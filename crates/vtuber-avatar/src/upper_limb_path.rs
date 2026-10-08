//! Feasible joint paths resampled at the camera cadence. Explicit profile
//! transitions ease to rest; live observations already carry render-clock damping.

use crate::{
    upper_limb::ArmJoints,
    upper_limb_body::{BodyCurve, BodyPose},
    upper_limb_solver::{Problem, SolveStatus},
};
use vtuber_core::arm_tracking::HandFingerPose;

#[derive(Clone)]
struct Segment {
    from: [Option<ArmJoints>; 2],
    to: [Option<ArmJoints>; 2],
    elapsed: f32,
    duration: f32,
    eased: bool,
    body: Option<BodyCurve>,
}

#[derive(Clone, Default)]
pub(crate) struct JointPath {
    pub current: [Option<ArmJoints>; 2],
    pub body: Option<BodyPose>,
    segment: Option<Segment>,
    queued: std::collections::VecDeque<Segment>,
    pub transition_seconds: Option<f32>,
}

const RESPONSE_SECONDS: f32 = 1.0 / 30.0;

impl JointPath {
    pub fn after_current_segment(&self) -> Self {
        let last = self.queued.back().or(self.segment.as_ref());
        Self {
            current: last.map_or(self.current, |s| s.to),
            body: last
                .and_then(|s| s.body.as_ref().map(|b| b.to.clone()))
                .or_else(|| self.body.clone()),
            segment: None,
            queued: Default::default(),
            transition_seconds: self.transition_seconds,
        }
    }

    /// Match playback to the producer period instead of displaying a
    /// slow result in two frames and then freezing until the next one arrives.
    /// A fast producer still supplies a pose each camera interval. A distant
    /// IK branch/route must take as long as the equivalent local SQP steps;
    /// a collision-free shortcut is not permission for a two-frame snap.
    pub fn retime(&mut self, producer_seconds: f32) {
        for segment in self.segment.iter_mut().chain(&mut self.queued) {
            if !segment.eased {
                let distance = segment
                    .from
                    .into_iter()
                    .zip(segment.to)
                    .filter_map(|(a, b)| a.zip(b))
                    .flat_map(|(a, b)| a.angles.into_iter().zip(b.angles))
                    .map(|(a, b)| (b - a).abs())
                    .fold(0.0_f32, f32::max);
                let steps = (distance / crate::upper_limb_solver::MAX_STEP_RADIANS).max(1.0);
                segment.duration = producer_seconds.max(RESPONSE_SECONDS * steps);
            }
        }
    }

    pub fn busy(&self) -> bool {
        self.segment.is_some()
    }

    /// Install a route whose edges passed the same continuous
    /// predicate as a direct SQP step. Changing the playback speed does not
    /// change its admitted geometric path.
    pub fn plan(
        &mut self,
        problem: &Problem<'_>,
        target: [Option<ArmJoints>; 2],
        obsolete: &std::sync::atomic::AtomicBool,
    ) -> Result<bool, SolveStatus> {
        let route = crate::upper_limb_planner::connect(problem, self.current, target, obsolete)?;
        let complete = route.last().is_some_and(|(pose, body)| {
            *pose == target
                && match (body, problem.body_curve) {
                    (Some(body), Some(curve)) => body.same(&curve.to),
                    (None, None) => true,
                    _ => false,
                }
        });
        let mut from = self.current;
        let mut body = self.body.clone();
        for (to, next_body) in route {
            self.queued.push_back(Segment {
                from,
                to,
                elapsed: 0.0,
                duration: self.transition_seconds.unwrap_or(RESPONSE_SECONDS),
                eased: self.transition_seconds.is_some(),
                body: body
                    .clone()
                    .zip(next_body.clone())
                    .map(|(from, to)| BodyCurve { from, to }),
            });
            from = to;
            body = next_body;
        }
        self.segment = self.queued.pop_front();
        self.transition_seconds = None;
        Ok(complete)
    }

    pub fn advance(
        &mut self,
        problem: &Problem<'_>,
        target: [Option<ArmJoints>; 2],
        dt: f32,
    ) -> Result<f32, SolveStatus> {
        if self.current.iter().all(Option::is_none) {
            let evaluation = problem
                .evaluate(target)
                .map_err(SolveStatus::InvalidGeometry)?;
            if !problem.feasible(&evaluation) {
                return Err(SolveStatus::NoFeasibleSolution);
            }
            self.current = target;
            self.body = problem.body_curve.map(|c| c.to.clone());
            return Ok(dt.max(0.0));
        }
        if self.segment.is_none() {
            if !check_transition(problem, self.current, target) {
                return Err(SolveStatus::BlockedPath);
            }
            self.segment = Some(Segment {
                from: self.current,
                to: target,
                elapsed: 0.0,
                duration: self.transition_seconds.unwrap_or(RESPONSE_SECONDS),
                eased: self.transition_seconds.is_some(),
                body: problem.body_curve.cloned(),
            });
            self.transition_seconds = None;
        }
        let mut remaining = dt.max(0.0);
        loop {
            let Some(segment) = self.segment.as_mut() else {
                return Ok(remaining);
            };
            let left = (segment.duration - segment.elapsed).max(0.0);
            if remaining >= left {
                segment.elapsed = segment.duration;
                remaining -= left;
            } else {
                segment.elapsed += remaining;
                remaining = 0.0;
            }
            let t = segment.elapsed / segment.duration;
            // Do not stop and restart every smoothed camera sample. At 60 Hz the
            // 30 Hz segment supplies the intervening pose without a second 150 ms
            // response. Longer user-requested profile changes still ease to rest.
            let amount = if segment.eased { smooth_progress(t) } else { t };
            // Use the stored endpoint exactly: a + (b-a) need not round to b.
            // The prefetched segment is anchored at that same admitted endpoint.
            let next = if t >= 1.0 {
                segment.to
            } else {
                interpolate(segment.from, segment.to, amount)
            };
            let next_body = segment.body.as_ref().map(|b| b.at(amount));
            // The worker checked this complete segment, including the body.
            // Rendering only advances its parameter; it cannot introduce a new
            // candidate. Repeating convex clipping here would block every tick.
            self.current = next;
            if next_body.is_some() {
                self.body = next_body;
            }
            if t < 1.0 {
                return Ok(0.0);
            }
            self.segment = self.queued.pop_front();
            if remaining == 0.0 {
                return Ok(0.0);
            }
        }
    }
}

fn smooth_progress(t: f32) -> f32 {
    // The f32 polynomial can exceed one near t=0.996, extrapolating a
    // certified segment past an endpoint at a joint limit. Evaluate its
    // mathematical [0, 1] range without that cancellation error.
    let t = f64::from(t);
    (t * t * t * (10.0 + t * (-15.0 + 6.0 * t))).clamp(0.0, 1.0) as f32
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (f64::from(a) + (f64::from(b) - f64::from(a)) * f64::from(t)) as f32
}

fn fingers(a: HandFingerPose, b: HandFingerPose, t: f32) -> HandFingerPose {
    let lerp = |a, b| lerp(a, b, t);
    let mut out = a;
    for (q, b) in out
        .fingers
        .iter_mut()
        .flatten()
        .zip(b.fingers.into_iter().flatten())
    {
        *q = lerp(*q, b);
    }
    for (q, b) in out.spread.iter_mut().zip(b.spread) {
        *q = lerp(*q, b);
    }
    for (q, b) in out.thumb.iter_mut().zip(b.thumb) {
        *q = lerp(*q, b);
    }
    for (q, b) in out.thumb_cmc.iter_mut().zip(b.thumb_cmc) {
        *q = lerp(*q, b);
    }
    out.thumb_spread = lerp(a.thumb_spread, b.thumb_spread);
    out
}

pub(crate) fn interpolate(
    a: [Option<ArmJoints>; 2],
    b: [Option<ArmJoints>; 2],
    t: f32,
) -> [Option<ArmJoints>; 2] {
    if t <= 0.0 {
        return a;
    }
    if t >= 1.0 {
        return b;
    }
    let mut out = a;
    for ((out, a), b) in out.iter_mut().zip(a).zip(b) {
        *out = match a.zip(b) {
            Some((mut a, b)) => {
                for (q, target) in a.angles.iter_mut().zip(b.angles) {
                    *q = lerp(*q, target, t);
                }
                a.finger_weight = lerp(a.finger_weight, b.finger_weight, t);
                a.rest_curl = lerp(a.rest_curl, b.rest_curl, t);
                a.fingers = match (a.fingers, b.fingers) {
                    (Some(a), Some(b)) => Some(fingers(a, b, t)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    _ => None,
                };
                Some(a)
            }
            None => b.or(a),
        };
    }
    out
}

/// Continuous ROM and capsule checks. Chord bounds allow 0.001 arm
/// lengths of spatial approximation per collider; this is not a strict
/// zero-penetration certificate. See the upper-limb anatomy ADR.
pub(crate) fn check_transition(
    problem: &Problem<'_>,
    from: [Option<ArmJoints>; 2],
    to: [Option<ArmJoints>; 2],
) -> bool {
    // Bound displacements about their actual joint pivots, not about the
    // world origin. A torso vertex that is static in this problem moves zero.
    let mut accelerations = vec![0.0_f32; problem.geometry.capsules.len()];
    // A moving left arm must not spend the stationary right arm's ROM
    // margin (or vice versa). Each shoulder has its own angular curve.
    let mut rom_bounds = [(0.0_f32, 0.0_f32); 2];
    for (((chain, a), b), rom_bound) in problem
        .chains
        .into_iter()
        .zip(from)
        .zip(to)
        .zip(&mut rom_bounds)
    {
        let Some(((chain, a), b)) = chain.zip(a).zip(b) else {
            continue;
        };
        let d = std::array::from_fn::<_, 9, _>(|i| {
            a.angles
                .get(i)
                .zip(b.angles.get(i))
                .map_or(0.0, |(a, b)| (a - b).abs())
        });
        let [p, e, axial, f, roll, wf, wd, ..] = d;
        let upper = 2.0
            * (a.angles
                .get(1)
                .copied()
                .unwrap_or(0.0)
                .max(b.angles.get(1).copied().unwrap_or(0.0))
                * 0.5)
                .sin()
            * p
            + e
            + axial;
        let girdle = crate::girdle::rotation_bound(chain, a.angles, b.angles);
        rom_bound.0 = girdle + upper;
        let shoulder_length = chain
            .rest
            .shoulder
            .map_or(0.0, |s| s.position.distance(chain.rest.upper_arm.position));
        let g2 = crate::girdle::acceleration_bound(chain, a.angles, b.angles);
        let upper_speed = 2.0 * p + e + axial;
        // log(q)=2*v*acos(w)/sqrt(1-w²), w>=0. On this hemisphere
        // |f|<=pi, |f'|<=2, |f''|<=pi. Bound the log's second derivative
        // and therefore every fitted DOP plane, without a first-order
        // penalty for motion tangent to an active ROM boundary.
        let q1 = (girdle + upper_speed) * 0.5;
        let q2 = g2 * 0.5 + girdle * upper_speed * 0.5 + upper_speed.powi(2) * 0.25;
        rom_bound.1 = (std::f32::consts::PI + 2.0) * q2 + (4.0 + std::f32::consts::PI) * q1 * q1;

        let Some(radius) = crate::arm_anatomy::RadiusGeometry::from_arm(
            chain.rest.elbow.position - chain.rest.upper_arm.position,
            chain.side,
        ) else {
            return false;
        };
        let (r1, r2) = radius.derivative_bounds();
        let lower_speed = upper_speed + f + r1 * roll;
        let lower_acceleration =
            (upper_speed + f).powi(2) + 2.0 * (upper_speed + f) * r1 * roll + r2 * roll * roll;
        let hand_acceleration =
            lower_acceleration + 2.0 * lower_speed * (wf + wd) + (wf + wd).powi(2);
        let point_acceleration = |bone, point: bevy::prelude::Vec3| {
            if Some(bone) == chain.shoulder {
                chain
                    .rest
                    .shoulder
                    .map_or(0.0, |s| point.distance(s.position) * g2)
            } else if bone == chain.upper_arm {
                shoulder_length * g2
                    + point.distance(chain.rest.upper_arm.position) * upper_speed.powi(2)
            } else if bone == chain.lower_arm {
                shoulder_length * g2
                    + chain.rest.upper_arm_length * upper_speed.powi(2)
                    + point.distance(chain.rest.elbow.position) * lower_acceleration
            } else if bone == chain.hand {
                shoulder_length * g2
                    + chain.rest.upper_arm_length * upper_speed.powi(2)
                    + chain.rest.forearm_length * lower_acceleration
                    + point.distance(chain.rest.wrist.position) * hand_acceleration
            } else {
                0.0
            }
        };
        for (capsule, acceleration) in problem.geometry.capsules.iter().zip(&mut accelerations) {
            *acceleration += capsule
                .endpoints
                .into_iter()
                .map(|p| point_acceleration(capsule.bone, p))
                .fold(0.0_f32, f32::max);
        }
    }
    if let Some(curve) = problem.body_curve {
        // Factor the hierarchy once per bone, not once per endpoint.
        let bounds: std::collections::HashMap<_, _> = problem
            .geometry
            .bones()
            .filter_map(|bone| curve.acceleration_bound(bone).map(|bound| (bone, bound)))
            .collect();
        let acceleration_at = |bone, point| bounds.get(&bone).map_or(0.0, |bound| bound.at(point));
        for (capsule, acceleration) in problem.geometry.capsules.iter().zip(&mut accelerations) {
            *acceleration += capsule
                .endpoints
                .into_iter()
                .map(|p| acceleration_at(capsule.bone, p))
                .fold(0.0_f32, f32::max);
        }
    }
    let domain = |t| {
        let mut result = [None; 2];
        for ((slot, chain), state) in result
            .iter_mut()
            .zip(problem.chains)
            .zip(interpolate(from, to, t))
        {
            if let Some((chain, state)) = chain.zip(state) {
                *slot = Some(state.joint_domain(chain)?);
            }
        }
        Some(result)
    };
    let check_rom = |start: f32, end: f32| -> Option<bool> {
        let mid = domain((start + end) * 0.5)?;
        let margin = mid
            .iter()
            .flatten()
            .flat_map(|(margins, _)| margins)
            .copied()
            .fold(f32::INFINITY, f32::min);
        if margin < -64.0 * f32::EPSILON {
            return None;
        }
        let endpoint_a = domain(start)?;
        let endpoint_b = domain(end)?;
        let rom_certified = endpoint_a
            .iter()
            .zip(&endpoint_b)
            .zip(&mid)
            .zip(rom_bounds)
            .all(|(((a, b), mid), (angle, acceleration))| {
                a.as_ref()
                    .zip(b.as_ref())
                    .zip(mid.as_ref())
                    .is_none_or(|((a, b), (_, chart))| {
                        *chart >= angle * (end - start) * 0.5
                            && a.0.iter().zip(b.0).skip(1).all(|(a, b)| {
                                // Retain the slope of each boundary margin. The chord
                                // error belongs to this arm, and is zero at endpoints.
                                let curvature = acceleration * (end - start).powi(2);
                                let t = if curvature > 0.0 {
                                    (0.5 - (b - a) / curvature).clamp(0.0, 1.0)
                                } else {
                                    0.0
                                };
                                let minimum = (a + (b - a) * t - curvature * t * (1.0 - t) * 0.5)
                                    .min(*a)
                                    .min(b);
                                minimum >= 64.0 * f32::EPSILON
                            })
                    })
            });

        Some(rom_certified)
    };
    let check_capsules = |start: f32, end: f32| -> Option<bool> {
        let endpoint_a = problem.kinematics(interpolate(from, to, start)).ok()?;
        let endpoint_b = problem.kinematics(interpolate(from, to, end)).ok()?;
        let body_a = problem
            .body_curve
            .map(|c| c.at(start).motions())
            .transpose();
        let body_b = problem.body_curve.map(|c| c.at(end).motions()).transpose();
        let (Ok(body_a), Ok(body_b)) = (body_a, body_b) else {
            return None;
        };
        let motion = |evaluation: &crate::upper_limb_solver::Evaluation,
                      body: &std::collections::HashMap<_, _>,
                      bone| {
            evaluation
                .arms
                .iter()
                .flatten()
                .find_map(|a| a.motion.get(&bone).copied())
                .or_else(|| body.get(&bone).copied())
        };
        // A twice differentiable point curve stays within M*(b-a)^2/8
        // of its endpoint chord when |x''| <= M. Swept capsules preserve
        // tangent motion; an isotropic first-order sweep stalls at contact.
        // Render-time CCD uses a sub-millimetre spatial resolution, like
        // the linear contact tolerance in realtime physics engines. Endpoint
        // collision and continuous ROM checks retain their original precision.
        let resolution = problem.contact_offset() * 0.5;
        let clear = problem.geometry.sweep_is_clear(
            &|bone| motion(&endpoint_a, body_a.as_ref().unwrap_or(problem.body), bone),
            &|bone| motion(&endpoint_b, body_b.as_ref().unwrap_or(problem.body), bone),
            |i| {
                (accelerations.get(i).copied().unwrap_or(0.0) * (end - start).powi(2) / 8.0
                    - resolution)
                    .max(0.0)
            },
        );
        if !clear.ok()? {
            // An overlapping enclosure is inconclusive, but an actual
            // midpoint collision disproves the entire edge. Reject it now
            // instead of spending the subdivision budget on an impossible
            // certificate (recursive bisection in motion validation).
            let mid = (start + end) * 0.5;
            let pose = problem.kinematics(interpolate(from, to, mid)).ok()?;
            let body = problem
                .body_curve
                .map(|c| c.at(mid).motions())
                .transpose()
                .ok()?;
            if !problem
                .geometry
                .pose_is_clear(
                    &|bone| motion(&pose, body.as_ref().unwrap_or(problem.body), bone),
                    0.0,
                )
                .ok()?
            {
                return None;
            }
            return Some(false);
        }
        Some(true)
    };
    check_subdivisions(&check_rom) && check_subdivisions(&check_capsules)
}

fn check_subdivisions(check: &impl Fn(f32, f32) -> Option<bool>) -> bool {
    let mut intervals = vec![(0.0_f32, 1.0_f32)];
    for _ in 0..32 {
        let Some((start, end)) = intervals.pop() else {
            return true;
        };
        match check(start, end) {
            Some(true) => {}
            Some(false) => {
                let mid = (start + end) * 0.5;
                if mid == start || mid == end {
                    return false;
                }
                intervals.push((mid, end));
                intervals.push((start, mid));
            }
            None => return false,
        }
    }
    intervals.is_empty()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;
    use crate::upper_limb::tests::{chain, state};
    use crate::{arm::ArmSide, collision::CollisionGeometry};
    use std::collections::HashMap;

    #[test]
    fn subdivision_stops_at_collision_and_never_accepts_an_exhausted_budget() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        assert!(!check_subdivisions(&|start, end| {
            calls.fetch_add(1, Ordering::Relaxed);
            if start == 0.0 && end == 1.0 {
                Some(false)
            } else {
                None
            }
        }));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        calls.store(0, Ordering::Relaxed);
        assert!(!check_subdivisions(&|_, _| {
            calls.fetch_add(1, Ordering::Relaxed);
            Some(false)
        }));
        assert_eq!(calls.load(Ordering::Relaxed), 32);
        assert!(check_subdivisions(&|_, _| Some(true)));
    }

    #[test]
    fn interpolation_stays_between_joint_limits_at_the_end_of_a_segment() {
        let mut a = state(0.0, 0.2, 0.0, 0.0);
        let mut b = a;
        a.angles = crate::upper_limb::BOUNDS.map(|(lo, _)| lo);
        b.angles = crate::upper_limb::BOUNDS.map(|(_, hi)| hi);
        let mut previous = 0.0;
        for i in 0..=100_000 {
            let t = i as f32 / 100_000.0;
            let amount = smooth_progress(t);
            assert!((previous..=1.0).contains(&amount));
            previous = amount;
            let both = interpolate([Some(a), Some(b)], [Some(b), Some(a)], amount);
            assert!(both.into_iter().flatten().all(|pose| pose.valid()));
        }
        assert_eq!(
            interpolate([Some(a), None], [Some(b), None], 1.0),
            [Some(b), None]
        );
    }

    #[test]
    fn live_samples_are_resampled_without_a_stop_start_envelope() {
        let chain = chain(ArmSide::Left);
        let geometry = CollisionGeometry::default();
        let body = HashMap::new();
        let problem = Problem {
            chains: [Some(&chain), None],
            goals: [None, None],
            geometry: &geometry,
            body: &body,
            body_curve: None,
            tolerance: 64.0 * f32::EPSILON,
        };
        let start = [Some(state(0.0, 0.5, -0.2, 0.7)), None];
        let target = [Some(state(0.12, 0.5, -0.2, 0.7)), None];
        for (samples, duration) in [(2, RESPONSE_SECONDS), (6, 0.1), (12, 0.2)] {
            let mut path = JointPath::default();
            path.advance(&problem, start, 0.0).unwrap();
            path.advance(&problem, target, 0.0).unwrap();
            path.retime(duration);
            for tick in 1..=samples {
                path.advance(&problem, target, duration / samples as f32)
                    .unwrap();
                let expected = 0.12 * tick as f32 / samples as f32;
                assert!((path.current[0].unwrap().angles[0] - expected).abs() < 1.0e-6);
                assert!(problem.feasible(&problem.evaluate(path.current).unwrap()));
            }
        }
        let mut path = JointPath::default();
        path.advance(&problem, start, 0.0).unwrap();
        let remaining = path.advance(&problem, target, 0.05).unwrap();
        assert!((remaining - (0.05 - RESPONSE_SECONDS)).abs() < 1.0e-7);
        let next = [Some(state(0.24, 0.5, -0.2, 0.7)), None];
        path.advance(&problem, next, remaining).unwrap();
        assert!((path.current[0].unwrap().angles[0] - 0.18).abs() < 1.0e-6);
    }

    #[test]
    fn a_distant_ik_branch_keeps_the_local_step_playback_rate() {
        let chain = chain(ArmSide::Left);
        let geometry = CollisionGeometry::default();
        let body = HashMap::new();
        let problem = Problem {
            chains: [Some(&chain), None],
            goals: [None, None],
            geometry: &geometry,
            body: &body,
            body_curve: None,
            tolerance: 64.0 * f32::EPSILON,
        };
        let mut a = state(0.0, 0.5, -0.2, 0.7);
        let mut b = a;
        a.angles[4] = -0.7;
        b.angles[4] = 0.7;
        let mut path = JointPath::default();
        path.advance(&problem, [Some(a), None], 0.0).unwrap();
        path.advance(&problem, [Some(b), None], 0.0).unwrap();
        path.retime(0.001);
        for _ in 0..12 {
            let previous = path.current[0].unwrap();
            path.advance(&problem, [Some(b), None], 1.0 / 60.0).unwrap();
            let current = path.current[0].unwrap();
            assert!((current.angles[4] - previous.angles[4]).abs() <= 0.125 + 1.0e-6);
            assert!(problem.feasible(&problem.evaluate(path.current).unwrap()));
        }
        assert_eq!(path.current, [Some(b), None]);
    }

    #[test]
    fn moving_one_arm_does_not_consume_the_other_arms_rom_margin() {
        let left = chain(ArmSide::Left);
        let right = chain(ArmSide::Right);
        let interior = state(0.0, 0.5, -0.2, 0.7);
        let margin = |pose: ArmJoints| {
            pose.joint_domain(&right)
                .unwrap()
                .0
                .into_iter()
                .fold(f32::INFINITY, f32::min)
        };
        let exterior = [-1.5, 0.0, 2.2]
            .into_iter()
            .flat_map(|p| {
                [0.17, 1.57, 3.0].into_iter().flat_map(move |e| {
                    [-1.5, 0.0, 0.34]
                        .into_iter()
                        .map(move |a| state(p, e, a, 1.0))
                })
            })
            .find(|pose| margin(*pose) < 0.0)
            .unwrap();
        assert!(margin(interior) > 0.0 && margin(exterior) < 0.0);
        let mut low = 0.0;
        let mut high = 1.0;
        let mut stationary = interior;
        for _ in 0..24 {
            let mid = (low + high) * 0.5;
            let pose = interpolate([Some(interior), None], [Some(exterior), None], mid)[0].unwrap();
            if margin(pose) >= 128.0 * f32::EPSILON {
                stationary = pose;
                low = mid;
            } else {
                high = mid;
            }
        }
        assert!(margin(stationary) < 0.0001);
        let geometry = CollisionGeometry::default();
        let body = HashMap::new();
        let problem = Problem {
            chains: [Some(&left), Some(&right)],
            goals: [None, None],
            geometry: &geometry,
            body: &body,
            body_curve: None,
            tolerance: 64.0 * f32::EPSILON,
        };
        let from = [Some(interior), Some(stationary)];
        let to = [Some(state(0.12, 0.5, -0.2, 0.7)), Some(stationary)];
        assert!(check_transition(&problem, from, to));
    }

    #[test]
    fn render_ticks_keep_the_path_and_endpoint_velocity_is_continuous() {
        let chain = chain(ArmSide::Left);
        let geometry = CollisionGeometry::default();
        let body = HashMap::new();
        let problem = Problem {
            chains: [Some(&chain), None],
            goals: [None, None],
            geometry: &geometry,
            body: &body,
            body_curve: None,
            tolerance: 64.0 * f32::EPSILON,
        };
        let start = [Some(state(0.0, 0.2, 0.0, 0.0)), None];
        let target = [Some(state(0.4, 0.5, -0.2, 0.7)), None];
        let mut path = JointPath {
            transition_seconds: Some(0.15),
            ..Default::default()
        };
        path.advance(&problem, start, 0.0).unwrap();
        let h = 0.15 / 100.0;
        let mut positions = vec![path.current[0].unwrap().angles[0]];
        for _ in 0..101 {
            path.advance(&problem, target, h).unwrap();
            assert!(problem.feasible(&problem.evaluate(path.current).unwrap()));
            positions.push(path.current[0].unwrap().angles[0]);
        }
        let start_velocity = (positions[1] - positions[0]) / h;
        let end_velocity = (positions[100] - positions[99]) / h;
        // Quintic starts with 10*t^3: finite differences decay as h^2.
        let difference_bound = 11.0 * 0.4 * h * h / 0.15_f32.powi(3);
        assert!(start_velocity.abs() <= difference_bound);
        assert!(end_velocity.abs() <= difference_bound + 64.0 * f32::EPSILON / h);
        assert!(
            (positions[50] - positions[0]).abs() > 0.1,
            "same goal must not restart each render tick"
        );
        assert_eq!(
            path.current, target,
            "prefetched endpoint must match exactly"
        );
    }
    #[test]
    fn safe_endpoints_never_authorize_a_hand_path_through_a_volume() {
        use crate::collision::{BoneMotion, CapsuleCollider, Region};
        use bevy::prelude::*;
        let chain = chain(ArmSide::Left);
        let a = state(0.0, 0.7, -0.2, 0.8);
        let b = state(1.8, 0.7, -0.2, 0.8);
        let midpoint = interpolate([Some(a), None], [Some(b), None], 0.5)[0]
            .unwrap()
            .forward(&chain)
            .unwrap()
            .wrist;
        let ball = |bone, region, centre: Vec3, radius: f32| CapsuleCollider {
            bone,
            region,
            endpoints: [centre; 2],
            radius,
        };
        let torso = Entity::from_raw_u32(99).unwrap();
        let geometry = CollisionGeometry::new(
            vec![
                ball(
                    chain.hand,
                    Region::Hand(ArmSide::Left),
                    chain.rest.wrist.position,
                    0.01,
                ),
                ball(torso, Region::Torso, midpoint, 0.04),
            ],
            &[],
        );
        let body = HashMap::from([(
            torso,
            BoneMotion {
                rotation: Quat::IDENTITY,
                translation: Vec3::ZERO,
            },
        )]);
        let problem = Problem {
            chains: [Some(&chain), None],
            goals: [None, None],
            geometry: &geometry,
            body: &body,
            body_curve: None,
            tolerance: 64.0 * f32::EPSILON * chain.rest.total_arm_length,
        };
        assert!(problem.feasible(&problem.evaluate([Some(a), None]).unwrap()));
        assert!(problem.feasible(&problem.evaluate([Some(b), None]).unwrap()));
        let mut path = JointPath::default();
        path.advance(&problem, [Some(a), None], 0.0).unwrap();
        assert!(matches!(
            path.advance(&problem, [Some(b), None], 1.0 / 60.0),
            Err(SolveStatus::BlockedPath)
        ));
        assert_eq!(path.current, [Some(a), None]);
        let mut planned = path.clone();
        assert!(
            !planned
                .plan(
                    &problem,
                    [Some(b), None],
                    &std::sync::atomic::AtomicBool::new(true)
                )
                .unwrap()
        );
        assert!(planned.busy());
        while planned.busy() {
            planned
                .advance(&problem, [Some(b), None], 1.0 / 60.0)
                .unwrap();
            assert!(problem.feasible(&problem.evaluate(planned.current).unwrap()));
        }
        assert_ne!(planned.current, path.current);
        assert_ne!(planned.current, [Some(b), None]);
        let mut planned = path.clone();
        // A fresh live target also gets a feasible prefix immediately. Repeated
        // requests for a held goal eventually use the global route if needed.
        for _ in 0..32 {
            if planned.current == [Some(b), None] {
                break;
            }
            planned
                .plan(
                    &problem,
                    [Some(b), None],
                    &std::sync::atomic::AtomicBool::new(false),
                )
                .unwrap();
            while planned.busy() {
                planned
                    .advance(&problem, [Some(b), None], 1.0 / 60.0)
                    .unwrap();
                assert!(problem.feasible(&problem.evaluate(planned.current).unwrap()));
            }
        }
        assert_eq!(planned.current, [Some(b), None]);
        let near = state(0.08, 0.72, -0.21, 0.81);
        // A changing observation may still display a short checked step;
        // only the now-obsolete global search must be cancelled.
        let mut direct = path.clone();
        direct
            .plan(
                &problem,
                [Some(near), None],
                &std::sync::atomic::AtomicBool::new(true),
            )
            .unwrap();
        assert!(direct.busy());
        for _ in 0..10 {
            path.advance(&problem, [Some(near), None], 1.0 / 60.0)
                .unwrap();
            assert!(problem.feasible(&problem.evaluate(path.current).unwrap()));
        }
    }
}
