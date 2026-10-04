//! Feasible joint paths with zero endpoint velocity/acceleration. Camera goal
//! changes queue during a segment instead of restarting the response each tick.

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
    body: Option<BodyCurve>,
}

#[derive(Clone)]
pub(crate) struct JointPath {
    pub current: [Option<ArmJoints>; 2],
    pub body: Option<BodyPose>,
    segment: Option<Segment>,
    queued: std::collections::VecDeque<Segment>,
    pub response_seconds: f32,
}

const RESPONSE_SECONDS: f32 = 0.15;

impl Default for JointPath {
    fn default() -> Self {
        Self {
            current: [None, None],
            body: None,
            segment: None,
            queued: Default::default(),
            response_seconds: RESPONSE_SECONDS,
        }
    }
}

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
            response_seconds: self.response_seconds,
        }
    }

    pub fn busy(&self) -> bool {
        self.segment.is_some()
    }

    /// Install a route whose edges passed the same continuous
    /// predicate as a direct SQP step. Each junction has zero velocity and
    /// acceleration; no raw interpolation bypasses the constraints.
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
                duration: self.response_seconds,
                body: body
                    .clone()
                    .zip(next_body.clone())
                    .map(|(from, to)| BodyCurve { from, to }),
            });
            from = to;
            body = next_body;
        }
        self.segment = self.queued.pop_front();
        self.response_seconds = RESPONSE_SECONDS;
        Ok(complete)
    }

    pub fn advance(
        &mut self,
        problem: &Problem<'_>,
        target: [Option<ArmJoints>; 2],
        dt: f32,
    ) -> Result<(), SolveStatus> {
        if self.current.iter().all(Option::is_none) {
            let evaluation = problem
                .evaluate(target)
                .map_err(SolveStatus::InvalidGeometry)?;
            if !problem.feasible(&evaluation) {
                return Err(SolveStatus::NoFeasibleSolution);
            }
            self.current = target;
            self.body = problem.body_curve.map(|c| c.to.clone());
            return Ok(());
        }
        if self.segment.is_none() {
            if !check_transition(problem, self.current, target) {
                return Err(SolveStatus::BlockedPath);
            }
            self.segment = Some(Segment {
                from: self.current,
                to: target,
                elapsed: 0.0,
                duration: self.response_seconds,
                body: problem.body_curve.cloned(),
            });
            self.response_seconds = RESPONSE_SECONDS;
        }
        let Some(segment) = self.segment.as_mut() else {
            return Ok(());
        };
        segment.elapsed = (segment.elapsed + dt.max(0.0)).min(segment.duration);
        let t = segment.elapsed / segment.duration;
        let amount = smooth_progress(t);
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
        if t >= 1.0 {
            self.segment = self.queued.pop_front();
        }
        Ok(())
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

/// Continuous ROM and skin checks. Skin chord bounds allow 0.001 arm
/// lengths of spatial approximation per hull/pivot; this is not a strict
/// zero-penetration certificate. See the upper-limb anatomy ADR.
pub(crate) fn check_transition(
    problem: &Problem<'_>,
    from: [Option<ArmJoints>; 2],
    to: [Option<ArmJoints>; 2],
) -> bool {
    // Bound displacements about their actual joint pivots, not about the
    // world origin. A torso vertex that is static in this problem moves zero.
    let mut pivot_accelerations = vec![0.0_f32; problem.geometry.joints.len()];
    let mut accelerations = vec![0.0_f32; problem.geometry.hulls.len()];
    let mut rom_angle = 0.0_f32;
    let mut rom_acceleration = 0.0_f32;
    for ((chain, a), b) in problem.chains.into_iter().zip(from).zip(to) {
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
        rom_angle = rom_angle.max(girdle + upper);
        let shoulder_length = chain
            .rest
            .shoulder
            .map_or(0.0, |s| s.position.distance(chain.rest.upper_arm.position));
        let mut fingers = std::collections::HashMap::new();
        for finger in [
            chain.finger_rest.thumb,
            chain.finger_rest.index,
            chain.finger_rest.middle,
            chain.finger_rest.ring,
            chain.finger_rest.little,
        ] {
            let mut previous = chain.rest.wrist.position;
            let mut length = 0.0;
            for joint in [
                finger.metacarpal,
                finger.proximal,
                finger.intermediate,
                finger.distal,
            ]
            .into_iter()
            .flatten()
            {
                length += joint.rest.position.distance(previous);
                fingers.insert(joint.entity, (length, joint.rest.position));
                previous = joint.rest.position;
            }
        }
        let g2 = crate::girdle::acceleration_bound(chain, a.angles, b.angles);
        let upper_speed = 2.0 * p + e + axial;
        // log(q)=2*v*acos(w)/sqrt(1-w²), w>=0. On this hemisphere
        // |f|<=pi, |f'|<=2, |f''|<=pi. Bound the log's second derivative
        // and therefore every fitted DOP plane, without a first-order
        // penalty for motion tangent to an active ROM boundary.
        let q1 = (girdle + upper_speed) * 0.5;
        let q2 = g2 * 0.5 + girdle * upper_speed * 0.5 + upper_speed.powi(2) * 0.25;
        rom_acceleration = rom_acceleration
            .max((std::f32::consts::PI + 2.0) * q2 + (4.0 + std::f32::consts::PI) * q1 * q1);

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
        let hand_speed = lower_speed + wf + wd;
        let hand_acceleration =
            lower_acceleration + 2.0 * lower_speed * (wf + wd) + (wf + wd).powi(2);
        // A point follows at most one digit (three bend axes and one
        // opening axis, or two CMC axes and MCP/IP). Independent digits
        // never inherit each other's rotations.
        let weight_change = (a.finger_weight - b.finger_weight).abs();
        let baseline_speed = 3.0
            * ((a.rest_curl - b.rest_curl).abs()
                + a.rest_curl.abs().max(b.rest_curl.abs()) * weight_change);
        let baseline_accel = 6.0 * (a.rest_curl - b.rest_curl).abs() * weight_change;
        let mut finger_speed = baseline_speed;
        let mut finger_accel = baseline_accel;
        if let Some((af, bf)) = a.fingers.or(b.fingers).zip(b.fingers.or(a.fingers)) {
            let coords = |p: HandFingerPose| {
                let [mcp, ip] = p.thumb;
                let [cmc_flex, cmc_abduct] = p.thumb_cmc;
                p.fingers
                    .into_iter()
                    .zip(p.spread)
                    .map(|([a, b, c], d)| [a, b, c, d])
                    .chain(std::iter::once([mcp, ip, cmc_flex, cmc_abduct]))
            };
            for (a, b) in coords(af).zip(coords(bf)) {
                let mut speed = baseline_speed + 4.0 * std::f32::consts::PI * weight_change;
                let mut acceleration = baseline_accel;
                for (x, y) in a.into_iter().zip(b) {
                    speed += (x - y).abs() + weight_change * x.abs().max(y.abs());
                    acceleration += 2.0 * (x - y).abs() * weight_change;
                }
                finger_speed = finger_speed.max(speed);
                finger_accel = finger_accel.max(acceleration);
            }
        }
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
            } else if bone == chain.hand || fingers.contains_key(&bone) {
                let (digit_speed, digit_accel) = if fingers.contains_key(&bone) {
                    (finger_speed, finger_accel)
                } else {
                    (0.0, 0.0)
                };
                let hand_length = fingers.get(&bone).map_or_else(
                    || point.distance(chain.rest.wrist.position),
                    |(length, rest)| length + point.distance(*rest),
                );
                shoulder_length * g2
                    + chain.rest.upper_arm_length * upper_speed.powi(2)
                    + chain.rest.forearm_length * lower_acceleration
                    + hand_length
                        * (hand_acceleration
                            + 2.0 * hand_speed * digit_speed
                            + digit_speed.powi(2)
                            + digit_accel)
            } else {
                0.0
            }
        };
        for (joint, acceleration) in problem.geometry.joints.iter().zip(&mut pivot_accelerations) {
            *acceleration += point_acceleration(joint.pivot_bone, joint.pivot);
        }
        for (hull, acceleration) in problem.geometry.hulls.iter().zip(&mut accelerations) {
            let value = if hull.skin.is_empty() {
                hull.shape
                    .points()
                    .iter()
                    .map(|p| {
                        point_acceleration(
                            hull.bone,
                            bevy::prelude::Vec3::from_array(p.to_array().map(|v| v as f32)),
                        )
                    })
                    .fold(0.0_f32, f32::max)
            } else {
                hull.skin
                    .iter()
                    .map(|v| {
                        v.influences
                            .iter()
                            .map(|(bone, p, w)| point_acceleration(*bone, *p) * w)
                            .sum::<f32>()
                    })
                    .fold(0.0_f32, f32::max)
            };
            *acceleration += value;
        }
    }
    if let Some(curve) = problem.body_curve {
        // Factor the hierarchy once per bone, not once per skin influence.
        let bounds: std::collections::HashMap<_, _> = problem
            .geometry
            .bones()
            .filter_map(|bone| curve.acceleration_bound(bone).map(|bound| (bone, bound)))
            .collect();
        let acceleration_at = |bone, point| bounds.get(&bone).map_or(0.0, |bound| bound.at(point));
        for (hull, acceleration) in problem.geometry.hulls.iter().zip(&mut accelerations) {
            let bound = if hull.skin.is_empty() {
                hull.shape
                    .points()
                    .iter()
                    .map(|p| {
                        acceleration_at(
                            hull.bone,
                            bevy::prelude::Vec3::from_array(p.to_array().map(|v| v as f32)),
                        )
                    })
                    .fold(0.0_f32, f32::max)
            } else {
                hull.skin
                    .iter()
                    .map(|v| {
                        v.influences
                            .iter()
                            .map(|(b, p, w)| acceleration_at(*b, *p) * w)
                            .sum::<f32>()
                    })
                    .fold(0.0_f32, f32::max)
            };
            *acceleration += bound;
        }
        for (joint, acceleration) in problem.geometry.joints.iter().zip(&mut pivot_accelerations) {
            *acceleration += acceleration_at(joint.pivot_bone, joint.pivot);
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
        let chart = mid
            .iter()
            .flatten()
            .map(|(_, chart)| *chart)
            .fold(f32::INFINITY, f32::min);
        let rom_certified = chart >= rom_angle * (end - start) * 0.5
            && endpoint_a.iter().zip(&endpoint_b).all(|(a, b)| {
                a.as_ref().zip(b.as_ref()).is_none_or(|(a, b)| {
                    a.0.iter().zip(b.0).skip(1).all(|(a, b)| {
                        // Retain the slope of each boundary margin. A path
                        // leaving contact need not pay its maximum chord
                        // error at the endpoint, where that error is zero.
                        let curvature = rom_acceleration * (end - start).powi(2);
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
    let check_skin = |start: f32, end: f32| -> Option<bool> {
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
        // of its endpoint chord when |x''| <= M. Endpoint hulls preserve
        // tangent motion; an isotropic first-order sweep stalls at contact.
        // Render-time CCD uses a sub-millimetre spatial resolution, like
        // the linear contact tolerance in realtime physics engines. Endpoint
        // collision and continuous ROM checks retain their original precision.
        let resolution = problem
            .chains
            .into_iter()
            .flatten()
            .map(|c| c.rest.total_arm_length)
            .fold(0.0_f32, f32::max)
            * 0.001;
        let clear = problem.geometry.sweep_is_clear(
            &|bone| motion(&endpoint_a, body_a.as_ref().unwrap_or(problem.body), bone),
            &|bone| motion(&endpoint_b, body_b.as_ref().unwrap_or(problem.body), bone),
            |i| {
                (accelerations.get(i).copied().unwrap_or(0.0) * (end - start).powi(2) / 8.0
                    - resolution)
                    .max(0.0)
            },
            |i| {
                (pivot_accelerations.get(i).copied().unwrap_or(0.0) * (end - start).powi(2) / 8.0
                    - resolution)
                    .max(0.0)
            },
        );
        clear.ok()
    };
    // ROM and skin constrain the same path, but need not subdivide it at
    // the same places. A tight angular boundary must not make us skin and
    // clip the mesh again for every otherwise unnecessary angular interval.
    check_subdivisions(&check_rom) && check_subdivisions(&check_skin)
}

fn check_subdivisions(check: &(impl Fn(f32, f32) -> Option<bool> + Sync)) -> bool {
    let pool = bevy::tasks::AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
    let mut intervals = vec![(0.0_f32, 1.0_f32)];
    let mut remaining = 256;
    // Independent interval checks share no mutable geometry. Evaluate
    // each subdivision level together on the same pool used by the SQP.
    // Budget exhaustion is unresolved; the spatial error budget is not enlarged.
    while !intervals.is_empty() && remaining > 0 {
        let count = intervals.len().min(remaining);
        remaining -= count;
        let checked = pool.scope(|scope| {
            for (start, end) in intervals.drain(..count) {
                scope.spawn(async move { (start, end, check(start, end)) });
            }
        });
        for (start, end, clear) in checked {
            match clear {
                Some(true) => {}
                Some(false) => {
                    let mid = (start + end) * 0.5;
                    if mid == start || mid == end {
                        return false;
                    }
                    intervals.push((start, mid));
                    intervals.push((mid, end));
                }
                None => return false,
            }
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
        let mut path = JointPath::default();
        path.advance(&problem, start, 0.0).unwrap();
        let h = RESPONSE_SECONDS / 100.0;
        let mut positions = vec![path.current[0].unwrap().angles[0]];
        for _ in 0..101 {
            path.advance(&problem, target, h).unwrap();
            assert!(problem.feasible(&problem.evaluate(path.current).unwrap()));
            positions.push(path.current[0].unwrap().angles[0]);
        }
        let start_velocity = (positions[1] - positions[0]) / h;
        let end_velocity = (positions[100] - positions[99]) / h;
        // Quintic starts with 10*t^3: finite differences decay as h^2.
        let difference_bound = 11.0 * 0.4 * h * h / RESPONSE_SECONDS.powi(3);
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
        use crate::collision::{BoneMotion, Hull, Region};
        use bevy::prelude::*;
        use parry3d::shape::ConvexPolyhedron;
        let chain = chain(ArmSide::Left);
        let a = state(0.0, 0.7, -0.2, 0.8);
        let b = state(1.8, 0.7, -0.2, 0.8);
        let midpoint = interpolate([Some(a), None], [Some(b), None], 0.5)[0]
            .unwrap()
            .forward(&chain)
            .unwrap()
            .wrist;
        let box_hull = |bone, region, centre: Vec3, radius: f32| {
            let points: Vec<_> = [-radius, radius]
                .into_iter()
                .flat_map(|x| {
                    [-radius, radius].into_iter().flat_map(move |y| {
                        [-radius, radius].map(move |z| {
                            parry3d::math::Vector::from_array(
                                (centre + Vec3::new(x, y, z)).to_array().map(f64::from),
                            )
                        })
                    })
                })
                .collect();
            Hull {
                bone,
                region,
                shape: ConvexPolyhedron::from_convex_hull(&points).unwrap(),
                skin: vec![],
            }
        };
        let torso = Entity::from_raw_u32(99).unwrap();
        let geometry = CollisionGeometry::new(
            vec![
                box_hull(
                    chain.hand,
                    Region::Hand(ArmSide::Left),
                    chain.rest.wrist.position,
                    0.01,
                ),
                box_hull(torso, Region::Torso, midpoint, 0.04),
            ],
            vec![],
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
