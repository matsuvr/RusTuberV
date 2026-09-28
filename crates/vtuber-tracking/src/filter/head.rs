//! Critically damped second-order smoothing for head rotation on SO(3).
//!
//! The filter operates directly on [`UnitQuaternion`] values in the canonical
//! tracking basis used by [`crate::pose`]. Smoothing in quaternion space avoids
//! Euler-angle wrapping, gimbal-lock singularities, and independent per-axis
//! low-pass artefacts.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use vtuber_core::types::MonoTimeNs;

/// Parameters for the head rotation filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadFilterParams {
    /// Smoothing time constant in seconds.
    ///
    /// One fixed constant covers the whole channel; it does not increase with
    /// the target's departure from the filtered rotation. Smaller values make
    /// the filter follow input changes faster.
    /// Values must be positive and finite; non-positive values are clamped to
    /// [`f32::EPSILON`] when the filter runs.
    pub time_constant_sec: f32,
    /// Maximum allowed delta-time in seconds.
    ///
    /// Larger gaps are clamped to this value so that a stale observation
    /// cannot fully snap the output.
    pub max_dt_sec: f32,
    /// Maximum accepted angular departure from the filtered rotation, in radians.
    ///
    /// A target farther from the current smoothed rotation is rejected without
    /// advancing the rotation, velocity, or timestamp. This compares the target
    /// with the filtered state, not with the preceding raw observation. A later
    /// target is accepted once it falls within this limit.
    pub max_step_rad: f32,
}

impl Default for HeadFilterParams {
    fn default() -> Self {
        Self {
            time_constant_sec: 0.025,
            max_dt_sec: super::damped::DEFAULT_MAX_DT_SEC,
            max_step_rad: 1.25,
        }
    }
}

impl HeadFilterParams {
    /// Returns parameters with the given smoothing time constant.
    #[must_use]
    pub fn with_time_constant(time_constant_sec: f32) -> Self {
        Self {
            time_constant_sec,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FilterState {
    quat: UnitQuaternion<f32>,
    velocity: Vector3<f32>,
    last_time: MonoTimeNs,
}

/// Quaternion-centered critically damped second-order filter for head rotation.
///
/// The filter maintains an internal quaternion state and a tangent-space
/// angular velocity. On each update it applies a critically damped second
/// order (biquad) step to the local rotation error. The step is derived from
/// elapsed time and a time constant, making the smoothing independent of the
/// input frame rate.
///
/// The filter handles quaternion sign ambiguity by choosing the sign of the
/// target quaternion that yields the shortest arc from the current state.
/// Switching between `q` and `-q` for the same physical rotation therefore
/// does not produce a discontinuity.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadRotationFilter {
    params: HeadFilterParams,
    state: Option<FilterState>,
    quarantined_samples: u64,
}

impl HeadRotationFilter {
    /// Creates a new filter with the given parameters.
    #[must_use]
    pub fn new(params: HeadFilterParams) -> Self {
        Self {
            params,
            state: None,
            quarantined_samples: 0,
        }
    }

    /// Returns `true` if the filter has received at least one observation.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.state.is_some()
    }

    /// Resets the filter, discarding all state.
    ///
    /// The next call to [`update`](Self::update) initializes the filter with
    /// that observation.
    pub fn reset(&mut self) {
        self.state = None;
        self.quarantined_samples = 0;
    }

    /// Reacquires tracking, discarding the previous smoothed state.
    ///
    /// This is equivalent to [`reset`](Self::reset): the stored rotation,
    /// angular velocity, timestamp, and rejection count are discarded. The
    /// next observation becomes the new initial state.
    pub fn reacquire(&mut self) {
        self.reset();
    }

    /// Number of targets rejected for exceeding the filtered-state angular limit.
    #[must_use]
    pub fn quarantined_samples(&self) -> u64 {
        self.quarantined_samples
    }

    /// Updates the filter with a new target rotation.
    ///
    /// `timestamp` is expected to be monotonically non-decreasing. If it is
    /// older than the previous observation the elapsed time is treated as
    /// zero, so the output is the current state (or the input if the filter
    /// was just reset).
    ///
    /// # Arguments
    ///
    /// * `target` - Desired rotation in the canonical tracking basis.
    /// * `timestamp` - Monotonic timestamp of the observation.
    ///
    /// # Returns
    ///
    /// The smoothed rotation. This equals `target` on the first update after
    /// a [`reset`](Self::reset) or [`reacquire`](Self::reacquire).
    #[must_use]
    pub fn update(
        &mut self,
        target: UnitQuaternion<f32>,
        timestamp: MonoTimeNs,
    ) -> UnitQuaternion<f32> {
        let Some(state) = self.state else {
            self.state = Some(FilterState {
                quat: target,
                velocity: Vector3::zeros(),
                last_time: timestamp,
            });
            return target;
        };

        // Compute elapsed seconds. `saturating_sub` clamps backwards
        // timestamps to zero and avoids overflow for very large differences.
        let dt_ns = timestamp.0.saturating_sub(state.last_time.0);
        let dt_sec = (dt_ns as f32) / 1_000_000_000.0;
        let dt_sec = dt_sec.min(self.params.max_dt_sec).max(0.0);

        // If dt is zero (same timestamp or backwards), keep the current
        // state. This also covers the zero/negative dt acceptance cases.
        if dt_sec <= 0.0 {
            return state.quat;
        }

        // Clamp tau away from zero before the damping step.
        let tau = self.params.time_constant_sec.max(f32::EPSILON);

        // Choose the quaternion sign that gives the shortest arc.
        let signed_target = choose_shortest_arc(state.quat, target);
        let error = (state.quat.inverse() * signed_target).scaled_axis();
        let max_step = if self.params.max_step_rad.is_finite() {
            self.params.max_step_rad.max(0.0)
        } else {
            f32::MAX
        };
        if error.norm() > max_step {
            self.quarantined_samples = self.quarantined_samples.saturating_add(1);
            return state.quat;
        }

        // A critically damped second-order response in the local rotation
        // vector is the SO(3) equivalent of a biquad. The quaternion remains
        // on SO(3); only the tangent-space error and velocity are filtered.
        let (smoothed, velocity) =
            critically_damped_rotation_step(state.quat, signed_target, state.velocity, dt_sec, tau);

        self.state = Some(FilterState {
            quat: smoothed,
            velocity,
            last_time: timestamp,
        });

        smoothed
    }

    /// Returns the current smoothed rotation without advancing the filter.
    ///
    /// Returns `None` if the filter has not been initialized.
    #[must_use]
    pub fn current(&self) -> Option<UnitQuaternion<f32>> {
        self.state.map(|s| s.quat)
    }
}

fn critically_damped_rotation_step(
    current: UnitQuaternion<f32>,
    target: UnitQuaternion<f32>,
    velocity: Vector3<f32>,
    dt_sec: f32,
    time_constant_sec: f32,
) -> (UnitQuaternion<f32>, Vector3<f32>) {
    let error = (current.inverse() * target).scaled_axis();
    let (step, new_velocity) =
        super::damped::critically_damped_step(error, velocity, dt_sec, time_constant_sec);
    (
        current * UnitQuaternion::from_scaled_axis(step),
        new_velocity,
    )
}

/// Returns `target` or `-target`, whichever is closer to `current`.
#[must_use]
fn choose_shortest_arc(
    current: UnitQuaternion<f32>,
    target: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let c = current.quaternion();
    let t = target.quaternion();
    let dot = c.w * t.w + c.i * t.i + c.j * t.j + c.k * t.k;
    if dot < 0.0 { negate(target) } else { target }
}

/// Explicitly negates a unit quaternion, preserving unit norm.
#[must_use]
fn negate(q: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    let inner = q.quaternion();
    UnitQuaternion::from_quaternion(Quaternion::new(-inner.w, -inner.i, -inner.j, -inner.k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    fn ts(ns: u64) -> MonoTimeNs {
        MonoTimeNs(ns)
    }

    #[test]
    fn first_update_initializes_state() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        assert!(!filter.is_initialized());
        let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.5);
        let out = filter.update(q, ts(16_666_667));
        assert!(filter.is_initialized());
        assert_relative_eq!(out.quaternion().w, q.quaternion().w, epsilon = 1e-6);
        assert_relative_eq!(out.quaternion().i, q.quaternion().i, epsilon = 1e-6);
        assert_relative_eq!(out.quaternion().j, q.quaternion().j, epsilon = 1e-6);
        assert_relative_eq!(out.quaternion().k, q.quaternion().k, epsilon = 1e-6);
    }

    #[test]
    fn zero_dt_returns_current_state() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.5);
        let out1 = filter.update(q, ts(1_000_000_000));
        let out2 = filter.update(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.5),
            ts(1_000_000_000),
        );
        assert_relative_eq!(out1.quaternion().w, out2.quaternion().w, epsilon = 1e-6);
        assert_relative_eq!(out1.quaternion().i, out2.quaternion().i, epsilon = 1e-6);
        assert_relative_eq!(out1.quaternion().j, out2.quaternion().j, epsilon = 1e-6);
        assert_relative_eq!(out1.quaternion().k, out2.quaternion().k, epsilon = 1e-6);
    }

    #[test]
    fn backwards_timestamp_returns_current_state() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.5);
        let out1 = filter.update(q, ts(2_000_000_000));
        let out2 = filter.update(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.5),
            ts(1_000_000_000),
        );
        assert_relative_eq!(out1.quaternion().w, out2.quaternion().w, epsilon = 1e-6);
        assert_relative_eq!(out1.quaternion().i, out2.quaternion().i, epsilon = 1e-6);
        assert_relative_eq!(out1.quaternion().j, out2.quaternion().j, epsilon = 1e-6);
        assert_relative_eq!(out1.quaternion().k, out2.quaternion().k, epsilon = 1e-6);
    }

    #[test]
    fn large_single_frame_rotation_is_quarantined() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let initial = UnitQuaternion::identity();
        let out1 = filter.update(initial, ts(1_000_000_000));
        let out2 = filter.update(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 2.0),
            ts(1_016_666_667),
        );
        assert_relative_eq!(out1.angle_to(&out2), 0.0, epsilon = 1e-6);
        assert_eq!(filter.quarantined_samples(), 1);
    }

    #[test]
    fn turns_at_2_and_8_rad_per_sec_have_similar_lag_per_rate() {
        // Compare steady turns at 2 and 8 rad/s with the same time constant.
        // Their lag per unit angular speed should be similar: a faster turn
        // must not stiffen the filter. The assertions below compare these two
        // sampled trajectories, not every possible speed.
        let lag_per_rate = |rate: f32| {
            let mut filter = HeadRotationFilter::new(HeadFilterParams::with_time_constant(0.025));
            let mut turn = UnitQuaternion::identity();
            let mut now = 0u64;
            let _ = filter.update(turn, ts(now));
            let mut out = turn;
            let step_ns = 16_666_667u64;
            for _ in 0..240 {
                turn *= UnitQuaternion::from_axis_angle(&Vector3::y_axis(), rate / 60.0);
                now += step_ns;
                out = filter.update(turn, ts(now));
            }
            out.angle_to(&turn) / rate
        };
        let slow = lag_per_rate(2.0);
        let fast = lag_per_rate(8.0);
        assert!(
            (fast - slow).abs() < 0.01,
            "the lag per rad/s of turn must not depend on the turn's speed: \
             {slow} vs {fast}"
        );
        assert!(
            slow < 0.06,
            "a 2 rad/s turn must be tracked closely: {slow}"
        );
    }

    #[test]
    fn a_fast_turn_is_not_quarantined_as_a_jump() {
        // A 4 rad/s turn at 30 Hz advances by about 0.133 rad per observation.
        // The target's departure from the filtered state, which also includes
        // smoothing lag, must stay within the 1.25 rad rejection limit.
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let mut turn = UnitQuaternion::identity();
        let mut now = 0u64;
        let _ = filter.update(turn, ts(now));
        let mut previous = 0.0f32;
        for _ in 0..60 {
            turn *= UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 4.0 / 30.0);
            now += 33_333_333;
            let out = filter.update(turn, ts(now));
            let step = out.angle_to(&turn);
            assert!(step >= previous - 1.0e-6, "a fast turn must not stall");
            previous = step;
        }
        assert_eq!(filter.quarantined_samples(), 0);
    }

    #[test]
    fn accepted_40_and_70_degree_steps_follow_the_same_fraction() {
        // With the same time constant, elapsed time, and zero initial velocity,
        // these two accepted steps should cover the same fraction of their
        // target angle. The filter must not increase damping for the larger step.
        let followed = |degrees: f32| {
            let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
            let identity = UnitQuaternion::identity();
            let _ = filter.update(identity, ts(0));
            let target = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), degrees.to_radians());
            let out = filter.update(target, ts(16_666_667));
            out.angle_to(&identity) / identity.angle_to(&target)
        };
        // Both the 40-degree and 70-degree targets are within 1.25 rad of the
        // initial filtered rotation, so both are smoothed rather than rejected.
        let small = followed(40.0);
        let large = followed(70.0);
        assert!(
            (small - large).abs() < 1.0e-3,
            "the followed fraction must not depend on the departure: {small} vs {large}"
        );
        assert!(small > 0.0 && small < 0.5);
    }
}
