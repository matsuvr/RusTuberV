//! Critically damped second-order smoothing for head rotation on SO(3).
//!
//! The filter operates directly on [`UnitQuaternion`] values in the canonical
//! tracking basis used by [`crate::pose`]. Smoothing in quaternion space avoids
//! Euler-angle wrapping, gimbal-lock singularities, and independent per-axis
//! low-pass artefacts.

use nalgebra::UnitQuaternion;
use vtuber_core::types::MonoTimeNs;

use super::{
    damped::{RotationSpring, shortest_arc},
    time::elapsed_seconds,
};

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
    /// Angular departure from the filtered rotation that requires confirmation.
    /// Measured in radians.
    ///
    /// A target farther from the current smoothed rotation is rejected while
    /// the previous accepted response continues. The comparison uses the
    /// filtered state, not the preceding raw observation. A later
    /// target is accepted once it falls within this limit, or after three
    /// consecutive observations stay within 0.15 radians of the first rejected
    /// heading. Reconnection after an absence explicitly accepts the new heading.
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
    spring: RotationSpring,
    target: UnitQuaternion<f32>,
    last_time: MonoTimeNs,
}

const RECOVERY_OBSERVATIONS: u8 = 3;
const RECOVERY_SPREAD_RAD: f32 = 0.15;

#[derive(Clone, Copy, Debug, PartialEq)]
struct RejectedHeading {
    anchor: UnitQuaternion<f32>,
    observations: u8,
}

/// Keeps a fixed anchor so a sequence of drifting outliers cannot accumulate
/// agreement merely by staying close to the preceding outlier.
fn observe_rejected_heading(
    previous: Option<RejectedHeading>,
    target: UnitQuaternion<f32>,
) -> RejectedHeading {
    match previous {
        Some(previous) if previous.anchor.angle_to(&target) <= RECOVERY_SPREAD_RAD => {
            RejectedHeading {
                observations: previous.observations.saturating_add(1),
                ..previous
            }
        }
        _ => RejectedHeading {
            anchor: target,
            observations: 1,
        },
    }
}

/// Quaternion-centered critically damped second-order filter for head rotation.
///
/// The filter maintains an internal quaternion state and a tangent-space
/// error derivative. On each update it applies a critically damped second
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
    rejected_heading: Option<RejectedHeading>,
    quarantined_samples: u64,
}

impl HeadRotationFilter {
    /// Creates a new filter with the given parameters.
    #[must_use]
    pub fn new(params: HeadFilterParams) -> Self {
        Self {
            params,
            state: None,
            rejected_heading: None,
            quarantined_samples: 0,
        }
    }

    /// Returns `true` if the filter has received at least one observation.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.state.is_some()
    }

    /// A new observation can initialize the spring or advance its render clock.
    pub(crate) fn can_observe_at(&self, timestamp: MonoTimeNs) -> bool {
        self.state.is_none_or(|state| timestamp > state.last_time)
    }

    /// Resets the filter, discarding all state.
    ///
    /// The next call to [`update`](Self::update) initializes the filter with
    /// that observation.
    pub fn reset(&mut self) {
        self.state = None;
        self.rejected_heading = None;
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

    /// Updates the filter with a new observed target rotation.
    ///
    /// Call this once per observation. Use [`advance`](Self::advance) on render
    /// ticks that reuse an observation so it cannot count toward recovery again.
    ///
    /// `timestamp` is expected to be monotonically non-decreasing. If it is
    /// older than the previous observation the elapsed time is treated as
    /// zero, so the output is the current state (or the input if the filter
    /// was just reset).
    ///
    /// # Arguments
    ///
    /// * `target` - Desired rotation in the canonical tracking basis.
    /// * `timestamp` - Monotonic time at which to advance the filter.
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
        let Some(mut state) = self.state else {
            self.state = Some(FilterState {
                spring: RotationSpring::new(target),
                target,
                last_time: timestamp,
            });
            return target;
        };

        // Compute elapsed seconds. `saturating_sub` clamps backwards
        // timestamps to zero and avoids overflow for very large differences.
        let dt_sec = elapsed_seconds(timestamp, state.last_time, self.params.max_dt_sec);

        // If dt is zero (same timestamp or backwards), keep the current
        // state. This also covers the zero/negative dt acceptance cases.
        if dt_sec <= 0.0 {
            return state.spring.value;
        }

        // Clamp tau away from zero before the damping step.
        let tau = self.params.time_constant_sec.max(f32::EPSILON);

        // Choose the quaternion sign that gives the shortest arc.
        let signed_target = shortest_arc(state.spring.value, target);
        let error = (state.spring.value.inverse() * signed_target).scaled_axis();
        let max_step = if self.params.max_step_rad.is_finite() {
            self.params.max_step_rad.max(0.0)
        } else {
            f32::MAX
        };
        if error.norm() > max_step {
            let rejected = observe_rejected_heading(self.rejected_heading, signed_target);
            if rejected.observations >= RECOVERY_OBSERVATIONS {
                // Use the existing smooth reconnection path, preserving both
                // rotation and angular velocity instead of reseeding the spring.
                return self.reconnect(signed_target, timestamp);
            }
            self.rejected_heading = Some(rejected);
            self.quarantined_samples = self.quarantined_samples.saturating_add(1);
            return self.advance(timestamp).unwrap_or(state.spring.value);
        }
        self.rejected_heading = None;

        // A critically damped second-order response in the local rotation
        // vector is the SO(3) equivalent of a biquad. The quaternion remains
        // on SO(3); only the tangent-space error and velocity are filtered.
        let smoothed = state.spring.step_local(signed_target, dt_sec, tau);
        state.target = signed_target;
        state.last_time = timestamp;
        self.state = Some(state);

        smoothed
    }

    /// Continues the existing response toward the last accepted observation.
    /// Missing or rejected observations never become an extrapolation target.
    pub fn advance(&mut self, timestamp: MonoTimeNs) -> Option<UnitQuaternion<f32>> {
        let state = self.state.as_mut()?;
        let dt = elapsed_seconds(timestamp, state.last_time, self.params.max_dt_sec);
        if dt > 0.0 {
            state.spring.step_local(
                state.target,
                dt,
                self.params.time_constant_sec.max(f32::EPSILON),
            );
            state.last_time = timestamp;
        }
        Some(state.spring.value)
    }

    /// Ends an interrupted run of rejected observations without stopping motion.
    pub(crate) fn interrupt_recovery(&mut self) {
        self.rejected_heading = None;
    }

    /// Accepts a recovered heading without discarding momentum.
    /// Used after a tracking absence or confirmation of a large new heading.
    pub fn reconnect(
        &mut self,
        target: UnitQuaternion<f32>,
        timestamp: MonoTimeNs,
    ) -> UnitQuaternion<f32> {
        self.rejected_heading = None;
        if let Some(state) = self.state.as_mut() {
            state.target = target;
            self.advance(timestamp).unwrap_or(target)
        } else {
            self.update(target, timestamp)
        }
    }

    /// Returns the current smoothed rotation without advancing the filter.
    ///
    /// Returns `None` if the filter has not been initialized.
    #[must_use]
    pub fn current(&self) -> Option<UnitQuaternion<f32>> {
        self.state.map(|s| s.spring.value)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    fn ts(ns: u64) -> MonoTimeNs {
        MonoTimeNs(ns)
    }

    #[test]
    fn missing_frames_finish_the_accepted_motion_at_each_render_rate() {
        for hz in [30u64, 60, 120] {
            let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
            let _ = filter.update(UnitQuaternion::identity(), ts(0));
            let target = UnitQuaternion::from_euler_angles(0.2, 0.6, -0.1);
            let first = filter.update(target, ts(16_000_000));
            let mut previous = first.angle_to(&target);
            for frame in 1..=hz * 10 {
                let out = filter
                    .advance(ts(16_000_000 + frame * 1_000_000_000 / hz))
                    .unwrap();
                let distance = out.angle_to(&target);
                assert!(distance <= previous + 1e-5, "{hz} Hz frame {frame}");
                previous = distance;
            }
            assert!(
                previous < 1e-5,
                "last accepted heading must survive prolonged loss"
            );
            filter.reset();
            assert!(filter.advance(ts(11_000_000_000)).is_none());
        }
    }

    #[test]
    fn a_rejected_target_does_not_stop_or_redirect_accepted_motion() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let _ = filter.update(UnitQuaternion::identity(), ts(0));
        let target = UnitQuaternion::from_euler_angles(0.0, 0.5, 0.0);
        let first = filter.update(target, ts(16_000_000));
        let mut missing = filter.clone();
        let out = filter.update(
            UnitQuaternion::from_euler_angles(0.0, -2.0, 0.0),
            ts(32_000_000),
        );
        let expected = missing.advance(ts(32_000_000)).unwrap();
        assert!(out.angle_to(&expected) < 1e-6);
        assert!(out.angle_to(&target) < first.angle_to(&target));
        assert_eq!(filter.quarantined_samples(), 1);
    }

    #[test]
    fn reconnect_keeps_momentum_and_can_reach_a_large_new_heading() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let _ = filter.update(UnitQuaternion::identity(), ts(0));
        let target = UnitQuaternion::from_euler_angles(0.0, 0.5, 0.0);
        let _ = filter.update(target, ts(16_000_000));
        let mut uninterrupted = filter.clone();
        let out = filter.reconnect(target, ts(32_000_000));
        assert!(out.angle_to(&uninterrupted.update(target, ts(32_000_000))) < 1e-6);
        let changed = UnitQuaternion::from_euler_angles(0.0, -1.2, 0.0);
        let first = filter.reconnect(changed, ts(48_000_000));
        assert!(first.angle_to(&changed) > 0.5, "reconnection must not snap");
        for frame in 1..=60 {
            let _ = filter.update(changed, ts(48_000_000 + frame * 16_666_667));
        }
        assert!(filter.current().unwrap().angle_to(&changed) < 1e-5);
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
    fn stable_rejected_headings_reconnect_without_discarding_momentum() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let _ = filter.update(UnitQuaternion::identity(), ts(0));
        let _ = filter.update(
            UnitQuaternion::from_euler_angles(0.0, 0.5, 0.0),
            ts(16_000_000),
        );
        for (time, yaw) in [(32_000_000, -1.5), (48_000_000, -1.55)] {
            let _ = filter.update(UnitQuaternion::from_euler_angles(0.0, yaw, 0.0), ts(time));
        }
        assert_eq!(filter.quarantined_samples(), 2);

        let target = UnitQuaternion::from_euler_angles(0.0, -1.48, 0.0);
        // The sign of a quaternion must not break agreement about a heading.
        let signed_target = UnitQuaternion::from_quaternion(-*target.quaternion());
        let mut expected = filter.clone();
        let expected = expected.reconnect(signed_target, ts(64_000_000));
        let recovered = filter.update(signed_target, ts(64_000_000));
        assert!(recovered.angle_to(&expected) < 1e-6);
        assert!(recovered.angle_to(&target) > 0.5, "recovery must not snap");
        assert_eq!(filter.quarantined_samples(), 2);

        let settled = filter.advance(ts(564_000_000)).unwrap();
        assert!(settled.angle_to(&target) < 1e-5);
    }

    #[test]
    fn drifting_outliers_cannot_confirm_a_heading_by_following_each_other() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let _ = filter.update(UnitQuaternion::identity(), ts(0));
        for sample in 0..8 {
            // Adjacent observations agree, but no three agree with one anchor.
            let yaw = 1.5 + sample as f32 * 0.12;
            let out = filter.update(
                UnitQuaternion::from_euler_angles(0.0, yaw, 0.0),
                ts((sample + 1) * 33_333_333),
            );
            assert!(out.angle() < 1e-6);
        }
        assert_eq!(filter.quarantined_samples(), 8);
    }

    #[test]
    fn a_normal_observation_interrupts_a_rejected_run() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let identity = UnitQuaternion::identity();
        let outlier = UnitQuaternion::from_euler_angles(0.0, 1.5, 0.0);
        let _ = filter.update(identity, ts(0));
        let _ = filter.update(outlier, ts(16_000_000));
        let _ = filter.update(outlier, ts(32_000_000));
        let _ = filter.update(identity, ts(48_000_000));
        let out = filter.update(outlier, ts(64_000_000));
        assert!(out.angle() < 1e-6);
        assert_eq!(filter.quarantined_samples(), 3);
    }

    #[test]
    fn interruption_reset_and_reconnect_discard_pending_agreement() {
        let identity = UnitQuaternion::identity();
        let outlier = UnitQuaternion::from_euler_angles(0.0, 1.5, 0.0);
        let mut pending = HeadRotationFilter::new(HeadFilterParams::default());
        let _ = pending.update(identity, ts(0));
        let _ = pending.update(outlier, ts(16_000_000));
        let _ = pending.update(outlier, ts(32_000_000));

        let mut interrupted = pending.clone();
        interrupted.interrupt_recovery();
        assert!(interrupted.update(outlier, ts(48_000_000)).angle() < 1e-6);

        for reset in [HeadRotationFilter::reset, HeadRotationFilter::reacquire] {
            let mut filter = pending.clone();
            reset(&mut filter);
            let _ = filter.update(identity, ts(48_000_000));
            assert!(filter.update(outlier, ts(64_000_000)).angle() < 1e-6);
            assert_eq!(filter.quarantined_samples(), 1);
        }

        let _ = pending.reconnect(identity, ts(48_000_000));
        assert!(pending.update(outlier, ts(64_000_000)).angle() < 1e-6);
    }

    #[test]
    fn duplicate_and_backwards_timestamps_do_not_confirm_a_rejected_heading() {
        let identity = UnitQuaternion::identity();
        let target = UnitQuaternion::from_euler_angles(0.0, 1.5, 0.0);
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let _ = filter.update(identity, ts(0));
        let _ = filter.update(target, ts(33_333_333));
        for _ in 0..30 {
            assert!(filter.update(target, ts(33_333_333)).angle() < 1e-6);
            assert!(filter.update(target, ts(16_666_667)).angle() < 1e-6);
        }
        assert_eq!(filter.quarantined_samples(), 1);
        assert!(filter.update(target, ts(66_666_666)).angle() < 1e-6);
        assert!(filter.update(target, ts(99_999_999)).angle() > 0.0);
        assert_eq!(filter.quarantined_samples(), 2);
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
