//! Quaternion-centered SO(3) biquad smoothing for head rotation.
//!
//! The filter operates directly on [`UnitQuaternion`] values in the canonical
//! tracking basis described in `DESIGN.md` §11.6. Smoothing in quaternion
//! space avoids Euler-angle wrapping, gimbal-lock singularities, and
//! independent per-axis low-pass artefacts.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use vtuber_core::types::MonoTimeNs;

/// Parameters for the head rotation filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadFilterParams {
    /// Smoothing time constant in seconds while observations continue the
    /// motion the filter is already following.
    ///
    /// Smaller values make the filter follow input changes faster. Values
    /// must be positive and finite; non-positive values are clamped to
    /// [`f32::EPSILON`] when the filter runs.
    pub time_constant_sec: f32,
    /// Smoothing time constant used when an observation jumps away from the
    /// previous one.
    ///
    /// The first observation after a detection jump (or a loss) eases with
    /// this value instead of the fast one, so the departure is absorbed. It is
    /// never smaller than [`Self::time_constant_sec`].
    pub slow_time_constant_sec: f32,
    /// Observation rate in radians per second at which the response has fully
    /// slowed to [`Self::slow_time_constant_sec`].
    ///
    /// A plausible head turn stays below this, so sustained motion keeps the
    /// fast response; only a discontinuity slows the filter down.
    pub jump_rate_rad_per_sec: f32,
    /// Maximum allowed delta-time in seconds.
    ///
    /// Larger gaps are clamped to this value so that a stale observation
    /// cannot fully snap the output.
    pub max_dt_sec: f32,
    /// Maximum accepted rotation step in radians.
    ///
    /// A larger single-frame step is treated as an outlier and quarantined
    /// until a later sample returns within the physical limit.
    pub max_step_rad: f32,
}

impl Default for HeadFilterParams {
    fn default() -> Self {
        Self {
            time_constant_sec: 0.025,
            slow_time_constant_sec: 0.1,
            jump_rate_rad_per_sec: 8.0,
            max_dt_sec: super::damped::DEFAULT_MAX_DT_SEC,
            max_step_rad: 1.25,
        }
    }
}

impl HeadFilterParams {
    /// Returns parameters with a fixed smoothing time constant.
    ///
    /// The jump response is disabled, so the filter smooths at
    /// `time_constant_sec` regardless of how far an observation departs from
    /// the previous one.
    #[must_use]
    pub fn with_time_constant(time_constant_sec: f32) -> Self {
        Self {
            time_constant_sec,
            slow_time_constant_sec: time_constant_sec,
            ..Self::default()
        }
    }

    /// The residual-adaptive response implied by these parameters.
    fn response(self) -> super::damped::ResidualResponse {
        super::damped::ResidualResponse::new(
            self.time_constant_sec,
            self.slow_time_constant_sec,
            self.jump_rate_rad_per_sec,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FilterState {
    quat: UnitQuaternion<f32>,
    velocity: Vector3<f32>,
    last_time: MonoTimeNs,
    /// The observation the filter last received, used to measure how far the
    /// next one departs from it.
    last_target: UnitQuaternion<f32>,
    /// When `last_target` first changed, so the departure is measured over the
    /// observation interval rather than over one render tick.
    last_target_time: MonoTimeNs,
}

/// Quaternion-centered exponential smoothing filter for head rotation.
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
    /// For this exponential smoothing filter this is equivalent to
    /// [`reset`](Self::reset). The next observation becomes the new initial
    /// state.
    pub fn reacquire(&mut self) {
        self.reset();
    }

    /// Number of single-frame target rotations rejected as outliers.
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
                last_target: target,
                last_target_time: timestamp,
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

        // How far this observation departs from the previous one, per second
        // of observation interval. A held sample repeats `last_target`, so it
        // contributes no departure and keeps the fast response.
        let observation_changed = target != state.last_target;
        let departure = if observation_changed {
            super::damped::observation_rate(
                state.last_target.angle_to(&target),
                (timestamp.0.saturating_sub(state.last_target_time.0) as f32) * 1.0e-9,
            )
        } else {
            0.0
        };
        let time_constant_sec = self.params.response().time_constant_sec(departure);

        // Clamp tau to avoid division by zero and non-finite parameters.
        let tau = time_constant_sec.max(f32::EPSILON);

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
            so3_biquad_step(state.quat, signed_target, state.velocity, dt_sec, tau);

        self.state = Some(FilterState {
            quat: smoothed,
            velocity,
            last_time: timestamp,
            last_target: if observation_changed {
                target
            } else {
                state.last_target
            },
            last_target_time: if observation_changed {
                timestamp
            } else {
                state.last_target_time
            },
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

fn so3_biquad_step(
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

    /// Feeds a steady turn at 30 Hz, holding each observation for two 60 Hz
    /// render ticks, and returns the lag after the last observation.
    fn sustained_turn_lag(filter: &mut HeadRotationFilter) -> f32 {
        let observation_step_sec = 1.0 / 30.0;
        let half_step_ns = (observation_step_sec * 0.5e9) as u64;
        let turn_per_observation = 2.0 * observation_step_sec;
        let mut target = UnitQuaternion::identity();
        let mut now = 0u64;
        let _ = filter.update(target, ts(now));
        let mut out = target;
        for _ in 0..60 {
            target *= UnitQuaternion::from_axis_angle(&Vector3::y_axis(), turn_per_observation);
            now += half_step_ns;
            let _ = filter.update(target, ts(now));
            now += half_step_ns;
            out = filter.update(target, ts(now));
        }
        out.angle_to(&target)
    }

    #[test]
    fn a_sustained_turn_lags_less_than_the_slow_response() {
        let mut adaptive = HeadRotationFilter::new(HeadFilterParams::default());
        let mut slow = HeadRotationFilter::new(HeadFilterParams::with_time_constant(0.1));
        let adaptive_lag = sustained_turn_lag(&mut adaptive);
        let slow_lag = sustained_turn_lag(&mut slow);
        assert!(
            adaptive_lag < slow_lag * 0.6,
            "a plausible turn must use the fast response: adaptive={adaptive_lag}, slow={slow_lag}"
        );
    }

    #[test]
    fn a_single_departure_is_absorbed() {
        let mut filter = HeadRotationFilter::new(HeadFilterParams::default());
        let identity = UnitQuaternion::identity();
        let _ = filter.update(identity, ts(0));
        // A 40-degree one-observation departure stays below the quarantine
        // limit, so only the adaptive response can absorb it.
        let departure = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 40.0f32.to_radians());
        let out = filter.update(departure, ts(16_666_667));
        let followed = out.angle_to(&identity);
        assert!(
            followed < identity.angle_to(&departure) * 0.1,
            "a detection jump must not reach the avatar in one frame: {followed}"
        );
    }
}
