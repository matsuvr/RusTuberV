//! Shared critically damped second-order step for tracked-channel smoothing.
//!
//! Head rotation and observed-arm positions run on the same render-clock
//! integrator so every tracked channel reconstructs a continuous signal from a
//! held observation with one dynamic response.

use nalgebra::Vector3;

/// Default maximum accepted delta-time in seconds.
///
/// Larger gaps are clamped so a stale observation cannot fully snap the output.
pub const DEFAULT_MAX_DT_SEC: f32 = 0.5;

/// Residual-adaptive smoothing response shared by tracked channels.
///
/// A channel whose observations continue the motion it is already following
/// stays at `fast_sec`, so healthy tracking follows with the agility the
/// subject's own motion allows. When an observation departs from the previous
/// one faster than `jump_rate_per_sec` — a detection jump, or the first frame
/// after a loss — the response eases toward `slow_sec`, so the departure is
/// absorbed instead of reaching the avatar as a snap. Because the residual is
/// measured between consecutive observations (not against the smoothed state),
/// a sustained fast motion keeps a small residual and stays agile; only a
/// discontinuity slows the channel down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ResidualResponse {
    /// Time constant used while observations follow the previous one.
    fast_sec: f32,
    /// Time constant used when an observation jumps at `jump_rate_per_sec` or
    /// faster.
    slow_sec: f32,
    /// Observation rate, in channel units per second, at which the response has
    /// fully slowed to `slow_sec`.
    jump_rate_per_sec: f32,
}

impl ResidualResponse {
    pub(crate) const fn new(fast_sec: f32, slow_sec: f32, jump_rate_per_sec: f32) -> Self {
        Self {
            fast_sec,
            slow_sec,
            jump_rate_per_sec,
        }
    }

    /// A response that never adapts, for callers that want fixed smoothing.
    #[cfg(test)]
    pub(crate) const fn fixed(time_constant_sec: f32) -> Self {
        Self::new(time_constant_sec, time_constant_sec, 0.0)
    }

    /// Time constant for an observation departing at `rate_per_sec`.
    pub(crate) fn time_constant_sec(self, rate_per_sec: f32) -> f32 {
        let fast = self.fast_sec.max(f32::EPSILON);
        let slow = self.slow_sec.max(fast);
        if !self.jump_rate_per_sec.is_finite() || self.jump_rate_per_sec <= 0.0 {
            return fast;
        }
        // A non-finite rate (a changed observation with no elapsed time) is
        // treated as a jump so the departure is absorbed rather than followed.
        let rate = if rate_per_sec.is_finite() {
            rate_per_sec
        } else {
            self.jump_rate_per_sec
        };
        let factor = (rate / self.jump_rate_per_sec).clamp(0.0, 1.0);
        fast + (slow - fast) * factor
    }
}

/// Observation rate in channel units per second, or infinity when a change
/// happened with no elapsed time.
#[must_use]
pub(crate) fn observation_rate(distance: f32, dt_sec: f32) -> f32 {
    if !distance.is_finite() || distance <= 0.0 {
        return 0.0;
    }
    if !dt_sec.is_finite() || dt_sec <= 0.0 {
        return f32::INFINITY;
    }
    distance / dt_sec
}

/// One critically damped second-order step toward a target error.
///
/// `error` is the target minus the current value expressed in the channel's own
/// space, `velocity` is the retained derivative of that space, and
/// `time_constant_sec` is the inverse bandwidth. Returns the correction to add
/// to the current value and the new velocity.
#[must_use]
pub fn critically_damped_step(
    error: Vector3<f32>,
    velocity: Vector3<f32>,
    dt_sec: f32,
    time_constant_sec: f32,
) -> (Vector3<f32>, Vector3<f32>) {
    let omega = 1.0 / time_constant_sec.max(f32::EPSILON);
    let decay = (-omega * dt_sec).exp();
    let combined = velocity + error * omega;
    let new_error = (error + combined * dt_sec) * decay;
    let new_velocity = (velocity - combined * (omega * dt_sec)) * decay;
    (error - new_error, new_velocity)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn a_smooth_observation_keeps_the_fast_response() {
        let response = ResidualResponse::new(0.02, 0.1, 8.0);
        assert_eq!(response.time_constant_sec(0.0), 0.02);
        let quarter = response.time_constant_sec(2.0);
        assert!((quarter - 0.04).abs() < 1.0e-6, "{quarter}");
    }

    #[test]
    fn a_jump_reaches_the_slow_response() {
        let response = ResidualResponse::new(0.02, 0.1, 8.0);
        assert!((response.time_constant_sec(8.0) - 0.1).abs() < 1.0e-6);
        assert!((response.time_constant_sec(80.0) - 0.1).abs() < 1.0e-6);
        let mid = response.time_constant_sec(4.0);
        assert!((mid - 0.06).abs() < 1.0e-6, "{mid}");
    }

    #[test]
    fn fixed_response_ignores_the_rate() {
        let response = ResidualResponse::fixed(0.05);
        assert_eq!(response.time_constant_sec(0.0), 0.05);
        assert_eq!(response.time_constant_sec(100.0), 0.05);
    }

    #[test]
    fn zero_dt_departure_counts_as_a_jump() {
        assert_eq!(observation_rate(0.1, 0.0), f32::INFINITY);
        assert_eq!(observation_rate(0.1, 0.1), 1.0);
        assert_eq!(observation_rate(0.0, 0.1), 0.0);
    }
}
