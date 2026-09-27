//! Shared critically damped second-order step for tracked-channel smoothing.
//!
//! Head rotation and observed-arm positions run on the same render-clock
//! integrator so every tracked channel reconstructs a continuous signal from a
//! held observation with one dynamic response.
//!
//! The response is a fixed per-channel time constant. It is deliberately not
//! stretched by how fast the observations move: a hand that really is moving
//! fast is not an outlier, and treating its speed as evidence of a jump slowed
//! exactly the motion that needed to follow. Discontinuities are handled where
//! they can be told apart from motion — the head filter quarantines a
//! single-observation rotation beyond its physical limit, and the arm tracker
//! quarantines a wrist step beyond the calibrated arm length — and a lost
//! channel is eased back by the shared loss blend rather than by stiffening the
//! filter.

use nalgebra::Vector3;

/// Default maximum accepted delta-time in seconds.
///
/// Larger gaps are clamped so a stale observation cannot fully snap the output.
pub const DEFAULT_MAX_DT_SEC: f32 = 0.5;

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
    let (cx, vx) = critically_damped_step_scalar(error.x, velocity.x, dt_sec, time_constant_sec);
    let (cy, vy) = critically_damped_step_scalar(error.y, velocity.y, dt_sec, time_constant_sec);
    let (cz, vz) = critically_damped_step_scalar(error.z, velocity.z, dt_sec, time_constant_sec);
    (Vector3::new(cx, cy, cz), Vector3::new(vx, vy, vz))
}

/// [`critically_damped_step`] for a single scalar channel, such as a finger
/// joint's flexion angle.
#[must_use]
pub fn critically_damped_step_scalar(
    error: f32,
    velocity: f32,
    dt_sec: f32,
    time_constant_sec: f32,
) -> (f32, f32) {
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
    fn a_fixed_response_steps_the_same_way_regardless_of_the_error_size() {
        // The response is a linear function of the error, so the correction a
        // channel applies is proportional to how far it still has to travel. A
        // large error therefore takes longer in absolute terms, but nothing
        // about its size stiffens the filter.
        let dt = 1.0 / 60.0;
        let (small, _) = critically_damped_step_scalar(0.1, 0.0, dt, 0.05);
        let (large, _) = critically_damped_step_scalar(1.0, 0.0, dt, 0.05);
        assert!(
            (large - 10.0 * small).abs() < 1.0e-6,
            "the step must scale with the error: {large} vs {small}"
        );
    }

    #[test]
    fn a_constant_rate_ramp_converges_without_lag_that_grows_with_speed() {
        // A critically damped second-order response lags a constant-rate ramp by
        // 2 * tau * rate. The lag is therefore proportional to how fast the
        // subject moves, but the response never changes: doubling the speed
        // doubles the lag instead of the filter stiffening and the channel
        // falling arbitrarily further behind.
        let dt = 1.0 / 60.0;
        let lag_after = |rate: f32| {
            let tau = 0.05;
            let mut error = 0.0;
            let mut velocity = 0.0;
            for _ in 0..600 {
                error -= rate * dt;
                let (correction, next) = critically_damped_step_scalar(error, velocity, dt, tau);
                error -= correction;
                velocity = next;
            }
            error.abs()
        };
        let slow = lag_after(1.0);
        let fast = lag_after(4.0);
        assert!(
            (fast - 4.0 * slow).abs() < 1.0e-3,
            "lag must scale linearly with a constant rate: {fast} vs {slow}"
        );
        assert!(
            (slow - 0.1).abs() < 1.0e-2,
            "a 1 unit/s ramp must settle at 2 * tau of lag: {slow}"
        );
    }

    #[test]
    fn a_held_observation_holds_the_settled_value() {
        // Re-feeding the same target is the render-clock case: the output stays
        // at the observation instead of drifting.
        let dt = 1.0 / 60.0;
        let (correction, _) = critically_damped_step_scalar(0.0, 0.0, dt, 0.05);
        assert_eq!(correction, 0.0);
    }
}
