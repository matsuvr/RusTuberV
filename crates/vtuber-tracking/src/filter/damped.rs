//! Shared critically damped second-order step for tracked-channel smoothing.
//!
//! Head rotation and observed-arm positions run on the same render-clock
//! integrator so every tracked channel reconstructs a continuous signal from a
//! held observation with one dynamic response.

use nalgebra::Vector3;

/// Default smoothing time constant in seconds, shared by tracked channels.
pub const DEFAULT_TIME_CONSTANT_SEC: f32 = 0.05;

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
    let omega = 1.0 / time_constant_sec.max(f32::EPSILON);
    let decay = (-omega * dt_sec).exp();
    let combined = velocity + error * omega;
    let new_error = (error + combined * dt_sec) * decay;
    let new_velocity = (velocity - combined * (omega * dt_sec)) * decay;
    (error - new_error, new_velocity)
}
