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

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Default maximum accepted delta-time in seconds.
///
/// Larger gaps are clamped so a stale observation cannot fully snap the output.
pub const DEFAULT_MAX_DT_SEC: f32 = 0.5;

/// One critically damped second-order step toward a target error.
///
/// `error` is a residual in the channel's coordinate space, `velocity` is its
/// retained derivative, and `time_constant_sec` is the inverse bandwidth.
/// Returns the amount removed from the residual and its new derivative.
/// Scalar/position callers use target minus current and add the correction;
/// world rotations use current relative to target and reconstruct the residual.
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

/// Retained state for a scalar joint coordinate. Anatomical limits belong to
/// the caller, which projects `value` and, where required, `velocity` afterward.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScalarSpring {
    /// Current scalar coordinate.
    pub value: f32,
    /// Retained error derivative.
    pub velocity: f32,
}

impl ScalarSpring {
    /// Seeds a coordinate at its first measurement with no residual motion.
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self {
            value,
            velocity: 0.0,
        }
    }

    /// Non-finite measurements hold the state, as for observed finger angles.
    pub fn step(&mut self, target: f32, dt_sec: f32, time_constant_sec: f32) -> f32 {
        if !target.is_finite() {
            return self.value;
        }
        let (correction, velocity) = critically_damped_step_scalar(
            target - self.value,
            self.velocity,
            dt_sec,
            time_constant_sec,
        );
        self.value += correction;
        self.velocity = velocity;
        self.value
    }
}

/// Retained state for positions or groups of independent joint coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorSpring {
    /// Current position or group of joint coordinates.
    pub position: Vector3<f32>,
    /// Retained error derivative in the same coordinate space.
    velocity: Vector3<f32>,
}

impl VectorSpring {
    /// Seeds at the first measurement with no residual motion.
    #[must_use]
    pub fn new(position: Vector3<f32>) -> Self {
        Self {
            position,
            velocity: Vector3::zeros(),
        }
    }

    /// Advances in the channel's coordinate space.
    pub fn step(&mut self, target: Vector3<f32>, dt_sec: f32, time_constant_sec: f32) {
        let (correction, velocity) = critically_damped_step(
            target - self.position,
            self.velocity,
            dt_sec,
            time_constant_sec,
        );
        self.position += correction;
        self.velocity = velocity;
    }
}

/// Rotation-vector spring state. Error coordinates stay in the frame chosen
/// by the channel; local head motion and world-frame shoulder motion must not
/// silently exchange those frames while retaining their error derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationSpring {
    /// Current unit rotation in the channel's coordinate frame.
    pub value: UnitQuaternion<f32>,
    velocity: Vector3<f32>,
}

impl RotationSpring {
    /// Seeds at the first measurement with no residual motion.
    #[must_use]
    pub fn new(value: UnitQuaternion<f32>) -> Self {
        Self {
            value,
            velocity: Vector3::zeros(),
        }
    }

    /// Head response: advance the local error from the current rotation.
    pub fn step_local(
        &mut self,
        target: UnitQuaternion<f32>,
        dt_sec: f32,
        tau: f32,
    ) -> UnitQuaternion<f32> {
        let target = shortest_arc(self.value, target);
        let error = (self.value.inverse() * target).scaled_axis();
        let (correction, velocity) = critically_damped_step(error, self.velocity, dt_sec, tau);
        self.value *= UnitQuaternion::from_scaled_axis(correction);
        self.velocity = velocity;
        self.value
    }

    /// Shoulder response: decay the world-frame residual onto the target.
    pub fn step_world(
        &mut self,
        target: UnitQuaternion<f32>,
        dt_sec: f32,
        tau: f32,
    ) -> UnitQuaternion<f32> {
        let target = shortest_arc(self.value, target);
        let error = (self.value * target.inverse()).scaled_axis();
        let (correction, velocity) = critically_damped_step(error, self.velocity, dt_sec, tau);
        self.value = UnitQuaternion::from_scaled_axis(error - correction) * target;
        self.value.renormalize();
        self.velocity = velocity;
        self.value
    }
}

/// Quaternion sign with the shortest arc from the current rotation.
#[must_use]
pub fn shortest_arc(
    current: UnitQuaternion<f32>,
    target: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    if current.quaternion().coords.dot(&target.quaternion().coords) < 0.0 {
        let q = target.quaternion();
        UnitQuaternion::from_quaternion(Quaternion::new(-q.w, -q.i, -q.j, -q.k))
    } else {
        target
    }
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

    #[test]
    fn seeded_scalar_and_rotation_responses_agree_across_render_rates() {
        // A held one-axis target has the same critical response in every
        // coordinate representation, including both rotation error frames.
        let initial = UnitQuaternion::from_scaled_axis(Vector3::new(0.3, -0.2, 0.1));
        let target = initial * UnitQuaternion::from_scaled_axis(Vector3::y() * 0.8);
        for fps in [30, 60, 120] {
            let mut scalar = ScalarSpring::new(0.0);
            let mut local = RotationSpring::new(initial);
            let mut world = RotationSpring::new(initial);
            for _ in 0..fps {
                scalar.step(0.8, 1.0 / fps as f32, 0.15);
                local.step_local(target, 1.0 / fps as f32, 0.15);
                world.step_world(target, 1.0 / fps as f32, 0.15);
            }
            let expected_residual = 0.8 * (1.0 + 1.0 / 0.15) * (-1.0_f32 / 0.15).exp();
            assert!((0.8 - scalar.value - expected_residual).abs() < 2.0e-6);
            assert!((local.value.angle_to(&target) - expected_residual).abs() < 2.0e-6);
            assert!((world.value.angle_to(&target) - expected_residual).abs() < 2.0e-6);
        }
    }

    #[test]
    fn multi_axis_rotation_response_preserves_basis_and_quaternion_sign() {
        let basis = UnitQuaternion::from_scaled_axis(Vector3::new(0.6, -0.9, 0.4));
        for world_frame in [false, true] {
            let mut original = RotationSpring::new(UnitQuaternion::identity());
            let mut transformed = RotationSpring::new(basis);
            for tick in 0..180 {
                let t = tick as f32 / 60.0;
                let target = UnitQuaternion::from_scaled_axis(Vector3::new(
                    0.4 * t.sin(),
                    0.6 * (t * 0.7).sin(),
                    0.3 * (t * 1.3).cos(),
                ));
                let transformed_target = basis * target;
                let q = transformed_target.quaternion();
                let transformed_target = if tick % 2 == 0 {
                    UnitQuaternion::from_quaternion(Quaternion::new(-q.w, -q.i, -q.j, -q.k))
                } else {
                    transformed_target
                };
                if world_frame {
                    original.step_world(target, 1.0 / 60.0, 0.15);
                    transformed.step_world(transformed_target, 1.0 / 60.0, 0.15);
                } else {
                    original.step_local(target, 1.0 / 60.0, 0.025);
                    transformed.step_local(transformed_target, 1.0 / 60.0, 0.025);
                }
                assert!(transformed.value.angle_to(&(basis * original.value)) < 2.0e-5);
                assert!((transformed.value.norm() - 1.0).abs() < 2.0e-5);
            }
        }
    }
}
