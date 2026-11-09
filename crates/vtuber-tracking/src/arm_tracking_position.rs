//! Position response for an observed wrist or elbow, in calibrated arm lengths.
//!
//! Quiet motion keeps the existing two-pole response. A larger planar tracking
//! error increases the bandwidth continuously; depth keeps its fixed response
//! so wrist depth noise does not turn into elbow flexion. This is a response
//! curve, not an observation gate: every adopted target is still followed.

use nalgebra::Vector3;

/// Errors within five percent of the calibrated arm length keep the quiet
/// response, including the existing stationary-camera noise fixtures.
const QUIET_PLANAR_ERROR: f32 = 0.05;
/// The response reaches its fastest setting at twenty percent of arm length.
const FAST_PLANAR_ERROR: f32 = 0.20;
/// Fast planar response; palm, depth and shoulder responses are independent.
const FAST_TIME_CONSTANT_SEC: f32 = 0.04;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ArmPositionSmoother {
    pub(super) position: Vector3<f32>,
    first_stage: Vector3<f32>,
    /// Remaining quiet response after observed control takes over from the virtual arm.
    settling_sec: f32,
}

impl ArmPositionSmoother {
    pub(super) fn new(position: Vector3<f32>) -> Self {
        Self {
            position,
            first_stage: position,
            settling_sec: 0.0,
        }
    }

    pub(super) fn step(&mut self, target: Vector3<f32>, dt_sec: f32, quiet_tau: f32, weight: f32) {
        if weight < 1.0 {
            // Six time constants leave less than 2% of a held step in the
            // two-pole quiet filter. Bound takeover even for continuous motion.
            self.settling_sec = 6.0 * quiet_tau;
        }
        if dt_sec == 0.0 {
            return;
        }
        let error = target - self.position;
        let distance = error.x.hypot(error.y);
        if weight >= 1.0 {
            self.settling_sec = if distance <= QUIET_PLANAR_ERROR {
                0.0
            } else {
                (self.settling_sec - dt_sec).max(0.0)
            };
        }
        let planar_tau = if self.settling_sec > 0.0 {
            quiet_tau
        } else {
            planar_time_constant(distance, quiet_tau)
        };
        let time_constants = Vector3::new(planar_tau, planar_tau, quiet_tau);
        for (((position, first), target), tau) in self
            .position
            .iter_mut()
            .zip(self.first_stage.iter_mut())
            .zip(target.iter())
            .zip(time_constants.iter())
        {
            (*position, *first) = cascade_step(*position, *first, *target, dt_sec, *tau);
        }
    }
}

/// Smoothly increases the planar bandwidth as the accepted target moves away.
/// X/Y share a response, so rotating the camera-plane axes cannot change it.
fn planar_time_constant(distance: f32, quiet_tau: f32) -> f32 {
    let amount = ((distance - QUIET_PLANAR_ERROR) / (FAST_PLANAR_ERROR - QUIET_PLANAR_ERROR))
        .clamp(0.0, 1.0);
    let amount = amount * amount * (3.0 - 2.0 * amount);
    quiet_tau + (FAST_TIME_CONSTANT_SEC - quiet_tau) * amount
}

/// Exact held-target step of two equal first-order poles in series.
///
/// At fixed tau this is the existing critically damped response. Keeping the
/// two pole positions instead of a velocity lets tau change without inheriting
/// momentum from another bandwidth. The new output is a convex combination of
/// the old output, first pole and target: a held step cannot overshoot even
/// while its response returns to the quiet setting.
fn cascade_step(position: f32, first: f32, target: f32, dt: f32, tau: f32) -> (f32, f32) {
    let elapsed = dt / tau;
    let decay = (-elapsed).exp();
    let first_error = first - target;
    (
        target + (position - target + first_error * elapsed) * decay,
        target + first_error * decay,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::damped::VectorSpring;

    #[test]
    fn quiet_motion_matches_the_existing_response_at_each_render_rate() {
        for fps in [30, 60, 120] {
            for tau in [0.125, 0.15] {
                let initial = Vector3::new(0.3, -0.2, 0.4);
                let mut actual = ArmPositionSmoother::new(initial);
                let mut reference = VectorSpring::new(initial);
                for tick in 0..fps * 8 {
                    let sample = tick / (fps / 30);
                    let t = sample as f32 / 30.0;
                    let offset = |phase: f32| {
                        0.006 * (std::f32::consts::TAU * 0.5 * t + phase).sin()
                            + 0.004 * (std::f32::consts::TAU * 3.0 * t + phase).sin()
                            + 0.003 * (std::f32::consts::TAU * 7.0 * t + phase).sin()
                    };
                    let target = initial + Vector3::new(offset(0.0), offset(0.7), offset(1.4));
                    actual.step(target, 1.0 / fps as f32, tau, 1.0);
                    reference.step(target, 1.0 / fps as f32, tau);
                    assert!((actual.position - reference.position).norm() < 2.0e-6);
                }
            }
        }
    }

    #[test]
    fn moving_in_the_plane_does_not_release_depth_noise() {
        for fps in [30, 60, 120] {
            let mut actual = ArmPositionSmoother::new(Vector3::zeros());
            let mut reference = VectorSpring::new(Vector3::zeros());
            for tick in 0..fps * 4 {
                let t = (tick / (fps / 30)) as f32 / 30.0;
                let target = Vector3::new(
                    0.4 * (std::f32::consts::TAU * t).sin(),
                    0.2 * (std::f32::consts::TAU * t).cos(),
                    0.03 * (std::f32::consts::TAU * 3.0 * t).sin(),
                );
                actual.step(target, 1.0 / fps as f32, 0.125, 1.0);
                reference.step(target, 1.0 / fps as f32, 0.125);
                assert!((actual.position.z - reference.position.z).abs() < 2.0e-6);
            }
        }
    }

    #[test]
    fn a_large_planar_step_arrives_earlier_without_overshooting() {
        for fps in [30, 60, 120] {
            let mut actual = ArmPositionSmoother::new(Vector3::zeros());
            let mut reference = VectorSpring::new(Vector3::zeros());
            let target = Vector3::new(0.4, 0.0, 0.0);
            let mut previous = 0.0;
            for tick in 1..=fps * 3 {
                actual.step(target, 1.0 / fps as f32, 0.125, 1.0);
                reference.step(target, 1.0 / fps as f32, 0.125);
                assert!(actual.position.x >= previous);
                assert!(actual.position.x <= target.x);
                previous = actual.position.x;
                if tick == fps / 10 {
                    assert!(actual.position.x >= 0.2, "{fps} Hz: halfway within 100 ms");
                    assert!(reference.position.x < 0.1);
                }
                if tick == fps * 3 / 10 {
                    assert!(actual.position.x >= 0.36, "{fps} Hz: 90% within 300 ms");
                }
            }
            assert!((actual.position - target).norm() < 1.0e-6);
        }
    }

    #[test]
    fn stopping_and_reversing_stays_inside_the_observed_range() {
        for fps in [30, 60, 120] {
            let mut actual = ArmPositionSmoother::new(Vector3::zeros());
            for target in [0.4, -0.4, 0.0] {
                for _ in 0..fps {
                    actual.step(Vector3::new(target, 0.0, 0.0), 1.0 / fps as f32, 0.125, 1.0);
                    assert!((-0.4..=0.4).contains(&actual.position.x));
                    assert_eq!(actual.position.y, 0.0);
                    assert_eq!(actual.position.z, 0.0);
                }
            }
            assert!(actual.position.x.abs() < 0.001);
        }
    }

    #[test]
    fn takeover_keeps_the_quiet_response_until_the_position_catches_up() {
        for tau in [0.125, 0.15] {
            let mut actual = ArmPositionSmoother::new(Vector3::zeros());
            let mut reference = VectorSpring::new(Vector3::zeros());
            let target = Vector3::new(0.4, 0.2, 0.1);
            for weight in [0.0, 0.5, 0.99, 1.0, 1.0] {
                actual.step(target, 1.0 / 60.0, tau, weight);
                reference.step(target, 1.0 / 60.0, tau);
                assert!((actual.position - reference.position).norm() < 2.0e-6);
            }
            // The two responses diverge as soon as a settled, fully observed
            // arm begins a new movement, before the settling timeout expires.
            let near = actual.position;
            actual.step(near, 1.0 / 60.0, tau, 1.0);
            reference.step(near, 1.0 / 60.0, tau);
            for _ in 0..6 {
                actual.step(target, 1.0 / 60.0, tau, 1.0);
                reference.step(target, 1.0 / 60.0, tau);
            }
            assert!(actual.position.x > reference.position.x + 0.03);
        }
    }

    #[test]
    fn continuous_motion_can_accelerate_after_takeover_without_a_pause() {
        let mut actual = ArmPositionSmoother::new(Vector3::zeros());
        let mut reference = VectorSpring::new(Vector3::zeros());
        let dt = 1.0 / 60.0;
        for tick in 1..=60 {
            let target = Vector3::new(0.2 + tick as f32 * 0.02, 0.0, 0.0);
            let weight = if tick <= 4 { 0.5 } else { 1.0 };
            actual.step(target, dt, 0.125, weight);
            reference.step(target, dt, 0.125);
            if tick == 30 {
                assert!((actual.position - reference.position).norm() < 2.0e-6);
            }
        }
        assert!(actual.position.x > reference.position.x + 0.05);
    }

    #[test]
    fn zero_elapsed_time_keeps_both_poles_unchanged() {
        let mut actual = ArmPositionSmoother::new(Vector3::new(0.3, -0.2, 0.4));
        let previous = actual;
        actual.step(Vector3::new(-0.2, 0.3, 0.4), 0.0, 0.125, 1.0);
        assert_eq!(actual, previous);
    }
}
