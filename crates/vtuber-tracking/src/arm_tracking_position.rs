//! Shoulder-relative position response. Estimate planar velocity once per
//! accepted camera sample; integrate both position poles on the render clock.

use nalgebra::{Vector2, Vector3};
use vtuber_core::MonoTimeNs;

const FAST_TAU_SEC: f32 = 0.040;
const VELOCITY_TAU_SEC: f32 = 0.060;
/// Seconds squared per squared calibrated arm length.
const SPEED_GAIN: f32 = 2.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PlanarMotion {
    previous: Option<(Vector2<f32>, MonoTimeNs)>,
    velocity: Vector2<f32>,
}

impl PlanarMotion {
    fn observed(self, xy: Vector2<f32>, time: MonoTimeNs) -> Self {
        let velocity = match self.previous {
            Some((previous, captured)) if time > captured => {
                let dt = (time.0 - captured.0) as f32 * 1e-9;
                let alpha = -(-dt / VELOCITY_TAU_SEC).exp_m1();
                self.velocity + alpha * ((xy - previous) / dt - self.velocity)
            }
            Some(_) => return self,
            None => Vector2::zeros(),
        };
        Self {
            previous: Some((xy, time)),
            velocity,
        }
    }

    fn time_constant(self, quiet_tau: f32) -> f32 {
        (quiet_tau / (1.0 + SPEED_GAIN * self.velocity.norm_squared())).max(FAST_TAU_SEC)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ArmPositionSmoother {
    pub(super) position: Vector3<f32>,
    first_stage: Vector3<f32>,
    motion: PlanarMotion,
}

impl ArmPositionSmoother {
    pub(super) fn new(position: Vector3<f32>) -> Self {
        Self {
            position,
            first_stage: position,
            motion: PlanarMotion::default(),
        }
    }

    pub(super) fn observe(&mut self, target: Option<[f32; 3]>, time: MonoTimeNs) {
        self.motion = target.map_or_else(PlanarMotion::default, |[x, y, _]| {
            self.motion.observed(Vector2::new(x, y), time)
        });
    }

    pub(super) fn step(&mut self, target: Vector3<f32>, dt_sec: f32, quiet_tau: f32) {
        if dt_sec == 0.0 {
            return;
        }
        let planar_tau = self.motion.time_constant(quiet_tau);
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
    fn onset_stop_and_reversal_follow_camera_samples_at_each_render_rate() {
        for fps in [30, 60, 120] {
            for amplitude in [0.04, 0.4] {
                let mut actual = ArmPositionSmoother::new(Vector3::zeros());
                let mut quiet = VectorSpring::new(Vector3::zeros());
                for tick in 0..fps * 3 {
                    let sample = tick / (fps / 30);
                    let x = match sample {
                        0..=2 => 0.0,
                        3..=29 => amplitude,
                        _ => -amplitude,
                    };
                    let target = Vector3::new(x, 0.0, 0.02 * (sample as f32).sin());
                    if tick % (fps / 30) == 0 {
                        actual.observe(
                            Some(target.into()),
                            MonoTimeNs(sample as u64 * 1_000_000_000 / 30),
                        );
                    }
                    actual.step(target, 1.0 / fps as f32, 0.125);
                    quiet.step(target, 1.0 / fps as f32, 0.125);
                    if sample == 3 {
                        assert!(actual.position.x > 1.25 * quiet.position.x);
                    }
                    assert!(actual.position.x.abs() <= amplitude + 1e-6);
                    assert!((actual.position.z - quiet.position.z).abs() < 2e-6);
                }
                assert!((actual.position.x + amplitude).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn loss_breaks_velocity_but_retaining_a_sample_preserves_it() {
        let mut smoother = ArmPositionSmoother::new(Vector3::zeros());
        smoother.observe(Some([0.0; 3]), MonoTimeNs(0));
        smoother.observe(Some([0.1; 3]), MonoTimeNs(33_333_333));
        let moving = smoother.motion;
        assert!(moving.time_constant(0.125) < 0.125);
        smoother.step(Vector3::repeat(0.1), 1.0 / 60.0, 0.125);
        assert_eq!(smoother.motion, moving);
        smoother.observe(None, MonoTimeNs(66_666_667));
        smoother.observe(Some([-0.4; 3]), MonoTimeNs(99_999_999));
        assert_eq!(smoother.motion.time_constant(0.125), 0.125);
        smoother.observe(Some([-0.3; 3]), MonoTimeNs(133_333_333));
        assert!(smoother.motion.time_constant(0.125) < 0.125);
    }

    #[test]
    fn noise_during_sustained_motion_is_attenuated() {
        for fps in [30, 60, 120] {
            let mut clean = ArmPositionSmoother::new(Vector3::zeros());
            let mut noisy = clean;
            let mut input_energy = 0.0;
            let mut output_energy = 0.0;
            for tick in 0..fps * 2 {
                let sample = tick / (fps / 30);
                let t = sample as f32 / 30.0;
                let target = Vector3::new(0.6 * t, 0.0, 0.0);
                let noise = 0.008 * (std::f32::consts::TAU * 6.0 * t).sin()
                    + 0.004 * (std::f32::consts::TAU * 3.0 * t).cos();
                let measured = target + Vector3::new(noise, noise, 0.0);
                if tick % (fps / 30) == 0 {
                    let captured = MonoTimeNs(sample as u64 * 1_000_000_000 / 30);
                    clean.observe(Some(target.into()), captured);
                    noisy.observe(Some(measured.into()), captured);
                }
                clean.step(target, 1.0 / fps as f32, 0.125);
                noisy.step(measured, 1.0 / fps as f32, 0.125);
                if tick >= fps {
                    input_energy += (measured - target).norm_squared();
                    output_energy += (noisy.position - clean.position).norm_squared();
                }
            }
            assert!(
                output_energy < input_energy,
                "{fps} Hz: {output_energy} / {input_energy}"
            );
        }
    }

    #[test]
    fn zero_elapsed_time_keeps_both_poles_unchanged() {
        let mut actual = ArmPositionSmoother::new(Vector3::new(0.3, -0.2, 0.4));
        let before = actual;
        actual.step(Vector3::new(-0.2, 0.3, 0.4), 0.0, 0.125);
        assert_eq!(actual, before);
    }
}
