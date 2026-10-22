//! Noise contracts for the actual six-channel wrist/elbow smoother.
//!
//! All coordinates are shoulder-relative positions divided by the same arm
//! length, so their input/output energy ratios have the same units. Small
//! perturbations of this bent pose leave the bend-plane speed cap inactive.
//! The nonlinear IK/bone-output regressions live in vtuber-avatar.
#![allow(clippy::indexing_slicing)]

use super::*;
use nalgebra::{Complex, SMatrix, SVector};

const AMPLITUDE: f32 = 0.01;
const CAMERA_HZ: usize = 30;
const SECONDS: usize = 4;
const SAMPLES: usize = CAMERA_HZ * SECONDS;
const BASE: ArmTrackingTarget = ArmTrackingTarget {
    wrist: [0.0, 0.0, 0.8],
    elbow_pole: [0.3, -0.2, 0.4],
    palm_normal: None,
    palm_forward: None,
    fingers: None,
};

fn coordinates(target: ArmTrackingTarget) -> SVector<f64, 6> {
    SVector::from_iterator(
        target
            .wrist
            .into_iter()
            .chain(target.elbow_pole)
            .map(f64::from),
    )
}

fn offset_target(offset: [f32; 6]) -> ArmTrackingTarget {
    let mut target = BASE;
    for (coordinate, delta) in target
        .wrist
        .iter_mut()
        .chain(target.elbow_pole.iter_mut())
        .zip(offset)
    {
        *coordinate += delta;
    }
    target
}

#[test]
fn six_channel_frequency_gain_bounds_every_combination_of_noise_phases() {
    // Excite each input independently and measure the COMPLEX transfer matrix:
    // retaining phase is essential, since two sources can reinforce/cancel.
    // The largest singular value is the worst gain over all six input
    // amplitudes/phases, not merely the gain for one chosen noisy trace.
    // Measure every render subframe of a held 30 Hz observation separately;
    // checking only its fundamental at render Hz would miss hold harmonics.
    for fps in [30, 60, 120] {
        let renders_per_sample = fps / CAMERA_HZ;
        for hz in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0] {
            let mut transfer = vec![SMatrix::<Complex<f64>, 6, 6>::zeros(); renders_per_sample];
            for input in 0..6 {
                let mut smoother = ArmSmootherState::new(BASE);
                for sample in 0..SAMPLES * 2 {
                    let phase = std::f64::consts::TAU * hz * sample as f64 / CAMERA_HZ as f64;
                    let mut offset = [0.0; 6];
                    offset[input] = AMPLITUDE * phase.sin() as f32;
                    for matrix in &mut transfer {
                        let output = smoother.advance(offset_target(offset), 1.0 / fps as f32);
                        if sample < SAMPLES {
                            continue; // Let initial conditions settle for four seconds.
                        }
                        let delta = coordinates(output) - coordinates(BASE);
                        let weight = Complex::new(phase.sin(), phase.cos())
                            * (2.0 / (SAMPLES as f64 * f64::from(AMPLITUDE)));
                        for row in 0..6 {
                            matrix[(row, input)] += weight * delta[row];
                        }
                    }
                }
            }
            for (subframe, matrix) in transfer.into_iter().enumerate() {
                let gain = matrix.svd(false, false).singular_values.max();
                assert!(
                    gain <= 1.0005,
                    "{fps} Hz, noise={hz} Hz, subframe={subframe}: worst combined gain={gain}"
                );
                // A concrete jitter budget independent of the configured tau:
                // >= 2 Hz noise must retain at most 35% of its amplitude.
                if hz >= 2.0 {
                    assert!(gain < 0.35, "{fps} Hz, noise={hz} Hz: jitter gain={gain}");
                }
                for row in 0..6 {
                    for column in 0..6 {
                        if row != column {
                            assert!(
                                matrix[(row, column)].norm() < 0.0005,
                                "{fps} Hz, noise={hz} Hz: channel {column} leaks into {row}: {}",
                                matrix[(row, column)]
                            );
                        } else if hz == 0.25 {
                            // Freezing everything also removes noise; reject it.
                            assert!(matrix[(row, column)].norm() > 0.90);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn simultaneous_multitone_noise_cannot_gain_energy_or_step_energy() {
    // Basis responses alone cannot catch nonlinear interactions. Also drive
    // all six inputs together, with correlated, opposing and distinct phases.
    // Measure the SUM of noise sources, including cross terms, as the input;
    // summing the individual source energies would miss constructive overlap.
    for fps in [30, 60, 120] {
        for phases in [0, 1, 2] {
            let mut smoother = ArmSmootherState::new(BASE);
            let mut input_energy = SVector::<f64, 6>::zeros();
            let mut output_energy = SVector::<f64, 6>::zeros();
            let mut input_step_energy = SVector::<f64, 6>::zeros();
            let mut output_step_energy = SVector::<f64, 6>::zeros();
            let mut previous_input = SVector::<f64, 6>::zeros();
            let mut previous_output = SVector::<f64, 6>::zeros();
            for tick in 0..fps * SECONDS * 2 {
                let sample = tick / (fps / CAMERA_HZ);
                let t = sample as f64 / CAMERA_HZ as f64;
                let offset = std::array::from_fn(|channel| {
                    if sample >= SAMPLES {
                        return 0.0; // Include four seconds of the entire settling tail.
                    }
                    let phase = match phases {
                        0 => 0.0,
                        1 => (channel % 2) as f64 * std::f64::consts::PI,
                        _ => channel as f64 * 0.7,
                    };
                    let signal: f64 = [(0.5, 0.4), (3.0, 0.35), (7.0, 0.25)]
                        .into_iter()
                        .map(|(hz, amplitude)| {
                            amplitude * (std::f64::consts::TAU * hz * t + phase).sin()
                        })
                        .sum();
                    AMPLITUDE * signal as f32
                });
                let observed = offset_target(offset);
                let output = smoother.advance(observed, 1.0 / fps as f32);
                let input = coordinates(observed) - coordinates(BASE);
                let output = coordinates(output) - coordinates(BASE);
                let input_step = input - previous_input;
                let output_step = output - previous_output;
                input_energy += input.component_mul(&input);
                output_energy += output.component_mul(&output);
                input_step_energy += input_step.component_mul(&input_step);
                output_step_energy += output_step.component_mul(&output_step);
                previous_input = input;
                previous_output = output;
            }
            for (metric, raw, filtered) in [
                ("position", input_energy, output_energy),
                ("step", input_step_energy, output_step_energy),
            ] {
                for channel in 0..6 {
                    assert!(
                        filtered[channel] <= raw[channel] * 1.0005 + 1.0e-10,
                        "{fps} Hz, phases={phases}, {metric}, channel={channel}: {} > {}",
                        filtered[channel],
                        raw[channel]
                    );
                }
            }
            assert!(previous_output.norm() < 1.0e-5, "the tail must settle");
        }
    }
}
