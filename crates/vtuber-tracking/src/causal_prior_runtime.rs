//! Bounded linear-prior corrections with reset semantics and causality
//! regression (GNM #68.4d).
//!
//! Wraps [`PriorInference`] with the runtime guarantees the tracking pipeline
//! needs before any production connection:
//!
//! - corrections are clamped into per-channel-group configured bounds,
//! - history/prior state resets on sequence gaps, tracking loss, and
//!   reacquires, so a stale prior never survives a discontinuity,
//! - missing/corrupt artifacts already degrade to the deterministic adaptive
//!   baseline at the #132 boundary; non-finite predictions fail closed to
//!   that same baseline here,
//! - causality is machine-checked: changing only future frames can never
//!   change current-or-earlier outputs.

use crate::causal_prior_inference::{LinearPriorLoadError, PriorInference};

/// One bounded channel group.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectionGroup {
    /// Group name (for example `mouth`, `brow`, `eyes`).
    pub name: String,
    /// Inclusive start canonical channel index.
    pub channel_start: usize,
    /// Exclusive end canonical channel index.
    pub channel_end: usize,
    /// Maximum absolute correction allowed for this group's channels.
    pub max_abs_correction: f32,
}

/// Typed configuration errors.
#[derive(Clone, Debug, PartialEq)]
pub enum PriorRuntimeError {
    /// A group range was empty, inverted, out of bounds, or its bound was
    /// non-finite/negative; or cadence settings were invalid.
    InvalidGroup {
        /// Offending group name (`<cadence>` for cadence errors).
        name: String,
    },
    /// Groups overlapped each other's channel ranges.
    OverlappingGroups,
}

/// Runtime configuration for bounded corrections and continuity checks.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorRuntimeConfig {
    /// Per-group correction bounds.
    pub groups: Vec<CorrectionGroup>,
    /// Expected nominal inter-frame interval in microseconds.
    pub expected_dt_micros: u64,
    /// Gap tolerance multiplier over the expected interval.
    pub gap_tolerance: f64,
}

impl PriorRuntimeConfig {
    /// Validates group ranges and the cadence settings.
    ///
    /// # Errors
    ///
    /// Returns typed errors for invalid groups, overlaps, or cadence values.
    pub fn validate(&self) -> Result<(), PriorRuntimeError> {
        if self.expected_dt_micros == 0
            || !self.gap_tolerance.is_finite()
            || self.gap_tolerance <= 1.0
        {
            return Err(PriorRuntimeError::InvalidGroup {
                name: "<cadence>".to_owned(),
            });
        }
        for group in &self.groups {
            let valid = group.channel_start < group.channel_end
                && group.max_abs_correction.is_finite()
                && group.max_abs_correction >= 0.0;
            if !valid {
                return Err(PriorRuntimeError::InvalidGroup {
                    name: group.name.clone(),
                });
            }
        }
        for (index, left) in self.groups.iter().enumerate() {
            for right in self.groups.iter().skip(index + 1) {
                if left.channel_start < right.channel_end && right.channel_start < left.channel_end
                {
                    return Err(PriorRuntimeError::OverlappingGroups);
                }
            }
        }
        Ok(())
    }

    fn gap_limit_micros(&self) -> u64 {
        (self.expected_dt_micros as f64 * self.gap_tolerance).min(u64::MAX as f64) as u64
    }
}

/// Why the runtime dropped its prior state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResetReason {
    /// Frame identity jumped beyond the cadence tolerance.
    SequenceGap,
    /// The caller explicitly reported tracking loss or reacquire.
    ExplicitReset,
}

/// Outcome of one runtime step.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorStepOutcome {
    /// Bounded prior state when a learned prior is active; `None` means the
    /// deterministic adaptive baseline stands alone this frame.
    pub prior_state: Option<Vec<f32>>,
    /// Whether any channel was clamped into its group bound.
    pub clamped: bool,
    /// Set on the frame where a reset occurred.
    pub reset: Option<ResetReason>,
}

/// Stateful runtime connecting the learned prior to one tracking stream.
#[derive(Debug)]
pub struct PriorRuntime {
    inference: PriorInference,
    config: PriorRuntimeConfig,
    last_identity: Option<(u64, u64)>,
    /// Whether the most recent step clamped any channel.
    clamped: bool,
}

impl PriorRuntime {
    /// Creates a validated runtime.
    ///
    /// # Errors
    ///
    /// Returns configuration errors from [`PriorRuntimeConfig::validate`].
    pub fn new(
        inference: PriorInference,
        config: PriorRuntimeConfig,
    ) -> Result<Self, PriorRuntimeError> {
        config.validate()?;
        Ok(Self {
            inference,
            config,
            last_identity: None,
            clamped: false,
        })
    }

    /// Explicitly drops prior state (tracking loss or reacquire boundary).
    pub fn reset(&mut self) {
        self.last_identity = None;
    }

    /// Advances one frame: predicts the prior from causal features with
    /// continuity checks, then clamps into group bounds.
    ///
    /// A discontinuity (sequence jump or timestamp gap) resets state first and
    /// reports it in the outcome; the prediction still runs on this frame's
    /// features because they are causal by contract. Baseline frames yield
    /// `prior_state: None`. Non-finite predictions fail closed to baseline.
    ///
    /// # Errors
    ///
    /// Propagates prediction failures from [`PriorInference::predict`].
    pub fn advance(
        &mut self,
        frame_seq: u64,
        timestamp_micros: u64,
        features: &[f32],
    ) -> Result<PriorStepOutcome, LinearPriorLoadError> {
        let mut reset = None;
        if let Some((previous_seq, previous_time)) = self.last_identity {
            let continuous = frame_seq == previous_seq + 1
                && timestamp_micros.saturating_sub(previous_time) <= self.config.gap_limit_micros();
            if !continuous {
                self.last_identity = None;
                reset = Some(ResetReason::SequenceGap);
            }
        }

        let prior_state = match self.inference.predict(features)? {
            Some(mut state) => {
                let mut clamped = false;
                for group in &self.config.groups {
                    for index in group.channel_start..group.channel_end.min(state.len()) {
                        if let Some(value) = state.get_mut(index) {
                            if *value > group.max_abs_correction {
                                *value = group.max_abs_correction;
                                clamped = true;
                            } else if *value < -group.max_abs_correction {
                                *value = -group.max_abs_correction;
                                clamped = true;
                            }
                        }
                    }
                }
                self.clamped = clamped;
                if state.iter().any(|value| !value.is_finite()) {
                    // Fail closed: the deterministic baseline stands alone.
                    None
                } else {
                    Some(state)
                }
            }
            None => None,
        };

        self.last_identity = Some((frame_seq, timestamp_micros));
        Ok(PriorStepOutcome {
            prior_state,
            clamped: self.clamped,
            reset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arkit_teacher::PairedTemporalSample;
    use crate::causal_dataset::{CausalFeatureConfig, build_causal_dataset};
    use crate::causal_prior::{LinearPriorTrainingConfig, fit_linear_prior};
    use std::collections::BTreeSet;
    use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};

    fn sample(seq: u64, jaw_open: f32) -> PairedTemporalSample {
        let mut values = [0.0_f32; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = jaw_open;
        PairedTemporalSample {
            frame_seq: seq,
            timestamp_micros: seq * 16_667,
            mediapipe_observation: None,
            gnm_state: Some(crate::arkit_teacher::DeterministicGnmState {
                projected_coefficients: Arkit52Coefficients::try_from_array(values).unwrap(),
                residual: 0.01,
            }),
            baseline_output: Arkit52Coefficients::default(),
            teacher: None,
            rgb_reference: None,
        }
    }

    fn learned_inference() -> PriorInference {
        let values = [0.1_f32, 0.5, 0.9, 0.7, 0.2];
        let samples: Vec<PairedTemporalSample> = values
            .iter()
            .enumerate()
            .map(|(index, value)| sample(index as u64 + 1, *value))
            .collect();
        let rows = build_causal_dataset(
            "take-a",
            &samples,
            CausalFeatureConfig {
                history_len: 2,
                max_gap_micros: 40_000,
            },
        )
        .unwrap()
        .rows;
        let takes = BTreeSet::from(["take-a".to_owned()]);
        let artifact = fit_linear_prior(
            &rows,
            &takes,
            LinearPriorTrainingConfig::default(),
            "history+velocity",
        )
        .unwrap();
        PriorInference::load_or_baseline(Some(artifact), "history+velocity")
    }

    fn runtime_config(expected_dt_micros: u64) -> PriorRuntimeConfig {
        PriorRuntimeConfig {
            groups: vec![CorrectionGroup {
                name: "all".to_owned(),
                channel_start: 0,
                channel_end: ARKIT52_CHANNEL_COUNT,
                max_abs_correction: 0.05,
            }],
            expected_dt_micros,
            gap_tolerance: 1.5,
        }
    }

    fn features(value: f32) -> Vec<f32> {
        // Same layout as #130 with history_len 2: 2 slots x (52 + residual +
        // quality) plus the velocity slot of 52.
        vec![value; CausalFeatureConfig::feature_dims() * 2 + 52]
    }

    #[test]
    fn corrections_are_clamped_into_group_bounds() {
        let mut runtime =
            PriorRuntime::new(learned_inference(), runtime_config(16_667)).expect("valid");
        let outcome = runtime
            .advance(1, 16_667, &features(10.0))
            .expect("advances");
        assert!(outcome.clamped);
        let state = outcome.prior_state.expect("learned prior active");
        assert!(state.iter().all(|value| value.abs() <= 0.05 + f32::EPSILON));
    }

    #[test]
    fn sequence_gaps_report_resets_at_multiple_frame_rates() {
        for dt in [33_333_u64, 16_667, 8_333] {
            let mut runtime =
                PriorRuntime::new(learned_inference(), runtime_config(dt)).expect("valid");
            runtime.advance(1, dt, &features(0.1)).expect("frame 1");
            runtime.advance(2, 2 * dt, &features(0.2)).expect("frame 2");
            // Dropout: sequences 3..6 vanish; frame 7 arrives.
            let outcome = runtime
                .advance(7, 7 * dt, &features(0.3))
                .expect("frame after dropout");
            assert_eq!(outcome.reset, Some(ResetReason::SequenceGap));
        }
    }

    #[test]
    fn explicit_loss_and_reacquire_resets_keep_runtime_alive() {
        let mut runtime = PriorRuntime::new(
            PriorInference::DeterministicBaseline,
            runtime_config(16_667),
        )
        .expect("valid");
        let outcome = runtime
            .advance(1, 16_667, &features(0.1))
            .expect("baseline step");
        assert!(outcome.prior_state.is_none());
        runtime.reset();
        let outcome = runtime
            .advance(9, 200_000, &features(0.2))
            .expect("reacquire step");
        assert!(outcome.prior_state.is_none());
        assert!(outcome.reset.is_none());
    }

    #[test]
    fn causality_future_changes_never_affect_current_or_earlier_outputs() {
        let run = |future_value: f32| {
            let mut runtime =
                PriorRuntime::new(learned_inference(), runtime_config(16_667)).expect("valid");
            let mut outputs = Vec::new();
            for seq in 1..=5_u64 {
                let value = if seq == 5 {
                    future_value
                } else {
                    0.1 * seq as f32
                };
                outputs.push(
                    runtime
                        .advance(seq, seq * 16_667, &features(value))
                        .expect("step"),
                );
            }
            outputs
        };
        let baseline_run = run(0.5);
        let perturbed_run = run(0.99);
        // Core property: current-or-earlier outputs are bit-identical even
        // though the future frame's input changed drastically.
        for (index, (baseline, perturbed)) in
            baseline_run.iter().zip(perturbed_run.iter()).enumerate()
        {
            if index < 4 {
                assert_eq!(
                    baseline.prior_state,
                    perturbed.prior_state,
                    "frame {}",
                    index + 1
                );
            }
        }
        // Prediction *sensitivity* to inputs is a fit-quality concern
        // covered by #131/#132 tests; this test pins causality only.
    }

    #[test]
    fn blink_pulse_step_and_dropout_regression_across_frame_rates() {
        for dt in [33_333_u64, 16_667, 8_333] {
            let mut runtime =
                PriorRuntime::new(learned_inference(), runtime_config(dt)).expect("valid");
            for seq in 1..=4_u64 {
                runtime
                    .advance(seq, seq * dt, &features(0.1 * seq as f32))
                    .expect("step");
            }
            runtime.advance(5, 5 * dt, &features(0.95)).expect("pulse");
            runtime
                .advance(6, 6 * dt, &features(0.05))
                .expect("release");
            let outcome = runtime
                .advance(12, 12 * dt, &features(0.1))
                .expect("post-dropout");
            assert_eq!(outcome.reset, Some(ResetReason::SequenceGap));
        }
    }

    #[test]
    fn invalid_groups_and_overlaps_are_typed_errors() {
        let inference = PriorInference::DeterministicBaseline;
        let bad_range = PriorRuntimeConfig {
            groups: vec![CorrectionGroup {
                name: "bad".to_owned(),
                channel_start: 10,
                channel_end: 5,
                max_abs_correction: 0.1,
            }],
            expected_dt_micros: 16_667,
            gap_tolerance: 1.5,
        };
        assert!(matches!(
            PriorRuntime::new(inference.clone(), bad_range),
            Err(PriorRuntimeError::InvalidGroup { .. })
        ));

        let overlapping = PriorRuntimeConfig {
            groups: vec![
                CorrectionGroup {
                    name: "a".to_owned(),
                    channel_start: 0,
                    channel_end: 20,
                    max_abs_correction: 0.1,
                },
                CorrectionGroup {
                    name: "b".to_owned(),
                    channel_start: 15,
                    channel_end: 30,
                    max_abs_correction: 0.1,
                },
            ],
            expected_dt_micros: 16_667,
            gap_tolerance: 1.5,
        };
        assert!(matches!(
            PriorRuntime::new(inference, overlapping),
            Err(PriorRuntimeError::OverlappingGroups)
        ));
    }
}
