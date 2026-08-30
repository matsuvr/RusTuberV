//! Robustness, cross-talk, and runtime-cost A/B report kernels (GNM #57.5).
//!
//! Pure, deterministic functions that turn source-aligned Direct/GNM numeric
//! traces into the comparison numbers required for the promotion decision:
//!
//! - one-frame jumps, partial/full dropouts, stale-output duration, and
//!   reacquire timing per backend,
//! - expression leakage during rigid-head-only motion and head-pose leakage
//!   during expression-only motion,
//! - backend latency breakdowns plus fitter iteration/drop counters carried
//!   through unchanged,
//! - a typed [`PromotionVerdict`] recording `Default` or `Experimental` with
//!   numerically justified blockers.
//!
//! Nothing here reads cameras, models, or runtimes; reports rerun from
//! synthetic or recorded numeric traces alone.

use crate::temporal_metrics::{TemporalMetricError, TemporalTrace};

/// Longest-gap and jump statistics for one scalar trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustnessMetrics {
    /// Largest absolute change between two consecutive samples.
    pub max_one_frame_jump: f64,
    /// Number of inter-sample gaps exceeding `expected_dt_micros * gap_tolerance`.
    ///
    /// Values of one are partial dropouts (single missing frame); larger
    /// counts indicate repeated loss. Full dropouts surface as the largest
    /// gap duration below.
    pub dropout_gap_count: usize,
    /// Duration of the longest detected dropout gap in milliseconds.
    pub longest_dropout_ms: f64,
    /// Longest span in milliseconds during which the value never changed by
    /// more than `stale_epsilon`, i.e. the published output stayed frozen.
    pub stale_duration_ms: f64,
    /// Delay from the end of the longest dropout until the value moved again
    /// by more than `stale_epsilon`, in milliseconds.
    pub reacquire_delay_ms: Option<f64>,
}

/// Measures robustness statistics for one validated trace.
///
/// `expected_dt_micros` is the nominal sample period; gaps above
/// `gap_tolerance * expected_dt` count as dropouts. `stale_epsilon` is the
/// smallest value change treated as real motion rather than quantization.
///
/// # Errors
///
/// Propagates trace validation failures; configuration errors return typed
/// variants instead of panicking on external input.
// Bounds are guaranteed by construction: TemporalTrace is non-empty and
// monotonic, so index-1 and index are in range inside 1..len loops.
#[allow(clippy::indexing_slicing)]
pub fn robustness_metrics(
    trace: &TemporalTrace,
    expected_dt_micros: u64,
    gap_tolerance: f64,
    stale_epsilon: f64,
) -> Result<RobustnessMetrics, TemporalMetricError> {
    if expected_dt_micros == 0 {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "expected_dt_micros must be positive",
        ));
    }
    if !gap_tolerance.is_finite() || gap_tolerance <= 1.0 {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "gap_tolerance must be finite and greater than 1",
        ));
    }
    if !stale_epsilon.is_finite() || stale_epsilon < 0.0 {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "stale_epsilon must be finite and non-negative",
        ));
    }
    let samples = trace.samples();
    let gap_limit = (expected_dt_micros as f64 * gap_tolerance).min(u64::MAX as f64) as u64;

    let mut max_one_frame_jump = 0.0_f64;
    let mut dropout_gap_count = 0_usize;
    let mut longest_gap_micros = 0_u64;
    let mut longest_gap_end_index = None;
    let mut stale_duration_ms = 0.0_f64;

    let mut stale_start_index = 0_usize;
    for index in 1..samples.len() {
        let previous = samples[index - 1];
        let current = samples[index];
        let delta = current.value - previous.value;
        max_one_frame_jump = max_one_frame_jump.max(delta.abs());

        let gap = current.timestamp_micros - previous.timestamp_micros;
        if gap > gap_limit {
            dropout_gap_count += 1;
            if gap > longest_gap_micros {
                longest_gap_micros = gap;
                longest_gap_end_index = Some(index);
            }
        }

        if delta.abs() > stale_epsilon {
            let stale_span =
                previous.timestamp_micros - samples[stale_start_index].timestamp_micros;
            stale_duration_ms = stale_duration_ms.max(stale_span as f64 / 1_000.0);
            stale_start_index = index;
        }
    }
    // Tail run: value stayed frozen until the end of the trace.
    let tail_span =
        samples[samples.len() - 1].timestamp_micros - samples[stale_start_index].timestamp_micros;
    stale_duration_ms = stale_duration_ms.max(tail_span as f64 / 1_000.0);

    let reacquire_delay_ms = longest_gap_end_index.and_then(|end_index| {
        // After the longest dropout, find the first sample that actually
        // moves again relative to the post-gap value.
        let resumed_value = samples.get(end_index)?.value;
        samples[end_index..]
            .iter()
            .find(|sample| (sample.value - resumed_value).abs() > stale_epsilon)
            .map(|moved| {
                (moved.timestamp_micros - samples[end_index].timestamp_micros) as f64 / 1_000.0
            })
    });

    Ok(RobustnessMetrics {
        max_one_frame_jump,
        dropout_gap_count,
        longest_dropout_ms: longest_gap_micros as f64 / 1_000.0,
        stale_duration_ms,
        reacquire_delay_ms,
    })
}

/// Expression-channel leakage measured against a driver channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossTalkMetrics {
    /// RMS velocity of the observed channel while the driver moved fast.
    pub driven_rms_per_second: f64,
    /// RMS velocity of the observed channel while the driver was quiet.
    pub rest_rms_per_second: f64,
    /// `driven_rms - rest_rms`; zero means no measurable cross-talk.
    pub crosstalk_excess: f64,
}

/// Measures how much `observed` moves while `driver` moves versus at rest.
///
/// Both traces must share the same nominal sample grid; samples pair by
/// position, so callers must pass identically sampled windows (as produced
/// by the same-frame fan-out). A window whose driver velocity exceeds
/// `driver_threshold_per_second` counts as "driven".
///
/// # Errors
///
/// Propagates trace validation failures and rejects mismatched lengths.
// Bounds are guaranteed by construction: TemporalTrace is non-empty and
// monotonic, so index-1 and index are in range inside 1..len loops.
#[allow(clippy::indexing_slicing)]
pub fn crosstalk_metrics(
    driver: &TemporalTrace,
    observed: &TemporalTrace,
    driver_threshold_per_second: f64,
) -> Result<CrossTalkMetrics, TemporalMetricError> {
    if !driver_threshold_per_second.is_finite() || driver_threshold_per_second <= 0.0 {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "driver_threshold_per_second must be finite and positive",
        ));
    }
    let driver_samples = driver.samples();
    let observed_samples = observed.samples();
    if driver_samples.len() != observed_samples.len() {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "cross-talk traces must have identical sample counts",
        ));
    }
    if driver_samples.len() < 2 {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "cross-talk traces need at least two samples",
        ));
    }

    let mut driven_sum_squares = 0.0;
    let mut driven_count = 0_usize;
    let mut rest_sum_squares = 0.0;
    let mut rest_count = 0_usize;
    for window in 1..driver_samples.len() {
        let dt = (driver_samples[window].timestamp_micros
            - driver_samples[window - 1].timestamp_micros) as f64
            / 1_000_000.0;
        if dt <= 0.0 {
            continue;
        }
        let driver_velocity =
            (driver_samples[window].value - driver_samples[window - 1].value).abs() / dt;
        let observed_velocity =
            (observed_samples[window].value - observed_samples[window - 1].value).abs() / dt;
        if driver_velocity > driver_threshold_per_second {
            driven_sum_squares += observed_velocity * observed_velocity;
            driven_count += 1;
        } else {
            rest_sum_squares += observed_velocity * observed_velocity;
            rest_count += 1;
        }
    }

    let driven_rms_per_second = if driven_count > 0 {
        (driven_sum_squares / driven_count as f64).sqrt()
    } else {
        0.0
    };
    let rest_rms_per_second = if rest_count > 0 {
        (rest_sum_squares / rest_count as f64).sqrt()
    } else {
        0.0
    };
    Ok(CrossTalkMetrics {
        driven_rms_per_second,
        rest_rms_per_second,
        crosstalk_excess: driven_rms_per_second - rest_rms_per_second,
    })
}

/// Promotion decision recorded at the end of an A/B report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromotionDecision {
    /// The backend may become the default tracking path.
    Default,
    /// The backend stays experimental behind explicit settings.
    Experimental,
}

/// One failed promotion criterion with its measured number.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionBlocker {
    /// Human-readable criterion name.
    pub criterion: String,
    /// Measured value that failed the criterion.
    pub measured: f64,
    /// Numeric bound the measurement had to satisfy.
    pub bound: f64,
}

/// Verdict combining a decision with its numeric justification.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionVerdict {
    /// Final decision for the report tail.
    pub decision: PromotionDecision,
    /// Empty when [`PromotionDecision::Default`]; otherwise lists every
    /// failed criterion with numbers so the decision is reproducible.
    pub blockers: Vec<PromotionBlocker>,
}

/// Numeric promotion criteria evaluated over measured A/B results.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionCriteria {
    /// Maximum tolerated additional end-to-end latency for GNM, ms.
    pub max_gnm_latency_overhead_ms: f64,
    /// Maximum tolerated expression excess during rigid head motion, 1/s.
    pub max_head_to_expression_crosstalk: f64,
    /// Maximum tolerated head-pose excess during expression motion, rad/s.
    pub max_expression_to_head_crosstalk: f64,
    /// Maximum tolerated GNM fit latency, ms.
    pub max_fit_latency_ms: f64,
    /// Maximum tolerated longest dropout before authority must fall back, ms.
    pub max_longest_dropout_ms: f64,
}

impl PromotionCriteria {
    fn validate(self) -> Result<(), TemporalMetricError> {
        let values = [
            self.max_gnm_latency_overhead_ms,
            self.max_head_to_expression_crosstalk,
            self.max_expression_to_head_crosstalk,
            self.max_fit_latency_ms,
            self.max_longest_dropout_ms,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(TemporalMetricError::InvalidResponseSpec(
                "promotion criteria must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Measured inputs to the promotion evaluation for one A/B run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbMeasuredInputs {
    /// Direct end-to-end latency, ms.
    pub direct_end_to_end_ms: f64,
    /// GNM end-to-end latency, ms.
    pub gnm_end_to_end_ms: f64,
    /// Worst observed GNM fit latency, ms.
    pub gnm_max_fit_ms: f64,
    /// Expression excess while only the head moved, GNM side.
    pub gnm_head_to_expression_crosstalk: f64,
    /// Head-pose excess while only expressions moved, GNM side.
    pub gnm_expression_to_head_crosstalk: f64,
    /// Longest GNM output dropout, ms.
    pub gnm_longest_dropout_ms: f64,
}

/// Evaluates the promotion criteria against measurements.
///
/// # Errors
///
/// Returns a typed error for invalid criteria instead of panicking.
pub fn promotion_verdict(
    inputs: AbMeasuredInputs,
    criteria: PromotionCriteria,
) -> Result<PromotionVerdict, TemporalMetricError> {
    criteria.validate()?;
    let latency_overhead = inputs.gnm_end_to_end_ms - inputs.direct_end_to_end_ms;
    let checks = [
        (
            ("gnm_latency_overhead_ms", latency_overhead),
            criteria.max_gnm_latency_overhead_ms,
        ),
        (
            (
                "gnm_head_to_expression_crosstalk",
                inputs.gnm_head_to_expression_crosstalk,
            ),
            criteria.max_head_to_expression_crosstalk,
        ),
        (
            (
                "gnm_expression_to_head_crosstalk",
                inputs.gnm_expression_to_head_crosstalk,
            ),
            criteria.max_expression_to_head_crosstalk,
        ),
        (
            ("gnm_max_fit_ms", inputs.gnm_max_fit_ms),
            criteria.max_fit_latency_ms,
        ),
        (
            ("gnm_longest_dropout_ms", inputs.gnm_longest_dropout_ms),
            criteria.max_longest_dropout_ms,
        ),
    ];
    let mut blockers = Vec::new();
    for ((criterion, measured), bound) in checks {
        if !(measured.is_finite() && measured <= bound) {
            blockers.push(PromotionBlocker {
                criterion: criterion.to_owned(),
                measured,
                bound,
            });
        }
    }
    let decision = if blockers.is_empty() {
        PromotionDecision::Default
    } else {
        PromotionDecision::Experimental
    };
    Ok(PromotionVerdict { decision, blockers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_metrics::TemporalSample;
    use approx::assert_relative_eq;

    const DT_MICROS: u64 = 16_667;

    fn trace(values: &[f64]) -> TemporalTrace {
        TemporalTrace::new(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| TemporalSample {
                    timestamp_micros: index as u64 * DT_MICROS,
                    value: *value,
                })
                .collect(),
        )
        .expect("valid trace")
    }

    #[test]
    fn dropout_fixture_counts_gaps_and_measures_reacquire_delay() {
        // 10 frames of motion, a 5-frame dropout (timestamps jump), then
        // 3 frozen frames before motion resumes.
        let mut values: Vec<f64> = (0..10).map(|index| index as f64 * 0.1).collect();
        let mut timestamps: Vec<u64> = (0..10).map(|index| index as u64 * DT_MICROS).collect();
        let gap_start = 10_u64 * DT_MICROS;
        let resume_index = 15_u64;
        let frozen_value = 9_f64 * 0.1;
        for offset in 0..4 {
            values.push(frozen_value);
            timestamps.push(gap_start + (resume_index + offset) * DT_MICROS);
        }
        values.push(frozen_value + 0.5);
        timestamps.push(gap_start + (resume_index + 4) * DT_MICROS);

        let samples: Vec<TemporalSample> = values
            .iter()
            .zip(timestamps.iter())
            .map(|(value, timestamp)| TemporalSample {
                timestamp_micros: *timestamp,
                value: *value,
            })
            .collect();
        let trace = TemporalTrace::new(samples).expect("valid trace");
        let metrics = robustness_metrics(&trace, DT_MICROS, 1.5, 1e-6).expect("measures");
        assert_eq!(metrics.dropout_gap_count, 1);
        assert_relative_eq!(
            metrics.longest_dropout_ms,
            (gap_start + resume_index * DT_MICROS - 9 * DT_MICROS) as f64 / 1_000.0
        );
        assert_relative_eq!(
            metrics.reacquire_delay_ms.expect("resumes"),
            4.0 * DT_MICROS as f64 / 1_000.0
        );
    }

    #[test]
    fn jump_and_stale_are_reported_from_numeric_traces() {
        let metrics = robustness_metrics(
            &trace(&[0.0, 0.0, 0.0, 0.9, 0.9, 0.2]),
            DT_MICROS,
            1.5,
            1e-6,
        )
        .expect("measures");
        assert_relative_eq!(metrics.max_one_frame_jump, 0.9);
        assert_eq!(metrics.dropout_gap_count, 0);
        // Three frozen frames = two full gaps ≈ 33.33 ms.
        assert!((30.0..=40.0).contains(&metrics.stale_duration_ms));
    }

    #[test]
    fn crosstalk_excess_is_zero_when_channels_are_independent() {
        // Driver moves in the first half only; observed moves only in the
        // second half, so no excess appears during driven frames.
        let driver_values: Vec<f64> = (0..20)
            .map(|index| if index < 10 { index as f64 * 0.2 } else { 2.0 })
            .collect();
        let observed_values: Vec<f64> = (0..20)
            .map(|index| {
                if index >= 10 {
                    (index - 10) as f64 * 0.2
                } else {
                    0.0
                }
            })
            .collect();
        let metrics = crosstalk_metrics(&trace(&driver_values), &trace(&observed_values), 5.0)
            .expect("measures");
        assert_eq!(metrics.driven_rms_per_second, 0.0);
        assert!(metrics.rest_rms_per_second > 0.0);
        assert!(metrics.crosstalk_excess < 0.0);
    }

    #[test]
    fn crosstalk_detects_expression_leakage_during_rigid_motion() {
        // Observed channel wiggles exactly while the driver moves.
        let driver_values: Vec<f64> = (0..20).map(|index| index as f64 * 0.2).collect();
        let observed_values: Vec<f64> = (0..20).map(|index| index as f64 * 0.05).collect();
        let metrics = crosstalk_metrics(&trace(&driver_values), &trace(&observed_values), 5.0)
            .expect("measures");
        assert!(metrics.crosstalk_excess > 0.0);
    }

    #[test]
    fn verdict_requires_default_only_when_all_criteria_pass() {
        let inputs = AbMeasuredInputs {
            direct_end_to_end_ms: 40.0,
            gnm_end_to_end_ms: 50.0,
            gnm_max_fit_ms: 8.0,
            gnm_head_to_expression_crosstalk: 0.01,
            gnm_expression_to_head_crosstalk: 0.002,
            gnm_longest_dropout_ms: 90.0,
        };
        let criteria = PromotionCriteria {
            max_gnm_latency_overhead_ms: 12.0,
            max_head_to_expression_crosstalk: 0.05,
            max_expression_to_head_crosstalk: 0.01,
            max_fit_latency_ms: 10.0,
            max_longest_dropout_ms: 120.0,
        };
        let verdict = promotion_verdict(inputs, criteria).expect("verdict");
        assert_eq!(verdict.decision, PromotionDecision::Default);
        assert!(verdict.blockers.is_empty());

        let failing = promotion_verdict(
            AbMeasuredInputs {
                gnm_end_to_end_ms: 60.0,
                ..inputs
            },
            criteria,
        )
        .expect("verdict");
        assert_eq!(failing.decision, PromotionDecision::Experimental);
        assert_eq!(failing.blockers.len(), 1);
        assert_eq!(failing.blockers[0].criterion, "gnm_latency_overhead_ms");
        assert_relative_eq!(failing.blockers[0].measured, 20.0);
        assert_relative_eq!(failing.blockers[0].bound, 12.0);

        assert!(
            promotion_verdict(
                inputs,
                PromotionCriteria {
                    max_gnm_latency_overhead_ms: -1.0,
                    ..criteria
                }
            )
            .is_err()
        );
    }
}
