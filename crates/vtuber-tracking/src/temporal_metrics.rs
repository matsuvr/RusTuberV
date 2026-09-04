//! Pure, timestamp-aware temporal quality metrics for tracking A/B evaluation.
//!
//! These functions intentionally know nothing about MediaPipe, GNM, ARKit,
//! cameras, or rendering. Callers can apply the same definitions to any scalar
//! channel or projected state component and compare jitter and latency together.

/// One scalar sample on a monotonic capture timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalSample {
    /// Monotonic source timestamp in microseconds.
    pub timestamp_micros: u64,
    /// Finite scalar state or coefficient value.
    pub value: f64,
}

/// Validated scalar time series.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalTrace {
    samples: Vec<TemporalSample>,
}

impl TemporalTrace {
    /// Creates a non-empty trace with finite values and strictly increasing
    /// timestamps.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn new(samples: Vec<TemporalSample>) -> Result<Self, TemporalMetricError> {
        if samples.is_empty() {
            return Err(TemporalMetricError::EmptyTrace);
        }
        for (index, sample) in samples.iter().enumerate() {
            if !sample.value.is_finite() {
                return Err(TemporalMetricError::NonFinite { index });
            }
            if index > 0 && sample.timestamp_micros <= samples[index - 1].timestamp_micros {
                return Err(TemporalMetricError::NonMonotonic {
                    index,
                    previous: samples[index - 1].timestamp_micros,
                    current: sample.timestamp_micros,
                });
            }
        }
        Ok(Self { samples })
    }

    /// Returns samples in timestamp order.
    pub fn samples(&self) -> &[TemporalSample] {
        &self.samples
    }

    /// Returns the number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns whether the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns the first source timestamp.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn start_micros(&self) -> u64 {
        self.samples[0].timestamp_micros
    }

    /// Returns the final source timestamp.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn end_micros(&self) -> u64 {
        self.samples[self.samples.len() - 1].timestamp_micros
    }
}

/// Stationary/noise-oriented metrics for one scalar trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalNoiseMetrics {
    /// Number of samples used.
    pub sample_count: usize,
    /// Arithmetic mean of the trace.
    pub mean: f64,
    /// RMS deviation from the trace mean.
    pub stationary_rms: f64,
    /// RMS first derivative in value units per second, when at least two samples exist.
    pub first_difference_rms_per_second: Option<f64>,
    /// RMS second derivative in value units per second squared, when at least three samples exist.
    pub second_difference_rms_per_second2: Option<f64>,
}

/// Computes dt-aware stationary, velocity, and acceleration noise metrics.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn temporal_noise_metrics(trace: &TemporalTrace) -> TemporalNoiseMetrics {
    let samples = trace.samples();
    let sample_count = samples.len();
    let mean = samples.iter().map(|sample| sample.value).sum::<f64>() / sample_count as f64;
    let stationary_rms = (samples
        .iter()
        .map(|sample| {
            let delta = sample.value - mean;
            delta * delta
        })
        .sum::<f64>()
        / sample_count as f64)
        .sqrt();

    let first_difference_rms_per_second = if sample_count >= 2 {
        let sum_squares = samples
            .windows(2)
            .map(|window| {
                let dt = micros_to_seconds(window[1].timestamp_micros - window[0].timestamp_micros);
                let velocity = (window[1].value - window[0].value) / dt;
                velocity * velocity
            })
            .sum::<f64>();
        Some((sum_squares / (sample_count - 1) as f64).sqrt())
    } else {
        None
    };

    let second_difference_rms_per_second2 = if sample_count >= 3 {
        let mut sum_squares = 0.0;
        for window in samples.windows(3) {
            let dt_left =
                micros_to_seconds(window[1].timestamp_micros - window[0].timestamp_micros);
            let dt_right =
                micros_to_seconds(window[2].timestamp_micros - window[1].timestamp_micros);
            let velocity_left = (window[1].value - window[0].value) / dt_left;
            let velocity_right = (window[2].value - window[1].value) / dt_right;
            let velocity_dt = 0.5 * (dt_left + dt_right);
            let acceleration = (velocity_right - velocity_left) / velocity_dt;
            sum_squares += acceleration * acceleration;
        }
        Some((sum_squares / (sample_count - 2) as f64).sqrt())
    } else {
        None
    };

    TemporalNoiseMetrics {
        sample_count,
        mean,
        stationary_rms,
        first_difference_rms_per_second,
        second_difference_rms_per_second2,
    }
}

/// Definition of a commanded scalar step or release.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepResponseSpec {
    /// Timestamp at which the target changes.
    pub command_micros: u64,
    /// Pre-command baseline value.
    pub baseline: f64,
    /// Post-command target value. May be lower than `baseline` for a release.
    pub target: f64,
    /// Fraction of the commanded amplitude used for the settling band.
    pub settling_tolerance_fraction: f64,
}

impl StepResponseSpec {
    fn validate(self, trace: &TemporalTrace) -> Result<Self, TemporalMetricError> {
        if !self.baseline.is_finite() || !self.target.is_finite() {
            return Err(TemporalMetricError::InvalidResponseSpec(
                "baseline and target must be finite",
            ));
        }
        if self.target == self.baseline {
            return Err(TemporalMetricError::InvalidResponseSpec(
                "target must differ from baseline",
            ));
        }
        if !self.settling_tolerance_fraction.is_finite()
            || self.settling_tolerance_fraction <= 0.0
            || self.settling_tolerance_fraction >= 1.0
        {
            return Err(TemporalMetricError::InvalidResponseSpec(
                "settling tolerance must be finite and within (0, 1)",
            ));
        }
        if self.command_micros < trace.start_micros() || self.command_micros > trace.end_micros() {
            return Err(TemporalMetricError::CommandOutsideTrace {
                command: self.command_micros,
                start: trace.start_micros(),
                end: trace.end_micros(),
            });
        }
        Ok(self)
    }
}

/// Measured timing and amplitude characteristics for a step response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepResponseMetrics {
    /// Delay from command to the 10% crossing, in milliseconds.
    pub onset_delay_ms: Option<f64>,
    /// Time from command to the 10% crossing, in milliseconds.
    pub t10_ms: Option<f64>,
    /// Time from command to the 50% crossing, in milliseconds.
    pub t50_ms: Option<f64>,
    /// Time from command to the 90% crossing, in milliseconds.
    pub t90_ms: Option<f64>,
    /// Time between the 10% and 90% crossings, in milliseconds.
    pub rise_time_10_90_ms: Option<f64>,
    /// Maximum normalized response amplitude after the command.
    /// `1.0` reaches the target and values above one indicate overshoot.
    pub peak_response_ratio: f64,
    /// Missing peak amplitude relative to the target, clamped at zero.
    pub peak_attenuation: f64,
    /// Overshoot beyond the target amplitude, clamped at zero.
    pub overshoot: f64,
    /// Earliest sampled time after the 90% crossing at which every remaining
    /// sample stays inside the configured target band, in milliseconds from command.
    pub settling_time_ms: Option<f64>,
}

/// Measures a rising or falling scalar response using identical definitions.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn step_response_metrics(
    trace: &TemporalTrace,
    spec: StepResponseSpec,
) -> Result<StepResponseMetrics, TemporalMetricError> {
    let spec = spec.validate(trace)?;
    let amplitude = spec.target - spec.baseline;

    let t10 = progress_crossing_micros(trace, spec, 0.10)?;
    let t50 = progress_crossing_micros(trace, spec, 0.50)?;
    let t90 = progress_crossing_micros(trace, spec, 0.90)?;

    let peak_response_ratio = trace
        .samples()
        .iter()
        .filter(|sample| sample.timestamp_micros >= spec.command_micros)
        .map(|sample| (sample.value - spec.baseline) / amplitude)
        .fold(f64::NEG_INFINITY, f64::max);

    let settling_time_ms = t90.and_then(|t90_micros| {
        trace
            .samples()
            .iter()
            .enumerate()
            .filter(|(_, sample)| sample.timestamp_micros >= t90_micros)
            .find_map(|(index, sample)| {
                let all_settled = trace.samples()[index..].iter().all(|candidate| {
                    let progress = (candidate.value - spec.baseline) / amplitude;
                    (progress - 1.0).abs() <= spec.settling_tolerance_fraction
                });
                if all_settled {
                    Some(duration_ms(spec.command_micros, sample.timestamp_micros))
                } else {
                    None
                }
            })
    });

    let t10_ms = t10.map(|time| duration_ms(spec.command_micros, time));
    let t50_ms = t50.map(|time| duration_ms(spec.command_micros, time));
    let t90_ms = t90.map(|time| duration_ms(spec.command_micros, time));
    let rise_time_10_90_ms = match (t10, t90) {
        (Some(start), Some(end)) if end >= start => Some(duration_ms(start, end)),
        _ => None,
    };

    Ok(StepResponseMetrics {
        onset_delay_ms: t10_ms,
        t10_ms,
        t50_ms,
        t90_ms,
        rise_time_10_90_ms,
        peak_response_ratio,
        peak_attenuation: (1.0 - peak_response_ratio).max(0.0),
        overshoot: (peak_response_ratio - 1.0).max(0.0),
        settling_time_ms,
    })
}

/// Definition of a short pulse whose peak preservation is important.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PulseResponseSpec {
    /// Timestamp at which the pulse begins.
    pub onset_micros: u64,
    /// Baseline value before the pulse.
    pub baseline: f64,
    /// Expected pulse peak value.
    pub target_peak: f64,
    /// Optional timestamp at which the reference pulse reaches its peak.
    pub expected_peak_micros: Option<u64>,
}

/// Peak-preservation metrics for a short pulse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PulseResponseMetrics {
    /// Maximum normalized pulse response, where `1.0` reaches the expected peak.
    pub peak_response_ratio: f64,
    /// Missing peak fraction, clamped at zero.
    pub peak_attenuation: f64,
    /// Timestamp of the observed peak.
    pub observed_peak_micros: u64,
    /// Delay from pulse onset to observed peak in milliseconds.
    pub observed_peak_delay_ms: f64,
    /// Observed-minus-expected peak timing in milliseconds when a reference peak is supplied.
    pub peak_timing_error_ms: Option<f64>,
}

/// Measures short-pulse peak preservation without applying any smoothing itself.
pub fn pulse_response_metrics(
    trace: &TemporalTrace,
    spec: PulseResponseSpec,
) -> Result<PulseResponseMetrics, TemporalMetricError> {
    if !spec.baseline.is_finite() || !spec.target_peak.is_finite() {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "pulse baseline and target peak must be finite",
        ));
    }
    if spec.target_peak == spec.baseline {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "pulse target peak must differ from baseline",
        ));
    }
    if spec.onset_micros < trace.start_micros() || spec.onset_micros > trace.end_micros() {
        return Err(TemporalMetricError::CommandOutsideTrace {
            command: spec.onset_micros,
            start: trace.start_micros(),
            end: trace.end_micros(),
        });
    }
    if let Some(expected_peak) = spec.expected_peak_micros
        && expected_peak < spec.onset_micros
    {
        return Err(TemporalMetricError::InvalidResponseSpec(
            "expected pulse peak cannot precede onset",
        ));
    }

    let amplitude = spec.target_peak - spec.baseline;
    let (observed_peak_micros, peak_response_ratio) = trace
        .samples()
        .iter()
        .filter(|sample| sample.timestamp_micros >= spec.onset_micros)
        .map(|sample| {
            (
                sample.timestamp_micros,
                (sample.value - spec.baseline) / amplitude,
            )
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .ok_or(TemporalMetricError::EmptyTrace)?;

    let peak_timing_error_ms = spec
        .expected_peak_micros
        .map(|expected| signed_duration_ms(expected, observed_peak_micros));

    Ok(PulseResponseMetrics {
        peak_response_ratio,
        peak_attenuation: (1.0 - peak_response_ratio).max(0.0),
        observed_peak_micros,
        observed_peak_delay_ms: duration_ms(spec.onset_micros, observed_peak_micros),
        peak_timing_error_ms,
    })
}

fn progress_crossing_micros(
    trace: &TemporalTrace,
    spec: StepResponseSpec,
    fraction: f64,
) -> Result<Option<u64>, TemporalMetricError> {
    let amplitude = spec.target - spec.baseline;
    let threshold = spec.baseline + amplitude * fraction;
    let start_value =
        value_at(trace, spec.command_micros).ok_or(TemporalMetricError::CommandOutsideTrace {
            command: spec.command_micros,
            start: trace.start_micros(),
            end: trace.end_micros(),
        })?;
    if reached(start_value, threshold, amplitude) {
        return Ok(Some(spec.command_micros));
    }

    let mut previous = TemporalSample {
        timestamp_micros: spec.command_micros,
        value: start_value,
    };
    for current in trace
        .samples()
        .iter()
        .copied()
        .filter(|sample| sample.timestamp_micros > spec.command_micros)
    {
        if reached(current.value, threshold, amplitude) {
            return Ok(Some(interpolate_crossing_micros(
                previous, current, threshold,
            )));
        }
        previous = current;
    }
    Ok(None)
}

fn reached(value: f64, threshold: f64, amplitude: f64) -> bool {
    if amplitude > 0.0 {
        value >= threshold
    } else {
        value <= threshold
    }
}

fn interpolate_crossing_micros(left: TemporalSample, right: TemporalSample, threshold: f64) -> u64 {
    let value_delta = right.value - left.value;
    if value_delta.abs() <= f64::EPSILON {
        return right.timestamp_micros;
    }
    let fraction = ((threshold - left.value) / value_delta).clamp(0.0, 1.0);
    let delta_micros = (right.timestamp_micros - left.timestamp_micros) as f64;
    left.timestamp_micros + (fraction * delta_micros).round() as u64
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn value_at(trace: &TemporalTrace, timestamp_micros: u64) -> Option<f64> {
    let samples = trace.samples();
    if timestamp_micros < samples[0].timestamp_micros
        || timestamp_micros > samples[samples.len() - 1].timestamp_micros
    {
        return None;
    }
    match samples.binary_search_by_key(&timestamp_micros, |sample| sample.timestamp_micros) {
        Ok(index) => Some(samples[index].value),
        Err(right_index) => {
            let left = samples[right_index - 1];
            let right = samples[right_index];
            let fraction = (timestamp_micros - left.timestamp_micros) as f64
                / (right.timestamp_micros - left.timestamp_micros) as f64;
            Some(left.value + fraction * (right.value - left.value))
        }
    }
}

fn micros_to_seconds(micros: u64) -> f64 {
    micros as f64 / 1_000_000.0
}

fn duration_ms(start_micros: u64, end_micros: u64) -> f64 {
    (end_micros - start_micros) as f64 / 1_000.0
}

fn signed_duration_ms(reference_micros: u64, observed_micros: u64) -> f64 {
    (observed_micros as i128 - reference_micros as i128) as f64 / 1_000.0
}

/// Typed validation failure for temporal metrics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TemporalMetricError {
    /// A trace contained no samples.
    EmptyTrace,
    /// A sample value was NaN or infinite.
    NonFinite {
        /// Invalid sample index.
        index: usize,
    },
    /// Timestamps were duplicated or moved backwards.
    NonMonotonic {
        /// Invalid sample index.
        index: usize,
        /// Previous timestamp.
        previous: u64,
        /// Current timestamp.
        current: u64,
    },
    /// A response definition was invalid.
    InvalidResponseSpec(&'static str),
    /// The command timestamp was outside the available trace.
    CommandOutsideTrace {
        /// Command timestamp.
        command: u64,
        /// Trace start timestamp.
        start: u64,
        /// Trace end timestamp.
        end: u64,
    },
}

impl std::fmt::Display for TemporalMetricError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTrace => write!(formatter, "temporal trace must not be empty"),
            Self::NonFinite { index } => {
                write!(formatter, "temporal sample {index} is non-finite")
            }
            Self::NonMonotonic {
                index,
                previous,
                current,
            } => write!(
                formatter,
                "temporal sample {index} timestamp {current} does not follow {previous}"
            ),
            Self::InvalidResponseSpec(reason) => {
                write!(formatter, "invalid response metric definition: {reason}")
            }
            Self::CommandOutsideTrace {
                command,
                start,
                end,
            } => write!(
                formatter,
                "response command {command} is outside trace [{start}, {end}]"
            ),
        }
    }
}

impl std::error::Error for TemporalMetricError {}

/// Lecture-style hold detection for one cheek-puff take.
///
/// A take is one fixed hold (the lecture keeps each expression for 4 s).
/// The hold counts as detected when the score stays above `threshold` on
/// more than half its samples. Empty traces never count as detected.
#[must_use]
pub fn cheek_hold_detected(trace: &TemporalTrace, threshold: f64) -> bool {
    let samples = trace.samples();
    if samples.is_empty() {
        return false;
    }
    let reactive = samples
        .iter()
        .filter(|sample| sample.value > threshold)
        .count();
    reactive * 2 > samples.len()
}

/// Take-level cheek evaluation over puff and non-puff takes.
///
/// Each slice holds one precomputed hold decision per take (`true` =
/// detected). The lecture's bar is all puff takes detected with zero false
/// alarms on the non-puff takes (talk/smile/purse); this struct reports
/// both counts so a threshold sweep can pick the middle of the flat
/// interval instead of the single best score.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CheekTakeEvaluation {
    /// Puff takes detected.
    pub detected_puff: usize,
    /// Total puff takes.
    pub total_puff: usize,
    /// Non-puff takes with a false alarm.
    pub false_alarms: usize,
    /// Total non-puff takes.
    pub total_non_puff: usize,
}

impl CheekTakeEvaluation {
    /// Aggregates per-take hold decisions.
    #[must_use]
    pub fn new(puff_detected: &[bool], non_puff_detected: &[bool]) -> Self {
        Self {
            detected_puff: puff_detected.iter().filter(|hit| **hit).count(),
            total_puff: puff_detected.len(),
            false_alarms: non_puff_detected.iter().filter(|hit| **hit).count(),
            total_non_puff: non_puff_detected.len(),
        }
    }

    /// Returns `true` when every puff take fired and no non-puff take did.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.total_puff > 0
            && self.detected_puff == self.total_puff
            && self.false_alarms == 0
    }
}

/// Returns `true` when the training and evaluation take-id sets are
/// disjoint. Take-disjoint splits are required before any cheek (or prior)
/// number may be read as generalization evidence.
#[must_use]
pub fn cheek_takes_are_disjoint(train: &[&str], eval: &[&str]) -> bool {
    train
        .iter()
        .all(|take| !eval.iter().any(|candidate| candidate == take))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(values: &[(u64, f64)]) -> TemporalTrace {
        TemporalTrace::new(
            values
                .iter()
                .map(|(timestamp_micros, value)| TemporalSample {
                    timestamp_micros: *timestamp_micros,
                    value: *value,
                })
                .collect(),
        )
        .unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn trace_rejects_non_finite_and_non_monotonic_samples() {        assert!(matches!(
            TemporalTrace::new(vec![TemporalSample {
                timestamp_micros: 0,
                value: f64::NAN,
            }]),
            Err(TemporalMetricError::NonFinite { index: 0 })
        ));
        assert!(matches!(
            TemporalTrace::new(vec![
                TemporalSample {
                    timestamp_micros: 10,
                    value: 0.0,
                },
                TemporalSample {
                    timestamp_micros: 10,
                    value: 0.0,
                },
            ]),
            Err(TemporalMetricError::NonMonotonic { index: 1, .. })
        ));
    }

    #[test]
    fn constant_trace_has_zero_jitter_velocity_and_acceleration() {
        let metrics = temporal_noise_metrics(&trace(&[
            (0, 0.5),
            (10_000, 0.5),
            (30_000, 0.5),
            (60_000, 0.5),
        ]));
        assert_close(metrics.stationary_rms, 0.0);
        assert_close(metrics.first_difference_rms_per_second.unwrap(), 0.0);
        assert_close(metrics.second_difference_rms_per_second2.unwrap(), 0.0);
    }

    #[test]
    fn linear_motion_is_dt_aware_and_has_zero_acceleration() {
        let metrics = temporal_noise_metrics(&trace(&[
            (0, 0.0),
            (100_000, 0.1),
            (350_000, 0.35),
            (1_000_000, 1.0),
        ]));
        assert_close(metrics.first_difference_rms_per_second.unwrap(), 1.0);
        assert_close(metrics.second_difference_rms_per_second2.unwrap(), 0.0);
    }

    #[test]
    fn rising_step_reports_interpolated_10_50_90_and_peak() {
        let samples = trace(&[
            (0, 0.0),
            (100_000, 0.1),
            (200_000, 0.5),
            (300_000, 0.9),
            (400_000, 1.0),
            (500_000, 1.0),
        ]);
        let metrics = step_response_metrics(
            &samples,
            StepResponseSpec {
                command_micros: 0,
                baseline: 0.0,
                target: 1.0,
                settling_tolerance_fraction: 0.02,
            },
        )
        .unwrap();
        assert_close(metrics.t10_ms.unwrap(), 100.0);
        assert_close(metrics.t50_ms.unwrap(), 200.0);
        assert_close(metrics.t90_ms.unwrap(), 300.0);
        assert_close(metrics.rise_time_10_90_ms.unwrap(), 200.0);
        assert_close(metrics.peak_response_ratio, 1.0);
        assert_close(metrics.peak_attenuation, 0.0);
        assert_close(metrics.overshoot, 0.0);
        assert_close(metrics.settling_time_ms.unwrap(), 400.0);
    }

    #[test]
    fn falling_release_uses_the_same_progress_definition() {
        let samples = trace(&[
            (0, 1.0),
            (100_000, 0.9),
            (200_000, 0.5),
            (300_000, 0.1),
            (400_000, 0.0),
        ]);
        let metrics = step_response_metrics(
            &samples,
            StepResponseSpec {
                command_micros: 0,
                baseline: 1.0,
                target: 0.0,
                settling_tolerance_fraction: 0.02,
            },
        )
        .unwrap();
        assert_close(metrics.t10_ms.unwrap(), 100.0);
        assert_close(metrics.t50_ms.unwrap(), 200.0);
        assert_close(metrics.t90_ms.unwrap(), 300.0);
        assert_close(metrics.peak_response_ratio, 1.0);
    }

    #[test]
    fn step_peak_reports_attenuation_and_overshoot_without_hiding_lag() {
        let attenuated = step_response_metrics(
            &trace(&[(0, 0.0), (100_000, 0.2), (200_000, 0.7), (300_000, 0.8)]),
            StepResponseSpec {
                command_micros: 0,
                baseline: 0.0,
                target: 1.0,
                settling_tolerance_fraction: 0.05,
            },
        )
        .unwrap();
        assert_close(attenuated.peak_response_ratio, 0.8);
        assert_close(attenuated.peak_attenuation, 0.2);
        assert!(attenuated.t90_ms.is_none());

        let overshot = step_response_metrics(
            &trace(&[(0, 0.0), (100_000, 0.5), (200_000, 1.2), (300_000, 1.0)]),
            StepResponseSpec {
                command_micros: 0,
                baseline: 0.0,
                target: 1.0,
                settling_tolerance_fraction: 0.05,
            },
        )
        .unwrap();
        assert_close(overshot.peak_attenuation, 0.0);
        assert_close(overshot.overshoot, 0.2);
    }

    #[test]
    fn short_pulse_reports_peak_preservation_and_timing() {
        let metrics = pulse_response_metrics(
            &trace(&[
                (0, 0.0),
                (20_000, 0.3),
                (40_000, 0.8),
                (60_000, 0.4),
                (80_000, 0.0),
            ]),
            PulseResponseSpec {
                onset_micros: 0,
                baseline: 0.0,
                target_peak: 1.0,
                expected_peak_micros: Some(30_000),
            },
        )
        .unwrap();
        assert_close(metrics.peak_response_ratio, 0.8);
        assert_close(metrics.peak_attenuation, 0.2);
        assert_eq!(metrics.observed_peak_micros, 40_000);
        assert_close(metrics.observed_peak_delay_ms, 40.0);
        assert_close(metrics.peak_timing_error_ms.unwrap(), 10.0);
    }

    #[test]
    fn command_between_samples_is_interpolated_without_precommand_cheating() {
        let metrics = step_response_metrics(
            &trace(&[(0, 0.0), (100_000, 0.0), (200_000, 1.0), (300_000, 1.0)]),
            StepResponseSpec {
                command_micros: 150_000,
                baseline: 0.0,
                target: 1.0,
                settling_tolerance_fraction: 0.05,
            },
        )
        .unwrap();
        assert_close(metrics.t10_ms.unwrap(), 0.0);
        assert_close(metrics.t50_ms.unwrap(), 0.0);
        assert_close(metrics.t90_ms.unwrap(), 40.0);
    }

    #[test]
    fn cheek_hold_needs_more_than_half_reactive_samples() {
        // 4 s hold at 30 fps would be 120 samples; the rule is the fraction,
        // not the count, so a short trace exercises the same boundary.
        let held = trace(&[(0, 0.8), (33_333, 0.8), (66_666, 0.0), (99_999, 0.8)]);
        assert!(cheek_hold_detected(&held, 0.5));
        let tied = trace(&[(0, 0.8), (33_333, 0.0)]);
        assert!(!cheek_hold_detected(&tied, 0.5));
        let quiet = trace(&[(0, 0.1), (33_333, 0.2)]);
        assert!(!cheek_hold_detected(&quiet, 0.5));
    }

    #[test]
    fn cheek_take_evaluation_reports_clean_only_without_false_alarms() {
        let clean = CheekTakeEvaluation::new(&[true, true], &[false, false, false]);
        assert!(clean.is_clean());
        assert_eq!(clean.detected_puff, 2);
        let missed = CheekTakeEvaluation::new(&[true, false], &[false]);
        assert!(!missed.is_clean());
        let false_alarm = CheekTakeEvaluation::new(&[true], &[true]);
        assert!(!false_alarm.is_clean());
        assert!(!CheekTakeEvaluation::new(&[], &[]).is_clean());
    }

    #[test]
    fn cheek_take_splits_must_be_disjoint_to_claim_generalization() {
        assert!(cheek_takes_are_disjoint(
            &["take_a", "take_b"],
            &["take_c", "take_d"]
        ));
        assert!(!cheek_takes_are_disjoint(&["take_a", "take_b"], &["take_b"]));
    }
}
