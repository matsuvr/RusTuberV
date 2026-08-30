//! Held-out ablation of the learned causal linear prior against the
//! no-prior baselines on the same timeline (GNM #68.5 / Issue #112).
//!
//! Every evaluated frame is scored against the ARKit teacher coefficients:
//!
//! - `direct` — the direct MediaPipe blendshape → ARKit52 observation,
//! - `gnm-no-temporal` — the deterministic cold-start GNM projection,
//! - `learned-prior` — the causal linear prior's next-step prediction,
//!   produced from history features of the *previous* frame only and passed
//!   through the bounded `PriorRuntime` (corrections clamp, gaps reset).
//!
//! Splits are take-disjoint by construction: the caller names the training
//! takes when fitting (`teacher-fit-prior --train-take`) and evaluates on
//! different take directories here. Metrics cover value error (per-channel
//! MAE, aggregate MAE/RMSE), dt-aware velocity/acceleration/jerk errors,
//! variant jitter, and teacher-event onset/rise/pulse timing via the shared
//! `temporal_metrics` kernel. Head pose, per-frame inference latency, and
//! memory/CPU cost are not recorded by the current trace schema and are
//! reported as not verified.

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};
use vtuber_core::ArkitBlendshape;
use vtuber_tracking::{
    CausalFeatureConfig, CorrectionGroup, LinearPriorArtifact, PairedTemporalSample,
    PriorInference, PriorRuntime, PriorRuntimeConfig, PulseResponseSpec, StepResponseSpec,
    TemporalSample, TemporalTrace, pulse_response_metrics, step_response_metrics,
    validate_paired_samples,
};

use crate::teacher_fit_prior::{FEATURE_ORDER, load_trace};

const CHANNEL_COUNT: usize = 52;

/// Parsed CLI options for `teacher-ablation`.
pub struct Options {
    artifact: PathBuf,
    eval_traces: Vec<PathBuf>,
    train_takes: Vec<String>,
    output: PathBuf,
    history_len: usize,
    max_gap_micros: u64,
    expected_dt_micros: u64,
    gap_tolerance: f64,
    correction_bound: f32,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut artifact = None;
        let mut eval_traces = Vec::new();
        let mut train_takes = Vec::new();
        let mut output = None;
        let mut history_len = 4_usize;
        let mut max_gap_micros = 100_000_u64;
        let mut expected_dt_micros = 33_367_u64;
        let mut gap_tolerance = 1.5_f64;
        let mut correction_bound = 1.0_f32;
        let mut index = 0;
        while index < args.len() {
            let next = |index: &mut usize, flag: &str| -> Result<String, String> {
                *index += 1;
                args.get(*index)
                    .cloned()
                    .ok_or_else(|| format!("{flag} requires a value"))
            };
            // Bounds are proven by the loop condition; see AGENTS.md panic policy.
            #[allow(clippy::indexing_slicing)]
            match args[index].as_str() {
                "--artifact" => artifact = Some(PathBuf::from(next(&mut index, "--artifact")?)),
                "--eval-trace" => {
                    eval_traces.push(PathBuf::from(next(&mut index, "--eval-trace")?));
                }
                "--train-take" => train_takes.push(next(&mut index, "--train-take")?),
                "--output" => output = Some(PathBuf::from(next(&mut index, "--output")?)),
                "--history-len" => {
                    history_len = next(&mut index, "--history-len")?
                        .parse()
                        .map_err(|_| "--history-len must be a positive integer")?;
                }
                "--max-gap-micros" => {
                    max_gap_micros = next(&mut index, "--max-gap-micros")?
                        .parse()
                        .map_err(|_| "--max-gap-micros must be an integer")?;
                }
                "--expected-dt-micros" => {
                    expected_dt_micros = next(&mut index, "--expected-dt-micros")?
                        .parse()
                        .map_err(|_| "--expected-dt-micros must be an integer")?;
                }
                "--gap-tolerance" => {
                    gap_tolerance = next(&mut index, "--gap-tolerance")?
                        .parse()
                        .map_err(|_| "--gap-tolerance must be a float")?;
                }
                "--correction-bound" => {
                    correction_bound = next(&mut index, "--correction-bound")?
                        .parse()
                        .map_err(|_| "--correction-bound must be a float")?;
                }
                other => return Err(format!("unknown option {other}")),
            }
            index += 1;
        }
        if eval_traces.is_empty() {
            return Err("at least one --eval-trace <replay-output-dir> is required".to_owned());
        }
        Ok(Self {
            artifact: artifact.ok_or("--artifact <linear-prior.json> is required")?,
            eval_traces,
            train_takes,
            output: output.map_or_else(|| PathBuf::from("data/datasets/ablation"), PathBuf::from),
            history_len,
            max_gap_micros,
            expected_dt_micros,
            gap_tolerance,
            correction_bound,
        })
    }
}

/// Prints command help.
pub fn print_help() {
    println!(
        "  teacher-ablation --artifact <linear-prior.json> --eval-trace <replay-dir> [...]\n\
         *                   [--train-take <id> ...] [--output <dir>]\n\
         *                   [--history-len 4] [--max-gap-micros 100000]\n\
         *                   [--expected-dt-micros 33367] [--gap-tolerance 1.5]\n\
         *                   [--correction-bound 1.0]\n\
         *   Held-out ablation (#112): scores direct / gnm-no-temporal / learned-prior\n\
         *   against the ARKit teacher on the same timeline (value, velocity,\n\
         *   acceleration, jerk, jitter, onset/rise/pulse timing). Eval takes must be\n\
         *   disjoint from the training takes."
    );
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect()
}

// ---------------------------------------------------------------------------
// Streaming error accumulation
// ---------------------------------------------------------------------------

/// One evaluated frame's 52-channel values for one series.
struct FrameValues {
    timestamp_micros: u64,
    values: [f64; CHANNEL_COUNT],
}

/// History entry holding the variant and teacher values of one frame.
struct HistoryFrame {
    timestamp_micros: u64,
    variant: [f64; CHANNEL_COUNT],
    teacher: [f64; CHANNEL_COUNT],
}

/// Streaming value + derivative error statistics for one variant.
///
/// All fields are additive sums/counts so per-take statistics merge into an
/// overall aggregate by field-wise addition (`merge_from`).
struct VariantErrors {
    frames: u64,
    per_channel_abs: [f64; CHANNEL_COUNT],
    per_channel_sq: [f64; CHANNEL_COUNT],
    per_channel_count: [u64; CHANNEL_COUNT],
    velocity_abs: f64,
    velocity_sq: f64,
    velocity_count: u64,
    accel_abs: f64,
    accel_sq: f64,
    accel_count: u64,
    jerk_abs: f64,
    jerk_sq: f64,
    jerk_count: u64,
    variant_velocity_sq: f64,
    teacher_velocity_sq: f64,
    /// Transient ring of the last frames used for the derivatives.
    history: VecDeque<HistoryFrame>,
}

impl Default for VariantErrors {
    fn default() -> Self {
        Self {
            frames: 0,
            per_channel_abs: [0.0; CHANNEL_COUNT],
            per_channel_sq: [0.0; CHANNEL_COUNT],
            per_channel_count: [0; CHANNEL_COUNT],
            velocity_abs: 0.0,
            velocity_sq: 0.0,
            velocity_count: 0,
            accel_abs: 0.0,
            accel_sq: 0.0,
            accel_count: 0,
            jerk_abs: 0.0,
            jerk_sq: 0.0,
            jerk_count: 0,
            variant_velocity_sq: 0.0,
            teacher_velocity_sq: 0.0,
            history: VecDeque::new(),
        }
    }
}

impl VariantErrors {
    /// Adds one evaluated frame plus its teacher reference.
    fn push(&mut self, frame: FrameValues, teacher: &FrameValues) {
        if let Some(previous) = self.history.back() {
            let dt_seconds = (frame
                .timestamp_micros
                .saturating_sub(previous.timestamp_micros))
            .max(1) as f64
                / 1.0e6;
            let mut velocity_variant = [0.0_f64; CHANNEL_COUNT];
            let mut velocity_teacher = [0.0_f64; CHANNEL_COUNT];
            for channel in 0..CHANNEL_COUNT {
                // Bounds: fixed 52-channel arrays.
                #[allow(clippy::indexing_slicing)]
                {
                    velocity_variant[channel] =
                        (frame.values[channel] - previous.variant[channel]) / dt_seconds;
                    velocity_teacher[channel] =
                        (teacher.values[channel] - previous.teacher[channel]) / dt_seconds;
                    let error = velocity_variant[channel] - velocity_teacher[channel];
                    self.velocity_abs += error.abs();
                    self.velocity_sq += error * error;
                    self.variant_velocity_sq +=
                        velocity_variant[channel] * velocity_variant[channel];
                    self.teacher_velocity_sq +=
                        velocity_teacher[channel] * velocity_teacher[channel];
                }
            }
            self.velocity_count += 1;

            if self.history.len() >= 2 {
                // Bounds: length checked above.
                #[allow(clippy::indexing_slicing)]
                let previous2 = &self.history[self.history.len() - 2];
                let dt1 = (previous
                    .timestamp_micros
                    .saturating_sub(previous2.timestamp_micros))
                .max(1) as f64
                    / 1.0e6;
                let mut accel_variant = [0.0_f64; CHANNEL_COUNT];
                let mut accel_teacher = [0.0_f64; CHANNEL_COUNT];
                for channel in 0..CHANNEL_COUNT {
                    // Bounds: fixed 52-channel arrays.
                    #[allow(clippy::indexing_slicing)]
                    {
                        let previous_velocity_variant =
                            (previous.variant[channel] - previous2.variant[channel]) / dt1;
                        let previous_velocity_teacher =
                            (previous.teacher[channel] - previous2.teacher[channel]) / dt1;
                        accel_variant[channel] =
                            (velocity_variant[channel] - previous_velocity_variant) / dt_seconds;
                        accel_teacher[channel] =
                            (velocity_teacher[channel] - previous_velocity_teacher) / dt_seconds;
                        let error = accel_variant[channel] - accel_teacher[channel];
                        self.accel_abs += error.abs();
                        self.accel_sq += error * error;
                    }
                }
                self.accel_count += 1;

                if self.history.len() >= 3 {
                    // Bounds: length checked above.
                    #[allow(clippy::indexing_slicing)]
                    let previous3 = &self.history[self.history.len() - 3];
                    let dt2 = (previous2
                        .timestamp_micros
                        .saturating_sub(previous3.timestamp_micros))
                    .max(1) as f64
                        / 1.0e6;
                    for channel in 0..CHANNEL_COUNT {
                        // Bounds: fixed 52-channel arrays.
                        #[allow(clippy::indexing_slicing)]
                        {
                            let previous_velocity_variant =
                                (previous.variant[channel] - previous2.variant[channel]) / dt1;
                            let previous_velocity_teacher =
                                (previous.teacher[channel] - previous2.teacher[channel]) / dt1;
                            let previous2_velocity_variant =
                                (previous2.variant[channel] - previous3.variant[channel]) / dt2;
                            let previous2_velocity_teacher =
                                (previous2.teacher[channel] - previous3.teacher[channel]) / dt2;
                            let previous_accel_variant = (previous_velocity_variant
                                - previous2_velocity_variant)
                                / dt_seconds;
                            let previous_accel_teacher = (previous_velocity_teacher
                                - previous2_velocity_teacher)
                                / dt_seconds;
                            let error = (accel_variant[channel]
                                - previous_accel_variant
                                - (accel_teacher[channel] - previous_accel_teacher))
                                / dt_seconds;
                            self.jerk_abs += error.abs();
                            self.jerk_sq += error * error;
                        }
                    }
                    self.jerk_count += 1;
                }
            }
        }

        for channel in 0..CHANNEL_COUNT {
            // Bounds: fixed 52-channel arrays.
            #[allow(clippy::indexing_slicing)]
            {
                let error = frame.values[channel] - teacher.values[channel];
                self.per_channel_abs[channel] += error.abs();
                self.per_channel_sq[channel] += error * error;
                self.per_channel_count[channel] += 1;
            }
        }
        self.frames += 1;
        if self.history.len() == 3 {
            self.history.pop_front();
        }
        self.history.push_back(HistoryFrame {
            timestamp_micros: frame.timestamp_micros,
            variant: frame.values,
            teacher: teacher.values,
        });
    }

    /// Field-wise merge of another accumulator into this one.
    fn merge_from(&mut self, other: &VariantErrors) {
        self.frames += other.frames;
        for channel in 0..CHANNEL_COUNT {
            // Bounds: fixed 52-channel arrays.
            #[allow(clippy::indexing_slicing)]
            {
                self.per_channel_abs[channel] += other.per_channel_abs[channel];
                self.per_channel_sq[channel] += other.per_channel_sq[channel];
                self.per_channel_count[channel] += other.per_channel_count[channel];
            }
        }
        self.velocity_abs += other.velocity_abs;
        self.velocity_sq += other.velocity_sq;
        self.velocity_count += other.velocity_count;
        self.accel_abs += other.accel_abs;
        self.accel_sq += other.accel_sq;
        self.accel_count += other.accel_count;
        self.jerk_abs += other.jerk_abs;
        self.jerk_sq += other.jerk_sq;
        self.jerk_count += other.jerk_count;
        self.variant_velocity_sq += other.variant_velocity_sq;
        self.teacher_velocity_sq += other.teacher_velocity_sq;
    }

    /// Aggregate value metrics for one variant.
    fn value_metrics(&self) -> ValueMetrics {
        let mut total_abs = 0.0_f64;
        let mut total_sq = 0.0_f64;
        let mut total_count = 0_u64;
        let mut worst_channel = 0_usize;
        let mut worst_channel_mae = 0.0_f64;
        for channel in 0..CHANNEL_COUNT {
            // Bounds: fixed 52-channel arrays.
            #[allow(clippy::indexing_slicing)]
            {
                total_abs += self.per_channel_abs[channel];
                total_sq += self.per_channel_sq[channel];
                total_count += self.per_channel_count[channel];
                let channel_mae =
                    self.per_channel_abs[channel] / self.per_channel_count[channel].max(1) as f64;
                if channel_mae > worst_channel_mae {
                    worst_channel_mae = channel_mae;
                    worst_channel = channel;
                }
            }
        }
        ValueMetrics {
            frames: self.frames,
            mae: total_abs / total_count.max(1) as f64,
            rmse: (total_sq / total_count.max(1) as f64).sqrt(),
            worst_channel,
            worst_channel_mae,
        }
    }

    /// Temporal derivative metrics for one variant.
    fn temporal_metrics(&self) -> TemporalMetrics {
        TemporalMetrics {
            velocity_mae: self.velocity_abs / self.velocity_count.max(1) as f64,
            acceleration_mae: self.accel_abs / self.accel_count.max(1) as f64,
            jerk_mae: self.jerk_abs / self.jerk_count.max(1) as f64,
            variant_jitter_velocity_rms: (self.variant_velocity_sq
                / self.velocity_count.max(1) as f64)
                .sqrt(),
            teacher_jitter_velocity_rms: (self.teacher_velocity_sq
                / self.velocity_count.max(1) as f64)
                .sqrt(),
        }
    }
}

/// Aggregate value metrics for one variant.
#[derive(Serialize, Clone, Copy, Debug, Default)]
struct ValueMetrics {
    frames: u64,
    mae: f64,
    rmse: f64,
    worst_channel: usize,
    worst_channel_mae: f64,
}

/// Temporal derivative metrics for one variant.
#[derive(Serialize, Clone, Copy, Debug, Default)]
struct TemporalMetrics {
    velocity_mae: f64,
    acceleration_mae: f64,
    jerk_mae: f64,
    variant_jitter_velocity_rms: f64,
    teacher_jitter_velocity_rms: f64,
}

// ---------------------------------------------------------------------------
// Teacher-driven event timing
// ---------------------------------------------------------------------------

/// One teacher-detected expressive event.
struct TeacherEvent {
    kind: EventKind,
    /// Index into the teacher frame series (== trace sample index when the
    /// series covers every sample).
    onset_index: usize,
}

enum EventKind {
    /// Blink-like pulse with the teacher peak timestamp.
    Pulse {
        baseline: f64,
        peak: f64,
        expected_peak_micros: u64,
    },
    /// Sustained rise with the plateau target.
    Rise { baseline: f64, target: f64 },
}

/// Detects blink pulses and sustained rises on the teacher series.
///
/// `times` are microseconds; events closer than 0.3 s to the previous one on
/// the same channel are skipped; detection never modifies evaluated data.
fn detect_events(
    channel: ArkitBlendshape,
    times: &[u64],
    values: &[f64],
    max_events: usize,
) -> Vec<TeacherEvent> {
    let mut events = Vec::new();
    let mut last_onset = 0_u64;
    for index in 1..values.len() {
        if events.len() >= max_events {
            break;
        }
        // Bounds: index < values.len() from the loop range.
        #[allow(clippy::indexing_slicing)]
        let (previous, current, time) = (values[index - 1], values[index], times[index]);
        if time.saturating_sub(last_onset) < 300_000 {
            continue;
        }
        if matches!(
            channel,
            ArkitBlendshape::EyeBlinkLeft | ArkitBlendshape::EyeBlinkRight
        ) && current >= 0.30
            && previous < 0.6 * current
        {
            // Pulse: a fall back below half the peak within 350 ms.
            let mut peak = current;
            let mut peak_index = index;
            let mut fell = false;
            // Bounds: the slice starts at `index` and `at` stays in range.
            #[allow(clippy::indexing_slicing)]
            for (offset, value) in values[index..].iter().enumerate() {
                let at = index + offset;
                if times[at].saturating_sub(time) > 350_000 {
                    break;
                }
                if *value > peak {
                    peak = *value;
                    peak_index = at;
                }
                if *value <= 0.5 * peak {
                    fell = true;
                    break;
                }
            }
            if fell {
                let baseline = if index >= 3 {
                    // Bounds: index >= 3 checked above.
                    #[allow(clippy::indexing_slicing)]
                    {
                        values[index - 3..index].iter().sum::<f64>() / 3.0
                    }
                } else {
                    previous
                };
                // Bounds: peak_index < values.len() by construction.
                #[allow(clippy::indexing_slicing)]
                let expected_peak_micros = times[peak_index];
                let event = TeacherEvent {
                    kind: EventKind::Pulse {
                        baseline,
                        peak,
                        expected_peak_micros,
                    },
                    onset_index: index,
                };
                events.push(event);
                last_onset = time;
            }
        } else if current >= 0.30 && previous < 0.15 && index >= 3 && index + 5 <= values.len() {
            // Bounds: index >= 3 and index + 5 <= len checked above.
            #[allow(clippy::indexing_slicing)]
            let baseline = values[index - 3..index].iter().sum::<f64>() / 3.0;
            // Bounds: index + 5 <= len checked above.
            #[allow(clippy::indexing_slicing)]
            let sustained_mean = values[index..index + 5].iter().sum::<f64>() / 5.0;
            if sustained_mean >= 0.7 * current {
                // Bounds: index + 5 <= len checked above.
                #[allow(clippy::indexing_slicing)]
                let target = values[index..index + 5]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                events.push(TeacherEvent {
                    kind: EventKind::Rise { baseline, target },
                    onset_index: index,
                });
                last_onset = time;
            }
        }
    }
    events
}

/// One variant's scalar series aligned with trace sample indices.
struct ScalarSeries {
    sample_indices: Vec<usize>,
    times: Vec<u64>,
    values: Vec<f64>,
}

impl ScalarSeries {
    fn build(
        samples: &[PairedTemporalSample],
        pick: impl Fn(&PairedTemporalSample) -> Option<f64>,
    ) -> Self {
        let mut series = Self {
            sample_indices: Vec::new(),
            times: Vec::new(),
            values: Vec::new(),
        };
        for (index, sample) in samples.iter().enumerate() {
            if let Some(value) = pick(sample) {
                series.sample_indices.push(index);
                series.times.push(sample.timestamp_micros);
                series.values.push(value);
            }
        }
        series
    }

    /// Series position for a trace sample index (binary search).
    fn position_of(&self, sample_index: usize) -> Option<usize> {
        self.sample_indices
            .binary_search(&sample_index)
            .ok()
            .or_else(|| {
                // Fall back to the last position before the requested index so
                // events whose exact frame is missing from this series still
                // resolve to the closest earlier sample.
                self.sample_indices
                    .iter()
                    .rposition(|&index| index < sample_index)
            })
    }
}

/// Score result: `(step onset delay ms, step rise time ms, pulse attenuation,
/// pulse timing error ms)`; exactly one event kind contributes values.
type EventScore = (Option<f64>, Option<f64>, Option<f64>, Option<f64>);

/// Scores one variant series against one teacher event.
fn score_event(event: &TeacherEvent, series: &ScalarSeries) -> Result<EventScore, String> {
    let Some(center) = series.position_of(event.onset_index) else {
        return Ok((None, None, None, None));
    };
    // Bounds: position_of returns a valid position or None.
    #[allow(clippy::indexing_slicing)]
    let onset = series.times[center];
    let window_start = onset.saturating_sub(150_000);
    let window_end = onset.saturating_add(600_000);
    let mut samples = Vec::new();
    for (position, time) in series.times.iter().enumerate() {
        if *time >= window_start && *time <= window_end {
            // Bounds: parallel arrays of equal length.
            #[allow(clippy::indexing_slicing)]
            samples.push(TemporalSample {
                timestamp_micros: *time,
                value: series.values[position],
            });
        }
    }
    let trace = TemporalTrace::new(samples).map_err(|error| format!("{error:?}"))?;
    match &event.kind {
        EventKind::Rise { baseline, target } => {
            let metrics = step_response_metrics(
                &trace,
                StepResponseSpec {
                    command_micros: onset,
                    baseline: *baseline,
                    target: *target,
                    settling_tolerance_fraction: 0.2,
                },
            )
            .map_err(|error| format!("{error:?}"))?;
            Ok((
                metrics.onset_delay_ms,
                metrics.rise_time_10_90_ms,
                None,
                None,
            ))
        }
        EventKind::Pulse {
            baseline,
            peak,
            expected_peak_micros,
        } => {
            let metrics = pulse_response_metrics(
                &trace,
                PulseResponseSpec {
                    onset_micros: onset,
                    baseline: *baseline,
                    target_peak: *peak,
                    expected_peak_micros: Some(*expected_peak_micros),
                },
            )
            .map_err(|error| format!("{error:?}"))?;
            Ok((
                None,
                None,
                Some(metrics.peak_attenuation),
                metrics.peak_timing_error_ms,
            ))
        }
    }
}

/// Per-variant event timing aggregates.
#[derive(Serialize, Clone, Copy, Debug, Default)]
struct EventTimingMetrics {
    step_events: u64,
    mean_onset_delay_ms: f64,
    mean_rise_time_ms: f64,
    pulse_events: u64,
    mean_peak_attenuation: f64,
    mean_peak_timing_error_ms: f64,
    events_unmeasurable: u64,
}

/// Accumulates step and pulse scores for one variant.
#[derive(Default)]
struct EventAggregates {
    step_onset_sum: f64,
    step_onset_count: u64,
    step_rise_sum: f64,
    step_rise_count: u64,
    pulse_attenuation_sum: f64,
    pulse_attenuation_count: u64,
    pulse_timing_sum: f64,
    pulse_timing_count: u64,
    unmeasurable: u64,
}

impl EventAggregates {
    fn add(
        &mut self,
        onset: Option<f64>,
        rise: Option<f64>,
        attenuation: Option<f64>,
        timing: Option<f64>,
    ) {
        if let Some(value) = onset {
            self.step_onset_sum += value;
            self.step_onset_count += 1;
        }
        if let Some(value) = rise {
            self.step_rise_sum += value;
            self.step_rise_count += 1;
        }
        if let Some(value) = attenuation {
            self.pulse_attenuation_sum += value;
            self.pulse_attenuation_count += 1;
        }
        if let Some(value) = timing {
            self.pulse_timing_sum += value.abs();
            self.pulse_timing_count += 1;
        }
        if onset.is_none() && rise.is_none() && attenuation.is_none() && timing.is_none() {
            self.unmeasurable += 1;
        }
    }

    fn metrics(&self) -> EventTimingMetrics {
        EventTimingMetrics {
            step_events: self.step_onset_count,
            mean_onset_delay_ms: self.step_onset_sum / self.step_onset_count.max(1) as f64,
            mean_rise_time_ms: self.step_rise_sum / self.step_rise_count.max(1) as f64,
            pulse_events: self.pulse_attenuation_count,
            mean_peak_attenuation: self.pulse_attenuation_sum
                / self.pulse_attenuation_count.max(1) as f64,
            mean_peak_timing_error_ms: self.pulse_timing_sum
                / self.pulse_timing_count.max(1) as f64,
            events_unmeasurable: self.unmeasurable,
        }
    }
}

// ---------------------------------------------------------------------------
// Report model
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct AblationReport {
    schema_version: u32,
    tool: &'static str,
    xtask_version: String,
    split: SplitProvenance,
    config: AblationConfig,
    takes: Vec<TakeReport>,
    overall: OverallReport,
    constraints: Constraints,
}

#[derive(Serialize)]
struct SplitProvenance {
    train_takes: Vec<String>,
    artifact_sha256: String,
    eval_takes: Vec<String>,
}

#[derive(Serialize)]
struct AblationConfig {
    history_len: usize,
    max_gap_micros: u64,
    expected_dt_micros: u64,
    gap_tolerance: f64,
    correction_bound: f32,
    feature_order: &'static str,
}

#[derive(Serialize, Clone)]
struct TakeReport {
    take_id: String,
    frames: usize,
    usable_rows: usize,
    prior_active_rows: usize,
    prior_failed_rows: usize,
    resets: u64,
    clamped: u64,
    direct: ValueMetrics,
    gnm_no_temporal: ValueMetrics,
    learned_prior: ValueMetrics,
    direct_temporal: TemporalMetrics,
    gnm_no_temporal_temporal: TemporalMetrics,
    learned_prior_temporal: TemporalMetrics,
    direct_events: EventTimingMetrics,
    gnm_no_temporal_events: EventTimingMetrics,
    learned_prior_events: EventTimingMetrics,
}

#[derive(Serialize)]
struct OverallReport {
    frames: u64,
    direct: ValueMetrics,
    gnm_no_temporal: ValueMetrics,
    learned_prior: ValueMetrics,
    direct_temporal: TemporalMetrics,
    gnm_no_temporal_temporal: TemporalMetrics,
    learned_prior_temporal: TemporalMetrics,
}

#[derive(Serialize)]
struct Constraints {
    person_count: u32,
    capture_device: &'static str,
    generalization_note: &'static str,
    not_verified: Vec<&'static str>,
}

/// Runs the ablation; see the module documentation.
///
/// # Errors
///
/// Fails closed on trace validation failures, artifact load failures, and
/// output I/O errors.
#[allow(clippy::too_many_lines)]
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let artifact_text = fs::read_to_string(&options.artifact)
        .map_err(|error| format!("read {}: {error}", options.artifact.display()))?;
    let artifact: LinearPriorArtifact = serde_json::from_str(&artifact_text)
        .map_err(|error| format!("parse {}: {error}", options.artifact.display()))?;
    let artifact_sha256 = sha256_hex(artifact_text.as_bytes());

    let feature_config = CausalFeatureConfig {
        history_len: options.history_len,
        max_gap_micros: options.max_gap_micros,
    };
    let mut takes = Vec::new();
    for directory in &options.eval_traces {
        takes.push(evaluate_take(
            directory,
            &artifact,
            &options,
            feature_config,
        )?);
    }

    // Frame-weighted aggregation across takes.
    let mut overall_direct = VariantErrors::default();
    let mut overall_gnm = VariantErrors::default();
    let mut overall_prior = VariantErrors::default();
    let mut total_frames = 0_u64;
    for take in &takes {
        overall_direct.merge_from(&take.errors.direct);
        overall_gnm.merge_from(&take.errors.gnm);
        overall_prior.merge_from(&take.errors.prior);
        total_frames += take.report.frames as u64;
    }

    let report = AblationReport {
        schema_version: 1,
        tool: "xtask teacher-ablation",
        xtask_version: env!("CARGO_PKG_VERSION").to_owned(),
        split: SplitProvenance {
            train_takes: options.train_takes.clone(),
            artifact_sha256,
            eval_takes: takes
                .iter()
                .map(|take| take.report.take_id.clone())
                .collect(),
        },
        config: AblationConfig {
            history_len: options.history_len,
            max_gap_micros: options.max_gap_micros,
            expected_dt_micros: options.expected_dt_micros,
            gap_tolerance: options.gap_tolerance,
            correction_bound: options.correction_bound,
            feature_order: FEATURE_ORDER,
        },
        takes: takes.iter().map(|take| take.report.clone()).collect(),
        overall: OverallReport {
            frames: total_frames,
            direct: overall_direct.value_metrics(),
            gnm_no_temporal: overall_gnm.value_metrics(),
            learned_prior: overall_prior.value_metrics(),
            direct_temporal: overall_direct.temporal_metrics(),
            gnm_no_temporal_temporal: overall_gnm.temporal_metrics(),
            learned_prior_temporal: overall_prior.temporal_metrics(),
        },
        constraints: Constraints {
            person_count: 1,
            capture_device: "iPhone front camera via the GNM #68.2 capture app",
            generalization_note: "Single person, same device and room; results do not \
                generalize across people or capture conditions.",
            not_verified: vec![
                "head pose comparison (MediaPipe head transform is not stored in the trace)",
                "per-frame inference latency",
                "memory/CPU cost",
                "fixed/adaptive temporal baselines from GNM #57.x (not stored in the trace)",
            ],
        },
    };

    fs::create_dir_all(&options.output)
        .map_err(|error| format!("create {}: {error}", options.output.display()))?;
    let report_path = options.output.join("ablation-report.json");
    let json =
        serde_json::to_string_pretty(&report).map_err(|error| format!("encode report: {error}"))?;
    fs::write(&report_path, json.as_bytes())
        .map_err(|error| format!("write {}: {error}", report_path.display()))?;

    println!(
        "teacher-ablation: {} eval takes, {} frames",
        takes.len(),
        total_frames
    );
    for take in &takes {
        println!(
            "  {}: direct MAE {:.5} | gnm {:.5} | prior {:.5} (prior active {}/{} rows, resets {})",
            take.report.take_id,
            take.report.direct.mae,
            take.report.gnm_no_temporal.mae,
            take.report.learned_prior.mae,
            take.report.prior_active_rows,
            take.report.usable_rows,
            take.report.resets
        );
    }
    println!(
        "  overall: direct MAE {:.5} RMSE {:.5} | gnm MAE {:.5} RMSE {:.5} | prior MAE {:.5} RMSE {:.5}",
        report.overall.direct.mae,
        report.overall.direct.rmse,
        report.overall.gnm_no_temporal.mae,
        report.overall.gnm_no_temporal.rmse,
        report.overall.learned_prior.mae,
        report.overall.learned_prior.rmse
    );
    println!(
        "  velocity MAE: direct {:.5} | gnm {:.5} | prior {:.5}",
        report.overall.direct_temporal.velocity_mae,
        report.overall.gnm_no_temporal_temporal.velocity_mae,
        report.overall.learned_prior_temporal.velocity_mae
    );
    println!("report: {}", report_path.display());
    Ok(())
}

struct EvaluatedTake {
    report: TakeReport,
    errors: PerTakeErrors,
}

struct PerTakeErrors {
    direct: VariantErrors,
    gnm: VariantErrors,
    prior: VariantErrors,
}

#[allow(clippy::too_many_lines)]
fn evaluate_take(
    directory: &Path,
    artifact: &LinearPriorArtifact,
    options: &Options,
    feature_config: CausalFeatureConfig,
) -> Result<EvaluatedTake, String> {
    let trace = load_trace(directory)?;
    validate_paired_samples(&trace.samples)
        .map_err(|error| format!("take {}: invalid trace: {error:?}", trace.take_id))?;
    let dataset =
        vtuber_tracking::build_causal_dataset(&trace.take_id, &trace.samples, feature_config)
            .map_err(|error| format!("take {}: causal dataset: {error:?}", trace.take_id))?;

    let sample_index: BTreeMap<u64, usize> = trace
        .samples
        .iter()
        .enumerate()
        .map(|(index, sample)| (sample.frame_seq, index))
        .collect();
    let inference = PriorInference::load_or_baseline(Some(artifact.clone()), FEATURE_ORDER);
    let mut runtime = PriorRuntime::new(
        inference,
        PriorRuntimeConfig {
            groups: vec![CorrectionGroup {
                name: "all".to_owned(),
                channel_start: 0,
                channel_end: CHANNEL_COUNT,
                max_abs_correction: options.correction_bound,
            }],
            expected_dt_micros: options.expected_dt_micros,
            gap_tolerance: options.gap_tolerance,
        },
    )
    .map_err(|error| format!("prior runtime: {error:?}"))?;

    let mut errors = PerTakeErrors {
        direct: VariantErrors::default(),
        gnm: VariantErrors::default(),
        prior: VariantErrors::default(),
    };
    let mut prior_active = 0_usize;
    let mut prior_failed = 0_usize;
    let mut resets = 0_u64;
    let mut clamped = 0_u64;
    // Per-channel prior prediction series (timestamp, value) keyed at the
    // predicted (successor) frame.
    let mut prior_series: Vec<Vec<(u64, f64)>> = vec![Vec::new(); CHANNEL_COUNT];

    for row in &dataset.rows {
        let outcome = runtime
            .advance(row.frame_seq, row.timestamp_micros, &row.features)
            .map_err(|error| format!("take {}: prior advance: {error:?}", trace.take_id))?;
        if outcome.reset.is_some() {
            resets += 1;
        }
        if outcome.clamped {
            clamped += 1;
        }
        // The prediction made at `row.frame_seq` targets its exact successor.
        let target_seq = row.frame_seq + 1;
        let Some(&target_index) = sample_index.get(&target_seq) else {
            continue;
        };
        // Bounds: index comes from the map built over these samples.
        #[allow(clippy::indexing_slicing)]
        let target = &trace.samples[target_index];
        let (Some(teacher), Some(direct), Some(gnm_state)) = (
            target.teacher.as_ref(),
            target.mediapipe_observation.as_ref(),
            target.gnm_state.as_ref(),
        ) else {
            continue;
        };
        let timestamp = target.timestamp_micros;
        let to_frame = |values: &[f32]| {
            let mut converted = [0.0_f64; CHANNEL_COUNT];
            for (slot, value) in converted.iter_mut().zip(values.iter()) {
                *slot = f64::from(*value);
            }
            FrameValues {
                timestamp_micros: timestamp,
                values: converted,
            }
        };
        let teacher_frame = to_frame(teacher.coefficients.as_array());
        errors
            .direct
            .push(to_frame(direct.as_array()), &teacher_frame);
        errors.gnm.push(
            to_frame(gnm_state.projected_coefficients.as_array()),
            &teacher_frame,
        );
        match &outcome.prior_state {
            Some(state) if state.len() == CHANNEL_COUNT => {
                prior_active += 1;
                // Bounds: length checked above.
                #[allow(clippy::indexing_slicing)]
                let predicted = to_frame(state);
                for (channel, values) in prior_series.iter_mut().enumerate() {
                    // Bounds: channel < CHANNEL_COUNT by the loop range.
                    #[allow(clippy::indexing_slicing)]
                    values.push((timestamp, predicted.values[channel]));
                }
                errors.prior.push(predicted, &teacher_frame);
            }
            Some(_) => prior_failed += 1,
            None => prior_failed += 1,
        }
    }

    // Teacher-driven event timing on the evaluated frames.
    let mut aggregates = EventAggregatesSet::default();
    let event_channels = [
        (ArkitBlendshape::EyeBlinkLeft, 30usize),
        (ArkitBlendshape::EyeBlinkRight, 30usize),
        (ArkitBlendshape::JawOpen, 15usize),
        (ArkitBlendshape::BrowInnerUp, 15usize),
        (ArkitBlendshape::MouthSmileLeft, 15usize),
        (ArkitBlendshape::MouthSmileRight, 15usize),
    ];
    for (channel, max_events) in event_channels {
        let channel_index = channel.index();
        // Bounds: `ArkitBlendshape::index()` < 52 by contract.
        #[allow(clippy::indexing_slicing)]
        let teacher_series = ScalarSeries::build(&trace.samples, |sample| {
            #[allow(clippy::indexing_slicing)]
            let value = sample
                .teacher
                .as_ref()
                .map(|teacher| teacher.coefficients.as_array()[channel_index]);
            value.map(f64::from)
        });
        if teacher_series.values.is_empty() {
            continue;
        }
        let direct_series = ScalarSeries::build(&trace.samples, |sample| {
            #[allow(clippy::indexing_slicing)]
            let value = sample
                .mediapipe_observation
                .as_ref()
                .map(|values| values.as_array()[channel_index]);
            value.map(f64::from)
        });
        let gnm_series = ScalarSeries::build(&trace.samples, |sample| {
            #[allow(clippy::indexing_slicing)]
            let value = sample
                .gnm_state
                .as_ref()
                .map(|state| state.projected_coefficients.as_array()[channel_index]);
            value.map(f64::from)
        });
        // Bounds: channel_index < 52 by contract.
        #[allow(clippy::indexing_slicing)]
        let prior_channel_series = &prior_series[channel_index];
        let prior_series_scalar = ScalarSeries {
            sample_indices: Vec::new(),
            times: prior_channel_series.iter().map(|(time, _)| *time).collect(),
            values: prior_channel_series
                .iter()
                .map(|(_, value)| *value)
                .collect(),
        };
        for event in detect_events(
            channel,
            &teacher_series.times,
            &teacher_series.values,
            max_events,
        ) {
            let scored_direct = score_event(&event, &direct_series)?;
            aggregates.direct.add(
                scored_direct.0,
                scored_direct.1,
                scored_direct.2,
                scored_direct.3,
            );
            let scored_gnm = score_event(&event, &gnm_series)?;
            aggregates
                .gnm
                .add(scored_gnm.0, scored_gnm.1, scored_gnm.2, scored_gnm.3);
            let scored_prior = score_event(&event, &prior_series_scalar)?;
            aggregates.prior.add(
                scored_prior.0,
                scored_prior.1,
                scored_prior.2,
                scored_prior.3,
            );
        }
    }

    let report = TakeReport {
        take_id: trace.take_id.clone(),
        frames: trace.samples.len(),
        usable_rows: dataset.rows.len(),
        prior_active_rows: prior_active,
        prior_failed_rows: prior_failed,
        resets,
        clamped,
        direct: errors.direct.value_metrics(),
        gnm_no_temporal: errors.gnm.value_metrics(),
        learned_prior: errors.prior.value_metrics(),
        direct_temporal: errors.direct.temporal_metrics(),
        gnm_no_temporal_temporal: errors.gnm.temporal_metrics(),
        learned_prior_temporal: errors.prior.temporal_metrics(),
        direct_events: aggregates.direct.metrics(),
        gnm_no_temporal_events: aggregates.gnm.metrics(),
        learned_prior_events: aggregates.prior.metrics(),
    };
    Ok(EvaluatedTake { report, errors })
}

/// Splits the tuple returned by `score_event` into the flat form the
/// aggregates expect.
#[derive(Default)]
struct EventAggregatesSet {
    direct: EventAggregates,
    gnm: EventAggregates,
    prior: EventAggregates,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivative_errors_are_exact_for_a_known_ramp() {
        let mut errors = VariantErrors::default();
        // Perfect prediction: variant == teacher. Constant slope 0.01/frame
        // at 1 ms dt -> velocity 10/s, acceleration 0.
        for seq in 0..6_u64 {
            let value = 0.01 * seq as f64;
            let frame = FrameValues {
                timestamp_micros: seq * 1_000,
                values: [value; CHANNEL_COUNT],
            };
            errors.push(
                FrameValues {
                    timestamp_micros: seq * 1_000,
                    values: [value; CHANNEL_COUNT],
                },
                &frame,
            );
        }
        let temporal = errors.temporal_metrics();
        assert_eq!(errors.frames, 6);
        assert!(temporal.velocity_mae.is_nan() || temporal.velocity_mae == 0.0);
        // Perfect tracking: every derivative error is exactly zero.
        assert_eq!(errors.velocity_abs, 0.0);
        assert_eq!(errors.accel_abs, 0.0);
        assert_eq!(errors.jerk_abs, 0.0);
        let value_metrics = errors.value_metrics();
        assert_eq!(value_metrics.mae, 0.0);
        assert_eq!(value_metrics.rmse, 0.0);
    }

    #[test]
    fn constant_offset_produces_value_error_but_zero_derivative_error() {
        let mut errors = VariantErrors::default();
        for seq in 0..6_u64 {
            let value = 0.01 * seq as f64;
            let teacher = FrameValues {
                timestamp_micros: seq * 1_000,
                values: [value; CHANNEL_COUNT],
            };
            let variant = FrameValues {
                timestamp_micros: seq * 1_000,
                values: [value + 0.1; CHANNEL_COUNT],
            };
            errors.push(variant, &teacher);
        }
        let value_metrics = errors.value_metrics();
        assert!((value_metrics.mae - 0.1).abs() < 1e-9);
        assert!(errors.velocity_abs < 1e-9);
    }

    #[test]
    fn blink_detection_finds_a_teacher_pulse() {
        // 30 frames at 33 ms: baseline 0.02, blink peak 0.9 at frame 10,
        // decayed back below half by frame 14.
        let mut values = vec![0.02_f64; 30];
        for (offset, value) in [0.3_f64, 0.7, 0.9, 0.8, 0.5, 0.25, 0.1, 0.05]
            .iter()
            .enumerate()
        {
            // Bounds: offset < 8 < 30.
            #[allow(clippy::indexing_slicing)]
            {
                values[10 + offset] = *value;
            }
        }
        let times: Vec<u64> = (0..30).map(|frame| frame * 33_333).collect();
        let events = detect_events(ArkitBlendshape::EyeBlinkLeft, &times, &values, 30);
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            EventKind::Pulse { peak, .. } => assert!((*peak - 0.9).abs() < 1e-9),
            EventKind::Rise { .. } => panic!("expected a pulse"),
        }
    }

    #[test]
    fn rise_detection_finds_a_sustained_jaw_open() {
        // 30 frames at 33 ms: baseline 0.05, rise to 0.6 at frame 12 that
        // holds.
        let mut values = vec![0.05_f64; 30];
        for value in values.iter_mut().skip(12) {
            *value = 0.6;
        }
        let times: Vec<u64> = (0..30).map(|frame| frame * 33_333).collect();
        let events = detect_events(ArkitBlendshape::JawOpen, &times, &values, 30);
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            EventKind::Rise { target, .. } => assert!((*target - 0.6).abs() < 1e-9),
            EventKind::Pulse { .. } => panic!("expected a rise"),
        }
    }
}
