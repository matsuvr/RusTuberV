//! Threshold fitting, time-series replay, and held-out evaluation metrics
//! (Issues #52/#64/#66).
//!
//! The fitter replays the shared trackers over each take's observed time
//! series, so sequence reuse, normal sequence jumps, capture-time gaps, and
//! missing observations are judged exactly as the runtime will judge them.
//! Candidate thresholds are fixed before fitting; the held-out test split is
//! never read by `fit`.
//!
//! Three metric groups are kept separate: reviewed-frame counts, continuous
//! labelled-interval time, and closure events. Sparse review frames are never
//! expanded into global time, and an interval with an unmeasured endpoint
//! contributes no duration.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use serde::Serialize;
use vtuber_core::{FrameSeq, MonoTimeNs};
use vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256;
use vtuber_tracking::{
    EYE_CLOSURE_ALGORITHM_VERSION, EYE_CLOSURE_FEATURE, EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
    EYE_CLOSURE_SAMPLE_GAP_NS, EyeClosureFeatures, EyeClosureFingerprints, EyeClosureObservation,
    EyeClosureProfileDocument, EyeClosureThresholds, EyeClosureTracker,
    EyeClosureVerificationStatus, EyeGeometryThreshold, EyeGeometryThresholdValues, EyeOpenness,
    EyeSide, EyeThreshold,
};

use super::Options;
use super::labels::{
    ExtractedData, FrameRow, LabelRow, LabelSource, LabelValue, Labels, Split, SplitFile,
};

/// Minimum closed-time recall required on train and validation.
pub(crate) const MIN_CLOSED_RECALL: f64 = 0.95;
/// Maximum not-closed time that may be latched closed.
pub(crate) const MAX_FALSE_CLOSE: f64 = 0.01;
/// Minimum fraction of reviewed closure events that must be reached.
pub(crate) const MIN_EVENT_ATTAINMENT: f64 = 0.95;
const CLOSE_STEP: f32 = 0.005;
const WIDTHS: [f32; 4] = [0.01, 0.02, 0.04, 0.08];
const QUANTILE_SEGMENTS: usize = 23;
const MIN_BLINK_STEPS: usize = 20;

/// One frame's judgement inputs and optional ground truth.
#[derive(Clone, Debug)]
pub(crate) struct SeriesFrame {
    pub frame_seq: u64,
    pub timestamp_micros: u64,
    pub openness: Option<f32>,
    pub raw_blink: Option<f32>,
    pub lid_gap: Option<f32>,
    pub label: Option<LabelValue>,
    pub source: Option<LabelSource>,
    pub event_id: Option<String>,
}

impl SeriesFrame {
    /// Whether this eye had an observation in this frame.
    fn observation_present(&self) -> bool {
        self.openness.is_some()
    }

    /// The geometry features for this eye, when both inputs exist.
    fn features(&self) -> Option<EyeClosureFeatures> {
        Some(EyeClosureFeatures {
            lid_gap_ratio: self.lid_gap?,
            raw_blink: self.raw_blink?,
        })
    }
}

/// One take's time-ordered series.
#[derive(Clone, Debug)]
pub(crate) struct TakeSeries {
    pub take_id: String,
    pub frames: Vec<SeriesFrame>,
}

/// One replayed judgement, keyed by take and frame.
#[derive(Clone, Debug)]
pub(crate) struct TakePrediction {
    pub take_id: String,
    pub frame_seq: u64,
    /// Capture time carried with the prediction record (the interval metric
    /// reads the same capture times from the series).
    #[allow(dead_code)]
    pub timestamp_micros: u64,
    pub state: EyeOpenness,
}

/// Metrics on the visually reviewed frames only.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ReviewedFrameMetrics {
    pub frames: u64,
    pub closed_frames: u64,
    pub not_closed_frames: u64,
    pub uncertain_frames: u64,
    pub unobservable_frames: u64,
    pub unknown_predictions: u64,
    pub closed_attained_frames: u64,
    pub false_close_frames: u64,
    pub closed_recall: Option<f64>,
    pub false_close: Option<f64>,
    pub observability: Option<f64>,
}

/// Event-attainment metrics keyed by `(take_id, event_id)` within one eye.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ReviewedEventMetrics {
    pub closed_events: u64,
    pub attained_events: u64,
    pub event_attainment: Option<f64>,
}

/// Continuous labelled-interval time metrics.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct LabeledIntervalMetrics {
    pub closed_time_ms: f64,
    pub closed_predicted_ms: f64,
    pub not_closed_time_ms: f64,
    pub false_close_ms: f64,
    pub closed_recall: Option<f64>,
    pub false_close: Option<f64>,
}

/// All metric groups for one eye over one replay.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct EyeMetrics {
    pub reviewed: ReviewedFrameMetrics,
    pub events: ReviewedEventMetrics,
    pub intervals: LabeledIntervalMetrics,
}

impl EyeMetrics {
    /// Whether every required group is measured and passes the targets.
    ///
    /// An unmeasured group is a failure, not a pass: sparse labels never
    /// fabricate a denominator.
    pub(crate) fn is_acceptable(&self) -> bool {
        self.reviewed
            .closed_recall
            .is_some_and(|value| value >= MIN_CLOSED_RECALL)
            && self
                .reviewed
                .false_close
                .is_some_and(|value| value <= MAX_FALSE_CLOSE)
            && self
                .events
                .event_attainment
                .is_some_and(|value| value >= MIN_EVENT_ATTAINMENT)
            && self
                .intervals
                .closed_recall
                .is_some_and(|value| value >= MIN_CLOSED_RECALL)
            && self
                .intervals
                .false_close
                .is_some_and(|value| value <= MAX_FALSE_CLOSE)
    }
}

/// Builds one eye's series for the selected takes.
pub(crate) fn build_series(
    data: &ExtractedData,
    labels: &Labels,
    takes: &[String],
    eye: EyeSide,
) -> Vec<TakeSeries> {
    let wanted: BTreeSet<&str> = takes.iter().map(String::as_str).collect();
    let mut grouped: BTreeMap<&str, Vec<&FrameRow>> = BTreeMap::new();
    for frame in &data.frames {
        if wanted.contains(frame.take_id.as_str()) {
            grouped
                .entry(frame.take_id.as_str())
                .or_default()
                .push(frame);
        }
    }
    grouped
        .into_iter()
        .map(|(take_id, mut frames)| {
            frames.sort_by_key(|frame| frame.frame_seq);
            let points = frames
                .into_iter()
                .map(|frame| {
                    let label_row = labels.get(&frame.take_id, frame.frame_seq, eye);
                    let (openness, raw_blink, lid_gap) = match eye {
                        EyeSide::Left => {
                            (frame.openness_left, frame.mp_blink_left, frame.lid_gap_left)
                        }
                        EyeSide::Right => (
                            frame.openness_right,
                            frame.mp_blink_right,
                            frame.lid_gap_right,
                        ),
                    };
                    SeriesFrame {
                        frame_seq: frame.frame_seq,
                        timestamp_micros: frame.timestamp_micros,
                        openness,
                        raw_blink: raw_blink.or(openness.map(|value| 1.0 - value)),
                        lid_gap,
                        label: label_row.map(|row| row.label),
                        source: label_row.map(|row| row.source),
                        event_id: label_row
                            .map(|row| row.event_id.clone())
                            .filter(|id| !id.is_empty()),
                    }
                })
                .collect();
            TakeSeries {
                take_id: take_id.to_owned(),
                frames: points,
            }
        })
        .collect()
}

fn mono_time(timestamp_micros: u64) -> MonoTimeNs {
    MonoTimeNs(timestamp_micros.saturating_mul(1000))
}

/// Replays one v1 raw-blink candidate over the series.
pub(crate) fn replay_blink_candidate(
    series: &[TakeSeries],
    threshold: EyeThreshold,
) -> Vec<TakePrediction> {
    let mut tracker = EyeClosureTracker::new(EyeClosureThresholds::new(threshold, threshold));
    let mut predictions = Vec::new();
    for take in series {
        tracker.reset();
        for frame in &take.frames {
            let state = tracker.observe(
                FrameSeq(frame.frame_seq),
                mono_time(frame.timestamp_micros),
                EyeClosureObservation {
                    left_openness: frame.openness,
                    right_openness: frame.openness,
                },
            );
            predictions.push(TakePrediction {
                take_id: take.take_id.clone(),
                frame_seq: frame.frame_seq,
                timestamp_micros: frame.timestamp_micros,
                state: state.left,
            });
        }
    }
    predictions
}

/// Replays one geometry candidate over the series.
pub(crate) fn replay_geometry_candidate(
    series: &[TakeSeries],
    threshold: EyeGeometryThreshold,
) -> Vec<TakePrediction> {
    let thresholds = vtuber_tracking::EyeGeometryThresholds::new(threshold, threshold);
    let mut tracker = vtuber_tracking::GeometryEyeClosureTracker::new(thresholds);
    let mut predictions = Vec::new();
    for take in series {
        tracker.reset();
        for frame in &take.frames {
            let state = tracker.observe(
                FrameSeq(frame.frame_seq),
                mono_time(frame.timestamp_micros),
                [frame.features(), frame.features()],
            );
            predictions.push(TakePrediction {
                take_id: take.take_id.clone(),
                frame_seq: frame.frame_seq,
                timestamp_micros: frame.timestamp_micros,
                state: state.left,
            });
        }
    }
    predictions
}

fn prediction_map(predictions: &[TakePrediction]) -> BTreeMap<(&str, u64), EyeOpenness> {
    predictions
        .iter()
        .map(|prediction| {
            (
                (prediction.take_id.as_str(), prediction.frame_seq),
                prediction.state,
            )
        })
        .collect()
}

/// Scores visually reviewed frames only.
pub(crate) fn reviewed_frame_metrics(
    series: &[TakeSeries],
    predictions: &[TakePrediction],
) -> ReviewedFrameMetrics {
    let predictions = prediction_map(predictions);
    let mut metrics = ReviewedFrameMetrics::default();
    for take in series {
        for frame in &take.frames {
            if frame.source != Some(LabelSource::VisualReview) {
                continue;
            }
            let Some(label) = frame.label else {
                continue;
            };
            match label {
                LabelValue::Uncertain => {
                    metrics.uncertain_frames += 1;
                    continue;
                }
                LabelValue::Unobservable => {
                    metrics.unobservable_frames += 1;
                    continue;
                }
                LabelValue::FullyClosed | LabelValue::NotClosed => {}
            }
            let state = predictions
                .get(&(take.take_id.as_str(), frame.frame_seq))
                .copied()
                .unwrap_or(EyeOpenness::Unknown);
            metrics.frames += 1;
            if state == EyeOpenness::Unknown {
                metrics.unknown_predictions += 1;
            }
            match label {
                LabelValue::FullyClosed => {
                    metrics.closed_frames += 1;
                    if state == EyeOpenness::Closed {
                        metrics.closed_attained_frames += 1;
                    }
                }
                LabelValue::NotClosed => {
                    metrics.not_closed_frames += 1;
                    if state == EyeOpenness::Closed {
                        metrics.false_close_frames += 1;
                    }
                }
                LabelValue::Uncertain | LabelValue::Unobservable => {}
            }
        }
    }
    if metrics.frames > 0 {
        metrics.observability =
            Some((metrics.frames - metrics.unknown_predictions) as f64 / metrics.frames as f64);
    }
    if metrics.closed_frames > 0 {
        metrics.closed_recall =
            Some(metrics.closed_attained_frames as f64 / metrics.closed_frames as f64);
    }
    if metrics.not_closed_frames > 0 {
        metrics.false_close =
            Some(metrics.false_close_frames as f64 / metrics.not_closed_frames as f64);
    }
    metrics
}

/// Duration of a labelled interval between two adjacent observed frames.
///
/// Both endpoints must carry the same definite visual label, be present in
/// the recording sequence, have an observation, and be capture-time adjacent.
/// Anything else is `None`; no nominal frame duration is substituted.
#[must_use]
pub(crate) fn labeled_interval_micros(current: &SeriesFrame, next: &SeriesFrame) -> Option<u64> {
    let current_label = current.label?;
    if current_label != next.label? {
        return None;
    }
    if !matches!(
        current_label,
        LabelValue::FullyClosed | LabelValue::NotClosed
    ) {
        return None;
    }
    if next.frame_seq != current.frame_seq.saturating_add(1) {
        return None;
    }
    if !current.observation_present() || !next.observation_present() {
        return None;
    }
    let delta = next
        .timestamp_micros
        .checked_sub(current.timestamp_micros)?;
    if delta == 0 || delta.saturating_mul(1000) >= EYE_CLOSURE_SAMPLE_GAP_NS {
        return None;
    }
    Some(delta)
}

/// Scores continuous visually labelled intervals only.
pub(crate) fn reviewed_interval_metrics(
    series: &[TakeSeries],
    predictions: &[TakePrediction],
) -> LabeledIntervalMetrics {
    let predictions = prediction_map(predictions);
    let mut metrics = LabeledIntervalMetrics::default();
    for take in series {
        for pair in take.frames.windows(2) {
            let [current, next] = pair else {
                continue;
            };
            if current.source != Some(LabelSource::VisualReview) {
                continue;
            }
            let Some(delta) = labeled_interval_micros(current, next) else {
                continue;
            };
            let current_state = predictions
                .get(&(take.take_id.as_str(), current.frame_seq))
                .copied()
                .unwrap_or(EyeOpenness::Unknown);
            let next_state = predictions
                .get(&(take.take_id.as_str(), next.frame_seq))
                .copied()
                .unwrap_or(EyeOpenness::Unknown);
            let both_closed =
                current_state == EyeOpenness::Closed && next_state == EyeOpenness::Closed;
            let milliseconds = delta as f64 / 1000.0;
            match current.label {
                Some(LabelValue::FullyClosed) => {
                    metrics.closed_time_ms += milliseconds;
                    if both_closed {
                        metrics.closed_predicted_ms += milliseconds;
                    }
                }
                Some(LabelValue::NotClosed) => {
                    metrics.not_closed_time_ms += milliseconds;
                    if both_closed {
                        metrics.false_close_ms += milliseconds;
                    }
                }
                Some(LabelValue::Uncertain | LabelValue::Unobservable) | None => {}
            }
        }
    }
    if metrics.closed_time_ms > 0.0 {
        metrics.closed_recall = Some(metrics.closed_predicted_ms / metrics.closed_time_ms);
    }
    if metrics.not_closed_time_ms > 0.0 {
        metrics.false_close = Some(metrics.false_close_ms / metrics.not_closed_time_ms);
    }
    metrics
}

/// Scores reviewed closure events keyed by `(take_id, event_id)`.
///
/// The caller passes one eye's series, so the effective key is
/// `(take_id, eye, event_id)`. The same event id in another take is a
/// different event and is never merged.
pub(crate) fn reviewed_event_metrics(
    series: &[TakeSeries],
    predictions: &[TakePrediction],
) -> ReviewedEventMetrics {
    let predictions = prediction_map(predictions);
    let mut events: BTreeMap<(&str, &str), bool> = BTreeMap::new();
    for take in series {
        for frame in &take.frames {
            if frame.source != Some(LabelSource::VisualReview)
                || frame.label != Some(LabelValue::FullyClosed)
            {
                continue;
            }
            let Some(event_id) = frame.event_id.as_deref() else {
                continue;
            };
            let state = predictions
                .get(&(take.take_id.as_str(), frame.frame_seq))
                .copied()
                .unwrap_or(EyeOpenness::Unknown);
            let attained = events
                .entry((take.take_id.as_str(), event_id))
                .or_insert(false);
            if state == EyeOpenness::Closed {
                *attained = true;
            }
        }
    }
    let mut metrics = ReviewedEventMetrics {
        closed_events: events.len() as u64,
        attained_events: events.values().filter(|attained| **attained).count() as u64,
        event_attainment: None,
    };
    if metrics.closed_events > 0 {
        metrics.event_attainment =
            Some(metrics.attained_events as f64 / metrics.closed_events as f64);
    }
    metrics
}

/// Scores one v1 raw-blink candidate with all metric groups.
pub(crate) fn evaluate_blink_candidate(
    series: &[TakeSeries],
    threshold: EyeThreshold,
) -> EyeMetrics {
    let predictions = replay_blink_candidate(series, threshold);
    EyeMetrics {
        reviewed: reviewed_frame_metrics(series, &predictions),
        events: reviewed_event_metrics(series, &predictions),
        intervals: reviewed_interval_metrics(series, &predictions),
    }
}

/// Scores one geometry candidate with all metric groups.
pub(crate) fn evaluate_geometry_candidate(
    series: &[TakeSeries],
    threshold: EyeGeometryThreshold,
) -> EyeMetrics {
    let predictions = replay_geometry_candidate(series, threshold);
    EyeMetrics {
        reviewed: reviewed_frame_metrics(series, &predictions),
        events: reviewed_event_metrics(series, &predictions),
        intervals: reviewed_interval_metrics(series, &predictions),
    }
}

/// The fixed, pre-declared v1 candidate grid.
pub(crate) fn grid_candidates() -> Vec<EyeThreshold> {
    let mut candidates = Vec::new();
    let mut close_at = 0.0_f32;
    while close_at <= 1.0 {
        for width in WIDTHS {
            let reopen_at = close_at + width;
            if reopen_at <= 1.0
                && let Ok(threshold) = EyeThreshold::new(close_at, reopen_at)
            {
                candidates.push(threshold);
            }
        }
        close_at += CLOSE_STEP;
    }
    candidates
}

fn quantile(sorted: &[f32], q: f64) -> Option<f32> {
    let last = sorted.len().checked_sub(1)?;
    let position = q * last as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let lower_value = *sorted.get(lower)?;
    let upper_value = *sorted.get(upper)?;
    let weight = (position - lower as f64) as f32;
    Some(lower_value + (upper_value - lower_value) * weight)
}

fn label_quantiles(values: &mut [f32]) -> Vec<f32> {
    values.sort_by(f32::total_cmp);
    let mut quantiles = Vec::with_capacity(QUANTILE_SEGMENTS + 2);
    quantiles.push(0.0_f32);
    for segment in 0..=QUANTILE_SEGMENTS {
        let q = segment as f64 / QUANTILE_SEGMENTS as f64;
        if let Some(value) = quantile(values, q) {
            quantiles.push(value);
        }
    }
    quantiles.sort_by(f32::total_cmp);
    quantiles.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-6);
    quantiles
}

/// Candidate gap pairs from the train split's definite visual labels.
///
/// Uses the 24 quantiles of the closed and not-closed lid-gap distributions
/// plus `0`, and every `close < reopen` pair. Returns an empty vector when
/// either class has no labelled samples; it never fabricates one.
pub(crate) fn geometry_gap_pairs(train: &[TakeSeries]) -> Vec<(f32, f32)> {
    let mut closed = Vec::new();
    let mut not_closed = Vec::new();
    for take in train {
        for frame in &take.frames {
            if frame.source != Some(LabelSource::VisualReview) {
                continue;
            }
            let Some(gap) = frame.lid_gap else {
                continue;
            };
            match frame.label {
                Some(LabelValue::FullyClosed) => closed.push(gap),
                Some(LabelValue::NotClosed) => not_closed.push(gap),
                _ => {}
            }
        }
    }
    if closed.is_empty() || not_closed.is_empty() {
        return Vec::new();
    }
    let mut values = label_quantiles(&mut closed);
    values.extend(label_quantiles(&mut not_closed));
    values.sort_by(f32::total_cmp);
    values.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-6);
    let mut pairs = Vec::new();
    for (index, close) in values.iter().enumerate() {
        for reopen in values.iter().skip(index + 1) {
            if close < reopen {
                pairs.push((*close, *reopen));
            }
        }
    }
    pairs
}

/// The full geometry candidate set: gap pairs times `min_blink` steps.
pub(crate) fn geometry_candidates(train: &[TakeSeries]) -> Vec<EyeGeometryThreshold> {
    let pairs = geometry_gap_pairs(train);
    let mut candidates = Vec::new();
    for (close_gap, reopen_gap) in pairs {
        for step in 0..=MIN_BLINK_STEPS {
            let min_blink = step as f32 / MIN_BLINK_STEPS as f32;
            if let Ok(threshold) = EyeGeometryThreshold::new(close_gap, reopen_gap, min_blink) {
                candidates.push(threshold);
            }
        }
    }
    candidates
}

/// One geometry candidate's train and validation metrics.
#[derive(Clone)]
pub(crate) struct GeometryCandidateMetrics {
    pub threshold: EyeGeometryThreshold,
    pub train: EyeMetrics,
    pub validation: EyeMetrics,
}

fn compare_geometry_candidates(
    left: &GeometryCandidateMetrics,
    right: &GeometryCandidateMetrics,
) -> Ordering {
    let left_validation = &left.validation;
    let right_validation = &right.validation;
    right_validation
        .events
        .event_attainment
        .map_or(0.0, |value| value)
        .total_cmp(
            &left_validation
                .events
                .event_attainment
                .map_or(0.0, |value| value),
        )
        .then_with(|| {
            right_validation
                .reviewed
                .closed_recall
                .map_or(0.0, |value| value)
                .total_cmp(
                    &left_validation
                        .reviewed
                        .closed_recall
                        .map_or(0.0, |value| value),
                )
        })
        .then_with(|| {
            left_validation
                .reviewed
                .false_close
                .map_or(1.0, |value| value)
                .total_cmp(
                    &right_validation
                        .reviewed
                        .false_close
                        .map_or(1.0, |value| value),
                )
        })
        .then_with(|| left.threshold.width().total_cmp(&right.threshold.width()))
        .then_with(|| {
            left.threshold
                .min_blink()
                .total_cmp(&right.threshold.min_blink())
        })
        .then_with(|| {
            left.threshold
                .close_gap()
                .total_cmp(&right.threshold.close_gap())
        })
}

/// Selects one geometry threshold: G (`min_blink == 0`) before H.
pub(crate) fn select_geometry_threshold(
    candidates: &[GeometryCandidateMetrics],
) -> Option<EyeGeometryThreshold> {
    let acceptable = |candidate: &GeometryCandidateMetrics| {
        candidate.train.is_acceptable() && candidate.validation.is_acceptable()
    };
    let best = |min_blink_is_zero: bool| {
        candidates
            .iter()
            .filter(|candidate| {
                acceptable(candidate)
                    && (candidate.threshold.min_blink() == 0.0) == min_blink_is_zero
            })
            .min_by(|left, right| compare_geometry_candidates(left, right))
            .map(|candidate| candidate.threshold)
    };
    best(true).or_else(|| best(false))
}

struct CandidateRow {
    eye: EyeSide,
    class: &'static str,
    close: f32,
    reopen: f32,
    min_blink: Option<f32>,
    status: &'static str,
    train: EyeMetrics,
    validation: Option<EyeMetrics>,
}

impl CandidateRow {
    fn csv(&self) -> String {
        format!(
            "{},{},{:.4},{:.4},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.eye.as_str(),
            self.class,
            self.close,
            self.reopen,
            self.min_blink
                .map_or_else(String::new, |value| format!("{value:.2}")),
            self.status,
            opt_metric(self.train.reviewed.closed_recall),
            opt_metric(self.train.reviewed.false_close),
            opt_metric(self.train.events.event_attainment),
            opt_metric(self.train.intervals.closed_recall),
            opt_metric(self.train.intervals.false_close),
            opt_metric(
                self.validation
                    .as_ref()
                    .and_then(|metrics| metrics.reviewed.closed_recall)
            ),
            opt_metric(
                self.validation
                    .as_ref()
                    .and_then(|metrics| metrics.reviewed.false_close)
            ),
            opt_metric(
                self.validation
                    .as_ref()
                    .and_then(|metrics| metrics.events.event_attainment)
            ),
            opt_metric(
                self.validation
                    .as_ref()
                    .and_then(|metrics| metrics.intervals.closed_recall)
            ),
            opt_metric(
                self.validation
                    .as_ref()
                    .and_then(|metrics| metrics.intervals.false_close)
            ),
        )
    }
}

fn opt_metric(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.4}"))
}

/// Runs `eye-closure fit`.
pub(crate) fn run_fit(options: &Options) -> Result<(), String> {
    let data_dir = options.data.as_deref().ok_or("missing --data")?;
    let labels_path = options.labels.as_deref().ok_or("missing --labels")?;
    let split_path = options.split.as_deref().ok_or("missing --split")?;
    let data = ExtractedData::load(data_dir)?;
    let labels = Labels::load(labels_path)?;
    let split = SplitFile::load(split_path, &data)?;
    for row in labels.iter() {
        if !data.takes.contains_key(&row.take_id) {
            return Err(format!(
                "{}: label references unknown take {}",
                labels_path.display(),
                row.take_id
            ));
        }
    }
    std::fs::create_dir_all(&options.output)
        .map_err(|error| format!("failed to create {}: {error}", options.output.display()))?;

    let train_takes = split.takes_for(Split::Train).to_vec();
    let validation_takes = split.takes_for(Split::Validation).to_vec();
    if train_takes.is_empty() || validation_takes.is_empty() {
        return Err(format!(
            "{}: fit requires non-empty train and validation take lists",
            split_path.display()
        ));
    }

    let mut candidates_csv = String::from(
        "eye,class,close,reopen,min_blink,status,train_frame_recall,train_frame_false_close,train_event_attainment,train_interval_recall,train_interval_false_close,validation_frame_recall,validation_frame_false_close,validation_event_attainment,validation_interval_recall,validation_interval_false_close\n",
    );
    let mut report = String::from("# Eye-closure fit report (algorithm v2)\n\n");
    let _ = writeln!(
        report,
        "- feature: `{EYE_CLOSURE_FEATURE}`\n- algorithm: {EYE_CLOSURE_ALGORITHM_VERSION}\n- label sha256: `{}`\n- train takes: {train_takes:?}\n- validation takes: {validation_takes:?}\n- targets: reviewed-frame recall >= {MIN_CLOSED_RECALL}, reviewed-frame false-close <= {MAX_FALSE_CLOSE}, event attainment >= {MIN_EVENT_ATTAINMENT}, interval recall >= {MIN_CLOSED_RECALL}, interval false-close <= {MAX_FALSE_CLOSE}\n- metric groups are separate: reviewed-frame counts, continuous labelled-interval time, and `(take_id, eye, event_id)` events. Unmeasured groups fail acceptance.\n",
        labels.sha256
    );

    let mut selected_thresholds: BTreeMap<EyeSide, EyeGeometryThreshold> = BTreeMap::new();
    for eye in [EyeSide::Left, EyeSide::Right] {
        let train = build_series(&data, &labels, &train_takes, eye);
        let validation = build_series(&data, &labels, &validation_takes, eye);
        let _ = writeln!(
            report,
            "\n### {} eye per-take counts\n\n{}",
            eye.as_str(),
            breakdown(&train, &validation)
        );
        report.push_str(&r_baseline_section(
            eye,
            &train,
            &validation,
            &mut candidates_csv,
        ));
        let geometry_status = geometry_section(
            eye,
            &train,
            &validation,
            &mut candidates_csv,
            &mut report,
            &mut selected_thresholds,
        );
        if let Some(status) = geometry_status {
            let _ = writeln!(report, "{status}");
        }
    }

    write(&options.output.join("candidates.csv"), &candidates_csv)?;
    write(&options.output.join("report.md"), &report)?;

    if let (Some(left), Some(right)) = (
        selected_thresholds.get(&EyeSide::Left),
        selected_thresholds.get(&EyeSide::Right),
    ) {
        let document = EyeClosureProfileDocument {
            schema_version: EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
            algorithm_version: EYE_CLOSURE_ALGORITHM_VERSION,
            feature: EYE_CLOSURE_FEATURE.into(),
            status: EyeClosureVerificationStatus::Candidate,
            left: values(*left),
            right: values(*right),
            fingerprints: EyeClosureFingerprints {
                task_bundle_sha256: Some(TASK_BUNDLE_SHA256.to_owned()),
                feature: EYE_CLOSURE_FEATURE.into(),
                preprocess: Some("mediapipe face landmarker landmarks and raw blendshapes".into()),
            },
            applies_to: None,
        };
        write_json(&options.output.join("candidate_profile.json"), &document)?;
        println!(
            "selected left close_gap={:.4}/reopen_gap={:.4}/min_blink={:.2}, right close_gap={:.4}/reopen_gap={:.4}/min_blink={:.2}",
            left.close_gap(),
            left.reopen_gap(),
            left.min_blink(),
            right.close_gap(),
            right.reopen_gap(),
            right.min_blink()
        );
    } else {
        println!(
            "no acceptable geometry threshold for both eyes; candidate_profile.json was not written"
        );
    }
    Ok(())
}

fn r_baseline_section(
    eye: EyeSide,
    train: &[TakeSeries],
    validation: &[TakeSeries],
    csv: &mut String,
) -> String {
    let mut section = format!(
        "\n### {} eye R baseline (raw blink openness)\n\n",
        eye.as_str()
    );
    let mut rows = Vec::new();
    for threshold in grid_candidates() {
        let train_metrics = evaluate_blink_candidate(train, threshold);
        let validation_metrics = evaluate_blink_candidate(validation, threshold);
        let status = if train_metrics.is_acceptable() && validation_metrics.is_acceptable() {
            "acceptable"
        } else if train_metrics.is_acceptable() {
            "rejected_validation"
        } else {
            "rejected_train"
        };
        rows.push(CandidateRow {
            eye,
            class: "R",
            close: threshold.close_at(),
            reopen: threshold.reopen_at(),
            min_blink: None,
            status,
            train: train_metrics,
            validation: Some(validation_metrics),
        });
    }
    let formatted: Vec<String> = rows.iter().map(CandidateRow::csv).collect();
    for line in formatted {
        let _ = writeln!(csv, "{line}");
    }
    let acceptable = rows.iter().filter(|row| row.status == "acceptable").count();
    let interval_measured = rows
        .iter()
        .filter(|row| {
            row.validation
                .as_ref()
                .is_some_and(|metrics| metrics.intervals.closed_recall.is_some())
        })
        .count();
    let _ = writeln!(
        section,
        "- candidates: {}, acceptable on train+validation: {acceptable}\n- candidates with measured validation interval recall: {interval_measured}",
        rows.len()
    );
    if let Some(best) = rows
        .iter()
        .min_by(|left, right| compare_candidate_rows(left, right))
    {
        let _ = writeln!(
            section,
            "- best R frontier (interval not used for ordering): close_at={:.3}, reopen_at={:.3}",
            best.close, best.reopen
        );
        section.push_str(&describe_validation(&best.validation));
    }
    let low_false_close: Vec<&CandidateRow> = rows
        .iter()
        .filter(|row| {
            row.validation.as_ref().is_some_and(|metrics| {
                metrics
                    .reviewed
                    .false_close
                    .is_some_and(|value| value <= MAX_FALSE_CLOSE)
            })
        })
        .collect();
    if let Some(best) = low_false_close
        .iter()
        .min_by(|left, right| compare_candidate_rows(left, right))
    {
        let _ = writeln!(
            section,
            "- best R with validation frame false-close <= 1%: close_at={:.3}, reopen_at={:.3}",
            best.close, best.reopen
        );
        section.push_str(&describe_validation(&best.validation));
    } else {
        section.push_str(
            "- no R candidate has a measured validation frame false-close at or below 1%\n",
        );
    }
    section
}

fn compare_candidate_rows(left: &CandidateRow, right: &CandidateRow) -> Ordering {
    match (&left.validation, &right.validation) {
        (Some(left), Some(right)) => compare_eye_metrics(left, right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn compare_eye_metrics(left: &EyeMetrics, right: &EyeMetrics) -> Ordering {
    let ret = |metrics: &EyeMetrics| metrics.events.event_attainment.map_or(0.0, |value| value);
    let recall = |metrics: &EyeMetrics| metrics.reviewed.closed_recall.map_or(0.0, |value| value);
    let false_close =
        |metrics: &EyeMetrics| metrics.reviewed.false_close.map_or(1.0, |value| value);
    ret(right)
        .total_cmp(&ret(left))
        .then_with(|| recall(right).total_cmp(&recall(left)))
        .then_with(|| false_close(left).total_cmp(&false_close(right)))
}

fn describe_validation(validation: &Option<EyeMetrics>) -> String {
    let Some(metrics) = validation else {
        return "- validation metrics: unmeasured\n".to_owned();
    };
    format!(
        "  - validation frame recall={}, frame false-close={}, event attainment={}, interval recall={}, interval false-close={}\n",
        opt_metric(metrics.reviewed.closed_recall),
        opt_metric(metrics.reviewed.false_close),
        opt_metric(metrics.events.event_attainment),
        opt_metric(metrics.intervals.closed_recall),
        opt_metric(metrics.intervals.false_close),
    )
}

fn geometry_section(
    eye: EyeSide,
    train: &[TakeSeries],
    validation: &[TakeSeries],
    csv: &mut String,
    report: &mut String,
    selected_thresholds: &mut BTreeMap<EyeSide, EyeGeometryThreshold>,
) -> Option<String> {
    let candidates = geometry_candidates(train);
    if candidates.is_empty() {
        let _ = writeln!(
            report,
            "\n### {} eye geometry\n\n- no candidate: the train split lacks visually labelled fully_closed or not_closed lid-gap samples\n",
            eye.as_str()
        );
        return None;
    }
    let total = candidates.len();
    let evaluated: Vec<GeometryCandidateMetrics> = candidates
        .into_iter()
        .map(|threshold| GeometryCandidateMetrics {
            threshold,
            train: evaluate_geometry_candidate(train, threshold),
            validation: evaluate_geometry_candidate(validation, threshold),
        })
        .collect();
    for candidate in &evaluated {
        let status = if candidate.train.is_acceptable() && candidate.validation.is_acceptable() {
            "acceptable"
        } else if candidate.train.is_acceptable() {
            "rejected_validation"
        } else {
            "rejected_train"
        };
        let row = CandidateRow {
            eye,
            class: if candidate.threshold.min_blink() == 0.0 {
                "G"
            } else {
                "H"
            },
            close: candidate.threshold.close_gap(),
            reopen: candidate.threshold.reopen_gap(),
            min_blink: Some(candidate.threshold.min_blink()),
            status,
            train: candidate.train.clone(),
            validation: Some(candidate.validation.clone()),
        };
        let _ = writeln!(csv, "{}", row.csv());
    }
    let acceptable: Vec<GeometryCandidateMetrics> = evaluated
        .iter()
        .filter(|candidate| candidate.train.is_acceptable() && candidate.validation.is_acceptable())
        .cloned()
        .collect();
    let mut status = format!(
        "\n### {} eye geometry\n\n- candidates: {total}, acceptable on train+validation: {}\n",
        eye.as_str(),
        acceptable.len()
    );
    for (class, is_g) in [("G", true), ("H", false)] {
        let subset: Vec<&GeometryCandidateMetrics> = evaluated
            .iter()
            .filter(|candidate| (candidate.threshold.min_blink() == 0.0) == is_g)
            .collect();
        if let Some(best) = subset
            .iter()
            .min_by(|left, right| compare_eye_metrics(&left.validation, &right.validation))
        {
            let _ = writeln!(
                status,
                "- best {class} frontier (interval not used for ordering): close_gap={:.4}, reopen_gap={:.4}, min_blink={:.2}",
                best.threshold.close_gap(),
                best.threshold.reopen_gap(),
                best.threshold.min_blink()
            );
            status.push_str(&describe_validation(&Some(best.validation.clone())));
        }
        let low_false_close: Vec<&&GeometryCandidateMetrics> = subset
            .iter()
            .filter(|candidate| {
                candidate
                    .validation
                    .reviewed
                    .false_close
                    .is_some_and(|value| value <= MAX_FALSE_CLOSE)
            })
            .collect();
        if let Some(best) = low_false_close
            .iter()
            .min_by(|left, right| compare_eye_metrics(&left.validation, &right.validation))
        {
            let _ = writeln!(
                status,
                "- best {class} with validation frame false-close <= 1%: close_gap={:.4}, reopen_gap={:.4}, min_blink={:.2}",
                best.threshold.close_gap(),
                best.threshold.reopen_gap(),
                best.threshold.min_blink()
            );
            status.push_str(&describe_validation(&Some(best.validation.clone())));
        }
    }
    match select_geometry_threshold(&acceptable) {
        Some(threshold) => {
            selected_thresholds.insert(eye, threshold);
            let _ = writeln!(
                status,
                "- selected: close_gap={:.4}, reopen_gap={:.4}, min_blink={:.2}, class={}",
                threshold.close_gap(),
                threshold.reopen_gap(),
                threshold.min_blink(),
                if threshold.min_blink() == 0.0 {
                    "G"
                } else {
                    "H"
                }
            );
        }
        None => {
            status.push_str("- no geometry candidate satisfies every measured target\n");
        }
    }
    Some(status)
}

fn values(threshold: EyeGeometryThreshold) -> EyeGeometryThresholdValues {
    EyeGeometryThresholdValues {
        close_gap: threshold.close_gap(),
        reopen_gap: threshold.reopen_gap(),
        min_blink: threshold.min_blink(),
    }
}

fn breakdown(train: &[TakeSeries], validation: &[TakeSeries]) -> String {
    let mut out = String::new();
    for (split, series) in [("train", train), ("validation", validation)] {
        for take in series {
            let closed = take
                .frames
                .iter()
                .filter(|frame| frame.label == Some(LabelValue::FullyClosed))
                .count();
            let not_closed = take
                .frames
                .iter()
                .filter(|frame| frame.label == Some(LabelValue::NotClosed))
                .count();
            let _ = writeln!(
                out,
                "- {split} {}: frames={}, closed={closed}, not_closed={not_closed}",
                take.take_id,
                take.frames.len()
            );
        }
    }
    if out.is_empty() {
        out.push_str("- (no takes)\n");
    }
    out
}

/// Loads a profile document from JSON.
pub(crate) fn read_profile(path: &Path) -> Result<EyeClosureProfileDocument, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
    write(path, &text)
}

/// Label row shape retained for report cross-checking.
#[allow(dead_code)]
fn _label_shape(row: &LabelRow) -> (&str, Option<&str>) {
    (&row.take_id, row.tag.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(
        seq: u64,
        label: Option<LabelValue>,
        event: Option<&str>,
        lid_gap: impl Into<Option<f32>>,
        raw_blink: impl Into<Option<f32>>,
    ) -> SeriesFrame {
        let lid_gap = lid_gap.into();
        let raw_blink = raw_blink.into();
        SeriesFrame {
            frame_seq: seq,
            timestamp_micros: seq * 33_333,
            openness: raw_blink.map(|value| 1.0 - value),
            raw_blink,
            lid_gap,
            label,
            source: label.map(|_| LabelSource::VisualReview),
            event_id: event.map(str::to_owned),
        }
    }

    fn missing_observation(seq: u64, label: Option<LabelValue>) -> SeriesFrame {
        SeriesFrame {
            frame_seq: seq,
            timestamp_micros: seq * 33_333,
            openness: None,
            raw_blink: None,
            lid_gap: None,
            label,
            source: label.map(|_| LabelSource::VisualReview),
            event_id: None,
        }
    }

    fn take(take_id: &str, frames: Vec<SeriesFrame>) -> TakeSeries {
        TakeSeries {
            take_id: take_id.to_owned(),
            frames,
        }
    }

    fn separated_frames(count: u64) -> Vec<SeriesFrame> {
        (0..count)
            .map(|seq| {
                if seq < 8 {
                    frame(
                        seq,
                        Some(LabelValue::FullyClosed),
                        Some("run"),
                        0.05,
                        Some(0.9),
                    )
                } else {
                    frame(seq, Some(LabelValue::NotClosed), None, 0.9, Some(0.05))
                }
            })
            .collect()
    }

    #[test]
    fn geometry_candidates_need_both_label_classes() {
        let only_open = vec![take(
            "t",
            (0..3)
                .map(|seq| frame(seq, Some(LabelValue::NotClosed), None, 0.9, Some(0.1)))
                .collect(),
        )];
        assert!(geometry_gap_pairs(&only_open).is_empty());
        assert!(geometry_candidates(&only_open).is_empty());
    }

    #[test]
    fn geometry_candidates_recover_a_separating_threshold() {
        let data = vec![take("t", separated_frames(16))];
        let candidates = geometry_candidates(&data);
        assert!(!candidates.is_empty());
        let threshold = EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap();
        let metrics = evaluate_geometry_candidate(&data, threshold);
        assert!(metrics.is_acceptable(), "{metrics:?}");
        assert_eq!(metrics.reviewed.closed_recall, Some(1.0));
        assert_eq!(metrics.reviewed.false_close, Some(0.0));
        assert_eq!(metrics.events.event_attainment, Some(1.0));
        assert!(metrics.intervals.closed_recall.is_some());
    }

    #[test]
    fn seven_labelled_frames_with_one_event_id_are_one_event() {
        let frames: Vec<SeriesFrame> = (0..7)
            .map(|seq| {
                frame(
                    seq,
                    Some(LabelValue::FullyClosed),
                    Some("wink"),
                    0.05,
                    Some(0.9),
                )
            })
            .collect();
        let data = vec![take("t", frames)];
        let metrics =
            evaluate_geometry_candidate(&data, EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap());
        assert_eq!(metrics.events.closed_events, 1);
        assert_eq!(metrics.events.attained_events, 1);
        assert_eq!(metrics.events.event_attainment, Some(1.0));
    }

    #[test]
    fn the_same_event_id_in_two_takes_is_not_merged() {
        let left = take(
            "a",
            vec![frame(
                0,
                Some(LabelValue::FullyClosed),
                Some("e1"),
                0.05,
                Some(0.9),
            )],
        );
        let right = take(
            "b",
            vec![frame(
                0,
                Some(LabelValue::FullyClosed),
                Some("e1"),
                0.05,
                Some(0.9),
            )],
        );
        let data = vec![left, right];
        let metrics =
            evaluate_geometry_candidate(&data, EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap());
        assert_eq!(metrics.events.closed_events, 2);
        assert_eq!(metrics.events.attained_events, 2);
    }

    #[test]
    fn an_isolated_review_frame_adds_no_interval_time() {
        let data = vec![take(
            "t",
            vec![
                frame(
                    10,
                    Some(LabelValue::FullyClosed),
                    Some("e"),
                    0.05,
                    Some(0.9),
                ),
                frame(11, None, None, 0.05, Some(0.9)),
                frame(12, None, None, 0.05, Some(0.9)),
            ],
        )];
        let metrics =
            evaluate_geometry_candidate(&data, EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap());
        assert_eq!(metrics.intervals.closed_time_ms, 0.0);
        assert_eq!(metrics.intervals.closed_recall, None);
        assert_eq!(
            labeled_interval_micros(&data[0].frames[0], &data[0].frames[1]),
            None
        );
    }

    #[test]
    fn a_sequence_jump_with_contiguous_time_keeps_the_latch() {
        // A 0.5 openness is inside the hysteresis band: it stays closed
        // only when the 100 -> 105 seq jump preserves the carried latch.
        let frames = vec![
            frame(100, None, None, 0.1, 0.9),
            frame(105, None, None, 0.5, 0.5),
        ];
        let data = vec![take("t", frames)];
        let predictions = replay_blink_candidate(&data, EyeThreshold::new(0.4, 0.7).unwrap());
        assert!(predictions.iter().all(|p| p.state == EyeOpenness::Closed));
    }

    #[test]
    fn a_long_capture_gap_resets_even_with_contiguous_sequence() {
        let mut frames = vec![
            frame(0, None, None, 0.1, 0.9),
            frame(1, None, None, 0.5, 0.5),
        ];
        frames[1].timestamp_micros = 500_000;
        let data = vec![take("t", frames)];
        let predictions = replay_blink_candidate(&data, EyeThreshold::new(0.4, 0.7).unwrap());
        assert_eq!(predictions[0].state, EyeOpenness::Closed);
        assert_eq!(
            predictions[1].state,
            EyeOpenness::Open,
            "entry condition after reset"
        );
    }

    #[test]
    fn a_reused_sequence_never_advances_the_latch() {
        let frames = vec![
            frame(0, None, None, 0.1, Some(0.9)),
            frame(0, None, None, 0.9, Some(0.1)),
        ];
        let data = vec![take("t", frames)];
        let predictions = replay_blink_candidate(&data, EyeThreshold::new(0.4, 0.7).unwrap());
        assert!(predictions.iter().all(|p| p.state == EyeOpenness::Closed));
    }

    #[test]
    fn a_missing_observation_is_unknown_and_then_recovers() {
        let data = vec![take(
            "t",
            vec![
                frame(0, Some(LabelValue::FullyClosed), Some("e"), 0.05, 0.9),
                missing_observation(1, Some(LabelValue::NotClosed)),
                frame(2, Some(LabelValue::NotClosed), None, 0.8, 0.05),
            ],
        )];
        let predictions =
            replay_geometry_candidate(&data, EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap());
        assert_eq!(predictions[1].state, EyeOpenness::Unknown);
        assert_eq!(predictions[2].state, EyeOpenness::Open);
        let metrics =
            evaluate_geometry_candidate(&data, EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap());
        assert_eq!(metrics.reviewed.unknown_predictions, 1);
        assert_eq!(metrics.events.attained_events, 1);
    }

    #[test]
    fn selection_prefers_g_over_h_and_is_deterministic() {
        let train = vec![take("t", separated_frames(16))];
        let validation = vec![take("v", separated_frames(16))];
        let g = EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap();
        let h = EyeGeometryThreshold::new(0.2, 0.4, 0.5).unwrap();
        let candidates = vec![
            GeometryCandidateMetrics {
                threshold: h,
                train: evaluate_geometry_candidate(&train, h),
                validation: evaluate_geometry_candidate(&validation, h),
            },
            GeometryCandidateMetrics {
                threshold: g,
                train: evaluate_geometry_candidate(&train, g),
                validation: evaluate_geometry_candidate(&validation, g),
            },
        ];
        assert_eq!(select_geometry_threshold(&candidates), Some(g));
    }

    #[test]
    fn empty_metric_groups_are_not_acceptable() {
        let metrics = EyeMetrics::default();
        assert!(!metrics.is_acceptable());
    }
}

#[cfg(test)]
mod end_to_end {
    use super::*;
    use crate::eye_closure::evaluate;
    use crate::eye_closure::labels::{ExtractedData, Labels, SplitFile};
    use std::path::PathBuf;

    fn take_metadata(take_id: &str, session: &str) -> serde_json::Value {
        serde_json::json!({
            "take_id": take_id,
            "session_id": session,
            "raw_frames_root": serde_json::Value::Null,
            "pixel_rotation_degrees": 0,
            "mirrored": false
        })
    }

    fn write_fixture(root: &Path) {
        let extracted = root.join("extracted");
        std::fs::create_dir_all(&extracted).unwrap();
        let takes = [
            ("train1", "sess_train1"),
            ("train2", "sess_train2"),
            ("val1", "sess_val1"),
            ("test1", "sess_test1"),
        ];
        let metadata = serde_json::json!({
            "schema_version": 2,
            "feature": vtuber_tracking::EYE_CLOSURE_FEATURE,
            "inputs": takes.iter().map(|(take, session)| take_metadata(take, session)).collect::<Vec<_>>()
        });
        std::fs::write(
            extracted.join("extraction-metadata.json"),
            serde_json::to_string_pretty(&metadata).unwrap(),
        )
        .unwrap();

        let mut frames = String::new();
        let mut labels = String::from("take_id,frame_seq,eye,label,label_source,tag,event_id\n");
        for (take, _) in takes {
            for seq in 0..16_u64 {
                let closed = seq < 8;
                let lid_gap = if closed { 0.05 } else { 0.9 };
                let blink = if closed { 0.9 } else { 0.05 };
                frames.push_str(
                    &serde_json::to_string(&serde_json::json!({
                        "take_id": take,
                        "frame_seq": seq,
                        "timestamp_micros": seq * 33_333,
                        "openness_left": 1.0 - blink,
                        "openness_right": 1.0 - blink,
                        "mp_blink_left": blink,
                        "mp_blink_right": blink,
                        "lid_gap_left": lid_gap,
                        "lid_gap_right": lid_gap,
                        "arkit_blink_left": blink,
                        "arkit_blink_right": blink,
                        "gap_before": false,
                        "rgb_reference": serde_json::Value::Null
                    }))
                    .unwrap(),
                );
                frames.push('\n');
                for (eye, event) in [
                    ("left", format!("{take}:left:run")),
                    ("right", format!("{take}:right:run")),
                ] {
                    let (label, event_id) = if closed {
                        ("fully_closed", event)
                    } else {
                        ("not_closed", String::new())
                    };
                    labels.push_str(&format!(
                        "{take},{seq},{eye},{label},visual_review,,{event_id}\n"
                    ));
                }
            }
        }
        std::fs::write(extracted.join("eye-frames.jsonl"), frames).unwrap();
        std::fs::write(root.join("labels.csv"), labels).unwrap();
        std::fs::write(
            root.join("split.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "train": ["train1", "train2"],
                "validation": ["val1"],
                "test": ["test1"]
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn options(root: &Path, output: PathBuf) -> Options {
        Options {
            inputs: None,
            data: Some(root.join("extracted")),
            labels: Some(root.join("labels.csv")),
            split: Some(root.join("split.json")),
            profile: None,
            config_dir: None,
            output,
            project_root: root.to_path_buf(),
        }
    }

    #[test]
    fn fit_recovers_geometry_thresholds_and_evaluate_verifies() {
        let root = tempfile::tempdir().unwrap();
        write_fixture(root.path());
        let fit_dir = root.path().join("fit");
        run_fit(&options(root.path(), fit_dir.clone())).unwrap();

        let document = read_profile(&fit_dir.join("candidate_profile.json")).unwrap();
        assert_eq!(document.status, EyeClosureVerificationStatus::Candidate);
        assert!(document.left.close_gap < 0.9, "{:?}", document.left);
        assert!(document.left.close_gap < document.left.reopen_gap);
        assert!(document.right.close_gap < document.right.reopen_gap);

        let test_dir = root.path().join("test");
        let mut evaluate_options = options(root.path(), test_dir.clone());
        evaluate_options.profile = Some(fit_dir.join("candidate_profile.json"));
        evaluate::run(&evaluate_options).unwrap();
        let verified = read_profile(&test_dir.join("eye_closure_profile.json")).unwrap();
        assert_eq!(verified.status, EyeClosureVerificationStatus::Verified);
        assert_eq!(document.left, verified.left);
        assert_eq!(document.right, verified.right);
    }

    #[test]
    fn a_proxy_only_test_set_never_yields_a_verified_profile() {
        let root = tempfile::tempdir().unwrap();
        write_fixture(root.path());
        let fit_dir = root.path().join("fit");
        run_fit(&options(root.path(), fit_dir.clone())).unwrap();
        let labels = std::fs::read_to_string(root.path().join("labels.csv")).unwrap();
        std::fs::write(
            root.path().join("labels.csv"),
            labels.replace("visual_review", "arkit_proxy"),
        )
        .unwrap();
        let test_dir = root.path().join("test");
        let mut evaluate_options = options(root.path(), test_dir.clone());
        evaluate_options.profile = Some(fit_dir.join("candidate_profile.json"));
        evaluate::run(&evaluate_options).unwrap();
        assert!(!test_dir.join("eye_closure_profile.json").exists());
    }

    #[test]
    fn an_unknown_fingerprint_never_yields_a_verified_profile() {
        let root = tempfile::tempdir().unwrap();
        write_fixture(root.path());
        let fit_dir = root.path().join("fit");
        run_fit(&options(root.path(), fit_dir.clone())).unwrap();
        let mut document = read_profile(&fit_dir.join("candidate_profile.json")).unwrap();
        document.fingerprints.task_bundle_sha256 = Some("0000".into());
        let tampered = root.path().join("tampered.json");
        write_json(&tampered, &document).unwrap();
        let test_dir = root.path().join("test");
        let mut evaluate_options = options(root.path(), test_dir.clone());
        evaluate_options.profile = Some(tampered);
        evaluate::run(&evaluate_options).unwrap();
        assert!(!test_dir.join("eye_closure_profile.json").exists());
    }

    #[test]
    fn a_v1_document_is_rejected_as_a_candidate() {
        let root = tempfile::tempdir().unwrap();
        write_fixture(root.path());
        let legacy = root.path().join("legacy.json");
        std::fs::write(
            &legacy,
            serde_json::to_string(&serde_json::json!({
                "schema_version": 1,
                "algorithm_version": 1,
                "feature": "mediapipe_raw_eye_blink_openness_v1",
                "status": "candidate",
                "left": { "close_at": 0.4, "reopen_at": 0.6 },
                "right": { "close_at": 0.4, "reopen_at": 0.6 },
                "fingerprints": {
                    "task_bundle_sha256": vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256,
                    "feature": "mediapipe_raw_eye_blink_openness_v1",
                    "preprocess": null
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let document = read_profile(&legacy);
        assert!(document.is_err(), "a v1 document is not parseable as v2");
        let test_dir = root.path().join("test");
        let mut evaluate_options = options(root.path(), test_dir);
        evaluate_options.profile = Some(legacy);
        assert!(evaluate::run(&evaluate_options).is_err());
    }

    #[test]
    fn split_leakage_and_unknown_takes_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        write_fixture(root.path());
        let data = ExtractedData::load(&root.path().join("extracted")).unwrap();
        let leaked = root.path().join("leaked.json");
        std::fs::write(
            &leaked,
            serde_json::to_string(&serde_json::json!({
                "train": ["train1"],
                "validation": ["train1"],
                "test": ["test1"]
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(SplitFile::load(&leaked, &data).is_err());

        let unknown = root.path().join("unknown.json");
        std::fs::write(
            &unknown,
            serde_json::to_string(&serde_json::json!({
                "train": ["missing"],
                "validation": ["val1"],
                "test": ["test1"]
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(SplitFile::load(&unknown, &data).is_err());
    }

    #[test]
    fn label_loading_is_required_by_fit() {
        let root = tempfile::tempdir().unwrap();
        write_fixture(root.path());
        let labels_path = root.path().join("labels.csv");
        let labels = Labels::load(&labels_path).unwrap();
        assert!(labels.sha256.len() == 64);
    }
}
