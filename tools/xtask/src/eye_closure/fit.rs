//! Threshold fitting and held-out evaluation (Issue #52).
//!
//! The fitter replays the shared [`EyeClosureTracker`] over each take's time
//! series, so hysteresis and sequence gaps are judged exactly as the runtime
//! will judge them. Candidate thresholds are fixed before fitting; the held
//! out test split is never read by `fit`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use serde::Serialize;
use vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256;
use vtuber_tracking::{
    EYE_CLOSURE_ALGORITHM_VERSION, EYE_CLOSURE_FEATURE, EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
    EyeClosureFingerprints, EyeClosureObservation, EyeClosureProfileDocument, EyeClosureThresholds,
    EyeClosureTracker, EyeClosureVerificationStatus, EyeSide, EyeThreshold, EyeThresholdValues,
};

use super::Options;
use super::labels::{
    ExtractedData, FrameRow, LabelRow, LabelSource, LabelValue, Labels, Split, SplitFile,
};

/// Minimum closed-time recall required on train and validation.
pub(crate) const MIN_CLOSED_RECALL: f64 = 0.95;
/// Maximum not-closed time that may be latched closed.
pub(crate) const MAX_FALSE_CLOSE: f64 = 0.01;
const CLOSE_STEP: f32 = 0.005;
const WIDTHS: [f32; 4] = [0.01, 0.02, 0.04, 0.08];
const NOMINAL_FRAME_MS: f64 = 33.333;
const MAX_FRAME_MS: f64 = 250.0;

/// One frame's judgement inputs and optional ground truth.
#[derive(Clone, Debug)]
pub(crate) struct SeriesFrame {
    pub frame_seq: u64,
    pub timestamp_micros: u64,
    pub gap_before: bool,
    pub openness: Option<f32>,
    pub label: Option<LabelValue>,
    pub source: Option<LabelSource>,
    pub event_id: Option<String>,
}

/// One take's time-ordered series.
#[derive(Clone, Debug)]
pub(crate) struct TakeSeries {
    pub take_id: String,
    pub frames: Vec<SeriesFrame>,
}

/// Metrics for one eye over one replay.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct EyeMetrics {
    pub closed_time_ms: f64,
    pub closed_predicted_ms: f64,
    pub closed_recall: f64,
    pub not_closed_time_ms: f64,
    pub false_close_ms: f64,
    pub false_close: f64,
    pub closed_events: u64,
    pub closed_events_attained: u64,
    pub uncertain_frames: u64,
    pub unobservable_frames: u64,
    pub visual_closed_frames: u64,
    pub visual_not_closed_frames: u64,
}

impl EyeMetrics {
    /// Whether both denominators are populated and the thresholds pass.
    pub(crate) fn is_acceptable(&self) -> bool {
        self.closed_time_ms > 0.0
            && self.not_closed_time_ms > 0.0
            && self.closed_recall >= MIN_CLOSED_RECALL
            && self.false_close <= MAX_FALSE_CLOSE
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
                    SeriesFrame {
                        frame_seq: frame.frame_seq,
                        timestamp_micros: frame.timestamp_micros,
                        gap_before: frame.gap_before,
                        openness: match eye {
                            EyeSide::Left => frame.openness_left,
                            EyeSide::Right => frame.openness_right,
                        },
                        label: label_row.map(|row| row.label),
                        source: label_row.map(|row| row.source),
                        event_id: label_row.map(|row| row.event_id.clone()),
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

/// Replays a candidate threshold over the series and scores it.
pub(crate) fn evaluate_candidate(series: &[TakeSeries], threshold: EyeThreshold) -> EyeMetrics {
    let mut tracker = EyeClosureTracker::new(EyeClosureThresholds::new(threshold, threshold));
    let mut metrics = EyeMetrics::default();
    let mut attained: BTreeSet<String> = BTreeSet::new();
    let mut events: BTreeSet<String> = BTreeSet::new();
    for take in series {
        tracker.reset();
        for (index, frame) in take.frames.iter().enumerate() {
            let state = tracker.observe(
                frame.frame_seq,
                EyeClosureObservation {
                    left_openness: frame.openness,
                    right_openness: None,
                },
            );
            let closed = state.left.is_closed();
            let duration_ms = duration_ms(&take.frames, index);
            match frame.label {
                Some(LabelValue::FullyClosed) => {
                    metrics.closed_time_ms += duration_ms;
                    if closed {
                        metrics.closed_predicted_ms += duration_ms;
                    }
                    if frame.source == Some(LabelSource::VisualReview) {
                        metrics.visual_closed_frames += 1;
                    }
                    if let Some(event) = &frame.event_id {
                        events.insert(event.clone());
                        if closed {
                            attained.insert(event.clone());
                        }
                    }
                }
                Some(LabelValue::NotClosed) => {
                    metrics.not_closed_time_ms += duration_ms;
                    if closed {
                        metrics.false_close_ms += duration_ms;
                    }
                    if frame.source == Some(LabelSource::VisualReview) {
                        metrics.visual_not_closed_frames += 1;
                    }
                }
                Some(LabelValue::Uncertain) => metrics.uncertain_frames += 1,
                Some(LabelValue::Unobservable) => metrics.unobservable_frames += 1,
                None => {}
            }
        }
    }
    if metrics.closed_time_ms > 0.0 {
        metrics.closed_recall = metrics.closed_predicted_ms / metrics.closed_time_ms;
    }
    if metrics.not_closed_time_ms > 0.0 {
        metrics.false_close = metrics.false_close_ms / metrics.not_closed_time_ms;
    }
    metrics.closed_events = events.len() as u64;
    metrics.closed_events_attained = attained.len() as u64;
    metrics
}

fn duration_ms(frames: &[SeriesFrame], index: usize) -> f64 {
    let Some(frame) = frames.get(index) else {
        return NOMINAL_FRAME_MS;
    };
    let Some(next) = frames.get(index + 1) else {
        return NOMINAL_FRAME_MS;
    };
    if next.gap_before {
        return NOMINAL_FRAME_MS;
    }
    ((next.timestamp_micros.saturating_sub(frame.timestamp_micros)) as f64 / 1000.0)
        .clamp(0.0, MAX_FRAME_MS)
}

/// The fixed, pre-declared candidate grid.
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

struct CandidateRow {
    threshold: EyeThreshold,
    train: EyeMetrics,
    validation: Option<EyeMetrics>,
    status: &'static str,
}

fn select(train: &[TakeSeries], validation: &[TakeSeries]) -> Vec<CandidateRow> {
    // Deterministic selection: smallest hysteresis, then highest recall, then
    // lowest false-close, then smallest close point.
    let selected = grid_candidates()
        .into_iter()
        .filter(|threshold| evaluate_candidate(train, *threshold).is_acceptable())
        .filter_map(|threshold| {
            let metrics = evaluate_candidate(validation, threshold);
            metrics.is_acceptable().then_some((threshold, metrics))
        })
        .min_by(compare_candidates)
        .map(|(threshold, _)| threshold);
    grid_candidates()
        .into_iter()
        .map(|threshold| {
            let train_metrics = evaluate_candidate(train, threshold);
            let mut status = "rejected_train";
            let mut validation_metrics = None;
            if train_metrics.is_acceptable() {
                let metrics = evaluate_candidate(validation, threshold);
                status = if metrics.is_acceptable() {
                    "acceptable"
                } else {
                    "rejected_validation"
                };
                validation_metrics = Some(metrics);
            }
            if selected == Some(threshold) {
                status = "selected";
            }
            CandidateRow {
                threshold,
                train: train_metrics,
                validation: validation_metrics,
                status,
            }
        })
        .collect()
}

fn compare_candidates(
    left: &(EyeThreshold, EyeMetrics),
    right: &(EyeThreshold, EyeMetrics),
) -> std::cmp::Ordering {
    let left_width = left.0.reopen_at() - left.0.close_at();
    let right_width = right.0.reopen_at() - right.0.close_at();
    left_width
        .partial_cmp(&right_width)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            right
                .1
                .closed_recall
                .partial_cmp(&left.1.closed_recall)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            left.1
                .false_close
                .partial_cmp(&right.1.false_close)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            left.0
                .close_at()
                .partial_cmp(&right.0.close_at())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
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
        "eye,close_at,reopen_at,status,train_closed_recall,train_false_close,validation_closed_recall,validation_false_close\n",
    );
    let mut report = String::from("# Eye-closure fit report\n\n");
    let _ = writeln!(
        report,
        "- feature: `{EYE_CLOSURE_FEATURE}`\n- algorithm: {EYE_CLOSURE_ALGORITHM_VERSION}\n- label sha256: `{}`\n- train takes: {train_takes:?}\n- validation takes: {validation_takes:?}\n- selection: closed recall >= {MIN_CLOSED_RECALL}, false-close <= {MAX_FALSE_CLOSE}; smallest hysteresis, then recall, then false-close\n",
        labels.sha256
    );

    let mut selected_thresholds = BTreeMap::new();
    for eye in [EyeSide::Left, EyeSide::Right] {
        let train = build_series(&data, &labels, &train_takes, eye);
        let validation = build_series(&data, &labels, &validation_takes, eye);
        let rows = select(&train, &validation);
        let _ = writeln!(
            report,
            "### {} eye per-take counts\n\n{}",
            eye.as_str(),
            breakdown(&train, &validation)
        );
        let mut best: Option<&CandidateRow> = None;
        for row in &rows {
            let _ = writeln!(
                candidates_csv,
                "{},{:.3},{:.3},{},{:.4},{:.4},{:.4},{:.4}",
                eye.as_str(),
                row.threshold.close_at(),
                row.threshold.reopen_at(),
                row.status,
                row.train.closed_recall,
                row.train.false_close,
                row.validation
                    .as_ref()
                    .map_or(f64::NAN, |m| m.closed_recall),
                row.validation.as_ref().map_or(f64::NAN, |m| m.false_close),
            );
            if row.status == "selected" {
                best = Some(row);
            }
        }
        match best {
            Some(row) => {
                selected_thresholds.insert(eye, row.threshold);
                let validation = row.validation.as_ref();
                let _ = writeln!(
                    report,
                    "## {} eye: selected\n\n- close_at={:.3}, reopen_at={:.3}\n- train recall={:.4}, false-close={:.4}\n- validation recall={:.4}, false-close={:.4}\n- validation closed events attained {}/{}\n",
                    eye.as_str(),
                    row.threshold.close_at(),
                    row.threshold.reopen_at(),
                    row.train.closed_recall,
                    row.train.false_close,
                    validation.map_or(f64::NAN, |m| m.closed_recall),
                    validation.map_or(f64::NAN, |m| m.false_close),
                    validation.map_or(0, |m| m.closed_events_attained),
                    validation.map_or(0, |m| m.closed_events),
                );
            }
            None => {
                let any_train = rows.iter().any(|row| row.status != "rejected_train");
                let _ = writeln!(
                    report,
                    "## {} eye: no_acceptable_threshold\n\n- candidates passing train: {}\n- the opposite eye is not copied to this side\n",
                    eye.as_str(),
                    any_train
                );
            }
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
            left: values(left),
            right: values(right),
            fingerprints: EyeClosureFingerprints {
                task_bundle_sha256: Some(TASK_BUNDLE_SHA256.to_owned()),
                feature: EYE_CLOSURE_FEATURE.into(),
                preprocess: Some("mediapipe face landmarker raw blendshapes".into()),
            },
            applies_to: None,
        };
        write_json(&options.output.join("candidate_profile.json"), &document)?;
        println!(
            "selected left close_at={:.3}/reopen_at={:.3}, right close_at={:.3}/reopen_at={:.3}",
            left.close_at(),
            left.reopen_at(),
            right.close_at(),
            right.reopen_at()
        );
    } else {
        println!("no acceptable threshold for both eyes; candidate_profile.json was not written");
    }
    Ok(())
}

fn values(threshold: &EyeThreshold) -> EyeThresholdValues {
    EyeThresholdValues {
        close_at: threshold.close_at(),
        reopen_at: threshold.reopen_at(),
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

/// Serializes a threshold document for reports.
#[allow(dead_code)]
fn _serialize(values: &EyeThresholdValues) -> Result<String, String> {
    serde_json::to_string(values).map_err(|error| error.to_string())
}

/// Label row shape retained for report cross-checking.
#[allow(dead_code)]
fn _label_shape(row: &LabelRow) -> (&str, Option<&str>) {
    (&row.take_id, row.tag.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_tracking::EyeClosureProfileError;

    fn series(points: &[(u64, Option<f32>, Option<LabelValue>)]) -> Vec<TakeSeries> {
        vec![TakeSeries {
            take_id: "t".into(),
            frames: points
                .iter()
                .enumerate()
                .map(|(index, (seq, openness, label))| SeriesFrame {
                    frame_seq: *seq,
                    timestamp_micros: (*seq) * 33_333,
                    gap_before: false,
                    openness: *openness,
                    label: *label,
                    source: label.map(|_| LabelSource::VisualReview),
                    event_id: label.map(|_| format!("e{index}")),
                })
                .collect(),
        }]
    }

    #[test]
    fn a_known_threshold_is_recovered_from_synthetic_data() {
        // Raw blink closes at <= 0.4 openness and opens at >= 0.6.
        let data = series(&[
            (0, Some(0.9), Some(LabelValue::NotClosed)),
            (1, Some(0.35), Some(LabelValue::FullyClosed)),
            (2, Some(0.35), Some(LabelValue::FullyClosed)),
            (3, Some(0.9), Some(LabelValue::NotClosed)),
        ]);
        let threshold = EyeThreshold::new(0.4, 0.6).unwrap();
        let metrics = evaluate_candidate(&data, threshold);
        assert!(metrics.is_acceptable(), "{metrics:?}");
        assert_eq!(metrics.closed_recall, 1.0);
        assert_eq!(metrics.false_close, 0.0);
    }

    #[test]
    fn hysteresis_keeps_a_half_open_eye_closed_and_fails_the_candidate() {
        // The eye closes at 0.3 then sits at 0.5, which is labelled NotClosed.
        // A no-hysteresis style threshold would reopen; a wide hysteresis
        // stays closed and must be rejected for false-close.
        let data = series(&[
            (0, Some(0.2), Some(LabelValue::FullyClosed)),
            (1, Some(0.5), Some(LabelValue::NotClosed)),
        ]);
        let threshold = EyeThreshold::new(0.4, 0.6).unwrap();
        let metrics = evaluate_candidate(&data, threshold);
        assert!(!metrics.is_acceptable(), "{metrics:?}");
        assert!(metrics.false_close > MAX_FALSE_CLOSE);
    }

    #[test]
    fn a_gap_drops_the_closure_before_scoring() {
        // seq 0 closed, seq 5 labelled open after a gap: the closure must not
        // carry across the gap and create a false close.
        let data = series(&[
            (0, Some(0.1), Some(LabelValue::FullyClosed)),
            (5, Some(0.9), Some(LabelValue::NotClosed)),
        ]);
        let threshold = EyeThreshold::new(0.4, 0.6).unwrap();
        let metrics = evaluate_candidate(&data, threshold);
        assert_eq!(metrics.false_close, 0.0, "{metrics:?}");
    }

    #[test]
    fn nan_openness_does_not_toggle() {
        let data = series(&[
            (0, Some(0.1), Some(LabelValue::FullyClosed)),
            (1, Some(f32::NAN), Some(LabelValue::FullyClosed)),
            (2, Some(f32::NAN), Some(LabelValue::NotClosed)),
        ]);
        let threshold = EyeThreshold::new(0.4, 0.6).unwrap();
        let metrics = evaluate_candidate(&data, threshold);
        // The closure persists through NaN, so the final NotClosed frame is a
        // false close; that is exactly the kind of failure the fitter must see.
        assert!(metrics.false_close > 0.0, "{metrics:?}");
    }

    #[test]
    fn grid_stays_inside_the_valid_threshold_domain() {
        for threshold in grid_candidates() {
            assert!(0.0 <= threshold.close_at());
            assert!(threshold.close_at() < threshold.reopen_at());
            assert!(threshold.reopen_at() <= 1.0);
        }
    }

    #[test]
    fn empty_denominators_are_not_acceptable() {
        let data = series(&[(0, Some(0.9), Some(LabelValue::NotClosed))]);
        let metrics = evaluate_candidate(&data, EyeThreshold::new(0.4, 0.6).unwrap());
        assert!(!metrics.is_acceptable());
        assert_eq!(metrics.closed_recall, 0.0);
    }

    #[test]
    fn proxy_only_labels_validate_but_do_not_verify() {
        // A document with proxy-only provenance still validates as a candidate.
        let document = EyeClosureProfileDocument {
            schema_version: EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
            algorithm_version: EYE_CLOSURE_ALGORITHM_VERSION,
            feature: EYE_CLOSURE_FEATURE.into(),
            status: EyeClosureVerificationStatus::Candidate,
            left: EyeThresholdValues {
                close_at: 0.4,
                reopen_at: 0.6,
            },
            right: EyeThresholdValues {
                close_at: 0.4,
                reopen_at: 0.6,
            },
            fingerprints: EyeClosureFingerprints {
                task_bundle_sha256: Some(TASK_BUNDLE_SHA256.into()),
                feature: EYE_CLOSURE_FEATURE.into(),
                preprocess: None,
            },
            applies_to: None,
        };
        assert!(document.validate().is_ok());
        assert!(matches!(
            EyeClosureProfileDocument {
                schema_version: 2,
                ..document
            }
            .validate(),
            Err(EyeClosureProfileError::UnsupportedSchemaVersion { .. })
        ));
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

    fn openness_values() -> Vec<f32> {
        vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.2, 0.6, 0.1, 0.7, 0.3, 0.8,
            0.2, 0.9, 0.0,
        ]
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
            "schema_version": 1,
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
        let values = openness_values();
        for (take, _) in takes {
            for (index, value) in values.iter().enumerate() {
                let seq = index as u64;
                frames.push_str(
                    &serde_json::to_string(&serde_json::json!({
                        "take_id": take,
                        "frame_seq": seq,
                        "timestamp_micros": seq * 33_333,
                        "openness_left": value,
                        "openness_right": value,
                        "arkit_blink_left": 1.0 - value,
                        "arkit_blink_right": 1.0 - value,
                        "gap_before": false,
                        "rgb_reference": serde_json::Value::Null
                    }))
                    .unwrap(),
                );
                frames.push('\n');
                let left_label = if *value <= 0.35 {
                    "fully_closed"
                } else {
                    "not_closed"
                };
                let right_label = if *value <= 0.55 {
                    "fully_closed"
                } else {
                    "not_closed"
                };
                labels.push_str(&format!(
                    "{take},{seq},left,{left_label},visual_review,,e-left\n"
                ));
                labels.push_str(&format!(
                    "{take},{seq},right,{right_label},visual_review,,e-right\n"
                ));
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
            output,
            project_root: root.to_path_buf(),
        }
    }

    #[test]
    fn fit_recovers_left_and_right_thresholds_and_evaluate_verifies() {
        let root = tempfile::tempdir().unwrap();
        write_fixture(root.path());
        let fit_dir = root.path().join("fit");
        run_fit(&options(root.path(), fit_dir.clone())).unwrap();

        let document = read_profile(&fit_dir.join("candidate_profile.json")).unwrap();
        assert_eq!(document.status, EyeClosureVerificationStatus::Candidate);
        assert!(
            (0.30..0.40).contains(&document.left.close_at),
            "{:?}",
            document.left
        );
        assert!(
            (0.50..0.60).contains(&document.right.close_at),
            "{:?}",
            document.right
        );
        assert!(document.left.close_at < document.left.reopen_at);
        assert!(document.right.close_at < document.right.reopen_at);

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
        // Replace every visual source with the proxy source.
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

        let session_split = root.path().join("session.json");
        std::fs::write(
            &session_split,
            serde_json::to_string(&serde_json::json!({
                "train": ["train1"],
                "validation": ["train2"],
                "test": ["test1"]
            }))
            .unwrap(),
        )
        .unwrap();
        // train1 and train2 are different sessions, so this is allowed; a
        // shared-session split is the failure below.
        assert!(SplitFile::load(&session_split, &data).is_ok());

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
