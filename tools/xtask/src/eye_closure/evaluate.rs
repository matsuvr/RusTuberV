//! Held-out evaluation and validated-profile emission (Issues #52/#66).
//!
//! `evaluate` freezes one candidate, reads the test split exactly once, and
//! writes an installable profile only when both eyes pass every measured
//! metric group with visual labels and matching inference fingerprints. A
//! failing evaluation still writes a report but leaves no runtime profile
//! behind.

use std::fmt::Write as _;
use std::path::Path;

use vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256;
use vtuber_tracking::{
    EYE_CLOSURE_ALGORITHM_VERSION, EYE_CLOSURE_FEATURE, EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
    EyeClosureFingerprints, EyeClosureProfileDocument, EyeClosureVerificationStatus,
    EyeGeometryThresholdValues, EyeSide,
};

use super::Options;
use super::fit::{
    MAX_FALSE_CLOSE, MIN_CLOSED_RECALL, MIN_EVENT_ATTAINMENT, build_series,
    evaluate_geometry_candidate, read_profile,
};
use super::labels::{ExtractedData, Labels, Split, SplitFile};

/// Runs `eye-closure evaluate`.
pub(crate) fn run(options: &Options) -> Result<(), String> {
    let data_dir = options.data.as_deref().ok_or("missing --data")?;
    let labels_path = options.labels.as_deref().ok_or("missing --labels")?;
    let split_path = options.split.as_deref().ok_or("missing --split")?;
    let profile_path = options.profile.as_deref().ok_or("missing --profile")?;
    let data = ExtractedData::load(data_dir)?;
    let labels = Labels::load(labels_path)?;
    let split = SplitFile::load(split_path, &data)?;
    let document = read_profile(profile_path)?;
    if document.status != EyeClosureVerificationStatus::Candidate {
        return Err(format!(
            "{}: evaluate requires a candidate profile, found {:?}",
            profile_path.display(),
            document.status
        ));
    }
    let thresholds = document
        .validate()
        .map_err(|error| format!("{}: {error}", profile_path.display()))?;
    let test_takes = split.takes_for(Split::Test).to_vec();
    if test_takes.is_empty() {
        return Err(format!("{}: test split is empty", split_path.display()));
    }
    let fingerprint_ok = document.fingerprints.feature == EYE_CLOSURE_FEATURE
        && document
            .fingerprints
            .task_bundle_sha256
            .as_deref()
            .is_some_and(|hash| hash.eq_ignore_ascii_case(TASK_BUNDLE_SHA256));

    let mut report = String::from("# Eye-closure held-out evaluation (algorithm v2)\n\n");
    let _ = writeln!(
        report,
        "- feature: `{EYE_CLOSURE_FEATURE}`\n- algorithm: {EYE_CLOSURE_ALGORITHM_VERSION}\n- label sha256: `{}`\n- test takes: {test_takes:?}\n- fingerprint match: {fingerprint_ok}\n- criteria: reviewed-frame recall >= {MIN_CLOSED_RECALL}, reviewed-frame false-close <= {MAX_FALSE_CLOSE}, event attainment >= {MIN_EVENT_ATTAINMENT}, interval recall >= {MIN_CLOSED_RECALL}, interval false-close <= {MAX_FALSE_CLOSE}, visual labels required\n",
        labels.sha256
    );

    let mut verified: Vec<(EyeSide, vtuber_tracking::EyeGeometryThreshold)> = Vec::new();
    for eye in [EyeSide::Left, EyeSide::Right] {
        let series = build_series(&data, &labels, &test_takes, eye);
        let metrics = evaluate_geometry_candidate(&series, thresholds.for_side(eye));
        let visual_ok =
            metrics.reviewed.closed_frames > 0 && metrics.reviewed.not_closed_frames > 0;
        let eye_ok = metrics.is_acceptable() && visual_ok;
        let _ = writeln!(
            report,
            "## {} eye\n\n- close_gap={:.4}, reopen_gap={:.4}, min_blink={:.2}\n- reviewed frames: {} (closed {}, not_closed {}, uncertain {}, unobservable {}), unknown predictions {}\n- frame recall: {} ({} / {}), frame false-close: {}\n- event attainment: {} ({} / {})\n- interval recall: {} ({} ms / {} ms), interval false-close: {}\n- observability: {}\n- result: {}\n",
            eye.as_str(),
            thresholds.for_side(eye).close_gap(),
            thresholds.for_side(eye).reopen_gap(),
            thresholds.for_side(eye).min_blink(),
            metrics.reviewed.frames,
            metrics.reviewed.closed_frames,
            metrics.reviewed.not_closed_frames,
            metrics.reviewed.uncertain_frames,
            metrics.reviewed.unobservable_frames,
            metrics.reviewed.unknown_predictions,
            opt_metric(metrics.reviewed.closed_recall),
            metrics.reviewed.closed_attained_frames,
            metrics.reviewed.closed_frames,
            opt_metric(metrics.reviewed.false_close),
            opt_metric(metrics.events.event_attainment),
            metrics.events.attained_events,
            metrics.events.closed_events,
            opt_metric(metrics.intervals.closed_recall),
            metrics.intervals.closed_predicted_ms,
            metrics.intervals.closed_time_ms,
            opt_metric(metrics.intervals.false_close),
            opt_metric(metrics.reviewed.observability),
            if eye_ok { "pass" } else { "unverified" }
        );
        if eye_ok {
            verified.push((eye, thresholds.for_side(eye)));
        }
    }

    std::fs::create_dir_all(&options.output)
        .map_err(|error| format!("failed to create {}: {error}", options.output.display()))?;
    write(&options.output.join("test-report.md"), &report)?;

    let mut left = None;
    let mut right = None;
    for (eye, threshold) in verified {
        match eye {
            EyeSide::Left => left = Some(threshold),
            EyeSide::Right => right = Some(threshold),
        }
    }
    let (Some(left), Some(right)) = (left, right) else {
        println!("both eyes did not pass; runtime eye_closure_profile.json was not written");
        return Ok(());
    };
    if !fingerprint_ok {
        println!(
            "fingerprints do not match the current inference contract; runtime profile was not written"
        );
        return Ok(());
    }
    let profile = EyeClosureProfileDocument {
        schema_version: EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
        algorithm_version: EYE_CLOSURE_ALGORITHM_VERSION,
        feature: EYE_CLOSURE_FEATURE.into(),
        status: EyeClosureVerificationStatus::Verified,
        left: values(left),
        right: values(right),
        fingerprints: EyeClosureFingerprints {
            task_bundle_sha256: Some(TASK_BUNDLE_SHA256.to_owned()),
            feature: EYE_CLOSURE_FEATURE.into(),
            preprocess: Some("mediapipe face landmarker landmarks and raw blendshapes".into()),
        },
        applies_to: Some("held-out captures; real webcam conditions not yet confirmed".into()),
    };
    write_json(&options.output.join("eye_closure_profile.json"), &profile)?;
    println!(
        "validated profile written: left close_gap={:.4}/reopen_gap={:.4}/min_blink={:.2}, right close_gap={:.4}/reopen_gap={:.4}/min_blink={:.2}",
        left.close_gap(),
        left.reopen_gap(),
        left.min_blink(),
        right.close_gap(),
        right.reopen_gap(),
        right.min_blink()
    );
    Ok(())
}

fn opt_metric(value: Option<f64>) -> String {
    value.map_or_else(|| "unmeasured".to_owned(), |value| format!("{value:.4}"))
}

fn values(threshold: vtuber_tracking::EyeGeometryThreshold) -> EyeGeometryThresholdValues {
    EyeGeometryThresholdValues {
        close_gap: threshold.close_gap(),
        reopen_gap: threshold.reopen_gap(),
        min_blink: threshold.min_blink(),
    }
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
    write(path, &text)
}
