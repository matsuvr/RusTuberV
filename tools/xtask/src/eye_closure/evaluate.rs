//! Held-out evaluation and validated-profile emission (Issue #52).
//!
//! `evaluate` freezes one candidate, reads the test split exactly once, and
//! writes an installable profile only when both eyes pass with visual labels
//! and matching inference fingerprints. A failing evaluation still writes a
//! report but leaves no runtime profile behind.

use std::fmt::Write as _;
use std::path::Path;

use vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256;
use vtuber_tracking::{
    EYE_CLOSURE_ALGORITHM_VERSION, EYE_CLOSURE_FEATURE, EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
    EyeClosureFingerprints, EyeClosureProfileDocument, EyeClosureThresholds,
    EyeClosureVerificationStatus, EyeSide,
};

use super::Options;
use super::fit::{
    MAX_FALSE_CLOSE, MIN_CLOSED_RECALL, build_series, evaluate_candidate, read_profile,
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

    let mut report = String::from("# Eye-closure held-out evaluation\n\n");
    let _ = writeln!(
        report,
        "- feature: `{EYE_CLOSURE_FEATURE}`\n- algorithm: {EYE_CLOSURE_ALGORITHM_VERSION}\n- label sha256: `{}`\n- test takes: {test_takes:?}\n- fingerprint match: {fingerprint_ok}\n- criteria: closed recall >= {MIN_CLOSED_RECALL}, false-close <= {MAX_FALSE_CLOSE}, visual labels required\n",
        labels.sha256
    );

    let mut verified: Vec<(EyeSide, vtuber_tracking::EyeThreshold)> = Vec::new();
    for eye in [EyeSide::Left, EyeSide::Right] {
        let series = build_series(&data, &labels, &test_takes, eye);
        let metrics = evaluate_candidate(&series, thresholds.for_side(eye));
        let visual_ok = metrics.visual_closed_frames > 0 && metrics.visual_not_closed_frames > 0;
        let eye_ok = metrics.is_acceptable() && visual_ok;
        let _ = writeln!(
            report,
            "## {} eye\n\n- close_at={:.3}, reopen_at={:.3}\n- closed recall: {:.4} ({} frames, events {}/{})\n- false-close: {:.4}\n- uncertain/unobservable frames: {}/{}\n- visual fully_closed / not_closed frames: {}/{}\n- result: {}\n",
            eye.as_str(),
            thresholds.for_side(eye).close_at(),
            thresholds.for_side(eye).reopen_at(),
            metrics.closed_recall,
            metrics.visual_closed_frames,
            metrics.closed_events_attained,
            metrics.closed_events,
            metrics.false_close,
            metrics.uncertain_frames,
            metrics.unobservable_frames,
            metrics.visual_closed_frames,
            metrics.visual_not_closed_frames,
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
            preprocess: Some("mediapipe face landmarker raw blendshapes".into()),
        },
        applies_to: Some(
            "held-out iPhone-derived captures; real webcam conditions not yet confirmed".into(),
        ),
    };
    write_json(&options.output.join("eye_closure_profile.json"), &profile)?;
    println!(
        "validated profile written: left close_at={:.3}/reopen_at={:.3}, right close_at={:.3}/reopen_at={:.3}",
        left.close_at(),
        left.reopen_at(),
        right.close_at(),
        right.reopen_at()
    );
    Ok(())
}

fn values(threshold: vtuber_tracking::EyeThreshold) -> vtuber_tracking::EyeThresholdValues {
    vtuber_tracking::EyeThresholdValues {
        close_at: threshold.close_at(),
        reopen_at: threshold.reopen_at(),
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

/// Retains the threshold bundle type for the loader tests.
#[allow(dead_code)]
fn _thresholds(thresholds: &EyeClosureThresholds) -> f32 {
    thresholds.left().close_at()
}
