// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for the calibration subsystem.
//!
//! These tests exercise the sample collector through its public API to verify
//! that valid neutral samples are retained, invalid samples are rejected with
//! stable reasons, and the same input stream always produces the same
//! decisions.

use vtuber_core::control::CalibrationSettings;
use vtuber_core::types::{
    FrameSeq, Landmark3, LandmarkSchemaId, MonoTimeNs, RawExpressionObservation,
};
use vtuber_tracking::{
    CalibrationCollector, CalibrationInput, NeutralContext, NeutralReference,
    NeutralValidationSettings, RejectionReason, SampleDecision,
};

fn settings() -> CalibrationSettings {
    CalibrationSettings::try_new(5, 5.0, 0.5, 5.0f32.to_radians(), 0.15).unwrap()
}

fn face_landmarks() -> Vec<Landmark3> {
    // A non-degenerate planar face-like shape with enough points for Kabsch.
    vec![
        Landmark3 {
            x: -1.0,
            y: 0.0,
            z: 0.05,
            visibility: 1.0,
        },
        Landmark3 {
            x: 1.0,
            y: 0.0,
            z: 0.05,
            visibility: 1.0,
        },
        Landmark3 {
            x: 0.0,
            y: 0.8,
            z: 0.0,
            visibility: 1.0,
        },
        Landmark3 {
            x: 0.0,
            y: -0.6,
            z: 0.1,
            visibility: 1.0,
        },
        Landmark3 {
            x: -0.5,
            y: 0.3,
            z: 0.02,
            visibility: 1.0,
        },
        Landmark3 {
            x: 0.5,
            y: 0.3,
            z: 0.02,
            visibility: 1.0,
        },
    ]
}

fn input(seq: u64, confidence: f32) -> CalibrationInput {
    CalibrationInput {
        source_seq: FrameSeq(seq),
        captured_at: MonoTimeNs(seq * 33_333_333),
        face_confidence: confidence,
        landmarks: face_landmarks(),
        expressions: RawExpressionObservation {
            blink_left: 0.1,
            blink_left_confidence: 0.9,
            blink_right: 0.1,
            blink_right_confidence: 0.9,
            mouth_open: 0.05,
            mouth_open_confidence: 0.9,
        },
        schema: LandmarkSchemaId("integration-test"),
    }
}

#[test]
fn calibration_collector_rejects_insufficient_landmarks() {
    let mut collector = CalibrationCollector::new(settings());
    let mut bad = input(1, 0.9);
    bad.landmarks = vec![
        Landmark3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            visibility: 1.0,
        },
        Landmark3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
            visibility: 1.0,
        },
    ];
    let decision = collector.offer(bad);
    assert_eq!(
        decision,
        SampleDecision::Rejected(RejectionReason::InvalidLandmarks)
    );
    assert_eq!(collector.metrics().rejected_invalid_landmarks, 1);
}

#[test]
fn calibration_collector_rejects_non_finite_landmark() {
    let mut collector = CalibrationCollector::new(settings());
    let mut bad = input(1, 0.9);
    bad.landmarks[0].x = f32::INFINITY;
    let decision = collector.offer(bad);
    assert_eq!(
        decision,
        SampleDecision::Rejected(RejectionReason::InvalidLandmarks)
    );
}

#[test]
fn calibration_collector_rejects_out_of_range_visibility() {
    let mut collector = CalibrationCollector::new(settings());
    let mut bad = input(1, 0.9);
    bad.landmarks[0].visibility = 1.5;
    let decision = collector.offer(bad);
    assert_eq!(
        decision,
        SampleDecision::Rejected(RejectionReason::InvalidLandmarks)
    );
}

#[test]
fn calibration_collector_rejects_faceless_low_confidence() {
    let mut collector = CalibrationCollector::new(settings());
    let decision = collector.offer(input(1, 0.0));
    assert!(
        matches!(
            decision,
            SampleDecision::Rejected(RejectionReason::LowConfidence { .. })
        ),
        "expected LowConfidence, got {decision:?}"
    );
}

#[test]
fn calibration_collector_rejects_sequence_regression() {
    let mut collector = CalibrationCollector::new(settings());
    assert_eq!(collector.offer(input(2, 0.9)), SampleDecision::Accepted);
    let decision = collector.offer(input(1, 0.9));
    assert!(
        matches!(
            decision,
            SampleDecision::Rejected(RejectionReason::DuplicateOrOldSeq { .. })
        ),
        "expected DuplicateOrOldSeq, got {decision:?}"
    );
}

#[test]
fn calibration_collector_rejects_schema_mismatch() {
    let mut collector = CalibrationCollector::new(settings());
    assert_eq!(collector.offer(input(1, 0.9)), SampleDecision::Accepted);

    let mut other = input(2, 0.9);
    other.schema = LandmarkSchemaId("other");
    let decision = collector.offer(other);
    assert_eq!(
        decision,
        SampleDecision::Rejected(RejectionReason::SchemaMismatch)
    );
}

#[test]
fn calibration_collector_stops_at_capacity() {
    let mut collector = CalibrationCollector::new(settings());
    for seq in 1..=5 {
        assert_eq!(collector.offer(input(seq, 0.9)), SampleDecision::Accepted);
    }
    let decision = collector.offer(input(6, 0.9));
    assert!(
        matches!(
            decision,
            SampleDecision::Rejected(RejectionReason::SessionFull { capacity: 5 })
        ),
        "expected SessionFull, got {decision:?}"
    );
    assert_eq!(collector.samples().len(), 5);
}

#[test]
fn calibration_collector_metrics_sum_correctly() {
    let mut collector = CalibrationCollector::new(settings());

    // One accepted, one low confidence, one invalid expression, one duplicate.
    collector.offer(input(1, 0.9));
    collector.offer(input(2, 0.1));
    let mut bad = input(3, 0.9);
    bad.expressions.blink_left_confidence = 1.5;
    collector.offer(bad);
    collector.offer(input(1, 0.9)); // duplicate of seq 1

    assert_eq!(collector.metrics().accepted, 1);
    assert_eq!(collector.metrics().rejected_low_confidence, 1);
    assert_eq!(collector.metrics().rejected_invalid_values, 1);
    assert_eq!(collector.metrics().rejected_duplicate_or_old_seq, 1);
    assert_eq!(collector.metrics().total_offered(), 4);
    assert_eq!(collector.metrics().total_rejected(), 3);
}

#[test]
fn calibration_collector_is_deterministic_for_same_stream() {
    let mut decisions_a = Vec::new();
    let mut collector_a = CalibrationCollector::new(settings());
    for seq in 1..=8 {
        let mut frame = input(seq, 0.9);
        if seq == 4 {
            // Introduce a single talking frame that should be rejected.
            frame.expressions.mouth_open = 0.8;
        }
        decisions_a.push(collector_a.offer(frame));
    }

    let mut decisions_b = Vec::new();
    let mut collector_b = CalibrationCollector::new(settings());
    for seq in 1..=8 {
        let mut frame = input(seq, 0.9);
        if seq == 4 {
            frame.expressions.mouth_open = 0.8;
        }
        decisions_b.push(collector_b.offer(frame));
    }

    assert_eq!(decisions_a, decisions_b);
    assert_eq!(collector_a.samples().len(), collector_b.samples().len());
    assert_eq!(collector_a.metrics(), collector_b.metrics());
}

#[test]
fn calibration_collector_ready_only_after_required_samples() {
    // Use the minimum allowed sample count so the test stays short.
    let settings = CalibrationSettings::try_new(5, 5.0, 0.5, 5.0f32.to_radians(), 0.15).unwrap();
    let mut collector = CalibrationCollector::new(settings);

    assert!(!collector.is_ready());
    for seq in 1..=4 {
        collector.offer(input(seq, 0.9));
        assert!(!collector.is_ready());
    }
    collector.offer(input(5, 0.9));
    assert!(collector.is_ready());
}

#[test]
fn neutral_reference_integration_builds_valid_profile() {
    let settings = CalibrationSettings::try_new(5, 5.0, 0.5, 5.0f32.to_radians(), 0.15).unwrap();
    let mut collector = CalibrationCollector::new(settings);
    for seq in 1..=5 {
        collector.offer(input(seq, 0.9));
    }

    let profile = NeutralReference::aggregate(
        &collector,
        &NeutralValidationSettings::default(),
        &NeutralContext::new(
            MonoTimeNs(1_000_000_000),
            Some("model-hash".into()),
            Some("camera-fp".into()),
        ),
    )
    .unwrap();

    assert_eq!(profile.schema, LandmarkSchemaId("integration-test"));
    assert!(!profile.landmarks.is_empty());
    assert!(profile.is_compatible_with(Some("model-hash")));
    assert!(!profile.is_compatible_with(Some("other-hash")));
}

#[test]
fn neutral_reference_integration_rejects_incomplete_collector() {
    let settings = CalibrationSettings::try_new(5, 5.0, 0.5, 5.0f32.to_radians(), 0.15).unwrap();
    let mut collector = CalibrationCollector::new(settings);
    collector.offer(input(1, 0.9));

    let err = NeutralReference::aggregate(
        &collector,
        &NeutralValidationSettings::default(),
        &NeutralContext::default(),
    )
    .unwrap_err();

    assert_eq!(err.code(), "CALIBRATION_INSUFFICIENT_SAMPLES");
}
