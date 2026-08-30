// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for confidence synthesis and hysteresis gating.

use vtuber_tracking::confidence::{
    ConfidenceAssessment, ConfidenceConfigError, ConfidenceError, ConfidenceGate,
    ConfidenceGateParams, ConfidenceInputs, ConfidencePolicies, ConfidenceSignal, ConfidenceSource,
    MissingSourcePolicy, synthesize,
};

fn test_params() -> ConfidenceGateParams {
    ConfidenceGateParams {
        enter_threshold: 0.8,
        exit_threshold: 0.4,
        required_consecutive_good: 2,
        required_consecutive_bad: 3,
        max_count: 100,
    }
}

fn good_inputs() -> ConfidenceInputs {
    ConfidenceInputs {
        detector: Some(0.95),
        landmark: Some(0.9),
        pose: Some(0.85),
        expression: Some(0.88),
    }
}

#[test]
fn confidence_hysteresis_synthesizes_minimum() {
    let frame = synthesize(&good_inputs(), &ConfidencePolicies::default()).unwrap();
    assert_eq!(frame, 0.85);
}

#[test]
fn confidence_hysteresis_missing_detector_treated_as_zero() {
    let inputs = ConfidenceInputs {
        detector: None,
        landmark: Some(0.9),
        ..ConfidenceInputs::default()
    };
    let policies = ConfidencePolicies::default();
    assert_eq!(synthesize(&inputs, &policies).unwrap(), 0.0);
}

#[test]
fn confidence_hysteresis_missing_expression_ignored_by_default() {
    let inputs = ConfidenceInputs {
        detector: Some(0.9),
        landmark: Some(0.8),
        expression: None,
        ..ConfidenceInputs::default()
    };
    assert_eq!(
        synthesize(&inputs, &ConfidencePolicies::default()).unwrap(),
        0.8
    );
}

#[test]
fn confidence_hysteresis_nan_source_rejected() {
    let inputs = ConfidenceInputs {
        detector: Some(f32::NAN),
        landmark: Some(0.9),
        ..ConfidenceInputs::default()
    };
    let err = synthesize(&inputs, &ConfidencePolicies::default()).unwrap_err();
    assert_eq!(
        err,
        ConfidenceError::InvalidValue(ConfidenceSource::Detector)
    );
}

#[test]
fn confidence_hysteresis_acquires_after_consecutive_good() {
    let mut gate = ConfidenceGate::new(test_params()).unwrap();

    let a1 = gate.update(0.85);
    assert!(!a1.is_confident);
    assert_eq!(a1.signal, ConfidenceSignal::None);
    assert_eq!(a1.consecutive_good, 1);

    let a2 = gate.update(0.9);
    assert!(a2.is_confident);
    assert_eq!(a2.signal, ConfidenceSignal::Acquire);
    assert_eq!(a2.consecutive_good, 0);
}

#[test]
fn confidence_hysteresis_degrades_after_consecutive_bad() {
    let mut gate = ConfidenceGate::new(test_params()).unwrap();
    gate.update(0.85);
    gate.update(0.9); // Acquire

    gate.update(0.35);
    gate.update(0.30);
    let a = gate.update(0.25);
    assert!(!a.is_confident);
    assert_eq!(a.signal, ConfidenceSignal::Degrade);
}

#[test]
fn confidence_hysteresis_no_oscillation_in_hysteresis_band() {
    let mut gate = ConfidenceGate::new(test_params()).unwrap();
    gate.update(0.85);
    gate.update(0.9); // Acquire

    for _ in 0..20 {
        let a = gate.update(0.6);
        assert!(
            a.is_confident,
            "gate should remain confident in the hysteresis band: {a:?}"
        );
        assert_eq!(a.signal, ConfidenceSignal::None);
        assert_eq!(a.consecutive_good, 0);
        assert_eq!(a.consecutive_bad, 0);
    }
}

#[test]
fn confidence_hysteresis_non_finite_input_is_rejected() {
    let mut gate = ConfidenceGate::new(test_params()).unwrap();
    let a = gate.update(f32::NAN);
    assert!(!a.is_confident);
    assert!(!a.valid);
    assert_eq!(a.frame_confidence, 0.0);
    assert_eq!(a.consecutive_bad, 1);
}

#[test]
fn confidence_hysteresis_counters_are_bounded() {
    let params = ConfidenceGateParams {
        max_count: 4,
        ..test_params()
    };
    let mut gate = ConfidenceGate::new(params).unwrap();

    for i in 0..10 {
        let a = gate.update(0.0);
        assert!(
            a.consecutive_bad <= 4,
            "counter overflow at frame {i}: {a:?}"
        );
    }
    let a = gate.update(0.0);
    assert_eq!(a.consecutive_bad, 4);
}

#[test]
fn confidence_hysteresis_ignoring_all_missing_sources_yields_zero() {
    let inputs = ConfidenceInputs::default();
    let policies = ConfidencePolicies {
        detector: MissingSourcePolicy::Ignore,
        landmark: MissingSourcePolicy::Ignore,
        pose: MissingSourcePolicy::Ignore,
        expression: MissingSourcePolicy::Ignore,
    };
    assert_eq!(synthesize(&inputs, &policies).unwrap(), 0.0);
}

#[test]
fn confidence_hysteresis_rejects_misordered_thresholds() {
    let params = ConfidenceGateParams {
        enter_threshold: 0.4,
        exit_threshold: 0.8,
        ..ConfidenceGateParams::default()
    };
    assert!(matches!(
        ConfidenceGate::new(params).unwrap_err(),
        ConfidenceConfigError::ThresholdOrder { .. }
    ));
}

#[test]
fn confidence_hysteresis_full_pipeline_emits_expected_signals() {
    let mut gate = ConfidenceGate::new(test_params()).unwrap();

    // Two good frames -> Acquire.
    let a = update_synthesized(&mut gate, &good_inputs(), &ConfidencePolicies::default());
    assert_eq!(a.signal, ConfidenceSignal::None);
    assert!(!a.is_confident);

    let a = update_synthesized(&mut gate, &good_inputs(), &ConfidencePolicies::default());
    assert_eq!(a.signal, ConfidenceSignal::Acquire);
    assert!(a.is_confident);

    // Three bad frames -> Degrade.
    let bad_inputs = ConfidenceInputs {
        detector: Some(0.2),
        landmark: Some(0.9),
        ..ConfidenceInputs::default()
    };
    let a = update_synthesized(&mut gate, &bad_inputs, &ConfidencePolicies::default());
    assert_eq!(a.signal, ConfidenceSignal::None);
    assert!(a.is_confident);

    let a = update_synthesized(&mut gate, &bad_inputs, &ConfidencePolicies::default());
    assert_eq!(a.signal, ConfidenceSignal::None);
    assert!(a.is_confident);

    let a = update_synthesized(&mut gate, &bad_inputs, &ConfidencePolicies::default());
    assert_eq!(a.signal, ConfidenceSignal::Degrade);
    assert!(!a.is_confident);
}

fn update_synthesized(
    gate: &mut ConfidenceGate,
    inputs: &ConfidenceInputs,
    policies: &ConfidencePolicies,
) -> ConfidenceAssessment {
    let c = synthesize(inputs, policies).unwrap_or(0.0);
    gate.update(c)
}
