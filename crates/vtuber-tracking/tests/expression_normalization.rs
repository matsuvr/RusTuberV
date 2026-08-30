// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for blink / mouth expression normalization.
//!
//! These tests verify the end-to-end contract of [`ExpressionFilter`]: raw
//! coefficients are normalized against calibrated neutral / closed / open
//! ranges, missing channels are filled according to the configured fallback
//! policy, and the output is always finite and clamped to `[0, 1]`.

use approx::assert_relative_eq;
use vtuber_core::types::{ExpressionCoefficients, MonoTimeNs, RawExpressionObservation};
use vtuber_tracking::filter::{
    ExpressionCalibration, ExpressionFilter, ExpressionFilterParams, ExpressionRange,
    MissingChannelFallback, MissingChannelPolicy,
};

fn default_calibration() -> ExpressionCalibration {
    // The neutral open-eye value equals the `open` endpoint so that a
    // relaxed face produces a normalized blink coefficient near zero.
    ExpressionCalibration::new(
        ExpressionRange::for_blink(0.05, 0.05, 0.90).expect("valid left blink range"),
        ExpressionRange::for_blink(0.05, 0.05, 0.90).expect("valid right blink range"),
        ExpressionRange::for_mouth(0.05, 0.80).expect("valid mouth range"),
    )
}

fn fast_params() -> ExpressionFilterParams {
    // Use fast attack/release so the tests do not have to run many frames
    // to observe the steady-state value, while still exercising smoothing.
    // The mouth dead zone is disabled for tests that assert exact maxima.
    ExpressionFilterParams {
        mouth_dead_zone: 0.0,
        ..ExpressionFilterParams::with_time_constants(0.01, 0.01)
    }
}

fn observation(blink_left: f32, blink_right: f32, mouth_open: f32) -> RawExpressionObservation {
    RawExpressionObservation {
        blink_left,
        blink_left_confidence: 0.9,
        blink_right,
        blink_right_confidence: 0.9,
        mouth_open,
        mouth_open_confidence: 0.9,
    }
}

fn assert_all_finite(expr: &ExpressionCoefficients) {
    assert!(
        expr.blink_left.is_finite() && (0.0..=1.0).contains(&expr.blink_left),
        "blink_left out of range: {}",
        expr.blink_left
    );
    assert!(
        expr.blink_right.is_finite() && (0.0..=1.0).contains(&expr.blink_right),
        "blink_right out of range: {}",
        expr.blink_right
    );
    assert!(
        expr.aa.is_finite() && (0.0..=1.0).contains(&expr.aa),
        "aa out of range: {}",
        expr.aa
    );
}

#[test]
fn expression_normalization_neutral_raw_values_yield_zero() {
    let mut filter = ExpressionFilter::new(default_calibration(), fast_params());
    let out = filter.update(&observation(0.05, 0.05, 0.05), MonoTimeNs(16_666_667));

    assert_all_finite(&out);
    assert_relative_eq!(out.blink_left, 0.0, epsilon = 1e-4);
    assert_relative_eq!(out.blink_right, 0.0, epsilon = 1e-4);
    assert_relative_eq!(out.aa, 0.0, epsilon = 1e-4);
}

#[test]
fn expression_normalization_fully_closed_eyes_and_open_mouth_yield_one() {
    let mut filter = ExpressionFilter::new(default_calibration(), fast_params());
    let out = filter.update(&observation(0.90, 0.90, 0.80), MonoTimeNs(16_666_667));

    assert_all_finite(&out);
    assert_relative_eq!(out.blink_left, 1.0, epsilon = 1e-4);
    assert_relative_eq!(out.blink_right, 1.0, epsilon = 1e-4);
    assert_relative_eq!(out.aa, 1.0, epsilon = 1e-4);
}

#[test]
fn expression_normalization_left_and_right_blink_are_independent() {
    let mut filter = ExpressionFilter::new(default_calibration(), fast_params());
    let out = filter.update(&observation(0.90, 0.05, 0.05), MonoTimeNs(16_666_667));

    assert_all_finite(&out);
    assert_relative_eq!(out.blink_left, 1.0, epsilon = 1e-4);
    assert_relative_eq!(out.blink_right, 0.0, epsilon = 1e-4);
    assert_relative_eq!(out.aa, 0.0, epsilon = 1e-4);
}

#[test]
fn expression_normalization_mouth_maps_to_aa_only() {
    let mut filter = ExpressionFilter::new(default_calibration(), fast_params());
    let out = filter.update(&observation(0.10, 0.10, 0.80), MonoTimeNs(16_666_667));

    assert_all_finite(&out);
    assert_relative_eq!(out.aa, 1.0, epsilon = 1e-4);
    assert_relative_eq!(out.ih, 0.0, epsilon = 1e-6);
    assert_relative_eq!(out.ou, 0.0, epsilon = 1e-6);
    assert_relative_eq!(out.ee, 0.0, epsilon = 1e-6);
    assert_relative_eq!(out.oh, 0.0, epsilon = 1e-6);
}

#[test]
fn expression_normalization_inverted_blink_range_rejects() {
    let err = ExpressionRange::for_blink(0.50, 0.90, 0.05).expect_err("open > closed");
    assert_eq!(err.code(), "EXPRESSION_CALIBRATION_INVERTED_RANGE");
}

#[test]
fn expression_normalization_zero_span_mouth_range_rejects() {
    let err = ExpressionRange::for_mouth(0.50, 0.50).expect_err("equal endpoints");
    assert_eq!(err.code(), "EXPRESSION_CALIBRATION_ZERO_SPAN");
}

#[test]
fn expression_normalization_out_of_range_calibration_value_rejects() {
    let err = ExpressionRange::for_blink(-0.1, 0.0, 1.0).expect_err("negative neutral");
    assert_eq!(err.code(), "EXPRESSION_CALIBRATION_VALUE_OUT_OF_RANGE");
}

#[test]
fn expression_normalization_missing_left_blink_mirrors_right() {
    let mut obs = observation(0.90, 0.90, 0.05);
    obs.blink_left_confidence = 0.1; // missing

    let mut filter = ExpressionFilter::new(default_calibration(), fast_params());
    let out = filter.update(&obs, MonoTimeNs(16_666_667));

    assert_all_finite(&out);
    assert_relative_eq!(out.blink_left, 1.0, epsilon = 1e-4);
}

#[test]
fn expression_normalization_missing_blink_with_zero_policy_outputs_zero() {
    let mut obs = observation(0.90, 0.05, 0.05);
    obs.blink_left_confidence = 0.1;

    let mut params = fast_params();
    params.missing_policy = MissingChannelPolicy {
        blink_left: MissingChannelFallback::Zero,
        blink_right: MissingChannelFallback::MirrorOpposite,
        mouth: MissingChannelFallback::Zero,
    };

    let mut filter = ExpressionFilter::new(default_calibration(), params);
    let out = filter.update(&obs, MonoTimeNs(16_666_667));

    assert_all_finite(&out);
    assert_relative_eq!(out.blink_left, 0.0, epsilon = 1e-4);
}

#[test]
fn expression_normalization_missing_mouth_outputs_zero() {
    let mut obs = observation(0.10, 0.10, 0.80);
    obs.mouth_open_confidence = 0.1;

    let mut filter = ExpressionFilter::new(default_calibration(), fast_params());
    let out = filter.update(&obs, MonoTimeNs(16_666_667));

    assert_all_finite(&out);
    assert_relative_eq!(out.aa, 0.0, epsilon = 1e-4);
}

#[test]
fn expression_normalization_mouth_dead_zone_silences_small_openings() {
    let cal = ExpressionCalibration::new(
        ExpressionRange::for_blink(0.10, 0.05, 0.90).unwrap(),
        ExpressionRange::for_blink(0.10, 0.05, 0.90).unwrap(),
        ExpressionRange::for_mouth(0.05, 0.80).unwrap(),
    );
    let mut params = fast_params();
    params.mouth_dead_zone = 0.2;

    let mut filter = ExpressionFilter::new(cal, params);
    // normalized mouth = (0.15 - 0.05) / (0.80 - 0.05) = 0.1333...
    let out = filter.update(&observation(0.10, 0.10, 0.15), MonoTimeNs(16_666_667));

    assert_all_finite(&out);
    assert_relative_eq!(out.aa, 0.0, epsilon = 1e-4);
}

#[test]
fn expression_normalization_attack_is_faster_than_release() {
    let mut params = fast_params();
    params.attack_time_constant_sec = 0.01;
    params.release_time_constant_sec = 0.50;

    let mut filter = ExpressionFilter::new(default_calibration(), params);

    // Step from neutral to fully open.
    let up = filter.update(&observation(0.90, 0.90, 0.80), MonoTimeNs(16_666_667));
    // Step back to neutral after the same elapsed interval.
    let down = filter.update(&observation(0.10, 0.10, 0.05), MonoTimeNs(33_333_334));

    // With fast attack and slow release, `up.aa` should be much closer to
    // 1.0 than `down.aa` is to 0.0.
    assert!(up.aa > 1.0 - down.aa, "up={}, down={}", up.aa, down.aa);
}

#[test]
fn expression_normalization_non_finite_or_out_of_range_raw_values_are_clamped() {
    let mut filter = ExpressionFilter::new(default_calibration(), fast_params());
    let bad = RawExpressionObservation {
        blink_left: f32::NAN,
        blink_left_confidence: 0.9,
        blink_right: f32::INFINITY,
        blink_right_confidence: 0.9,
        mouth_open: 1.5,
        mouth_open_confidence: 0.9,
    };

    let out = filter.update(&bad, MonoTimeNs(16_666_667));
    assert_all_finite(&out);
}

#[test]
fn expression_normalization_reset_clears_smoothed_state() {
    let mut filter = ExpressionFilter::new(default_calibration(), fast_params());
    let _ = filter.update(&observation(0.90, 0.90, 0.80), MonoTimeNs(16_666_667));
    filter.reset();
    let out = filter.update(&observation(0.90, 0.90, 0.80), MonoTimeNs(33_333_334));

    // After reset the first update initializes directly to the target.
    assert_all_finite(&out);
    assert_relative_eq!(out.blink_left, 1.0, epsilon = 1e-4);
    assert_relative_eq!(out.blink_right, 1.0, epsilon = 1e-4);
    assert_relative_eq!(out.aa, 1.0, epsilon = 1e-4);
}
