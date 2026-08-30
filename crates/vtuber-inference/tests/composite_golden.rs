// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Golden contract tests for the detector-to-crop-to-landmark boundaries.
//!
//! The positive fixture is a synthetic detection list and synthetic landmark
//! tensor. No unlicensed or user-captured image is stored in the repository;
//! the actual model provenance remains pinned by the production descriptor.

#![cfg(feature = "legacy-face-stack")]

#[path = "support/composite_fixture.rs"]
mod fixture;

use vtuber_core::types::NormalizedRect;
use vtuber_inference::crop::FaceCropTransform;
use vtuber_inference::detector::DetectorDecodeOutcome;
use vtuber_inference::{
    DetectorPostprocessConfig, FaceCropConfig, FrameFaceInference, FrameInferenceOutcome,
};

#[test]
fn composite_golden_production_model_and_crop_contract_is_pinned() {
    let descriptor = fixture::production_descriptor();
    assert_eq!(descriptor.id, "ultraface-rfb-320-peppapig-98");
    assert_eq!(descriptor.detector.byte_size, 1_270_727);
    assert_eq!(
        descriptor.detector.sha256,
        "34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017"
    );
    assert_eq!(descriptor.landmarks.byte_size, 13_728_231);
    assert_eq!(
        descriptor.landmarks.sha256,
        "73EDF90954F05EBEF4639E7FA8620C5F83CCA09D2476DE66AB100F26C2B25E7A"
    );
    assert_eq!(
        descriptor.detector_postprocess,
        DetectorPostprocessConfig {
            score_threshold: 0.7,
            nms_iou: 0.3,
            max_pre_nms_candidates: 256,
            max_post_nms_detections: 16,
        }
    );
    assert_eq!(
        descriptor.crop,
        FaceCropConfig {
            square_scale: 1.35,
            center_y_offset_fraction: -0.05,
            output_size: [256, 256],
            interpolation: vtuber_inference::CropInterpolation::Bilinear,
            outside_fill: vtuber_inference::CropOutsideFill::NormalizationMean,
        }
    );
}

#[test]
fn composite_golden_mean_frame_is_no_face_without_landmark_execution() {
    let detector = fixture::MockDetector::new([DetectorDecodeOutcome::NoFace]);
    let detector_calls = detector.calls.clone();
    let landmark = fixture::MockLandmark::new([]);
    let landmark_calls = landmark.calls.clone();
    let mut runtime = fixture::runtime(detector, landmark);

    let outcome = runtime
        .infer_frame(&fixture::mean_frame(1))
        .expect("no-face is ordinary state");
    assert!(matches!(outcome, FrameInferenceOutcome::NoFace));
    let timing = runtime.take_timing();
    assert!(timing.detector.is_some());
    assert!(timing.crop.is_none());
    assert!(timing.landmark.is_none());
    assert!(timing.total > std::time::Duration::ZERO);
    assert_eq!(detector_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(landmark_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
#[ignore = "requires downloaded legacy UltraFace and Peppa Pig ONNX research artifacts"]
fn composite_golden_actual_ultraface_mean_frame_is_no_face() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("crate is nested beneath workspace root");
    let mut runtime = vtuber_inference::CompositeFrameInference::from_pipeline_descriptor(
        &fixture::production_descriptor(),
        &root.join("assets").join("models"),
        &vtuber_inference::RuntimeSettings::default(),
    )
    .expect("manifest-tracked production models should construct");

    let outcome = runtime
        .infer_frame(&fixture::mean_frame(3))
        .expect("mean frame should not produce a runtime error");
    assert!(matches!(outcome, FrameInferenceOutcome::NoFace));
}

#[test]
fn composite_golden_single_face_preserves_box_crop_and_98_source_landmarks() {
    let detection = fixture::detection();
    let detector = fixture::MockDetector::new([DetectorDecodeOutcome::Detections(vec![detection])]);
    let landmark = fixture::MockLandmark::new([fixture::valid_landmarks()]);
    let mut runtime = fixture::runtime(detector, landmark);

    let outcome = runtime
        .infer_frame(&fixture::mean_frame(2))
        .expect("synthetic positive should decode");
    let FrameInferenceOutcome::Face(observation) = outcome else {
        panic!("expected a synthetic single face");
    };

    let descriptor = fixture::production_descriptor();
    let transform = FaceCropTransform::from_detector_box(64, 48, &detection.rect, descriptor.crop)
        .expect("golden detector box is valid");
    let (left, top, side) = transform.source_pixel_square();
    assert!((left - 7.8).abs() < 1e-5);
    assert!((top - 6.6).abs() < 1e-5);
    assert!((side - 32.4).abs() < 1e-5);
    assert_eq!(observation.landmarks.len(), 98);
    assert_eq!(observation.face_confidence, 0.9);
    assert_eq!(observation.roi, transform.source_roi());
    assert_eq!(observation.schema.0, "peppapig-98");
    assert!(observation.landmarks.iter().all(|landmark| {
        landmark.x.is_finite()
            && landmark.y.is_finite()
            && landmark.z.is_finite()
            && (0.0..=1.0).contains(&landmark.x)
            && (0.0..=1.0).contains(&landmark.y)
    }));
    for landmark in &observation.landmarks {
        assert!((landmark.x - 0.375).abs() < 1e-5);
        assert!((landmark.y - 0.475).abs() < 1e-5);
    }
    let timing = runtime.take_timing();
    assert!(timing.detector.is_some());
    assert!(timing.crop.is_some());
    assert!(timing.landmark.is_some());
    assert!(timing.decode.is_some());
    assert!(timing.total > std::time::Duration::ZERO);
}

#[test]
fn composite_golden_synthetic_primary_box_is_explicit_and_unmirrored() {
    let detection = fixture::detection();
    let expected = NormalizedRect {
        x: 0.25,
        y: 0.25,
        width: 0.25,
        height: 0.5,
        rotation_rad: 0.0,
    };
    assert_eq!(detection.rect, expected);
    assert!(detection.rect.x < detection.rect.x + detection.rect.width);
    assert!(detection.rect.y < detection.rect.y + detection.rect.height);
}
