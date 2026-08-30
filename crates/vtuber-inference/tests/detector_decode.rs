// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration coverage for UltraFace output decoding and primary selection.

#![cfg(feature = "legacy-face-stack")]

use vtuber_core::types::NormalizedRect;
use vtuber_inference::DetectorPostprocessConfig;
use vtuber_inference::detector::{
    DetectorDecodeOutcome, DetectorRawOutputs, DetectorRawTensor, FaceDetection, decode_detections,
    hard_nms, select_primary_face,
};

fn config() -> DetectorPostprocessConfig {
    DetectorPostprocessConfig {
        score_threshold: 0.7,
        nms_iou: 0.3,
        max_pre_nms_candidates: 256,
        max_post_nms_detections: 16,
    }
}

fn raw_outputs(scores: &[(f32, f32)], boxes: &[[f32; 4]]) -> DetectorRawOutputs {
    DetectorRawOutputs {
        tensors: vec![
            DetectorRawTensor {
                name: "scores".into(),
                shape: vec![1, scores.len(), 2],
                values: scores
                    .iter()
                    .flat_map(|&(background, face)| [background, face])
                    .collect(),
            },
            DetectorRawTensor {
                name: "boxes".into(),
                shape: vec![1, boxes.len(), 4],
                values: boxes.iter().flat_map(|item| item.iter().copied()).collect(),
            },
        ],
    }
}

#[test]
fn detector_decode_thresholds_and_nms_known_fixture() {
    let raw = raw_outputs(
        &[(0.1, 0.95), (0.1, 0.9), (0.1, 0.85)],
        &[
            [0.0, 0.0, 0.6, 0.6],
            [0.05, 0.05, 0.65, 0.65],
            [0.7, 0.7, 0.9, 0.9],
        ],
    );

    let DetectorDecodeOutcome::Detections(detections) = decode_detections(&raw, config()).unwrap()
    else {
        panic!("expected detections");
    };
    assert_eq!(
        detections
            .iter()
            .map(|item| item.anchor_index)
            .collect::<Vec<_>>(),
        [0, 2]
    );
}

#[test]
fn detector_decode_rejects_output_name_and_shape_mismatch() {
    let mut raw = raw_outputs(&[(0.1, 0.9)], &[[0.1, 0.1, 0.2, 0.2]]);
    raw.tensors[0].name = "unexpected".into();
    assert!(decode_detections(&raw, config()).is_err());

    let mut raw = raw_outputs(&[(0.1, 0.9)], &[[0.1, 0.1, 0.2, 0.2]]);
    raw.tensors[0].shape = vec![1, 1, 3];
    assert!(decode_detections(&raw, config()).is_err());
}

#[test]
fn detector_nms_is_bounded_and_keeps_the_best_pre_nms_candidate() {
    let raw = raw_outputs(
        &[(0.1, 0.8), (0.1, 0.95)],
        &[[0.0, 0.0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6]],
    );
    let mut limited = config();
    limited.max_pre_nms_candidates = 1;

    let DetectorDecodeOutcome::Detections(detections) = decode_detections(&raw, limited).unwrap()
    else {
        panic!("expected one detection");
    };
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].anchor_index, 1);

    let many = (0..32)
        .map(|index| FaceDetection {
            rect: NormalizedRect {
                x: index as f32 / 64.0,
                y: 0.0,
                width: 0.01,
                height: 0.01,
                rotation_rad: 0.0,
            },
            confidence: 0.5,
            anchor_index: index,
        })
        .collect::<Vec<_>>();
    assert_eq!(hard_nms(&many, 0.3, usize::MAX).len(), 16);
}

#[test]
fn primary_face_selection_prefers_previous_roi_continuity() {
    let detections = [
        FaceDetection {
            rect: NormalizedRect {
                x: 0.02,
                y: 0.02,
                width: 0.3,
                height: 0.3,
                rotation_rad: 0.0,
            },
            confidence: 0.75,
            anchor_index: 3,
        },
        FaceDetection {
            rect: NormalizedRect {
                x: 0.65,
                y: 0.65,
                width: 0.3,
                height: 0.3,
                rotation_rad: 0.0,
            },
            confidence: 0.99,
            anchor_index: 4,
        },
    ];
    let previous = NormalizedRect {
        x: 0.0,
        y: 0.0,
        width: 0.3,
        height: 0.3,
        rotation_rad: 0.0,
    };

    assert_eq!(
        select_primary_face(&detections, Some(&previous))
            .unwrap()
            .anchor_index,
        3
    );
}
