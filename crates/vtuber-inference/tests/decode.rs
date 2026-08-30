// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for inference output decode.

#![cfg(feature = "legacy-face-stack")]

use vtuber_inference::decode::landmarks::*;
use vtuber_inference::error::InferenceError;
use vtuber_inference::roi::FaceRoi;

fn peppa_contract() -> LandmarkOutputContract {
    LandmarkOutputContract {
        tensor_name: "/Concat_1".into(),
        expected_shape: vec![1, 98, 3],
        expected_dtype: TensorDtype::F32,
        landmark_count: 98,
        coordinate_range: CoordinateRange::ZeroToOne,
        channels: LandmarkChannels::XyConfidence,
        layout: LandmarkLayout::BatchLandmarkChannels,
        canonical_size: 256,
    }
}

fn tensor(data: Vec<f32>, shape: Vec<usize>, dtype: TensorDtype) -> OutputTensor<'static> {
    let shape = Box::leak(shape.into_boxed_slice());
    let data = Box::leak(data.into_boxed_slice());
    OutputTensor { shape, data, dtype }
}

fn centered_roi() -> FaceRoi {
    FaceRoi {
        center_x: 320.0,
        center_y: 240.0,
        rotation_rad: 0.0,
        scale: 0.5,
        confidence: 1.0,
    }
}

#[test]
fn landmark_decode_golden_count() {
    let mut data = vec![0.5_f32; 98 * 3];
    data[2] = 0.9;
    let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);

    let landmarks = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap();

    assert_eq!(landmarks.len(), 98);
    assert!((landmarks[0].x - 0.5).abs() < 1e-6);
    assert!((landmarks[0].y - 0.5).abs() < 1e-6);
    assert!((landmarks[0].visibility - 0.9).abs() < 1e-6);
    assert!(
        landmarks
            .iter()
            .all(|l| l.x.is_finite() && l.visibility.is_finite())
    );
}

#[test]
fn landmark_decode_invalid_shape() {
    let t = tensor(vec![0.0_f32; 97 * 3], vec![1, 97, 3], TensorDtype::F32);
    let err = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap_err();
    assert!(
        matches!(err, InferenceError::OutputShapeMismatch { .. }),
        "expected shape mismatch, got {err:?}"
    );
}

#[test]
fn landmark_decode_element_count_mismatch() {
    let mut data = vec![0.0_f32; 98 * 3];
    data.push(0.0);
    let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);
    let err = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap_err();
    assert!(
        matches!(err, InferenceError::OutputElementCountMismatch { .. }),
        "expected element count mismatch, got {err:?}"
    );
}

#[test]
fn landmark_decode_nan_value() {
    let mut data = vec![0.5_f32; 98 * 3];
    data[10] = f32::NAN;
    let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);
    let err = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap_err();
    assert!(
        matches!(err, InferenceError::InvalidOutputValue { .. }),
        "expected invalid value, got {err:?}"
    );
}

#[test]
fn landmark_decode_dtype_mismatch() {
    let t = tensor(vec![0.0_f32; 98 * 3], vec![1, 98, 3], TensorDtype::F16);
    let err = decode_landmarks(&t, &peppa_contract(), &centered_roi(), 640, 480).unwrap_err();
    assert!(
        matches!(err, InferenceError::OutputDtypeMismatch { .. }),
        "expected dtype mismatch, got {err:?}"
    );
}

#[test]
fn landmark_decode_minus_one_to_one_range() {
    let contract = LandmarkOutputContract {
        coordinate_range: CoordinateRange::MinusOneToOne,
        ..peppa_contract()
    };
    let mut data = vec![0.0_f32; 98 * 3];
    data[0] = -1.0;
    data[1] = 1.0;
    data[2] = 1.0;
    let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);

    let landmarks = decode_landmarks(&t, &contract, &centered_roi(), 640, 480).unwrap();

    // (-1, 1) maps to (0, 1) in ROI-local normalized space.
    // side = 0.5 * 480 = 240, so top-center of ROI is (200, 360).
    assert!((landmarks[0].x - 200.0 / 640.0).abs() < 1e-6);
    assert!((landmarks[0].y - 360.0 / 480.0).abs() < 1e-6);
}

#[test]
fn landmark_decode_pixel_range() {
    let contract = LandmarkOutputContract {
        coordinate_range: CoordinateRange::Pixel,
        ..peppa_contract()
    };
    let mut data = vec![0.0_f32; 98 * 3];
    data[0] = 128.0;
    data[1] = 128.0;
    data[2] = 0.8;
    let t = tensor(data, vec![1, 98, 3], TensorDtype::F32);

    let landmarks = decode_landmarks(&t, &contract, &centered_roi(), 640, 480).unwrap();

    // (128, 128) in a 256x256 canonical input maps to the ROI center.
    assert!((landmarks[0].x - 0.5).abs() < 1e-6);
    assert!((landmarks[0].y - 0.5).abs() < 1e-6);
    assert!((landmarks[0].visibility - 0.8).abs() < 1e-6);
}

#[test]
fn landmark_decode_roi_transform_round_trip() {
    let roi = FaceRoi {
        center_x: 123.0,
        center_y: 456.0,
        rotation_rad: 0.7,
        scale: 0.3,
        confidence: 1.0,
    };
    let local = (0.2_f32, 0.8_f32);

    let (x_img, y_img) = roi_local_to_image(local.0, local.1, &roi, 1280, 720).unwrap();
    let (x_back, y_back) = image_to_roi_local(x_img, y_img, &roi, 1280, 720).unwrap();

    assert!((x_back - local.0).abs() < 1e-5);
    assert!((y_back - local.1).abs() < 1e-5);
}

#[test]
fn landmark_decode_canonical_round_trip() {
    let local = (0.35_f32, 0.65_f32);
    let canonical = roi_local_to_canonical(local.0, local.1, 256);
    let back = canonical_to_roi_local(canonical.0, canonical.1, 256);
    assert!((back.0 - local.0).abs() < 1e-6);
    assert!((back.1 - local.1).abs() < 1e-6);
}

#[test]
fn landmark_decode_roi_center_maps_to_image_center() {
    let (x, y) = roi_local_to_image(0.5, 0.5, &centered_roi(), 640, 480).unwrap();
    assert!((x - 320.0).abs() < 1e-6);
    assert!((y - 240.0).abs() < 1e-6);
}
