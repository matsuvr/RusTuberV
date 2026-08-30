// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for detector-box crop geometry and preprocessing.

#![cfg(feature = "legacy-face-stack")]

use std::sync::Arc;

use vtuber_core::types::{
    FrameSeq, Landmark3, MonoTimeNs, NormalizedRect, PixelFormat, VideoFrame,
};
use vtuber_inference::descriptor::{
    ChannelOrder, CropInterpolation, CropOutsideFill, FaceCropConfig, InputValueDomain,
    NormalizationContract, TensorContract, TensorLayout,
};
use vtuber_inference::{FaceCropPreprocessBuffers, FaceCropTransform, LandmarkCoordinateEncoding};

fn crop_config(scale: f32) -> FaceCropConfig {
    FaceCropConfig {
        square_scale: scale,
        center_y_offset_fraction: -0.05,
        output_size: [256, 256],
        interpolation: CropInterpolation::Bilinear,
        outside_fill: CropOutsideFill::NormalizationMean,
    }
}

fn peppa_input() -> TensorContract {
    TensorContract {
        shape: vec![1, 3, 256, 256],
        dtype: "float32".into(),
        layout: TensorLayout::Nchw,
        channel_order: ChannelOrder::Rgb,
        value_domain: InputValueDomain::UnitFloat,
        normalization: NormalizationContract {
            mean: [0.485, 0.456, 0.406],
            scale: [0.229, 0.224, 0.225],
        },
    }
}

fn frame(width: u32, height: u32, rgb: [u8; 3]) -> VideoFrame {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for _ in 0..width * height {
        data.extend_from_slice(&rgb);
    }
    VideoFrame {
        seq: FrameSeq(1),
        captured_at: MonoTimeNs(1),
        width,
        height,
        stride_bytes: width as usize * 3,
        format: PixelFormat::Rgb8,
        data: Arc::from(data),
    }
}

#[test]
fn face_crop_transform_keeps_square_source_pixels_on_wide_frame() {
    let detector_box = NormalizedRect {
        x: 0.4,
        y: 0.3,
        width: 0.1,
        height: 0.2,
        rotation_rad: 0.0,
    };
    let transform =
        FaceCropTransform::from_detector_box(640, 480, &detector_box, crop_config(1.35)).unwrap();
    let (_, _, side) = transform.source_pixel_square();
    let roi = transform.source_roi();

    assert!((side - 129.6).abs() < 1e-4);
    assert!(((roi.width * 640.0) - (roi.height * 480.0)).abs() < 1e-4);
    assert!(roi.y < 0.3);
}

#[test]
fn crop_round_trip_preserves_source_coordinates() {
    let detector_box = NormalizedRect {
        x: 0.2,
        y: 0.25,
        width: 0.2,
        height: 0.25,
        rotation_rad: 0.0,
    };
    let transform =
        FaceCropTransform::from_detector_box(640, 480, &detector_box, crop_config(1.35)).unwrap();
    let source = (0.31, 0.57);

    let crop = transform.source_normalized_to_crop_pixels(source.0, source.1);
    let round_trip = transform.crop_pixels_to_source_normalized(crop.0, crop.1);

    assert!((round_trip.0 - source.0).abs() < 1e-6);
    assert!((round_trip.1 - source.1).abs() < 1e-6);
}

#[test]
fn face_crop_maps_normalized_and_pixel_landmarks_to_source() {
    let detector_box = NormalizedRect {
        x: 0.25,
        y: 0.25,
        width: 0.25,
        height: 0.25,
        rotation_rad: 0.0,
    };
    let transform =
        FaceCropTransform::from_detector_box(640, 480, &detector_box, crop_config(1.35)).unwrap();

    let normalized = transform.landmark_model_to_source_normalized(
        0.5,
        0.5,
        LandmarkCoordinateEncoding::Normalized0To1,
    );
    let pixels = transform.landmark_model_to_source_normalized(
        128.0,
        128.0,
        LandmarkCoordinateEncoding::CropPixels,
    );
    assert!((normalized.0 - pixels.0).abs() < 1e-6);
    assert!((normalized.1 - pixels.1).abs() < 1e-6);

    let mut landmarks = vec![
        Landmark3 {
            x: 0.5,
            y: 0.5,
            z: 0.25,
            visibility: 0.9,
        };
        98
    ];
    transform
        .map_landmarks_to_source_normalized(
            &mut landmarks,
            LandmarkCoordinateEncoding::Normalized0To1,
        )
        .unwrap();
    assert_eq!(landmarks.len(), 98);
    assert_eq!(landmarks[0].z, 0.25);
    assert_eq!(landmarks[0].visibility, 0.9);
    assert!((landmarks[0].x - normalized.0).abs() < 1e-6);
}

#[test]
fn face_crop_preprocess_pads_outside_with_normalized_zero_and_uses_nchw() {
    let detector_box = NormalizedRect {
        x: 0.0,
        y: 0.0,
        width: 0.25,
        height: 0.25,
        rotation_rad: 0.0,
    };
    let config = crop_config(2.0);
    let transform = FaceCropTransform::from_detector_box(4, 4, &detector_box, config).unwrap();
    let input = peppa_input();
    let mut buffers = FaceCropPreprocessBuffers::new(config.output_size).unwrap();
    assert_eq!(buffers.tensor_shape(), [1, 3, 256, 256]);
    let tensor = buffers
        .preprocess(&frame(4, 4, [255, 0, 0]), &transform, &input, config)
        .unwrap();

    assert!(tensor.iter().all(|value| value.is_finite()));
    assert!(tensor.iter().any(|value| value.abs() < 1e-6));
    assert!(tensor.iter().any(|value| *value > 1.0));
}

#[test]
fn face_crop_config_change_changes_geometry_and_tensor_contract_is_checked() {
    let detector_box = NormalizedRect {
        x: 0.3,
        y: 0.3,
        width: 0.2,
        height: 0.2,
        rotation_rad: 0.0,
    };
    let first =
        FaceCropTransform::from_detector_box(640, 480, &detector_box, crop_config(1.35)).unwrap();
    let second =
        FaceCropTransform::from_detector_box(640, 480, &detector_box, crop_config(1.5)).unwrap();
    assert_ne!(first.source_pixel_square(), second.source_pixel_square());

    let mut bad_input = peppa_input();
    bad_input.layout = TensorLayout::Nhwc;
    let config = crop_config(1.35);
    let mut buffers = FaceCropPreprocessBuffers::new(config.output_size).unwrap();
    let error = buffers.preprocess(&frame(640, 480, [0, 0, 0]), &first, &bad_input, config);
    assert!(error.is_err());
}
