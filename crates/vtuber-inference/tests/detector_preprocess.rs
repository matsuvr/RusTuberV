// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for the UltraFace detector preprocessor.

#![cfg(feature = "legacy-face-stack")]

use std::sync::Arc;

use vtuber_core::types::{FrameSeq, MonoTimeNs, PixelFormat, VideoFrame};
use vtuber_inference::detector::preprocess::{
    DetectorNormalization, DetectorPreprocessError, UltraFacePreprocessBuffers,
};

fn frame(
    width: u32,
    height: u32,
    stride_bytes: usize,
    format: PixelFormat,
    data: Vec<u8>,
) -> VideoFrame {
    VideoFrame {
        seq: FrameSeq(7),
        captured_at: MonoTimeNs(11),
        width,
        height,
        stride_bytes,
        format,
        data: Arc::<[u8]>::from(data),
    }
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1.0e-6,
        "actual={actual} expected={expected}"
    );
}

#[test]
fn detector_preprocess_rgb_stride_normalization_and_nchw_order_are_exact() {
    let source = frame(
        2,
        2,
        8,
        PixelFormat::Rgb8,
        vec![
            1, 2, 3, 10, 20, 30, 200, 201, // row 0 plus padding
            40, 50, 60, 70, 80, 90, 202, 203, // row 1 plus padding
        ],
    );
    let mut buffers =
        UltraFacePreprocessBuffers::with_dimensions(2, 2, DetectorNormalization::default());

    let tensor = buffers
        .preprocess(&source)
        .expect("synthetic RGB frame is valid");
    let expected_pixels = [
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0],
    ];
    for channel in 0..3 {
        for pixel in 0..4 {
            assert_close(
                tensor[channel * 4 + pixel],
                (expected_pixels[pixel][channel] - 127.0) / 128.0,
            );
        }
    }
}

#[test]
fn detector_preprocess_bgr_is_converted_to_rgb_before_resize() {
    let source = frame(2, 1, 6, PixelFormat::Bgr8, vec![3, 2, 1, 30, 20, 10]);
    let mut buffers =
        UltraFacePreprocessBuffers::with_dimensions(2, 1, DetectorNormalization::default());

    let tensor = buffers
        .preprocess(&source)
        .expect("synthetic BGR frame is valid");
    assert_close(tensor[0], (1.0 - 127.0) / 128.0);
    assert_close(tensor[1], (10.0 - 127.0) / 128.0);
    assert_close(tensor[2], (2.0 - 127.0) / 128.0);
    assert_close(tensor[3], (20.0 - 127.0) / 128.0);
    assert_close(tensor[4], (3.0 - 127.0) / 128.0);
    assert_close(tensor[5], (30.0 - 127.0) / 128.0);
}

#[test]
fn detector_preprocess_bilinear_resize_preserves_unmirrored_width_and_height_orientation() {
    let source = frame(
        2,
        2,
        6,
        PixelFormat::Rgb8,
        vec![0, 0, 0, 100, 0, 0, 200, 0, 0, 255, 0, 0],
    );
    let mut buffers =
        UltraFacePreprocessBuffers::with_dimensions(4, 4, DetectorNormalization::default());

    let tensor = buffers
        .preprocess(&source)
        .expect("synthetic resize frame is valid");
    let red = &tensor[..16];
    let expected_top = [0.0, 25.0, 75.0, 100.0];
    let expected_bottom = [200.0, 213.75, 241.25, 255.0];
    for (index, expected) in expected_top.into_iter().enumerate() {
        assert_close(red[index], (expected - 127.0) / 128.0);
    }
    for (index, expected) in expected_bottom.into_iter().enumerate() {
        assert_close(red[12 + index], (expected - 127.0) / 128.0);
    }
}

#[test]
fn detector_preprocess_reusable_tensor_storage_keeps_its_allocation() {
    let source = frame(1, 1, 3, PixelFormat::Rgb8, vec![127, 127, 127]);
    let mut buffers =
        UltraFacePreprocessBuffers::with_dimensions(2, 2, DetectorNormalization::default());
    let first_ptr = {
        let tensor = buffers.preprocess(&source).expect("first frame is valid");
        tensor.as_ptr()
    };
    let second_ptr = {
        let tensor = buffers.preprocess(&source).expect("second frame is valid");
        tensor.as_ptr()
    };
    assert_eq!(first_ptr, second_ptr);
}

#[test]
fn detector_preprocess_malformed_frames_and_normalization_are_typed_errors() {
    let zero = frame(0, 2, 0, PixelFormat::Rgb8, Vec::new());
    let mut buffers = UltraFacePreprocessBuffers::new();
    assert!(matches!(
        buffers.preprocess(&zero),
        Err(DetectorPreprocessError::ZeroDimension { .. })
    ));

    let short = frame(2, 2, 6, PixelFormat::Rgb8, vec![0; 6]);
    assert!(matches!(
        buffers.preprocess(&short),
        Err(DetectorPreprocessError::FrameBufferTooSmall { .. })
    ));

    let non_finite = frame(1, 1, 3, PixelFormat::Rgb8, vec![0, 0, 0]);
    let mut invalid_buffers = UltraFacePreprocessBuffers::with_dimensions(
        2,
        2,
        DetectorNormalization {
            mean: [f32::NAN, 127.0, 127.0],
            scale: [128.0; 3],
        },
    );
    assert!(matches!(
        invalid_buffers.preprocess(&non_finite),
        Err(DetectorPreprocessError::NonFiniteNormalization { channel: 0, .. })
    ));
}
