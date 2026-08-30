// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration test for the worker-owned UltraFace runtime boundary.

#![cfg(feature = "legacy-face-stack")]

use std::path::PathBuf;
use std::sync::Arc;

use vtuber_core::types::{FrameSeq, MonoTimeNs, PixelFormat, VideoFrame};
use vtuber_inference::detector::{UltraFaceDetector, UltraFacePreprocessBuffers};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("assets")
        .join("models")
        .join("version-RFB-320.onnx")
}

#[test]
#[ignore = "requires the downloaded legacy UltraFace ONNX research artifact"]
fn detector_runtime_ultraface_runnable_receives_tensor_and_returns_raw_outputs() {
    let detector = UltraFaceDetector::from_path(model_path())
        .expect("the exact supplied UltraFace artifact should be runnable");
    let frame = VideoFrame {
        seq: FrameSeq(1),
        captured_at: MonoTimeNs(2),
        width: 320,
        height: 240,
        stride_bytes: 320 * 3,
        format: PixelFormat::Rgb8,
        data: Arc::<[u8]>::from(vec![127; 320 * 240 * 3]),
    };
    let mut buffers = UltraFacePreprocessBuffers::new();

    let outputs = detector
        .infer(&mut buffers, &frame)
        .expect("the fixed tensor should execute in tract-onnx");
    assert_eq!(outputs.tensors.len(), 2);
    assert_eq!(outputs.tensors[0].name, "scores");
    assert_eq!(outputs.tensors[0].shape, [1, 4420, 2]);
    assert_eq!(outputs.tensors[0].values.len(), 8840);
    assert_eq!(outputs.tensors[1].name, "boxes");
    assert_eq!(outputs.tensors[1].shape, [1, 4420, 4]);
    assert_eq!(outputs.tensors[1].values.len(), 17680);
    assert!(
        outputs
            .tensors
            .iter()
            .flat_map(|tensor| tensor.values.iter())
            .all(|value| value.is_finite())
    );
}
