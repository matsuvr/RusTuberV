//! Detector-specific preprocessing and runtime APIs.

/// Decode and post-process UltraFace outputs.
#[cfg(feature = "onnx")]
pub mod decode;
/// Hard non-maximum suppression for detector boxes.
pub mod nms;
/// UltraFace input preprocessing with reusable worker-owned buffers.
pub mod preprocess;
/// UltraFace ONNX runtime and validated raw detector outputs.
#[cfg(feature = "onnx")]
pub mod runtime;

#[cfg(feature = "onnx")]
pub use decode::{
    DetectorDecodeError, DetectorDecodeOutcome, decode_detections, select_primary_face,
};
pub use nms::{FaceDetection, hard_nms, intersection_over_union};
pub use preprocess::{
    DetectorNormalization, DetectorPreprocessError, ULTRAFACE_INPUT_HEIGHT, ULTRAFACE_INPUT_WIDTH,
    UltraFacePreprocessBuffers,
};
#[cfg(feature = "onnx")]
pub use runtime::{DetectorRawOutputs, DetectorRawTensor, DetectorRuntimeError, UltraFaceDetector};
