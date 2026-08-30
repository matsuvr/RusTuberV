//! Model descriptor and runtime settings for inference workers.
//!
//! A [`ModelDescriptor`] is a plain, sendable description of a face model that
//! can cross thread boundaries. The actual runtime object is constructed from
//! the descriptor inside the inference worker, so controller code never owns
//! a live runtime instance.

use std::path::PathBuf;

use vtuber_core::types::LandmarkSchemaId;

/// Role of an artifact in the production face pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelRole {
    /// Full-frame face detector.
    FaceDetector,
    /// Landmark model that consumes a face crop.
    FaceLandmarks,
}

/// Tensor layout recorded in the model manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    /// Batch, channel, height, width.
    Nchw,
    /// Batch, height, width, channel.
    Nhwc,
}

/// Value domain before the manifest normalization is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputValueDomain {
    /// Unsigned 8-bit image samples represented as float tensor values.
    RawU8,
    /// Floating-point image samples in the unit interval.
    UnitFloat,
}

/// Per-channel normalization contract from the model manifest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizationContract {
    /// Value subtracted from each channel.
    pub mean: [f32; 3],
    /// Value dividing each channel after the mean is subtracted.
    pub scale: [f32; 3],
}

/// Input tensor contract for a model artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorContract {
    /// Tensor dimensions in the model's native order.
    pub shape: Vec<usize>,
    /// Tensor element type as named by the manifest.
    pub dtype: String,
    /// Tensor memory layout.
    pub layout: TensorLayout,
    /// Image channel order, when the tensor represents an image.
    pub channel_order: ChannelOrder,
    /// Value domain before normalization.
    pub value_domain: InputValueDomain,
    /// Per-channel normalization.
    pub normalization: NormalizationContract,
}

/// One named output tensor contract from a model artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputTensorContract {
    /// Runtime output name.
    pub name: String,
    /// Output tensor dimensions.
    pub shape: Vec<usize>,
    /// Tensor element type as named by the manifest.
    pub dtype: String,
    /// Human-readable meaning of the output.
    pub description: String,
}

/// Provenance and tensor contract for one pipeline model artifact.
///
/// The file path is relative to the manifest directory. This type intentionally
/// contains no backend or live runtime object and is safe to move to a worker.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelArtifactDescriptor {
    /// Stable manifest ID.
    pub id: String,
    /// Pipeline role.
    pub role: ModelRole,
    /// Artifact path relative to the manifest directory.
    pub file: std::path::PathBuf,
    /// Exact expected file size in bytes.
    pub byte_size: u64,
    /// Exact expected SHA-256 digest, encoded as hexadecimal.
    pub sha256: String,
    /// Input tensor name.
    pub input_name: String,
    /// Authoritative download or source location.
    pub source: String,
    /// Upstream project location.
    pub upstream: String,
    /// Artifact license identifier.
    pub license: String,
    /// Optional direct license URL.
    pub license_url: Option<String>,
    /// Input tensor contract.
    pub input: TensorContract,
    /// Named output tensor contracts.
    pub outputs: Vec<OutputTensorContract>,
    /// Whether the model requires a detector-provided face crop.
    pub requires_crop: bool,
    /// Landmark schema produced by the artifact, when applicable.
    pub schema: Option<String>,
    /// Landmark output coordinate encoding, when applicable.
    pub landmark_coordinate_encoding: Option<String>,
    /// Pose method, when applicable.
    pub pose_method: Option<String>,
    /// Representative landmark indices used by the pose adapter.
    pub representative_indices: Vec<usize>,
}

/// Detector post-processing parameters from the production pipeline manifest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectorPostprocessConfig {
    /// Minimum face score.
    pub score_threshold: f32,
    /// IoU threshold for hard NMS.
    pub nms_iou: f32,
    /// Maximum candidates retained before NMS.
    pub max_pre_nms_candidates: usize,
    /// Maximum detections retained after NMS.
    pub max_post_nms_detections: usize,
}

/// Crop interpolation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CropInterpolation {
    /// Bilinear interpolation.
    Bilinear,
}

/// Fill policy for pixels outside the source image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CropOutsideFill {
    /// Fill with the landmark model's normalization mean.
    NormalizationMean,
}

/// Detector-to-landmark crop contract from the production manifest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceCropConfig {
    /// Square crop side length relative to the detector box side.
    pub square_scale: f32,
    /// Vertical center offset as a fraction of detector box height.
    pub center_y_offset_fraction: f32,
    /// Crop output dimensions in pixels as `[width, height]`.
    pub output_size: [usize; 2],
    /// Crop interpolation mode.
    pub interpolation: CropInterpolation,
    /// Outside-image fill policy.
    pub outside_fill: CropOutsideFill,
}

/// Fully resolved, runtime-free production detector plus landmark pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct FacePipelineDescriptor {
    /// Stable pipeline ID.
    pub id: String,
    /// Resolved full-frame detector artifact.
    pub detector: ModelArtifactDescriptor,
    /// Resolved crop-based landmark artifact.
    pub landmarks: ModelArtifactDescriptor,
    /// Detector post-processing contract.
    pub detector_postprocess: DetectorPostprocessConfig,
    /// Detector-to-landmark crop contract.
    pub crop: FaceCropConfig,
}

/// Manifest mapping from backend blendshape names to canonical expressions.
///
/// A model may export coefficients under different naming conventions.  Each
/// canonical expression stores a list of candidate names; the decoder uses
/// the first match it finds in the runtime output.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExpressionMapping {
    /// Candidate names for the left eye blink coefficient.
    pub blink_left: Vec<String>,
    /// Candidate names for the right eye blink coefficient.
    pub blink_right: Vec<String>,
    /// Candidate names for the mouth openness coefficient.
    pub mouth_open: Vec<String>,
}

/// Supported model file formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// TensorFlow Lite flatbuffer.
    Tflite,
    /// ONNX.
    #[cfg(feature = "onnx")]
    Onnx,
}

/// Describes a face model to load inside the inference worker.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDescriptor {
    /// Human-readable model identifier.
    pub id: String,
    /// Model format.
    pub format: ModelFormat,
    /// Absolute path to the model file.
    pub path: PathBuf,
    /// Expected SHA-256 hex digest of the model file.
    pub sha256: String,
    /// Input tensor name.
    pub input_name: String,
    /// Input tensor shape in the model's native order.
    pub input_shape: Vec<usize>,
    /// Input data type, e.g. `f32` or `u8`.
    pub input_dtype: String,
    /// Channel order of the input image.
    pub channel_order: ChannelOrder,
    /// Normalization applied after converting to float.
    pub normalization: Normalization,
    /// Landmark schema produced by this model.
    pub schema: LandmarkSchemaId,
    /// Optional blendshape-to-expression mapping for models that export
    /// expression coefficients as named blendshape output.
    pub expression_mapping: Option<ExpressionMapping>,
}

/// Channel order of the input image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChannelOrder {
    /// Red, green, blue.
    Rgb,
    /// Red, green, blue, alpha.
    Rgba,
    /// Blue, green, red.
    Bgr,
    /// Blue, green, red, alpha.
    Bgra,
    /// Single luminance channel.
    Gray,
}

/// Normalization applied to input pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Normalization {
    /// Scale `[0, 255]` to `[0, 1]`.
    ZeroToOne,
    /// Scale `[0, 255]` to `[-1, 1]`.
    MinusOneToOne,
    /// Subtract mean and divide by standard deviation per channel.
    MeanStd {
        /// Per-channel mean.
        mean: [f32; 3],
        /// Per-channel standard deviation.
        std: [f32; 3],
    },
}

/// Runtime settings that control worker behavior without owning runtime state.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeSettings {
    /// Maximum time to wait for a new frame before polling the stop token.
    pub frame_wait_timeout_ms: u64,
    /// Interval between face detector runs while tracking is active.
    pub detector_interval_frames: u32,
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            frame_wait_timeout_ms: 100,
            detector_interval_frames: 5,
        }
    }
}

impl ModelDescriptor {
    /// Returns a short display name for diagnostics.
    #[must_use]
    pub fn display_name(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_settings_default() {
        let settings = RuntimeSettings::default();
        assert_eq!(settings.frame_wait_timeout_ms, 100);
        assert_eq!(settings.detector_interval_frames, 5);
    }

    #[test]
    fn descriptor_display_name() {
        let desc = ModelDescriptor {
            id: "test-model".into(),
            format: ModelFormat::Tflite,
            path: PathBuf::from("/tmp/model.tflite"),
            sha256: "abcd".into(),
            input_name: "input".into(),
            input_shape: vec![1, 256, 256, 3],
            input_dtype: "f32".into(),
            channel_order: ChannelOrder::Rgb,
            normalization: Normalization::ZeroToOne,
            schema: LandmarkSchemaId("test-schema"),
            expression_mapping: None,
        };
        assert_eq!(desc.display_name(), "test-model");
    }
}
