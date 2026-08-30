//! Face inference runtime traits and implementations.

#[cfg(feature = "onnx")]
use crate::error::InferenceError;
use crate::error::Result;
use std::time::Duration;

use vtuber_core::FaceTrackingOutcome;
use vtuber_core::types::{LandmarkSchemaId, RawFaceObservation, VideoFrame};

/// Result of one production frame-level inference attempt.
///
/// `NoFace` is an ordinary observation and must not be counted as a runtime
/// failure. A `Face` observation uses unmirrored source-image normalized
/// coordinates. Implementations own all live model state in the inference
/// worker thread; this contract moves only borrowed frame data across the
/// call boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum FrameInferenceOutcome {
    /// A face was found and decoded into a source-image observation.
    Face(RawFaceObservation),
    /// No face met the detector and landmark validity policy.
    NoFace,
}

/// Latest-only outcome published by an inference worker.
///
/// Unlike [`FrameInferenceOutcome`], this type carries enough source metadata
/// for the application bridge to explicitly clear the last face observation
/// when a frame has no face. It is intentionally bounded by the same
/// capacity-one slot as face observations.
#[derive(Clone, Debug, PartialEq)]
pub enum InferenceOutcome {
    /// A validated face observation.
    Face(RawFaceObservation),
    /// A frame was processed but no valid face was found.
    NoFace {
        /// Source frame sequence that produced the result.
        source_seq: vtuber_core::types::FrameSeq,
        /// Source capture timestamp.
        captured_at: vtuber_core::types::MonoTimeNs,
        /// Monotonic completion timestamp.
        inference_finished_at: vtuber_core::types::MonoTimeNs,
    },
}

/// Per-frame timing reported by a composite runtime.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameInferenceTiming {
    /// Detector preprocessing and execution time, when the detector ran.
    pub detector: Option<Duration>,
    /// Face crop preprocessing time, when a crop was available.
    pub crop: Option<Duration>,
    /// Landmark model execution time, when a crop was available.
    pub landmark: Option<Duration>,
    /// Landmark validation, mapping, and observation construction time.
    pub decode: Option<Duration>,
    /// Total time measured by the runtime itself.
    pub total: Duration,
    /// Detector confidence associated with the active ROI, if known.
    pub detector_confidence: Option<f32>,
    /// Human-readable ROI lifecycle state after the frame attempt.
    pub roi_state: Option<String>,
}

/// Frame-level production inference boundary.
///
/// The worker owns the implementing value and its model runtimes. The caller
/// supplies a borrowed [`VideoFrame`]; detector-specific tensors and runtime
/// values remain private to the inference crate.
pub trait FrameFaceInference: Send {
    /// Runs the complete detector-to-landmark pipeline for one frame.
    fn infer_frame(&mut self, frame: &VideoFrame) -> Result<FrameInferenceOutcome>;

    /// Takes the timing for the most recent frame attempt.
    ///
    /// Implementations that do not expose stage timings return the default;
    /// workers still record their own end-to-end duration.
    fn take_timing(&mut self) -> FrameInferenceTiming {
        FrameInferenceTiming::default()
    }
}

/// Canonical MediaPipe face-tracking boundary owned by the inference worker.
///
/// Implementations own all live native/runtime state. The worker passes only a
/// borrowed, engine-independent [`VideoFrame`] and receives a validated
/// [`FaceTrackingOutcome`].
pub trait FaceTrackingInference: Send {
    /// Runs one frame through the canonical face-tracking backend.
    fn infer_face_tracking(&mut self, frame: &VideoFrame) -> Result<FaceTrackingOutcome>;
}

/// Trait for a face inference runtime.
pub trait FaceInference: Send + Sync {
    /// Performs inference on a preprocessed input tensor.
    ///
    /// `tensor` contains normalized float values in the layout described by
    /// `input_shape`. The shape is passed explicitly because the trait object
    /// may be created from descriptors with varying input layouts.
    fn infer(&self, tensor: &[f32], input_shape: &[usize; 4]) -> Result<RawFaceObservation>;
    /// Returns the schema ID used by this runtime.
    fn schema_id(&self) -> LandmarkSchemaId;
}

/// ONNX implementation of the face inference runtime.
#[cfg(feature = "onnx")]
pub struct OnnxRuntime {
    model: std::sync::Arc<tract_core::prelude::TypedRunnableModel>,
    schema: LandmarkSchemaId,
}

#[cfg(feature = "onnx")]
impl OnnxRuntime {
    /// Constructs a new OnnxRuntime from a model file.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be read, optimized, or made runnable.
    pub fn new(
        path: impl AsRef<std::path::Path>,
        schema: LandmarkSchemaId,
    ) -> crate::error::Result<Self> {
        use tract_onnx::prelude::*;

        let mut file = std::fs::File::open(path.as_ref())
            .map_err(|e| InferenceError::LoadFailed(e.to_string()))?;
        let model = tract_onnx::onnx()
            .model_for_read(&mut file)
            .map_err(|e| InferenceError::LoadFailed(format!("{e:?}")))?
            .into_optimized()
            .map_err(|e| InferenceError::OptimizationFailed(format!("{e:?}")))?
            .into_runnable()
            .map_err(|e| InferenceError::OptimizationFailed(format!("{e:?}")))?;

        Ok(Self { model, schema })
    }

    /// Runs the landmark model and decodes its exact `[1, 98, 3]` output.
    pub fn infer_landmarks(
        &self,
        tensor: &[f32],
        input_shape: &[usize; 4],
    ) -> Result<Vec<vtuber_core::types::Landmark3>> {
        use tract_onnx::prelude::*;

        let input_array = tensors_to_tract(tensor, input_shape)
            .map_err(|e| InferenceError::ExecutionFailed(format!("invalid tensor shape: {e:?}")))?;

        let result = self
            .model
            .run(tvec!(input_array.into_tvalue()))
            .map_err(|e| InferenceError::ExecutionFailed(format!("{e:?}")))?;

        let output = result
            .first()
            .ok_or_else(|| InferenceError::OutputShapeMismatch {
                expected: vec![1, 98, 3],
                actual: Vec::new(),
            })?
            .to_plain_array_view::<f32>()
            .map_err(|e| InferenceError::ExecutionFailed(format!("{e:?}")))?;
        let actual = output.shape().to_vec();
        if actual != [1, 98, 3] {
            return Err(InferenceError::OutputShapeMismatch {
                expected: vec![1, 98, 3],
                actual,
            });
        }
        for (index, value) in output.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(InferenceError::InvalidOutputValue { index, value });
            }
        }

        decode_landmarks(output)
    }
}

#[cfg(feature = "onnx")]
impl FaceInference for OnnxRuntime {
    fn infer(&self, tensor: &[f32], input_shape: &[usize; 4]) -> Result<RawFaceObservation> {
        use vtuber_core::types::{FrameSeq, MonoTimeNs, NormalizedRect};

        let landmarks = self.infer_landmarks(tensor, input_shape)?;
        let expressions = crate::decode::expressions::decode_expressions(
            None,
            None,
            Some(&landmarks),
            self.schema,
            1.0,
        )
        .ok()
        .flatten()
        .unwrap_or_default();

        Ok(RawFaceObservation {
            source_seq: FrameSeq(0),
            captured_at: MonoTimeNs(0),
            inference_started_at: MonoTimeNs(0),
            inference_finished_at: MonoTimeNs(0),
            face_confidence: 1.0,
            landmarks,
            blendshapes: None,
            expressions,
            roi: NormalizedRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                rotation_rad: 0.0,
            },
            schema: self.schema,
        })
    }

    fn schema_id(&self) -> LandmarkSchemaId {
        self.schema
    }
}

#[cfg(feature = "onnx")]
fn tensors_to_tract(
    data: &[f32],
    input_shape: &[usize; 4],
) -> std::result::Result<tract_core::ndarray::Array4<f32>, tract_core::ndarray::ShapeError> {
    tract_core::ndarray::Array::from_shape_vec(
        (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ),
        data.to_vec(),
    )
}

#[cfg(feature = "onnx")]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn decode_landmarks(
    output: tract_core::ndarray::ArrayViewD<f32>,
) -> Result<Vec<vtuber_core::types::Landmark3>> {
    use vtuber_core::types::Landmark3;

    if output.shape() != [1, 98, 3] {
        return Err(InferenceError::OutputShapeMismatch {
            expected: vec![1, 98, 3],
            actual: output.shape().to_vec(),
        });
    }
    let mut landmarks = Vec::with_capacity(98);
    for i in 0..98 {
        let x = output[[0, i, 0]];
        let y = output[[0, i, 1]];
        let conf = output[[0, i, 2]];
        landmarks.push(Landmark3 {
            x,
            y,
            z: 0.0,
            visibility: conf.clamp(0.0, 1.0),
        });
    }
    Ok(landmarks)
}

#[cfg(all(test, feature = "legacy-face-stack"))]
mod tests {
    use super::decode_landmarks;

    #[test]
    fn landmark_runtime_clamps_model_visibility_to_contract() {
        let mut values = vec![0.5_f32; 98 * 3];
        values[2] = 1.25;
        values[5] = -0.25;
        let output = tract_core::ndarray::Array::from_shape_vec((1, 98, 3), values)
            .expect("test output shape is valid")
            .into_dyn();

        let landmarks = decode_landmarks(output.view()).expect("test output decodes");

        assert_eq!(landmarks[0].visibility, 1.0);
        assert_eq!(landmarks[1].visibility, 0.0);
    }
}
