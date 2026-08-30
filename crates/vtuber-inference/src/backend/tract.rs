//! Tract-based runtime construction for inference workers.
//!
//! This module performs manifest verification, model loading, optimization,
//! and runnable conversion inside the worker thread. The runnable object is
//! owned by the worker and never moved to the controller thread.

use std::path::Path;
use std::sync::Arc;

use tract_tflite::prelude::{Framework, IntoRunnable, TypedRunnableModel};
use vtuber_core::types::{LandmarkSchemaId, RawFaceObservation};

use crate::descriptor::{ModelDescriptor, ModelFormat};
use crate::error::{InferenceError, Result};
use crate::runtime::FaceInference;

/// Loads and verifies a face model described by `descriptor` and returns a
/// runnable runtime object.
///
/// The returned object is owned by the worker thread; it is constructed here
/// and never moved to the caller thread.
pub fn load_model_runtime(descriptor: &ModelDescriptor) -> Result<Box<dyn FaceInference>> {
    verify_model_file(&descriptor.path, &descriptor.sha256)?;

    match descriptor.format {
        ModelFormat::Tflite => Ok(Box::new(TfliteRuntime::new(descriptor)?)),
        #[cfg(feature = "onnx")]
        ModelFormat::Onnx => Ok(Box::new(load_onnx(descriptor)?)),
    }
}

/// Verifies a model file against the SHA-256 recorded in its manifest.
pub fn verify_model_file(path: &Path, expected_sha256: &str) -> Result<()> {
    let bytes = std::fs::read(path)
        .map_err(|e| InferenceError::LoadFailed(format!("failed to read model file: {e}")))?;
    let actual = sha256_hex(&bytes);
    if actual.eq_ignore_ascii_case(expected_sha256) {
        Ok(())
    } else {
        Err(InferenceError::HashMismatch {
            expected: expected_sha256.to_owned(),
            actual,
        })
    }
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

#[cfg(feature = "onnx")]
fn load_onnx(descriptor: &ModelDescriptor) -> Result<crate::runtime::OnnxRuntime> {
    crate::runtime::OnnxRuntime::new(&descriptor.path, descriptor.schema)
}

/// TFLite face inference runtime owned by the worker thread.
struct TfliteRuntime {
    /// The runnable tract model. Kept alive so that resources are dropped in
    /// the worker thread when this runtime is dropped.
    #[allow(dead_code)]
    model: Arc<TypedRunnableModel>,
    schema: LandmarkSchemaId,
}

impl TfliteRuntime {
    fn new(descriptor: &ModelDescriptor) -> Result<Self> {
        let bytes = std::fs::read(&descriptor.path)
            .map_err(|e| InferenceError::LoadFailed(format!("failed to read model file: {e}")))?;
        let model = tract_tflite::tflite()
            .model_for_read(&mut &bytes[..])
            .map_err(|e| InferenceError::LoadFailed(format!("{e:?}")))?
            .into_optimized()
            .map_err(|e| InferenceError::OptimizationFailed(format!("{e:?}")))?
            .into_runnable()
            .map_err(|e| InferenceError::OptimizationFailed(format!("{e:?}")))?;

        Ok(Self {
            model,
            schema: descriptor.schema,
        })
    }
}

impl FaceInference for TfliteRuntime {
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn infer(&self, tensor: &[f32], input_shape: &[usize; 4]) -> Result<RawFaceObservation> {
        use tract_tflite::prelude::*;
        use vtuber_core::types::{FrameSeq, MonoTimeNs, NormalizedRect};

        let input = tract_ndarray::Array::from_shape_vec(
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ),
            tensor.to_vec(),
        )
        .map_err(|e| InferenceError::ExecutionFailed(format!("invalid input tensor: {e}")))?;
        let result = self
            .model
            .run(tvec!(input.into_tensor().into_tvalue()))
            .map_err(|e| InferenceError::ExecutionFailed(format!("{e:?}")))?;
        let output = result
            .first()
            .ok_or_else(|| InferenceError::ExecutionFailed("model returned no outputs".into()))?
            .to_plain_array_view::<f32>()
            .map_err(|e| InferenceError::ExecutionFailed(format!("output is not f32: {e:?}")))?;
        let shape = output.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 3 {
            return Err(InferenceError::OutputShapeMismatch {
                expected: vec![1, 98, 3],
                actual: shape.to_vec(),
            });
        }
        let count = shape[1];
        let mut landmarks = Vec::with_capacity(count);
        for index in 0..count {
            let x = output[[0, index, 0]];
            let y = output[[0, index, 1]];
            let confidence = output[[0, index, 2]].clamp(0.0, 1.0);
            if !x.is_finite() || !y.is_finite() || !confidence.is_finite() {
                return Err(InferenceError::InvalidOutputValue {
                    index,
                    value: if !x.is_finite() {
                        x
                    } else if !y.is_finite() {
                        y
                    } else {
                        confidence
                    },
                });
            }
            landmarks.push(vtuber_core::types::Landmark3 {
                x,
                y,
                z: 0.0,
                visibility: confidence,
            });
        }
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
