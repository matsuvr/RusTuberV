//! Typed UltraFace detector runtime.

use std::path::Path;
use std::sync::Arc;

use thiserror::Error;
use tract_onnx::prelude::*;
use vtuber_core::types::VideoFrame;

use crate::detector::preprocess::{
    DetectorPreprocessError, ULTRAFACE_INPUT_HEIGHT, ULTRAFACE_INPUT_WIDTH,
    UltraFacePreprocessBuffers,
};
use crate::probe::{OnnxProbeError, build_ultraface_runnable};

const OUTPUT_NAMES: [&str; 2] = ["scores", "boxes"];
const OUTPUT_SHAPES: [[usize; 3]; 2] = [[1, 4420, 2], [1, 4420, 4]];

/// One validated raw output tensor from the detector stage.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorRawTensor {
    /// Manifest output name.
    pub name: String,
    /// Model output shape.
    pub shape: Vec<usize>,
    /// Raw F32 values in the model's output layout.
    pub values: Vec<f32>,
}

/// Raw detector outputs in the model's stable output order.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorRawOutputs {
    /// Output tensors, ordered as `scores`, then `boxes`.
    pub tensors: Vec<DetectorRawTensor>,
}

/// Typed failures from detector runtime construction or execution.
#[derive(Debug, Error)]
pub enum DetectorRuntimeError {
    /// Input frame could not be converted into the detector tensor.
    #[error("detector preprocessing failed: {0}")]
    Preprocess(#[from] DetectorPreprocessError),
    /// The exact accepted model could not be constructed.
    #[error("detector model construction failed: {0}")]
    ModelConstruction(#[source] OnnxProbeError),
    /// The preprocessed tensor shape does not match the fixed model contract.
    #[error("detector input shape mismatch: actual={actual:?} expected={expected:?}")]
    InputShapeMismatch {
        /// Actual buffer shape.
        actual: [usize; 4],
        /// Required model shape.
        expected: [usize; 4],
    },
    /// The runtime returned a different number of tensors.
    #[error("detector output count mismatch: actual={actual} expected=2")]
    OutputCountMismatch {
        /// Actual number of outputs.
        actual: usize,
    },
    /// A runtime output violates the exact fixed output contract.
    #[error(
        "detector output mismatch at index {index}: name={name} shape={actual_shape:?} expected_name={expected_name} expected_shape={expected_shape:?}"
    )]
    OutputContractMismatch {
        /// Output index.
        index: usize,
        /// Runtime-assigned output name.
        name: String,
        /// Actual runtime output shape.
        actual_shape: Vec<usize>,
        /// Expected manifest output name.
        expected_name: &'static str,
        /// Expected manifest output shape.
        expected_shape: [usize; 3],
    },
    /// The runtime failed while creating or executing an input tensor.
    #[error("detector execution failed: {detail}")]
    Execution {
        /// Technical tract error.
        detail: String,
    },
    /// A runtime output contained a non-finite value.
    #[error("detector output {name} contains non-finite values")]
    NonFiniteOutput {
        /// Output name.
        name: &'static str,
    },
}

/// Worker-owned UltraFace detector runnable.
///
/// Construct and drop this value in the inference worker. The application
/// orchestration layer should pass a worker-owned instance rather than create
/// one on the Bevy/main thread.
pub struct UltraFaceDetector {
    runnable: Arc<TypedRunnableModel>,
}

impl UltraFaceDetector {
    /// Load, optimize, and make the exact manifest UltraFace artifact runnable.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, DetectorRuntimeError> {
        let runnable =
            build_ultraface_runnable(path).map_err(DetectorRuntimeError::ModelConstruction)?;
        Ok(Self { runnable })
    }

    /// Run detector preprocessing and inference, returning validated raw tensors.
    pub fn infer(
        &self,
        buffers: &mut UltraFacePreprocessBuffers,
        frame: &VideoFrame,
    ) -> Result<DetectorRawOutputs, DetectorRuntimeError> {
        let actual_shape = buffers.shape();
        let expected_shape = [1, 3, ULTRAFACE_INPUT_HEIGHT, ULTRAFACE_INPUT_WIDTH];
        if actual_shape != expected_shape {
            return Err(DetectorRuntimeError::InputShapeMismatch {
                actual: actual_shape,
                expected: expected_shape,
            });
        }
        let tensor = buffers.preprocess(frame)?;
        let input = Tensor::from_shape(&expected_shape, tensor).map_err(|error| {
            DetectorRuntimeError::Execution {
                detail: format!("input tensor: {error:?}"),
            }
        })?;
        let values = self.runnable.run(tvec![input.into()]).map_err(|error| {
            DetectorRuntimeError::Execution {
                detail: format!("runnable: {error:?}"),
            }
        })?;
        raw_outputs(values)
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn raw_outputs(values: TVec<TValue>) -> Result<DetectorRawOutputs, DetectorRuntimeError> {
    if values.len() != OUTPUT_NAMES.len() {
        return Err(DetectorRuntimeError::OutputCountMismatch {
            actual: values.len(),
        });
    }
    let mut tensors = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let shape = value.shape().to_vec();
        if shape != OUTPUT_SHAPES[index] {
            return Err(DetectorRuntimeError::OutputContractMismatch {
                index,
                name: OUTPUT_NAMES[index].to_string(),
                actual_shape: shape,
                expected_name: OUTPUT_NAMES[index],
                expected_shape: OUTPUT_SHAPES[index],
            });
        }
        let array = value.to_plain_array_view::<f32>().map_err(|error| {
            DetectorRuntimeError::Execution {
                detail: format!("output index={index}: {error:?}"),
            }
        })?;
        let output_values = array.iter().copied().collect::<Vec<_>>();
        if output_values.iter().any(|value| !value.is_finite()) {
            return Err(DetectorRuntimeError::NonFiniteOutput {
                name: OUTPUT_NAMES[index],
            });
        }
        tensors.push(DetectorRawTensor {
            name: OUTPUT_NAMES[index].to_string(),
            shape,
            values: output_values,
        });
    }
    Ok(DetectorRawOutputs { tensors })
}
