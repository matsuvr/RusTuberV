//! Typed errors for the inference subsystem.

use thiserror::Error;

/// Errors that can occur during inference model loading or execution.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum InferenceError {
    /// Failure while loading the approved MediaPipe native runtime or task.
    #[error("MediaPipe runtime load failed: {0}")]
    MediaPipeLoadFailed(String),
    /// The captured timestamp could not be represented in MediaPipe VIDEO mode.
    #[error("MediaPipe timestamp is out of range")]
    MediaPipeTimestampOutOfRange,
    /// The captured frame could not be converted to packed RGB pixels.
    #[error("MediaPipe frame conversion failed: {0}")]
    MediaPipeFrameConversion(String),
    /// MediaPipe returned a result outside the canonical face contract.
    #[error("MediaPipe output contract failed: {0}")]
    MediaPipeOutputContract(String),
    /// MediaPipe rejected or failed to process a frame.
    #[error("MediaPipe frame inference failed: {0}")]
    MediaPipeFrameInference(String),
    /// Failure to load the model from disk.
    #[error("model load failed: {0}")]
    LoadFailed(String),
    /// Model hash mismatch.
    #[error("model hash mismatch: expected {expected}, got {actual}")]
    HashMismatch {
        /// Expected SHA-256 digest.
        expected: String,
        /// Actual SHA-256 digest.
        actual: String,
    },
    /// Failure to optimize the model for the runtime.
    #[error("model optimization failed: {0}")]
    OptimizationFailed(String),
    /// Failure during tensor execution.
    #[error("inference execution failed: {0}")]
    ExecutionFailed(String),
    /// The provided video frame was incompatible with the model input.
    #[error("input frame incompatible: {0}")]
    InvalidInput(String),
    /// The tracked face ROI is invalid or out of bounds.
    #[error("invalid face ROI: {0}")]
    InvalidRoi(String),
    /// The video frame stride does not match its width and pixel format.
    #[error("frame stride mismatch: expected at least {expected} bytes, got {actual}")]
    FrameStrideMismatch {
        /// Minimum stride required.
        expected: usize,
        /// Actual stride in bytes.
        actual: usize,
    },
    /// The video frame buffer is too small for the declared resolution.
    #[error("frame buffer too small: expected at least {expected} bytes, got {actual}")]
    FrameBufferTooSmall {
        /// Minimum buffer size required.
        expected: usize,
        /// Actual buffer size.
        actual: usize,
    },
    /// The model input tensor layout is not supported.
    #[error("unsupported input layout: {shape:?}")]
    UnsupportedInputLayout {
        /// Input shape that could not be interpreted.
        shape: Vec<usize>,
    },
    /// The model output tensor shape does not match the manifest contract.
    #[error("output shape mismatch: expected {expected:?}, got {actual:?}")]
    OutputShapeMismatch {
        /// Expected shape from the manifest contract.
        expected: Vec<usize>,
        /// Actual shape of the runtime output tensor.
        actual: Vec<usize>,
    },
    /// The model output tensor element count does not match its declared shape.
    #[error("output element count mismatch: expected {expected}, got {actual}")]
    OutputElementCountMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// The model output tensor data type does not match the manifest contract.
    #[error("output dtype mismatch: expected {expected}, got {actual}")]
    OutputDtypeMismatch {
        /// Expected data type.
        expected: String,
        /// Actual data type.
        actual: String,
    },
    /// The model output tensor contains an invalid numeric value.
    #[error("invalid output value at index {index}: {value}")]
    InvalidOutputValue {
        /// Index of the offending element.
        index: usize,
        /// The invalid value.
        value: f32,
    },
    /// Runtime-specific internal error.
    #[error("internal runtime error: {0}")]
    Internal(String),
    /// Worker is already running and cannot be started again.
    #[error("inference worker already running")]
    AlreadyRunning,
    /// The worker thread panicked.
    #[error("inference worker panicked")]
    WorkerPanicked,
    /// The input channel or frame slot was closed unexpectedly.
    #[error("inference input closed")]
    InputClosed,
}

/// Result type alias for inference operations.
pub type Result<T> = std::result::Result<T, InferenceError>;
