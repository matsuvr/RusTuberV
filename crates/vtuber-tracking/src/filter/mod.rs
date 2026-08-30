//! Tracking filters: quaternion-centered rotation smoothing and expression
//! normalization / smoothing.

pub mod expression;
pub mod gaze;
pub mod head;

pub use expression::{
    ExpressionCalibration, ExpressionCalibrationError, ExpressionChannel, ExpressionFilter,
    ExpressionFilterParams, ExpressionRange, MissingChannelFallback, MissingChannelPolicy,
};
pub use gaze::{
    DEFAULT_RETURN_HALF_LIFE, DEFAULT_TRACKED_HALF_LIFE, DEFAULT_UNAVAILABLE_HOLD, GazeFilter,
    GazeFilterParams,
};
pub use head::{HeadFilterParams, HeadRotationFilter};
