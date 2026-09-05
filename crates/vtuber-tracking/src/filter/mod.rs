//! Tracking filters: quaternion-centered rotation smoothing and expression
//! normalization / smoothing.

pub mod detailed;
pub mod expression;
pub mod gaze;
pub mod head;
pub mod translation;

pub use detailed::DetailedExpressionFilter;
pub use expression::{
    ExpressionCalibration, ExpressionCalibrationError, ExpressionChannel, ExpressionFilter,
    ExpressionFilterParams, ExpressionRange, MissingChannelFallback, MissingChannelPolicy,
};
pub use gaze::{
    DEFAULT_RETURN_HALF_LIFE, DEFAULT_TRACKED_HALF_LIFE, DEFAULT_UNAVAILABLE_HOLD, GazeFilter,
    GazeFilterParams,
};
pub use head::{HeadFilterParams, HeadRotationFilter};
pub use translation::{TranslationFilter, TranslationFilterParams};
