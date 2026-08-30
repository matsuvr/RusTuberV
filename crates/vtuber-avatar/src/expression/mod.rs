//! Expression command building and application.
//!
//! This module handles the conversion from tracking data (blink, mouth, gaze)
//! into VRM expression weight commands. The builder is pure and testable
//! without Bevy ECS.

pub mod blink;
pub mod command;
pub mod mouth;
pub mod system;

pub use blink::{RawBlinkInput, map_blink_to_expressions, map_blink_with_fallback};
pub use command::{
    ExpressionCommand, ExpressionCommandBuilder, ExpressionCommandMetrics,
    build_detailed_face_commands, build_detailed_face_commands_for_authority, build_face_commands,
};
pub use mouth::{
    RawMouthInput, is_valid_mouth_preset, map_mouth_to_expressions, map_mouth_with_fallback,
};
pub use system::{
    DEFAULT_CHANGE_EPSILON, ExpressionStateTracker, apply_tracked_expressions, coalesce_commands,
};
