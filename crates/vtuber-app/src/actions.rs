//! UI actions — commands emitted by the UI layer.
//!
//! The UI reads [`crate::ui_model::UiViewModel`] snapshots and emits these
//! actions. The orchestrator processes them and updates domain state.
//! The UI never calls camera, filesystem, or VRM APIs directly.

use std::path::PathBuf;

use vtuber_avatar::ArmPoseProfileOverride;

/// Actions that the UI can emit.
///
/// These are processed by the orchestrator, which translates them into
/// domain service calls. The UI layer should only construct these values
/// and send them — it should not perform the actual operations.
#[derive(Clone, Debug, PartialEq)]
pub enum UiAction {
    // --- Screen navigation ---
    /// Switch to a different screen.
    SwitchScreen(crate::ui_model::Screen),

    // --- Camera actions ---
    /// Refresh the list of available cameras.
    RefreshCameras,
    /// Select a camera by index.
    SelectCamera {
        /// Camera index from the available list.
        index: usize,
    },
    /// Restore the current avatar's last successful auto-framed camera pose.
    ResetAvatarCamera,

    // --- Avatar actions ---
    /// Import a VRM model from the given path.
    ImportAvatar {
        /// Path to the VRM file.
        path: PathBuf,
    },
    /// Unload the current avatar.
    UnloadAvatar,

    // --- Lifecycle actions ---
    /// Start all workers (capture → inference → tracking).
    Start,
    /// Stop all workers in reverse order.
    Stop,

    // --- NDI output actions ---
    /// Start the optional transparent avatar NDI output.
    StartNdiOutput,
    /// Stop the optional transparent avatar NDI output.
    StopNdiOutput,

    // --- Calibration actions ---
    /// Begin calibration sequence.
    BeginCalibration,
    /// Cancel in-progress calibration.
    CancelCalibration,
    /// Retry calibration after failure.
    RetryCalibration,

    // --- Preview actions ---
    /// Toggle preview mirroring.
    ToggleMirror,
    /// Toggle preview visibility.
    TogglePreview,
    /// Toggle mirror-style avatar motion.
    ToggleAvatarMotionMirror,

    // --- Avatar pose settings ---
    /// Store a bounded per-model default-arm profile and re-resolve it.
    SetArmPoseProfile {
        /// The six validated profile parameters edited by the settings UI.
        profile: ArmPoseProfileOverride,
    },
    /// Remove the active model's override and return to geometry-derived pose.
    ResetArmPoseProfile,

    // --- Face tracking backend ---
    /// Select the face tracking backend (Direct, GNM Temporal, GNM Shadow).
    SetFaceTrackingBackend(crate::face_backend::FaceTrackingBackendSelection),

    // --- Error actions ---
    /// Dismiss the current error (does not clear domain failure state).
    DismissError,
    /// Retry after a recoverable error.
    RetryAfterError,
}

impl UiAction {
    /// Check if this action requires a running pipeline.
    #[must_use]
    pub fn requires_running_pipeline(&self) -> bool {
        matches!(
            self,
            UiAction::BeginCalibration | UiAction::CancelCalibration | UiAction::RetryCalibration
        )
    }

    /// Check if this action is a screen navigation action.
    #[must_use]
    pub fn is_navigation(&self) -> bool {
        matches!(self, UiAction::SwitchScreen(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui_model::Screen;

    #[test]
    fn actions_start_does_not_require_pipeline() {
        assert!(!UiAction::Start.requires_running_pipeline());
    }

    #[test]
    fn actions_calibration_requires_pipeline() {
        assert!(UiAction::BeginCalibration.requires_running_pipeline());
        assert!(UiAction::CancelCalibration.requires_running_pipeline());
        assert!(UiAction::RetryCalibration.requires_running_pipeline());
    }

    #[test]
    fn actions_stop_does_not_require_pipeline() {
        assert!(!UiAction::Stop.requires_running_pipeline());
    }

    #[test]
    fn ndi_output_actions_are_independent_of_tracking_pipeline() {
        assert!(!UiAction::StartNdiOutput.requires_running_pipeline());
        assert!(!UiAction::StopNdiOutput.requires_running_pipeline());
        assert!(!UiAction::StartNdiOutput.is_navigation());
        assert!(!UiAction::StopNdiOutput.is_navigation());
    }

    #[test]
    fn actions_navigation_is_navigation() {
        assert!(UiAction::SwitchScreen(Screen::Live).is_navigation());
        assert!(UiAction::SwitchScreen(Screen::Setup).is_navigation());
        assert!(UiAction::SwitchScreen(Screen::Diagnostics).is_navigation());
    }

    #[test]
    fn actions_non_navigation_is_not_navigation() {
        assert!(!UiAction::Start.is_navigation());
        assert!(!UiAction::Stop.is_navigation());
        assert!(!UiAction::RefreshCameras.is_navigation());
    }

    #[test]
    fn actions_import_avatar_carries_path() {
        let path = PathBuf::from("/tmp/model.vrm");
        let action = UiAction::ImportAvatar { path: path.clone() };
        match action {
            UiAction::ImportAvatar { path: p } => assert_eq!(p, path),
            _ => panic!("expected ImportAvatar"),
        }
    }

    #[test]
    fn actions_select_camera_carries_index() {
        let action = UiAction::SelectCamera { index: 3 };
        match action {
            UiAction::SelectCamera { index } => assert_eq!(index, 3),
            _ => panic!("expected SelectCamera"),
        }
    }

    #[test]
    fn reset_camera_is_a_distinct_one_shot_action() {
        assert!(!UiAction::ResetAvatarCamera.is_navigation());
        assert!(!UiAction::ResetAvatarCamera.requires_running_pipeline());
    }

    #[test]
    fn actions_are_clone_and_eq() {
        let a = UiAction::Start;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn actions_are_debug() {
        let action = UiAction::ImportAvatar {
            path: PathBuf::from("/tmp/model.vrm"),
        };
        let debug = format!("{:?}", action);
        assert!(debug.contains("ImportAvatar"));
        assert!(debug.contains("model.vrm"));
    }
}
