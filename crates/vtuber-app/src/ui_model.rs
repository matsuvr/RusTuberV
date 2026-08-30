//! UI view models — immutable snapshots for rendering the UI.
//!
//! These types hide Bevy query details from the UI layer. The UI reads
//! these snapshots and emits [`crate::actions::UiAction`] commands.

use bevy::prelude::Resource;
use std::path::PathBuf;

use crate::import::VrmGeneration;
use vtuber_avatar::ArmPoseProfile;

/// Which screen the UI is currently showing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Screen {
    /// Initial setup: camera selection, avatar import.
    #[default]
    Setup,
    /// Live preview with tracking active.
    Live,
    /// Performance diagnostics and metrics.
    Diagnostics,
}

/// Overall application lifecycle state for UI display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AppLifecycle {
    /// Nothing configured yet.
    #[default]
    Idle,
    /// Workers are starting up.
    Starting,
    /// All workers running, tracking active.
    Running,
    /// Workers are shutting down.
    Stopping,
    /// A recoverable error occurred.
    Failed,
}

/// Camera state for the UI.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CameraViewModel {
    /// Available camera descriptors (display name, index).
    pub available_cameras: Vec<CameraDescriptor>,
    /// Currently selected camera index, if any.
    pub selected_index: Option<usize>,
    /// Whether the camera is currently capturing.
    pub is_capturing: bool,
    /// Camera backend name (e.g. "MSMF", "AVFoundation").
    pub backend: Option<String>,
    /// Active resolution (width x height).
    pub resolution: Option<(u32, u32)>,
}

/// A camera descriptor for display in the UI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraDescriptor {
    /// Human-readable camera name.
    pub name: String,
    /// Platform-specific index or identifier.
    pub index: usize,
}

/// Avatar state for the UI.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AvatarViewModel {
    /// Imported model summary, if any.
    pub imported_model: Option<ImportedModelSummary>,
    /// Avatar lifecycle state.
    pub lifecycle: AvatarLifecycleState,
    /// Whether the avatar is ready for tracking.
    pub is_ready: bool,
    /// Whether the avatar load/binding failed (recoverable).
    pub load_failed: bool,
}

/// Settings view model for the active avatar's default arm pose.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArmPoseViewModel {
    /// Current validated profile, automatic or model-specific.
    pub profile: ArmPoseProfile,
    /// Whether the active model has an explicit persisted override.
    pub has_override: bool,
}

/// Summary of an imported model for display.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedModelSummary {
    /// VRM generation accepted by preflight and the runtime adapter.
    pub generation: VrmGeneration,
    /// Stable asset ID (SHA-256 short).
    pub id: String,
    /// Model name from VRM metadata.
    pub name: String,
    /// Original file path (for display only, not used for loading).
    pub original_path: PathBuf,
    /// Whether the model has all required bones.
    pub has_required_bones: bool,
    /// Available expression preset count.
    pub expression_count: usize,
}

/// Avatar lifecycle state for UI display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AvatarLifecycleState {
    /// No avatar loaded.
    #[default]
    None,
    /// Avatar is loading.
    Loading,
    /// Avatar is binding bones.
    Binding,
    /// Avatar is ready.
    Ready,
    /// Avatar is unloading.
    Unloading,
    /// Avatar loading/binding failed.
    Failed,
}

/// Calibration state for the UI.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CalibrationViewModel {
    /// Whether calibration is in progress.
    pub is_calibrating: bool,
    /// Number of samples collected so far.
    pub samples_collected: u32,
    /// Target number of samples.
    pub samples_target: u32,
    /// Quality score (0.0 to 1.0), if available.
    pub quality_score: Option<f32>,
    /// Last rejection reason, if any.
    pub last_reject_reason: Option<String>,
    /// Whether calibration completed successfully.
    pub is_complete: bool,
}

/// Tracking state for the UI.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrackingViewModel {
    /// Whether tracking is active.
    pub is_tracking: bool,
    /// Current tracking state (Tracking/Lost/Initializing).
    pub state: TrackingState,
    /// Current confidence (0.0 to 1.0).
    pub confidence: f32,
    /// Face detected.
    pub face_detected: bool,
}

/// Tracking state enum for UI display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrackingState {
    /// Not tracking.
    #[default]
    Idle,
    /// Initializing / calibrating.
    Initializing,
    /// Actively tracking.
    Tracking,
    /// Face lost, attempting recovery.
    Lost,
}

/// Display state for the optional NDI output sender.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NdiOutputUiState {
    /// No sender is active.
    #[default]
    Off,
    /// The sender is being initialized.
    Starting,
    /// Frames are being offered to the sender.
    Live,
    /// The last start/send operation failed.
    Error,
}

/// Small immutable snapshot of the optional NDI output state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NdiOutputViewModel {
    /// Whether this build contains the explicit NDI SDK feature.
    pub available: bool,
    /// Current sender state.
    pub state: NdiOutputUiState,
    /// Fixed source name used by the application.
    pub source_name: Option<String>,
    /// Number of currently connected receivers, when reported by the backend.
    pub connections: Option<u32>,
    /// Frames discarded because the sender mailbox was not running.
    pub dropped_frames: u64,
    /// Frames replaced in the bounded latest-value mailbox.
    pub replaced_frames: u64,
    /// Stable backend error code, when present.
    pub error_code: Option<String>,
    /// Short backend error message, when present.
    pub error_message: Option<String>,
}

/// Complete UI view model snapshot.
#[derive(Clone, Debug, Default, Resource)]
pub struct UiViewModel {
    /// Current screen.
    pub screen: Screen,
    /// Application lifecycle.
    pub lifecycle: AppLifecycle,
    /// Camera state.
    pub camera: CameraViewModel,
    /// Avatar state.
    pub avatar: AvatarViewModel,
    /// Active model arm-pose settings.
    pub arm_pose: ArmPoseViewModel,
    /// Calibration state.
    pub calibration: CalibrationViewModel,
    /// Tracking state.
    pub tracking: TrackingViewModel,
    /// Optional NDI output state.
    pub ndi_output: NdiOutputViewModel,
    /// Whether preview mirroring is enabled.
    pub mirror_preview: bool,
    /// Whether avatar motion is reflected for the operator.
    pub mirror_avatar_motion: bool,
    /// Whether preview is visible.
    pub preview_visible: bool,
}

impl UiViewModel {
    /// Check if the Start action is available.
    #[must_use]
    pub fn can_start(&self) -> bool {
        self.lifecycle == AppLifecycle::Idle
            && self.avatar.is_ready
            && self.camera.selected_index.is_some()
    }

    /// Check if the Stop action is available.
    #[must_use]
    pub fn can_stop(&self) -> bool {
        self.lifecycle == AppLifecycle::Running
    }

    /// Check if calibration can begin.
    #[must_use]
    pub fn can_calibrate(&self) -> bool {
        self.lifecycle == AppLifecycle::Running
            && !self.calibration.is_calibrating
            && !self.calibration.is_complete
    }

    /// Check if the current avatar has a camera default that can be reset.
    #[must_use]
    pub fn can_reset_camera(&self) -> bool {
        self.avatar.is_ready && self.avatar.lifecycle == AvatarLifecycleState::Ready
    }

    /// Check if NDI output can be started for the current ready avatar.
    #[must_use]
    pub fn can_start_ndi_output(&self) -> bool {
        self.avatar.is_ready
            && self.avatar.lifecycle == AvatarLifecycleState::Ready
            && self.ndi_output.available
            && matches!(
                self.ndi_output.state,
                NdiOutputUiState::Off | NdiOutputUiState::Error
            )
    }

    /// Check if an active or starting NDI sender can be stopped.
    #[must_use]
    pub fn can_stop_ndi_output(&self) -> bool {
        matches!(
            self.ndi_output.state,
            NdiOutputUiState::Starting | NdiOutputUiState::Live
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_model_default_is_idle() {
        let vm = UiViewModel::default();
        assert_eq!(vm.lifecycle, AppLifecycle::Idle);
        assert_eq!(vm.screen, Screen::Setup);
        assert!(!vm.can_start());
        assert!(!vm.can_stop());
    }

    #[test]
    fn ui_model_can_start_when_ready() {
        let mut vm = UiViewModel::default();
        vm.avatar.is_ready = true;
        vm.camera.selected_index = Some(0);
        assert!(vm.can_start());
    }

    #[test]
    fn ui_model_cannot_start_without_camera() {
        let mut vm = UiViewModel::default();
        vm.avatar.is_ready = true;
        // No camera selected.
        assert!(!vm.can_start());
    }

    #[test]
    fn ui_model_cannot_start_without_avatar() {
        let mut vm = UiViewModel::default();
        vm.camera.selected_index = Some(0);
        // Avatar not ready.
        assert!(!vm.can_start());
    }

    #[test]
    fn ui_model_can_stop_when_running() {
        let vm = UiViewModel {
            lifecycle: AppLifecycle::Running,
            ..Default::default()
        };
        assert!(vm.can_stop());
    }

    #[test]
    fn ui_model_cannot_stop_when_idle() {
        let vm = UiViewModel::default();
        assert!(!vm.can_stop());
    }

    #[test]
    fn ui_model_ndi_start_requires_ready_avatar_but_not_tracking() {
        let mut vm = UiViewModel::default();
        vm.ndi_output.available = true;
        assert!(!vm.can_start_ndi_output());

        vm.avatar.is_ready = true;
        vm.avatar.lifecycle = AvatarLifecycleState::Ready;
        assert!(vm.can_start_ndi_output());

        vm.lifecycle = AppLifecycle::Running;
        assert!(vm.can_start_ndi_output());
    }

    #[test]
    fn ui_model_ndi_stop_is_available_only_while_starting_or_live() {
        let mut vm = UiViewModel::default();
        assert!(!vm.can_stop_ndi_output());
        vm.ndi_output.state = NdiOutputUiState::Starting;
        assert!(vm.can_stop_ndi_output());
        vm.ndi_output.state = NdiOutputUiState::Live;
        assert!(vm.can_stop_ndi_output());
    }

    #[test]
    fn ui_model_can_reset_camera_only_for_a_ready_avatar() {
        let mut vm = UiViewModel::default();
        assert!(!vm.can_reset_camera());

        vm.avatar.is_ready = true;
        assert!(!vm.can_reset_camera());

        vm.avatar.lifecycle = AvatarLifecycleState::Ready;
        assert!(vm.can_reset_camera());
    }

    #[test]
    fn ui_model_can_calibrate_when_running() {
        let vm = UiViewModel {
            lifecycle: AppLifecycle::Running,
            ..Default::default()
        };
        assert!(vm.can_calibrate());
    }

    #[test]
    fn ui_model_cannot_calibrate_when_calibrating() {
        let vm = UiViewModel {
            lifecycle: AppLifecycle::Running,
            calibration: CalibrationViewModel {
                is_calibrating: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!vm.can_calibrate());
    }

    #[test]
    fn ui_model_cannot_calibrate_when_complete() {
        let vm = UiViewModel {
            lifecycle: AppLifecycle::Running,
            calibration: CalibrationViewModel {
                is_complete: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!vm.can_calibrate());
    }

    #[test]
    fn ui_model_screen_transitions() {
        let mut vm = UiViewModel::default();
        assert_eq!(vm.screen, Screen::Setup);
        vm.screen = Screen::Live;
        assert_eq!(vm.screen, Screen::Live);
        vm.screen = Screen::Diagnostics;
        assert_eq!(vm.screen, Screen::Diagnostics);
    }
}
