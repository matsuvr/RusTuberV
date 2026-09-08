//! UI view models — immutable snapshots for rendering the UI.
//! These types hide Bevy queries from the UI, which emits UiAction commands.

use crate::import::VrmGeneration;
use bevy::prelude::Resource;
use std::path::PathBuf;
use vtuber_avatar::ArmPoseProfile;

/// Destination selected by the navigation-only sidebar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pane {
    /// Guided setup overview. The desktop shell selects this on startup.
    Studio,
    /// Capture device and explicitly confirmed camera preview.
    #[default]
    Camera,
    /// VRM import and arm pose.
    Avatar,
    /// Neutral pose calibration.
    Calibration,
    /// Legacy camera-preview destination, rendered as the camera page.
    Preview,
    /// Clean avatar output, including optional NDI.
    NdiOutput,
    /// Tracking health and technical diagnostics.
    Diagnostics,
    /// Application language and appearance settings.
    Settings,
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
    /// Available camera descriptors.
    pub available_cameras: Vec<CameraDescriptor>,
    /// Currently selected camera index.
    pub selected_index: Option<usize>,
    /// Whether the camera is capturing.
    pub is_capturing: bool,
    /// Backend name.
    pub backend: Option<String>,
    /// Active resolution.
    pub resolution: Option<(u32, u32)>,
}

/// A camera descriptor for display.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraDescriptor {
    /// Human-readable device name.
    pub name: String,
    /// Platform index or identifier.
    pub index: usize,
}

/// Avatar state for the UI.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AvatarViewModel {
    /// Imported model summary.
    pub imported_model: Option<ImportedModelSummary>,
    /// Avatar lifecycle state.
    pub lifecycle: AvatarLifecycleState,
    /// Whether the avatar is ready for tracking.
    pub is_ready: bool,
    /// Whether loading or binding failed.
    pub load_failed: bool,
}

/// Settings for the active avatar's default arm pose.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ArmPoseViewModel {
    /// Current validated profile.
    pub profile: ArmPoseProfile,
    /// Whether there is a model-specific persisted override.
    pub has_override: bool,
}

/// Summary of an imported model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedModelSummary {
    /// VRM generation.
    pub generation: VrmGeneration,
    /// Stable asset ID.
    pub id: String,
    /// Name from VRM metadata.
    pub name: String,
    /// Original source path, not used for loading.
    pub original_path: PathBuf,
    /// Whether required bones exist.
    pub has_required_bones: bool,
    /// Expression preset count.
    pub expression_count: usize,
}

/// Avatar lifecycle state for display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AvatarLifecycleState {
    /// No avatar loaded.
    #[default]
    None,
    /// Loading.
    Loading,
    /// Binding bones.
    Binding,
    /// Ready.
    Ready,
    /// Unloading.
    Unloading,
    /// Failed.
    Failed,
}

/// Calibration state for the UI.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CalibrationViewModel {
    /// Whether calibration is collecting samples.
    pub is_calibrating: bool,
    /// Samples collected.
    pub samples_collected: u32,
    /// Target samples.
    pub samples_target: u32,
    /// Quality score in 0..1.
    pub quality_score: Option<f32>,
    /// Last rejection reason.
    pub last_reject_reason: Option<String>,
    /// Whether calibration succeeded.
    pub is_complete: bool,
}

/// Tracking state for the UI.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrackingViewModel {
    /// Whether tracking is active.
    pub is_tracking: bool,
    /// Current state.
    pub state: TrackingState,
    /// Confidence in 0..1.
    pub confidence: f32,
    /// Face detected.
    pub face_detected: bool,
}

/// Tracking state for display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrackingState {
    /// Not tracking.
    #[default]
    Idle,
    /// Initializing.
    Initializing,
    /// Tracking.
    Tracking,
    /// Face lost.
    Lost,
}

/// State of the optional NDI sender.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NdiOutputUiState {
    /// Off.
    #[default]
    Off,
    /// Initializing.
    Starting,
    /// Offering frames to receivers.
    Live,
    /// Failed.
    Error,
}

/// Immutable NDI output snapshot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NdiOutputViewModel {
    /// Whether the build includes the SDK feature.
    pub available: bool,
    /// Whether the NDI runtime was found.
    pub runtime_installed: bool,
    /// Sender state.
    pub state: NdiOutputUiState,
    /// Source name.
    pub source_name: Option<String>,
    /// Connected receiver count.
    pub connections: Option<u32>,
    /// Discarded frames.
    pub dropped_frames: u64,
    /// Replaced mailbox frames.
    pub replaced_frames: u64,
    /// Backend error code.
    pub error_code: Option<String>,
    /// Backend error detail.
    pub error_message: Option<String>,
}

/// Complete UI snapshot.
#[derive(Clone, Debug, Default, Resource)]
pub struct UiViewModel {
    /// Selected destination.
    pub pane: Pane,
    /// Application lifecycle.
    pub lifecycle: AppLifecycle,
    /// Camera state.
    pub camera: CameraViewModel,
    /// Avatar state.
    pub avatar: AvatarViewModel,
    /// Arm pose settings.
    pub arm_pose: ArmPoseViewModel,
    /// Calibration state.
    pub calibration: CalibrationViewModel,
    /// Tracking state.
    pub tracking: TrackingViewModel,
    /// NDI state.
    pub ndi_output: NdiOutputViewModel,
    /// Camera-preview mirroring.
    pub mirror_preview: bool,
    /// Operator-facing avatar mirroring.
    pub mirror_avatar_motion: bool,
    /// Camera-preview visibility.
    pub preview_visible: bool,
}

impl UiViewModel {
    /// Whether Start is available.
    #[must_use]
    pub fn can_start(&self) -> bool {
        self.lifecycle == AppLifecycle::Idle
            && self.avatar.is_ready
            && self.camera.selected_index.is_some()
    }
    /// Whether Stop is available.
    #[must_use]
    pub fn can_stop(&self) -> bool {
        self.lifecycle == AppLifecycle::Running
    }
    /// Whether neutral calibration can begin.
    #[must_use]
    pub fn can_calibrate(&self) -> bool {
        self.lifecycle == AppLifecycle::Running
            && !self.calibration.is_calibrating
            && !self.calibration.is_complete
    }
    /// Whether the avatar has a camera default to restore.
    #[must_use]
    pub fn can_reset_camera(&self) -> bool {
        self.avatar.is_ready && self.avatar.lifecycle == AvatarLifecycleState::Ready
    }
    /// Whether NDI can start independently of tracking.
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
    /// Whether NDI can stop.
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
        assert_eq!(vm.pane, Pane::Camera);
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
        assert!(!vm.can_start());
    }
    #[test]
    fn ui_model_cannot_start_without_avatar() {
        let mut vm = UiViewModel::default();
        vm.camera.selected_index = Some(0);
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
        assert!(!UiViewModel::default().can_stop());
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
    fn ui_model_pane_transitions() {
        let mut vm = UiViewModel::default();
        assert_eq!(vm.pane, Pane::Camera);
        for pane in [
            Pane::Studio,
            Pane::Diagnostics,
            Pane::Settings,
            Pane::Camera,
        ] {
            vm.pane = pane;
            assert_eq!(vm.pane, pane);
        }
    }
}
