//! App orchestrator — processes UI actions and manages domain state.
//!
//! The orchestrator receives [`UiAction`](crate::actions::UiAction) commands from
//! the UI layer and translates them into domain service calls (camera, import,
//! tracking, etc.). The UI model projects this state into display snapshots.
//!
//! Avatar file requests are prepared by a one-shot worker and returned with
//! their request ID. The lifecycle bridge emits prepared load messages and
//! commits the model only when the avatar plugin accepts the request.
//!
//! State transitions live here. `action_system` dispatches ECS effects,
//! `expressions` owns expression bindings and snapshots, and `lifecycle`
//! submits and commits avatar loads. Public system paths are re-exported here.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use bevy::prelude::Resource;
use vtuber_camera::device::CameraDescriptor;

use crate::actions::UiAction;
use crate::avatar_io::{AvatarFileResult, AvatarFileWork};
use crate::import::ImportedModel;
use crate::license_review::VrmLicenseReview;
use vtuber_avatar::AvatarLifecycleState;

mod action_system;
mod expressions;
mod lifecycle;

pub use action_system::{LookSystemParams, process_ui_actions_system};
pub use expressions::sync_expression_view_model;
pub use lifecycle::sync_avatar_lifecycle_system;

/// An already imported model awaiting background load preparation.
///
/// Used for CLI startup and retries. The file worker reads its managed copy
/// and settings before the lifecycle bridge submits a prepared request.
#[derive(Clone, Debug)]
pub struct PendingLoadRequest {
    /// Monotonically increasing correlation identifier.
    pub request_id: u64,
    /// The imported model to load.
    pub model: ImportedModel,
}

/// Model and saved look waiting for the corresponding lifecycle result.
#[derive(Debug)]
pub(crate) struct SubmittedAvatarLoad {
    pub(crate) model: ImportedModel,
    pub(crate) look: Option<vtuber_avatar::RichLookSettings>,
}

/// A selected VRM awaiting explicit license acceptance before import.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingAvatarImport {
    /// Source path chosen by the user.
    pub path: PathBuf,
    /// Review shown in the consent sheet.
    pub review: VrmLicenseReview,
    /// Whether the user checked the acceptance checkbox.
    pub accepted: bool,
}

/// Resource managing the application orchestration state.
#[derive(Resource, Debug)]
pub struct Orchestrator {
    /// Asset root for imported models.
    asset_root: PathBuf,
    /// Current import state.
    import_state: ImportState,
    /// Model accepted by the avatar lifecycle, if any.
    pub(crate) imported_model: Option<ImportedModel>,
    /// Camera descriptors.
    pub(crate) cameras: Vec<CameraDescriptor>,
    /// Selected camera index.
    pub(crate) selected_camera: Option<usize>,
    /// Last error, if any.
    last_error: Option<OrchestratorError>,
    /// Pipeline lifecycle state.
    pub(crate) pipeline_state: PipelineState,
    /// Pending avatar load request not yet submitted to the lifecycle.
    pending_load: Option<PendingLoadRequest>,
    pub(crate) pending_file_work: Option<(u64, AvatarFileWork)>,
    active_request_id: Option<u64>,
    prepared_load: Option<(
        vtuber_avatar::LoadImportedAvatarRequest,
        SubmittedAvatarLoad,
    )>,
    /// Requests submitted to the engine but not yet confirmed.
    submitted_loads: BTreeMap<u64, SubmittedAvatarLoad>,
    /// Selected VRM awaiting license acceptance.
    pub(crate) pending_avatar_import: Option<PendingAvatarImport>,
    /// Next avatar load request correlation identifier.
    next_load_request_id: u64,
    /// Mirror of the avatar lifecycle state, updated by the sync system.
    pub(crate) lifecycle_state: AvatarLifecycleState,
    /// Whether capture should be running for the selected avatar and camera.
    capture_desired: bool,
    /// Whether the capture system has acknowledged the current intent.
    capture_ack: bool,
    /// Whether a camera enumeration is pending.
    camera_refresh_requested: bool,
    /// Calibration command waiting for the tracking bridge.
    calibration_request: Option<CalibrationRequest>,
    /// Whether inference should be restarted after a recoverable worker error.
    inference_retry_requested: bool,
}

/// Calibration intent passed from UI orchestration to the tracking domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationRequest {
    /// Start a fresh neutral collection.
    Begin,
    /// Cancel and clear the current collection.
    Cancel,
    /// Retry after resetting the current collection.
    Retry,
}

/// State of the tracking pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PipelineState {
    /// Pipeline is idle.
    #[default]
    Idle,
    /// Pipeline is starting up.
    Starting,
    /// Pipeline is running.
    Running,
    /// Pipeline is stopping.
    Stopping,
    /// Pipeline failed to start or crashed.
    Failed,
}

/// State of an in-progress or completed import.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ImportState {
    /// No import in progress.
    #[default]
    Idle,
    /// Import is in progress.
    InProgress,
    /// Import completed successfully.
    Success,
    /// Import failed.
    Failed(String),
}

/// Errors that can occur in the orchestrator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchestratorError {
    /// Model import failed.
    ImportFailed(String),
    /// No camera selected.
    NoCameraSelected,
    /// No avatar loaded.
    NoAvatarLoaded,
    /// The avatar lifecycle rejected an imported model load request.
    AvatarLoadRejected(String),
    /// The avatar lifecycle entered a failed state while loading or binding.
    AvatarLifecycleFailed(String),
    /// Persistent avatar pose settings could not be written.
    ArmPoseSettingsFailed(String),
    /// Persistent expression-key bindings could not be written.
    ExpressionSettingsFailed(String),
    /// The selected model's license metadata could not be reviewed.
    LicenseReviewFailed(String),
    /// Camera enumeration, opening, capture, or reconnect failed.
    CameraFailed(String),
    /// Inference model load, execution, or worker failure.
    InferenceFailed(String),
    /// Observed-arm inference failed to start, load, or process frames.
    PoseInferenceFailed(String),
}

impl fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImportFailed(msg) => write!(f, "Import failed: {msg}"),
            Self::NoCameraSelected => write!(f, "No camera selected"),
            Self::NoAvatarLoaded => write!(f, "No avatar loaded"),
            Self::AvatarLoadRejected(msg) => write!(f, "Avatar load rejected: {msg}"),
            Self::AvatarLifecycleFailed(msg) => write!(f, "Avatar lifecycle failed: {msg}"),
            Self::ArmPoseSettingsFailed(msg) => write!(f, "Arm-pose settings failed: {msg}"),
            Self::ExpressionSettingsFailed(msg) => {
                write!(f, "Expression settings failed: {msg}")
            }
            Self::LicenseReviewFailed(msg) => write!(f, "License review failed: {msg}"),
            Self::CameraFailed(msg) => write!(f, "Camera failed: {msg}"),
            Self::InferenceFailed(msg) => write!(f, "Inference failed: {msg}"),
            Self::PoseInferenceFailed(msg) => {
                write!(f, "Pose inference failed: {msg}")
            }
        }
    }
}

impl std::error::Error for OrchestratorError {}

impl Default for Orchestrator {
    fn default() -> Self {
        Self {
            asset_root: PathBuf::from("assets"),
            import_state: ImportState::Idle,
            imported_model: None,
            cameras: Vec::new(),
            selected_camera: None,
            last_error: None,
            pipeline_state: PipelineState::Idle,
            pending_load: None,
            submitted_loads: BTreeMap::new(),
            pending_file_work: None,
            active_request_id: None,
            prepared_load: None,
            pending_avatar_import: None,
            next_load_request_id: 1,
            lifecycle_state: AvatarLifecycleState::NoAvatar,
            capture_desired: false,
            capture_ack: true,
            camera_refresh_requested: true,
            calibration_request: None,
            inference_retry_requested: false,
        }
    }
}

impl Orchestrator {
    /// Create a new orchestrator with the given asset root.
    #[must_use]
    pub fn new(asset_root: PathBuf) -> Self {
        Self {
            asset_root,
            ..Default::default()
        }
    }

    /// Process a UI action and update internal state.
    pub fn process_action(&mut self, action: &UiAction) {
        match action {
            UiAction::RefreshCameras => {
                // Camera enumeration is handled by the capture bridge system,
                // never by the UI renderer or a fabricated device list.
                self.camera_refresh_requested = true;
            }
            UiAction::SelectCamera { index } => {
                self.select_camera(*index);
            }
            UiAction::ImportAvatar { path } => {
                self.import_avatar(path);
            }
            UiAction::RequestAvatarImportReview { path } => {
                self.request_avatar_import_review(path);
            }
            UiAction::SetAvatarImportReviewAccepted { accepted } => {
                self.set_avatar_import_review_accepted(*accepted);
            }
            UiAction::AcceptAvatarImportReview => {
                self.accept_avatar_import_review();
            }
            UiAction::CancelAvatarImportReview => {
                self.cancel_avatar_import_review();
            }
            UiAction::UnloadAvatar => {
                self.unload_avatar();
            }
            UiAction::DismissError => {
                self.last_error = None;
            }
            UiAction::RetryAfterError => {
                if matches!(self.last_error, Some(OrchestratorError::InferenceFailed(_))) {
                    self.inference_retry_requested = true;
                    self.last_error = None;
                    if self.capture_desired {
                        self.pipeline_state = PipelineState::Starting;
                        // Capture is still owned by the current session; only
                        // the inference worker needs to be restarted.
                        self.capture_ack = true;
                    } else if self.selected_camera.is_some() {
                        // An inference failure stops capture as part of
                        // recovery. The retry must therefore request a fresh
                        // camera start instead of reusing a stopped session.
                        self.pipeline_state = PipelineState::Starting;
                        self.capture_desired = true;
                        self.capture_ack = false;
                    }
                }
                self.retry_avatar_load();
            }
            UiAction::BeginCalibration => {
                if self.pipeline_state == PipelineState::Running {
                    self.calibration_request = Some(CalibrationRequest::Begin);
                }
            }
            UiAction::CancelCalibration => {
                self.calibration_request = Some(CalibrationRequest::Cancel);
            }
            UiAction::RetryCalibration => {
                if self.pipeline_state == PipelineState::Running {
                    self.calibration_request = Some(CalibrationRequest::Retry);
                }
            }
            // These effects are owned by the outer application dispatcher.
            UiAction::SwitchPane(_)
            | UiAction::ResetAvatarCamera
            | UiAction::StartNdiOutput
            | UiAction::StopNdiOutput
            | UiAction::ToggleMirror
            | UiAction::TogglePreview
            | UiAction::ToggleAvatarMotionMirror
            | UiAction::SetArmTrackingEnabled { .. }
            | UiAction::RecalibrateArms
            | UiAction::SetArmPoseProfile { .. }
            | UiAction::ResetArmPoseProfile { .. }
            | UiAction::AssignExpressionKey { .. }
            | UiAction::ResetExpressionBindings { .. }
            | UiAction::ToggleExpressionKey { .. }
            | UiAction::ClearManualExpression { .. }
            | UiAction::SetLanguage(_)
            | UiAction::ChangeRichLook { .. }
            | UiAction::SaveRichLook { .. } => {}
        }
    }

    /// Refresh the list of available cameras from an external source.
    pub fn set_camera_list(&mut self, cameras: Vec<CameraDescriptor>) {
        let selected_id = self
            .selected_camera
            .and_then(|index| self.cameras.get(index))
            .map(|camera| camera.id.clone());
        self.cameras = cameras;
        self.selected_camera = selected_id
            .as_ref()
            .and_then(|id| self.cameras.iter().position(|camera| camera.id == *id));
    }

    /// Select a camera, stopping the previous session before automatic startup.
    fn select_camera(&mut self, index: usize) {
        if index >= self.cameras.len()
            || (self.selected_camera == Some(index) && self.pipeline_state != PipelineState::Failed)
        {
            return;
        }
        self.selected_camera = Some(index);
        self.last_error = None;
        self.stop_pipeline();
    }

    /// Import an avatar from the given path.
    ///
    /// On success a [`PendingLoadRequest`] is queued while the accepted model
    /// remains selected. The sync system drains the pending request and emits a
    /// `LoadImportedAvatarRequest` that the avatar lifecycle consumes.
    fn import_avatar(&mut self, path: &Path) {
        self.import_state = ImportState::InProgress;
        self.last_error = None;

        let request_id = self.begin_avatar_request();
        self.pending_file_work = Some((request_id, AvatarFileWork::Import(path.to_path_buf())));
    }

    /// Queues an imported model for background load preparation without replacing
    /// the accepted model or its look until the engine confirms the request.
    pub fn queue_imported_model(&mut self, model: ImportedModel) {
        let request_id = self.begin_avatar_request();
        self.pending_load = Some(PendingLoadRequest { request_id, model });
        self.import_state = ImportState::InProgress;
        self.last_error = None;
    }

    fn begin_avatar_request(&mut self) -> u64 {
        let request_id = self.next_load_request_id;
        self.next_load_request_id = self.next_load_request_id.wrapping_add(1);
        self.active_request_id = Some(request_id);
        self.pending_file_work = None;
        self.pending_load = None;
        self.prepared_load = None;
        self.pending_avatar_import = None;
        request_id
    }

    pub(crate) fn complete_avatar_work(
        &mut self,
        request_id: u64,
        result: Result<AvatarFileResult, OrchestratorError>,
    ) {
        if self.active_request_id != Some(request_id) {
            return;
        }
        match result {
            Ok(AvatarFileResult::Review { path, review }) => {
                self.import_state = ImportState::Idle;
                self.pending_avatar_import = Some(PendingAvatarImport {
                    path,
                    review,
                    accepted: false,
                });
            }
            Ok(AvatarFileResult::Load(request, submitted)) => {
                self.prepared_load = Some((request, submitted));
                self.import_state = ImportState::Success;
            }
            Err(error) => {
                self.import_state = ImportState::Failed(error.to_string());
                self.last_error = Some(error);
            }
        }
    }

    /// Unload the current avatar.
    ///
    /// Stops tracking and clears the imported model and pending load requests.
    /// The sync system requests removal of the active avatar.
    fn unload_avatar(&mut self) {
        self.stop_pipeline();
        self.imported_model = None;
        self.import_state = ImportState::Idle;
        self.pending_load = None;
        self.submitted_loads.clear();
        self.pending_file_work = None;
        self.active_request_id = None;
        self.prepared_load = None;
        self.pending_avatar_import = None;
    }

    /// Requests background license extraction for the selected file.
    ///
    /// The model is not imported here. A failed extraction clears any previous
    /// review and surfaces a recoverable error; there is no bypass that would
    /// import a model whose license could not be reviewed.
    fn request_avatar_import_review(&mut self, path: &Path) {
        self.last_error = None;
        let request_id = self.begin_avatar_request();
        self.import_state = ImportState::InProgress;
        self.pending_file_work = Some((request_id, AvatarFileWork::Review(path.to_path_buf())));
    }

    /// Records the review checkbox state.
    fn set_avatar_import_review_accepted(&mut self, accepted: bool) {
        if let Some(pending) = &mut self.pending_avatar_import {
            pending.accepted = accepted;
        }
    }

    /// Imports the reviewed model only after explicit acceptance.
    fn accept_avatar_import_review(&mut self) {
        let Some(pending) = &self.pending_avatar_import else {
            return;
        };
        if !pending.accepted {
            return;
        }
        let path = pending.path.clone();
        self.pending_avatar_import = None;
        self.import_avatar(&path);
    }

    /// Dismisses the review without importing.
    fn cancel_avatar_import_review(&mut self) {
        self.import_state = ImportState::Idle;
        self.pending_avatar_import = None;
        self.pending_file_work = None;
        self.active_request_id = None;
    }

    /// Retry a failed avatar load by re-submitting the current imported model.
    fn retry_avatar_load(&mut self) {
        if self.lifecycle_state != AvatarLifecycleState::Failed {
            return;
        }
        if let Some(model) = self.imported_model.clone() {
            self.queue_imported_model(model);
            self.last_error = None;
            self.lifecycle_state = AvatarLifecycleState::NoAvatar;
        }
    }

    /// Starts tracking automatically once setup is complete: the avatar is
    /// ready, a camera is selected, and the pipeline is idle.
    pub fn maybe_auto_start_tracking(&mut self) {
        if self.pipeline_state != PipelineState::Idle
            || self.lifecycle_state != AvatarLifecycleState::Ready
            || self.selected_camera.is_none()
            || self.imported_model.is_none()
        {
            return;
        }
        self.pipeline_state = PipelineState::Starting;
        self.capture_desired = true;
        self.capture_ack = false;
        self.inference_retry_requested = false;
    }

    /// Stop the tracking pipeline.
    fn stop_pipeline(&mut self) {
        if self.pipeline_state == PipelineState::Idle
            || self.pipeline_state == PipelineState::Stopping
        {
            return;
        }
        self.pipeline_state = PipelineState::Stopping;
        self.capture_desired = false;
        self.capture_ack = false;
        self.inference_retry_requested = false;
    }

    /// Get the current pipeline state.
    #[must_use]
    pub fn pipeline_state(&self) -> PipelineState {
        self.pipeline_state
    }

    /// Get the last error, if any.
    #[must_use]
    pub fn last_error(&self) -> Option<&OrchestratorError> {
        self.last_error.as_ref()
    }

    /// Get the import state.
    #[must_use]
    pub fn import_state(&self) -> &ImportState {
        &self.import_state
    }

    /// Take the pending load request, if any.
    ///
    /// The file worker bridge drains this once to prepare the model and settings.
    pub fn take_pending_load_request(&mut self) -> Option<PendingLoadRequest> {
        self.pending_load.take()
    }

    /// Update the lifecycle state mirror.
    ///
    /// The sync system calls this after reading the `AvatarLifecycle` resource
    /// so that state decisions and the display projection use the true engine state.
    pub fn set_lifecycle_state(&mut self, state: AvatarLifecycleState) {
        self.lifecycle_state = state;
    }

    /// Whether the orchestrator has an imported model.
    #[must_use]
    pub fn has_imported_model(&self) -> bool {
        self.imported_model.is_some()
    }

    /// Test-only imported-model seam for NDI orchestration contracts.
    #[cfg(test)]
    pub(crate) fn set_imported_model_for_tests(
        &mut self,
        model: Option<crate::import::ImportedModel>,
    ) {
        self.imported_model = model;
    }

    /// Returns the stable ID of the currently imported model, if any.
    #[must_use]
    pub fn active_model_id(&self) -> Option<&str> {
        self.imported_model.as_ref().map(|model| model.id.as_str())
    }

    /// The asset root used for model imports.
    #[must_use]
    pub fn asset_root(&self) -> &Path {
        &self.asset_root
    }

    /// Whether capture should be running.
    #[must_use]
    pub fn capture_desired(&self) -> bool {
        self.capture_desired
    }

    /// Whether the capture system has acknowledged the current intent.
    #[must_use]
    pub fn capture_ack(&self) -> bool {
        self.capture_ack
    }

    /// Whether a fresh camera enumeration is pending.
    #[must_use]
    pub fn camera_refresh_requested(&self) -> bool {
        self.camera_refresh_requested
    }

    /// Marks a camera enumeration request as handled.
    pub fn clear_camera_refresh_request(&mut self) {
        self.camera_refresh_requested = false;
    }

    /// Acknowledge the current capture state.
    pub fn set_capture_ack(&mut self, ack: bool) {
        self.capture_ack = ack;
    }

    /// Set the pipeline state (called by the capture bridge system).
    pub fn set_pipeline_state(&mut self, state: PipelineState) {
        self.pipeline_state = state;
    }

    /// Set the last error (called by the capture bridge system).
    pub fn set_last_error(&mut self, error: Option<OrchestratorError>) {
        self.last_error = error;
    }

    /// Moves an inference failure through the normal reverse-order shutdown.
    ///
    /// The inference bridge calls this after observing a failed worker. The
    /// capture bridge then stops the camera, while the inference bridge joins
    /// the failed inference worker first.
    pub fn fail_inference(&mut self, message: String) {
        self.last_error = Some(OrchestratorError::InferenceFailed(message));
        self.capture_desired = false;
        self.capture_ack = false;
        self.pipeline_state = PipelineState::Stopping;
    }

    /// Moves a capture-worker failure through the same recoverable shutdown.
    pub fn fail_camera(&mut self, message: String) {
        self.last_error = Some(OrchestratorError::CameraFailed(message));
        self.capture_desired = false;
        self.capture_ack = false;
        self.pipeline_state = PipelineState::Stopping;
    }

    /// Completes a capture stop while preserving a recoverable failure state.
    pub fn complete_capture_stop(&mut self) {
        if self.last_error.is_some() {
            self.pipeline_state = PipelineState::Failed;
        } else {
            self.pipeline_state = PipelineState::Idle;
        }
    }

    /// Get the selected camera descriptor, if any.
    #[must_use]
    pub fn selected_camera_descriptor(&self) -> Option<CameraDescriptor> {
        self.selected_camera
            .and_then(|idx| self.cameras.get(idx).cloned())
    }

    /// Takes the pending calibration intent, if any.
    pub fn take_calibration_request(&mut self) -> Option<CalibrationRequest> {
        self.calibration_request.take()
    }

    /// Takes a pending inference restart request.
    pub fn take_inference_retry_request(&mut self) -> bool {
        let requested = self.inference_retry_requested;
        self.inference_retry_requested = false;
        requested
    }
}

#[cfg(test)]
mod tests;
