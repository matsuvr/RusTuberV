//! App orchestrator — processes UI actions and manages domain state.
//!
//! The orchestrator receives [`UiAction`] commands from the UI layer and
//! translates them into domain service calls (camera, import, tracking, etc.).
//! It updates the [`UiViewModel`] snapshot that the UI reads each frame.
//!
//! Avatar loading is bridged to the `vtuber-avatar` lifecycle through a
//! pending-request protocol: after a successful import the orchestrator stores
//! a [`PendingLoadRequest`]; a Bevy system (in the same crate) drains it and
//! emits the corresponding `LoadImportedAvatarRequest` message that the avatar
//! plugin consumes.

use std::path::{Path, PathBuf};

use bevy::prelude::*;

use crate::actions::UiAction;
use crate::import::VrmGeneration;
use crate::import::{self, ImportedModel, ModelImportError};
use crate::license_review::{self, VrmLicenseReview, VrmLicenseReviewError};
use crate::ndi_output::NdiOutputIntent;
use crate::preview::PreviewState;
use crate::settings::ArmPoseSettings;
use crate::ui::UiState;
use crate::ui_model::*;
use vtuber_avatar::{
    ArmPoseOverrideStore, ArmPoseProfileChange, ArmPoseProfileOverride, AvatarAssetId,
    AvatarMotionMirror,
};
use vtuber_camera::device::CameraDescriptor;

/// A pending avatar load request waiting to be submitted to the lifecycle.
///
/// After a successful `import_vrm()` call the orchestrator stores the
/// [`ImportedModel`] here. A Bevy system drains this value and emits a
/// `LoadImportedAvatarRequest` message that the avatar plugin consumes.
#[derive(Clone, Debug)]
pub struct PendingLoadRequest {
    /// Monotonically increasing correlation identifier.
    pub request_id: u64,
    /// The imported model to load.
    pub model: ImportedModel,
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
    /// Imported model, if any.
    imported_model: Option<ImportedModel>,
    /// Camera descriptors.
    cameras: Vec<CameraDescriptor>,
    /// Selected camera index.
    selected_camera: Option<usize>,
    /// Last error, if any.
    last_error: Option<OrchestratorError>,
    /// Pipeline lifecycle state.
    pipeline_state: PipelineState,
    /// Current UI pane.
    current_pane: Pane,
    /// Pending avatar load request not yet submitted to the lifecycle.
    pending_load: Option<PendingLoadRequest>,
    /// Selected VRM awaiting license acceptance.
    pending_avatar_import: Option<PendingAvatarImport>,
    /// Next avatar load request correlation identifier.
    next_load_request_id: u64,
    /// Mirror of the avatar lifecycle state, updated by the sync system.
    lifecycle_state: crate::ui_model::AvatarLifecycleState,
    /// Whether capture should be running (set by Start/Stop actions).
    capture_desired: bool,
    /// Whether the capture system has acknowledged the current intent.
    capture_ack: bool,
    /// Whether a camera enumeration is pending.
    camera_refresh_requested: bool,
    /// Calibration command waiting for the tracking bridge.
    calibration_request: Option<CalibrationRequest>,
    /// Whether inference should be restarted after a recoverable worker error.
    inference_retry_requested: bool,
    /// Whether the automatic start after setup completion is still pending.
    auto_start_armed: bool,
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
    /// Pipeline already running.
    PipelineAlreadyRunning,
    /// Pipeline not running.
    PipelineNotRunning,
    /// The avatar lifecycle rejected an imported model load request.
    AvatarLoadRejected(String),
    /// The avatar lifecycle entered a failed state while loading or binding.
    AvatarLifecycleFailed(String),
    /// Persistent avatar pose settings could not be written.
    ArmPoseSettingsFailed(String),
    /// The selected model's license metadata could not be reviewed.
    LicenseReviewFailed(String),
    /// Camera enumeration, opening, capture, or reconnect failed.
    CameraFailed(String),
    /// Inference model load, execution, or worker failure.
    InferenceFailed(String),
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImportFailed(msg) => write!(f, "Import failed: {msg}"),
            Self::NoCameraSelected => write!(f, "No camera selected"),
            Self::NoAvatarLoaded => write!(f, "No avatar loaded"),
            Self::PipelineAlreadyRunning => write!(f, "Pipeline already running"),
            Self::PipelineNotRunning => write!(f, "Pipeline not running"),
            Self::AvatarLoadRejected(msg) => write!(f, "Avatar load rejected: {msg}"),
            Self::AvatarLifecycleFailed(msg) => write!(f, "Avatar lifecycle failed: {msg}"),
            Self::ArmPoseSettingsFailed(msg) => write!(f, "Arm-pose settings failed: {msg}"),
            Self::LicenseReviewFailed(msg) => write!(f, "License review failed: {msg}"),
            Self::CameraFailed(msg) => write!(f, "Camera failed: {msg}"),
            Self::InferenceFailed(msg) => write!(f, "Inference failed: {msg}"),
        }
    }
}

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
            current_pane: Pane::default(),
            pending_load: None,
            pending_avatar_import: None,
            next_load_request_id: 1,
            lifecycle_state: AvatarLifecycleState::None,
            capture_desired: false,
            capture_ack: true,
            camera_refresh_requested: true,
            calibration_request: None,
            inference_retry_requested: false,
            auto_start_armed: true,
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
            UiAction::SwitchPane(pane) => {
                self.current_pane = *pane;
            }
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
            UiAction::Start => {
                self.start_pipeline();
            }
            UiAction::Stop => {
                self.stop_pipeline();
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
            UiAction::RetryCalibration if self.pipeline_state == PipelineState::Running => {
                self.calibration_request = Some(CalibrationRequest::Retry);
            }
            _ => {
                // Other actions handled by specific subsystems.
            }
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

    /// Select a camera by index.
    fn select_camera(&mut self, index: usize) {
        if index < self.cameras.len() {
            self.selected_camera = Some(index);
            // Choosing a camera completes the camera setup step, so tracking
            // may start once the avatar is also ready.
            self.auto_start_armed = true;
        }
    }

    /// Import an avatar from the given path.
    ///
    /// On success the imported model is stored and a [`PendingLoadRequest`] is
    /// queued. The sync system drains the pending request and emits a
    /// `LoadImportedAvatarRequest` that the avatar lifecycle consumes.
    fn import_avatar(&mut self, path: &PathBuf) {
        self.import_state = ImportState::InProgress;
        self.last_error = None;

        match import::import_vrm(path, &self.asset_root, import::DEFAULT_SIZE_LIMIT) {
            Ok(model) => {
                let request_id = self.next_load_request_id;
                self.next_load_request_id += 1;
                self.pending_load = Some(PendingLoadRequest {
                    request_id,
                    model: model.clone(),
                });
                self.imported_model = Some(model);
                self.import_state = ImportState::Success;
                // A fresh successful import is a setup completion; tracking
                // starts once the avatar is ready and a camera is selected.
                self.auto_start_armed = true;
                // Reset lifecycle from any previous Failed state so the new
                // load can proceed.
                if self.lifecycle_state == AvatarLifecycleState::Failed {
                    self.lifecycle_state = AvatarLifecycleState::None;
                }
            }
            Err(e) => {
                let msg = format_import_error(&e);
                self.import_state = ImportState::Failed(msg.clone());
                self.last_error = Some(OrchestratorError::ImportFailed(msg));
            }
        }
    }

    /// Unload the current avatar.
    ///
    /// Clears the imported model and any pending load request. The sync system
    /// detects the removal and emits an `UnloadAvatarRequest`.
    fn unload_avatar(&mut self) {
        self.imported_model = None;
        self.import_state = ImportState::Idle;
        self.pending_load = None;
    }

    /// Reads the selected file and opens a license review.
    ///
    /// The model is not imported here. A failed extraction clears any previous
    /// review and surfaces a recoverable error; there is no bypass that would
    /// import a model whose license could not be reviewed.
    fn request_avatar_import_review(&mut self, path: &Path) {
        self.last_error = None;
        let result = read_reviewable_bytes(path)
            .and_then(|bytes| license_review::extract_vrm_license_review(path, &bytes));
        match result {
            Ok(review) => {
                self.pending_avatar_import = Some(PendingAvatarImport {
                    path: path.to_path_buf(),
                    review,
                    accepted: false,
                });
            }
            Err(error) => {
                self.pending_avatar_import = None;
                self.last_error = Some(OrchestratorError::LicenseReviewFailed(error.to_string()));
            }
        }
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
        self.pending_avatar_import = None;
    }

    /// Retry a failed avatar load by re-submitting the current imported model.
    fn retry_avatar_load(&mut self) {
        if self.lifecycle_state != AvatarLifecycleState::Failed {
            return;
        }
        if let Some(model) = self.imported_model.clone() {
            let request_id = self.next_load_request_id;
            self.next_load_request_id += 1;
            self.pending_load = Some(PendingLoadRequest { request_id, model });
            self.last_error = None;
            self.lifecycle_state = AvatarLifecycleState::None;
        }
    }

    /// Start the tracking pipeline.
    fn start_pipeline(&mut self) {
        if self.pipeline_state == PipelineState::Running
            || self.pipeline_state == PipelineState::Starting
        {
            self.last_error = Some(OrchestratorError::PipelineAlreadyRunning);
            return;
        }
        if self.selected_camera.is_none() {
            self.last_error = Some(OrchestratorError::NoCameraSelected);
            return;
        }
        if self.imported_model.is_none() {
            self.last_error = Some(OrchestratorError::NoAvatarLoaded);
            return;
        }
        self.pipeline_state = PipelineState::Starting;
        self.capture_desired = true;
        self.capture_ack = false;
        self.inference_retry_requested = false;
        self.auto_start_armed = false;
    }

    /// Starts tracking automatically once setup is complete: the avatar is
    /// ready and a camera is selected.
    ///
    /// A successful import or an explicit camera selection arms this start,
    /// which then fires once the lifecycle reports ready. Starting (manually
    /// or automatically) and stopping disarm it, so a user-requested stop is
    /// not overridden on the next frame.
    pub fn maybe_auto_start_tracking(&mut self) {
        if !self.auto_start_armed
            || self.pipeline_state != PipelineState::Idle
            || self.lifecycle_state != AvatarLifecycleState::Ready
            || self.selected_camera.is_none()
            || self.imported_model.is_none()
        {
            return;
        }
        self.start_pipeline();
    }

    /// Stop the tracking pipeline.
    fn stop_pipeline(&mut self) {
        self.auto_start_armed = false;
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

    /// Update the UI view model from current orchestrator state.
    ///
    /// The avatar `lifecycle` and `is_ready` fields are driven by the
    /// lifecycle mirror maintained by the sync system, not by the presence of
    /// an imported model. A model becomes ready only after the `bevy_vrm1`
    /// asset has initialized and humanoid binding has completed.
    pub fn update_view_model(&self, vm: &mut UiViewModel) {
        // The view model is rebuilt only when an input to it actually changed;
        // rebuilding it every frame clones strings and path buffers for the
        // camera list and imported-model summary even in the steady state.
        if !self.view_model_source_unchanged(vm) {
            self.rebuild_view_model(vm);
        }
    }

    /// Returns `true` when `vm` already reflects every current source value
    /// consumed by [`Self::rebuild_view_model`].
    fn view_model_source_unchanged(&self, vm: &UiViewModel) -> bool {
        let lifecycle = match self.pipeline_state {
            PipelineState::Idle => AppLifecycle::Idle,
            PipelineState::Starting => AppLifecycle::Starting,
            PipelineState::Running => AppLifecycle::Running,
            PipelineState::Stopping => AppLifecycle::Stopping,
            PipelineState::Failed => AppLifecycle::Failed,
        };
        if vm.pane != self.current_pane || vm.lifecycle != lifecycle {
            return false;
        }
        if vm.camera.selected_index != self.selected_camera
            || vm.camera.available_cameras.len() != self.cameras.len()
        {
            return false;
        }
        for (i, camera) in self.cameras.iter().enumerate() {
            match vm.camera.available_cameras.get(i) {
                Some(descriptor) if descriptor.name == camera.label => {}
                _ => return false,
            }
        }
        match (&vm.avatar.imported_model, &self.imported_model) {
            (None, None) => {}
            (Some(summary), model) => match model {
                Some(model) => {
                    let has_required_bones = model.summary.humanoid_nodes.hips < 1000
                        && model.summary.humanoid_nodes.head < 1000;
                    if summary.generation != model.summary.generation
                        || summary.id != model.id
                        || summary.name != model.name
                        || summary.original_path != model.original_path
                        || summary.has_required_bones != has_required_bones
                        || summary.expression_count != model.summary.expression_presets.len()
                    {
                        return false;
                    }
                }
                None => return false,
            },
            (None, Some(_)) => return false,
        }
        let pending_import = self.pending_avatar_import.as_ref();
        if vm.avatar_import_review.review.as_ref() != pending_import.map(|pending| &pending.review)
            || vm.avatar_import_review.accepted
                != pending_import.is_some_and(|pending| pending.accepted)
        {
            return false;
        }
        vm.avatar.lifecycle == self.lifecycle_state
    }

    /// Rebuilds every view-model field from current orchestrator state.
    fn rebuild_view_model(&self, vm: &mut UiViewModel) {
        // Pane.
        vm.pane = self.current_pane;

        // Lifecycle.
        vm.lifecycle = match self.pipeline_state {
            PipelineState::Idle => AppLifecycle::Idle,
            PipelineState::Starting => AppLifecycle::Starting,
            PipelineState::Running => AppLifecycle::Running,
            PipelineState::Stopping => AppLifecycle::Stopping,
            PipelineState::Failed => AppLifecycle::Failed,
        };

        // Camera — convert from vtuber_camera descriptors to UI model descriptors.
        vm.camera.available_cameras = self
            .cameras
            .iter()
            .enumerate()
            .map(|(i, c)| crate::ui_model::CameraDescriptor {
                name: c.label.clone(),
                index: i,
            })
            .collect();
        vm.camera.selected_index = self.selected_camera;

        // Avatar — imported model summary for display.
        vm.avatar.imported_model = self.imported_model.as_ref().map(|m| ImportedModelSummary {
            generation: m.summary.generation,
            id: m.id.clone(),
            name: m.name.clone(),
            original_path: m.original_path.clone(),
            has_required_bones: m.summary.humanoid_nodes.hips < 1000
                && m.summary.humanoid_nodes.head < 1000,
            expression_count: m.summary.expression_presets.len(),
        });

        // Avatar lifecycle — driven by the sync system, not by import state.
        vm.avatar.lifecycle = self.lifecycle_state;
        vm.avatar.is_ready = self.lifecycle_state == AvatarLifecycleState::Ready;
        vm.avatar.load_failed = self.lifecycle_state == AvatarLifecycleState::Failed;

        // License review — present only while an import waits for acceptance.
        vm.avatar_import_review.review = self
            .pending_avatar_import
            .as_ref()
            .map(|pending| pending.review.clone());
        vm.avatar_import_review.accepted = self
            .pending_avatar_import
            .as_ref()
            .is_some_and(|pending| pending.accepted);
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
    /// The sync system calls this once per request to obtain the data needed
    /// to construct a `LoadImportedAvatarRequest` message.
    pub fn take_pending_load_request(&mut self) -> Option<PendingLoadRequest> {
        self.pending_load.take()
    }

    /// Update the lifecycle state mirror.
    ///
    /// The sync system calls this after reading the `AvatarLifecycle` resource
    /// so that `update_view_model` can report the true lifecycle state.
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
    pub fn asset_root(&self) -> &PathBuf {
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

/// Reads a file for review under the same size policy as `import_vrm`.
fn read_reviewable_bytes(path: &Path) -> Result<Vec<u8>, VrmLicenseReviewError> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > import::DEFAULT_SIZE_LIMIT {
        return Err(VrmLicenseReviewError::SizeExceeded {
            size: metadata.len(),
            limit: import::DEFAULT_SIZE_LIMIT,
        });
    }
    Ok(std::fs::read(path)?)
}

/// Format an import error for user display.
fn format_import_error(error: &ModelImportError) -> String {
    match error {
        ModelImportError::InvalidExtension => "File must have .vrm extension".to_string(),
        ModelImportError::NotRegularFile => "Not a regular file".to_string(),
        ModelImportError::SizeExceeded { size, limit } => {
            format!("File size ({size} bytes) exceeds limit ({limit} bytes)")
        }
        ModelImportError::NotVrm { reason } => {
            format!("File is not a supported VRM model: {reason}")
        }
        ModelImportError::AmbiguousVrmVersion { reason } => {
            format!("Model declares both VRM generations: {reason}")
        }
        ModelImportError::UnsupportedVersion(v) => format!("Unsupported VRM version: {v}"),
        ModelImportError::DuplicateHumanBone(bone) => {
            format!("Duplicate human bone declaration: {bone}")
        }
        ModelImportError::MissingRequiredBone(bone) => format!("Missing required bone: {bone}"),
        ModelImportError::GlbParse(msg) => format!("Failed to parse model: {msg}"),
        ModelImportError::ExternalUri(uri) => format!("External URI not allowed: {uri}"),
        ModelImportError::InvalidNodeIndex { index } => {
            format!("Invalid node index: {index}")
        }
        ModelImportError::InvalidMeshIndex { index } => format!("Invalid mesh index: {index}"),
        ModelImportError::InvalidMorphTargetIndex { mesh, index } => {
            format!("Invalid morph target index {index} for mesh {mesh}")
        }
        ModelImportError::InvalidVrmField { path, reason } => {
            format!("Invalid VRM field {path}: {reason}")
        }
        ModelImportError::Io(e) => format!("I/O error: {e}"),
        ModelImportError::LimitExceedsHardCap { .. } => "Configuration error".to_string(),
    }
}

/// System that processes pending UI actions through the orchestrator.
#[allow(clippy::too_many_arguments)]
pub fn process_ui_actions_system(
    mut orchestrator: ResMut<Orchestrator>,
    mut ui_state: ResMut<UiState>,
    mut view_model: ResMut<UiViewModel>,
    mut ndi_intent: Option<ResMut<NdiOutputIntent>>,
    mut preview: ResMut<PreviewState>,
    mut avatar_motion_mirror: ResMut<AvatarMotionMirror>,
    mut arm_pose_overrides: Option<ResMut<ArmPoseOverrideStore>>,
    mut arm_pose_settings: Option<ResMut<ArmPoseSettings>>,
    mut arm_pose_changes: Option<MessageWriter<ArmPoseProfileChange>>,
    lifecycle: Option<Res<vtuber_avatar::AvatarLifecycle>>,
    mut reset_camera_requests: Option<MessageWriter<vtuber_avatar::ResetCameraRequest>>,
) {
    let actions = ui_state.take_actions();
    for action in &actions {
        match action {
            UiAction::TogglePreview => preview.toggle_visible(),
            UiAction::ToggleMirror => preview.toggle_mirrored(),
            UiAction::ToggleAvatarMotionMirror => avatar_motion_mirror.toggle(),
            UiAction::SetArmPoseProfile { profile } => {
                apply_arm_pose_profile_action(
                    &mut orchestrator,
                    *profile,
                    &mut arm_pose_overrides,
                    arm_pose_settings.as_deref(),
                    &mut arm_pose_changes,
                    false,
                );
            }
            UiAction::ResetArmPoseProfile => {
                reset_arm_pose_profile_action(
                    &mut orchestrator,
                    &mut arm_pose_overrides,
                    arm_pose_settings.as_deref(),
                    &mut arm_pose_changes,
                );
            }
            UiAction::ResetAvatarCamera => {
                if let (Some(lifecycle), Some(requests)) =
                    (lifecycle.as_deref(), reset_camera_requests.as_mut())
                    && lifecycle.state() == vtuber_avatar::AvatarLifecycleState::Ready
                {
                    requests.write(vtuber_avatar::ResetCameraRequest {
                        generation: lifecycle.current_generation(),
                    });
                }
                // Side-placed capture cameras observe a yawed face. Recentering
                // makes the currently observed facing direction the new front,
                // so the reset is visible in avatar head orientation instead of
                // only restoring the viewport orbit.
                orchestrator.process_action(&UiAction::BeginCalibration);
            }
            UiAction::StartNdiOutput => {
                if let Some(intent) = ndi_intent.as_deref_mut() {
                    intent.request_start();
                }
            }
            UiAction::StopNdiOutput => {
                if let Some(intent) = ndi_intent.as_deref_mut() {
                    intent.request_stop();
                }
            }
            UiAction::SetLanguage(language) => {
                if let Some(settings) = arm_pose_settings.as_mut()
                    && let Err(error) = settings.set_language(*language)
                {
                    orchestrator.set_last_error(Some(OrchestratorError::ArmPoseSettingsFailed(
                        error.to_string(),
                    )));
                }
            }
            _ => orchestrator.process_action(action),
        }
    }
    orchestrator.update_view_model(&mut view_model);
    sync_arm_pose_view_model(
        &orchestrator,
        &mut view_model,
        arm_pose_overrides.as_deref(),
    );
    view_model.preview_visible = preview.visible;
    view_model.mirror_preview = preview.mirrored;
    view_model.mirror_avatar_motion = avatar_motion_mirror.is_enabled();
}

fn apply_arm_pose_profile_action(
    orchestrator: &mut Orchestrator,
    profile: ArmPoseProfileOverride,
    overrides: &mut Option<ResMut<ArmPoseOverrideStore>>,
    settings: Option<&ArmPoseSettings>,
    changes: &mut Option<MessageWriter<ArmPoseProfileChange>>,
    return_to_default: bool,
) {
    let Some(model_id) = orchestrator.active_model_id().map(str::to_owned) else {
        return;
    };
    let Some(store) = overrides.as_deref_mut() else {
        return;
    };
    if let Err(error) = store.set(model_id.clone(), profile) {
        orchestrator.set_last_error(Some(OrchestratorError::ArmPoseSettingsFailed(
            error.to_string(),
        )));
        return;
    }
    if let Some(settings) = settings
        && let Err(error) = settings.save(store)
    {
        orchestrator.set_last_error(Some(OrchestratorError::ArmPoseSettingsFailed(
            error.to_string(),
        )));
    }
    if let Some(changes) = changes.as_mut() {
        changes.write(ArmPoseProfileChange {
            model_id: AvatarAssetId::new(model_id),
            return_to_default,
        });
    }
}

fn reset_arm_pose_profile_action(
    orchestrator: &mut Orchestrator,
    overrides: &mut Option<ResMut<ArmPoseOverrideStore>>,
    settings: Option<&ArmPoseSettings>,
    changes: &mut Option<MessageWriter<ArmPoseProfileChange>>,
) {
    let Some(model_id) = orchestrator.active_model_id().map(str::to_owned) else {
        return;
    };
    let Some(store) = overrides.as_deref_mut() else {
        return;
    };
    store.reset(&AvatarAssetId::new(&model_id));
    if let Some(settings) = settings
        && let Err(error) = settings.save(store)
    {
        orchestrator.set_last_error(Some(OrchestratorError::ArmPoseSettingsFailed(
            error.to_string(),
        )));
    }
    if let Some(changes) = changes.as_mut() {
        changes.write(ArmPoseProfileChange {
            model_id: AvatarAssetId::new(model_id),
            return_to_default: true,
        });
    }
}

fn sync_arm_pose_view_model(
    orchestrator: &Orchestrator,
    view_model: &mut UiViewModel,
    overrides: Option<&ArmPoseOverrideStore>,
) {
    let Some(model_id) = orchestrator.active_model_id() else {
        view_model.arm_pose = ArmPoseViewModel::default();
        return;
    };
    let id = AvatarAssetId::new(model_id);
    let Some(overrides) = overrides else {
        view_model.arm_pose = ArmPoseViewModel::default();
        return;
    };
    let profile = overrides.profile_for(&id);
    view_model.arm_pose.profile = profile.unwrap_or_default();
    view_model.arm_pose.has_override = profile.is_some();
}

/// Converts the avatar lifecycle's internal state to the UI model's state.
fn map_avatar_lifecycle_state(
    state: vtuber_avatar::lifecycle::AvatarLifecycleState,
) -> AvatarLifecycleState {
    use vtuber_avatar::lifecycle::AvatarLifecycleState as Engine;
    match state {
        Engine::NoAvatar => AvatarLifecycleState::None,
        Engine::Loading => AvatarLifecycleState::Loading,
        Engine::Binding => AvatarLifecycleState::Binding,
        Engine::Ready => AvatarLifecycleState::Ready,
        Engine::Unloading => AvatarLifecycleState::Unloading,
        Engine::Failed => AvatarLifecycleState::Failed,
    }
}

/// System that bridges the orchestrator to the avatar lifecycle.
///
/// 1. Reads the [`AvatarLifecycle`] state and mirrors it into the orchestrator
///    so that `update_view_model` reports the true engine state.
/// 2. Drains any pending load request from the orchestrator and emits a
///    [`LoadImportedAvatarRequest`] message.
/// 3. Detects when the user has cleared the imported model while the lifecycle
///    still has an active avatar, and emits an [`UnloadAvatarRequest`].
///
/// This system must run after [`process_ui_actions_system`] so that it sees
/// the latest orchestrator mutations.
pub fn sync_avatar_lifecycle_system(
    mut orchestrator: ResMut<Orchestrator>,
    lifecycle: Res<vtuber_avatar::lifecycle::AvatarLifecycle>,
    mut load_requests: MessageWriter<vtuber_avatar::LoadImportedAvatarRequest>,
    mut load_results: MessageReader<vtuber_avatar::LoadImportedAvatarResult>,
    mut unload_requests: MessageWriter<vtuber_avatar::lifecycle::UnloadAvatarRequest>,
) {
    // 1. Mirror the lifecycle state into the orchestrator.
    let engine_state = lifecycle.state();
    let ui_state = map_avatar_lifecycle_state(engine_state);
    orchestrator.set_lifecycle_state(ui_state);

    // Consume the request result so a malformed path or lifecycle rejection
    // becomes a recoverable UI error instead of remaining invisible. Accepted
    // requests are intentionally not treated as ready: readiness is driven by
    // bevy_vrm1 initialization and humanoid binding below.
    for result in load_results.read() {
        if let vtuber_avatar::LoadImportedAvatarResult::Rejected { error, .. } = result {
            let message = error.to_string();
            orchestrator.set_last_error(Some(OrchestratorError::AvatarLoadRejected(message)));
            orchestrator.set_lifecycle_state(AvatarLifecycleState::Failed);
        }
    }

    // A load can also fail after acceptance, for example when the asset
    // handle never initializes or binding times out. Preserve that typed
    // reason for the UI instead of reducing every failure to `Failed`.
    if engine_state == vtuber_avatar::lifecycle::AvatarLifecycleState::Failed
        && !matches!(
            orchestrator.last_error(),
            Some(OrchestratorError::AvatarLoadRejected(_))
        )
    {
        let message = lifecycle
            .failure()
            .map(ToString::to_string)
            .unwrap_or_else(|| "avatar lifecycle failed without a recorded reason".to_string());
        orchestrator.set_last_error(Some(OrchestratorError::AvatarLifecycleFailed(message)));
    }

    // 2. Drain pending load requests.
    if let Some(pending) = orchestrator.take_pending_load_request() {
        let id = vtuber_avatar::AvatarAssetId::new(&pending.model.id);
        let asset_path = vtuber_avatar::UserAssetPath::avatar_model_path(&id);

        match asset_path {
            Ok(path) => {
                let expected_generation = match pending.model.summary.generation {
                    VrmGeneration::Vrm0 => vtuber_avatar::ExpectedVrmGeneration::Vrm0,
                    VrmGeneration::Vrm1 => vtuber_avatar::ExpectedVrmGeneration::Vrm1,
                };
                let imported = vtuber_avatar::ImportedAvatar::new(
                    id,
                    path,
                    pending.model.name.clone(),
                    expected_generation,
                );
                load_requests.write(vtuber_avatar::LoadImportedAvatarRequest {
                    request_id: pending.request_id,
                    imported,
                });
            }
            Err(e) => {
                // Should never happen for a well-formed SHA-256 id, but handle
                // gracefully rather than panicking.
                bevy::log::error!(
                    "failed to construct user asset path for import {}: {e}",
                    pending.model.id
                );
                orchestrator.set_lifecycle_state(AvatarLifecycleState::Failed);
                orchestrator
                    .set_last_error(Some(OrchestratorError::AvatarLoadRejected(e.to_string())));
            }
        }
    }

    // 3. Detect unload: model cleared while lifecycle is still active.
    if !orchestrator.has_imported_model() {
        use vtuber_avatar::lifecycle::AvatarLifecycleState as Engine;
        match engine_state {
            Engine::Ready | Engine::Loading | Engine::Binding => {
                unload_requests.write(vtuber_avatar::lifecycle::UnloadAvatarRequest);
            }
            Engine::NoAvatar | Engine::Unloading | Engine::Failed => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preview::PreviewState;

    #[test]
    fn orchestrator_default_state() {
        let orch = Orchestrator::default();
        assert_eq!(orch.import_state(), &ImportState::Idle);
        assert!(orch.last_error().is_none());
        assert!(orch.camera_refresh_requested());
    }

    #[test]
    fn preview_actions_update_preview_state_and_view_model() {
        let mut app = App::new();
        app.init_resource::<Orchestrator>()
            .init_resource::<UiState>()
            .init_resource::<UiViewModel>()
            .init_resource::<PreviewState>()
            .init_resource::<AvatarMotionMirror>()
            .add_systems(Update, process_ui_actions_system);

        {
            let mut actions = app.world_mut().resource_mut::<UiState>();
            actions.emit(UiAction::TogglePreview);
            actions.emit(UiAction::ToggleMirror);
            actions.emit(UiAction::ToggleAvatarMotionMirror);
        }
        app.update();

        let preview = app.world().resource::<PreviewState>();
        assert!(!preview.visible);
        assert!(!preview.mirrored);

        let view_model = app.world().resource::<UiViewModel>();
        assert!(!view_model.preview_visible);
        assert!(!view_model.mirror_preview);
        assert!(!view_model.mirror_avatar_motion);
    }

    #[test]
    fn ndi_output_actions_update_session_intent_without_starting_tracking() {
        let mut app = App::new();
        app.init_resource::<Orchestrator>()
            .init_resource::<UiState>()
            .init_resource::<UiViewModel>()
            .init_resource::<crate::ndi_output::NdiOutputIntent>()
            .init_resource::<PreviewState>()
            .init_resource::<AvatarMotionMirror>()
            .add_systems(Update, process_ui_actions_system);

        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::StartNdiOutput);
        app.update();
        let intent = app.world().resource::<crate::ndi_output::NdiOutputIntent>();
        assert!(intent.is_requested());
        assert_eq!(intent.generation(), 1);
        assert_eq!(
            app.world().resource::<Orchestrator>().pipeline_state(),
            PipelineState::Idle
        );

        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::StopNdiOutput);
        app.update();
        assert!(
            !app.world()
                .resource::<crate::ndi_output::NdiOutputIntent>()
                .is_requested()
        );
    }

    #[test]
    fn reset_camera_action_bridges_the_ready_generation_once() {
        let mut app = App::new();
        app.init_resource::<Orchestrator>()
            .init_resource::<UiState>()
            .init_resource::<UiViewModel>()
            .init_resource::<PreviewState>()
            .init_resource::<AvatarMotionMirror>()
            .init_resource::<vtuber_avatar::AvatarLifecycle>()
            .add_message::<vtuber_avatar::ResetCameraRequest>()
            .add_systems(Update, process_ui_actions_system);

        let root = app.world_mut().spawn_empty().id();
        let generation = {
            let mut lifecycle = app
                .world_mut()
                .resource_mut::<vtuber_avatar::AvatarLifecycle>();
            lifecycle.request_load(root).expect("test load is valid");
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
            lifecycle.current_generation()
        };
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .set_pipeline_state(PipelineState::Running);
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ResetAvatarCamera);
        app.update();

        let messages = app
            .world()
            .resource::<Messages<vtuber_avatar::ResetCameraRequest>>();
        let mut cursor = messages.get_cursor();
        let requests = cursor.read(messages).collect::<Vec<_>>();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].generation, generation);
        assert_eq!(
            app.world_mut()
                .resource_mut::<Orchestrator>()
                .take_calibration_request(),
            Some(CalibrationRequest::Begin)
        );
    }

    #[test]
    fn reset_camera_action_skips_tracking_recenter_while_idle() {
        let mut app = App::new();
        app.init_resource::<Orchestrator>()
            .init_resource::<UiState>()
            .init_resource::<UiViewModel>()
            .init_resource::<PreviewState>()
            .init_resource::<AvatarMotionMirror>()
            .init_resource::<vtuber_avatar::AvatarLifecycle>()
            .add_message::<vtuber_avatar::ResetCameraRequest>()
            .add_systems(Update, process_ui_actions_system);

        let root = app.world_mut().spawn_empty().id();
        {
            let mut lifecycle = app
                .world_mut()
                .resource_mut::<vtuber_avatar::AvatarLifecycle>();
            lifecycle.request_load(root).expect("test load is valid");
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
        }
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ResetAvatarCamera);
        app.update();

        assert!(
            app.world_mut()
                .resource_mut::<Orchestrator>()
                .take_calibration_request()
                .is_none()
        );
    }

    #[test]
    fn arm_pose_settings_action_updates_runtime_store_and_persists_reset() {
        let directory = tempfile::tempdir().expect("temporary settings directory");
        let path = directory.path().join("settings.toml");
        let model_id = "sha256:active".to_string();
        let profile = vtuber_avatar::ArmPoseProfile {
            arm_drop_radians: 0.4,
            ..Default::default()
        };
        let mut app = App::new();
        app.init_resource::<Orchestrator>()
            .init_resource::<UiState>()
            .init_resource::<UiViewModel>()
            .init_resource::<PreviewState>()
            .init_resource::<AvatarMotionMirror>()
            .init_resource::<vtuber_avatar::ArmPoseOverrideStore>()
            .insert_resource(ArmPoseSettings::empty_at(&path))
            .add_message::<ArmPoseProfileChange>()
            .add_systems(Update, process_ui_actions_system);

        app.world_mut()
            .resource_mut::<Orchestrator>()
            .imported_model = Some(ImportedModel {
            id: model_id.clone(),
            name: "test".to_string(),
            asset_path: directory.path().join("model.vrm"),
            meta_path: directory.path().join("import.toml"),
            summary: crate::import::VrmInspectionSummary::default(),
            original_path: directory.path().join("original.vrm"),
            size: 1,
        });
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::SetArmPoseProfile {
                profile: ArmPoseProfileOverride::from_profile(profile),
            });
        app.update();

        let id = vtuber_avatar::AvatarAssetId::new(&model_id);
        let store = app
            .world()
            .resource::<vtuber_avatar::ArmPoseOverrideStore>();
        assert_eq!(store.profile_for(&id), Some(profile));
        assert!(path.is_file());
        assert_eq!(
            app.world().resource::<UiViewModel>().arm_pose.profile,
            profile
        );

        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ResetArmPoseProfile);
        app.update();

        assert!(
            app.world()
                .resource::<vtuber_avatar::ArmPoseOverrideStore>()
                .profile_for(&id)
                .is_none()
        );
        let restored = crate::settings::load_arm_pose_overrides(&path).expect("reset file");
        assert!(restored.profile_for(&id).is_none());
    }

    #[test]
    fn orchestrator_refresh_cameras() {
        let mut orch = Orchestrator::default();
        orch.process_action(&UiAction::RefreshCameras);
        // Refresh only signals enumeration; it must not manufacture devices.
        assert!(orch.cameras.is_empty());
        orch.set_camera_list(vec![
            CameraDescriptor {
                id: "test:0".into(),
                label: "Test camera 0".into(),
            },
            CameraDescriptor {
                id: "test:1".into(),
                label: "Test camera 1".into(),
            },
        ]);
        assert_eq!(orch.cameras.len(), 2);
    }

    #[test]
    fn orchestrator_select_camera() {
        let mut orch = Orchestrator::default();
        orch.process_action(&UiAction::RefreshCameras);
        orch.set_camera_list(vec![CameraDescriptor {
            id: "test:0".into(),
            label: "Test camera 0".into(),
        }]);
        orch.process_action(&UiAction::SelectCamera { index: 0 });
        assert_eq!(orch.selected_camera, Some(0));
    }

    #[test]
    fn orchestrator_select_invalid_camera_ignored() {
        let mut orch = Orchestrator::default();
        orch.process_action(&UiAction::RefreshCameras);
        orch.set_camera_list(vec![CameraDescriptor {
            id: "test:0".into(),
            label: "Test camera 0".into(),
        }]);
        orch.process_action(&UiAction::SelectCamera { index: 99 });
        assert_eq!(orch.selected_camera, None);
    }

    #[test]
    fn camera_refresh_preserves_selected_identity_across_reordering() {
        let mut orch = Orchestrator::default();
        let c922 = CameraDescriptor {
            id: "msmf:c922-symbolic-link".into(),
            label: "C922".into(),
        };
        let elecom = CameraDescriptor {
            id: "msmf:elecom-symbolic-link".into(),
            label: "ELECOM".into(),
        };
        orch.set_camera_list(vec![c922.clone(), elecom.clone()]);
        orch.process_action(&UiAction::SelectCamera { index: 0 });

        orch.set_camera_list(vec![elecom, c922.clone()]);

        assert_eq!(orch.selected_camera, Some(1));
        assert_eq!(orch.selected_camera_descriptor(), Some(c922));
    }

    #[test]
    fn orchestrator_unload_avatar() {
        let mut orch = Orchestrator {
            imported_model: Some(ImportedModel {
                id: "test".into(),
                name: "test".into(),
                asset_path: PathBuf::new(),
                meta_path: PathBuf::new(),
                summary: Default::default(),
                original_path: PathBuf::new(),
                size: 0,
            }),
            ..Default::default()
        };
        orch.process_action(&UiAction::UnloadAvatar);
        assert!(orch.imported_model.is_none());
    }

    #[test]
    fn orchestrator_start_without_camera_sets_error() {
        let mut orch = Orchestrator {
            imported_model: Some(ImportedModel {
                id: "test".into(),
                name: "test".into(),
                asset_path: PathBuf::new(),
                meta_path: PathBuf::new(),
                summary: Default::default(),
                original_path: PathBuf::new(),
                size: 0,
            }),
            ..Default::default()
        };
        orch.process_action(&UiAction::Start);
        assert_eq!(
            orch.last_error(),
            Some(&OrchestratorError::NoCameraSelected)
        );
    }

    #[test]
    fn orchestrator_start_without_avatar_sets_error() {
        let mut orch = Orchestrator {
            selected_camera: Some(0),
            ..Default::default()
        };
        orch.process_action(&UiAction::Start);
        assert_eq!(orch.last_error(), Some(&OrchestratorError::NoAvatarLoaded));
    }

    #[test]
    fn auto_start_fires_when_setup_is_complete() {
        let mut orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            selected_camera: Some(0),
            lifecycle_state: AvatarLifecycleState::Ready,
            ..Default::default()
        };

        orch.maybe_auto_start_tracking();

        assert_eq!(orch.pipeline_state(), PipelineState::Starting);
        assert!(orch.capture_desired());
        assert!(!orch.capture_ack());
    }

    #[test]
    fn auto_start_waits_for_ready_lifecycle() {
        let mut orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            selected_camera: Some(0),
            lifecycle_state: AvatarLifecycleState::Binding,
            ..Default::default()
        };

        orch.maybe_auto_start_tracking();

        assert_eq!(orch.pipeline_state(), PipelineState::Idle);
        assert!(!orch.capture_desired());
    }

    #[test]
    fn auto_start_waits_for_camera_selection() {
        let mut orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            lifecycle_state: AvatarLifecycleState::Ready,
            ..Default::default()
        };

        orch.maybe_auto_start_tracking();

        assert_eq!(orch.pipeline_state(), PipelineState::Idle);
        assert!(!orch.capture_desired());
    }

    #[test]
    fn manual_stop_disarms_auto_start() {
        let mut orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            selected_camera: Some(0),
            lifecycle_state: AvatarLifecycleState::Ready,
            ..Default::default()
        };
        orch.maybe_auto_start_tracking();
        orch.process_action(&UiAction::Stop);
        orch.complete_capture_stop();
        assert_eq!(orch.pipeline_state(), PipelineState::Idle);

        orch.maybe_auto_start_tracking();

        assert_eq!(orch.pipeline_state(), PipelineState::Idle);
        assert!(!orch.capture_desired());
    }

    #[test]
    fn selecting_a_camera_rearms_auto_start_after_stop() {
        let mut orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            selected_camera: Some(0),
            lifecycle_state: AvatarLifecycleState::Ready,
            ..Default::default()
        };
        orch.maybe_auto_start_tracking();
        orch.process_action(&UiAction::Stop);
        orch.complete_capture_stop();

        orch.set_camera_list(vec![CameraDescriptor {
            id: "test:0".into(),
            label: "Test camera".into(),
        }]);
        orch.process_action(&UiAction::SelectCamera { index: 0 });
        orch.maybe_auto_start_tracking();

        assert_eq!(orch.pipeline_state(), PipelineState::Starting);
        assert!(orch.capture_desired());
    }

    #[test]
    fn orchestrator_dismiss_error() {
        let mut orch = Orchestrator {
            last_error: Some(OrchestratorError::NoCameraSelected),
            ..Default::default()
        };
        orch.process_action(&UiAction::DismissError);
        assert!(orch.last_error().is_none());
    }

    #[test]
    fn orchestrator_update_view_model() {
        let mut orch = Orchestrator::default();
        orch.process_action(&UiAction::RefreshCameras);
        orch.set_camera_list(vec![
            CameraDescriptor {
                id: "test:0".into(),
                label: "Test camera 0".into(),
            },
            CameraDescriptor {
                id: "test:1".into(),
                label: "Test camera 1".into(),
            },
        ]);
        orch.process_action(&UiAction::SelectCamera { index: 0 });

        let mut vm = UiViewModel::default();
        orch.update_view_model(&mut vm);

        assert_eq!(vm.camera.available_cameras.len(), 2);
        assert_eq!(vm.camera.selected_index, Some(0));
    }

    #[test]
    fn format_import_error_not_vrm() {
        let err = ModelImportError::NotVrm {
            reason: "missing VRM or VRMC_vrm extension".into(),
        };
        let msg = format_import_error(&err);
        assert!(msg.contains("supported VRM"));
    }

    #[test]
    fn format_import_error_missing_bone() {
        let err = ModelImportError::MissingRequiredBone("hips".to_string());
        let msg = format_import_error(&err);
        assert!(msg.contains("hips"));
    }

    fn stub_imported_model() -> ImportedModel {
        ImportedModel {
            id: "abc123".into(),
            name: "Test Model".into(),
            asset_path: PathBuf::new(),
            meta_path: PathBuf::new(),
            summary: Default::default(),
            original_path: PathBuf::new(),
            size: 0,
        }
    }

    #[test]
    fn orchestrator_unload_clears_pending_load() {
        let mut orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            pending_load: Some(PendingLoadRequest {
                request_id: 1,
                model: stub_imported_model(),
            }),
            ..Default::default()
        };
        orch.process_action(&UiAction::UnloadAvatar);
        assert!(orch.imported_model.is_none());
        assert!(orch.take_pending_load_request().is_none());
    }

    #[test]
    fn orchestrator_retry_after_failure_creates_pending_load() {
        let model = stub_imported_model();
        let mut orch = Orchestrator {
            imported_model: Some(model.clone()),
            lifecycle_state: AvatarLifecycleState::Failed,
            ..Default::default()
        };
        orch.process_action(&UiAction::RetryAfterError);
        let pending = orch
            .take_pending_load_request()
            .expect("should have pending load");
        assert_eq!(pending.request_id, 1);
        assert_eq!(pending.model.id, model.id);
        assert_eq!(orch.lifecycle_state, AvatarLifecycleState::None);
    }

    #[test]
    fn orchestrator_retry_ignored_when_not_failed() {
        let mut orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            lifecycle_state: AvatarLifecycleState::Ready,
            ..Default::default()
        };
        orch.process_action(&UiAction::RetryAfterError);
        assert!(orch.take_pending_load_request().is_none());
    }

    #[test]
    fn orchestrator_retry_after_inference_failure_restarts_only_inference() {
        let mut orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            capture_desired: true,
            capture_ack: true,
            pipeline_state: PipelineState::Failed,
            last_error: Some(OrchestratorError::InferenceFailed("model failed".into())),
            ..Default::default()
        };

        orch.process_action(&UiAction::RetryAfterError);

        assert_eq!(orch.pipeline_state, PipelineState::Starting);
        assert!(orch.capture_desired);
        assert!(orch.capture_ack);
        assert!(orch.last_error.is_none());
        assert!(orch.take_inference_retry_request());
        assert!(!orch.take_inference_retry_request());
        assert!(orch.take_pending_load_request().is_none());
    }

    #[test]
    fn inference_failure_requests_reverse_order_capture_shutdown() {
        let mut orch = Orchestrator {
            capture_desired: true,
            capture_ack: true,
            pipeline_state: PipelineState::Running,
            ..Default::default()
        };

        orch.fail_inference("landmark failed".into());

        assert_eq!(orch.pipeline_state, PipelineState::Stopping);
        assert!(!orch.capture_desired);
        assert!(!orch.capture_ack);
        assert!(matches!(
            orch.last_error(),
            Some(OrchestratorError::InferenceFailed(message)) if message == "landmark failed"
        ));

        orch.complete_capture_stop();
        assert_eq!(orch.pipeline_state, PipelineState::Failed);
    }

    #[test]
    fn retry_after_failed_inference_restarts_capture_when_it_was_stopped() {
        let mut orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            selected_camera: Some(0),
            capture_desired: false,
            capture_ack: true,
            pipeline_state: PipelineState::Failed,
            last_error: Some(OrchestratorError::InferenceFailed("model failed".into())),
            ..Default::default()
        };

        orch.process_action(&UiAction::RetryAfterError);

        assert_eq!(orch.pipeline_state, PipelineState::Starting);
        assert!(orch.capture_desired);
        assert!(!orch.capture_ack);
        assert!(orch.take_inference_retry_request());
    }

    #[test]
    fn orchestrator_view_model_reflects_lifecycle_not_import() {
        let orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            lifecycle_state: AvatarLifecycleState::Loading,
            ..Default::default()
        };
        let mut vm = UiViewModel::default();
        orch.update_view_model(&mut vm);

        // Model is imported but lifecycle is Loading, so is_ready must be false.
        assert!(vm.avatar.imported_model.is_some());
        assert!(!vm.avatar.is_ready);
        assert_eq!(vm.avatar.lifecycle, AvatarLifecycleState::Loading);
        assert!(!vm.avatar.load_failed);
    }

    #[test]
    fn orchestrator_view_model_ready_only_when_lifecycle_ready() {
        let orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            lifecycle_state: AvatarLifecycleState::Ready,
            ..Default::default()
        };
        let mut vm = UiViewModel::default();
        orch.update_view_model(&mut vm);

        assert!(vm.avatar.is_ready);
        assert_eq!(vm.avatar.lifecycle, AvatarLifecycleState::Ready);
        assert!(!vm.avatar.load_failed);
    }

    #[test]
    fn orchestrator_view_model_failed_sets_load_failed() {
        let orch = Orchestrator {
            imported_model: Some(stub_imported_model()),
            lifecycle_state: AvatarLifecycleState::Failed,
            ..Default::default()
        };
        let mut vm = UiViewModel::default();
        orch.update_view_model(&mut vm);

        assert!(!vm.avatar.is_ready);
        assert!(vm.avatar.load_failed);
        assert_eq!(vm.avatar.lifecycle, AvatarLifecycleState::Failed);
    }

    #[test]
    fn map_lifecycle_state_round_trip() {
        use vtuber_avatar::lifecycle::AvatarLifecycleState as Engine;

        assert_eq!(
            map_avatar_lifecycle_state(Engine::NoAvatar),
            AvatarLifecycleState::None
        );
        assert_eq!(
            map_avatar_lifecycle_state(Engine::Loading),
            AvatarLifecycleState::Loading
        );
        assert_eq!(
            map_avatar_lifecycle_state(Engine::Binding),
            AvatarLifecycleState::Binding
        );
        assert_eq!(
            map_avatar_lifecycle_state(Engine::Ready),
            AvatarLifecycleState::Ready
        );
        assert_eq!(
            map_avatar_lifecycle_state(Engine::Unloading),
            AvatarLifecycleState::Unloading
        );
        assert_eq!(
            map_avatar_lifecycle_state(Engine::Failed),
            AvatarLifecycleState::Failed
        );
    }

    const REVIEW_FIXTURE_JSON: &str = r#"{
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Hips", "children": [1]}, {"name": "Head"}],
        "extensionsUsed": ["VRMC_vrm"],
        "extensions": {
            "VRMC_vrm": {
                "specVersion": "1.0",
                "meta": {
                    "name": "Review Fixture",
                    "authors": ["Fixture Author"],
                    "avatarPermission": "onlyAuthor",
                    "allowExcessivelyViolentUsage": false,
                    "allowExcessivelySexualUsage": false,
                    "commercialUsage": "personalNonProfit",
                    "modification": "prohibited",
                    "licenseUrl": "https://vrm.dev/licenses/1.0/"
                },
                "humanoid": {
                    "humanBones": {
                        "hips": {"node": 0},
                        "head": {"node": 1}
                    }
                }
            }
        }
    }"#;

    fn write_review_fixture(dir: &tempfile::TempDir) -> PathBuf {
        let mut json_chunk = REVIEW_FIXTURE_JSON.as_bytes().to_vec();
        while !json_chunk.len().is_multiple_of(4) {
            json_chunk.push(b' ');
        }
        let bin_chunk = [0_u8; 12];
        let total_length = 12 + 8 + json_chunk.len() + 8 + bin_chunk.len();
        let mut bytes = Vec::with_capacity(total_length);
        bytes.extend_from_slice(&0x46546C67_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&(total_length as u32).to_le_bytes());
        bytes.extend_from_slice(&(json_chunk.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0x4E4F534A_u32.to_le_bytes());
        bytes.extend_from_slice(&json_chunk);
        bytes.extend_from_slice(&(bin_chunk.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0x004E4942_u32.to_le_bytes());
        bytes.extend_from_slice(&bin_chunk);

        let path = dir.path().join("review.vrm");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn review_request_holds_the_model_until_explicit_acceptance() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_review_fixture(&dir);
        let mut orch = Orchestrator::new(dir.path().join("assets"));

        orch.process_action(&UiAction::RequestAvatarImportReview { path: path.clone() });

        let pending = orch.pending_avatar_import.as_ref().expect("review pending");
        assert_eq!(pending.path, path);
        assert!(!pending.accepted);
        assert!(!orch.has_imported_model());

        orch.process_action(&UiAction::SetAvatarImportReviewAccepted { accepted: true });
        assert!(
            orch.pending_avatar_import
                .as_ref()
                .expect("review pending")
                .accepted
        );
        assert!(!orch.has_imported_model());

        orch.process_action(&UiAction::AcceptAvatarImportReview);

        assert!(orch.has_imported_model());
        assert!(orch.pending_avatar_import.is_none());
        assert!(orch.take_pending_load_request().is_some());
    }

    #[test]
    fn import_is_blocked_while_the_review_is_unchecked() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_review_fixture(&dir);
        let mut orch = Orchestrator::new(dir.path().join("assets"));

        orch.process_action(&UiAction::RequestAvatarImportReview { path });
        orch.process_action(&UiAction::AcceptAvatarImportReview);

        assert!(!orch.has_imported_model());
        assert!(orch.pending_avatar_import.is_some());
    }

    #[test]
    fn cancel_discards_the_pending_review_and_allows_a_fresh_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_review_fixture(&dir);
        let mut orch = Orchestrator::new(dir.path().join("assets"));

        orch.process_action(&UiAction::RequestAvatarImportReview { path: path.clone() });
        orch.process_action(&UiAction::SetAvatarImportReviewAccepted { accepted: true });
        orch.process_action(&UiAction::CancelAvatarImportReview);

        assert!(orch.pending_avatar_import.is_none());
        assert!(!orch.has_imported_model());

        // The same file must be reviewed again; acceptance is never remembered.
        orch.process_action(&UiAction::RequestAvatarImportReview { path });
        let pending = orch.pending_avatar_import.as_ref().expect("review pending");
        assert!(!pending.accepted);
    }

    #[test]
    fn unreviewable_model_is_rejected_without_a_pending_review() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.vrm");
        std::fs::write(&path, b"not a glb").unwrap();
        let mut orch = Orchestrator::new(dir.path().join("assets"));

        orch.process_action(&UiAction::RequestAvatarImportReview { path });

        assert!(orch.pending_avatar_import.is_none());
        assert!(!orch.has_imported_model());
        assert!(matches!(
            orch.last_error(),
            Some(OrchestratorError::LicenseReviewFailed(_))
        ));
    }

    #[test]
    fn oversized_file_is_rejected_before_reading() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oversized.vrm");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(import::DEFAULT_SIZE_LIMIT + 1).unwrap();
        drop(file);
        let mut orch = Orchestrator::new(dir.path().join("assets"));

        orch.process_action(&UiAction::RequestAvatarImportReview { path });

        assert!(orch.pending_avatar_import.is_none());
        assert!(matches!(
            orch.last_error(),
            Some(OrchestratorError::LicenseReviewFailed(_))
        ));
    }

    #[test]
    fn view_model_exposes_review_and_acceptance_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_review_fixture(&dir);
        let mut orch = Orchestrator::new(dir.path().join("assets"));
        let mut vm = UiViewModel::default();

        orch.process_action(&UiAction::RequestAvatarImportReview { path });
        orch.process_action(&UiAction::SetAvatarImportReviewAccepted { accepted: true });
        orch.update_view_model(&mut vm);

        assert!(vm.avatar_import_review.review.is_some());
        assert!(vm.avatar_import_review.accepted);

        orch.process_action(&UiAction::CancelAvatarImportReview);
        orch.update_view_model(&mut vm);

        assert!(vm.avatar_import_review.review.is_none());
        assert!(!vm.avatar_import_review.accepted);
    }
}
