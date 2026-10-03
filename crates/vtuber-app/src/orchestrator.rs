//! App orchestrator — processes UI actions and manages domain state.
//!
//! The orchestrator receives [`UiAction`](crate::actions::UiAction) commands from
//! the UI layer and translates them into domain service calls (camera, import,
//! tracking, etc.). It updates the [`UiViewModel`](crate::ui_model::UiViewModel)
//! snapshot that the UI reads each frame.
//!
//! Avatar loading is bridged to the `vtuber-avatar` lifecycle through a
//! pending-request protocol: after a successful import the orchestrator stores
//! a [`PendingLoadRequest`](crate::orchestrator::PendingLoadRequest). A Bevy system
//! drains it and emits the corresponding `LoadImportedAvatarRequest` message
//! that the avatar plugin consumes.
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
use crate::import::{self, ImportedModel, ModelImportError};
use crate::license_review::{self, VrmLicenseReview, VrmLicenseReviewError};
use crate::ui_model::*;

mod action_system;
mod expressions;
mod lifecycle;

pub use action_system::{LookSystemParams, process_ui_actions_system};
pub use expressions::sync_expression_view_model;
pub use lifecycle::sync_avatar_lifecycle_system;

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

/// Model and saved look waiting for the corresponding lifecycle result.
#[derive(Debug)]
struct SubmittedAvatarLoad {
    model: ImportedModel,
    look: Option<vtuber_avatar::RichLookSettings>,
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
    /// Requests submitted to the engine but not yet confirmed.
    submitted_loads: BTreeMap<u64, SubmittedAvatarLoad>,
    /// Selected VRM awaiting license acceptance.
    pending_avatar_import: Option<PendingAvatarImport>,
    /// Next avatar load request correlation identifier.
    next_load_request_id: u64,
    /// Mirror of the avatar lifecycle state, updated by the sync system.
    lifecycle_state: crate::ui_model::AvatarLifecycleState,
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
            current_pane: Pane::default(),
            pending_load: None,
            submitted_loads: BTreeMap::new(),
            pending_avatar_import: None,
            next_load_request_id: 1,
            lifecycle_state: AvatarLifecycleState::None,
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

        match import::import_vrm(path, &self.asset_root, import::DEFAULT_SIZE_LIMIT) {
            Ok(model) => {
                self.queue_imported_model(model);
            }
            Err(e) => {
                let msg = format_import_error(&e);
                self.import_state = ImportState::Failed(msg.clone());
                self.last_error = Some(OrchestratorError::ImportFailed(msg));
            }
        }
    }

    /// Queues an imported model, including CLI startup, without replacing the
    /// accepted model or its look until the engine confirms the request.
    pub fn queue_imported_model(&mut self, model: ImportedModel) {
        let request_id = self.next_load_request_id;
        self.next_load_request_id += 1;
        self.pending_load = Some(PendingLoadRequest { request_id, model });
        self.import_state = ImportState::Success;
        self.last_error = None;
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

#[cfg(test)]
mod tests;
