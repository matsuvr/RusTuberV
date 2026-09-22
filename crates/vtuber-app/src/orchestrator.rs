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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use crate::actions::UiAction;
use crate::expression_keys::{
    ExpressionBindingStore, ExpressionBindings, ExpressionKey, can_assign_expression,
    effective_bindings, manual_request_for_key,
};
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
    AvatarExpressionCatalog, AvatarGeneration, AvatarLifecycle, AvatarMotionMirror,
    ManualExpressionRequest, ManualExpressionSelection,
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

/// Model and saved look waiting for the corresponding lifecycle result.
#[derive(Debug)]
struct SubmittedAvatarLoad {
    model: ImportedModel,
    look: Option<vtuber_avatar::RichLookSettings>,
    roles: Vec<vtuber_avatar::MaterialRoleOverride>,
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
    /// Persistent expression-key bindings could not be written.
    ExpressionSettingsFailed(String),
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
            Self::ExpressionSettingsFailed(msg) => {
                write!(f, "Expression settings failed: {msg}")
            }
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
            submitted_loads: BTreeMap::new(),
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
    /// On success a [`PendingLoadRequest`] is queued while the accepted model
    /// remains selected. The sync system drains the pending request and emits a
    /// `LoadImportedAvatarRequest` that the avatar lifecycle consumes.
    fn import_avatar(&mut self, path: &PathBuf) {
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
    /// Clears the imported model and any pending load request. The sync system
    /// detects the removal and emits an `UnloadAvatarRequest`.
    fn unload_avatar(&mut self) {
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

/// The look-related system parameters grouped to stay within the system
/// parameter limit: the live settings, the material roles and their change
/// messages.
#[derive(SystemParam)]
pub struct LookSystemParams<'w> {
    look_settings: Option<ResMut<'w, vtuber_avatar::AvatarLookSettings>>,
    look_changes: Option<MessageWriter<'w, vtuber_avatar::LookSettingsChanged>>,
    material_roles: Option<Res<'w, vtuber_avatar::AvatarMaterialRoles>>,
    role_changes: Option<MessageWriter<'w, vtuber_avatar::MaterialRoleOverridesChanged>>,
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
    mut expression_store: Option<ResMut<ExpressionBindingStore>>,
    mut manual_requests: Option<MessageWriter<ManualExpressionRequest>>,
    mut pose_runtime: Option<ResMut<crate::pose_runtime::PoseRuntime>>,
    mut look: LookSystemParams,
) {
    let actions = ui_state.take_actions();
    for action in &actions {
        match action {
            UiAction::TogglePreview => preview.toggle_visible(),
            UiAction::ToggleMirror => preview.toggle_mirrored(),
            UiAction::ToggleAvatarMotionMirror => avatar_motion_mirror.toggle(),
            UiAction::SetArmTrackingEnabled { enabled } => {
                if let Some(pose) = pose_runtime.as_deref_mut() {
                    pose.set_enabled(*enabled);
                    if !*enabled {
                        // Dropping the observation state also returns the arms
                        // to the virtual anchors on the next compositor frame.
                        pose.recalibrate();
                    }
                }
                if let Some(settings) = arm_pose_settings.as_mut()
                    && let Err(error) = settings.set_arm_tracking_enabled(*enabled)
                {
                    orchestrator.set_last_error(Some(OrchestratorError::ArmPoseSettingsFailed(
                        error.to_string(),
                    )));
                }
            }
            UiAction::RecalibrateArms => {
                if let Some(pose) = pose_runtime.as_deref_mut() {
                    pose.recalibrate();
                }
            }
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
            UiAction::UnloadAvatar => {
                orchestrator.process_action(action);
                if let Some(persistent) = arm_pose_settings.as_deref_mut() {
                    persistent.look_model_id = None;
                }
                if let (Some(look), Some(changes)) = (
                    look.look_settings.as_deref_mut(),
                    look.look_changes.as_mut(),
                ) {
                    look.0 = vtuber_avatar::RichLookSettings::default();
                    changes.write(vtuber_avatar::LookSettingsChanged(look.0));
                }
                if let Some(role_changes) = look.role_changes.as_mut() {
                    role_changes.write(vtuber_avatar::MaterialRoleOverridesChanged(Vec::new()));
                }
            }
            UiAction::ChangeRichLook(change) => {
                apply_rich_look_action(
                    look.look_settings.as_deref_mut(),
                    look.look_changes.as_mut(),
                    *change,
                );
            }
            UiAction::SetMaterialRole {
                material_index,
                selected,
            } => {
                apply_material_role_action(
                    &mut orchestrator,
                    *material_index,
                    *selected,
                    look.material_roles.as_deref(),
                    look.role_changes.as_mut(),
                    arm_pose_settings.as_deref(),
                );
            }
            UiAction::SaveRichLook => {
                if let (Some(persistent), Some(look)) =
                    (arm_pose_settings.as_deref(), look.look_settings.as_deref())
                {
                    let result = persistent
                        .look_model_id
                        .as_ref()
                        .ok_or(OrchestratorError::NoAvatarLoaded)
                        .and_then(|id| {
                            persistent
                                .save_rich_look(id.clone(), look.0)
                                .map_err(|error| {
                                    OrchestratorError::ArmPoseSettingsFailed(error.to_string())
                                })
                        });
                    if let Err(error) = result {
                        orchestrator.set_last_error(Some(error));
                    }
                }
            }
            UiAction::AssignExpressionKey {
                model_id,
                generation,
                key,
                expression,
            } => {
                apply_expression_binding_action(
                    &mut orchestrator,
                    ExpressionBindingAction::Assign {
                        target: ExpressionTarget {
                            model_id: model_id.clone(),
                            generation: *generation,
                        },
                        key: *key,
                        expression: expression.clone(),
                    },
                    &mut expression_store,
                    arm_pose_settings.as_deref(),
                    lifecycle.as_deref(),
                    &mut manual_requests,
                );
            }
            UiAction::ResetExpressionBindings {
                model_id,
                generation,
            } => {
                apply_expression_binding_action(
                    &mut orchestrator,
                    ExpressionBindingAction::Reset {
                        target: ExpressionTarget {
                            model_id: model_id.clone(),
                            generation: *generation,
                        },
                    },
                    &mut expression_store,
                    arm_pose_settings.as_deref(),
                    lifecycle.as_deref(),
                    &mut manual_requests,
                );
            }
            UiAction::ToggleExpressionKey { generation, key } => {
                toggle_expression_key_action(
                    &orchestrator,
                    *generation,
                    *key,
                    expression_store.as_deref(),
                    lifecycle.as_deref(),
                    &mut manual_requests,
                );
            }
            UiAction::ClearManualExpression { generation } => {
                clear_manual_expression_action(*generation, &mut manual_requests);
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
    view_model.arm_tracking_enabled = pose_runtime.as_deref().is_some_and(|pose| pose.enabled());
    if let Some(settings) = look.look_settings.as_deref() {
        view_model.look.enabled = settings.0.enabled;
        view_model.look.strength = settings.0.strength;
    }
}

/// Applies one rich-look edit.
///
/// The resource is the immediate state the settings screen renders, and the
/// message carries the same value so every look listener sees the change.
fn apply_rich_look_action(
    settings: Option<&mut vtuber_avatar::AvatarLookSettings>,
    changes: Option<&mut MessageWriter<vtuber_avatar::LookSettingsChanged>>,
    change: crate::actions::RichLookChange,
) {
    let (Some(settings), Some(changes)) = (settings, changes) else {
        return;
    };
    let next = crate::actions::reduce_rich_look(settings.0, change);
    if next == settings.0 {
        return;
    }
    settings.0 = next;
    changes.write(vtuber_avatar::LookSettingsChanged(next));
}

/// Applies one material-role selection: commit-copy-save.
///
/// The runtime resource is updated through the same replace-all message the
/// avatar side consumes, and the selection is saved immediately. A failed save
/// keeps the live value and reports the existing typed error.
fn apply_material_role_action(
    orchestrator: &mut Orchestrator,
    material_index: usize,
    selected: Option<vtuber_avatar::MaterialRole>,
    roles: Option<&vtuber_avatar::AvatarMaterialRoles>,
    role_changes: Option<&mut MessageWriter<vtuber_avatar::MaterialRoleOverridesChanged>>,
    settings: Option<&ArmPoseSettings>,
) {
    let (Some(roles), Some(role_changes)) = (roles, role_changes) else {
        return;
    };
    let previous: Vec<vtuber_avatar::MaterialRoleOverride> = roles
        .entries()
        .into_iter()
        .map(|(index, _, current)| vtuber_avatar::MaterialRoleOverride {
            material_index: index,
            selected: current,
        })
        .collect();
    let mut next = previous.clone();
    match next
        .iter_mut()
        .find(|entry| entry.material_index == material_index)
    {
        Some(entry) => entry.selected = selected,
        None => next.push(vtuber_avatar::MaterialRoleOverride {
            material_index,
            selected,
        }),
    }
    if next == previous {
        return;
    }
    role_changes.write(vtuber_avatar::MaterialRoleOverridesChanged(next.clone()));
    if let Some(settings) = settings {
        let result = settings
            .look_model_id
            .as_ref()
            .ok_or(OrchestratorError::NoAvatarLoaded)
            .and_then(|model_id| {
                settings
                    .save_material_roles(model_id.clone(), next)
                    .map_err(|error| OrchestratorError::ArmPoseSettingsFailed(error.to_string()))
            });
        if let Err(error) = result {
            orchestrator.set_last_error(Some(error));
        }
    }
}

/// Rebuilds the material-role view model whenever the runtime roles change.
pub fn sync_look_material_view_model(
    roles: Option<Res<vtuber_avatar::AvatarMaterialRoles>>,
    mut view_model: ResMut<UiViewModel>,
    mut last_change: Local<Option<bevy::ecs::change_detection::Tick>>,
) {
    let Some(roles) = roles else {
        return;
    };
    let change_tick = roles.last_changed();
    if *last_change == Some(change_tick) {
        return;
    }
    *last_change = Some(change_tick);
    view_model.look_materials = roles
        .entries()
        .into_iter()
        .map(
            |(material_index, name, selected)| MaterialRoleEntryViewModel {
                material_index,
                name: name.to_owned(),
                selected,
            },
        )
        .collect();
}

/// Returns the one catalog that expression operations may use right now.
///
/// The imported-model ID and the render-side catalog can disagree while a
/// pending load has not reached the lifecycle yet. Only a `Ready` avatar whose
/// catalog generation matches the lifecycle generation and whose catalog
/// model matches the orchestrator's imported model yields a consistent
/// snapshot; otherwise expression operations are unavailable.
fn current_expression_catalog<'a>(
    orchestrator: &Orchestrator,
    lifecycle: &'a AvatarLifecycle,
) -> Option<&'a AvatarExpressionCatalog> {
    let catalog = lifecycle.expression_catalog()?;
    (lifecycle.state() == vtuber_avatar::AvatarLifecycleState::Ready
        && catalog.generation == lifecycle.current_generation().0
        && orchestrator.active_model_id() == Some(catalog.model_id.as_str()))
    .then_some(catalog)
}

/// Model and generation captured when the UI issued an action.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpressionTarget {
    model_id: String,
    generation: AvatarGeneration,
}

impl ExpressionTarget {
    /// Returns `true` when the consistent catalog still matches the target
    /// captured at issuance.
    fn is_current(&self, orchestrator: &Orchestrator, lifecycle: &AvatarLifecycle) -> bool {
        current_expression_catalog(orchestrator, lifecycle).is_some_and(|catalog| {
            catalog.model_id == self.model_id && catalog.generation == self.generation.0
        })
    }
}

/// Expression key mutation requested by the UI.
enum ExpressionBindingAction {
    /// Assign an expression (or unassign with `None`) to one key.
    Assign {
        target: ExpressionTarget,
        key: ExpressionKey,
        expression: Option<String>,
    },
    /// Restore the target model's deterministic default assignment.
    Reset { target: ExpressionTarget },
}

impl ExpressionBindingAction {
    fn target(&self) -> &ExpressionTarget {
        match self {
            Self::Assign { target, .. } | Self::Reset { target } => target,
        }
    }
}

/// Applies one expression binding mutation through copy-save-commit.
///
/// The target model and generation must still be current; an action collected
/// before a model swap never changes the new model's settings. The candidate
/// is written to `settings.toml` first; only a successful save commits it to
/// the runtime store and clears the manual selection. A failed save keeps the
/// previous assignment and reports the existing typed error.
fn apply_expression_binding_action(
    orchestrator: &mut Orchestrator,
    action: ExpressionBindingAction,
    store: &mut Option<ResMut<ExpressionBindingStore>>,
    settings: Option<&ArmPoseSettings>,
    lifecycle: Option<&AvatarLifecycle>,
    manual_requests: &mut Option<MessageWriter<ManualExpressionRequest>>,
) {
    let Some(lifecycle) = lifecycle else {
        return;
    };
    let target = action.target().clone();
    if !target.is_current(orchestrator, lifecycle) {
        return;
    }
    let catalog = lifecycle.expression_catalog();
    let model_id = target.model_id;
    let Some(store) = store.as_deref_mut() else {
        return;
    };
    let current = effective_bindings(store, &model_id, catalog);
    let mut candidate = current.clone();
    match action {
        ExpressionBindingAction::Assign {
            expression, key, ..
        } => match expression {
            Some(expression) => {
                if !can_assign_expression(catalog, &expression) {
                    return;
                }
                candidate.assign(key, expression);
            }
            None => candidate.unassign(key),
        },
        ExpressionBindingAction::Reset { .. } => {
            let Some(catalog) = catalog else {
                return;
            };
            candidate.reset_to_defaults(catalog);
        }
    }
    if candidate == current {
        return;
    }
    let mut next = store.clone();
    next.set(model_id, candidate);
    if let Some(settings) = settings
        && let Err(error) = settings.save_expression_bindings(&next)
    {
        orchestrator.set_last_error(Some(OrchestratorError::ExpressionSettingsFailed(
            error.to_string(),
        )));
        return;
    }
    *store = next;
    if let Some(requests) = manual_requests.as_mut() {
        requests.write(ManualExpressionRequest::Clear {
            generation: target.generation,
        });
    }
}

/// Resolves the key's binding snapshot and emits a toggle intent for the
/// generation the input was collected against.
///
/// Missing or not-ready expressions are dropped rather than substituted, and
/// the issued generation is passed through unchanged.
fn toggle_expression_key_action(
    orchestrator: &Orchestrator,
    generation: AvatarGeneration,
    key: ExpressionKey,
    store: Option<&ExpressionBindingStore>,
    lifecycle: Option<&AvatarLifecycle>,
    manual_requests: &mut Option<MessageWriter<ManualExpressionRequest>>,
) {
    let Some(lifecycle) = lifecycle else {
        return;
    };
    // The catalog must be the one consistent with the live model; a pending
    // replacement must not contribute its saved bindings to an old-generation
    // toggle.
    let Some(catalog) = current_expression_catalog(orchestrator, lifecycle) else {
        return;
    };
    if catalog.generation != generation.0 {
        return;
    }
    let Some(store) = store else {
        return;
    };
    let bindings = effective_bindings(store, &catalog.model_id, Some(catalog));
    let Some(expression) = bindings.expression_for(key) else {
        return;
    };
    if !can_assign_expression(Some(catalog), expression) {
        return;
    }
    let Some(request) = manual_request_for_key(generation, key, &bindings) else {
        return;
    };
    if let Some(requests) = manual_requests.as_mut() {
        requests.write(request);
    }
}

/// Emits a manual clear for exactly the generation the UI acted on.
fn clear_manual_expression_action(
    generation: AvatarGeneration,
    manual_requests: &mut Option<MessageWriter<ManualExpressionRequest>>,
) {
    let Some(requests) = manual_requests.as_mut() else {
        return;
    };
    requests.write(ManualExpressionRequest::Clear { generation });
}

/// Cheap view-model rebuild key: model, catalog generation, store revision,
/// manual generation, and the selected ID.
type ExpressionViewModelSignature = (String, u64, u64, u64, String);

/// Rebuilds the expression catalog/binding view model.
///
/// Registered by the shell after manual request processing so the selected
/// marker is consistent with the same frame that handled the toggle.
pub fn sync_expression_view_model(
    orchestrator: Res<Orchestrator>,
    lifecycle: Option<Res<AvatarLifecycle>>,
    store: Option<Res<ExpressionBindingStore>>,
    manual: Option<Res<ManualExpressionSelection>>,
    mut view_model: ResMut<UiViewModel>,
    mut last_signature: Local<Option<ExpressionViewModelSignature>>,
) {
    // A pending import may have updated the orchestrator model while the
    // render-side catalog is still the old one. Only the consistent catalog
    // may produce an operable snapshot; the avatar itself keeps rendering.
    let catalog = match lifecycle.as_deref() {
        Some(lifecycle) => current_expression_catalog(&orchestrator, lifecycle),
        None => None,
    };
    let signature = (
        catalog.map_or_else(String::new, |catalog| catalog.model_id.clone()),
        catalog.map_or(0, |catalog| catalog.generation),
        store.as_deref().map_or(0, ExpressionBindingStore::revision),
        manual.as_deref().map_or(0, |manual| manual.generation.0),
        manual
            .as_deref()
            .and_then(|manual| manual.selected.clone())
            .unwrap_or_default(),
    );
    if last_signature.as_ref() == Some(&signature) {
        return;
    }
    *last_signature = Some(signature);

    let selected = manual.as_deref().and_then(|manual| manual.selected.clone());
    // Actions emitted from this snapshot carry exactly the model/generation
    // shown here so a later swap cannot apply them to the new avatar.
    let snapshot_model_id = catalog.map(|catalog| catalog.model_id.clone());
    let snapshot_generation = catalog.map(|catalog| AvatarGeneration(catalog.generation));
    let bindings = match (catalog, store.as_deref()) {
        (Some(catalog), Some(store)) => effective_bindings(store, &catalog.model_id, Some(catalog)),
        (Some(catalog), None) => ExpressionBindings::default_for(catalog),
        (None, _) => ExpressionBindings::default(),
    };
    let entries = catalog
        .map(|catalog| {
            catalog
                .selectable_entries()
                .iter()
                .map(|entry| ExpressionEntryViewModel {
                    id: entry.id.clone(),
                    source_name: entry.source_name.clone(),
                    kind: entry.kind,
                    availability: entry.availability.clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    let binding_rows = ExpressionKey::ALL
        .into_iter()
        .map(|key| {
            let expression = bindings.expression_for(key).map(str::to_owned);
            ExpressionKeyBindingViewModel {
                key,
                selected: expression
                    .as_deref()
                    .is_some_and(|id| Some(id) == selected.as_deref()),
                expression,
            }
        })
        .collect();
    view_model.expression = ExpressionViewModel {
        model_id: snapshot_model_id,
        generation: snapshot_generation,
        has_catalog: catalog.is_some(),
        entries,
        bindings: binding_rows,
        selected,
    };
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

fn prepare_avatar_load(
    pending: PendingLoadRequest,
    persistent: Option<&ArmPoseSettings>,
) -> Result<
    (
        vtuber_avatar::LoadImportedAvatarRequest,
        SubmittedAvatarLoad,
    ),
    OrchestratorError,
> {
    let look = persistent
        .map(|settings| settings.rich_look_for(&pending.model.id))
        .transpose()
        .map_err(|error| OrchestratorError::ArmPoseSettingsFailed(error.to_string()))?;
    let roles = persistent
        .map(|settings| settings.material_roles_for(&pending.model.id))
        .transpose()
        .map_err(|error| OrchestratorError::ArmPoseSettingsFailed(error.to_string()))?
        .unwrap_or_default();
    let id = AvatarAssetId::new(&pending.model.id);
    let path = vtuber_avatar::UserAssetPath::avatar_model_path(&id)
        .map_err(|error| OrchestratorError::AvatarLoadRejected(error.to_string()))?;
    let expected_generation = match pending.model.summary.generation {
        VrmGeneration::Vrm0 => vtuber_avatar::ExpectedVrmGeneration::Vrm0,
        VrmGeneration::Vrm1 => vtuber_avatar::ExpectedVrmGeneration::Vrm1,
    };
    let imported =
        vtuber_avatar::ImportedAvatar::new(id, path, &pending.model.name, expected_generation);
    Ok((
        vtuber_avatar::LoadImportedAvatarRequest {
            request_id: pending.request_id,
            imported,
        },
        SubmittedAvatarLoad {
            model: pending.model,
            look,
            roles,
        },
    ))
}

/// Applies already-read settings only after the model load is accepted.
#[allow(clippy::too_many_arguments)]
fn restore_model_look(
    model_id: &str,
    restored: vtuber_avatar::RichLookSettings,
    roles: Vec<vtuber_avatar::MaterialRoleOverride>,
    persistent: &mut ArmPoseSettings,
    look: &mut vtuber_avatar::AvatarLookSettings,
    changes: &mut MessageWriter<vtuber_avatar::LookSettingsChanged>,
    role_changes: &mut MessageWriter<vtuber_avatar::MaterialRoleOverridesChanged>,
) {
    persistent.look_model_id = Some(model_id.to_owned());
    look.0 = restored;
    changes.write(vtuber_avatar::LookSettingsChanged(restored));
    role_changes.write(vtuber_avatar::MaterialRoleOverridesChanged(roles));
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
/// Runs after UI actions and the engine's load/request/unload systems, so
/// accepted results commit the model, look, save owner and UI snapshot together.
#[allow(clippy::too_many_arguments)]
pub fn sync_avatar_lifecycle_system(
    mut orchestrator: ResMut<Orchestrator>,
    lifecycle: Res<vtuber_avatar::lifecycle::AvatarLifecycle>,
    mut load_requests: MessageWriter<vtuber_avatar::LoadImportedAvatarRequest>,
    mut load_results: MessageReader<vtuber_avatar::LoadImportedAvatarResult>,
    mut unload_requests: MessageWriter<vtuber_avatar::lifecycle::UnloadAvatarRequest>,
    mut persistent: Option<ResMut<ArmPoseSettings>>,
    mut look: Option<ResMut<vtuber_avatar::AvatarLookSettings>>,
    mut changes: Option<MessageWriter<vtuber_avatar::LookSettingsChanged>>,
    mut role_changes: Option<MessageWriter<vtuber_avatar::MaterialRoleOverridesChanged>>,
    mut view_model: Option<ResMut<UiViewModel>>,
) {
    // 1. Mirror the lifecycle state into the orchestrator.
    let engine_state = lifecycle.state();
    let ui_state = map_avatar_lifecycle_state(engine_state);
    orchestrator.set_lifecycle_state(ui_state);

    // Only acceptance commits the selected model and its prepared look. A
    // rejection reports the error while the previously accepted model continues.
    for result in load_results.read() {
        match result {
            vtuber_avatar::LoadImportedAvatarResult::Accepted { request_id, .. } => {
                if let Some(submitted) = orchestrator.submitted_loads.remove(request_id) {
                    if let (
                        Some(restored),
                        Some(persistent),
                        Some(look),
                        Some(changes),
                        Some(role_changes),
                    ) = (
                        submitted.look,
                        persistent.as_deref_mut(),
                        look.as_deref_mut(),
                        changes.as_mut(),
                        role_changes.as_mut(),
                    ) {
                        restore_model_look(
                            &submitted.model.id,
                            restored,
                            submitted.roles,
                            persistent,
                            look,
                            changes,
                            role_changes,
                        );
                    }
                    orchestrator.imported_model = Some(submitted.model);
                    orchestrator.auto_start_armed = true;
                }
            }
            vtuber_avatar::LoadImportedAvatarResult::Rejected { request_id, error } => {
                orchestrator.submitted_loads.remove(request_id);
                orchestrator.set_last_error(Some(OrchestratorError::AvatarLoadRejected(
                    error.to_string(),
                )));
            }
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

    // 2. Read the selected model's settings without changing the accepted model.
    if let Some(pending) = orchestrator.take_pending_load_request() {
        match prepare_avatar_load(pending, persistent.as_deref()) {
            Ok((request, submitted)) => {
                orchestrator
                    .submitted_loads
                    .insert(request.request_id, submitted);
                load_requests.write(request);
            }
            Err(error) => orchestrator.set_last_error(Some(error)),
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
    if let Some(view_model) = view_model.as_deref_mut() {
        orchestrator.update_view_model(view_model);
        if let Some(look) = look.as_deref() {
            view_model.look.enabled = look.0.enabled;
            view_model.look.strength = look.0.strength;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preview::PreviewState;

    fn rich_look_app(path: &std::path::Path) -> App {
        let mut app = App::new();
        app.init_resource::<Orchestrator>()
            .init_resource::<UiState>()
            .init_resource::<UiViewModel>()
            .init_resource::<PreviewState>()
            .init_resource::<AvatarMotionMirror>()
            .init_resource::<NdiOutputIntent>()
            .init_resource::<AvatarLifecycle>()
            .init_resource::<vtuber_avatar::AvatarLookSettings>()
            .init_resource::<vtuber_avatar::AvatarMaterialRoles>()
            .init_resource::<vtuber_avatar::StandardLookBases>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<bevy_vrm1::prelude::MToonMaterial>>()
            .insert_resource(ArmPoseSettings::empty_at(path))
            .add_message::<vtuber_avatar::LookSettingsChanged>()
            .add_message::<vtuber_avatar::MaterialRoleOverridesChanged>()
            .add_message::<vtuber_avatar::LoadImportedAvatarRequest>()
            .add_message::<vtuber_avatar::LoadImportedAvatarResult>()
            .add_message::<vtuber_avatar::lifecycle::UnloadAvatarRequest>()
            .add_systems(
                Update,
                (
                    process_ui_actions_system,
                    sync_avatar_lifecycle_system,
                    vtuber_avatar::look::clear_look_materials_on_unload,
                    vtuber_avatar::look::apply_look_settings_changes,
                    vtuber_avatar::look::apply_material_role_overrides,
                    vtuber_avatar::look::initialize_look_materials,
                    vtuber_avatar::look::initialize_mtoon_look_materials,
                )
                    .chain(),
            );
        use vtuber_avatar::lifecycle::*;
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<bevy_vrm1::prelude::VrmAsset>()
            .add_message::<LoadAvatarRequest>()
            .add_message::<LoadAvatarResult>()
            .add_message::<ReplaceAvatarRequest>()
            .add_message::<ReplaceAvatarResult>()
            .add_message::<UnloadAvatarResult>()
            .add_systems(
                Update,
                (
                    vtuber_avatar::load::handle_load_imported_avatar_requests,
                    apply_avatar_request_events,
                    vtuber_avatar::unload::despawn_unloading_avatar,
                )
                    .chain()
                    .before(sync_avatar_lifecycle_system),
            );
        app
    }

    fn select_look_model(app: &mut App, suffix: char) {
        let model =
            stub_imported_model_with_id(&format!("sha256:{}", suffix.to_string().repeat(64)));
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .queue_imported_model(model);
        app.update(); // submit; the caller's next update receives acceptance/rejection.
    }

    fn look_action(app: &mut App, action: UiAction) {
        app.world_mut().resource_mut::<UiState>().emit(action);
        app.update();
    }

    #[test]
    fn rich_look_live_edits_commit_switch_and_restart_restore_per_model() {
        use crate::actions::RichLookChange;
        use vtuber_avatar::{AvatarLookSettings, RichLookSettings};
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = rich_look_app(&path);
        select_look_model(&mut app, 'a');
        app.update();
        finish_look_model(&mut app);
        let preview = app.world().resource::<PreviewState>().visible;
        let ndi_generation = app.world().resource::<NdiOutputIntent>().generation();
        look_action(
            &mut app,
            UiAction::ChangeRichLook(RichLookChange::Enabled(true)),
        );
        for strength in [1.0, 0.5, 0.0] {
            look_action(
                &mut app,
                UiAction::ChangeRichLook(RichLookChange::Strength(strength)),
            );
            assert_eq!(
                app.world().resource::<AvatarLookSettings>().0,
                RichLookSettings {
                    enabled: true,
                    strength
                }
            );
            assert_eq!(
                app.world().resource::<UiViewModel>().look.strength,
                strength
            );
            assert!(!path.exists(), "live edits must not write files");
        }
        look_action(&mut app, UiAction::SaveRichLook);
        let zero = RichLookSettings {
            enabled: true,
            strength: 0.0,
        };
        let saved = std::fs::read_to_string(&path).unwrap();
        look_action(
            &mut app,
            UiAction::ChangeRichLook(RichLookChange::Strength(0.5)),
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), saved);
        look_action(
            &mut app,
            UiAction::ChangeRichLook(RichLookChange::Enabled(false)),
        );
        look_action(
            &mut app,
            UiAction::ChangeRichLook(RichLookChange::Enabled(true)),
        );
        assert_eq!(app.world().resource::<AvatarLookSettings>().0.strength, 0.5);
        assert_eq!(app.world().resource::<PreviewState>().visible, preview);
        assert!(!app.world().resource::<NdiOutputIntent>().is_requested());
        assert_eq!(
            app.world().resource::<NdiOutputIntent>().generation(),
            ndi_generation
        );
        select_look_model(&mut app, 'b');
        app.update();
        finish_look_model(&mut app);
        assert_eq!(
            app.world().resource::<AvatarLookSettings>().0,
            RichLookSettings::default()
        );
        look_action(
            &mut app,
            UiAction::ChangeRichLook(RichLookChange::Strength(0.5)),
        );
        look_action(&mut app, UiAction::SaveRichLook);
        select_look_model(&mut app, 'a');
        app.update();
        finish_look_model(&mut app);
        assert_eq!(app.world().resource::<AvatarLookSettings>().0, zero);
        select_look_model(&mut app, 'b');
        app.update();
        finish_look_model(&mut app);
        assert_eq!(
            app.world().resource::<AvatarLookSettings>().0,
            RichLookSettings {
                enabled: false,
                strength: 0.5
            }
        );
        look_action(&mut app, UiAction::UnloadAvatar);
        assert_eq!(
            app.world().resource::<AvatarLookSettings>().0,
            RichLookSettings::default()
        );
        assert!(
            app.world()
                .resource::<ArmPoseSettings>()
                .look_model_id
                .is_none()
        );
        drop(app);
        let mut restarted = rich_look_app(&path);
        select_look_model(&mut restarted, 'a');
        restarted.update();
        finish_look_model(&mut restarted);
        assert_eq!(restarted.world().resource::<AvatarLookSettings>().0, zero);
    }

    #[test]
    fn material_role_selections_commit_save_and_restore_per_model() {
        use vtuber_avatar::MaterialRole;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = rich_look_app(&path);
        select_look_model(&mut app, 'a');
        app.update();
        finish_look_model(&mut app);
        look_action(
            &mut app,
            UiAction::SetMaterialRole {
                material_index: 2,
                selected: Some(MaterialRole::Face),
            },
        );
        assert_eq!(
            app.world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(2),
            MaterialRole::Face
        );
        assert!(path.is_file(), "role selection saves immediately");
        look_action(
            &mut app,
            UiAction::SetMaterialRole {
                material_index: 2,
                selected: None,
            },
        );
        assert_eq!(
            app.world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(2),
            MaterialRole::General
        );
        look_action(
            &mut app,
            UiAction::SetMaterialRole {
                material_index: 2,
                selected: Some(MaterialRole::Face),
            },
        );

        // A second model starts on Auto and keeps its own selection.
        select_look_model(&mut app, 'b');
        app.update();
        finish_look_model(&mut app);
        assert_eq!(
            app.world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(2),
            MaterialRole::General
        );
        look_action(
            &mut app,
            UiAction::SetMaterialRole {
                material_index: 4,
                selected: Some(MaterialRole::Hair),
            },
        );
        look_action(
            &mut app,
            UiAction::SetMaterialRole {
                material_index: 2,
                selected: Some(MaterialRole::Skin),
            },
        );

        select_look_model(&mut app, 'a');
        app.update();
        finish_look_model(&mut app);
        assert_eq!(
            app.world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(2),
            MaterialRole::Face,
            "model A's selection is restored on switch"
        );
        assert_eq!(
            app.world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(4),
            MaterialRole::General
        );
        select_look_model(&mut app, 'b');
        app.update();
        finish_look_model(&mut app);
        assert_eq!(
            app.world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(2),
            MaterialRole::Skin
        );
        assert_eq!(
            app.world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(4),
            MaterialRole::Hair
        );

        // A restart restores both models' selections from the settings file.
        drop(app);
        let mut restarted = rich_look_app(&path);
        select_look_model(&mut restarted, 'a');
        restarted.update();
        finish_look_model(&mut restarted);
        assert_eq!(
            restarted
                .world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(2),
            MaterialRole::Face
        );
        select_look_model(&mut restarted, 'b');
        restarted.update();
        finish_look_model(&mut restarted);
        assert_eq!(
            restarted
                .world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(2),
            MaterialRole::Skin
        );
        assert_eq!(
            restarted
                .world()
                .resource::<vtuber_avatar::AvatarMaterialRoles>()
                .role(4),
            MaterialRole::Hair
        );
    }

    #[test]
    fn rich_look_load_errors_reach_ui_without_submitting_new_model() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        std::fs::write(&path, "broken = [").unwrap();
        let mut app = rich_look_app(&path);
        select_look_model(&mut app, 'a');
        app.update();
        assert!(matches!(
            app.world().resource::<Orchestrator>().last_error(),
            Some(OrchestratorError::ArmPoseSettingsFailed(_))
        ));
        assert!(
            app.world()
                .resource::<Messages<vtuber_avatar::LoadImportedAvatarRequest>>()
                .is_empty()
        );
    }

    #[test]
    fn rich_look_save_errors_reach_ui_and_keep_live_value() {
        use crate::actions::RichLookChange;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = rich_look_app(&path);
        select_look_model(&mut app, 'a');
        app.update();
        std::fs::create_dir(&path).unwrap();
        look_action(
            &mut app,
            UiAction::ChangeRichLook(RichLookChange::Enabled(true)),
        );
        look_action(&mut app, UiAction::SaveRichLook);
        assert!(
            app.world()
                .resource::<vtuber_avatar::AvatarLookSettings>()
                .0
                .enabled
        );
        assert!(matches!(
            app.world().resource::<Orchestrator>().last_error(),
            Some(OrchestratorError::ArmPoseSettingsFailed(_))
        ));
    }

    fn rich_models(dir: &tempfile::TempDir) -> (ImportedModel, ImportedModel) {
        let first = write_review_fixture(dir);
        let mut bytes = std::fs::read(&first).unwrap();
        let index = bytes
            .windows(b"Review Fixture".len())
            .position(|part| part == b"Review Fixture")
            .unwrap();
        bytes[index] = b'B';
        let second = dir.path().join("second.vrm");
        std::fs::write(&second, bytes).unwrap();
        let root = dir.path().join("assets");
        (
            import::import_vrm(&first, &root, import::DEFAULT_SIZE_LIMIT).unwrap(),
            import::import_vrm(&second, &root, import::DEFAULT_SIZE_LIMIT).unwrap(),
        )
    }

    fn import_look_model(app: &mut App, model: &ImportedModel) {
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .import_avatar(&model.original_path);
        // First update submits; second runs the real handler and consumes its result.
        app.update();
        app.update();
    }

    fn finish_look_model(app: &mut App) {
        {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            let root = lifecycle.active_root().unwrap();
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
        }
        app.update();
    }

    fn assert_model_look(
        app: &App,
        model: &ImportedModel,
        expected: vtuber_avatar::RichLookSettings,
    ) {
        let world = app.world();
        let root = world.resource::<AvatarLifecycle>().active_root().unwrap();
        assert_eq!(
            world.get::<AvatarAssetId>(root).unwrap().0,
            model.id,
            "actual avatar"
        );
        assert_eq!(
            world.resource::<Orchestrator>().active_model_id(),
            Some(model.id.as_str()),
            "UI model owner"
        );
        assert_eq!(
            world.resource::<ArmPoseSettings>().look_model_id.as_deref(),
            Some(model.id.as_str()),
            "save owner"
        );
        assert_eq!(
            world.resource::<vtuber_avatar::AvatarLookSettings>().0,
            expected,
            "runtime look"
        );
        let vm = world.resource::<UiViewModel>();
        assert_eq!(
            vm.avatar.imported_model.as_ref().unwrap().id,
            model.id,
            "UI snapshot model"
        );
        assert_eq!(vm.look.enabled, expected.enabled);
        assert_eq!(vm.look.strength, expected.strength);
    }

    #[test]
    fn rich_lifecycle_rejected_switch_keeps_loading_model() {
        rejected_switch_keeps_model(false);
    }

    #[test]
    fn rich_lifecycle_rejected_switch_keeps_binding_model() {
        rejected_switch_keeps_model(true);
    }

    #[test]
    fn rich_lifecycle_parse_failure_keeps_model_and_allows_reselection() {
        restore_failure_keeps_model(false);
    }

    #[test]
    fn rich_lifecycle_read_failure_keeps_model_and_allows_reselection() {
        restore_failure_keeps_model(true);
    }

    fn rejected_switch_keeps_model(binding: bool) {
        use crate::actions::RichLookChange;
        use vtuber_avatar::{LoadImportedAvatarResult, RichLookSettings};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let (a, b) = rich_models(&dir);
        let a_look = RichLookSettings {
            enabled: true,
            strength: 0.25,
        };
        let b_look = RichLookSettings {
            enabled: false,
            strength: 0.75,
        };
        let settings = ArmPoseSettings::empty_at(&path);
        settings.save_rich_look(a.id.clone(), a_look).unwrap();
        settings.save_rich_look(b.id.clone(), b_look).unwrap();
        let mut app = rich_look_app(&path);
        app.world_mut().resource_mut::<Orchestrator>().asset_root = dir.path().join("assets");
        import_look_model(&mut app, &a);
        if binding {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            let root = lifecycle.active_root().unwrap();
            lifecycle.start_binding(root);
        }
        let mut results = app
            .world()
            .resource::<Messages<LoadImportedAvatarResult>>()
            .get_cursor_current();
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .import_avatar(&b.original_path);
        let request_id = app
            .world()
            .resource::<Orchestrator>()
            .pending_load
            .as_ref()
            .unwrap()
            .request_id;
        assert_model_look(&app, &a, a_look);
        app.update(); // prepared and submitted, but not accepted
        assert_model_look(&app, &a, a_look);
        app.update(); // production rejection and result handling
        assert_model_look(&app, &a, a_look);
        assert!(
            results
                .read(app.world().resource::<Messages<LoadImportedAvatarResult>>())
                .any(|result| matches!(result, LoadImportedAvatarResult::Rejected { request_id: rejected, .. } if *rejected == request_id))
        );
        finish_look_model(&mut app);
        assert_model_look(&app, &a, a_look);
        assert!(matches!(
            app.world().resource::<Orchestrator>().last_error(),
            Some(OrchestratorError::AvatarLoadRejected(_))
        ));
        look_action(
            &mut app,
            UiAction::ChangeRichLook(RichLookChange::Strength(0.5)),
        );
        look_action(&mut app, UiAction::SaveRichLook);
        assert_eq!(settings.rich_look_for(&a.id).unwrap().strength, 0.5);
        assert_eq!(settings.rich_look_for(&b.id).unwrap(), b_look);
    }

    fn restore_failure_keeps_model(read_error: bool) {
        use vtuber_avatar::{LoadImportedAvatarRequest, RichLookSettings};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let (a, b) = rich_models(&dir);
        let a_look = RichLookSettings {
            enabled: true,
            strength: 0.0,
        };
        let b_look = RichLookSettings {
            enabled: false,
            strength: 0.75,
        };
        let settings = ArmPoseSettings::empty_at(&path);
        settings.save_rich_look(a.id.clone(), a_look).unwrap();
        settings.save_rich_look(b.id.clone(), b_look).unwrap();
        let valid = std::fs::read_to_string(&path).unwrap();
        let mut app = rich_look_app(&path);
        app.world_mut().resource_mut::<Orchestrator>().asset_root = dir.path().join("assets");
        import_look_model(&mut app, &a);
        finish_look_model(&mut app);
        if read_error {
            std::fs::remove_file(&path).unwrap();
            std::fs::create_dir(&path).unwrap();
        } else {
            std::fs::write(&path, "broken = [").unwrap();
        }
        let mut requests = app
            .world()
            .resource::<Messages<LoadImportedAvatarRequest>>()
            .get_cursor_current();
        import_look_model(&mut app, &b);
        assert_eq!(
            requests
                .read(
                    app.world()
                        .resource::<Messages<LoadImportedAvatarRequest>>()
                )
                .count(),
            0
        );
        assert_model_look(&app, &a, a_look);
        assert_eq!(
            app.world().resource::<AvatarLifecycle>().state(),
            vtuber_avatar::AvatarLifecycleState::Ready
        );
        assert!(matches!(
            app.world().resource::<Orchestrator>().last_error(),
            Some(OrchestratorError::ArmPoseSettingsFailed(_))
        ));
        if read_error {
            std::fs::remove_dir(&path).unwrap();
        } else {
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "broken = [");
        }
        std::fs::write(&path, valid).unwrap();
        import_look_model(&mut app, &b);
        finish_look_model(&mut app);
        assert_model_look(&app, &b, b_look);
    }

    #[test]
    fn rich_lifecycle_accepted_switch_restores_and_saves_new_model() {
        use crate::actions::RichLookChange;
        use vtuber_avatar::RichLookSettings;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let (a, b) = rich_models(&dir);
        let a_look = RichLookSettings {
            enabled: false,
            strength: 0.25,
        };
        let b_look = RichLookSettings {
            enabled: true,
            strength: 0.0,
        };
        let settings = ArmPoseSettings::empty_at(&path);
        settings.save_rich_look(a.id.clone(), a_look).unwrap();
        settings.save_rich_look(b.id.clone(), b_look).unwrap();
        let mut app = rich_look_app(&path);
        app.world_mut().resource_mut::<Orchestrator>().asset_root = dir.path().join("assets");
        import_look_model(&mut app, &a);
        finish_look_model(&mut app);
        let old_root = app
            .world()
            .resource::<AvatarLifecycle>()
            .active_root()
            .unwrap();
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .import_avatar(&b.original_path);
        assert_model_look(&app, &a, a_look);
        app.update();
        assert_model_look(&app, &a, a_look);
        app.update();
        finish_look_model(&mut app);
        assert!(app.world().get_entity(old_root).is_err());
        assert_model_look(&app, &b, b_look);
        look_action(
            &mut app,
            UiAction::ChangeRichLook(RichLookChange::Strength(0.5)),
        );
        look_action(&mut app, UiAction::SaveRichLook);
        assert_eq!(settings.rich_look_for(&a.id).unwrap(), a_look);
        assert_eq!(
            settings.rich_look_for(&b.id).unwrap(),
            RichLookSettings {
                enabled: true,
                strength: 0.5
            }
        );
    }

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

        assert!(!orch.has_imported_model());
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

    fn expression_action_app(settings_path: PathBuf) -> App {
        let mut app = App::new();
        app.init_resource::<Orchestrator>()
            .init_resource::<UiState>()
            .init_resource::<UiViewModel>()
            .init_resource::<PreviewState>()
            .init_resource::<AvatarMotionMirror>()
            .init_resource::<vtuber_avatar::AvatarLifecycle>()
            .init_resource::<ExpressionBindingStore>()
            .init_resource::<vtuber_avatar::ManualExpressionSelection>()
            .init_resource::<crate::ndi_output::NdiOutputIntent>()
            .insert_resource(ArmPoseSettings::empty_at(settings_path))
            .add_message::<vtuber_avatar::ManualExpressionRequest>()
            .add_message::<vtuber_avatar::ArmPoseProfileChange>()
            .add_message::<vtuber_avatar::ResetCameraRequest>()
            .add_systems(Update, process_ui_actions_system);
        let root = app.world_mut().spawn_empty().id();
        let generation = {
            let mut lifecycle = app
                .world_mut()
                .resource_mut::<vtuber_avatar::AvatarLifecycle>();
            lifecycle.request_load(root).unwrap();
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
            lifecycle.current_generation()
        };
        let catalog = vtuber_avatar::AvatarExpressionCatalog::build(
            "model-a".into(),
            generation.0,
            [
                ("happy", true),
                ("angry", true),
                ("smile", false),
                ("custom49", false),
            ]
            .into_iter()
            .map(|(id, preset)| vtuber_avatar::ExpressionCatalogInput {
                id,
                declared_as_preset: preset,
                declared_morph_bind_count: 1,
                resolved_morph_bind_count: 1,
                declared_material_bind_count: 0,
                resolved_material_bind_count: 0,
                unresolved_material_bind_count: 0,
                unsupported_material_bind_count: 0,
            }),
        );
        app.world_mut()
            .resource_mut::<vtuber_avatar::AvatarLifecycle>()
            .set_expression_catalog(Some(catalog));
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .set_imported_model_for_tests(Some(stub_imported_model_with_id("model-a")));
        app
    }

    fn stub_imported_model_with_id(id: &str) -> ImportedModel {
        ImportedModel {
            id: id.into(),
            name: "Test Model".into(),
            asset_path: PathBuf::new(),
            meta_path: PathBuf::new(),
            summary: Default::default(),
            original_path: PathBuf::new(),
            size: 0,
        }
    }

    fn take_manual_requests(app: &mut App) -> Vec<vtuber_avatar::ManualExpressionRequest> {
        app.world_mut()
            .resource_mut::<Messages<vtuber_avatar::ManualExpressionRequest>>()
            .drain()
            .collect()
    }

    /// Model ID and generation of the currently loaded test avatar.
    fn expression_target(app: &App) -> (String, vtuber_avatar::AvatarGeneration) {
        (
            "model-a".to_string(),
            app.world()
                .resource::<vtuber_avatar::AvatarLifecycle>()
                .current_generation(),
        )
    }

    #[test]
    fn assigning_a_far_catalog_expression_persists_and_clears_manual() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = expression_action_app(path.clone());
        let (model_id, generation) = expression_target(&app);
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::AssignExpressionKey {
                model_id,
                generation,
                key: ExpressionKey::Digit1,
                expression: Some("custom49".into()),
            });
        app.update();

        let store = app.world().resource::<ExpressionBindingStore>();
        assert_eq!(
            store
                .bindings_for("model-a")
                .expect("saved entry")
                .expression_for(ExpressionKey::Digit1),
            Some("custom49")
        );
        let saved = crate::settings::load_expression_bindings(&path).expect("reload");
        assert_eq!(
            saved
                .bindings_for("model-a")
                .expect("persisted model")
                .expression_for(ExpressionKey::Digit1),
            Some("custom49")
        );
        assert!(matches!(
            take_manual_requests(&mut app).as_slice(),
            [vtuber_avatar::ManualExpressionRequest::Clear { .. }]
        ));
        assert!(
            app.world()
                .resource::<Orchestrator>()
                .last_error()
                .is_none()
        );
    }

    #[test]
    fn reassign_moves_the_expression_and_vacates_the_old_key() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = expression_action_app(path);
        // Default: Digit1=happy, Digit2=angry.
        let (model_id, generation) = expression_target(&app);
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::AssignExpressionKey {
                model_id,
                generation,
                key: ExpressionKey::Digit1,
                expression: Some("angry".into()),
            });
        app.update();

        let store = app.world().resource::<ExpressionBindingStore>();
        let bindings = store.bindings_for("model-a").expect("saved entry");
        assert_eq!(
            bindings.expression_for(ExpressionKey::Digit1),
            Some("angry")
        );
        assert_eq!(bindings.expression_for(ExpressionKey::Digit2), None);
        assert_eq!(bindings.key_for("happy"), None);
    }

    #[test]
    fn reset_restores_the_deterministic_defaults() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = expression_action_app(path);
        let (model_id, generation) = expression_target(&app);
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::AssignExpressionKey {
                model_id,
                generation,
                key: ExpressionKey::Digit1,
                expression: None,
            });
        app.update();
        assert_eq!(
            app.world()
                .resource::<ExpressionBindingStore>()
                .bindings_for("model-a")
                .expect("saved entry")
                .expression_for(ExpressionKey::Digit1),
            None
        );

        let (model_id, generation) = expression_target(&app);
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ResetExpressionBindings {
                model_id,
                generation,
            });
        app.update();
        assert_eq!(
            app.world()
                .resource::<ExpressionBindingStore>()
                .bindings_for("model-a")
                .expect("saved entry")
                .expression_for(ExpressionKey::Digit1),
            Some("happy")
        );
    }

    #[test]
    fn save_failure_keeps_the_previous_assignment_and_reports_an_error() {
        let directory = tempfile::tempdir().unwrap();
        // A directory path makes the atomic settings write fail.
        let path = directory.path().to_path_buf();
        let mut app = expression_action_app(path);
        let defaults = ExpressionBindings::default_for(
            app.world()
                .resource::<vtuber_avatar::AvatarLifecycle>()
                .expression_catalog()
                .expect("catalog"),
        );
        app.world_mut()
            .resource_mut::<ExpressionBindingStore>()
            .set("model-a".into(), defaults);
        let (model_id, generation) = expression_target(&app);
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::AssignExpressionKey {
                model_id,
                generation,
                key: ExpressionKey::Digit1,
                expression: Some("smile".into()),
            });
        app.update();

        let store = app.world().resource::<ExpressionBindingStore>();
        assert_eq!(
            store
                .bindings_for("model-a")
                .expect("previous entry")
                .expression_for(ExpressionKey::Digit1),
            Some("happy"),
            "a failed save must not apply the candidate"
        );
        assert!(matches!(
            app.world().resource::<Orchestrator>().last_error(),
            Some(OrchestratorError::ExpressionSettingsFailed(_))
        ));
        assert!(take_manual_requests(&mut app).is_empty());
    }

    #[test]
    fn toggle_key_emits_a_generation_bound_manual_request() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = expression_action_app(path);
        let generation = app
            .world()
            .resource::<vtuber_avatar::AvatarLifecycle>()
            .current_generation();
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ToggleExpressionKey {
                generation,
                key: ExpressionKey::Digit1,
            });
        app.update();

        assert_eq!(
            take_manual_requests(&mut app),
            vec![vtuber_avatar::ManualExpressionRequest::Toggle {
                generation,
                expression: "happy".into(),
            }]
        );

        // A key with no binding emits nothing.
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ToggleExpressionKey {
                generation,
                key: ExpressionKey::KeyM,
            });
        app.update();
        assert!(take_manual_requests(&mut app).is_empty());
    }

    #[test]
    fn stale_model_actions_never_reach_the_replacement_model() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = expression_action_app(path);
        let (model_a, generation_a) = expression_target(&app);

        // Actions issued from model A's snapshot are already queued.
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ToggleExpressionKey {
                generation: generation_a,
                key: ExpressionKey::Digit1,
            });
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ClearManualExpression {
                generation: generation_a,
            });
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::AssignExpressionKey {
                model_id: model_a.clone(),
                generation: generation_a,
                key: ExpressionKey::Digit1,
                expression: Some("smile".into()),
            });
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ResetExpressionBindings {
                model_id: model_a.clone(),
                generation: generation_a,
            });

        // Replace model A with model B before the orchestrator consumes them.
        let root_b = app.world_mut().spawn_empty().id();
        let generation_b = {
            let mut lifecycle = app
                .world_mut()
                .resource_mut::<vtuber_avatar::AvatarLifecycle>();
            lifecycle.request_replace(root_b).unwrap();
            lifecycle.finish_unload();
            lifecycle.start_binding(root_b);
            let catalog = vtuber_avatar::AvatarExpressionCatalog::build(
                "model-b".into(),
                lifecycle.current_generation().0,
                [vtuber_avatar::ExpressionCatalogInput {
                    id: "happy",
                    declared_as_preset: true,
                    declared_morph_bind_count: 1,
                    resolved_morph_bind_count: 1,
                    declared_material_bind_count: 0,
                    resolved_material_bind_count: 0,
                    unresolved_material_bind_count: 0,
                    unsupported_material_bind_count: 0,
                }],
            );
            lifecycle.set_expression_catalog(Some(catalog));
            lifecycle.finish_ready();
            lifecycle.current_generation()
        };
        assert_ne!(generation_a, generation_b);
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .set_imported_model_for_tests(Some(stub_imported_model_with_id("model-b")));

        app.update();

        let requests = take_manual_requests(&mut app);
        assert!(
            !requests.iter().any(|request| matches!(
                request,
                vtuber_avatar::ManualExpressionRequest::Toggle { .. }
            )),
            "an A toggle must not become a B toggle"
        );
        assert!(
            requests.iter().all(|request| !matches!(
                request,
                vtuber_avatar::ManualExpressionRequest::Clear { generation } if *generation == generation_b
            )),
            "an A clear must not clear B"
        );

        let store = app.world().resource::<ExpressionBindingStore>();
        assert!(
            store.bindings_for("model-b").is_none(),
            "an A assignment must not be saved for B"
        );
        assert!(
            store.bindings_for("model-a").is_none(),
            "a stale assignment must not be saved at all"
        );
    }

    #[test]
    fn expression_snapshot_rejects_pending_model_catalog_mismatch() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = expression_action_app(path);
        app.add_systems(Update, sync_expression_view_model);

        // The import of B succeeded, but the lifecycle and catalog still
        // describe the rendered model A.
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .set_imported_model_for_tests(Some(stub_imported_model_with_id("model-b")));
        app.update();

        let vm = app.world().resource::<UiViewModel>();
        assert!(
            !vm.expression.has_catalog,
            "a mixed B/gA/A snapshot must not be operable"
        );
        assert!(vm.expression.model_id.is_none());
        assert!(vm.expression.generation.is_none());
        assert!(vm.expression.entries.is_empty());
        assert!(vm.expression.selected.is_none());
    }

    #[test]
    fn pending_model_bindings_are_not_used_for_old_generation_actions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = expression_action_app(path.clone());
        let generation_a = app
            .world()
            .resource::<vtuber_avatar::AvatarLifecycle>()
            .current_generation();

        // A's default is Digit1=happy; B has a conflicting saved binding.
        let mut b_bindings = ExpressionBindings::default();
        b_bindings.assign(ExpressionKey::Digit1, "angry");
        app.world_mut()
            .resource_mut::<ExpressionBindingStore>()
            .set("model-b".into(), b_bindings);

        // Pending-import state: only the orchestrator model changed.
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .set_imported_model_for_tests(Some(stub_imported_model_with_id("model-b")));

        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ToggleExpressionKey {
                generation: generation_a,
                key: ExpressionKey::Digit1,
            });
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::AssignExpressionKey {
                model_id: "model-b".into(),
                generation: generation_a,
                key: ExpressionKey::Digit1,
                expression: Some("smile".into()),
            });
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ResetExpressionBindings {
                model_id: "model-b".into(),
                generation: generation_a,
            });
        app.update();

        assert!(
            take_manual_requests(&mut app).is_empty(),
            "an A-generation toggle must not use pending B bindings"
        );
        let store = app.world().resource::<ExpressionBindingStore>();
        assert_eq!(
            store
                .bindings_for("model-b")
                .expect("B entry")
                .expression_for(ExpressionKey::Digit1),
            Some("angry"),
            "B settings must be unchanged"
        );
        assert!(store.bindings_for("model-a").is_none());
        assert!(
            !path.is_file(),
            "a mismatched action must not write the settings file"
        );

        // Complete the swap; normal B operations still work.
        let root_b = app.world_mut().spawn_empty().id();
        let generation_b = {
            let mut lifecycle = app
                .world_mut()
                .resource_mut::<vtuber_avatar::AvatarLifecycle>();
            lifecycle.request_replace(root_b).unwrap();
            lifecycle.finish_unload();
            lifecycle.start_binding(root_b);
            let catalog = vtuber_avatar::AvatarExpressionCatalog::build(
                "model-b".into(),
                lifecycle.current_generation().0,
                [vtuber_avatar::ExpressionCatalogInput {
                    id: "angry",
                    declared_as_preset: true,
                    declared_morph_bind_count: 1,
                    resolved_morph_bind_count: 1,
                    declared_material_bind_count: 0,
                    resolved_material_bind_count: 0,
                    unresolved_material_bind_count: 0,
                    unsupported_material_bind_count: 0,
                }],
            );
            lifecycle.set_expression_catalog(Some(catalog));
            lifecycle.finish_ready();
            lifecycle.current_generation()
        };
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::ToggleExpressionKey {
                generation: generation_b,
                key: ExpressionKey::Digit1,
            });
        app.update();
        assert_eq!(
            take_manual_requests(&mut app),
            vec![vtuber_avatar::ManualExpressionRequest::Toggle {
                generation: generation_b,
                expression: "angry".into(),
            }]
        );
    }

    #[test]
    fn expression_view_model_lists_every_entry_and_all_36_rows() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let mut app = expression_action_app(path);
        app.add_systems(Update, sync_expression_view_model);
        let (model_id, generation) = expression_target(&app);
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::AssignExpressionKey {
                model_id,
                generation,
                key: ExpressionKey::Digit1,
                expression: Some("custom49".into()),
            });
        app.world_mut()
            .resource_mut::<vtuber_avatar::ManualExpressionSelection>()
            .toggle(generation, "smile");
        app.update();
        app.update();

        let vm = app.world().resource::<UiViewModel>();
        assert_eq!(vm.expression.entries.len(), 4);
        assert_eq!(vm.expression.bindings.len(), 36);
        assert_eq!(
            vm.expression.bindings[0].expression.as_deref(),
            Some("custom49")
        );
        assert!(vm.expression.has_catalog);
        assert_eq!(vm.expression.model_id.as_deref(), Some("model-a"));
        assert_eq!(vm.expression.generation, Some(generation));
        assert_eq!(vm.expression.selected.as_deref(), Some("smile"));
        let smile_row = vm
            .expression
            .bindings
            .iter()
            .find(|row| row.expression.as_deref() == Some("smile"))
            .expect("smile keeps its default key");
        assert!(
            smile_row.selected,
            "selection is marked on the assigned key"
        );
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
