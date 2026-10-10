//! One-shot avatar file work. The update loop only submits and polls values.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, mpsc};

use bevy::prelude::*;
use vtuber_avatar::{AvatarAssetId, LoadImportedAvatarRequest};

use crate::import::{self, ImportedModel, ModelImportError};
use crate::license_review::{self, VrmLicenseReview, VrmLicenseReviewError};
use crate::orchestrator::{
    Orchestrator, OrchestratorError, PendingLoadRequest, SubmittedAvatarLoad,
};
use crate::settings::AppSettings;

#[derive(Debug)]
pub(crate) enum AvatarFileWork {
    Review(PathBuf),
    Import(PathBuf),
}

#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "One owned result crosses the file boundary once; it is neither retained in a collection nor copied on the frame path"
)]
pub(crate) enum AvatarFileResult {
    Review {
        path: PathBuf,
        review: VrmLicenseReview,
    },
    Load(LoadImportedAvatarRequest, SubmittedAvatarLoad),
}

#[expect(
    clippy::large_enum_variant,
    reason = "One owned model is moved to a transient worker; boxing it would add an allocation without reducing retained frame state"
)]
enum Work {
    File(AvatarFileWork),
    Load(ImportedModel),
}

struct RunningWork {
    request_id: u64,
    reviewing: bool,
    result: Mutex<mpsc::Receiver<Result<AvatarFileResult, OrchestratorError>>>,
}

/// Owns at most one transient file worker, keeping model conversions sequential.
#[derive(Resource, Default)]
pub struct AvatarIoRuntime {
    running: Option<RunningWork>,
}

/// Polls completed file work and starts the next requested avatar operation.
/// File reads, conversion, hashing and persistence all run on the worker.
pub fn prepare_avatar_io_system(
    mut runtime: ResMut<AvatarIoRuntime>,
    mut orchestrator: ResMut<Orchestrator>,
    persistent: Option<Res<AppSettings>>,
) {
    if let Some(running) = runtime.running.as_mut() {
        let received = running
            .result
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .try_recv();
        let result = match received {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(work_error(
                running.reviewing,
                "avatar file worker exited without a result".into(),
            ))),
        };
        if let Some(result) = result {
            orchestrator.complete_avatar_work(running.request_id, result);
            runtime.running = None;
        }
    }
    if runtime.running.is_some() {
        return;
    }
    let next = orchestrator
        .pending_file_work
        .take()
        .map(|(id, work)| (id, Work::File(work)))
        .or_else(|| {
            orchestrator
                .take_pending_load_request()
                .map(|pending| (pending.request_id, Work::Load(pending.model)))
        });
    let Some((request_id, work)) = next else {
        return;
    };
    let reviewing = matches!(work, Work::File(AvatarFileWork::Review(_)));
    let asset_root = orchestrator.asset_root().to_path_buf();
    let persistent = persistent.as_deref().cloned();
    let (sender, receiver) = mpsc::channel();
    match std::thread::Builder::new()
        .name("avatar-file".into())
        .spawn(move || {
            let result = perform_work(request_id, work, &asset_root, persistent.as_ref());
            let _ = sender.send(result);
        }) {
        Ok(_) => {
            runtime.running = Some(RunningWork {
                request_id,
                reviewing,
                result: Mutex::new(receiver),
            });
        }
        Err(error) => {
            orchestrator
                .complete_avatar_work(request_id, Err(work_error(reviewing, error.to_string())));
        }
    }
}

fn work_error(reviewing: bool, message: String) -> OrchestratorError {
    if reviewing {
        OrchestratorError::LicenseReviewFailed(message)
    } else {
        OrchestratorError::AvatarLoadRejected(message)
    }
}

fn perform_work(
    request_id: u64,
    work: Work,
    asset_root: &Path,
    persistent: Option<&AppSettings>,
) -> Result<AvatarFileResult, OrchestratorError> {
    let model = match work {
        Work::File(AvatarFileWork::Review(path)) => {
            let review = read_reviewable_bytes(&path)
                .and_then(|bytes| license_review::extract_vrm_license_review(&path, &bytes))
                .map_err(|error| OrchestratorError::LicenseReviewFailed(error.to_string()))?;
            return Ok(AvatarFileResult::Review { path, review });
        }
        Work::File(AvatarFileWork::Import(path)) => {
            import::import_vrm(&path, asset_root, import::DEFAULT_SIZE_LIMIT)
                .map_err(|error| OrchestratorError::ImportFailed(format_import_error(&error)))?
        }
        Work::Load(model) => model,
    };
    let (request, submitted) =
        prepare_avatar_load(PendingLoadRequest { request_id, model }, persistent)?;
    Ok(AvatarFileResult::Load(request, submitted))
}

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

pub(crate) fn prepare_avatar_load(
    pending: PendingLoadRequest,
    persistent: Option<&AppSettings>,
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
    let id = AvatarAssetId::new(&pending.model.id);
    let path = vtuber_avatar::UserAssetPath::avatar_model_path(&id)
        .map_err(|error| OrchestratorError::AvatarLoadRejected(error.to_string()))?;
    // Managed copies stored by older versions may predate the VRM 0.x
    // conversion or the VRM 1.0 expression adaptation. The managed copy
    // alone is adapted in place; the user's original file is neither
    // required nor rewritten.
    crate::import::ensure_managed_model_ready(&pending.model.asset_path)
        .map_err(|error| OrchestratorError::AvatarLoadRejected(error.to_string()))?;
    // Preserve parser failures: missing facts and unreadable facts are distinct.
    let (expressions, constraints) =
        crate::import::read_runtime_source_facts(&pending.model.asset_path)
            .map_err(|error| OrchestratorError::AvatarLoadRejected(error.to_string()))?;
    let imported = vtuber_avatar::ImportedAvatar::new(id, path, &pending.model.name)
        .with_warnings(pending.model.summary.compatibility_warnings.clone())
        .with_expressions(expressions)
        .with_node_constraints(constraints);
    Ok((
        vtuber_avatar::LoadImportedAvatarRequest {
            request_id: pending.request_id,
            imported,
        },
        SubmittedAvatarLoad {
            model: pending.model,
            look,
        },
    ))
}

/// Format an import error for user display.
pub(crate) fn format_import_error(error: &ModelImportError) -> String {
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
