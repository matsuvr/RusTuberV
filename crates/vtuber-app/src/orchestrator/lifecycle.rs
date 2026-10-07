//! Avatar load preparation, submission, and lifecycle result handling.

use bevy::prelude::*;
use vtuber_avatar::{AvatarAssetId, AvatarLifecycle};

use super::{Orchestrator, OrchestratorError, PendingLoadRequest, SubmittedAvatarLoad};
use crate::settings::AppSettings;
use crate::ui_model::{AvatarLifecycleState, UiViewModel};

/// Pairs the accepted model owner with the ready lifecycle instance.
pub(super) fn current_model_target(
    orchestrator: &Orchestrator,
    lifecycle: &AvatarLifecycle,
) -> Option<crate::actions::ModelActionTarget> {
    if lifecycle.state() != vtuber_avatar::AvatarLifecycleState::Ready {
        return None;
    }
    Some(crate::actions::ModelActionTarget {
        model_id: orchestrator.active_model_id()?.to_owned(),
        generation: lifecycle.current_generation(),
    })
}

fn prepare_avatar_load(
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
    // The managed copy's `VRMC_vrm.expressions` (custom-origin record and
    // material bind entries included) is the single source of truth the
    // bind step resolves against the live scene. An unreadable managed copy
    // also fails the runtime asset load, so the facts are best-effort here.
    let (expressions, constraints) =
        crate::import::read_runtime_source_facts(&pending.model.asset_path).unwrap_or_default();
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

/// Converts the avatar lifecycle's internal state to the UI model's state.
pub(super) fn map_avatar_lifecycle_state(
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
///    [`LoadImportedAvatarRequest`](vtuber_avatar::LoadImportedAvatarRequest) message.
/// 3. Detects when the user has cleared the imported model while the lifecycle
///    still has an active avatar, and emits an
///    [`UnloadAvatarRequest`](vtuber_avatar::lifecycle::UnloadAvatarRequest).
///
/// Runs after UI actions and the engine's load/request/unload systems, so
/// accepted results commit the model, look, save owner and UI snapshot together.
#[expect(
    clippy::too_many_arguments,
    reason = "Bevy injects this system's resources and message streams, so the parameter list is the declared ECS contract and has no call site to restructure"
)]
pub fn sync_avatar_lifecycle_system(
    mut orchestrator: ResMut<Orchestrator>,
    lifecycle: Res<vtuber_avatar::lifecycle::AvatarLifecycle>,
    mut load_requests: MessageWriter<vtuber_avatar::LoadImportedAvatarRequest>,
    mut load_results: MessageReader<vtuber_avatar::LoadImportedAvatarResult>,
    mut unload_requests: MessageWriter<vtuber_avatar::lifecycle::UnloadAvatarRequest>,
    persistent: Option<Res<AppSettings>>,
    mut look: Option<ResMut<vtuber_avatar::AvatarLookSettings>>,
    mut changes: Option<MessageWriter<vtuber_avatar::LookSettingsChanged>>,
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
                    if let (Some(restored), Some(look), Some(changes)) =
                        (submitted.look, look.as_deref_mut(), changes.as_mut())
                    {
                        look.0 = restored;
                        changes.write(vtuber_avatar::LookSettingsChanged(restored));
                    }
                    orchestrator.imported_model = Some(submitted.model);
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

    // 2. Prepare and submit the pending load. Its model becomes active only
    // after the engine returns an Accepted result.
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
        view_model.model_target = current_model_target(&orchestrator, &lifecycle);
        if let Some(look) = look.as_deref() {
            view_model.look.enabled = look.0.enabled();
            view_model.look.strength = look.0.strength();
        }
    }
}
