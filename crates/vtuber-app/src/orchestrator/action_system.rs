//! UI command dispatch and its Bevy resource/message boundary.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use vtuber_avatar::{
    ArmPoseOverrideStore, ArmPoseProfileChange, ArmPoseProfileOverride, AvatarAssetId,
    AvatarMotionMirror, ManualExpressionRequest,
};

use super::expressions::{
    ExpressionBindingAction, apply_expression_binding_action, clear_manual_expression_action,
    toggle_expression_key_action,
};
use super::lifecycle::current_model_target;
use super::{Orchestrator, OrchestratorError};
use crate::actions::{ActionQueue, UiAction};
use crate::expression_keys::ExpressionBindingStore;
use crate::ndi_output::NdiOutputIntent;
use crate::preview::PreviewState;
use crate::settings::AppSettings;
use crate::ui_model::{ArmPoseViewModel, UiViewModel};

/// The look-related system parameters grouped to stay within the system
/// parameter limit: the live settings and their change messages.
#[derive(SystemParam)]
pub struct LookSystemParams<'w> {
    look_settings: Option<ResMut<'w, vtuber_avatar::AvatarLookSettings>>,
    look_changes: Option<MessageWriter<'w, vtuber_avatar::LookSettingsChanged>>,
}

/// System that processes pending UI actions through the orchestrator.
#[expect(
    clippy::too_many_arguments,
    reason = "Bevy injects this system's resources and message streams, so the parameter list is the declared ECS contract and has no call site to restructure"
)]
pub fn process_ui_actions_system(
    mut orchestrator: ResMut<Orchestrator>,
    mut action_queue: ResMut<ActionQueue>,
    mut view_model: ResMut<UiViewModel>,
    mut ndi_intent: Option<ResMut<NdiOutputIntent>>,
    mut preview: ResMut<PreviewState>,
    mut avatar_motion_mirror: ResMut<AvatarMotionMirror>,
    mut arm_pose_overrides: Option<ResMut<ArmPoseOverrideStore>>,
    mut arm_pose_settings: Option<ResMut<AppSettings>>,
    mut arm_pose_changes: Option<MessageWriter<ArmPoseProfileChange>>,
    lifecycle: Option<Res<vtuber_avatar::AvatarLifecycle>>,
    mut reset_camera_requests: Option<MessageWriter<vtuber_avatar::ResetCameraRequest>>,
    mut expression_store: Option<ResMut<ExpressionBindingStore>>,
    mut manual_requests: Option<MessageWriter<ManualExpressionRequest>>,
    mut pose_runtime: Option<ResMut<crate::pose_runtime::PoseRuntime>>,
    mut look: LookSystemParams,
) {
    let actions = action_queue.take_actions();
    for action in &actions {
        if let Some(target) = action.model_target()
            && lifecycle
                .as_deref()
                .and_then(|lifecycle| current_model_target(&orchestrator, lifecycle))
                .as_ref()
                != Some(target)
        {
            continue;
        }
        match action {
            UiAction::TogglePreview => preview.toggle_visible(),
            UiAction::ToggleMirror => preview.toggle_mirrored(),
            UiAction::ToggleAvatarMotionMirror => avatar_motion_mirror.toggle(),
            UiAction::SetArmTrackingEnabled { enabled } => {
                // Persist first: a refused save must leave the runtime switch
                // and the observation state exactly as they were.
                if let Some(settings) = arm_pose_settings.as_deref_mut()
                    && let Err(error) = settings.set_arm_tracking_enabled(*enabled)
                {
                    orchestrator.set_last_error(Some(OrchestratorError::ArmPoseSettingsFailed(
                        error.to_string(),
                    )));
                    continue;
                }
                if let Some(pose) = pose_runtime.as_deref_mut() {
                    pose.set_enabled(*enabled);
                    if !*enabled {
                        // Dropping the observation state also returns the arms
                        // to the virtual anchors on the next compositor frame.
                        pose.recalibrate();
                    }
                }
            }
            UiAction::RecalibrateArms => {
                if let Some(pose) = pose_runtime.as_deref_mut() {
                    pose.recalibrate();
                }
            }
            UiAction::SetArmPoseProfile { target, profile } => {
                edit_arm_pose_profile_action(
                    &mut orchestrator,
                    target,
                    Some(*profile),
                    &mut arm_pose_overrides,
                    arm_pose_settings.as_deref(),
                    &mut arm_pose_changes,
                );
            }
            UiAction::ResetArmPoseProfile { target } => {
                edit_arm_pose_profile_action(
                    &mut orchestrator,
                    target,
                    None,
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
                if let (Some(look), Some(changes)) = (
                    look.look_settings.as_deref_mut(),
                    look.look_changes.as_mut(),
                ) {
                    look.0 = vtuber_avatar::RichLookSettings::default();
                    changes.write(vtuber_avatar::LookSettingsChanged(look.0));
                }
            }
            UiAction::ChangeRichLook { change, .. } => {
                if let Err(error) = apply_rich_look_action(
                    look.look_settings.as_deref_mut(),
                    look.look_changes.as_mut(),
                    *change,
                ) {
                    orchestrator.set_last_error(Some(OrchestratorError::ArmPoseSettingsFailed(
                        error.to_string(),
                    )));
                }
            }
            UiAction::SaveRichLook { target } => {
                if let (Some(persistent), Some(look)) =
                    (arm_pose_settings.as_deref(), look.look_settings.as_deref())
                {
                    let result = persistent
                        .save_rich_look(target.model_id.clone(), look.0)
                        .map_err(|error| {
                            OrchestratorError::ArmPoseSettingsFailed(error.to_string())
                        });
                    if let Err(error) = result {
                        orchestrator.set_last_error(Some(error));
                    }
                }
            }
            UiAction::AssignExpressionKey {
                target,
                key,
                expression,
            } => {
                apply_expression_binding_action(
                    &mut orchestrator,
                    ExpressionBindingAction::Assign {
                        target,
                        key: *key,
                        expression: expression.as_deref(),
                    },
                    &mut expression_store,
                    arm_pose_settings.as_deref(),
                    lifecycle.as_deref(),
                    &mut manual_requests,
                );
            }
            UiAction::ResetExpressionBindings { target } => {
                apply_expression_binding_action(
                    &mut orchestrator,
                    ExpressionBindingAction::Reset { target },
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
            UiAction::SwitchPane(pane) => view_model.pane = *pane,
            UiAction::RefreshCameras
            | UiAction::SelectCamera { .. }
            | UiAction::ImportAvatar { .. }
            | UiAction::RequestAvatarImportReview { .. }
            | UiAction::SetAvatarImportReviewAccepted { .. }
            | UiAction::AcceptAvatarImportReview
            | UiAction::CancelAvatarImportReview
            | UiAction::BeginCalibration
            | UiAction::CancelCalibration
            | UiAction::RetryCalibration
            | UiAction::DismissError
            | UiAction::RetryAfterError => orchestrator.process_action(action),
        }
    }
    view_model.update_from_orchestrator(&orchestrator);
    view_model.model_target = lifecycle
        .as_deref()
        .and_then(|lifecycle| current_model_target(&orchestrator, lifecycle));
    sync_arm_pose_view_model(&mut view_model, arm_pose_overrides.as_deref());
    view_model.preview_visible = preview.visible;
    view_model.mirror_preview = preview.mirrored;
    view_model.mirror_avatar_motion = avatar_motion_mirror.is_enabled();
    view_model.arm_tracking_enabled = pose_runtime.as_deref().is_some_and(|pose| pose.enabled());
    if let Some(settings) = look.look_settings.as_deref() {
        view_model.look.enabled = settings.0.enabled();
        view_model.look.strength = settings.0.strength();
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
) -> Result<(), vtuber_avatar::RichLookSettingsError> {
    let (Some(settings), Some(changes)) = (settings, changes) else {
        return Ok(());
    };
    let next = crate::actions::reduce_rich_look(settings.0, change)?;
    if next == settings.0 {
        return Ok(());
    }
    settings.0 = next;
    changes.write(vtuber_avatar::LookSettingsChanged(next));
    Ok(())
}

fn edit_arm_pose_profile_action(
    orchestrator: &mut Orchestrator,
    target: &crate::actions::ModelActionTarget,
    profile: Option<ArmPoseProfileOverride>,
    overrides: &mut Option<ResMut<ArmPoseOverrideStore>>,
    settings: Option<&AppSettings>,
    changes: &mut Option<MessageWriter<ArmPoseProfileChange>>,
) {
    let model_id = target.model_id.clone();
    let Some(store) = overrides.as_deref_mut() else {
        return;
    };
    let mut candidate = store.clone();
    if let Some(profile) = profile {
        if let Err(error) = candidate.set(model_id.clone(), profile) {
            orchestrator.set_last_error(Some(OrchestratorError::ArmPoseSettingsFailed(
                error.to_string(),
            )));
            return;
        }
    } else {
        candidate.reset(&AvatarAssetId::new(&model_id));
    }
    if let Some(settings) = settings
        && let Err(error) = settings.save(&candidate)
    {
        orchestrator.set_last_error(Some(OrchestratorError::ArmPoseSettingsFailed(
            error.to_string(),
        )));
        return;
    }
    *store = candidate;
    if let Some(changes) = changes.as_mut() {
        changes.write(ArmPoseProfileChange {
            model_id: AvatarAssetId::new(model_id),
            return_to_default: profile.is_none(),
        });
    }
}

fn sync_arm_pose_view_model(
    view_model: &mut UiViewModel,
    overrides: Option<&ArmPoseOverrideStore>,
) {
    let Some(target) = view_model.model_target.as_ref() else {
        view_model.arm_pose = ArmPoseViewModel::default();
        return;
    };
    let id = AvatarAssetId::new(&target.model_id);
    let Some(overrides) = overrides else {
        view_model.arm_pose = ArmPoseViewModel::default();
        return;
    };
    let profile = overrides.profile_for(&id);
    view_model.arm_pose.profile = profile.unwrap_or_default();
    view_model.arm_pose.has_override = profile.is_some();
}
