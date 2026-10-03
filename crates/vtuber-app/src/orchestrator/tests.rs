#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)] // tests may panic (AGENTS.md)
use super::lifecycle::{current_model_target, map_avatar_lifecycle_state};
use super::*;
use crate::expression_keys::{ExpressionBindingStore, ExpressionBindings, ExpressionKey};
use crate::ndi_output::NdiOutputIntent;
use crate::preview::PreviewState;
use crate::settings::AppSettings;
use crate::ui::UiState;
use bevy::prelude::*;
use vtuber_avatar::{
    ArmPoseOverrideStore, ArmPoseProfileChange, ArmPoseProfileOverride, AvatarAssetId,
    AvatarLifecycle, AvatarMotionMirror,
};

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
        .init_resource::<Assets<StandardMaterial>>()
        .init_resource::<Assets<bevy_vrm1::prelude::MToonMaterial>>()
        .insert_resource(AppSettings::empty_at(path))
        .add_message::<vtuber_avatar::LookSettingsChanged>()
        .add_message::<vtuber_avatar::LoadImportedAvatarRequest>()
        .add_message::<vtuber_avatar::LoadImportedAvatarResult>()
        .add_message::<vtuber_avatar::lifecycle::UnloadAvatarRequest>()
        .add_systems(
            Update,
            (
                process_ui_actions_system,
                sync_avatar_lifecycle_system,
                vtuber_avatar::look::apply_look_settings_changes,
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
    let model = stub_imported_model_with_id(&format!("sha256:{}", suffix.to_string().repeat(64)));
    app.world_mut()
        .resource_mut::<Orchestrator>()
        .queue_imported_model(model);
    app.update(); // submit; the caller's next update receives acceptance/rejection.
}

fn model_action_target(app: &App) -> crate::actions::ModelActionTarget {
    current_model_target(
        app.world().resource::<Orchestrator>(),
        app.world().resource::<AvatarLifecycle>(),
    )
    .unwrap()
}

fn look_change(app: &mut App, change: crate::actions::RichLookChange) {
    let action = UiAction::ChangeRichLook {
        target: model_action_target(app),
        change,
    };
    look_action(app, action);
}

fn save_look(app: &mut App) {
    let action = UiAction::SaveRichLook {
        target: model_action_target(app),
    };
    look_action(app, action);
}

fn look_action(app: &mut App, action: UiAction) {
    app.world_mut().resource_mut::<UiState>().emit(action);
    app.update();
}

#[test]
fn orchestrator_error_has_the_standard_error_contract() {
    let error = OrchestratorError::ImportFailed("original detail".into());
    let standard: &dyn std::error::Error = &error;
    assert_eq!(standard.to_string(), "Import failed: original detail");
    assert!(standard.source().is_none());
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
    look_change(&mut app, RichLookChange::Enabled(true));
    for strength in [1.0, 0.5, 0.0] {
        look_change(&mut app, RichLookChange::Strength(strength));
        assert_eq!(
            app.world().resource::<AvatarLookSettings>().0,
            RichLookSettings::try_new(true, strength).unwrap()
        );
        assert_eq!(
            app.world().resource::<UiViewModel>().look.strength,
            strength
        );
        assert!(!path.exists(), "live edits must not write files");
    }
    save_look(&mut app);
    let zero = RichLookSettings::try_new(true, 0.0).unwrap();
    let saved = std::fs::read_to_string(&path).unwrap();
    look_change(&mut app, RichLookChange::Strength(0.5));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), saved);
    look_change(&mut app, RichLookChange::Enabled(false));
    look_change(&mut app, RichLookChange::Enabled(true));
    assert_eq!(
        app.world().resource::<AvatarLookSettings>().0.strength(),
        0.5
    );
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
    look_change(&mut app, RichLookChange::Strength(0.5));
    save_look(&mut app);
    select_look_model(&mut app, 'a');
    app.update();
    finish_look_model(&mut app);
    assert_eq!(app.world().resource::<AvatarLookSettings>().0, zero);
    select_look_model(&mut app, 'b');
    app.update();
    finish_look_model(&mut app);
    assert_eq!(
        app.world().resource::<AvatarLookSettings>().0,
        RichLookSettings::try_new(false, 0.5).unwrap()
    );
    look_action(&mut app, UiAction::UnloadAvatar);
    assert_eq!(
        app.world().resource::<AvatarLookSettings>().0,
        RichLookSettings::default()
    );
    assert!(app.world().resource::<UiViewModel>().model_target.is_none());
    drop(app);
    let mut restarted = rich_look_app(&path);
    select_look_model(&mut restarted, 'a');
    restarted.update();
    finish_look_model(&mut restarted);
    assert_eq!(restarted.world().resource::<AvatarLookSettings>().0, zero);
}

#[test]
fn invalid_rich_look_edits_keep_live_state_and_saved_bytes() {
    use crate::actions::RichLookChange;
    use vtuber_avatar::AvatarLookSettings;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("settings.toml");
    let mut app = rich_look_app(&path);
    select_look_model(&mut app, 'a');
    app.update();
    finish_look_model(&mut app);
    look_change(&mut app, RichLookChange::Strength(0.5));
    save_look(&mut app);
    let previous = app.world().resource::<AvatarLookSettings>().0;
    let bytes = std::fs::read(&path).unwrap();
    for strength in [-0.1, 1.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        look_change(&mut app, RichLookChange::Strength(strength));
        assert_eq!(app.world().resource::<AvatarLookSettings>().0, previous);
        assert_eq!(
            app.world().resource::<UiViewModel>().look.strength,
            previous.strength()
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(matches!(
            app.world().resource::<Orchestrator>().last_error(),
            Some(OrchestratorError::ArmPoseSettingsFailed(_))
        ));
    }
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
    finish_look_model(&mut app);
    std::fs::create_dir(&path).unwrap();
    look_change(&mut app, RichLookChange::Enabled(true));
    save_look(&mut app);
    assert!(
        app.world()
            .resource::<vtuber_avatar::AvatarLookSettings>()
            .0
            .enabled()
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

fn assert_model_look(app: &App, model: &ImportedModel, expected: vtuber_avatar::RichLookSettings) {
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
    assert_eq!(vm.look.enabled, expected.enabled());
    assert_eq!(vm.look.strength, expected.strength());
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
    let a_look = RichLookSettings::try_new(true, 0.25).unwrap();
    let b_look = RichLookSettings::try_new(false, 0.75).unwrap();
    let settings = AppSettings::empty_at(&path);
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
    look_change(&mut app, RichLookChange::Strength(0.5));
    save_look(&mut app);
    assert_eq!(settings.rich_look_for(&a.id).unwrap().strength(), 0.5);
    assert_eq!(settings.rich_look_for(&b.id).unwrap(), b_look);
}

fn restore_failure_keeps_model(read_error: bool) {
    use vtuber_avatar::{LoadImportedAvatarRequest, RichLookSettings};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.toml");
    let (a, b) = rich_models(&dir);
    let a_look = RichLookSettings::try_new(true, 0.0).unwrap();
    let b_look = RichLookSettings::try_new(false, 0.75).unwrap();
    let settings = AppSettings::empty_at(&path);
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
    let a_look = RichLookSettings::try_new(false, 0.25).unwrap();
    let b_look = RichLookSettings::try_new(true, 0.0).unwrap();
    let settings = AppSettings::empty_at(&path);
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
    look_change(&mut app, RichLookChange::Strength(0.5));
    save_look(&mut app);
    assert_eq!(settings.rich_look_for(&a.id).unwrap(), a_look);
    assert_eq!(
        settings.rich_look_for(&b.id).unwrap(),
        RichLookSettings::try_new(true, 0.5).unwrap()
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
fn arm_pose_settings_edits_commit_and_notify_only_after_a_successful_save() {
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
        .init_resource::<AvatarLifecycle>()
        .insert_resource(AppSettings::empty_at(&path))
        .add_message::<ArmPoseProfileChange>()
        .add_systems(Update, process_ui_actions_system);

    let root = app.world_mut().spawn_empty().id();
    {
        let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
    }

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
    let target = model_action_target(&app);
    app.world_mut()
        .resource_mut::<UiState>()
        .emit(UiAction::SetArmPoseProfile {
            target: target.clone(),
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
    assert_eq!(
        crate::settings::load_arm_pose_overrides(&path)
            .unwrap()
            .profile_for(&id),
        Some(profile)
    );
    let notifications: Vec<_> = app
        .world_mut()
        .resource_mut::<Messages<ArmPoseProfileChange>>()
        .drain()
        .collect();
    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].model_id, id);
    assert!(!notifications[0].return_to_default);

    let saved = std::fs::read(&path).unwrap();
    let foreign = "schema_version = 99\n";
    std::fs::write(&path, foreign).unwrap();
    for action in [
        UiAction::SetArmPoseProfile {
            target: target.clone(),
            profile: ArmPoseProfileOverride::from_profile(vtuber_avatar::ArmPoseProfile {
                arm_drop_radians: 0.7,
                ..profile
            }),
        },
        UiAction::ResetArmPoseProfile {
            target: target.clone(),
        },
    ] {
        app.world_mut().resource_mut::<UiState>().emit(action);
        app.update();
        assert_eq!(
            app.world()
                .resource::<ArmPoseOverrideStore>()
                .profile_for(&id),
            Some(profile)
        );
        assert_eq!(
            app.world().resource::<UiViewModel>().arm_pose.profile,
            profile
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), foreign);
        assert!(
            app.world()
                .resource::<Messages<ArmPoseProfileChange>>()
                .is_empty()
        );
        assert!(matches!(
            app.world().resource::<Orchestrator>().last_error(),
            Some(OrchestratorError::ArmPoseSettingsFailed(_))
        ));
    }
    std::fs::write(&path, saved).unwrap();

    app.world_mut()
        .resource_mut::<UiState>()
        .emit(UiAction::ResetArmPoseProfile {
            target: target.clone(),
        });
    app.update();

    assert!(
        app.world()
            .resource::<vtuber_avatar::ArmPoseOverrideStore>()
            .profile_for(&id)
            .is_none()
    );
    let restored = crate::settings::load_arm_pose_overrides(&path).expect("reset file");
    assert!(restored.profile_for(&id).is_none());
    let notifications: Vec<_> = app
        .world_mut()
        .resource_mut::<Messages<ArmPoseProfileChange>>()
        .drain()
        .collect();
    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].model_id, id);
    assert!(notifications[0].return_to_default);
}

/// Minimal app that runs the UI action processor with a real settings
/// resource, so a save is observed through the same path the shell uses.
fn arm_tracking_action_app(settings: AppSettings) -> App {
    let mut app = App::new();
    app.init_resource::<Orchestrator>()
        .init_resource::<UiState>()
        .init_resource::<UiViewModel>()
        .init_resource::<PreviewState>()
        .init_resource::<AvatarMotionMirror>()
        .init_resource::<crate::pose_runtime::PoseRuntime>()
        .insert_resource(settings)
        .add_systems(Update, process_ui_actions_system);
    app
}

/// The persisted switch, the runtime switch, and the rendered switch.
fn arm_tracking_switches(app: &App) -> (bool, bool, bool) {
    (
        app.world().resource::<AppSettings>().arm_tracking_enabled(),
        app.world()
            .resource::<crate::pose_runtime::PoseRuntime>()
            .enabled(),
        app.world().resource::<UiViewModel>().arm_tracking_enabled,
    )
}

fn toggle_arm_tracking(app: &mut App, enabled: bool) {
    app.world_mut()
        .resource_mut::<UiState>()
        .emit(UiAction::SetArmTrackingEnabled { enabled });
    app.update();
}

fn assert_settings_save_refused(app: &App) {
    assert!(matches!(
        app.world().resource::<Orchestrator>().last_error(),
        Some(OrchestratorError::ArmPoseSettingsFailed(_))
    ));
}

#[test]
fn a_successful_arm_tracking_save_switches_the_runtime_and_the_file() {
    let directory = tempfile::tempdir().expect("temporary settings directory");
    let path = directory.path().join("settings.toml");
    let mut app = arm_tracking_action_app(AppSettings::load(&path).expect("initial values"));
    assert_eq!(arm_tracking_switches(&app), (false, false, false));

    toggle_arm_tracking(&mut app, true);

    assert_eq!(arm_tracking_switches(&app), (true, true, true));
    assert!(crate::settings::load_arm_tracking_enabled(&path).expect("saved switch"));
    toggle_arm_tracking(&mut app, false);

    assert_eq!(arm_tracking_switches(&app), (false, false, false));
    assert!(!crate::settings::load_arm_tracking_enabled(&path).expect("saved switch"));
}

#[test]
fn a_refused_arm_tracking_save_keeps_the_runtime_switch() {
    let directory = tempfile::tempdir().expect("temporary settings directory");
    let path = directory.path().join("settings.toml");
    let current = "schema_version = 1\n";
    std::fs::write(&path, current).expect("settings seed");
    let mut app = arm_tracking_action_app(AppSettings::load(&path).expect("current schema"));
    assert_eq!(arm_tracking_switches(&app), (false, false, false));

    // A newer document on the same path refuses every later save, in both
    // directions, and leaves the runtime switch where the operator had it.
    let foreign = "schema_version = 99\n";
    std::fs::write(&path, foreign).expect("foreign schema");
    toggle_arm_tracking(&mut app, true);

    assert_eq!(arm_tracking_switches(&app), (false, false, false));
    assert_settings_save_refused(&app);
    assert_eq!(
        std::fs::read_to_string(&path).expect("settings bytes"),
        foreign
    );

    std::fs::write(&path, current).expect("current schema");
    toggle_arm_tracking(&mut app, true);
    assert_eq!(arm_tracking_switches(&app), (true, true, true));

    std::fs::write(&path, foreign).expect("foreign schema");
    toggle_arm_tracking(&mut app, false);

    assert_eq!(arm_tracking_switches(&app), (true, true, true));
    assert_settings_save_refused(&app);
    assert_eq!(
        std::fs::read_to_string(&path).expect("settings bytes"),
        foreign
    );
}

#[test]
fn an_unreadable_settings_path_keeps_the_runtime_switch() {
    let directory = tempfile::tempdir().expect("temporary settings directory");
    let mut app = arm_tracking_action_app(AppSettings::empty_at(directory.path()));

    toggle_arm_tracking(&mut app, true);

    assert_eq!(arm_tracking_switches(&app), (false, false, false));
    assert_settings_save_refused(&app);
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
fn auto_start_waits_quietly_for_avatar() {
    let mut orch = Orchestrator {
        selected_camera: Some(0),
        lifecycle_state: AvatarLifecycleState::Ready,
        ..Default::default()
    };
    orch.maybe_auto_start_tracking();
    assert_eq!(orch.pipeline_state(), PipelineState::Idle);
    assert!(!orch.capture_desired());
    assert!(orch.last_error().is_none());
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
fn camera_change_waits_for_capture_stop_then_starts_latest_selection() {
    let mut orch = Orchestrator {
        imported_model: Some(stub_imported_model()),
        selected_camera: Some(0),
        lifecycle_state: AvatarLifecycleState::Ready,
        cameras: (0..3)
            .map(|index| CameraDescriptor {
                id: format!("test:{index}"),
                label: format!("Test camera {index}"),
            })
            .collect(),
        pipeline_state: PipelineState::Running,
        capture_desired: true,
        ..Default::default()
    };

    orch.process_action(&UiAction::SelectCamera { index: 1 });
    orch.maybe_auto_start_tracking();
    assert_eq!(orch.pipeline_state(), PipelineState::Stopping);
    assert!(!orch.capture_desired());
    assert!(!orch.capture_ack());

    // Another dropdown choice during shutdown replaces the pending camera.
    orch.process_action(&UiAction::SelectCamera { index: 2 });
    orch.maybe_auto_start_tracking();
    assert_eq!(orch.pipeline_state(), PipelineState::Stopping);
    assert_eq!(orch.selected_camera, Some(2));

    orch.complete_capture_stop();
    orch.maybe_auto_start_tracking();
    assert_eq!(orch.pipeline_state(), PipelineState::Starting);
    assert!(orch.capture_desired());
    assert!(!orch.capture_ack());
}

#[test]
fn selecting_the_running_camera_keeps_the_session() {
    let mut orch = Orchestrator {
        imported_model: Some(stub_imported_model()),
        selected_camera: Some(0),
        lifecycle_state: AvatarLifecycleState::Ready,
        cameras: vec![CameraDescriptor {
            id: "test:0".into(),
            label: "Test camera".into(),
        }],
        pipeline_state: PipelineState::Running,
        capture_desired: true,
        ..Default::default()
    };

    orch.process_action(&UiAction::SelectCamera { index: 0 });
    orch.maybe_auto_start_tracking();

    assert_eq!(orch.pipeline_state(), PipelineState::Running);
    assert!(orch.capture_desired());
    assert!(orch.capture_ack());
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
        .insert_resource(AppSettings::empty_at(settings_path))
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
            target: crate::actions::ModelActionTarget {
                model_id,
                generation,
            },
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
            target: crate::actions::ModelActionTarget {
                model_id,
                generation,
            },
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
            target: crate::actions::ModelActionTarget {
                model_id,
                generation,
            },
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
            target: crate::actions::ModelActionTarget {
                model_id,
                generation,
            },
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
            target: crate::actions::ModelActionTarget {
                model_id,
                generation,
            },
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
    let mut app = expression_action_app(path.clone());
    app.init_resource::<ArmPoseOverrideStore>()
        .init_resource::<vtuber_avatar::AvatarLookSettings>()
        .add_message::<vtuber_avatar::LookSettingsChanged>();
    let b_profile = ArmPoseProfileOverride::from_profile(vtuber_avatar::ArmPoseProfile {
        arm_drop_radians: 0.4,
        ..Default::default()
    });
    app.world_mut()
        .resource_mut::<ArmPoseOverrideStore>()
        .set("model-b", b_profile)
        .unwrap();
    let arms_before = app.world().resource::<ArmPoseOverrideStore>().clone();
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
            target: crate::actions::ModelActionTarget {
                model_id: model_a.clone(),
                generation: generation_a,
            },
            key: ExpressionKey::Digit1,
            expression: Some("smile".into()),
        });
    app.world_mut()
        .resource_mut::<UiState>()
        .emit(UiAction::ResetExpressionBindings {
            target: crate::actions::ModelActionTarget {
                model_id: model_a.clone(),
                generation: generation_a,
            },
        });
    let target_a = crate::actions::ModelActionTarget {
        model_id: model_a.clone(),
        generation: generation_a,
    };
    for action in [
        UiAction::SetArmPoseProfile {
            target: target_a.clone(),
            profile: ArmPoseProfileOverride::from_profile(vtuber_avatar::ArmPoseProfile::default()),
        },
        UiAction::ResetArmPoseProfile {
            target: target_a.clone(),
        },
        UiAction::ChangeRichLook {
            target: target_a.clone(),
            change: crate::actions::RichLookChange::Enabled(true),
        },
        UiAction::SaveRichLook { target: target_a },
    ] {
        app.world_mut().resource_mut::<UiState>().emit(action);
    }

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

    // A matching model ID also cannot authorize an old instance generation.
    app.world_mut()
        .resource_mut::<UiState>()
        .emit(UiAction::ResetArmPoseProfile {
            target: crate::actions::ModelActionTarget {
                model_id: "model-b".into(),
                generation: generation_a,
            },
        });

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
    assert_eq!(app.world().resource::<ArmPoseOverrideStore>(), &arms_before);
    assert!(
        app.world()
            .resource::<Messages<ArmPoseProfileChange>>()
            .is_empty()
    );
    assert!(
        !app.world()
            .resource::<vtuber_avatar::AvatarLookSettings>()
            .0
            .enabled()
    );
    assert!(
        app.world()
            .resource::<Messages<vtuber_avatar::LookSettingsChanged>>()
            .is_empty()
    );
    assert!(
        !path.exists(),
        "stale operations must not save either model"
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
    assert!(vm.expression.target.is_none());
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
            target: crate::actions::ModelActionTarget {
                model_id: "model-b".into(),
                generation: generation_a,
            },
            key: ExpressionKey::Digit1,
            expression: Some("smile".into()),
        });
    app.world_mut()
        .resource_mut::<UiState>()
        .emit(UiAction::ResetExpressionBindings {
            target: crate::actions::ModelActionTarget {
                model_id: "model-b".into(),
                generation: generation_a,
            },
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
            target: crate::actions::ModelActionTarget {
                model_id,
                generation,
            },
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
    assert_eq!(
        vm.expression
            .target
            .as_ref()
            .map(|target| target.model_id.as_str()),
        Some("model-a")
    );
    assert_eq!(
        vm.expression
            .target
            .as_ref()
            .map(|target| target.generation),
        Some(generation)
    );
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
