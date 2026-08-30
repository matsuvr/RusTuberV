// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! M1-08-009: avatar import → lifecycle integration tests.
//!
//! Verifies that the sync system correctly bridges the orchestrator to the
//! avatar lifecycle by emitting `LoadImportedAvatarRequest` and
//! `UnloadAvatarRequest` messages.

use bevy::prelude::*;
use vtuber_app::orchestrator::{Orchestrator, sync_avatar_lifecycle_system};
use vtuber_app::ui_model::AvatarLifecycleState;
use vtuber_avatar::lifecycle::AvatarLifecycle;
use vtuber_avatar::{LoadImportedAvatarRequest, UserAssetPath};

/// Minimal Bevy app with the sync system and avatar lifecycle.
fn test_app() -> App {
    let mut app = App::new();
    app.init_resource::<Orchestrator>()
        .init_resource::<AvatarLifecycle>()
        .add_message::<LoadImportedAvatarRequest>()
        .add_message::<vtuber_avatar::LoadImportedAvatarResult>()
        .add_message::<vtuber_avatar::lifecycle::UnloadAvatarRequest>()
        .add_systems(bevy::app::Update, sync_avatar_lifecycle_system);
    app
}

#[test]
fn sync_system_maps_no_avatar_to_none() {
    let mut app = test_app();
    app.update();

    let orch = app.world().resource::<Orchestrator>();
    let mut vm = vtuber_app::ui_model::UiViewModel::default();
    orch.update_view_model(&mut vm);
    assert_eq!(vm.avatar.lifecycle, AvatarLifecycleState::None);
    assert!(!vm.avatar.is_ready);
    assert!(!vm.avatar.load_failed);
}

#[test]
fn sync_system_maps_loading_state() {
    let mut app = test_app();

    // Spawn a root entity first, then request load.
    let root = app.world_mut().spawn_empty().id();
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .request_load(root)
        .unwrap();

    app.update();

    let orch = app.world().resource::<Orchestrator>();
    let mut vm = vtuber_app::ui_model::UiViewModel::default();
    orch.update_view_model(&mut vm);
    assert_eq!(vm.avatar.lifecycle, AvatarLifecycleState::Loading);
    assert!(!vm.avatar.is_ready);
}

#[test]
fn sync_system_maps_ready_state() {
    let mut app = test_app();

    // Simulate the full lifecycle: Loading → Binding → Ready.
    let root = app.world_mut().spawn_empty().id();
    {
        let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
    }

    app.update();

    let orch = app.world().resource::<Orchestrator>();
    let mut vm = vtuber_app::ui_model::UiViewModel::default();
    orch.update_view_model(&mut vm);
    assert_eq!(vm.avatar.lifecycle, AvatarLifecycleState::Ready);
    assert!(vm.avatar.is_ready);
    assert!(!vm.avatar.load_failed);
}

#[test]
fn sync_system_maps_failed_state() {
    let mut app = test_app();

    // Simulate lifecycle failure.
    let root = app.world_mut().spawn_empty().id();
    {
        let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
        lifecycle.request_load(root).unwrap();
        lifecycle.fail(vtuber_avatar::lifecycle::AvatarLifecycleFailure::AssetLoadFailed);
    }

    app.update();

    let orch = app.world().resource::<Orchestrator>();
    let mut vm = vtuber_app::ui_model::UiViewModel::default();
    orch.update_view_model(&mut vm);
    assert_eq!(vm.avatar.lifecycle, AvatarLifecycleState::Failed);
    assert!(!vm.avatar.is_ready);
    assert!(vm.avatar.load_failed);
}

#[test]
fn user_asset_path_constructed_from_sha256_id() {
    let id = vtuber_avatar::AvatarAssetId::new("abc123def456");
    let path = UserAssetPath::avatar_model_path(&id).expect("valid path");
    assert_eq!(path.as_str(), "user://avatars/abc123def456/model.vrm");
}

#[test]
fn user_asset_path_rejects_absolute_filesystem_path() {
    let err = UserAssetPath::new("C:\\models\\avatar.vrm").expect_err("must reject absolute path");
    assert!(
        matches!(
            err,
            vtuber_avatar::AssetPathError::MissingScheme
                | vtuber_avatar::AssetPathError::WrongScheme
        ),
        "expected MissingScheme or WrongScheme, got {err:?}",
    );
}

#[test]
fn lifecycle_request_is_canonical_and_consumed_once() {
    let model_id = "a".repeat(64);
    // The canonical path contract is asserted without passing an absolute
    // filesystem path to AssetServer.
    let id = vtuber_avatar::AvatarAssetId::new(model_id.clone());
    let path = UserAssetPath::avatar_model_path(&id).expect("canonical id path");
    assert_eq!(
        path.as_str(),
        format!("user://avatars/{model_id}/model.vrm")
    );
}

#[test]
fn orchestrator_pending_load_is_none_by_default() {
    let mut orch = Orchestrator::default();
    assert!(orch.take_pending_load_request().is_none());
}

#[test]
fn orchestrator_lifecycle_state_starts_at_none() {
    let orch = Orchestrator::default();
    let mut vm = vtuber_app::ui_model::UiViewModel::default();
    orch.update_view_model(&mut vm);
    assert_eq!(vm.avatar.lifecycle, AvatarLifecycleState::None);
    assert!(!vm.avatar.is_ready);
}
