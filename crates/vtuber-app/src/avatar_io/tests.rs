#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use super::*;
use crate::actions::UiAction;
use crate::orchestrator::{ImportState, sync_avatar_lifecycle_system};
use serde_json::json;

fn model_bytes(constraint: Option<serde_json::Value>) -> Vec<u8> {
    let mut document = json!({
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Hips", "children": [1]}, {"name": "Head"}],
        "extensionsUsed": ["VRMC_vrm"],
        "extensions": {"VRMC_vrm": {
            "specVersion": "1.0", "meta": {"name": "Test", "authors": ["Test"]},
            "humanoid": {"humanBones": {"hips": {"node": 0}, "head": {"node": 1}}},
            "expressions": {"preset": {"happy": {"isBinary": false}}}
        }}
    });
    if let Some(constraint) = constraint {
        document["nodes"][1]["extensions"] =
            json!({"VRMC_node_constraint": {"constraint": constraint}});
    }
    vtuber_avatar::glb::Glb::new(document, None)
        .to_vec()
        .unwrap()
}

fn model(path: PathBuf, id: &str) -> ImportedModel {
    ImportedModel {
        id: id.into(),
        name: id.into(),
        asset_path: path,
        meta_path: PathBuf::new(),
        summary: Default::default(),
        original_path: PathBuf::new(),
        size: 0,
    }
}

fn bridge_app() -> App {
    let mut app = App::new();
    app.init_resource::<Orchestrator>()
        .init_resource::<AvatarIoRuntime>()
        .init_resource::<vtuber_avatar::AvatarLifecycle>()
        .add_message::<vtuber_avatar::LoadImportedAvatarRequest>()
        .add_message::<vtuber_avatar::LoadImportedAvatarResult>()
        .add_message::<vtuber_avatar::UnloadAvatarRequest>()
        .add_systems(
            Update,
            (prepare_avatar_io_system, sync_avatar_lifecycle_system).chain(),
        );
    app
}

fn finish_work(app: &mut App) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while app.world().resource::<Orchestrator>().import_state() == &ImportState::InProgress {
        assert!(std::time::Instant::now() < deadline, "file work timed out");
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn invalid_constraints_emit_no_load_and_preserve_the_accepted_model() {
    for constraint in [
        json!({"rotation": {"source": 9}}),
        json!({"roll": {"source": 0, "rollAxis": "invalid"}}),
        json!({"rotation": {"source": 0, "weight": 1.1}}),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.vrm");
        std::fs::write(&path, model_bytes(Some(constraint))).unwrap();
        let mut app = bridge_app();
        let mut orchestrator = app.world_mut().resource_mut::<Orchestrator>();
        orchestrator.set_imported_model_for_tests(Some(model(PathBuf::new(), "accepted")));
        orchestrator.queue_imported_model(model(path, "candidate"));
        finish_work(&mut app);
        let orchestrator = app.world().resource::<Orchestrator>();
        assert_eq!(orchestrator.active_model_id(), Some("accepted"));
        assert!(
            matches!(orchestrator.last_error(), Some(OrchestratorError::AvatarLoadRejected(message)) if message.contains("VRMC_node_constraint"))
        );
        assert!(
            app.world()
                .resource::<Messages<LoadImportedAvatarRequest>>()
                .is_empty()
        );
    }
}

#[test]
fn a_model_without_constraints_is_prepared_and_submitted_once() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.vrm");
    std::fs::write(&path, model_bytes(None)).unwrap();
    let mut app = bridge_app();
    app.world_mut()
        .resource_mut::<Orchestrator>()
        .queue_imported_model(model(path, "candidate"));
    finish_work(&mut app);
    let requests: Vec<_> = app
        .world_mut()
        .resource_mut::<Messages<LoadImportedAvatarRequest>>()
        .drain()
        .collect();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].request_id, 1);
    assert!(requests[0].imported.node_constraints.0.is_empty());
    assert_eq!(requests[0].imported.expressions.entries.len(), 1);
    app.update();
    assert!(
        app.world()
            .resource::<Messages<LoadImportedAvatarRequest>>()
            .is_empty()
    );
}

#[test]
fn missing_managed_data_propagates_as_a_load_error() {
    let directory = tempfile::tempdir().unwrap();
    let result = prepare_avatar_load(
        PendingLoadRequest {
            request_id: 1,
            model: model(directory.path().join("missing.vrm"), "candidate"),
        },
        None,
    );
    assert!(matches!(
        result,
        Err(OrchestratorError::AvatarLoadRejected(_))
    ));
}

#[test]
fn review_file_failures_are_preserved_before_import() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("broken.vrm");
    std::fs::write(&path, b"not a GLB").unwrap();
    let result = perform_work(
        1,
        Work::File(AvatarFileWork::Review(path)),
        directory.path(),
        None,
    );
    assert!(matches!(
        result,
        Err(OrchestratorError::LicenseReviewFailed(_))
    ));
    let path = directory.path().join("oversized.vrm");
    std::fs::File::create(&path)
        .unwrap()
        .set_len(import::DEFAULT_SIZE_LIMIT + 1)
        .unwrap();
    assert!(matches!(
        read_reviewable_bytes(&path),
        Err(VrmLicenseReviewError::SizeExceeded { .. })
    ));
    assert!(!directory.path().join("avatars").exists());
}

#[test]
fn an_unfinished_job_leaves_update_free_and_the_next_job_queued() {
    let mut app = bridge_app();
    let (sender, receiver) = mpsc::channel();
    app.world_mut().resource_mut::<AvatarIoRuntime>().running = Some(RunningWork {
        request_id: 1,
        reviewing: false,
        result: Mutex::new(receiver),
    });
    app.world_mut()
        .resource_mut::<Orchestrator>()
        .queue_imported_model(model(PathBuf::new(), "waiting"));
    app.update();
    app.update();
    assert_eq!(
        app.world().resource::<Orchestrator>().import_state(),
        &ImportState::InProgress
    );
    assert!(
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .take_pending_load_request()
            .is_some()
    );
    drop(sender);
}

#[test]
fn the_import_action_queues_work_without_touching_the_filesystem() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let mut orchestrator = Orchestrator::new(root.clone());
    orchestrator.process_action(&UiAction::ImportAvatar {
        path: directory.path().join("missing.vrm"),
    });
    assert_eq!(orchestrator.import_state(), &ImportState::InProgress);
    assert!(orchestrator.last_error().is_none());
    assert!(!root.exists());
}
