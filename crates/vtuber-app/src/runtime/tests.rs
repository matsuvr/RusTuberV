#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use super::*;
use crate::actions::UiAction;
use crate::preview::PreviewState;
use crate::ui::UiState;
use crate::ui_model::UiViewModel;
use std::sync::Arc;
use vtuber_avatar::AvatarMotionMirror;

#[test]
fn a_pre_wired_pose_capture_pair_survives_the_shell_initialization() {
    let mut app = App::new();
    let pose = PoseRuntime::new(std::path::PathBuf::from("."));
    let pose_slot = pose.frame_slot();
    app.insert_resource(pose).insert_resource(
        crate::capture_runtime::CaptureRuntime::with_backend_and_pose_output(
            crate::capture_runtime::CameraBackendKind::Mock,
            Some(Arc::clone(&pose_slot)),
        ),
    );

    init_tracking_runtimes(&mut app, std::path::PathBuf::from("."));

    // The connected Pose runtime is kept, not replaced by a fresh one, and
    // the caller's capture runtime keeps its Mock backend.
    assert!(Arc::ptr_eq(
        &app.world().resource::<PoseRuntime>().frame_slot(),
        &pose_slot
    ));
    assert!(matches!(
        app.world()
            .resource::<crate::capture_runtime::CaptureRuntime>()
            .backend_kind(),
        crate::capture_runtime::CameraBackendKind::Mock
    ));
}

#[test]
fn initialization_creates_default_capture_pose_and_face_runtimes() {
    let mut app = App::new();
    init_tracking_runtimes(&mut app, std::path::PathBuf::from("."));

    assert_eq!(
        app.world()
            .resource::<crate::capture_runtime::CaptureRuntime>()
            .backend_kind()
            .name(),
        default_camera_backend().name()
    );
    assert!(app.world().contains_resource::<PoseRuntime>());
    assert!(app.world().contains_resource::<InferenceRuntime>());
}

#[test]
fn sync_error_presenter_records_camera_error_code_and_presentation() {
    let mut app = App::new();
    app.init_resource::<Orchestrator>()
        .init_resource::<ErrorPresenter>()
        .init_resource::<DiagnosticsSnapshot>()
        .init_resource::<AppSettings>()
        .add_systems(Update, sync_error_presenter);
    app.world_mut()
        .resource_mut::<Orchestrator>()
        .set_last_error(Some(
            crate::orchestrator::OrchestratorError::NoCameraSelected,
        ));
    app.update();
    assert_eq!(
        app.world()
            .resource::<DiagnosticsSnapshot>()
            .last_error_code
            .as_deref(),
        Some("NO_CAMERA")
    );
    assert!(app.world().resource::<ErrorPresenter>().current().is_some());
}

#[test]
fn auto_start_system_starts_tracking_when_lifecycle_reports_ready() {
    let mut app = App::new();
    app.init_resource::<Orchestrator>()
        .init_resource::<UiState>()
        .init_resource::<UiViewModel>()
        .init_resource::<PreviewState>()
        .init_resource::<AvatarMotionMirror>()
        .init_resource::<vtuber_avatar::AvatarLifecycle>()
        .add_message::<vtuber_avatar::LoadImportedAvatarRequest>()
        .add_message::<vtuber_avatar::LoadImportedAvatarResult>()
        .add_message::<vtuber_avatar::lifecycle::UnloadAvatarRequest>()
        .add_systems(
            Update,
            (sync_avatar_lifecycle_system, auto_start_tracking_system).chain(),
        );

    {
        let mut orchestrator = app.world_mut().resource_mut::<Orchestrator>();
        orchestrator.set_imported_model_for_tests(Some(crate::import::ImportedModel {
            id: "test".into(),
            name: "test".into(),
            asset_path: std::path::PathBuf::new(),
            meta_path: std::path::PathBuf::new(),
            summary: crate::import::VrmInspectionSummary::default(),
            original_path: std::path::PathBuf::new(),
            size: 0,
        }));
        orchestrator.set_camera_list(vec![vtuber_camera::device::CameraDescriptor {
            id: "test:0".into(),
            label: "Test camera".into(),
        }]);
        orchestrator.process_action(&UiAction::SelectCamera { index: 0 });
    }

    let root = app.world_mut().spawn_empty().id();
    {
        let mut lifecycle = app
            .world_mut()
            .resource_mut::<vtuber_avatar::AvatarLifecycle>();
        lifecycle.request_load(root).expect("test load is valid");
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
    }

    app.update();

    let orchestrator = app.world().resource::<Orchestrator>();
    assert_eq!(
        orchestrator.pipeline_state(),
        crate::orchestrator::PipelineState::Starting
    );
    assert!(orchestrator.capture_desired());
}
