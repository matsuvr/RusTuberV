// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Desktop UI vertical smoke test.
//!
//! Exercises the full UI action → orchestrator → view model pipeline
//! without requiring a real camera or VRM model.

use vtuber_app::actions::UiAction;
use vtuber_app::orchestrator::Orchestrator;
use vtuber_app::ui_model::*;
use vtuber_camera::device::CameraDescriptor;

/// Smoke test: full UI action flow.
#[test]
fn smoke_ui_action_flow() {
    let mut orch = Orchestrator::default();
    let mut vm = UiViewModel::default();

    // 1. Refresh cameras.
    orch.process_action(&UiAction::RefreshCameras);
    orch.set_camera_list(vec![CameraDescriptor {
        id: "test:0".into(),
        label: "Test camera".into(),
    }]);
    vm.update_from_orchestrator(&orch);
    assert!(!vm.camera.available_cameras.is_empty());

    // 2. Select camera.
    orch.process_action(&UiAction::SelectCamera { index: 0 });
    vm.update_from_orchestrator(&orch);
    assert_eq!(vm.camera.selected_index, Some(0));

    // 3. Camera selection waits quietly for an avatar.
    orch.maybe_auto_start_tracking();
    assert!(!orch.capture_desired());
    assert!(orch.last_error().is_none());

    // 4. Unload when no avatar is safe.
    orch.process_action(&UiAction::UnloadAvatar);
    assert!(orch.last_error().is_none());
}

/// Smoke test: view model reflects orchestrator state.
#[test]
fn smoke_view_model_reflects_state() {
    let mut orch = Orchestrator::default();
    let mut vm = UiViewModel::default();

    // Initial state.
    vm.update_from_orchestrator(&orch);
    assert_eq!(vm.lifecycle, AppLifecycle::Idle);
    assert!(vm.camera.available_cameras.is_empty());
    assert!(vm.camera.selected_index.is_none());

    // After camera refresh + selection.
    orch.process_action(&UiAction::RefreshCameras);
    orch.set_camera_list(vec![CameraDescriptor {
        id: "test:0".into(),
        label: "Test camera".into(),
    }]);
    orch.process_action(&UiAction::SelectCamera { index: 0 });
    vm.update_from_orchestrator(&orch);
    assert!(!vm.camera.available_cameras.is_empty());
    assert_eq!(vm.camera.selected_index, Some(0));
}

/// Smoke test: error presenter doesn't duplicate errors.
#[test]
fn smoke_error_presenter_no_duplicate() {
    use vtuber_app::error_presenter::ErrorPresenter;
    use vtuber_app::orchestrator::OrchestratorError;
    use vtuber_app::settings::UiLanguage;

    let mut presenter = ErrorPresenter::default();

    // First error is presented.
    let err = OrchestratorError::NoCameraSelected;
    assert!(presenter.update(Some(&err), UiLanguage::Ja));

    // Same error is not re-presented.
    assert!(!presenter.update(Some(&err), UiLanguage::Ja));

    // Dismiss clears it.
    presenter.dismiss();
    assert!(presenter.current().is_none());

    // Can present again after dismiss.
    assert!(presenter.update(Some(&err), UiLanguage::Ja));
}

/// Smoke test: preview toggle doesn't affect tracking.
#[test]
fn smoke_preview_toggle_safe() {
    use vtuber_app::preview::PreviewState;

    let mut preview = PreviewState::default();
    assert!(preview.visible);
    assert!(preview.mirrored);

    preview.toggle_mirrored();
    assert!(!preview.mirrored);

    preview.toggle_visible();
    assert!(!preview.visible);

    // No tracking state is affected.
}

/// Smoke test: diagnostics snapshot.
#[test]
fn smoke_diagnostics_snapshot() {
    use vtuber_app::diagnostics::DiagnosticsSnapshot;

    let snap = DiagnosticsSnapshot::default();
    assert!(!snap.has_active_workers());
    assert!(snap.model_hash.is_none());

    let snap = DiagnosticsSnapshot {
        capture_rate: 30.0,
        inference_rate: 25.0,
        ..Default::default()
    };
    assert!(snap.has_active_workers());
}

/// A camera failure can be cleared through the UI action boundary.
#[test]
fn smoke_camera_error_can_be_cleared() {
    let mut orch = Orchestrator::default();
    orch.fail_camera("camera disconnected".into());
    assert!(orch.last_error().is_some());
    orch.process_action(&UiAction::RefreshCameras);
    orch.set_camera_list(vec![CameraDescriptor {
        id: "test:0".into(),
        label: "Test camera".into(),
    }]);
    orch.process_action(&UiAction::SelectCamera { index: 0 });
    orch.process_action(&UiAction::DismissError);
    assert!(orch.last_error().is_none());
}
