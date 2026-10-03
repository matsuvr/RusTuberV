//! Application pipeline wiring: worker ownership, system ordering, and shutdown.
//!
//! Called by `UiShellPlugin` after its shared resources exist. The face and Pose
//! workers share the capture source; shutdown stops consumers before capture.

use bevy::prelude::*;
use vtuber_avatar::{AvatarOutputState, apply_arm_pose_profile_changes};

use crate::capture_runtime::{
    CaptureRuntime, LatestVideoFrame, capture_bridge_system, default_camera_backend,
    read_latest_frame, register_preview_texture_system, sync_capture_diagnostics,
    update_preview_texture_system,
};
use crate::diagnostics::{DiagnosticsSnapshot, sync_engine_diagnostics};
use crate::error_presenter::ErrorPresenter;
use crate::inference_runtime::{
    InferenceProjectRoot, InferenceRuntime, inference_bridge_system, read_inference_output_system,
};
use crate::metrics_export::export_diagnostics_system;
use crate::ndi_output::{
    NdiOutputRuntime, ndi_output_bridge_system, shutdown_ndi_output,
    sync_ndi_output_view_model_system,
};
use crate::orchestrator::{
    Orchestrator, process_ui_actions_system, sync_avatar_lifecycle_system,
    sync_expression_view_model,
};
use crate::pose_runtime::{
    PoseRuntime, pose_source_selection_system, pose_worker_bridge_system, read_pose_output_system,
};
use crate::preview_landmarks::sync_preview_landmark_system;
use crate::settings::{
    AppSettings, restore_arm_pose_settings_system, restore_expression_binding_settings_system,
};
use crate::tracking_runtime::{TrackingRuntime, tracking_bridge_system};

pub(crate) fn configure_pipeline(app: &mut App) {
    let project_root = app
        .world()
        .get_resource::<InferenceProjectRoot>()
        .map(|root| root.0.clone())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    init_tracking_runtimes(app, project_root);
    app.add_systems(
        Startup,
        (
            restore_arm_pose_settings_system,
            restore_expression_binding_settings_system,
            crate::pose_runtime::restore_pose_settings_system,
            crate::tracking_runtime::load_eye_closure_profile_system,
        ),
    )
    .add_systems(
        Update,
        (
            // Look-change messages written by the action processing must
            // reach the avatar side in the same frame.
            process_ui_actions_system.before(vtuber_avatar::look::apply_look_settings_changes),
            apply_arm_pose_profile_changes,
            sync_avatar_lifecycle_system
                .after(vtuber_avatar::unload::despawn_unloading_avatar)
                .before(vtuber_avatar::look::apply_look_settings_changes),
        )
            .chain(),
    )
    .configure_sets(
        Update,
        vtuber_avatar::ManualExpressionSet.after(process_ui_actions_system),
    )
    .add_systems(
        Update,
        sync_expression_view_model
            .after(process_ui_actions_system)
            .after(vtuber_avatar::ManualExpressionSet),
    )
    .add_systems(
        Update,
        auto_start_tracking_system
            .after(sync_avatar_lifecycle_system)
            .before(inference_bridge_system),
    )
    .add_systems(
        Update,
        sync_error_presenter
            .after(sync_avatar_lifecycle_system)
            .after(sync_capture_diagnostics),
    )
    .add_systems(
        Update,
        ndi_output_bridge_system.after(sync_avatar_lifecycle_system),
    )
    .add_systems(
        Update,
        sync_ndi_output_view_model_system.after(ndi_output_bridge_system),
    )
    .add_systems(
        Update,
        (
            capture_bridge_system,
            read_latest_frame,
            update_preview_texture_system,
            register_preview_texture_system,
            sync_capture_diagnostics,
        )
            .chain(),
    )
    .add_systems(
        Update,
        (inference_bridge_system, read_inference_output_system)
            .chain()
            .before(capture_bridge_system),
    )
    .add_systems(
        Update,
        sync_preview_landmark_system.after(read_inference_output_system),
    )
    .add_systems(
        Update,
        tracking_bridge_system.after(read_inference_output_system),
    )
    .add_systems(
        Update,
        (
            pose_worker_bridge_system,
            read_pose_output_system,
            pose_source_selection_system,
        )
            .chain()
            .after(read_inference_output_system)
            .after(capture_bridge_system),
    )
    .add_systems(
        Last,
        (sync_engine_diagnostics, export_diagnostics_system)
            .chain()
            .before(shutdown_workers_on_exit),
    )
    .add_systems(Last, shutdown_workers_on_exit);
}

/// Installs the Pose, capture and inference runtimes, keeping any the caller
/// inserted first.
///
/// A caller that wires a Pose/Capture pair on one slot before this plugin runs
/// keeps both halves: replacing only the Pose runtime would leave the capture
/// worker publishing into the previous slot, so the arm input never arrives.
/// The Pose slot is taken from the runtime that is actually retained, and an
/// existing capture runtime keeps its backend, worker and sinks untouched; no
/// running controller is rebuilt here.
fn init_tracking_runtimes(app: &mut App, project_root: std::path::PathBuf) {
    // The Pose consumer exists before the capture runtime so its slot can be
    // handed to the constructor: one camera open serves both face and Pose, and
    // the output is fixed before any worker can start.
    if !app.world().contains_resource::<PoseRuntime>() {
        app.insert_resource(PoseRuntime::new(project_root.clone()));
    }
    app.init_resource::<TrackingRuntime>()
        .insert_resource(LatestVideoFrame::default());
    let pose_slot = app.world().resource::<PoseRuntime>().frame_slot();
    if !app.world().contains_resource::<CaptureRuntime>() {
        app.insert_resource(CaptureRuntime::with_backend_and_pose_output(
            default_camera_backend(),
            Some(pose_slot),
        ));
    }
    let frame_slot = app.world().resource::<CaptureRuntime>().frame_slot();
    app.insert_resource(InferenceRuntime::new(frame_slot, project_root));
}

/// Starts tracking as soon as the avatar is ready and a camera is selected,
/// so completing setup is enough and no extra Start press is needed.
fn auto_start_tracking_system(mut orchestrator: ResMut<Orchestrator>) {
    orchestrator.maybe_auto_start_tracking();
}

fn shutdown_workers_on_exit(
    mut exits: MessageReader<AppExit>,
    mut pose: ResMut<crate::pose_runtime::PoseRuntime>,
    mut inference: ResMut<InferenceRuntime>,
    mut capture: ResMut<CaptureRuntime>,
    ndi: Option<ResMut<NdiOutputRuntime>>,
    output: Option<ResMut<AvatarOutputState>>,
) {
    if exits.read().next().is_some() {
        shutdown_ndi_output(ndi, output);
        if let Err(error) = pose.stop() {
            error!("pose shutdown failed: {error}");
        }
        if let Err(error) = inference.stop_model() {
            error!("inference shutdown failed: {error}");
        }
        if let Err(error) = capture.shutdown() {
            error!("capture shutdown failed: {error}");
        }
    }
}

fn sync_error_presenter(
    orchestrator: Res<Orchestrator>,
    mut presenter: ResMut<ErrorPresenter>,
    mut diagnostics: ResMut<DiagnosticsSnapshot>,
    settings: Option<Res<AppSettings>>,
) {
    let lang = settings
        .as_deref()
        .map(|settings| settings.language())
        .unwrap_or_default();
    let error = orchestrator.last_error();
    presenter.update(error, lang);
    match error {
        Some(error) => {
            let presentation = crate::error_presenter::present_error(error, lang);
            diagnostics.last_error = Some(presentation.user_message);
            diagnostics.last_error_code = Some(presentation.code.to_owned());
        }
        None => {
            diagnostics.last_error = None;
            diagnostics.last_error_code = None;
        }
    }
}

#[cfg(test)]
mod tests;
