//! Bevy/egui integration. Domain workers keep their existing scheduling;
//! layout and consent decisions live in studio and privacy respectively.

use super::fonts::{UiFonts, configure_fonts};
use super::privacy::{CameraPreviewConsent, CameraPreviewEvent, next_camera_preview_consent};
use crate::actions::UiAction;
#[cfg(not(feature = "dev-synthetic-input"))]
use crate::avatar_bridge::publish_control_frame_system;
use crate::avatar_bridge::sync_avatar_diagnostics;
use crate::capture_runtime::{
    CaptureRuntime, LatestVideoFrame, capture_bridge_system, read_latest_frame,
    register_preview_texture_system, sync_capture_diagnostics, update_preview_texture_system,
};
use crate::diagnostics::{DiagnosticsSnapshot, sync_engine_diagnostics};
use crate::error_presenter::ErrorPresenter;
use crate::expression_keys::ExpressionBindingStore;
use crate::inference_runtime::{
    InferenceProjectRoot, InferenceRuntime, inference_bridge_system, read_inference_output_system,
};
use crate::metrics_export::{MetricsExportState, export_diagnostics_system};
use crate::ndi_output::{
    NdiOutputIntent, NdiOutputRuntime, ndi_output_bridge_system, shutdown_ndi_output,
    sync_ndi_output_view_model_system,
};
use crate::orchestrator::{
    Orchestrator, process_ui_actions_system, sync_avatar_lifecycle_system,
    sync_expression_view_model,
};
use crate::pose_runtime::{
    PoseRuntime, pose_source_selection_system, pose_worker_bridge_system, read_pose_output_system,
};
use crate::preview::PreviewState;
use crate::preview_landmarks::{PreviewLandmarkState, sync_preview_landmark_system};
use crate::settings::{
    ArmPoseSettings, restore_arm_pose_settings_system, restore_expression_binding_settings_system,
};
use crate::tracking_runtime::{TrackingRuntime, tracking_bridge_system};
use crate::ui_model::{Pane, UiViewModel};
use bevy::prelude::*;
use bevy_egui::{
    EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPostUpdateSet, EguiPrimaryContextPass,
    PrimaryEguiContext, egui,
};
use vtuber_avatar::{
    AvatarMotionMirror, AvatarOutputCamera, AvatarOutputState, AvatarOutputTarget,
    AvatarViewportCamera, CameraInputSet, CameraPointerInputGate, apply_arm_pose_profile_changes,
};

/// Session-local UI state. Camera consent is deliberately never persisted.
#[derive(Resource, Debug)]
pub struct UiState {
    /// Commands awaiting the application orchestrator.
    pub pending_actions: Vec<UiAction>,
    pub(crate) controls_open: bool,
    /// Avatar-only transition progress owned by egui: `0.0` is the full
    /// workspace, `1.0` is avatar-only. Intermediate values mean the workspace
    /// is collapsing (or expanding) and the monitor preview is expanding (or
    /// collapsing) with it.
    pub(crate) avatar_only_progress: f32,
    /// Last laid-out avatar monitor image rectangle, reused as the starting
    /// rect of the next preview transition.
    pub(crate) monitor_image_rect: Option<egui::Rect>,
    pub(crate) camera_consent: CameraPreviewConsent,
    last_pane: Option<Pane>,
    pub(crate) import_requested: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            pending_actions: Vec::new(),
            controls_open: true,
            avatar_only_progress: 0.0,
            monitor_image_rect: None,
            camera_consent: CameraPreviewConsent::Hidden,
            last_pane: None,
            import_requested: false,
        }
    }
}

impl UiState {
    /// Emit a command, deduplicating the existing one-shot UI actions.
    pub fn emit(&mut self, action: UiAction) {
        if matches!(
            action,
            UiAction::SwitchPane(_)
                | UiAction::SelectCamera { .. }
                | UiAction::Stop
                | UiAction::UnloadAvatar
                | UiAction::RequestAvatarImportReview { .. }
        ) {
            self.preview_event(CameraPreviewEvent::Hide);
        }
        if is_deduplicatable(&action) && self.pending_actions.contains(&action) {
            return;
        }
        self.pending_actions.push(action);
    }

    /// Drain pending commands.
    pub fn take_actions(&mut self) -> Vec<UiAction> {
        std::mem::take(&mut self.pending_actions)
    }

    pub(crate) fn preview_event(&mut self, event: CameraPreviewEvent) {
        self.camera_consent = next_camera_preview_consent(self.camera_consent, event);
    }

    pub(crate) fn set_controls_open(&mut self, open: bool) {
        self.controls_open = open;
        self.preview_event(CameraPreviewEvent::Hide);
    }

    fn sync_pane(&mut self, pane: Pane) {
        if self.last_pane != Some(pane) {
            self.preview_event(CameraPreviewEvent::Hide);
            self.last_pane = Some(pane);
        }
    }
}

fn is_deduplicatable(action: &UiAction) -> bool {
    matches!(
        action,
        UiAction::SwitchPane(_)
            | UiAction::ToggleMirror
            | UiAction::TogglePreview
            | UiAction::ToggleAvatarMotionMirror
            | UiAction::SetLanguage(_)
            | UiAction::DismissError
            | UiAction::StartNdiOutput
            | UiAction::StopNdiOutput
    )
}

/// Installs the studio UI and its bridges to existing domain services.
pub struct UiShellPlugin;

impl Plugin for UiShellPlugin {
    fn build(&self, app: &mut App) {
        // Invariant: the desktop installs EguiPlugin before this plugin.
        assert!(
            app.is_plugin_added::<EguiPlugin>(),
            "UiShellPlugin requires EguiPlugin to be installed first"
        );
        app.add_plugins(super::avatar_preview::AvatarPreviewPlugin);
        // The first camera may be the offscreen output camera. Never attach
        // egui there: UI pixels must stay on the window-targeting camera only.
        app.world_mut()
            .resource_mut::<EguiGlobalSettings>()
            .auto_create_primary_context = false;
        app.add_systems(Update, attach_primary_egui_context_to_viewport_camera);
        // One background color for the 3D avatar-only view and the egui
        // transition mask, so the preview card can dissolve into the scene.
        let background = super::studio::STUDIO_BACKGROUND;
        app.insert_resource(ClearColor(Color::srgb_u8(
            background.r(),
            background.g(),
            background.b(),
        )));
        app.init_resource::<UiState>()
            .init_resource::<UiFonts>()
            .init_resource::<UiViewModel>()
            .init_resource::<Orchestrator>()
            .init_resource::<ArmPoseSettings>()
            .init_resource::<ExpressionBindingStore>()
            .insert_resource(PreviewState {
                visible: false,
                ..Default::default()
            })
            .init_resource::<PreviewLandmarkState>()
            .init_resource::<NdiOutputIntent>()
            .init_resource::<NdiOutputRuntime>()
            .init_resource::<AvatarMotionMirror>()
            .init_resource::<CameraPointerInputGate>()
            .init_resource::<UiSurfaceHover>()
            .init_resource::<DiagnosticsSnapshot>()
            .init_resource::<MetricsExportState>()
            .init_resource::<ErrorPresenter>()
            .init_resource::<super::file_dialog::FileDialogState>()
            .init_resource::<CaptureRuntime>()
            .init_resource::<LatestVideoFrame>();
        app.world_mut()
            .resource_mut::<UiState>()
            .emit(UiAction::SwitchPane(Pane::Studio));
        let frame_slot = app.world().resource::<CaptureRuntime>().frame_slot();
        let project_root = app
            .world()
            .get_resource::<InferenceProjectRoot>()
            .map(|root| root.0.clone())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        app.insert_resource(InferenceRuntime::new(frame_slot, project_root.clone()))
            .insert_resource(PoseRuntime::new(project_root))
            .init_resource::<TrackingRuntime>();
        // Wire the camera fan-out to the Pose slot before the capture worker is
        // ever started, so one camera open serves both face and Pose.
        let pose_slot = app.world().resource::<PoseRuntime>().frame_slot();
        app.world_mut()
            .resource_mut::<CaptureRuntime>()
            .set_pose_output(Some(pose_slot));
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
                process_ui_actions_system,
                apply_arm_pose_profile_changes,
                sync_avatar_lifecycle_system,
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
            auto_start_tracking_system.after(sync_avatar_lifecycle_system),
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
        .add_systems(Last, shutdown_workers_on_exit)
        .add_systems(
            PostUpdate,
            sync_camera_pointer_input_gate
                .after(EguiPostUpdateSet::ProcessOutput)
                .before(CameraInputSet),
        )
        .add_systems(
            EguiPrimaryContextPass,
            (configure_fonts, ui_render_system).chain(),
        );
        #[cfg(not(feature = "dev-synthetic-input"))]
        app.add_systems(
            Update,
            publish_control_frame_system.after(tracking_bridge_system),
        )
        .add_systems(
            Update,
            sync_avatar_diagnostics.after(publish_control_frame_system),
        );
        #[cfg(feature = "dev-synthetic-input")]
        app.insert_resource(crate::synthetic_tracking::SyntheticTrackingSource::default())
            .add_systems(
                Update,
                crate::synthetic_tracking::synthetic_tracking_system.after(tracking_bridge_system),
            )
            .add_systems(
                Update,
                sync_avatar_diagnostics.after(crate::synthetic_tracking::synthetic_tracking_system),
            );
    }
}

/// Starts tracking as soon as the avatar is ready and a camera is selected,
/// so completing setup is enough and no extra Start press is needed.
fn auto_start_tracking_system(mut orchestrator: ResMut<Orchestrator>) {
    orchestrator.maybe_auto_start_tracking();
}

fn attach_primary_egui_context_to_viewport_camera(
    mut commands: Commands,
    viewport_cameras: Query<Entity, (With<AvatarViewportCamera>, Without<AvatarOutputCamera>)>,
    primary_contexts: Query<(), With<PrimaryEguiContext>>,
) {
    if !primary_contexts.is_empty() {
        return;
    }
    if let Ok(entity) = viewport_cameras.single() {
        commands.entity(entity).insert(PrimaryEguiContext);
    }
}

/// Pointer ownership for panels drawn in the shell's own egui root.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UiSurfaceHover {
    over_ui: bool,
}
impl UiSurfaceHover {
    /// Record pointer ownership for this frame.
    pub fn set(&mut self, over_ui: bool) {
        self.over_ui = over_ui;
    }
    /// Whether a studio surface owns the pointer.
    #[must_use]
    pub const fn over_ui(self) -> bool {
        self.over_ui
    }
}

fn sync_camera_pointer_input_gate(
    egui_wants_input: Res<bevy_egui::input::EguiWantsInput>,
    hover: Res<UiSurfaceHover>,
    mut gate: ResMut<CameraPointerInputGate>,
) {
    gate.set_egui_owns_pointer(egui_wants_input.wants_any_pointer_input() || hover.over_ui());
}

fn shutdown_workers_on_exit(
    mut exits: MessageReader<AppExit>,
    mut inference: ResMut<InferenceRuntime>,
    mut capture: ResMut<CaptureRuntime>,
    ndi: Option<ResMut<NdiOutputRuntime>>,
    output: Option<ResMut<AvatarOutputState>>,
) {
    if exits.read().next().is_some() {
        shutdown_ndi_output(ndi, output);
        inference.stop_model();
        capture.shutdown();
    }
}

fn sync_error_presenter(
    orchestrator: Res<Orchestrator>,
    mut presenter: ResMut<ErrorPresenter>,
    mut diagnostics: ResMut<DiagnosticsSnapshot>,
    settings: Option<Res<ArmPoseSettings>>,
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

#[allow(clippy::too_many_arguments)]
fn ui_render_system(
    mut contexts: EguiContexts,
    vm: Res<UiViewModel>,
    mut state: ResMut<UiState>,
    diagnostics: Res<DiagnosticsSnapshot>,
    errors: Res<ErrorPresenter>,
    mut preview: ResMut<PreviewState>,
    landmarks: Res<PreviewLandmarkState>,
    avatar_mirror: Res<AvatarMotionMirror>,
    settings: Res<ArmPoseSettings>,
    fonts: Res<UiFonts>,
    target: Option<Res<AvatarOutputTarget>>,
    mut output: Option<ResMut<AvatarOutputState>>,
    mut file_dialog: ResMut<super::file_dialog::FileDialogState>,
    mut hover: ResMut<UiSurfaceHover>,
) -> Result {
    super::file_dialog::poll_file_dialog(&mut file_dialog, &mut state);
    let camera_texture = preview
        .image_handle
        .as_ref()
        .and_then(|image| contexts.image_id(image.id()));
    let avatar_texture = target.as_ref().map(|target| {
        super::avatar_preview::AvatarPreviewTexture::new(
            target.image().clone(),
            target.profile(),
        )
    });
    let ctx = contexts.ctx_mut()?;
    state.sync_pane(vm.pane);
    if ctx.input(|input| input.key_pressed(bevy_egui::egui::Key::F1)) {
        let open = !state.controls_open;
        state.set_controls_open(open);
    }
    if ctx.input(|input| input.key_pressed(bevy_egui::egui::Key::Escape)) {
        state.preview_event(CameraPreviewEvent::Hide);
    }
    let over_ui = super::studio::render_studio(
        ctx,
        &vm,
        &mut state,
        &diagnostics,
        errors.current(),
        &preview,
        &landmarks,
        *avatar_mirror,
        camera_texture,
        avatar_texture,
        file_dialog.is_active(),
        fonts.error.as_deref(),
        settings.language(),
    );
    hover.set(over_ui);
    // Expression keys are collected after the UI pass so the same frame's
    // keyboard ownership (text edit, combo popup, modal) is respected. This
    // runs outside `render_studio`, so avatar-only (F1 hidden) still works.
    super::studio::expression_key_input(ctx, &vm, &mut state, file_dialog.is_active());
    // Uploads depend on explicit session consent; capture and inference do not.
    preview.visible = state.controls_open
        && matches!(vm.pane, Pane::Camera | Pane::Preview)
        && state.camera_consent == CameraPreviewConsent::Visible;
    // A running transition samples the same avatar texture the monitor uses,
    // so the offscreen render outlives the settings until the card is gone.
    let transitioning = state.avatar_only_progress > 0.0 && state.avatar_only_progress < 1.0;
    if let Some(output) = output.as_deref_mut() {
        output.set_preview_visible(vm.avatar.is_ready && (state.controls_open || transitioning));
    }
    // The UI requests a dialog; this boundary owns filesystem/dialog effects.
    if std::mem::take(&mut state.import_requested) {
        file_dialog.start(settings.language());
    }
    super::file_dialog::handle_dropped_files(ctx, &mut state);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_state_emit_and_take() {
        let mut state = UiState::default();
        assert!(state.pending_actions.is_empty());
        state.emit(UiAction::Start);
        state.emit(UiAction::Stop);
        assert_eq!(state.take_actions(), vec![UiAction::Start, UiAction::Stop]);
        assert!(state.pending_actions.is_empty());
    }

    #[test]
    fn navigation_is_deduplicated_and_always_revokes_camera_consent() {
        let mut state = UiState {
            camera_consent: CameraPreviewConsent::Visible,
            ..Default::default()
        };
        state.emit(UiAction::SwitchPane(Pane::Camera));
        state.emit(UiAction::SwitchPane(Pane::Camera));
        assert_eq!(state.pending_actions.len(), 1);
        assert_eq!(state.camera_consent, CameraPreviewConsent::Hidden);
        state.take_actions();
        state.emit(UiAction::SwitchPane(Pane::Camera));
        assert_eq!(state.pending_actions.len(), 1);
    }

    #[test]
    fn closing_and_reopening_controls_never_reveals_the_camera() {
        let mut state = UiState::default();
        assert!(state.controls_open);
        state.camera_consent = CameraPreviewConsent::Visible;
        state.set_controls_open(false);
        state.set_controls_open(true);
        assert_eq!(state.camera_consent, CameraPreviewConsent::Hidden);
    }

    #[test]
    fn camera_change_stop_and_unload_revoke_preview() {
        for action in [
            UiAction::SelectCamera { index: 1 },
            UiAction::Stop,
            UiAction::UnloadAvatar,
        ] {
            let mut state = UiState {
                camera_consent: CameraPreviewConsent::Visible,
                ..Default::default()
            };
            state.emit(action);
            assert_eq!(state.camera_consent, CameraPreviewConsent::Hidden);
        }
    }

    #[test]
    fn same_page_rerender_preserves_consent_but_external_navigation_does_not() {
        let mut state = UiState::default();
        state.sync_pane(Pane::Camera);
        state.camera_consent = CameraPreviewConsent::Visible;
        state.sync_pane(Pane::Camera);
        assert_eq!(state.camera_consent, CameraPreviewConsent::Visible);
        state.sync_pane(Pane::Studio);
        assert_eq!(state.camera_consent, CameraPreviewConsent::Hidden);
    }

    #[test]
    fn ui_state_emit_deduplicates_toggles_not_start() {
        let mut state = UiState::default();
        state.emit(UiAction::ToggleMirror);
        state.emit(UiAction::ToggleMirror);
        state.emit(UiAction::ToggleAvatarMotionMirror);
        state.emit(UiAction::ToggleAvatarMotionMirror);
        assert_eq!(state.take_actions().len(), 2);
        state.emit(UiAction::Start);
        state.emit(UiAction::Start);
        assert_eq!(state.take_actions().len(), 2);
    }

    #[test]
    fn egui_attaches_only_to_the_window_avatar_camera_not_the_output_target() {
        let mut app = App::new();
        let output = app.world_mut().spawn(AvatarOutputCamera).id();
        let viewport = app
            .world_mut()
            .spawn(AvatarViewportCamera::from_default_transform(
                Transform::default(),
            ))
            .id();
        app.add_systems(Update, attach_primary_egui_context_to_viewport_camera);
        app.update();
        assert!(app.world().get::<PrimaryEguiContext>(viewport).is_some());
        assert!(app.world().get::<PrimaryEguiContext>(output).is_none());
    }

    #[test]
    fn sync_error_presenter_updates_translated_summary_and_diagnostics() {
        let mut app = App::new();
        app.init_resource::<Orchestrator>()
            .init_resource::<ErrorPresenter>()
            .init_resource::<DiagnosticsSnapshot>()
            .init_resource::<ArmPoseSettings>()
            .add_systems(Update, sync_error_presenter);
        // Exercise the existing public action boundary; Start without a
        // selected camera produces NoCameraSelected without camera/file I/O.
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .process_action(&UiAction::Start);
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
}
