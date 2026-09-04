//! UI shell — the bevy_egui integration layer.
//!
//! Provides the [`UiShellPlugin`] which sets up egui and renders the
//! three main screens: Setup, Live, and Diagnostics.

use bevy::prelude::*;
use bevy_egui::{
    EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPostUpdateSet, EguiPrimaryContextPass,
    PrimaryEguiContext,
};

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
use crate::face_backend::{
    FaceTrackingBackendState, restore_face_backend_selection_system,
    sync_face_backend_diagnostics_system,
};
use crate::inference_runtime::{
    InferenceProjectRoot, InferenceRuntime, inference_bridge_system, read_inference_output_system,
};
use crate::metrics_export::{MetricsExportState, export_diagnostics_system};
use crate::ndi_output::{
    NdiOutputIntent, NdiOutputRuntime, ndi_output_bridge_system, shutdown_ndi_output,
    sync_ndi_output_view_model_system,
};
use crate::orchestrator::{Orchestrator, process_ui_actions_system, sync_avatar_lifecycle_system};
use crate::preview::PreviewState;
use crate::preview_landmarks::{PreviewLandmarkState, sync_preview_landmark_system};
use crate::settings::{ArmPoseSettings, restore_arm_pose_settings_system};
use crate::tracking_runtime::{TrackingRuntime, tracking_bridge_system};
use crate::ui_model::{Screen, UiViewModel};
use vtuber_avatar::{
    AvatarMotionMirror, AvatarViewportCamera, CameraInputSet, CameraPointerInputGate,
    apply_arm_pose_profile_changes,
};

use super::diagnostics::render_diagnostics_screen;
use super::live::render_live_screen;
use super::setup::render_setup_screen;

const DRAWER_WIDTH: f32 = 340.0;
const JAPANESE_FONT_NAME: &str = "LINESeedJP_A_Rg";
static JAPANESE_FONT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/fonts/LINESeedJP_A_TTF_Rg.ttf"
));

/// Fill of the drawer and its pull handle: near-black with a slight
/// translucency so the avatar scene stays perceptible at the rounded edge.
fn drawer_fill() -> bevy_egui::egui::Color32 {
    bevy_egui::egui::Color32::from_rgba_unmultiplied(17, 19, 26, 242)
}

/// The controls drawer: a fixed-width panel docked to the left edge of the
/// window, spanning the full height so it slides out over the avatar scene
/// like a vertical drawer.
fn control_drawer_panel() -> bevy_egui::egui::Panel {
    // `exact_size` pins the drawer geometry: egui persists panel sizes per Id,
    // so a stale in-session state can never widen or narrow the drawer.
    // Content growth is absorbed by the scroll area inside instead.
    bevy_egui::egui::Panel::left(bevy_egui::egui::Id::new("control_drawer"))
        .exact_size(DRAWER_WIDTH)
        .resizable(false)
        .show_separator_line(false)
        .frame(drawer_frame())
}

/// Flat modern style for the drawer: dark translucent fill with rounded
/// corners on the free (right) edge, flush against the left window edge.
fn drawer_frame() -> bevy_egui::egui::Frame {
    bevy_egui::egui::Frame::new()
        .fill(drawer_fill())
        .inner_margin(bevy_egui::egui::Margin::same(10))
        .corner_radius(bevy_egui::egui::CornerRadius {
            ne: 12,
            se: 12,
            ..bevy_egui::egui::CornerRadius::ZERO
        })
}

/// Full-viewport background-layer Ui that panels must be shown on.
///
/// bevy_egui hands systems only the raw egui Context, so this mirrors the
/// construction used by bevy_egui's own multipass examples.
fn root_viewport_ui(ctx: &bevy_egui::egui::Context) -> bevy_egui::egui::Ui {
    bevy_egui::egui::Ui::new(
        ctx.clone(),
        "control_drawer_root".into(),
        bevy_egui::egui::UiBuilder::new()
            .layer_id(bevy_egui::egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    )
}

/// Drawer header: title on the left, collapse control on the right.
fn drawer_header(ui: &mut bevy_egui::egui::Ui, hide_requested: &mut bool) {
    ui.horizontal(|ui| {
        ui.heading("Controls");
        ui.with_layout(
            bevy_egui::egui::Layout::right_to_left(bevy_egui::egui::Align::Center),
            |ui| {
                ui.small("F1");
                let close = bevy_egui::egui::Button::new("‹")
                    .min_size(bevy_egui::egui::vec2(26.0, 24.0))
                    .corner_radius(bevy_egui::egui::CornerRadius::same(6));
                if ui.add(close).on_hover_text("Hide Controls (F1)").clicked() {
                    *hide_requested = true;
                }
            },
        );
    });
    ui.add_space(4.0);
}

/// Vertical navigation list styled as full-width rounded items.
fn drawer_navigation(
    ui: &mut bevy_egui::egui::Ui,
    view_model: &UiViewModel,
    ui_state: &mut UiState,
) {
    for (screen, label) in [
        (Screen::Setup, "Setup"),
        (Screen::Live, "Live"),
        (Screen::Diagnostics, "Diagnostics"),
    ] {
        let item = bevy_egui::egui::Button::selectable(view_model.screen == screen, label)
            .min_size(bevy_egui::egui::vec2(ui.available_width(), 26.0))
            .corner_radius(bevy_egui::egui::CornerRadius::same(6));
        if ui.add(item).clicked() {
            ui_state.emit(UiAction::SwitchScreen(screen));
        }
    }
    ui.add_space(6.0);
}

/// Plugin that sets up the egui-based UI shell.
///
/// Requires `EguiPlugin` to be installed before this plugin.
pub struct UiShellPlugin;

impl Plugin for UiShellPlugin {
    fn build(&self, app: &mut App) {
        // Assert that EguiPlugin is already installed.
        assert!(
            app.is_plugin_added::<EguiPlugin>(),
            "UiShellPlugin requires EguiPlugin to be installed first"
        );

        // bevy_egui attaches the primary context to the first camera it sees,
        // which may be the inactive offscreen avatar-output camera spawned by
        // the avatar plugin's Startup systems. A context on that image target
        // renders the whole UI into a texture that is never presented, leaving
        // the window blank. Disable auto-attachment and bind the context to
        // the avatar viewport camera explicitly instead.
        app.world_mut()
            .resource_mut::<EguiGlobalSettings>()
            .auto_create_primary_context = false;
        app.add_systems(Update, attach_primary_egui_context_to_viewport_camera);

        app.init_resource::<UiState>()
            .init_resource::<JapaneseFontState>()
            .init_resource::<UiViewModel>()
            .init_resource::<Orchestrator>()
            .init_resource::<ArmPoseSettings>()
            .init_resource::<PreviewState>()
            .init_resource::<PreviewLandmarkState>()
            .init_resource::<NdiOutputIntent>()
            .init_resource::<NdiOutputRuntime>()
            .init_resource::<AvatarMotionMirror>()
            .init_resource::<CameraPointerInputGate>()
            .init_resource::<DiagnosticsSnapshot>()
            .init_resource::<MetricsExportState>()
            .init_resource::<ErrorPresenter>()
            .init_resource::<super::file_dialog::FileDialogState>()
            .init_resource::<CaptureRuntime>()
            .init_resource::<LatestVideoFrame>()
            .init_resource::<FaceTrackingBackendState>();

        let frame_slot = app.world().resource::<CaptureRuntime>().frame_slot();
        let project_root = app
            .world()
            .get_resource::<InferenceProjectRoot>()
            .map(|root| root.0.clone())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        app.insert_resource(InferenceRuntime::new(frame_slot, project_root))
            .init_resource::<TrackingRuntime>()
            .add_systems(Startup, restore_arm_pose_settings_system)
            .add_systems(Startup, restore_face_backend_selection_system)
            // Action processing, avatar pose re-resolution, backend authority,
            // and lifecycle sync are chained so a settings action reaches the
            // compositor in the same Update frame.
            .add_systems(
                Update,
                (
                    process_ui_actions_system,
                    apply_arm_pose_profile_changes,
                    sync_face_backend_diagnostics_system,
                    sync_avatar_lifecycle_system,
                )
                    .chain(),
            )
            // Synchronise the user-facing error after every producer has had a
            // chance to publish the current frame's error. This keeps import,
            // avatar lifecycle, camera, and inference failures visible in the
            // Setup tab without a race against another diagnostics sync.
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
            // Capture bridge: connects orchestrator intent to real camera.
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
                    // Inference owns the first half of shutdown. Capture is
                    // started by the following bridge once its state is
                    // visible, but is stopped only after inference has joined.
                    .before(capture_bridge_system),
            )
            .add_systems(Update, sync_preview_landmark_system.after(read_inference_output_system))
            .add_systems(Update, tracking_bridge_system.after(read_inference_output_system))
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
            // Install the embedded Japanese font before rendering the UI in
            // EguiPrimaryContextPass.
            .add_systems(
                EguiPrimaryContextPass,
                (configure_japanese_font, ui_render_system).chain(),
            );

        // The synthetic source is an explicit diagnostic build mode. It
        // replaces the real bridge rather than running beside it, so two
        // producers can never race on ActiveControlFrame.
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

/// Binds the primary Egui context to the avatar viewport camera.
///
/// Runs until exactly one primary context exists. The viewport camera is the
/// only window-targeting camera in the app, so egui renders onto the frame the
/// user actually sees. Re-attachment is skipped while any primary context is
/// alive so a camera replacement can never produce two primaries, which
/// bevy_egui rejects when schedules are reused.
fn attach_primary_egui_context_to_viewport_camera(
    mut commands: Commands,
    viewport_cameras: Query<Entity, (With<AvatarViewportCamera>, Without<PrimaryEguiContext>)>,
    primary_contexts: Query<(), With<PrimaryEguiContext>>,
) {
    if !primary_contexts.is_empty() {
        return;
    }
    if let Ok(entity) = viewport_cameras.single() {
        commands.entity(entity).insert(PrimaryEguiContext);
    }
}

/// Bridges the official egui pointer-ownership resource into the avatar camera
/// domain without letting UI code touch a camera transform or projection.
fn sync_camera_pointer_input_gate(
    egui_wants_input: Res<bevy_egui::input::EguiWantsInput>,
    mut gate: ResMut<CameraPointerInputGate>,
) {
    gate.set_egui_owns_pointer(egui_wants_input.wants_any_pointer_input());
}

/// Performs the explicit reverse-order shutdown required by the worker
/// ownership contract when Bevy is closing the application.
fn shutdown_workers_on_exit(
    mut exit_messages: MessageReader<AppExit>,
    mut inference: ResMut<InferenceRuntime>,
    mut capture: ResMut<CaptureRuntime>,
    ndi_output: Option<ResMut<crate::ndi_output::NdiOutputRuntime>>,
    output_state: Option<ResMut<vtuber_avatar::AvatarOutputState>>,
) {
    if exit_messages.read().next().is_some() {
        shutdown_ndi_output(ndi_output, output_state);
        inference.stop_model();
        capture.shutdown();
    }
}

/// Resource holding the current UI state and pending actions.
#[derive(Resource, Debug, Default)]
pub struct UiState {
    /// Actions emitted by the UI this frame.
    pub pending_actions: Vec<UiAction>,
    /// Session-local open state for the left control drawer.
    control_drawer: ControlDrawerState,
}

/// Tracks whether the embedded application font has been installed in egui.
#[derive(Resource, Debug, Default)]
struct JapaneseFontState {
    configured: bool,
}

/// Build the egui font definitions with LINE Seed JP as the primary UI font.
fn japanese_font_definitions() -> bevy_egui::egui::FontDefinitions {
    let mut fonts = bevy_egui::egui::FontDefinitions::default();
    fonts.font_data.insert(
        JAPANESE_FONT_NAME.to_owned(),
        std::sync::Arc::new(bevy_egui::egui::FontData::from_static(JAPANESE_FONT_BYTES)),
    );

    for family in [
        bevy_egui::egui::FontFamily::Proportional,
        bevy_egui::egui::FontFamily::Monospace,
    ] {
        if let Some(fonts_for_family) = fonts.families.get_mut(&family) {
            fonts_for_family.insert(0, JAPANESE_FONT_NAME.to_owned());
        }
    }

    fonts
}

/// Installs the bundled Japanese-capable font once the egui context exists.
fn configure_japanese_font(
    mut contexts: EguiContexts,
    mut font_state: ResMut<JapaneseFontState>,
) -> Result {
    if font_state.configured {
        return Ok(());
    }

    contexts.ctx_mut()?.set_fonts(japanese_font_definitions());
    font_state.configured = true;
    Ok(())
}

/// Session-local open state for the left control drawer.
///
/// The drawer starts open so first-run setup remains discoverable. Its
/// visibility is deliberately not persisted until the settings task owns the
/// configuration schema.
#[derive(Resource, Debug)]
struct ControlDrawerState {
    open: bool,
}

impl Default for ControlDrawerState {
    fn default() -> Self {
        Self { open: true }
    }
}

impl ControlDrawerState {
    fn toggle(&mut self) {
        self.open = !self.open;
    }
}

impl UiState {
    /// Emit a UI action, deduplicating one-shot actions within the same batch.
    pub fn emit(&mut self, action: UiAction) {
        // Deduplicate navigation and toggle actions within the same batch.
        if is_deduplicatable(&action) && self.pending_actions.contains(&action) {
            return;
        }
        self.pending_actions.push(action);
    }

    /// Take all pending actions, clearing the internal list.
    pub fn take_actions(&mut self) -> Vec<UiAction> {
        std::mem::take(&mut self.pending_actions)
    }
}

/// Check if an action should be deduplicated within a batch.
fn is_deduplicatable(action: &UiAction) -> bool {
    if matches!(action, UiAction::ToggleAvatarMotionMirror) {
        return true;
    }
    matches!(
        action,
        UiAction::SwitchScreen(_)
            | UiAction::ToggleMirror
            | UiAction::TogglePreview
            | UiAction::DismissError
            | UiAction::StartNdiOutput
            | UiAction::StopNdiOutput
    )
}

/// System that synchronizes the error presenter from the orchestrator.
fn sync_error_presenter(
    orchestrator: Res<Orchestrator>,
    mut error_presenter: ResMut<ErrorPresenter>,
    mut diagnostics: ResMut<DiagnosticsSnapshot>,
) {
    error_presenter.update(orchestrator.last_error());
    diagnostics.last_error = orchestrator.last_error().map(ToString::to_string);
    diagnostics.last_error_code = orchestrator
        .last_error()
        .map(crate::error_presenter::present_error)
        .map(|presentation| presentation.code.to_string());
}

/// System that renders the UI using egui.
///
/// Reads [`UiViewModel`] and emits [`UiAction`] via [`UiState`].
#[allow(clippy::too_many_arguments)]
fn ui_render_system(
    mut contexts: EguiContexts,
    view_model: Res<UiViewModel>,
    mut ui_state: ResMut<UiState>,
    diagnostics: Res<DiagnosticsSnapshot>,
    error_presenter: Res<ErrorPresenter>,
    preview: Res<PreviewState>,
    landmarks: Res<PreviewLandmarkState>,
    avatar_motion_mirror: Res<AvatarMotionMirror>,
    face_backend: Res<FaceTrackingBackendState>,
    mut file_dialog: ResMut<super::file_dialog::FileDialogState>,
) -> Result {
    let preview_texture = preview
        .image_handle
        .as_ref()
        .and_then(|handle| contexts.image_id(handle.id()));
    let ctx = contexts.ctx_mut()?;

    // Poll file dialog.
    super::file_dialog::poll_file_dialog(&mut file_dialog, &mut ui_state);

    if ctx.input(|input| input.key_pressed(bevy_egui::egui::Key::F1)) {
        ui_state.control_drawer.toggle();
    }

    let mut drawer_open = ui_state.control_drawer.open;
    let mut hide_requested = false;

    // Left control drawer. Its width is pinned by the panel builder; the
    // scroll area below must absorb content growth instead of feeding it back
    // into the drawer geometry. `show_collapsible` slides the drawer in and
    // out toward the left edge with egui's built-in easing.
    let mut viewport_ui = root_viewport_ui(ctx);
    let drawer_shown =
        control_drawer_panel().show_collapsible(&mut viewport_ui, &mut drawer_open, |ui| {
            drawer_header(ui, &mut hide_requested);
            drawer_navigation(ui, &view_model, &mut ui_state);

            // Screen content in a vertical scroll area.
            bevy_egui::egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match view_model.screen {
                    Screen::Setup => render_setup_screen(
                        ui,
                        &view_model,
                        &mut ui_state,
                        face_backend.requested,
                        &mut file_dialog,
                        error_presenter.current(),
                    ),
                    Screen::Live => render_live_screen(
                        ui,
                        &view_model,
                        &mut ui_state,
                        &preview,
                        &landmarks,
                        *avatar_motion_mirror,
                        preview_texture,
                    ),
                    Screen::Diagnostics => render_diagnostics_screen(ui, &view_model, &diagnostics),
                });
        });

    ui_state.control_drawer.open = drawer_open && !hide_requested;

    if drawer_shown.is_none() {
        // Drawer fully closed: leave a slim pull handle docked to the left
        // edge instead of a floating button that hides the avatar.
        bevy_egui::egui::Area::new(bevy_egui::egui::Id::new("control_drawer_handle"))
            .anchor(
                bevy_egui::egui::Align2::LEFT_CENTER,
                bevy_egui::egui::vec2(0.0, 0.0),
            )
            .movable(false)
            .show(ctx, |ui| {
                let open = bevy_egui::egui::Button::new("»")
                    .min_size(bevy_egui::egui::vec2(24.0, 72.0))
                    .corner_radius(bevy_egui::egui::CornerRadius {
                        ne: 10,
                        se: 10,
                        ..bevy_egui::egui::CornerRadius::ZERO
                    })
                    .fill(drawer_fill());
                if ui.add(open).on_hover_text("Show Controls (F1)").clicked() {
                    ui_state.control_drawer.open = true;
                }
            });
    }

    // Handle drag-and-drop for VRM files.
    super::file_dialog::handle_dropped_files(ctx, &mut ui_state);

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
        assert_eq!(state.pending_actions.len(), 2);

        let actions = state.take_actions();
        assert_eq!(actions.len(), 2);
        assert!(state.pending_actions.is_empty());
    }

    #[test]
    fn ui_state_default_is_empty() {
        let state = UiState::default();
        assert!(state.pending_actions.is_empty());
    }

    #[test]
    fn japanese_font_is_primary_for_both_ui_families() {
        let definitions = japanese_font_definitions();
        let font = definitions
            .font_data
            .get(JAPANESE_FONT_NAME)
            .expect("the bundled Japanese font must be registered");

        assert_eq!(font.font.as_ref(), JAPANESE_FONT_BYTES);
        for family in [
            bevy_egui::egui::FontFamily::Proportional,
            bevy_egui::egui::FontFamily::Monospace,
        ] {
            assert_eq!(
                definitions
                    .families
                    .get(&family)
                    .and_then(|fonts| fonts.first())
                    .map(String::as_str),
                Some(JAPANESE_FONT_NAME)
            );
        }
    }

    #[test]
    fn ui_state_emit_deduplicates_navigation() {
        let mut state = UiState::default();
        state.emit(UiAction::SwitchScreen(Screen::Live));
        state.emit(UiAction::SwitchScreen(Screen::Live)); // duplicate
        assert_eq!(state.pending_actions.len(), 1);

        state.emit(UiAction::SwitchScreen(Screen::Setup)); // different
        assert_eq!(state.pending_actions.len(), 2);
    }

    #[test]
    fn ui_state_emit_deduplicates_toggle() {
        let mut state = UiState::default();
        state.emit(UiAction::ToggleMirror);
        state.emit(UiAction::ToggleMirror); // duplicate
        assert_eq!(state.pending_actions.len(), 1);

        state.emit(UiAction::ToggleAvatarMotionMirror);
        state.emit(UiAction::ToggleAvatarMotionMirror); // duplicate
        assert_eq!(state.pending_actions.len(), 2);
    }

    #[test]
    fn ui_state_take_allows_same_action_next_batch() {
        let mut state = UiState::default();
        state.emit(UiAction::SwitchScreen(Screen::Live));
        let _ = state.take_actions();

        // Same action in next batch should work.
        state.emit(UiAction::SwitchScreen(Screen::Live));
        assert_eq!(state.pending_actions.len(), 1);
    }

    #[test]
    fn ui_state_emit_does_not_deduplicate_non_deduplicatable() {
        let mut state = UiState::default();
        state.emit(UiAction::Start);
        state.emit(UiAction::Start); // not deduplicated
        assert_eq!(state.pending_actions.len(), 2);
    }

    #[test]
    fn control_drawer_is_open_by_default_and_toggles() {
        let mut state = ControlDrawerState::default();
        assert!(state.open);

        state.toggle();
        assert!(!state.open);

        state.toggle();
        assert!(state.open);
    }

    #[test]
    fn sync_error_presenter_exposes_import_error_to_setup_and_diagnostics() {
        let mut app = App::new();
        app.init_resource::<Orchestrator>()
            .init_resource::<ErrorPresenter>()
            .init_resource::<DiagnosticsSnapshot>()
            .add_systems(Update, sync_error_presenter);

        app.world_mut()
            .resource_mut::<Orchestrator>()
            .set_last_error(Some(crate::orchestrator::OrchestratorError::ImportFailed(
                "invalid VRM".to_string(),
            )));
        app.update();

        let presenter = app.world().resource::<ErrorPresenter>();
        let presentation = presenter
            .current()
            .expect("import failures must produce a UI presentation");
        assert_eq!(presentation.code, "IMPORT_FAILED");
        assert!(presentation.user_message.contains("invalid VRM"));
        assert_eq!(
            app.world()
                .resource::<DiagnosticsSnapshot>()
                .last_error
                .as_deref(),
            Some("Import failed: invalid VRM")
        );
    }

    #[test]
    fn control_drawer_pins_geometry_against_stale_oversized_state() {
        let ctx = bevy_egui::egui::Context::default();
        let screen_rect = bevy_egui::egui::Rect::from_min_size(
            bevy_egui::egui::Pos2::ZERO,
            bevy_egui::egui::vec2(1920.0, 1080.0),
        );
        let raw_input = bevy_egui::egui::RawInput {
            screen_rect: Some(screen_rect),
            ..Default::default()
        };
        let drawer_id = bevy_egui::egui::Id::new("control_drawer");

        // Simulate a stale persisted state left by an oversized in-session
        // resize of a previous build.
        ctx.data_mut(|data| {
            data.insert_persisted(
                drawer_id,
                bevy_egui::egui::PanelState {
                    outer_rect: bevy_egui::egui::Rect::from_min_size(
                        bevy_egui::egui::Pos2::ZERO,
                        bevy_egui::egui::vec2(1500.0, 700.0),
                    ),
                },
            );
        });

        // The current builder must clamp that stale state to the exact drawer
        // geometry and keep oversized content inside the scrollable viewport.
        let _ = ctx.run_ui(raw_input.clone(), |ui| {
            control_drawer_panel().show(ui, |ui| {
                bevy_egui::egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.allocate_space(bevy_egui::egui::vec2(1500.0, 700.0));
                    });
            });
        });

        let pinned_rect = bevy_egui::egui::PanelState::load(&ctx, drawer_id)
            .expect("pinned drawer should have persisted state")
            .outer_rect;
        assert_eq!(pinned_rect.width(), DRAWER_WIDTH);
        assert_eq!(pinned_rect.height(), 1080.0);

        // Closing the drawer (a frame where it is not shown) and reopening it
        // must not let the next appearance depend on that gap either.
        let _ = ctx.run_ui(raw_input.clone(), |_ui| {});
        let _ = ctx.run_ui(raw_input, |ui| {
            control_drawer_panel().show(ui, |_ui| {});
        });

        let reopened_rect = bevy_egui::egui::PanelState::load(&ctx, drawer_id)
            .expect("reopened drawer should have persisted state")
            .outer_rect;
        assert_eq!(reopened_rect.size(), pinned_rect.size());
    }
}
