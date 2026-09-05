//! UI shell — the bevy_egui integration layer.
//!
//! Provides the [`UiShellPlugin`] which sets up egui and renders the
//! two main screens: Setup (Controls) and Diagnostics. The former Live tab
//! was folded into these two so controls live in Setup and health in
//! Diagnostics.

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
use crate::ui_model::{Pane, UiViewModel};
use vtuber_avatar::{
    AvatarMotionMirror, AvatarViewportCamera, CameraInputSet, CameraPointerInputGate,
    apply_arm_pose_profile_changes,
};

use super::diagnostics::render_diagnostics_pane;
use super::panes::{
    render_avatar_pane, render_calibration_pane, render_camera_pane, render_ndi_pane,
    render_preview_pane,
};
use super::widgets::{self, app_lifecycle_text};

/// macOS sidebar proportions; the drawer holds navigation, the selected
/// pane's grouped content, and the session footer, all in one column.
const DRAWER_WIDTH: f32 = 300.0;
/// Sidebar row height from the macOS source-list metric.
const SIDEBAR_ROW_HEIGHT: f32 = 28.0;
/// macOS systemBlue (dark mode) for the sidebar selection highlight.
const SIDEBAR_ACCENT: bevy_egui::egui::Color32 = bevy_egui::egui::Color32::from_rgb(10, 132, 255);
/// Secondary label gray for unselected sidebar icons.
const SIDEBAR_ICON_GRAY: bevy_egui::egui::Color32 = bevy_egui::egui::Color32::from_rgb(152, 155, 163);

const JAPANESE_FONT_NAME: &str = "LINESeedJP_A_Rg";
static JAPANESE_FONT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/fonts/LINESeedJP_A_TTF_Rg.ttf"
));

/// Fill of the drawer and its pull handle: near-black with a slight
/// translucency so the avatar scene stays perceptible behind the sidebar.
fn drawer_fill() -> bevy_egui::egui::Color32 {
    bevy_egui::egui::Color32::from_rgba_unmultiplied(17, 19, 26, 242)
}

/// The controls drawer: a fixed-width sidebar docked to the left edge of the
/// window, spanning the full height like a macOS source-list sidebar.
fn control_drawer_panel() -> bevy_egui::egui::Panel {
    // `exact_size` pins the drawer geometry: egui persists panel sizes per Id,
    // so a stale in-session state can never widen or narrow the drawer.
    // Content growth is absorbed by the scroll area inside instead.
    bevy_egui::egui::Panel::left(bevy_egui::egui::Id::new("control_drawer"))
        .exact_size(DRAWER_WIDTH)
        .resizable(false)
        .show_separator_line(true)
        .frame(drawer_frame())
}

/// macOS sidebar style: square, edge-to-edge translucent fill separated from
/// the content by the panel's own hairline — no floating rounded edge.
fn drawer_frame() -> bevy_egui::egui::Frame {
    bevy_egui::egui::Frame::new()
        .fill(drawer_fill())
        .inner_margin(bevy_egui::egui::Margin::same(8))
}

/// The single error banner, shown once at the top of the drawer content.
fn error_banner(
    ui: &mut bevy_egui::egui::Ui,
    presentation: &crate::error_presenter::ErrorPresentation,
    ui_state: &mut UiState,
) {
    bevy_egui::egui::Frame::new()
        .fill(bevy_egui::egui::Color32::from_rgba_unmultiplied(127, 29, 29, 90))
        .corner_radius(bevy_egui::egui::CornerRadius::same(8))
        .inner_margin(bevy_egui::egui::Margin::same(12))
        .stroke(bevy_egui::egui::Stroke::new(
            1.0,
            bevy_egui::egui::Color32::from_rgba_unmultiplied(248, 113, 113, 90),
        ))
        .show(ui, |ui| {
            super::error::render_error_panel(ui, presentation, ui_state);
        });
    ui.add_space(8.0);
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

/// Sidebar header: macOS toolbar row — one semibold 13pt app title on the
/// left and a small sidebar-toggle control on the right. The F1 shortcut
/// lives in the tooltip only, keeping the bar to a single quiet row.
fn drawer_header(ui: &mut bevy_egui::egui::Ui, hide_requested: &mut bool) {
    ui.horizontal(|ui| {
        ui.label(bevy_egui::egui::RichText::new("RusTuber").size(13.0).strong());
        ui.with_layout(
            bevy_egui::egui::Layout::right_to_left(bevy_egui::egui::Align::Center),
            |ui| {
                let close = bevy_egui::egui::Button::new(
                    bevy_egui::egui::RichText::new("‹").size(12.0),
                )
                .min_size(bevy_egui::egui::vec2(24.0, 22.0))
                .corner_radius(bevy_egui::egui::CornerRadius::same(5));
                if ui.add(close).on_hover_text("Hide Controls (F1)").clicked() {
                    *hide_requested = true;
                }
            },
        );
    });
    ui.add_space(8.0);
}

/// Setup categories shown in the sidebar, in workflow order.
const SETUP_ROWS: [(&str, &str, Pane); 3] = [
    ("◎", "Camera", Pane::Camera),
    ("◍", "Avatar", Pane::Avatar),
    ("◉", "Calibration", Pane::Calibration),
];

/// Output categories shown in the sidebar.
const OUTPUT_ROWS: [(&str, &str, Pane); 2] = [
    ("▣", "Preview & Display", Pane::Preview),
    ("⬡", "NDI Output", Pane::NdiOutput),
];

/// Source-list navigation: pane categories grouped under section captions,
/// styled like a macOS sidebar — 28pt rows, 13pt regular labels, and a
/// full-width rounded highlight in systemBlue for the selected item.
fn drawer_navigation(
    ui: &mut bevy_egui::egui::Ui,
    view_model: &UiViewModel,
    ui_state: &mut UiState,
) {
    widgets::section_caption(ui, "Setup");
    for (icon, title, pane) in SETUP_ROWS {
        let selected = view_model.pane == pane;
        if sidebar_row(ui, selected, icon, title).clicked() {
            ui_state.emit(UiAction::SwitchPane(pane));
        }
        ui.add_space(2.0);
    }
    ui.add_space(6.0);
    widgets::section_caption(ui, "Output");
    for (icon, title, pane) in OUTPUT_ROWS {
        let selected = view_model.pane == pane;
        if sidebar_row(ui, selected, icon, title).clicked() {
            ui_state.emit(UiAction::SwitchPane(pane));
        }
        ui.add_space(2.0);
    }
    ui.add_space(6.0);
    widgets::section_caption(ui, "System");
    if sidebar_row(ui, view_model.pane == Pane::Diagnostics, "▦", "Diagnostics").clicked() {
        ui_state.emit(UiAction::SwitchPane(Pane::Diagnostics));
    }
}

/// Selected pane content: the macOS System Settings grouped list for the
/// category picked in the sidebar, rendered inside the drawer so nothing
/// ever covers the avatar scene.
#[allow(clippy::too_many_arguments)]
fn drawer_pane_content(
    ui: &mut bevy_egui::egui::Ui,
    view_model: &UiViewModel,
    ui_state: &mut UiState,
    diagnostics: &crate::diagnostics::DiagnosticsSnapshot,
    preview: &PreviewState,
    landmarks: &PreviewLandmarkState,
    avatar_motion_mirror: AvatarMotionMirror,
    preview_texture: Option<bevy_egui::egui::TextureId>,
    file_dialog: &mut super::file_dialog::FileDialogState,
    error_presenter: &ErrorPresenter,
) {
    if let Some(presentation) = error_presenter.current() {
        error_banner(ui, presentation, ui_state);
    }
    match view_model.pane {
        Pane::Camera => render_camera_pane(ui, view_model, ui_state),
        Pane::Avatar => render_avatar_pane(ui, view_model, ui_state, file_dialog),
        Pane::Calibration => render_calibration_pane(ui, view_model, ui_state),
        Pane::Preview => render_preview_pane(
            ui,
            ui_state,
            preview,
            landmarks,
            avatar_motion_mirror,
            preview_texture,
        ),
        Pane::NdiOutput => render_ndi_pane(ui, view_model, ui_state),
        Pane::Diagnostics => render_diagnostics_pane(ui, view_model, diagnostics),
    }
}

/// One macOS sidebar row: 28pt tall, icon plus 13pt label, full-width
/// selection highlight.
fn sidebar_row(
    ui: &mut bevy_egui::egui::Ui,
    selected: bool,
    icon: &str,
    label: &str,
) -> bevy_egui::egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        bevy_egui::egui::vec2(ui.available_width(), SIDEBAR_ROW_HEIGHT),
        bevy_egui::egui::Sense::click(),
    );

    let highlight = if selected {
        Some(SIDEBAR_ACCENT)
    } else if response.hovered() {
        Some(bevy_egui::egui::Color32::from_rgba_unmultiplied(
            255, 255, 255, 16,
        ))
    } else {
        None
    };
    if let Some(fill) = highlight {
        ui.painter()
            .rect_filled(rect, bevy_egui::egui::CornerRadius::same(5), fill);
    }

    let text_color = if selected {
        bevy_egui::egui::Color32::WHITE
    } else {
        ui.visuals().text_color()
    };
    let icon_color = if selected { text_color } else { SIDEBAR_ICON_GRAY };
    let font = bevy_egui::egui::FontId::proportional(13.0);
    let icon_galley = ui
        .painter()
        .layout_no_wrap(icon.to_owned(), font.clone(), icon_color);
    let label_galley = ui
        .painter()
        .layout_no_wrap(label.to_owned(), font, text_color);

    let mid_y = rect.center().y;
    let icon_pos = bevy_egui::egui::pos2(rect.left() + 8.0, mid_y - icon_galley.size().y / 2.0);
    let label_pos = bevy_egui::egui::pos2(
        icon_pos.x + icon_galley.size().x + 6.0,
        mid_y - label_galley.size().y / 2.0,
    );
    ui.painter().galley(icon_pos, icon_galley, icon_color);
    ui.painter().galley(label_pos, label_galley, text_color);

    response
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
            .init_resource::<UiSurfaceHover>()
            .init_resource::<DiagnosticsSnapshot>()
            .init_resource::<MetricsExportState>()
            .init_resource::<ErrorPresenter>()
            .init_resource::<super::file_dialog::FileDialogState>()
            .init_resource::<CaptureRuntime>()
            .init_resource::<LatestVideoFrame>();

        let frame_slot = app.world().resource::<CaptureRuntime>().frame_slot();
        let project_root = app
            .world()
            .get_resource::<InferenceProjectRoot>()
            .map(|root| root.0.clone())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        app.insert_resource(InferenceRuntime::new(frame_slot, project_root))
            .init_resource::<TrackingRuntime>()
            .add_systems(Startup, restore_arm_pose_settings_system)
            // Action processing, avatar pose re-resolution, and lifecycle sync
            // are chained so a settings action reaches the compositor in the
            // same Update frame.
            .add_systems(
                Update,
                (
                    process_ui_actions_system,
                    apply_arm_pose_profile_changes,
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

/// Whether the egui pointer currently rests over a surface this shell drew.
///
/// bevy_egui wraps UI systems in [`bevy_egui::egui::Context::run_ui`], whose
/// background-layer pointer probe only knows its internal root Ui. Panels
/// drawn in this plugin's own root Ui are invisible to it, so
/// [`bevy_egui::input::EguiWantsInput`] alone would leave the main viewport
/// input ungated while the pointer hovers the drawer.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UiSurfaceHover {
    over_ui: bool,
}

impl UiSurfaceHover {
    /// Records whether the pointer rests over a UI surface this frame.
    pub fn set(&mut self, over_ui: bool) {
        self.over_ui = over_ui;
    }

    /// Whether the pointer rests over a UI surface.
    #[must_use]
    pub const fn over_ui(self) -> bool {
        self.over_ui
    }
}

/// Whether the egui pointer position lands inside any UI surface.
fn pointer_over_ui(pointer: Option<bevy_egui::egui::Pos2>, surfaces: &[bevy_egui::egui::Rect]) -> bool {
    pointer.is_some_and(|pointer| surfaces.iter().any(|rect| rect.contains(pointer)))
}

/// Bridges egui pointer ownership into the avatar camera domain without
/// letting UI code touch a camera transform or projection.
///
/// [`bevy_egui::input::EguiWantsInput`] still covers active drags and open
/// popups, but its hover probe misses this shell's panels (see
/// [`UiSurfaceHover`]), so the surfaces drawn this frame supply the rest.
fn sync_camera_pointer_input_gate(
    egui_wants_input: Res<bevy_egui::input::EguiWantsInput>,
    ui_surface_hover: Res<UiSurfaceHover>,
    mut gate: ResMut<CameraPointerInputGate>,
) {
    gate.set_egui_owns_pointer(
        egui_wants_input.wants_any_pointer_input() || ui_surface_hover.over_ui(),
    );
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
        UiAction::SwitchPane(_)
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
    mut file_dialog: ResMut<super::file_dialog::FileDialogState>,
    mut ui_surface_hover: ResMut<UiSurfaceHover>,
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

            // Footer session controls are pinned with a bottom_up layout
            // (widgets added first land at the bottom); the selected pane's
            // grouped content fills the space above them in a scroll area.
            // Everything stays inside the drawer so the avatar scene is
            // never covered.
            ui.with_layout(
                bevy_egui::egui::Layout::bottom_up(bevy_egui::egui::Align::LEFT),
                |ui| {
                    ui.horizontal(|ui| {
                        if widgets::filled_button(ui, "Start", view_model.can_start()).clicked() {
                            ui_state.emit(UiAction::Start);
                        }
                        if widgets::plain_button(ui, "Stop", view_model.can_stop()).clicked() {
                            ui_state.emit(UiAction::Stop);
                        }
                    });
                    ui.add_space(6.0);
                    let (color, label) = app_lifecycle_text(view_model.lifecycle);
                    widgets::status_text(ui, color, label);
                    ui.add_space(6.0);
                    ui.separator();

                    // Pane content in a vertical scroll area. Width is
                    // pinned to the viewport so wide content never expands
                    // the drawer beyond DRAWER_WIDTH.
                    bevy_egui::egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .id_salt("control_drawer_scroll")
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            drawer_pane_content(
                                ui,
                                &view_model,
                                &mut ui_state,
                                &diagnostics,
                                &preview,
                                &landmarks,
                                *avatar_motion_mirror,
                                preview_texture,
                                &mut file_dialog,
                                &error_presenter,
                            );
                        });
                },
            );
        });

    ui_state.control_drawer.open = drawer_open && !hide_requested;

    let mut handle_rect = None;
    if drawer_shown.is_none() {
        // Drawer fully closed: leave a slim pull handle docked to the left
        // edge instead of a floating button that hides the avatar.
        let handle = bevy_egui::egui::Area::new(bevy_egui::egui::Id::new("control_drawer_handle"))
            .anchor(
                bevy_egui::egui::Align2::LEFT_CENTER,
                bevy_egui::egui::vec2(0.0, 0.0),
            )
            .movable(false)
            .show(ctx, |ui| {
                let open = bevy_egui::egui::Button::new("»")
                    .min_size(bevy_egui::egui::vec2(24.0, 72.0))
                    .corner_radius(bevy_egui::egui::CornerRadius {
                        ne: 6,
                        se: 6,
                        ..bevy_egui::egui::CornerRadius::ZERO
                    })
                    .fill(drawer_fill());
                ui.add(open).on_hover_text("Show Controls (F1)")
            });
        handle_rect = Some(handle.response.rect);
        if handle.inner.clicked() {
            ui_state.control_drawer.open = true;
        }
    }

    // egui's own pointer-over-area probe misses panels drawn in this plugin's
    // own root Ui (see `UiSurfaceHover`), so report hover from the surfaces
    // the shell just drew.
    let mut surfaces = Vec::new();
    if let Some(drawer) = drawer_shown.as_ref() {
        surfaces.push(drawer.response.rect);
    }
    if let Some(handle) = handle_rect {
        surfaces.push(handle);
    }
    ui_surface_hover.set(pointer_over_ui(
        ctx.input(|input| input.pointer.interact_pos()),
        &surfaces,
    ));

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
    fn pointer_over_ui_covers_every_drawn_surface_but_not_the_scene() {
        let drawer = bevy_egui::egui::Rect::from_min_size(
            bevy_egui::egui::Pos2::ZERO,
            bevy_egui::egui::vec2(DRAWER_WIDTH, 1080.0),
        );
        let handle = bevy_egui::egui::Rect::from_min_size(
            bevy_egui::egui::Pos2::ZERO,
            bevy_egui::egui::vec2(24.0, 72.0),
        );

        let surfaces = [drawer, handle];
        assert!(pointer_over_ui(
            Some(bevy_egui::egui::pos2(150.0, 300.0)),
            &surfaces
        ));
        assert!(!pointer_over_ui(
            Some(bevy_egui::egui::pos2(900.0, 300.0)),
            &surfaces
        ));
        assert!(pointer_over_ui(Some(bevy_egui::egui::pos2(12.0, 40.0)), &surfaces));
        assert!(!pointer_over_ui(
            Some(bevy_egui::egui::pos2(12.0, 300.0)),
            &[]
        ));
        assert!(!pointer_over_ui(None, &surfaces));
    }

    #[test]
    fn open_drawer_surface_rect_tracks_pointer_hover() {
        let ctx = bevy_egui::egui::Context::default();
        let raw_input = bevy_egui::egui::RawInput {
            screen_rect: Some(bevy_egui::egui::Rect::from_min_size(
                bevy_egui::egui::Pos2::ZERO,
                bevy_egui::egui::vec2(1920.0, 1080.0),
            )),
            ..Default::default()
        };

        let surface = |ctx: &bevy_egui::egui::Context, raw: bevy_egui::egui::RawInput| {
            let mut drawer_rect = None;
            let _ = ctx.run_ui(raw, |_top_ui| {
                // Mirror `ui_render_system`: the drawer lives in the shell's
                // own root Ui, not in run_ui's internal root.
                let mut viewport_ui = root_viewport_ui(ctx);
                let mut open = true;
                let shown =
                    control_drawer_panel().show_collapsible(&mut viewport_ui, &mut open, |ui| {
                        ui.heading("Controls");
                    });
                drawer_rect = shown.map(|response| response.response.rect);
            });
            drawer_rect.expect("open drawer is shown")
        };

        // Warm-up pass so egui animation state exists, then measure.
        let _ = surface(&ctx, raw_input.clone());
        let drawer_rect = surface(&ctx, raw_input);

        assert_eq!(drawer_rect.width(), DRAWER_WIDTH);
        assert!(
            drawer_rect.contains(bevy_egui::egui::pos2(150.0, 300.0)),
            "pointer over the drawer must count as over UI"
        );
        assert!(
            !drawer_rect.contains(bevy_egui::egui::pos2(900.0, 300.0)),
            "pointer over the 3D scene must not count as over UI"
        );
    }

    #[test]
    fn drawer_footer_is_pinned_above_the_scene_and_pane_content_scrolls_above_it() {
        let ctx = bevy_egui::egui::Context::default();
        let raw_input = bevy_egui::egui::RawInput {
            screen_rect: Some(bevy_egui::egui::Rect::from_min_size(
                bevy_egui::egui::Pos2::ZERO,
                bevy_egui::egui::vec2(1920.0, 1080.0),
            )),
            ..Default::default()
        };

        let surface = |ctx: &bevy_egui::egui::Context, raw: bevy_egui::egui::RawInput| {
            let mut result = None;
            let _ = ctx.run_ui(raw, |_top_ui| {
                let mut viewport_ui = root_viewport_ui(ctx);
                control_drawer_panel().show(&mut viewport_ui, |ui| {
                    // Mirror the drawer body: bottom_up pins the footer,
                    // the scroll area fills the space above it.
                    ui.with_layout(
                        bevy_egui::egui::Layout::bottom_up(bevy_egui::egui::Align::LEFT),
                        |ui| {
                            ui.horizontal(|ui| {
                                let _ = ui.button("Start");
                                let _ = ui.button("Stop");
                            });
                            ui.add_space(6.0);
                            let status = ui.label("Idle");                            ui.separator();
                            let footer_top = status.rect.top();
                            let scroll = bevy_egui::egui::ScrollArea::vertical()
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    ui.allocate_space(bevy_egui::egui::vec2(100.0, 4000.0));
                                });
                            result = Some((footer_top, scroll.inner_rect, scroll.content_size));
                        },
                    );
                });
            });
            result.expect("drawer body laid out")
        };

        // Warm-up pass so egui animation state exists, then measure.
        let _ = surface(&ctx, raw_input.clone());
        let (footer_top, content_rect, content_size) = surface(&ctx, raw_input);

        assert!(
            content_rect.bottom() <= footer_top,
            "pane content must sit above the pinned footer, not overlap it"
        );
        assert!(
            content_size.y > content_rect.height(),
            "pane content taller than the space above the footer must scroll"
        );
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
        state.emit(UiAction::SwitchPane(Pane::Diagnostics));
        state.emit(UiAction::SwitchPane(Pane::Diagnostics)); // duplicate
        assert_eq!(state.pending_actions.len(), 1);

        state.emit(UiAction::SwitchPane(Pane::Camera)); // different
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
        state.emit(UiAction::SwitchPane(Pane::Diagnostics));
        let _ = state.take_actions();

        // Same action in next batch should work.
        state.emit(UiAction::SwitchPane(Pane::Diagnostics));
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
