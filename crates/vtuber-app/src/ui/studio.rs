//! Apple-style desktop workspace. The sidebar contains destinations only.
//! Camera controls, session commands, and state never live in that sidebar.

use super::avatar_preview::{AvatarPreviewTexture, paint_avatar_preview, paint_avatar_preview_at};
use super::shell::UiState;
use crate::actions::{RichLookChange, UiAction};
use crate::diagnostics::DiagnosticsSnapshot;
use crate::error_presenter::ErrorPresentation;
use crate::licenses::LicenseGroup;
use crate::preview::PreviewState;
use crate::preview_landmarks::PreviewLandmarkState;
use crate::settings::UiLanguage;
use crate::ui_model::{AvatarLifecycleState, NdiOutputUiState, Pane, TrackingState, UiViewModel};
use bevy_egui::egui::{self, Color32, CornerRadius, Frame, Id, RichText, TextureId, Ui, vec2};
use vtuber_avatar::{ArmPoseProfileOverride, AvatarMotionMirror};

mod camera;
mod expression_keys;
mod import_review;

use camera::{camera_select_page, camera_status_page};
pub(crate) use expression_keys::expression_key_input;
use expression_keys::expression_keys_page;
use import_review::render_avatar_import_review;

/// Sidebar destinations in display order. There is no hierarchy: every
/// destination shows all of its sections in the right pane at once.
const DESTINATIONS: [Pane; 6] = [
    Pane::VrmCamera,
    Pane::PoseCamera,
    Pane::ExpressionKeys,
    Pane::NdiOutput,
    Pane::Diagnostics,
    Pane::OssLicenses,
];

/// Every sidebar destination in display order.
fn destinations() -> impl Iterator<Item = Pane> {
    DESTINATIONS.into_iter()
}

fn page_title(pane: Pane, lang: UiLanguage) -> &'static str {
    match pane {
        Pane::VrmCamera => lang.pick(
            "VRM・カメラ選択",
            "VRM & camera",
            "VRM与摄像头",
            "VRM 및 카메라",
        ),
        Pane::PoseCamera => lang.pick(
            "姿勢・カメラ調整",
            "Pose & camera",
            "姿势与摄像头调整",
            "자세·카메라 조정",
        ),
        Pane::ExpressionKeys => lang.pick("表情設定", "Expressions", "表情设置", "표정 설정"),
        Pane::NdiOutput => lang.pick("NDI出力", "NDI output", "NDI输出", "NDI 출력"),
        Pane::Diagnostics => lang.pick("診断", "Diagnostics", "诊断", "진단"),
        Pane::OssLicenses => lang.pick(
            "OSSライセンス",
            "OSS licenses",
            "开源许可证",
            "OSS 라이선스",
        ),
    }
}

/// Language buttons shown in the left pane and, without one, in the toolbar.
const LANGUAGES: [(UiLanguage, &str); 4] = [
    (UiLanguage::Ja, "日本語"),
    (UiLanguage::En, "ENGLISH"),
    (UiLanguage::Zh, "中文"),
    (UiLanguage::Ko, "한국어"),
];

fn avatar_label(state: AvatarLifecycleState, lang: UiLanguage) -> &'static str {
    match state {
        AvatarLifecycleState::None => {
            lang.pick("未読み込み", "Not loaded", "未加载", "불러오지 않음")
        }
        AvatarLifecycleState::Loading => {
            lang.pick("読み込み中…", "Loading…", "正在加载…", "불러오는 중…")
        }
        AvatarLifecycleState::Binding => {
            lang.pick("準備中…", "Preparing…", "正在准备…", "준비 중…")
        }
        AvatarLifecycleState::Ready => lang.pick("準備完了", "Ready", "就绪", "준비 완료"),
        AvatarLifecycleState::Unloading => {
            lang.pick("解除中…", "Unloading…", "正在卸载…", "해제 중…")
        }
        AvatarLifecycleState::Failed => {
            lang.pick("読み込み失敗", "Load failed", "加载失败", "불러오기 실패")
        }
    }
}

fn panel_frame(gray: u8) -> Frame {
    Frame::new()
        .fill(Color32::from_gray(gray))
        .inner_margin(egui::Margin::same(16))
}

fn section(ui: &mut Ui, title: &str, contents: impl FnOnce(&mut Ui)) {
    setup_section(ui, title, false, false, contents);
}

fn setup_section(
    ui: &mut Ui,
    title: &str,
    unfinished: bool,
    animate: bool,
    contents: impl FnOnce(&mut Ui),
) {
    ui.add_space(12.0);
    let mut frame = Frame::new()
        .fill(Color32::WHITE)
        .stroke(egui::Stroke::new(1.0, Color32::from_gray(220)))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(egui::Margin::same(16));
    let attention = unfinished.then(|| attention_style(ui, animate));
    if let Some((_, shadow)) = attention {
        frame = frame.shadow(shadow);
    }
    let response = frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new(title).size(17.0).strong());
        ui.add_space(10.0);
        contents(ui);
    });
    if let Some((stroke, _)) = attention {
        ui.painter().rect_stroke(
            response.response.rect,
            CornerRadius::same(10),
            stroke,
            egui::StrokeKind::Inside,
        );
    }
}

fn attention_style(ui: &Ui, animate: bool) -> (egui::Stroke, egui::epaint::Shadow) {
    // A smooth fade directs attention without moving controls or blinking.
    // A three-point outline reaches full accent color at each peak, so the
    // change remains visible on both the white cards and dark avatar preview.
    // Timing and opacity are application choices, not Apple-prescribed values.
    let pulse = if animate {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(33));
        ui.input(|input| (0.5 - 0.5 * (input.time * std::f64::consts::TAU / 2.4).cos()) as f32)
    } else {
        0.5
    };
    let accent = ui.visuals().hyperlink_color;
    (
        egui::Stroke::new(3.0, accent.gamma_multiply(0.18 + 0.82 * pulse)),
        egui::epaint::Shadow {
            offset: [0, 0],
            blur: 16,
            spread: 2,
            color: accent.gamma_multiply(0.04 + 0.34 * pulse),
        },
    )
}

fn primary_button(ui: &mut Ui, label: &str, enabled: bool) -> egui::Response {
    ui.add_enabled(
        enabled,
        egui::Button::new(RichText::new(label).color(Color32::WHITE))
            .fill(ui.visuals().hyperlink_color)
            .min_size(vec2(0.0, 30.0)),
    )
}

/// Width of the left pane that lists the destinations.
const SIDEBAR_PANEL_WIDTH: f32 = 200.0;
/// The monitor panel frame adds 16px of margin on each side of the preview.
const MONITOR_PANEL_PADDING: f32 = 32.0;
/// Preview width the monitor panel aims for. Wide enough that the avatar
/// preview reads as a real preview instead of a thumbnail; a narrower window
/// shrinks it rather than dropping back to the compact layout.
const MONITOR_TARGET_WIDTH: f32 = 660.0;
/// Narrowest preview worth showing. The compact layout shows exactly this.
const MIN_MONITOR_WIDTH: f32 = 200.0;
/// Narrowest the settings column gets before the monitor moves above it.
const MIN_SETTINGS_WIDTH: f32 = 400.0;
/// Narrowest viewport that still fits the left pane, a minimum-width monitor
/// panel, and the settings column side by side.
const SIDEBAR_LAYOUT_MIN_WIDTH: f32 =
    SIDEBAR_PANEL_WIDTH + MONITOR_PANEL_PADDING + MIN_MONITOR_WIDTH + MIN_SETTINGS_WIDTH;

/// Below this width the side-by-side layout would squeeze either the preview or
/// the settings page past its minimum, so the monitor moves above the settings
/// instead.
fn use_sidebar(width: f32) -> bool {
    width >= SIDEBAR_LAYOUT_MIN_WIDTH
}

/// Width of the right-hand monitor panel: the target preview width on a wide
/// window, and whatever the settings column can spare on a narrower one. The
/// preview card is capped just under it so the frame margin never spills out.
fn monitor_panel_width(viewport_width: f32) -> f32 {
    (viewport_width - SIDEBAR_PANEL_WIDTH - MIN_SETTINGS_WIDTH).clamp(
        MIN_MONITOR_WIDTH + MONITOR_PANEL_PADDING,
        MONITOR_TARGET_WIDTH + MONITOR_PANEL_PADDING,
    )
}

const FLOATING_CONTROL_MARGIN: f32 = 16.0;
const FLOATING_CONTROL_SIZE: f32 = 40.0;
/// egui-managed duration of the workspace <-> avatar-only transition. Long
/// enough that the workspace collapse into the settings icon stays readable.
const AVATAR_ONLY_TRANSITION_SECONDS: f32 = 0.36;
/// Corner radius shared by the avatar monitor card and the expanding preview.
const MONITOR_CARD_RADIUS: u8 = 8;
/// Background of the avatar-only view. `UiShellPlugin` installs the same color
/// as Bevy's window `ClearColor`, and the monitor card fills with it, so the
/// small preview and the fullscreen view show the avatar on one background and
/// their colors can be compared directly.
pub(crate) const STUDIO_BACKGROUND: Color32 = Color32::from_rgb(43, 44, 47);

/// Center of the floating settings control, and therefore the point the
/// workspace collapses into when the avatar-only view opens.
fn settings_icon_center(viewport: egui::Rect) -> egui::Pos2 {
    egui::pos2(
        viewport.right() - FLOATING_CONTROL_MARGIN - FLOATING_CONTROL_SIZE / 2.0,
        viewport.top() + FLOATING_CONTROL_MARGIN + FLOATING_CONTROL_SIZE / 2.0,
    )
}

fn paint_settings_glyph(painter: &egui::Painter, center: egui::Pos2, color: Color32) {
    let stroke = egui::Stroke::new(1.7, color);
    let half_track = 10.0;
    for (dy, knob_dx) in [(-6.0_f32, -3.0_f32), (0.0, 4.0), (6.0, -5.0)] {
        let y = center.y + dy;
        painter.line_segment(
            [
                egui::pos2(center.x - half_track, y),
                egui::pos2(center.x + half_track, y),
            ],
            stroke,
        );
        painter.circle_filled(egui::pos2(center.x + knob_dx, y), 2.7, color);
    }
}

// The floating settings control reopens the workspace from the top-right corner.
// During a transition it fades and scales with `progress`, appearing to grow
// out of the collapsing workspace it replaces.
fn floating_settings_control(
    ctx: &egui::Context,
    state: &mut UiState,
    lang: UiLanguage,
    progress: f32,
) -> egui::Response {
    let area = egui::Area::new(Id::new("floating_settings_control"))
        .anchor(
            egui::Align2::RIGHT_TOP,
            vec2(-FLOATING_CONTROL_MARGIN, FLOATING_CONTROL_MARGIN),
        )
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(
                vec2(FLOATING_CONTROL_SIZE, FLOATING_CONTROL_SIZE),
                egui::Sense::click(),
            );
            let hover = ui
                .ctx()
                .animate_bool_with_time(response.id, response.hovered(), 0.12);
            // macOS system accent blue, with the pressed and hover variants
            // macOS applies to a filled accent button.
            let accent = Color32::from_rgb(0, 122, 255);
            let fill = if response.is_pointer_button_down_on() {
                accent.gamma_multiply(0.80)
            } else {
                accent.gamma_multiply(1.0 + 0.18 * hover)
            };
            let border = Color32::from_white_alpha((50.0 + 50.0 * hover) as u8);
            let center = rect.center();
            let radius = FLOATING_CONTROL_SIZE / 2.0;
            let scale = 0.75 + 0.25 * progress;
            ui.scope(|ui| {
                ui.multiply_opacity(progress);
                ui.with_visual_transform(
                    egui::emath::TSTransform::new(center.to_vec2() * (1.0 - scale), scale),
                    |ui| {
                        let painter = ui.painter();
                        painter.add(
                            egui::epaint::Shadow {
                                offset: [0, 3],
                                blur: 10,
                                spread: 0,
                                color: Color32::from_black_alpha(80),
                            }
                            .as_shape(rect, radius),
                        );
                        painter.circle_filled(center, radius, fill);
                        painter.circle_stroke(center, radius - 0.5, egui::Stroke::new(1.0, border));
                        paint_settings_glyph(painter, center, Color32::from_white_alpha(235));
                    },
                );
            });
            response
        });
    let response = area
        .inner
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(format!(
            "{} (F1)",
            lang.pick("設定", "Settings", "设置", "설정")
        ));
    if response.clicked() {
        state.set_controls_open(true);
    }
    response
}

fn navigation(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    ui.add_space(12.0);
    for pane in destinations() {
        navigation_button(ui, vm, state, pane, lang);
    }
}

fn navigation_button(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    pane: Pane,
    lang: UiLanguage,
) {
    if ui
        .add_sized(
            [ui.available_width(), 30.0],
            egui::Button::selectable(vm.pane == pane, page_title(pane, lang)),
        )
        .clicked()
    {
        state.emit(UiAction::SwitchPane(pane));
    }
}

/// Full-width language rows for the left pane.
fn language_buttons(ui: &mut Ui, state: &mut UiState, lang: UiLanguage) {
    ui.add_space(8.0);
    ui.label(
        RichText::new(lang.pick("表示言語", "Display language", "显示语言", "표시 언어"))
            .small()
            .weak(),
    );
    ui.add_space(4.0);
    for (language, label) in LANGUAGES {
        if ui
            .add_sized(
                [ui.available_width(), 24.0],
                egui::Button::selectable(lang == language, label),
            )
            .clicked()
            && lang != language
        {
            state.emit(UiAction::SetLanguage(language));
        }
    }
    ui.add_space(8.0);
}

/// Inline language buttons for the toolbar when no left pane is shown.
fn compact_language_buttons(ui: &mut Ui, state: &mut UiState, lang: UiLanguage) {
    for (language, label) in LANGUAGES {
        if ui.selectable_label(lang == language, label).clicked() && lang != language {
            state.emit(UiAction::SetLanguage(language));
        }
    }
}

fn compact_navigation(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    egui::ComboBox::from_id_salt("studio_navigation")
        .selected_text(page_title(vm.pane, lang))
        .show_ui(ui, |ui| {
            for pane in destinations() {
                if ui
                    .selectable_label(vm.pane == pane, page_title(pane, lang))
                    .clicked()
                {
                    state.emit(UiAction::SwitchPane(pane));
                }
            }
        });
}

fn toolbar(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, sidebar: bool, lang: UiLanguage) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("RusTuberV").strong());
        if !sidebar {
            compact_navigation(ui, vm, state, lang);
            compact_language_buttons(ui, state, lang);
        }
    });
}

fn avatar_monitor(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    texture: Option<AvatarPreviewTexture>,
    max_width: f32,
    draw_preview: bool,
    lang: UiLanguage,
) -> Option<egui::Rect> {
    ui.label(
        RichText::new(lang.pick(
            "アバタープレビュー",
            "Avatar preview",
            "虚拟形象预览",
            "아바타 미리 보기",
        ))
        .strong(),
    );
    ui.add_space(8.0);
    let mut preview_rect = None;
    if vm.avatar.is_ready {
        if let Some(texture) = texture {
            let width = ui.available_width().min(max_width);
            let profile = texture.profile();
            let height = width * profile.height as f32 / profile.width as f32;
            let attention = (draw_preview && vm.camera.selected_index.is_some())
                .then(|| attention_style(ui, vm.avatar_import_review.review.is_none()));
            let mut frame = Frame::new()
                .fill(STUDIO_BACKGROUND)
                .corner_radius(CornerRadius::same(MONITOR_CARD_RADIUS));
            if let Some((_, shadow)) = attention {
                frame = frame.shadow(shadow);
            }
            let response = frame
                .show(ui, |ui| {
                    let size = vec2(width, height);
                    // The expanding transition card owns the texture while the
                    // workspace collapses; the monitor keeps its layout only,
                    // so the preview is never drawn twice.
                    if draw_preview {
                        paint_avatar_preview(
                            ui,
                            texture.image().clone(),
                            size,
                            f32::from(MONITOR_CARD_RADIUS),
                        )
                    } else {
                        ui.allocate_exact_size(size, egui::Sense::hover()).1
                    }
                })
                .inner;
            preview_rect = Some(response.rect);
            if let Some((stroke, _)) = attention {
                // Paint the border inside the image rect so the glow never
                // changes the preview's layout or transition origin.
                ui.painter().rect_stroke(
                    response.rect,
                    CornerRadius::same(MONITOR_CARD_RADIUS),
                    stroke,
                    egui::StrokeKind::Inside,
                );
            }
            if draw_preview {
                state.monitor_pointer_hovered = response.hovered()
                    || (response.contains_pointer() && response.is_pointer_button_down_on());
                if state.monitor_pointer_hovered || response.dragged() || response.drag_stopped() {
                    state.monitor_input_size = Some(response.rect.size());
                }
                let hint = lang.pick(
                    "クリックでアバターを全面表示",
                    "Click to show the avatar full screen",
                    "点击全屏显示虚拟形象",
                    "클릭하여 아바타 전체 화면 표시",
                );
                if response
                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                    .on_hover_text(hint)
                    .clicked()
                {
                    state.set_controls_open(false);
                }
                ui.add_space(8.0);
                ui.label(RichText::new(hint).small().weak());
            }
        } else if draw_preview {
            ui.spinner();
        }
    } else {
        ui.label(avatar_label(vm.avatar.lifecycle, lang));
    }
    ui.add_space(8.0);
    ui.label(
        RichText::new(lang.pick(
            "NDIと同じアバター専用描画です。設定UIとカメラ映像は含みません。",
            "The same avatar-only render used by NDI. Settings and camera pixels are excluded.",
            "与NDI共用的虚拟形象画面，不含设置界面或摄像头影像。",
            "NDI와 동일한 아바타 전용 화면입니다. 설정과 카메라 영상은 포함되지 않습니다.",
        ))
        .small()
        .weak(),
    );
    preview_rect
}

#[expect(
    clippy::too_many_arguments,
    reason = "the pane draws from one argument per widget input, so each widget reads exactly the state it is given"
)]
pub(crate) fn render_studio(
    ctx: &egui::Context,
    vm: &UiViewModel,
    state: &mut UiState,
    diagnostics: &DiagnosticsSnapshot,
    error: Option<&ErrorPresentation>,
    preview: &PreviewState,
    landmarks: &PreviewLandmarkState,
    avatar_mirror: AvatarMotionMirror,
    camera_texture: Option<TextureId>,
    avatar_texture: Option<AvatarPreviewTexture>,
    dialog_active: bool,
    lang: UiLanguage,
) -> bool {
    state.monitor_pointer_hovered = false;
    state.monitor_input_size = None;
    let viewport = ctx.viewport_rect();
    // `avatar_only` is the single linear animation clock: 0.0 is the
    // workspace, 1.0 is the fullscreen avatar. egui stores it per `Id`,
    // advances it while it requests repaints, and reverses from the current
    // value when F1 toggles. Each element below shapes its own curve so the
    // collapse into the settings icon is the primary, readable motion.
    let avatar_only = ctx.animate_bool_with_time(
        Id::new("studio_avatar_only_transition"),
        !state.controls_open,
        AVATAR_ONLY_TRANSITION_SECONDS,
    );
    state.avatar_only_progress = avatar_only;
    let transitioning = avatar_only > 0.0 && avatar_only < 1.0;
    // The workspace holds its size, then accelerates into the icon.
    let collapse = egui::emath::easing::quadratic_in(avatar_only);
    // The preview card trails the collapse, then finishes fullscreen.
    let expand =
        egui::emath::easing::cubic_out(egui::emath::remap_clamp(avatar_only, 0.3..=1.0, 0.0..=1.0));
    // The workspace stays opaque while collapsing and only fades at the end,
    // so the shrink itself is what the viewer tracks.
    let fade = egui::emath::easing::quadratic_in(egui::emath::remap_clamp(
        avatar_only,
        0.6..=1.0,
        0.0..=1.0,
    ));
    let floating_control =
        (avatar_only > 0.0).then(|| floating_settings_control(ctx, state, lang, collapse));
    if avatar_only >= 1.0 {
        let over_ui = ctx
            .input(|input| input.pointer.interact_pos())
            .is_some_and(|pos| floating_control.is_some_and(|control| control.rect.contains(pos)));
        return render_avatar_import_review(ctx, vm, state, lang, over_ui);
    }
    let sidebar = use_sidebar(viewport.width());
    let mut root = Ui::new(
        ctx.clone(),
        Id::new("studio_root"),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(viewport),
    );
    let previous_monitor_rect = state.monitor_image_rect;
    let mut monitor_rect = previous_monitor_rect;
    let mut draw_workspace = |ui: &mut Ui| {
        let mut workspace = ui.new_child(egui::UiBuilder::new().id(Id::new("studio_workspace")));
        let ui = &mut workspace;
        egui::Panel::top("studio_toolbar")
            .resizable(false)
            .frame(panel_frame(250))
            .show(ui, |ui| toolbar(ui, vm, state, sidebar, lang));
        if sidebar {
            egui::Panel::left("studio_sidebar")
                .exact_size(SIDEBAR_PANEL_WIDTH)
                .resizable(false)
                .frame(panel_frame(242))
                .show(ui, |ui| {
                    egui::Panel::bottom("studio_sidebar_language")
                        .resizable(false)
                        .frame(Frame::new())
                        .show(ui, |ui| language_buttons(ui, state, lang));
                    egui::ScrollArea::vertical().show(ui, |ui| navigation(ui, vm, state, lang));
                });
            let panel = monitor_panel_width(viewport.width());
            egui::Panel::right("studio_monitor")
                .exact_size(panel)
                .resizable(false)
                .frame(panel_frame(250))
                .show(ui, |ui| {
                    if let Some(rect) = avatar_monitor(
                        ui,
                        vm,
                        state,
                        avatar_texture.clone(),
                        panel - MONITOR_PANEL_PADDING,
                        !transitioning,
                        lang,
                    ) {
                        monitor_rect = Some(rect);
                    }
                });
        } else {
            // On smaller windows keep the monitor outside the settings scroll area.
            egui::Panel::top("studio_compact_monitor")
                .resizable(false)
                .frame(panel_frame(250))
                .show(ui, |ui| {
                    if let Some(rect) = avatar_monitor(
                        ui,
                        vm,
                        state,
                        avatar_texture.clone(),
                        MIN_MONITOR_WIDTH,
                        !transitioning,
                        lang,
                    ) {
                        monitor_rect = Some(rect);
                    }
                });
        }
        egui::CentralPanel::default()
            .frame(panel_frame(247))
            .show(ui, |ui| {
                ui.push_id(("studio_page", vm.pane as u8), |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt(("studio_detail", vm.pane as u8))
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(RichText::new(page_title(vm.pane, lang)).size(26.0).strong());
                            if let Some(error) = error {
                                section(
                                    ui,
                                    lang.pick(
                                        "操作を確認してください",
                                        "Check this operation",
                                        "请检查此操作",
                                        "작업을 확인하세요",
                                    ),
                                    |ui| {
                                        ui.label(&error.user_message);
                                        for action in &error.suggested_actions {
                                            let label = match action {
                                                UiAction::RefreshCameras => lang.pick(
                                                    "カメラを再検出",
                                                    "Refresh cameras",
                                                    "重新检测摄像头",
                                                    "카메라 새로 고침",
                                                ),
                                                UiAction::RetryAfterError => lang.pick(
                                                    "再試行",
                                                    "Retry",
                                                    "重试",
                                                    "다시 시도",
                                                ),
                                                UiAction::DismissError => {
                                                    lang.pick("閉じる", "Dismiss", "关闭", "닫기")
                                                }
                                                _ => continue,
                                            };
                                            if ui.button(label).clicked() {
                                                state.emit(action.clone());
                                            }
                                        }
                                    },
                                );
                            }
                            match vm.pane {
                                Pane::VrmCamera => {
                                    vrm_page(ui, vm, state, dialog_active, lang);
                                    camera_select_page(ui, vm, state, dialog_active, lang);
                                    render_rich_look_controls(ui, vm, state, lang);
                                }
                                Pane::PoseCamera => {
                                    calibration_page(ui, vm, state, lang);
                                    camera_status_page(
                                        ui,
                                        vm,
                                        state,
                                        preview,
                                        landmarks,
                                        camera_texture,
                                        avatar_mirror,
                                        lang,
                                    );
                                }
                                Pane::ExpressionKeys => expression_keys_page(ui, vm, state, lang),
                                Pane::NdiOutput => output_page(ui, vm, state, lang),
                                Pane::Diagnostics => diagnostics_page(ui, vm, diagnostics, lang),
                                Pane::OssLicenses => oss_licenses_page(ui, lang),
                            }
                            ui.add_space(16.0);
                        });
                });
            });
    };
    if transitioning {
        paint_avatar_transition(
            &mut root,
            viewport,
            expand,
            avatar_texture.clone(),
            previous_monitor_rect,
        );
        let transform = egui::emath::TSTransform::new(
            settings_icon_center(viewport).to_vec2() * collapse,
            1.0 - collapse,
        );
        root.multiply_opacity(1.0 - fade);
        root.with_visual_transform(transform, &mut draw_workspace);
        transition_input_blocker(ctx, viewport);
    } else {
        root.with_visual_transform(egui::emath::TSTransform::IDENTITY, &mut draw_workspace);
    }
    state.monitor_image_rect = monitor_rect;
    if dialog_active || vm.avatar_import_review.review.is_some() || egui::Popup::is_any_open(ctx) {
        state.monitor_pointer_hovered = false;
        state.monitor_input_size = None;
    }
    let over_ui = ctx
        .input(|input| input.pointer.interact_pos())
        .is_some_and(|pos| viewport.contains(pos));
    render_avatar_import_review(ctx, vm, state, lang, over_ui)
}

/// Paints the avatar preview card expanding from the monitor rectangle to the
/// whole viewport. Drawn before the workspace, so the collapsing panels stay
/// on top of it while it grows underneath; `expand` is already delayed and
/// eased so the workspace collapse remains the primary motion.
fn paint_avatar_transition(
    root: &mut Ui,
    viewport: egui::Rect,
    expand: f32,
    texture: Option<AvatarPreviewTexture>,
    monitor_rect: Option<egui::Rect>,
) {
    // Mask the live 3D scene with the avatar-only background: the workspace is
    // translucent during the transition, and the card must be the only avatar
    // surface the viewer can read.
    root.painter()
        .rect_filled(viewport, CornerRadius::ZERO, STUDIO_BACKGROUND);
    let (Some(texture), Some(monitor)) = (texture, monitor_rect) else {
        return;
    };
    let card = monitor.lerp_towards(&viewport, expand);
    let radius = ((1.0 - expand) * f32::from(MONITOR_CARD_RADIUS)).round() as u8;
    let background_alpha = (255.0 * (1.0 - expand)).round() as u8;
    root.painter().rect_filled(
        card,
        CornerRadius::same(radius),
        Color32::from_rgba_unmultiplied(
            STUDIO_BACKGROUND.r(),
            STUDIO_BACKGROUND.g(),
            STUDIO_BACKGROUND.b(),
            background_alpha,
        ),
    );
    let overlay = root.new_child(
        egui::UiBuilder::new()
            .id_salt("studio_transition_preview")
            .max_rect(viewport),
    );
    paint_avatar_preview_at(
        overlay.painter(),
        card,
        texture.image().clone(),
        f32::from(radius),
    );
}

/// Swallows pointer input while the workspace is visually transformed away
/// from its layout rectangles. The floating settings control lives in a higher
/// layer and therefore stays clickable.
fn transition_input_blocker(ctx: &egui::Context, viewport: egui::Rect) {
    egui::Area::new(Id::new("studio_transition_input_blocker"))
        .order(egui::Order::Middle)
        .fixed_pos(viewport.min)
        .show(ctx, |ui| {
            ui.allocate_exact_size(viewport.size(), egui::Sense::click_and_drag());
        });
}

fn import_controls(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    active: bool,
    lang: UiLanguage,
) {
    if let Some(model) = &vm.avatar.imported_model {
        ui.label(RichText::new(&model.name).strong());
        ui.label(match model.generation {
            crate::import::VrmGeneration::Vrm0 => "VRM 0.x",
            crate::import::VrmGeneration::Vrm1 => "VRM 1.0",
        });
    }
    ui.label(avatar_label(vm.avatar.lifecycle, lang));
    if primary_button(
        ui,
        lang.pick("VRMを読み込む…", "Choose VRM…", "选择VRM…", "VRM 불러오기…"),
        !active,
    )
    .clicked()
    {
        state.import_requested = true;
    }
    ui.label(
        RichText::new(lang.pick(
            "VRM 0.x / 1.0。ファイルをウインドウにドロップしても読み込めます。",
            "VRM 0.x / 1.0. You can also drop a file into this window.",
            "支持VRM 0.x / 1.0，也可将文件拖入窗口。",
            "VRM 0.x / 1.0을 지원합니다. 파일을 창에 끌어 놓아도 됩니다.",
        ))
        .small()
        .weak(),
    );
}

fn vrm_page(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    dialog_active: bool,
    lang: UiLanguage,
) {
    setup_section(
        ui,
        lang.pick("モデル", "Model", "模型", "모델"),
        matches!(
            vm.avatar.lifecycle,
            AvatarLifecycleState::None | AvatarLifecycleState::Failed
        ),
        !dialog_active && vm.avatar_import_review.review.is_none(),
        |ui| {
            import_controls(ui, vm, state, dialog_active, lang);
            if vm.avatar.imported_model.is_some()
                && ui
                    .button(lang.pick(
                        "アバターを解除",
                        "Unload avatar",
                        "卸载虚拟形象",
                        "아바타 해제",
                    ))
                    .clicked()
            {
                state.emit(UiAction::UnloadAvatar);
            }
            if vm.avatar.load_failed
                && ui
                    .button(lang.pick("再読み込み", "Retry load", "重新加载", "다시 불러오기"))
                    .clicked()
            {
                state.emit(UiAction::RetryAfterError);
            }
        },
    );
}

/// Per-model default arm pose sliders, shown on the calibration page.
fn arm_pose_section(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    let Some(target) = vm.model_target.as_ref() else {
        return;
    };
    section(
        ui,
        lang.pick("腕の姿勢", "Arm pose", "手臂姿势", "팔 자세"),
        |ui| {
            ui.label(lang.pick(
                "このモデルの設定として保存されます。",
                "Saved for this model.",
                "设置将为此模型保存。",
                "이 모델의 설정으로 저장됩니다.",
            ));
            let mut profile = vm.arm_pose.profile;
            let mut drop_degrees = profile.arm_drop_radians.to_degrees();
            let mut curl_degrees = profile.finger_curl_radians.to_degrees();
            let mut changed = ui
                .add(
                    egui::Slider::new(&mut drop_degrees, 0.0..=90.0).text(lang.pick(
                        "腕下げ",
                        "Arm drop",
                        "手臂下垂",
                        "팔 내리기",
                    )),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut profile.reach_ratio, 0.01..=1.0).text(lang.pick(
                        "リーチ比",
                        "Reach ratio",
                        "伸展比例",
                        "뻗기 비율",
                    )),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut profile.forward_hand_offset_ratio, -1.0..=1.0).text(
                        lang.pick(
                            "前方オフセット",
                            "Forward offset",
                            "前向偏移",
                            "앞쪽 오프셋",
                        ),
                    ),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut profile.elbow_pole_offset_ratio, 0.0..=1.0)
                        .text(lang.pick("肘の位置", "Elbow pole", "肘部位置", "팔꿈치 위치")),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut curl_degrees, 0.0..=90.0).text(lang.pick(
                        "指の曲げ",
                        "Finger curl",
                        "手指弯曲",
                        "손가락 굽힘",
                    )),
                )
                .changed();
            if changed {
                profile.arm_drop_radians = drop_degrees.to_radians();
                profile.finger_curl_radians = curl_degrees.to_radians();
                state.emit(UiAction::SetArmPoseProfile {
                    target: target.clone(),
                    profile: ArmPoseProfileOverride::from_profile(profile),
                });
            }
            if vm.arm_pose.has_override
                && ui
                    .button(lang.pick(
                        "自動に戻す",
                        "Reset to automatic",
                        "恢复自动",
                        "자동으로 되돌리기",
                    ))
                    .clicked()
            {
                state.emit(UiAction::ResetArmPoseProfile {
                    target: target.clone(),
                });
            }
        },
    );
}

fn calibration_page(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    section(
        ui,
        lang.pick("中立姿勢", "Neutral pose", "自然姿势", "중립 자세"),
        |ui| {
            ui.label(lang.pick("カメラに向かい、自然な表情で開始してください。映像を表示する必要はありません。", "Face the camera with a relaxed expression. There is no need to reveal the camera image.", "面向摄像头，保持自然表情后开始。无需显示摄像头影像。", "카메라를 향해 편안한 표정으로 시작하세요. 카메라 영상을 표시할 필요는 없습니다."));
            if vm.calibration.is_calibrating {
                ui.label(lang.pick(
                    "顔を検出したら中立姿勢を更新します。カメラに顔を映してください。",
                    "The neutral pose will update when a face is detected. Face the camera.",
                    "检测到脸部后将更新自然姿势。请面向摄像头。",
                    "얼굴이 감지되면 중립 자세를 갱신합니다. 카메라를 바라보세요.",
                ));
                if ui
                    .button(lang.pick("キャンセル", "Cancel", "取消", "취소"))
                    .clicked()
                {
                    state.emit(UiAction::CancelCalibration);
                }
            } else if vm.calibration.is_complete {
                ui.label(lang.pick(
                    "キャリブレーション完了",
                    "Calibrated",
                    "校准完成",
                    "캘리브레이션 완료",
                ));
                if ui
                    .button(lang.pick("やり直す", "Redo", "重新校准", "다시 하기"))
                    .clicked()
                {
                    state.emit(UiAction::RetryCalibration);
                }
            } else {
                if vm.tracking.state == TrackingState::WaitingForFace {
                    ui.label(lang.pick(
                        "顔の検出待ちです。最初に検出した顔から中立姿勢を自動設定します。",
                        "Waiting for a face. The first detected face sets the neutral pose automatically.",
                        "正在等待检测脸部。首次检测到的脸部会自动设置自然姿势。",
                        "얼굴 감지를 기다리고 있습니다. 처음 감지된 얼굴로 중립 자세를 자동 설정합니다.",
                    ));
                }
                if primary_button(
                    ui,
                    lang.pick(
                        "キャリブレーション開始",
                        "Begin calibration",
                        "开始校准",
                        "캘리브레이션 시작",
                    ),
                    vm.can_calibrate(),
                )
                .clicked()
                {
                    state.emit(UiAction::BeginCalibration);
                }
                if !vm.can_calibrate() {
                    ui.label(lang.pick(
                        "VRMとカメラを選択し、映像が届くまでお待ちください。",
                        "Select a VRM and camera, then wait for the camera feed.",
                        "请选择VRM和摄像头，并等待摄像头影像。",
                        "VRM과 카메라를 선택하고 영상이 들어올 때까지 기다리세요.",
                    ));
                }
            }
            if let Some(score) = vm.calibration.quality_score {
                ui.label(format!(
                    "{}: {:.0}%",
                    lang.pick("品質", "Quality", "质量", "품질"),
                    score * 100.0
                ));
            }
            if let Some(reason) = &vm.calibration.last_reject_reason {
                ui.label(format!(
                    "{}: {reason}",
                    lang.pick("詳細", "Details", "详情", "세부 정보")
                ));
            }
        },
    );
    section(
        ui,
        lang.pick("腕のトラッキング", "Arm tracking", "手臂跟踪", "팔 트래킹"),
        |ui| {
            let mut enabled = vm.arm_tracking_enabled;
            if ui
                .checkbox(
                    &mut enabled,
                    lang.pick(
                        "Webカメラで腕を追跡",
                        "Track arms from the webcam",
                        "使用网络摄像头跟踪手臂",
                        "웹캠으로 팔 추적",
                    ),
                )
                .changed()
            {
                state.emit(UiAction::SetArmTrackingEnabled { enabled });
            }
            if ui
                .add_enabled(
                    vm.arm_tracking_enabled,
                    egui::Button::new(lang.pick(
                        "腕の長さを再校正",
                        "Recalibrate arm length",
                        "重新校准手臂长度",
                        "팔 길이 재보정",
                    )),
                )
                .clicked()
            {
                state.emit(UiAction::RecalibrateArms);
            }
            ui.label(
                RichText::new(lang.pick(
                    "肩・肘・手首に加えて手も画面に入れてください。手が見えない腕はPoseの推定を信頼できないため追跡せず、仮想の腕へゆっくり戻します。指の動きは対象外ですが、手のひらの向きは追跡します。",
                    "Keep the hands in frame along with the shoulders, elbows, and wrists. An arm whose hand is not detected is not trusted from Pose alone and eases back to the virtual arm. Finger articulation is out of scope; the palm orientation is tracked.",
                    "请将手与肩、肘、手腕一起保持在画面内。看不到手的另一侧手臂不信任Pose估计，会缓慢回到虚拟手臂。手指动作不在范围内，手掌朝向会被跟踪。",
                    "어깨, 팔꿈치, 손목과 함께 손도 화면에 유지하세요. 손이 보이지 않는 팔은 Pose 추정을 신뢰하지 않고 가상 팔로 천천히 돌아갑니다. 손가락 동작은 범위 밖이지만 손바닥 방향은 추적합니다.",
                ))
                .small()
                .weak(),
            );
        },
    );
    arm_pose_section(ui, vm, state, lang);
}

fn output_page(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    section(
        ui,
        lang.pick(
            "NDI — アバターのみ",
            "NDI — avatar only",
            "NDI — 仅虚拟形象",
            "NDI — 아바타만",
        ),
        |ui| {
            ui.label(lang.pick("常時プレビューと同じアバター専用描画を送信します。設定UI、カメラ映像、プレビュー枠は送信しません。", "Sends the same avatar-only render as the persistent preview. Settings, camera pixels, and preview chrome are never sent.", "发送与常驻预览相同的虚拟形象画面，不发送设置界面、摄像头影像或预览边框。", "항상 표시되는 미리 보기와 같은 아바타 전용 화면을 전송합니다. 설정, 카메라 영상, 미리 보기 테두리는 전송하지 않습니다."));
            let status = match vm.ndi_output.state {
                NdiOutputUiState::Off => lang.pick("停止", "Off", "已停止", "중지"),
                NdiOutputUiState::Starting => {
                    lang.pick("開始中…", "Starting…", "正在启动…", "시작 중…")
                }
                NdiOutputUiState::Live => lang.pick("送信中", "Sending", "发送中", "전송 중"),
                NdiOutputUiState::Error => lang.pick("エラー", "Error", "错误", "오류"),
            };
            ui.label(format!(
                "{}: {status}",
                lang.pick("状態", "Status", "状态", "상태")
            ));
            if let Some(name) = &vm.ndi_output.source_name {
                ui.label(format!(
                    "{}: {name}",
                    lang.pick("ソース名", "Source name", "源名称", "소스 이름")
                ));
            }
            if let Some(count) = vm.ndi_output.connections {
                ui.label(format!(
                    "{}: {count}",
                    lang.pick("受信数", "Receivers", "接收端数量", "수신 수")
                ));
            }
            if !vm.ndi_output.available {
                ui.label(lang.pick(
                    "このビルドにはNDI出力が含まれていません。",
                    "This build does not include NDI output.",
                    "此构建未包含NDI输出。",
                    "이 빌드에는 NDI 출력이 포함되어 있지 않습니다.",
                ));
            } else if !vm.ndi_output.runtime_installed {
                ui.label(lang.pick(
                    "NDIランタイムが見つかりません。NDIを使う場合のみ必要です。",
                    "NDI runtime not found. It is needed only when using NDI.",
                    "未找到NDI运行库。仅使用NDI时需要它。",
                    "NDI 런타임을 찾지 못했습니다. NDI를 사용할 때만 필요합니다.",
                ));
                ui.hyperlink_to(
                    lang.pick(
                        "NDI公式サイト",
                        "NDI website",
                        "NDI官方网站",
                        "NDI 공식 사이트",
                    ),
                    "https://ndi.video",
                );
            }
            ui.horizontal_wrapped(|ui| {
                if primary_button(
                    ui,
                    lang.pick("NDI送信を開始", "Start NDI", "开始NDI发送", "NDI 전송 시작"),
                    vm.can_start_ndi_output() && vm.ndi_output.runtime_installed,
                )
                .clicked()
                {
                    state.emit(UiAction::StartNdiOutput);
                }
                if ui
                    .add_enabled(
                        vm.can_stop_ndi_output(),
                        egui::Button::new(lang.pick(
                            "NDI送信を停止",
                            "Stop NDI",
                            "停止NDI发送",
                            "NDI 전송 중지",
                        )),
                    )
                    .clicked()
                {
                    state.emit(UiAction::StopNdiOutput);
                }
            });
            if let Some(code) = &vm.ndi_output.error_code {
                ui.label(format!(
                    "{}: {code}",
                    lang.pick("エラーコード", "Error code", "错误代码", "오류 코드")
                ));
            }
            if let Some(detail) = &vm.ndi_output.error_message {
                egui::CollapsingHeader::new(lang.pick(
                    "技術情報（原文）",
                    "Technical details (original)",
                    "技术详情（原文）",
                    "기술 정보 (원문)",
                ))
                .show(ui, |ui| {
                    ui.label(detail);
                });
            }
        },
    );
    section(
        ui,
        lang.pick(
            "画面キャプチャで使う",
            "Use screen capture",
            "使用屏幕捕获",
            "화면 캡처 사용",
        ),
        |ui| {
            ui.label(lang.pick("F1で設定を隠し、アバター表示に戻せます。ただし、デスクトップ／ウインドウキャプチャには、後から開いた設定や表示を許可したカメラ映像も映ります。アバターのみを確実に送る場合はNDIを選んでください。", "F1 hides settings and returns to the avatar. Desktop/window capture can still include settings opened later or a camera preview you explicitly reveal. Use NDI for an avatar-only feed.", "按F1隐藏设置并返回虚拟形象。桌面或窗口捕获仍会包含之后打开的设置以及您确认显示的摄像头影像。需要仅发送虚拟形象时，请使用NDI。", "F1로 설정을 숨기고 아바타 화면으로 돌아갑니다. 데스크톱이나 창 캡처에는 나중에 연 설정이나 직접 표시한 카메라 영상도 포함될 수 있습니다. 아바타만 전송하려면 NDI를 사용하세요."));
            if ui
                .button(lang.pick(
                    "アバターだけを表示 (F1)",
                    "Show avatar only (F1)",
                    "仅显示虚拟形象 (F1)",
                    "아바타만 표시 (F1)",
                ))
                .clicked()
            {
                state.set_controls_open(false);
            }
        },
    );
}

/// Rich-look switch and lighting/effect strength.
///
/// The controls edit the live [`vtuber_avatar::AvatarLookSettings`] the look
/// systems read, so the change is visible without a reload; the save button
/// persists the same values for the loaded model.
fn render_rich_look_controls(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    let Some(target) = vm.model_target.as_ref() else {
        return;
    };
    let switch = lang.pick("リッチ表示", "Enhanced look", "增强显示", "고급 렌더링");
    section(ui, switch, |ui| {
        let mut enabled = vm.look.enabled;
        if ui.checkbox(&mut enabled, switch).changed() {
            state.emit(UiAction::ChangeRichLook {
                target: target.clone(),
                change: RichLookChange::Enabled(enabled),
            });
        }
        let mut percent = vm.look.strength * 100.0;
        if ui
            .add(
                egui::Slider::new(&mut percent, 0.0..=100.0)
                    .suffix("%")
                    .text(lang.pick("効果の強さ", "Effect strength", "效果强度", "효과 강도")),
            )
            .changed()
        {
            state.emit(UiAction::ChangeRichLook {
                target: target.clone(),
                change: RichLookChange::Strength(percent / 100.0),
            });
        }
        if vm.avatar.imported_model.is_some()
            && ui
                .button(lang.pick(
                    "このモデルに設定を保存",
                    "Save for this model",
                    "为此模型保存设置",
                    "이 모델에 설정 저장",
                ))
                .clicked()
        {
            state.emit(UiAction::SaveRichLook {
                target: target.clone(),
            });
        }
    });
}

fn diagnostics_page(
    ui: &mut Ui,
    vm: &UiViewModel,
    snapshot: &DiagnosticsSnapshot,
    lang: UiLanguage,
) {
    section(
        ui,
        lang.pick("トラッキング", "Tracking", "跟踪", "트래킹"),
        |ui| {
            let tracking = match vm.tracking.state {
                TrackingState::Idle => lang.pick("停止中", "Idle", "已停止", "중지됨"),
                TrackingState::Initializing => {
                    lang.pick("初期化中", "Initializing", "初始化中", "초기화 중")
                }
                TrackingState::WaitingForFace => lang.pick(
                    "顔の検出待ち（カメラに顔を映してください）",
                    "Waiting for a face (face the camera)",
                    "等待检测脸部（请面向摄像头）",
                    "얼굴 감지 대기 중 (카메라를 바라보세요)",
                ),
                TrackingState::Tracking => lang.pick("追跡中", "Tracking", "跟踪中", "트래킹 중"),
                TrackingState::Lost => lang.pick(
                    "顔を検出できません",
                    "Face lost",
                    "未检测到脸部",
                    "얼굴을 찾지 못함",
                ),
            };
            ui.label(tracking);
            ui.label(format!(
                "{}: {:.0}%",
                lang.pick("信頼度", "Confidence", "置信度", "신뢰도"),
                vm.tracking.confidence * 100.0
            ));
            ui.label(avatar_label(vm.avatar.lifecycle, lang));
        },
    );
    section(
        ui,
        lang.pick("詳細診断", "Detailed diagnostics", "详细诊断", "상세 진단"),
        |ui| {
            ui.label(lang.pick("技術的な識別子とバックエンドのメッセージは原文で表示します。カメラ映像は表示しません。", "Technical identifiers and backend messages retain their original text. No camera image is shown.", "技术标识符和后端消息保留原文，不显示摄像头影像。", "기술 식별자와 백엔드 메시지는 원문으로 표시합니다. 카메라 영상은 표시하지 않습니다."));
            egui::CollapsingHeader::new(lang.pick(
                "技術情報を開く",
                "Open technical details",
                "打开技术详情",
                "기술 정보 열기",
            ))
            .show(ui, |ui| {
                ui.label(RichText::new(format!("{snapshot:#?}")).monospace());
            });
        },
    );
}

/// Licenses of the built application. Only the headings are localized; every
/// license body stays in the language its author published it in.
fn oss_licenses_page(ui: &mut Ui, lang: UiLanguage) {
    section(
        ui,
        lang.pick(
            "表示について",
            "About this list",
            "关于本列表",
            "이 목록에 대하여",
        ),
        |ui| {
            ui.label(lang.pick(
                "ライセンスの本文は配布元のファイルのまま表示し、表示言語では翻訳しません。",
                "License bodies are shown exactly as their publisher shipped them and are never translated into the display language.",
                "许可证正文按发布方提供的原文件显示，不随界面语言翻译。",
                "라이선스 본문은 배포된 파일 그대로 표시하며, 표시 언어로 번역하지 않습니다.",
            ));
        },
    );
    section(
        ui,
        lang.pick("同梱アセット", "Bundled assets", "内置资源", "번들 에셋"),
        |ui| {
            license_groups(ui, "bundled", crate::licenses::bundled_groups(), lang);
        },
    );
    section(
        ui,
        lang.pick(
            "依存パッケージ",
            "Dependency packages",
            "依赖包",
            "의존 패키지",
        ),
        |ui| {
            ui.label(lang.pick(
                "Cargo.lock が記録している、このアプリのビルドに使ったすべてのパッケージです。",
                "Every package Cargo.lock records as an input to this application's build.",
                "Cargo.lock 记录的、构建本应用所用的全部软件包。",
                "Cargo.lock에 기록된 이 애플리케이션 빌드에 사용된 모든 패키지입니다.",
            ));
            license_groups(
                ui,
                "dependencies",
                crate::licenses::dependency_groups(),
                lang,
            );
        },
    );
}

fn license_groups(ui: &mut Ui, list_id: &str, groups: &[LicenseGroup], lang: UiLanguage) {
    for (group_index, group) in groups.iter().enumerate() {
        let heading = format!(
            "{}  ({} {})",
            group.expression,
            group.items.len(),
            lang.pick("件", "items", "项", "개")
        );
        egui::CollapsingHeader::new(RichText::new(heading).strong())
            .id_salt(("oss_licenses", list_id, group_index))
            .show(ui, |ui| {
                let mut items = group
                    .items
                    .iter()
                    .map(|item| {
                        if item.version.is_empty() {
                            item.name.to_string()
                        } else {
                            format!("{} {}", item.name, item.version)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                items.push('\n');
                ui.label(RichText::new(items).monospace());
                if group.texts.is_empty() {
                    ui.label(lang.pick(
                        "このパッケージはライセンス本文を同梱していません。上流の配布元を参照してください。",
                        "These packages ship no license text. See their upstream distribution.",
                        "这些包未附带许可证正文，请参阅其上游分发。",
                        "이 패키지에는 라이선스 본문이 포함되어 있지 않습니다. 업스트림 배포를 참조하세요.",
                    ));
                }
                for (text_index, text) in group.texts.iter().enumerate() {
                    // Two crates under one expression rarely share a file, and
                    // their copyright lines are what tells them apart.
                    let holder = text.body.lines().next().unwrap_or_default().trim();
                    let title = if holder.is_empty() {
                        text.file_name.to_string()
                    } else {
                        format!("{} — {holder}", text.file_name)
                    };
                    egui::CollapsingHeader::new(RichText::new(title).monospace())
                        .id_salt(("license_text", text_index))
                        .show(ui, |ui| {
                            ui.label(RichText::new(text.body).monospace());
                        });
                }
            });
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;
    #[test]
    fn compact_layout_keeps_navigation_and_a_persistent_monitor() {
        assert!(!use_sidebar(720.0));
        assert!(use_sidebar(1600.0));
        let destinations: Vec<Pane> = destinations().collect();
        assert_eq!(destinations.len(), 6);
        assert_eq!(destinations.first(), Some(&Pane::VrmCamera));
        assert!(destinations.contains(&Pane::PoseCamera));
        assert!(destinations.contains(&Pane::ExpressionKeys));
        assert!(destinations.contains(&Pane::NdiOutput));
        assert!(destinations.contains(&Pane::Diagnostics));
        assert_eq!(destinations.last(), Some(&Pane::OssLicenses));
    }

    /// The regression: a 1280px window used to fall back to the compact layout
    /// and a 200px preview even though the side-by-side layout still fits.
    #[test]
    fn the_preview_stays_wide_near_1280_and_the_layout_switches_only_at_its_minimum() {
        let preview = |width: f32| monitor_panel_width(width) - MONITOR_PANEL_PADDING;
        // Wide windows get the full target; 1280 gets a preview that is still
        // three times the one the old 1292px threshold fell back to.
        assert_eq!(preview(1600.0), MONITOR_TARGET_WIDTH);
        assert_eq!(preview(1280.0), 648.0);
        assert_eq!(preview(1000.0), 368.0);
        assert!(use_sidebar(SIDEBAR_LAYOUT_MIN_WIDTH));
        assert!(!use_sidebar(SIDEBAR_LAYOUT_MIN_WIDTH - 1.0));
        for width in [SIDEBAR_LAYOUT_MIN_WIDTH, 900.0, 1280.0, 1600.0, 2400.0] {
            assert!(use_sidebar(width), "{width} keeps the side-by-side layout");
            assert!(
                width - SIDEBAR_PANEL_WIDTH - monitor_panel_width(width) >= MIN_SETTINGS_WIDTH,
                "{width} keeps the settings column usable"
            );
        }
    }

    /// The width the monitor card really gets, measured through the rendered
    /// panels rather than through the width arithmetic alone.
    fn rendered_preview_width(viewport_width: f32) -> f32 {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                vec2(viewport_width, 900.0),
            )),
            ..Default::default()
        };
        let mut vm = UiViewModel::default();
        vm.avatar.is_ready = true;
        vm.avatar.lifecycle = AvatarLifecycleState::Ready;
        let mut state = UiState::default();
        let _ = ctx.run_ui(input, |ui| {
            render_studio(
                ui.ctx(),
                &vm,
                &mut state,
                &DiagnosticsSnapshot::default(),
                None,
                &PreviewState::default(),
                &PreviewLandmarkState::default(),
                AvatarMotionMirror::default(),
                None,
                Some(AvatarPreviewTexture::new(
                    bevy::asset::Handle::default(),
                    vtuber_core::VideoOutputProfile::default(),
                )),
                false,
                UiLanguage::Ja,
            );
        });
        state.monitor_image_rect.map_or(0.0, |rect| rect.width())
    }

    #[test]
    fn preview_accepts_pointer_gestures_in_wide_and_compact_layouts() {
        for width in [1600.0, 700.0] {
            for button in [egui::PointerButton::Primary, egui::PointerButton::Secondary] {
                let ctx = egui::Context::default();
                let mut state = UiState::default();
                let mut vm = UiViewModel::default();
                vm.avatar.is_ready = true;
                vm.avatar.lifecycle = AvatarLifecycleState::Ready;
                let render = |state: &mut UiState, events: Vec<egui::Event>, dialog_active| {
                    let _ = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                vec2(width, 900.0),
                            )),
                            events,
                            ..Default::default()
                        },
                        |ui| {
                            render_studio(
                                ui.ctx(),
                                &vm,
                                state,
                                &DiagnosticsSnapshot::default(),
                                None,
                                &PreviewState::default(),
                                &PreviewLandmarkState::default(),
                                AvatarMotionMirror::default(),
                                None,
                                Some(AvatarPreviewTexture::new(
                                    bevy::asset::Handle::default(),
                                    vtuber_core::VideoOutputProfile::default(),
                                )),
                                dialog_active,
                                UiLanguage::Ja,
                            );
                        },
                    );
                };
                // Let egui finish panel sizing and register widgets for hit testing.
                render(&mut state, vec![], false);
                render(&mut state, vec![], false);
                let rect = state.monitor_image_rect.expect("preview rectangle");
                let start = rect.center();
                let pointer_button = |pos, pressed| egui::Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                };
                render(&mut state, vec![egui::Event::PointerMoved(start)], false);
                assert!(
                    state.monitor_pointer_hovered,
                    "width={width}, button={button:?}, rect={rect:?}, current={:?}",
                    state.monitor_image_rect
                );
                render(
                    &mut state,
                    vec![egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Line,
                        delta: vec2(0.0, 1.0),
                        modifiers: egui::Modifiers::NONE,
                        phase: egui::TouchPhase::Move,
                    }],
                    false,
                );
                assert!(state.monitor_pointer_hovered, "wheel remains camera input");
                render(&mut state, vec![pointer_button(start, true)], false);
                assert!(state.monitor_pointer_hovered);
                let end = start + vec2(30.0, 20.0);
                render(&mut state, vec![egui::Event::PointerMoved(end)], false);
                assert_eq!(state.monitor_input_size, Some(rect.size()));
                render(&mut state, vec![pointer_button(end, false)], false);
                assert!(
                    state.controls_open,
                    "drag release must not open full screen"
                );

                render(&mut state, vec![], true);
                assert!(
                    !state.monitor_pointer_hovered,
                    "dialog blocks preview input"
                );
                assert_eq!(state.monitor_input_size, None);
                render(
                    &mut state,
                    vec![egui::Event::PointerMoved(egui::pos2(10.0, 10.0))],
                    false,
                );
                assert!(
                    !state.monitor_pointer_hovered,
                    "outside preview is settings UI"
                );
                assert_eq!(state.monitor_input_size, None);

                // A plain click still opens the avatar-only view.
                if button == egui::PointerButton::Primary {
                    render(&mut state, vec![egui::Event::PointerMoved(start)], false);
                    render(&mut state, vec![pointer_button(start, true)], false);
                    render(&mut state, vec![pointer_button(start, false)], false);
                    assert!(!state.controls_open);
                }
            }
        }
    }

    #[test]
    fn the_rendered_preview_matches_the_layout_at_each_width() {
        assert_eq!(rendered_preview_width(1600.0), MONITOR_TARGET_WIDTH);
        assert_eq!(rendered_preview_width(1280.0), 648.0);
        assert_eq!(rendered_preview_width(900.0), 268.0);
        // Below the minimum the monitor moves above the settings and keeps the
        // same narrow preview the compact layout has always shown.
        assert_eq!(
            rendered_preview_width(SIDEBAR_LAYOUT_MIN_WIDTH - 1.0),
            MIN_MONITOR_WIDTH
        );
    }

    #[test]
    fn floating_control_reopens_settings_on_click() {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        state.set_controls_open(false);
        // Frame 1 is the area sizing pass (invisible); frame 2 registers the
        // interactive widget for egui's next-frame hit test.
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            floating_settings_control(ui.ctx(), &mut state, UiLanguage::Ja, 1.0);
        });
        let mut rect = egui::Rect::NOTHING;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            rect = floating_settings_control(ui.ctx(), &mut state, UiLanguage::Ja, 1.0).rect;
        });
        assert!(!state.controls_open);
        assert!(rect.width() > 0.0 && rect.height() > 0.0);
        let pos = rect.center();
        let mut input = egui::RawInput::default();
        input.events.push(egui::Event::PointerMoved(pos));
        for pressed in [true, false] {
            input.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            });
        }
        let _ = ctx.run_ui(input, |ui| {
            floating_settings_control(ui.ctx(), &mut state, UiLanguage::Ja, 1.0);
        });
        assert!(state.controls_open);
    }

    #[test]
    fn transition_card_expands_from_the_monitor_to_the_whole_viewport() {
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, vec2(1280.0, 720.0));
        let monitor = egui::Rect::from_min_size(egui::pos2(940.0, 60.0), vec2(268.0, 150.75));
        assert_eq!(monitor.lerp_towards(&viewport, 0.0), monitor);
        assert_eq!(monitor.lerp_towards(&viewport, 1.0), viewport);
        let half = monitor.lerp_towards(&viewport, 0.5);
        assert!(half.min.x < monitor.min.x && half.max.x > monitor.max.x);
        assert!(half.min.y < monitor.min.y && half.max.y > monitor.max.y);
    }

    #[test]
    fn collapse_transform_converges_on_the_settings_icon() {
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, vec2(1280.0, 720.0));
        let icon = settings_icon_center(viewport);
        let transformed = |progress: f32| {
            let transform = egui::emath::TSTransform::new(
                settings_icon_center(viewport).to_vec2() * progress,
                1.0 - progress,
            );
            (
                transform.mul_pos(viewport.min),
                transform.mul_pos(viewport.max),
            )
        };
        assert_eq!(transformed(0.0), (viewport.min, viewport.max));
        let (min, max) = transformed(1.0);
        assert!(min.distance(icon) < f32::EPSILON);
        assert!(max.distance(icon) < f32::EPSILON);
        let (min, max) = transformed(0.5);
        assert!(min.x > viewport.min.x && max.x < viewport.max.x);
    }

    #[test]
    fn transition_advances_and_paints_the_collapsing_workspace() {
        let ctx = egui::Context::default();
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                vec2(1280.0, 720.0),
            )),
            ..Default::default()
        };
        let mut vm = UiViewModel::default();
        vm.avatar.is_ready = true;
        vm.avatar.lifecycle = AvatarLifecycleState::Ready;
        let mut state = UiState::default();
        let diagnostics = DiagnosticsSnapshot::default();
        let preview = PreviewState::default();
        let landmarks = PreviewLandmarkState::default();
        let render = |state: &mut UiState| {
            let _ = ctx.run_ui(input(), |ui| {
                render_studio(
                    ui.ctx(),
                    &vm,
                    state,
                    &diagnostics,
                    None,
                    &preview,
                    &landmarks,
                    AvatarMotionMirror::default(),
                    None,
                    Some(AvatarPreviewTexture::new(
                        bevy::asset::Handle::default(),
                        vtuber_core::VideoOutputProfile::default(),
                    )),
                    false,
                    UiLanguage::Ja,
                );
            });
        };
        render(&mut state);
        assert_eq!(state.avatar_only_progress, 0.0);
        assert!(state.monitor_image_rect.is_some());
        state.set_controls_open(false);
        let mut saw_mid_transition = false;
        for _ in 0..40 {
            render(&mut state);
            let progress = state.avatar_only_progress;
            saw_mid_transition |= progress > 0.0 && progress < 1.0;
        }
        assert!(
            saw_mid_transition,
            "the transition must animate over frames"
        );
        assert_eq!(state.avatar_only_progress, 1.0);
    }

    #[test]
    fn the_license_page_renders_every_group_in_every_language() {
        for lang in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            let ctx = egui::Context::default();
            let mut state = UiState::default();
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                oss_licenses_page(ui, lang);
            });
            // The license bodies are reference material, so the page only reads.
            assert!(state.take_actions().is_empty());
        }
    }

    #[test]
    fn every_destination_has_a_label_in_all_four_languages() {
        for pane in destinations() {
            for lang in [
                UiLanguage::Ja,
                UiLanguage::En,
                UiLanguage::Zh,
                UiLanguage::Ko,
            ] {
                assert!(!page_title(pane, lang).is_empty());
            }
        }
    }

    fn test_generation() -> vtuber_avatar::AvatarGeneration {
        vtuber_avatar::AvatarGeneration(1)
    }

    #[test]
    fn rich_look_view_model_default_matches_the_look_settings() {
        let model = crate::ui_model::RichLookViewModel::default();
        let settings = vtuber_avatar::RichLookSettings::default();
        assert_eq!(model.enabled, settings.enabled());
        assert_eq!(model.strength, settings.strength());
    }

    #[test]
    fn rich_look_controls_emit_no_actions_without_input_in_any_language() {
        for lang in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            let ctx = egui::Context::default();
            let mut state = UiState::default();
            let vm = UiViewModel {
                model_target: Some(crate::actions::ModelActionTarget {
                    model_id: "model".into(),
                    generation: test_generation(),
                }),
                look: crate::ui_model::RichLookViewModel {
                    enabled: true,
                    strength: 0.5,
                },
                ..Default::default()
            };
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                render_rich_look_controls(ui, &vm, &mut state, lang);
            });
            let mut off = vm;
            off.look.enabled = false;
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                render_rich_look_controls(ui, &off, &mut state, lang);
            });
            // Neither enabled nor disabled controls emit actions without input.
            assert!(state.take_actions().is_empty());
        }
    }
}
