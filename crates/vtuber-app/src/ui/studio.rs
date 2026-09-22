//! Apple-style desktop workspace. The sidebar contains destinations only.
//! Camera controls, session commands, and state never live in that sidebar.

use super::avatar_preview::{AvatarPreviewTexture, paint_avatar_preview, paint_avatar_preview_at};
use super::privacy::{CameraPreviewConsent, CameraPreviewEvent};
use super::shell::UiState;
use crate::actions::UiAction;
use crate::diagnostics::DiagnosticsSnapshot;
use crate::error_presenter::ErrorPresentation;
use crate::expression_keys::ExpressionKey;
use crate::license_review::VrmLicenseReview;
use crate::preview::PreviewState;
use crate::preview_landmarks::PreviewLandmarkState;
use crate::settings::UiLanguage;
use crate::ui_model::{
    AppLifecycle, AvatarLifecycleState, ExpressionEntryViewModel, NdiOutputUiState, Pane,
    TrackingState, UiViewModel,
};
use bevy_egui::egui::{self, Color32, CornerRadius, Frame, Id, RichText, TextureId, Ui, vec2};
use vtuber_avatar::{
    ArmPoseProfileOverride, AvatarMotionMirror, ExpressionAvailability, ExpressionKind,
};
use vtuber_core::monotonic_now;

const NAVIGATION: [Pane; 7] = [
    Pane::Studio,
    Pane::Avatar,
    Pane::Camera,
    Pane::Calibration,
    Pane::NdiOutput,
    Pane::Settings,
    Pane::Diagnostics,
];

fn page_title(pane: Pane, lang: UiLanguage) -> &'static str {
    match pane {
        Pane::Studio => lang.pick("スタジオ", "Studio", "工作室", "스튜디오"),
        Pane::Avatar => lang.pick("アバター", "Avatar", "虚拟形象", "아바타"),
        Pane::Camera | Pane::Preview => lang.pick("カメラ", "Camera", "摄像头", "카메라"),
        Pane::Calibration => lang.pick("キャリブレーション", "Calibration", "校准", "캘리브레이션"),
        Pane::NdiOutput => lang.pick("出力", "Output", "输出", "출력"),
        Pane::Settings => lang.pick(
            "表情のキー割り当て",
            "Expression Key Bindings",
            "表情按键绑定",
            "표정 키 할당",
        ),
        Pane::Diagnostics => lang.pick("診断", "Diagnostics", "诊断", "진단"),
    }
}

fn session_label(state: AppLifecycle, lang: UiLanguage) -> &'static str {
    match state {
        AppLifecycle::Idle => lang.pick("待機中", "Idle", "待机", "대기 중"),
        AppLifecycle::Starting => lang.pick("開始中…", "Starting…", "正在启动…", "시작 중…"),
        AppLifecycle::Running => lang.pick("トラッキング中", "Tracking", "跟踪中", "트래킹 중"),
        AppLifecycle::Stopping => lang.pick("停止中…", "Stopping…", "正在停止…", "중지 중…"),
        AppLifecycle::Failed => lang.pick("エラー", "Error", "错误", "오류"),
    }
}

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
    ui.add_space(12.0);
    Frame::new()
        .fill(Color32::WHITE)
        .stroke(egui::Stroke::new(1.0, Color32::from_gray(220)))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(egui::Margin::same(16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(title).size(17.0).strong());
            ui.add_space(10.0);
            contents(ui);
        });
}

fn primary_button(ui: &mut Ui, label: &str, enabled: bool) -> egui::Response {
    ui.add_enabled(
        enabled,
        egui::Button::new(RichText::new(label).color(Color32::WHITE))
            .fill(ui.visuals().hyperlink_color)
            .min_size(vec2(0.0, 30.0)),
    )
}

fn use_sidebar(width: f32) -> bool {
    width >= 1000.0
}

const FLOATING_CONTROL_MARGIN: f32 = 16.0;
const FLOATING_CONTROL_SIZE: f32 = 40.0;
/// egui-managed duration of the workspace <-> avatar-only transition. Long
/// enough that the workspace collapse into the settings icon stays readable.
const AVATAR_ONLY_TRANSITION_SECONDS: f32 = 0.36;
/// Corner radius shared by the avatar monitor card and the expanding preview.
const MONITOR_CARD_RADIUS: u8 = 8;
/// Background of the avatar-only view. `UiShellPlugin` installs the same color
/// as Bevy's window `ClearColor`, so the preview card can dissolve into the 3D
/// scene without a seam.
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

/// Minimal shield-and-check glyph for the license consent sheet, drawn in the
/// same monochrome style as the other controls.
fn paint_license_glyph(painter: &egui::Painter, center: egui::Pos2, color: Color32) {
    let stroke = egui::Stroke::new(1.6, color);
    let width = 11.0;
    let top = center.y - 12.5;
    let shoulder = center.y + 1.0;
    painter.add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(center.x, top),
            egui::pos2(center.x + width, top + 4.0),
            egui::pos2(center.x + width, shoulder),
            egui::pos2(center.x, top + 24.0),
            egui::pos2(center.x - width, shoulder),
            egui::pos2(center.x - width, top + 4.0),
        ],
        Color32::TRANSPARENT,
        stroke,
    ));
    painter.line_segment(
        [
            egui::pos2(center.x - 5.0, center.y - 1.0),
            egui::pos2(center.x - 1.5, center.y + 2.5),
        ],
        stroke,
    );
    painter.line_segment(
        [
            egui::pos2(center.x - 1.5, center.y + 2.5),
            egui::pos2(center.x + 5.5, center.y - 4.5),
        ],
        stroke,
    );
}

// The settings toggle keeps the top-right corner in both states: a floating
// accent control while the avatar fills the window, and the toolbar's
// rightmost button while the workspace is open (Apple HIG: consistent placement).
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
        .on_hover_text(format!("{} (F1)", page_title(Pane::Settings, lang)));
    if response.clicked() {
        state.set_controls_open(true);
    }
    response
}

fn navigation(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    for pane in NAVIGATION {
        let group = match pane {
            Pane::Studio => Some(lang.pick("セットアップ", "Setup", "准备", "준비")),
            Pane::NdiOutput => Some(lang.pick("配信", "Broadcast", "直播", "방송")),
            Pane::Settings => Some(lang.pick("アプリ", "App", "应用", "앱")),
            _ => None,
        };
        if let Some(group) = group {
            ui.add_space(12.0);
            ui.label(RichText::new(group).small().weak());
            ui.add_space(4.0);
        }
        let selected = vm.pane == pane || (pane == Pane::Camera && vm.pane == Pane::Preview);
        if ui
            .add_sized(
                [ui.available_width(), 30.0],
                egui::Button::selectable(selected, page_title(pane, lang)),
            )
            .clicked()
        {
            state.emit(UiAction::SwitchPane(pane));
        }
    }
}

fn compact_navigation(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    egui::ComboBox::from_id_salt("studio_navigation")
        .selected_text(page_title(vm.pane, lang))
        .show_ui(ui, |ui| {
            for pane in NAVIGATION {
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
        }
        ui.separator();
        ui.label(session_label(vm.lifecycle, lang));
        if vm.can_stop() {
            if ui
                .button(lang.pick(
                    "トラッキングを停止",
                    "Stop tracking",
                    "停止跟踪",
                    "트래킹 중지",
                ))
                .clicked()
            {
                state.emit(UiAction::Stop);
            }
        } else if primary_button(
            ui,
            lang.pick(
                "トラッキングを開始",
                "Start tracking",
                "开始跟踪",
                "트래킹 시작",
            ),
            vm.can_start(),
        )
        .clicked()
        {
            state.emit(UiAction::Start);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button(lang.pick(
                    "アバターのみ (F1)",
                    "Avatar only (F1)",
                    "仅虚拟形象 (F1)",
                    "아바타만 (F1)",
                ))
                .clicked()
            {
                state.set_controls_open(false);
            }
        });
    });
}

fn avatar_monitor(
    ui: &mut Ui,
    vm: &UiViewModel,
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
            Frame::new()
                .fill(Color32::from_gray(228))
                .corner_radius(CornerRadius::same(MONITOR_CARD_RADIUS))
                .show(ui, |ui| {
                    let size = vec2(width, height);
                    // The expanding transition card owns the texture while the
                    // workspace collapses; the monitor keeps its layout only,
                    // so the preview is never drawn twice.
                    let response = if draw_preview {
                        paint_avatar_preview(
                            ui,
                            texture.image().clone(),
                            size,
                            f32::from(MONITOR_CARD_RADIUS),
                        )
                    } else {
                        ui.allocate_exact_size(size, egui::Sense::hover()).1
                    };
                    preview_rect = Some(response.rect);
                });
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

#[allow(clippy::too_many_arguments)]
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
    font_error: Option<&str>,
    lang: UiLanguage,
) -> bool {
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
        egui::Panel::top("studio_toolbar")
            .resizable(false)
            .frame(panel_frame(250))
            .show(ui, |ui| toolbar(ui, vm, state, sidebar, lang));
        if sidebar {
            egui::Panel::left("studio_sidebar")
                .exact_size(200.0)
                .resizable(false)
                .frame(panel_frame(242))
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| navigation(ui, vm, state, lang));
                });
            egui::Panel::right("studio_monitor")
                .exact_size(300.0)
                .resizable(false)
                .frame(panel_frame(250))
                .show(ui, |ui| {
                    if let Some(rect) =
                        avatar_monitor(ui, vm, avatar_texture.clone(), 268.0, !transitioning, lang)
                    {
                        monitor_rect = Some(rect);
                    }
                });
        } else {
            // On smaller windows keep the monitor outside the settings scroll area.
            egui::Panel::top("studio_compact_monitor")
                .resizable(false)
                .frame(panel_frame(250))
                .show(ui, |ui| {
                    if let Some(rect) =
                        avatar_monitor(ui, vm, avatar_texture.clone(), 200.0, !transitioning, lang)
                    {
                        monitor_rect = Some(rect);
                    }
                });
        }
        egui::CentralPanel::default()
            .frame(panel_frame(247))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt(("studio_detail", vm.pane as u8))
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new(page_title(vm.pane, lang)).size(26.0).strong());
                        if let Some(detail) = font_error {
                            section(ui, "Font could not be loaded", |ui| {
                                ui.label(detail);
                                ui.horizontal(|ui| {
                                    if ui.button("Japanese").clicked() {
                                        state.emit(UiAction::SetLanguage(UiLanguage::Ja));
                                    }
                                    if ui.button("English").clicked() {
                                        state.emit(UiAction::SetLanguage(UiLanguage::En));
                                    }
                                });
                            });
                        }
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
                                            UiAction::RetryAfterError => {
                                                lang.pick("再試行", "Retry", "重试", "다시 시도")
                                            }
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
                            Pane::Studio => overview(ui, vm, state, dialog_active, lang),
                            Pane::Avatar => {
                                avatar_page(ui, vm, state, avatar_mirror, dialog_active, lang)
                            }
                            Pane::Camera | Pane::Preview => {
                                camera_page(ui, vm, state, preview, landmarks, camera_texture, lang)
                            }
                            Pane::Calibration => calibration_page(ui, vm, state, lang),
                            Pane::NdiOutput => output_page(ui, vm, state, lang),
                            Pane::Settings => settings_page(ui, vm, state, lang),
                            Pane::Diagnostics => diagnostics_page(ui, vm, diagnostics, lang),
                        }
                        ui.add_space(16.0);
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
        root.with_visual_transform(transform, |ui| draw_workspace(ui));
        transition_input_blocker(ctx, viewport);
    } else {
        draw_workspace(&mut root);
    }
    state.monitor_image_rect = monitor_rect;
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
        Color32::from_rgba_unmultiplied(228, 228, 228, background_alpha),
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

/// Renders the license review sheet on top of the workspace when a model is
/// waiting for acceptance. Returns whether the pointer is over any UI.
fn render_avatar_import_review(
    ctx: &egui::Context,
    vm: &UiViewModel,
    state: &mut UiState,
    lang: UiLanguage,
    over_ui: bool,
) -> bool {
    let Some(review) = &vm.avatar_import_review.review else {
        return over_ui;
    };
    avatar_import_review_modal(ctx, review, vm.avatar_import_review.accepted, state, lang);
    true
}

/// Apple-style centered consent sheet shown before an imported VRM reaches the
/// asset store. The newly selected model stays unloaded until this sheet is
/// accepted.
fn avatar_import_review_modal(
    ctx: &egui::Context,
    review: &VrmLicenseReview,
    accepted: bool,
    state: &mut UiState,
    lang: UiLanguage,
) {
    let mut checked = accepted;
    let mut import_requested = false;
    let mut cancel_requested = false;
    let modal = egui::Modal::new(Id::new("avatar_import_review")).show(ctx, |ui| {
        ui.set_width(540.0);
        review_sheet_header(ui, review, lang);
        ui.add_space(12.0);
        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("avatar_import_review_details")
            .max_height(330.0)
            .auto_shrink([false, true])
            .show(ui, |ui| review_sheet_details(ui, review, lang));
        ui.separator();
        ui.add_space(8.0);
        ui.checkbox(
            &mut checked,
            lang.pick(
                "この VRM のライセンス・利用条件を確認しました",
                "I have reviewed this VRM's license and usage terms",
                "我已确认该 VRM 的许可与使用条件",
                "이 VRM의 라이선스 및 이용 조건을 확인했습니다",
            ),
        );
        ui.add_space(8.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if primary_button(
                ui,
                lang.pick("読み込む", "Import", "导入", "불러오기"),
                checked,
            )
            .clicked()
            {
                import_requested = true;
            }
            if ui
                .button(lang.pick("キャンセル", "Cancel", "取消", "취소"))
                .clicked()
            {
                cancel_requested = true;
            }
        });
    });
    if checked != accepted {
        state.emit(UiAction::SetAvatarImportReviewAccepted { accepted: checked });
    }
    if modal.should_close() || cancel_requested {
        state.emit(UiAction::CancelAvatarImportReview);
    } else if import_requested {
        state.emit(UiAction::AcceptAvatarImportReview);
    }
}

fn review_sheet_header(ui: &mut Ui, review: &VrmLicenseReview, lang: UiLanguage) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(44.0, 44.0), egui::Sense::hover());
        ui.painter()
            .circle_filled(rect.center(), 22.0, Color32::from_gray(238));
        paint_license_glyph(ui.painter(), rect.center(), Color32::from_gray(80));
        ui.add_space(10.0);
        ui.vertical(|ui| {
            ui.label(
                RichText::new(lang.pick(
                    "ライセンス確認",
                    "License review",
                    "许可确认",
                    "라이선스 확인",
                ))
                .size(19.0)
                .strong(),
            );
            ui.label(RichText::new(&review.model_name).size(15.0));
            ui.label(
                RichText::new(match review.generation {
                    crate::import::VrmGeneration::Vrm0 => "VRM 0.x",
                    crate::import::VrmGeneration::Vrm1 => "VRM 1.0",
                })
                .small()
                .weak(),
            );
        });
    });
}

fn review_sheet_details(ui: &mut Ui, review: &VrmLicenseReview, lang: UiLanguage) {
    review_sheet_group(
        ui,
        lang.pick("モデル情報", "Model information", "模型信息", "모델 정보"),
    );
    review_field(
        ui,
        lang.pick("バージョン", "Version", "版本", "버전"),
        review.version.as_deref(),
        lang,
    );
    let authors = joined_values(&review.authors);
    review_field(
        ui,
        lang.pick("作者", "Author", "作者", "제작자"),
        authors.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("連絡先", "Contact", "联系方式", "연락처"),
        review.contact_information.as_deref(),
        lang,
    );
    let references = joined_values(&review.references);
    review_field(
        ui,
        lang.pick("参照", "Reference", "参考", "참조"),
        references.as_deref(),
        lang,
    );

    review_sheet_group(
        ui,
        lang.pick("利用条件", "Usage permission", "使用条件", "이용 조건"),
    );
    review_field(
        ui,
        lang.pick(
            "アバター利用",
            "Avatar permission",
            "虚拟形象使用",
            "아바타 이용",
        ),
        review.avatar_permission.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("暴力表現", "Violent", "暴力表现", "폭력 표현"),
        review.allow_violent_usage.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("性的表現", "Sexual", "性表现", "성적 표현"),
        review.allow_sexual_usage.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("商用利用", "Commercial", "商业用途", "상업적 이용"),
        review.commercial_usage.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("その他の許諾", "Other permission", "其他许可", "기타 허가"),
        review.other_permission_url.as_deref(),
        lang,
    );

    review_sheet_group(
        ui,
        lang.pick(
            "配布・改変ライセンス",
            "Distribution & modification",
            "分发与修改许可",
            "배포·개조 라이선스",
        ),
    );
    review_field(
        ui,
        lang.pick("改変", "Modification", "修改", "개조"),
        review.modification_license.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("ライセンス名", "License name", "许可名称", "라이선스 이름"),
        review.license_name.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick("ライセンスURL", "License URL", "许可URL", "라이선스 URL"),
        review.license_url.as_deref(),
        lang,
    );
    review_field(
        ui,
        lang.pick(
            "その他ライセンスURL",
            "Other license URL",
            "其他许可URL",
            "기타 라이선스 URL",
        ),
        review.other_license_url.as_deref(),
        lang,
    );
}

fn review_sheet_group(ui: &mut Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(title).size(14.0).strong());
    ui.add_space(4.0);
}

fn review_field(ui: &mut Ui, label: &str, value: Option<&str>, lang: UiLanguage) {
    ui.horizontal_top(|ui| {
        ui.add_sized(
            [150.0, 18.0],
            egui::Label::new(RichText::new(label).weak()).truncate(),
        );
        match value.filter(|value| !value.is_empty()) {
            Some(value) if is_external_url(value) => {
                ui.hyperlink_to(value, value);
            }
            Some(value) => {
                ui.label(value);
            }
            None => {
                ui.label(
                    RichText::new(lang.pick("未指定", "Not specified", "未指定", "미지정")).weak(),
                );
            }
        }
    });
}

fn joined_values(values: &[String]) -> Option<String> {
    (!values.is_empty()).then(|| values.join(", "))
}

fn is_external_url(value: &str) -> bool {
    value.starts_with("https://") || value.starts_with("http://")
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

fn camera_controls(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    let selected_label = vm
        .camera
        .selected_index
        .and_then(|index| vm.camera.available_cameras.get(index))
        .map(|camera| camera.name.as_str())
        .unwrap_or_else(|| lang.pick("カメラを選択", "Select camera", "选择摄像头", "카메라 선택"));
    let can_change = matches!(vm.lifecycle, AppLifecycle::Idle | AppLifecycle::Failed);
    ui.add_enabled_ui(can_change, |ui| {
        egui::ComboBox::from_id_salt("studio_camera_device")
            .selected_text(selected_label)
            .width(ui.available_width().min(360.0))
            .show_ui(ui, |ui| {
                for (index, camera) in vm.camera.available_cameras.iter().enumerate() {
                    if ui
                        .selectable_label(vm.camera.selected_index == Some(index), &camera.name)
                        .clicked()
                    {
                        state.emit(UiAction::SelectCamera { index });
                    }
                }
            });
    });
    if !can_change {
        ui.label(lang.pick(
            "カメラを変更するには、トラッキングを停止してください。",
            "Stop tracking before changing the camera.",
            "更换摄像头前请停止跟踪。",
            "카메라를 변경하려면 트래킹을 중지하세요.",
        ));
    }
    if ui
        .button(lang.pick(
            "カメラを再検出",
            "Refresh cameras",
            "重新检测摄像头",
            "카메라 새로 고침",
        ))
        .clicked()
    {
        state.emit(UiAction::RefreshCameras);
    }
    if vm.camera.available_cameras.is_empty() {
        ui.label(lang.pick(
            "カメラが見つかりません。接続とOSのカメラ権限を確認してください。",
            "No camera found. Check the connection and OS camera permission.",
            "未找到摄像头。请检查连接及系统摄像头权限。",
            "카메라를 찾지 못했습니다. 연결과 OS 카메라 권한을 확인하세요.",
        ));
    }
}

fn overview(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    dialog_active: bool,
    lang: UiLanguage,
) {
    ui.label(lang.pick(
        "アバターを読み込み、カメラを選ぶとトラッキングが自動で始まります。",
        "Load an avatar and select a camera; tracking starts automatically.",
        "加载虚拟形象并选择摄像头后，将自动开始跟踪。",
        "아바타를 불러오고 카메라를 선택하면 트래킹이 자동으로 시작됩니다.",
    ));
    section(
        ui,
        lang.pick(
            "1. アバターを読み込む",
            "1. Load an avatar",
            "1. 加载虚拟形象",
            "1. 아바타 불러오기",
        ),
        |ui| import_controls(ui, vm, state, dialog_active, lang),
    );
    section(
        ui,
        lang.pick(
            "2. カメラを選ぶ",
            "2. Choose a camera",
            "2. 选择摄像头",
            "2. 카메라 선택",
        ),
        |ui| {
            camera_controls(ui, vm, state, lang);
            ui.add_space(8.0);
            ui.label(lang.pick(
                "カメラ映像は非表示です。メニューを開くだけでは表示されません。",
                "Camera preview is hidden. Opening a menu never reveals it.",
                "摄像头预览默认隐藏，打开菜单不会显示影像。",
                "카메라 미리 보기는 숨겨져 있습니다. 메뉴를 열어도 영상이 표시되지 않습니다.",
            ));
            if ui
                .button(lang.pick(
                    "カメラ調整へ",
                    "Camera settings",
                    "摄像头设置",
                    "카메라 설정",
                ))
                .clicked()
            {
                state.emit(UiAction::SwitchPane(Pane::Camera));
            }
        },
    );
    section(
        ui,
        lang.pick(
            "3. 動きと出力を確認する",
            "3. Check motion and output",
            "3. 检查动作与输出",
            "3. 움직임과 출력 확인",
        ),
        |ui| {
            ui.label(lang.pick("アバターとカメラが揃うと自動で始まります。動きは常時表示のアバタープレビューで確認できます。停止後は上部のボタンで再開できます。", "Tracking starts automatically once the avatar and camera are ready. Check motion in the persistent avatar preview; use the toolbar button to resume after stopping.", "虚拟形象与摄像头就绪后将自动开始。可在常驻预览中确认动作，停止后可用顶部按钮重新开始。", "아바타와 카메라가 준비되면 자동으로 시작됩니다. 항상 표시되는 미리 보기에서 움직임을 확인하고, 중지 후에는 상단 버튼으로 다시 시작할 수 있습니다."));
            ui.horizontal_wrapped(|ui| {
                if ui.button(page_title(Pane::Calibration, lang)).clicked() {
                    state.emit(UiAction::SwitchPane(Pane::Calibration));
                }
                if ui
                    .button(lang.pick("出力を設定", "Configure output", "设置输出", "출력 설정"))
                    .clicked()
                {
                    state.emit(UiAction::SwitchPane(Pane::NdiOutput));
                }
            });
        },
    );
    expression_status_section(ui, vm, state, lang);
    render_rich_look_controls(ui, vm, state, lang);
}

fn avatar_page(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    mirror: AvatarMotionMirror,
    dialog_active: bool,
    lang: UiLanguage,
) {
    section(
        ui,
        lang.pick("モデル", "Model", "模型", "모델"),
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
    section(
        ui,
        lang.pick(
            "表示と構図",
            "Display and framing",
            "显示与构图",
            "표시 및 구도",
        ),
        |ui| {
            let mut reflected = mirror.is_enabled();
            if ui
                .checkbox(
                    &mut reflected,
                    lang.pick(
                        "アバターの動きを左右反転",
                        "Mirror avatar motion",
                        "镜像虚拟形象动作",
                        "아바타 움직임 좌우 반전",
                    ),
                )
                .changed()
            {
                state.emit(UiAction::ToggleAvatarMotionMirror);
            }
            if ui
                .add_enabled(
                    vm.can_reset_camera(),
                    egui::Button::new(lang.pick(
                        "アバターの画角をリセット",
                        "Reset avatar framing",
                        "重置虚拟形象构图",
                        "아바타 구도 초기화",
                    )),
                )
                .clicked()
            {
                state.emit(UiAction::ResetAvatarCamera);
            }
            ui.label(lang.pick("F1でアバターのみを表示。左ドラッグで回転、右ドラッグで移動、ホイールでズームします。", "Use F1 for avatar-only view. Left-drag to orbit, right-drag to pan, and scroll to zoom.", "按F1仅显示虚拟形象。左键拖动旋转，右键拖动平移，滚轮缩放。", "F1로 아바타만 표시합니다. 왼쪽 드래그로 회전, 오른쪽 드래그로 이동, 휠로 확대·축소합니다."));
        },
    );
    if vm.avatar.imported_model.is_some() {
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
                        egui::Slider::new(&mut profile.shoulder_follow_weight, 0.0..=1.0).text(
                            lang.pick("肩の追従", "Shoulder follow", "肩部跟随", "어깨 추종"),
                        ),
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
                    state.emit(UiAction::ResetArmPoseProfile);
                }
            },
        );
    }
}

fn camera_page(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    preview: &PreviewState,
    landmarks: &PreviewLandmarkState,
    texture: Option<TextureId>,
    lang: UiLanguage,
) {
    section(
        ui,
        lang.pick("入力カメラ", "Input camera", "输入摄像头", "입력 카메라"),
        |ui| camera_controls(ui, vm, state, lang),
    );
    section(
        ui,
        lang.pick(
            "カメラ映像の確認",
            "Check camera preview",
            "查看摄像头预览",
            "카메라 영상 확인",
        ),
        |ui| {
            match state.camera_consent {
                CameraPreviewConsent::Hidden => {
                    ui.label(lang.pick(
                        "カメラ映像は非表示です。トラッキングとは独立した表示設定です。",
                        "Camera pixels are hidden. Preview visibility is independent of tracking.",
                        "摄像头影像已隐藏。预览显示与跟踪功能相互独立。",
                        "카메라 영상은 숨겨져 있습니다. 미리 보기 표시는 트래킹과 별개입니다.",
                    ));
                    if ui
                        .button(lang.pick(
                            "カメラ映像を確認…",
                            "Preview camera…",
                            "查看摄像头影像…",
                            "카메라 영상 확인…",
                        ))
                        .clicked()
                    {
                        state.preview_event(CameraPreviewEvent::Request);
                    }
                }
                CameraPreviewConsent::Confirming => {
                    ui.label(
                        RichText::new(lang.pick(
                            "このウインドウに実際のカメラ映像を表示します。",
                            "This will show the actual camera image in this window.",
                            "此操作将在本窗口显示真实摄像头影像。",
                            "이 창에 실제 카메라 영상이 표시됩니다.",
                        ))
                        .strong(),
                    );
                    ui.label(lang.pick("デスクトップ／ウインドウキャプチャで配信中の場合、顔や部屋も配信に映ります。NDI出力には含まれません。", "Desktop or window capture can broadcast your face and room. NDI output will not include them.", "若正在通过桌面或窗口捕获直播，您的脸部和房间也会被播出。NDI输出不包含这些影像。", "데스크톱이나 창 캡처로 방송 중이면 얼굴과 방도 방송에 보입니다. NDI 출력에는 포함되지 않습니다."));
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .button(lang.pick("キャンセル", "Cancel", "取消", "취소"))
                            .clicked()
                        {
                            state.preview_event(CameraPreviewEvent::Hide);
                        }
                        if ui
                            .button(lang.pick(
                                "確認して映像を表示",
                                "Show camera image",
                                "确认并显示影像",
                                "확인 후 영상 표시",
                            ))
                            .clicked()
                        {
                            state.preview_event(CameraPreviewEvent::Confirm);
                        }
                    });
                }
                CameraPreviewConsent::Visible => {
                    if ui
                        .button(lang.pick(
                            "カメラ映像を隠す (Esc)",
                            "Hide camera (Esc)",
                            "隐藏摄像头影像 (Esc)",
                            "카메라 영상 숨기기 (Esc)",
                        ))
                        .clicked()
                    {
                        state.preview_event(CameraPreviewEvent::Hide);
                    }
                    // Check after the Hide button: do not emit even one more image mesh.
                    if state.camera_consent == CameraPreviewConsent::Visible {
                        if let Some(texture) = texture {
                            let width = ui.available_width().min(640.0);
                            let size = vec2(width, width * 9.0 / 16.0);
                            let uv = if preview.mirrored {
                                egui::Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0))
                            } else {
                                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0))
                            };
                            let rect = ui
                                .add(
                                    egui::Image::from_texture((texture, size))
                                        .uv(uv)
                                        .corner_radius(CornerRadius::same(8)),
                                )
                                .rect;
                            if let Some(snapshot) = landmarks.latest_fresh_at(monotonic_now()) {
                                let painter = ui.painter().with_clip_rect(rect);
                                for point in snapshot.landmarks.iter() {
                                    if point.x.is_finite()
                                        && point.y.is_finite()
                                        && (0.0..=1.0).contains(&point.x)
                                        && (0.0..=1.0).contains(&point.y)
                                    {
                                        let x = if preview.mirrored {
                                            1.0 - point.x
                                        } else {
                                            point.x
                                        };
                                        painter.circle_filled(
                                            egui::pos2(
                                                rect.left() + x * rect.width(),
                                                rect.top() + point.y * rect.height(),
                                            ),
                                            1.5,
                                            Color32::YELLOW,
                                        );
                                    }
                                }
                            }
                        } else {
                            ui.label(lang.pick("映像を待っています。停止中の場合は、上部でトラッキングを開始してください。", "Waiting for frames. Start tracking in the toolbar if it is stopped.", "正在等待影像。如已停止，请在顶部开始跟踪。", "영상을 기다리고 있습니다. 중지 상태라면 상단에서 트래킹을 시작하세요."));
                        }
                    }
                }
            }
            ui.add_space(8.0);
            let mut mirrored = preview.mirrored;
            if ui
                .checkbox(
                    &mut mirrored,
                    lang.pick(
                        "カメラプレビューを左右反転",
                        "Mirror camera preview",
                        "镜像摄像头预览",
                        "카메라 미리 보기 좌우 반전",
                    ),
                )
                .changed()
            {
                state.emit(UiAction::ToggleMirror);
            }
            ui.label(
                RichText::new(lang.pick(
                    "別の画面への移動、設定を閉じる操作、Escで再び非表示になります。",
                    "Changing pages, closing settings, or pressing Esc hides it again.",
                    "切换页面、关闭设置或按Esc后，影像会再次隐藏。",
                    "다른 화면으로 이동하거나 설정을 닫거나 Esc를 누르면 다시 숨겨집니다.",
                ))
                .small()
                .weak(),
            );
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
                let value = if vm.calibration.samples_target > 0 {
                    vm.calibration.samples_collected as f32 / vm.calibration.samples_target as f32
                } else {
                    0.0
                };
                ui.add(egui::ProgressBar::new(value.clamp(0.0, 1.0)).text(format!(
                    "{} / {}",
                    vm.calibration.samples_collected, vm.calibration.samples_target
                )));
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
                        "先にトラッキングを開始してください。",
                        "Start tracking first.",
                        "请先开始跟踪。",
                        "먼저 트래킹을 시작하세요.",
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
/// The strength scales the whole look, so at 0 the scene is the standard
/// display even while the switch is on.
fn render_rich_look_controls(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    let switch = lang.pick("リッチ表示", "Enhanced look", "增强显示", "고급 렌더링");
    section(ui, switch, |ui| {
        let mut enabled = vm.look.enabled;
        if ui.checkbox(&mut enabled, switch).changed() {
            state.emit(UiAction::ChangeRichLook(
                crate::actions::RichLookChange::Enabled(enabled),
            ));
            state.emit(UiAction::SaveRichLook);
        }
        let mut percent = vm.look.strength * 100.0;
        let slider = ui.add_enabled(
            enabled,
            egui::Slider::new(&mut percent, 0.0..=100.0)
                .suffix("%")
                .text(lang.pick("効果の強さ", "Effect strength", "效果强度", "효과 강도")),
        );
        if slider.changed() {
            state.emit(UiAction::ChangeRichLook(
                crate::actions::RichLookChange::Strength(percent / 100.0),
            ));
        }
        if slider.drag_stopped() || (slider.changed() && !slider.dragged()) {
            state.emit(UiAction::SaveRichLook);
        }
        ui.add_space(4.0);
        ui.label(lang.pick(
            "照明と質感をまとめて調整します。VRMファイルは変更しません。",
            "Adjust lighting and materials together. Your VRM file is not modified.",
            "同时调整灯光和材质，不修改 VRM 文件。",
            "조명과 재질을 함께 조정합니다. VRM 파일은 변경하지 않습니다.",
        ));
    });
}

/// The four-language label of a material role; `None` is the "Auto" entry.
fn material_role_label(
    lang: UiLanguage,
    role: Option<vtuber_avatar::MaterialRole>,
) -> &'static str {
    match role {
        None => lang.pick("自動", "Auto", "自动", "자동"),
        Some(vtuber_avatar::MaterialRole::General) => lang.pick("一般", "General", "通用", "일반"),
        Some(vtuber_avatar::MaterialRole::Face) => lang.pick("顔", "Face", "面部", "얼굴"),
        Some(vtuber_avatar::MaterialRole::Skin) => lang.pick("肌", "Skin", "皮肤", "피부"),
        Some(vtuber_avatar::MaterialRole::Hair) => lang.pick("髪", "Hair", "头发", "머리카락"),
        Some(vtuber_avatar::MaterialRole::Fabric) => lang.pick("布", "Fabric", "布料", "천"),
        Some(vtuber_avatar::MaterialRole::Metal) => lang.pick("金属", "Metal", "金属", "금속"),
        Some(vtuber_avatar::MaterialRole::Eye) => lang.pick("目", "Eyes", "眼睛", "눈"),
    }
}

/// Per-material role selection inside the look's detail section.
///
/// The list only names the loaded model's materials and their current
/// selection; every numeric shader detail stays hidden.
fn render_material_role_controls(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    lang: UiLanguage,
) {
    if vm.look_materials.is_empty() {
        return;
    }
    section(
        ui,
        lang.pick("材質の役割", "Material roles", "材质角色", "재질 역할"),
        |ui| {
            ui.label(lang.pick(
                "各材質の役割を選ぶと、光沢や陰影の調整がその役割に合います。「自動」は材質名からの判定です。",
                "Pick each material's role so the gloss and shading fit it. \"Auto\" decides from the material name.",
                "为每种材质选择角色，让光泽与阴影更贴合。“自动”根据材质名判断。",
                "각 재질의 역할을 선택하면 광택과 음영이 역할에 맞게 조정됩니다. \"자동\"은 재질 이름으로 판단합니다.",
            ));
            for entry in &vm.look_materials {
                ui.horizontal(|ui| {
                    let label = if entry.name.is_empty() {
                        format!("#{}", entry.material_index)
                    } else {
                        entry.name.clone()
                    };
                    ui.label(label);
                    let current = material_role_label(lang, entry.selected);
                    egui::ComboBox::from_id_salt(("material_role", entry.material_index))
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for role in [
                                None,
                                Some(vtuber_avatar::MaterialRole::General),
                                Some(vtuber_avatar::MaterialRole::Face),
                                Some(vtuber_avatar::MaterialRole::Skin),
                                Some(vtuber_avatar::MaterialRole::Hair),
                                Some(vtuber_avatar::MaterialRole::Fabric),
                                Some(vtuber_avatar::MaterialRole::Metal),
                                Some(vtuber_avatar::MaterialRole::Eye),
                            ] {
                                if ui
                                    .selectable_label(
                                        entry.selected == role,
                                        material_role_label(lang, role),
                                    )
                                    .clicked()
                                {
                                    state.emit(UiAction::SetMaterialRole {
                                        material_index: entry.material_index,
                                        selected: role,
                                    });
                                }
                            }
                        });
                });
            }
        },
    );
}

fn settings_page(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    render_rich_look_controls(ui, vm, state, lang);
    render_material_role_controls(ui, vm, state, lang);
    expression_settings_section(ui, vm, state, lang);
    section(
        ui,
        lang.pick("表示言語", "Display language", "显示语言", "표시 언어"),
        |ui| {
            let languages = [
                (
                    UiLanguage::Ja,
                    lang.pick("日本語", "Japanese", "日语", "일본어"),
                ),
                (UiLanguage::En, lang.pick("英語", "English", "英语", "영어")),
                (
                    UiLanguage::Zh,
                    lang.pick(
                        "中国語（簡体字）",
                        "Chinese (Simplified)",
                        "简体中文",
                        "중국어 (간체)",
                    ),
                ),
                (
                    UiLanguage::Ko,
                    lang.pick("韓国語", "Korean", "韩语", "한국어"),
                ),
            ];
            for (language, label) in languages {
                if ui.radio(lang == language, label).clicked() && lang != language {
                    state.emit(UiAction::SetLanguage(language));
                }
            }
            ui.label(lang.pick("初期設定は日本語です。変更はすぐに反映され、再起動後も維持されます。", "Japanese is the initial language. Changes apply immediately and persist across restarts.", "初始语言为日语。更改立即生效，重启后仍会保留。", "초기 언어는 일본어입니다. 변경 사항은 즉시 적용되며 재시작 후에도 유지됩니다."));
        },
    );
}

/// Translates standard presets and keeps the exact runtime ID visible.
fn expression_entry_name(entry: &ExpressionEntryViewModel, lang: UiLanguage) -> String {
    // Only expressions classified as standard presets get a translated label.
    // A custom expression that happens to be named `happy` keeps its author
    // string and is never displayed as the standard 喜 label.
    let standard = match entry.kind {
        ExpressionKind::EmotionalPreset | ExpressionKind::Neutral => match entry.id.as_str() {
            "happy" => Some(lang.pick("喜", "Happy", "喜悦", "기쁨")),
            "angry" => Some(lang.pick("怒", "Angry", "愤怒", "분노")),
            "sad" => Some(lang.pick("哀", "Sad", "悲伤", "슬픔")),
            "relaxed" => Some(lang.pick("楽", "Relaxed", "放松", "편안함")),
            "surprised" => Some(lang.pick("驚き", "Surprised", "惊讶", "놀람")),
            "neutral" => Some(lang.pick("通常", "Neutral", "自然", "중립")),
            _ => None,
        },
        ExpressionKind::Custom | ExpressionKind::Tracking => None,
    };
    let base = match standard {
        Some(label) => format!("{label} ({})", entry.id),
        None => entry.source_name.clone(),
    };
    if entry.kind == ExpressionKind::Tracking {
        format!(
            "{} [{}]",
            base,
            lang.pick("追跡用", "Tracking", "跟踪用", "트래킹용")
        )
    } else {
        base
    }
}

/// Describes availability as a disabled-candidate suffix.
fn expression_availability_suffix(
    availability: &ExpressionAvailability,
    lang: UiLanguage,
) -> String {
    match availability {
        ExpressionAvailability::Ready => String::new(),
        ExpressionAvailability::Empty => {
            format!(" — {}", lang.pick("未定義", "Empty", "空定义", "비어 있음"))
        }
        ExpressionAvailability::Unresolved { reason } => format!(
            " — {}: {reason}",
            lang.pick("利用できません", "Unavailable", "不可用", "사용할 수 없음")
        ),
        ExpressionAvailability::Unsupported { reason } => format!(
            " — {}: {reason}",
            lang.pick("利用できません", "Unavailable", "不可用", "사용할 수 없음")
        ),
    }
}

fn expression_settings_section(
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    lang: UiLanguage,
) {
    section(
        ui,
        lang.pick(
            "表情・キー割り当て",
            "Expressions & Key Bindings",
            "表情与按键绑定",
            "표정 및 키 할당",
        ),
        |ui| {
            ui.label(lang.pick(
                "ESCまたはスペースキーで標準の表情に戻ります",
                "Press Escape or Space to return to the standard expression.",
                "按 Esc 或空格键可恢复为标准表情。",
                "Esc 또는 스페이스 키를 누르면 표준 표정으로 돌아갑니다.",
            ));
            ui.add_space(6.0);
            if !vm.expression.has_catalog {
                let message = if vm.avatar.imported_model.is_some() {
                    lang.pick(
                        "選択できる表情がありません",
                        "No selectable expressions.",
                        "没有可选择的表情。",
                        "선택할 수 있는 표정이 없습니다.",
                    )
                } else {
                    lang.pick(
                        "モデルを読み込んでください",
                        "Load a model to continue.",
                        "请先加载模型。",
                        "먼저 모델을 불러오세요.",
                    )
                };
                ui.label(message);
                return;
            }
            // Every assignment action carries the exact model/generation this
            // snapshot was built for.
            let target = vm.expression.model_id.clone().zip(vm.expression.generation);
            ui.label(lang.pick(
                "最大36個のキーに割り当てられます。自動割り当てに含まれなかった表情も、ここで選択できます。",
                "Up to 36 keys can be assigned. You can also choose expressions that were not assigned automatically.",
                "最多可绑定36个按键。未被自动分配的表情也可以在这里选择。",
                "최대 36개 키에 할당할 수 있습니다. 자동 할당되지 않은 표정도 여기에서 선택할 수 있습니다.",
            ));
            ui.add_space(6.0);
            for row in &vm.expression.bindings {
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [28.0, 18.0],
                        egui::Label::new(RichText::new(row.key.label()).strong()),
                    );
                    let selected_label = row
                        .expression
                        .as_deref()
                        .and_then(|id| {
                            vm.expression
                                .entries
                                .iter()
                                .find(|entry| entry.id == id)
                                .map(|entry| expression_entry_name(entry, lang))
                        })
                        .unwrap_or_else(|| {
                            row.expression.clone().unwrap_or_else(|| {
                                lang.pick("未割り当て", "Unassigned", "未分配", "할당되지 않음")
                                    .to_owned()
                            })
                        });
                    let combo = egui::ComboBox::from_id_salt(("expression_key", row.key.label()))
                        .selected_text(selected_label)
                        .width(ui.available_width().min(320.0))
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    row.expression.is_none(),
                                    lang.pick(
                                        "未割り当て",
                                        "Unassigned",
                                        "未分配",
                                        "할당되지 않음",
                                    ),
                                )
                                .clicked()
                                && row.expression.is_some()
                                && let Some((model_id, generation)) = target.clone()
                            {
                                state.emit(UiAction::AssignExpressionKey {
                                    model_id,
                                    generation,
                                    key: row.key,
                                    expression: None,
                                });
                            }
                            // A saved binding that no longer exists in the
                            // current catalog stays visible, never silently
                            // replaced with a different expression.
                            if let Some(current) = &row.expression
                                && !vm
                                    .expression
                                    .entries
                                    .iter()
                                    .any(|entry| &entry.id == current)
                            {
                                ui.add_enabled(
                                    false,
                                    egui::Button::selectable(
                                        true,
                                        format!(
                                            "{current} — {}",
                                            lang.pick(
                                                "利用できません",
                                                "Unavailable",
                                                "不可用",
                                                "사용할 수 없음"
                                            )
                                        ),
                                    ),
                                );
                            }
                            for entry in &vm.expression.entries {
                                let label = format!(
                                    "{}{}",
                                    expression_entry_name(entry, lang),
                                    expression_availability_suffix(&entry.availability, lang)
                                );
                                let selected = row.expression.as_deref() == Some(entry.id.as_str());
                                if ui
                                    .add_enabled(
                                        entry.availability.is_ready(),
                                        egui::Button::selectable(selected, label),
                                    )
                                    .clicked()
                                    && let Some((model_id, generation)) = target.clone()
                                {
                                    state.emit(UiAction::AssignExpressionKey {
                                        model_id,
                                        generation,
                                        key: row.key,
                                        expression: Some(entry.id.clone()),
                                    });
                                }
                            }
                        });
                    let _ = combo;
                    if row.selected {
                        ui.label(
                            RichText::new(lang.pick("選択中", "Selected", "已选中", "선택됨"))
                                .small()
                                .weak(),
                        );
                    }
                });
            }
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button(lang.pick("表情を解除", "Clear Expression", "取消表情", "표정 해제"))
                    .clicked()
                    && let Some(generation) = vm.expression.generation
                {
                    state.emit(UiAction::ClearManualExpression { generation });
                }
                if ui
                    .button(lang.pick(
                        "初期割り当てに戻す",
                        "Restore Default Bindings",
                        "恢复默认绑定",
                        "기본 키 할당 복원",
                    ))
                    .clicked()
                    && let Some((model_id, generation)) = target.clone()
                {
                    state.emit(UiAction::ResetExpressionBindings {
                        model_id,
                        generation,
                    });
                }
            });
            ui.label(
                RichText::new(lang.pick(
                    "キーを押すと表情を選択し、同じキーをもう一度押すと解除します。文字入力中やダイアログ表示中は反応しません。",
                    "Press a key to select an expression. Press the same key again to clear it. Expression keys are inactive while typing or using dialogs.",
                    "按键可选择表情，再次按下同一按键可取消。输入文字或使用对话框时，表情按键不会生效。",
                    "키를 눌러 표정을 선택하고 같은 키를 다시 누르면 해제합니다. 텍스트 입력 중이거나 대화상자를 사용하는 동안에는 표정 키가 작동하지 않습니다.",
                ))
                .small()
                .weak(),
            );
        },
    );
}

/// Studio list of assigned expressions. Clicking runs the same toggle action
/// as the physical key.
fn expression_status_section(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    if !vm.expression.has_catalog {
        return;
    }
    let Some(generation) = vm.expression.generation else {
        return;
    };
    section(
        ui,
        lang.pick("表情キー", "Expression keys", "表情按键", "표정 키"),
        |ui| {
            let mut any = false;
            for row in &vm.expression.bindings {
                let Some(id) = &row.expression else {
                    continue;
                };
                any = true;
                let name = vm
                    .expression
                    .entries
                    .iter()
                    .find(|entry| &entry.id == id)
                    .map_or_else(|| id.clone(), |entry| expression_entry_name(entry, lang));
                let text = if row.selected {
                    format!(
                        "● {}  {} ({})",
                        row.key.label(),
                        name,
                        lang.pick("選択中", "Selected", "已选中", "선택됨")
                    )
                } else {
                    format!("{}  {name}", row.key.label())
                };
                if ui
                    .add(egui::Button::selectable(row.selected, text))
                    .clicked()
                {
                    state.emit(UiAction::ToggleExpressionKey {
                        generation,
                        key: row.key,
                    });
                }
            }
            if !any {
                ui.label(
                    RichText::new(lang.pick(
                        "割り当て済みの表情はありません。",
                        "No expressions are assigned.",
                        "尚未分配表情。",
                        "할당된 표정이 없습니다.",
                    ))
                    .weak(),
                );
            }
        },
    );
}

/// Translates an egui physical key into the fixed expression key, if bound.
#[must_use]
pub(crate) fn expression_key_from_egui(key: egui::Key) -> Option<ExpressionKey> {
    use egui::Key;
    Some(match key {
        Key::Num1 => ExpressionKey::Digit1,
        Key::Num2 => ExpressionKey::Digit2,
        Key::Num3 => ExpressionKey::Digit3,
        Key::Num4 => ExpressionKey::Digit4,
        Key::Num5 => ExpressionKey::Digit5,
        Key::Num6 => ExpressionKey::Digit6,
        Key::Num7 => ExpressionKey::Digit7,
        Key::Num8 => ExpressionKey::Digit8,
        Key::Num9 => ExpressionKey::Digit9,
        Key::Num0 => ExpressionKey::Digit0,
        Key::Q => ExpressionKey::KeyQ,
        Key::W => ExpressionKey::KeyW,
        Key::E => ExpressionKey::KeyE,
        Key::R => ExpressionKey::KeyR,
        Key::T => ExpressionKey::KeyT,
        Key::Y => ExpressionKey::KeyY,
        Key::U => ExpressionKey::KeyU,
        Key::I => ExpressionKey::KeyI,
        Key::O => ExpressionKey::KeyO,
        Key::P => ExpressionKey::KeyP,
        Key::A => ExpressionKey::KeyA,
        Key::S => ExpressionKey::KeyS,
        Key::D => ExpressionKey::KeyD,
        Key::F => ExpressionKey::KeyF,
        Key::G => ExpressionKey::KeyG,
        Key::H => ExpressionKey::KeyH,
        Key::J => ExpressionKey::KeyJ,
        Key::K => ExpressionKey::KeyK,
        Key::L => ExpressionKey::KeyL,
        Key::Z => ExpressionKey::KeyZ,
        Key::X => ExpressionKey::KeyX,
        Key::C => ExpressionKey::KeyC,
        Key::V => ExpressionKey::KeyV,
        Key::B => ExpressionKey::KeyB,
        Key::N => ExpressionKey::KeyN,
        Key::M => ExpressionKey::KeyM,
        _ => return None,
    })
}

/// Emits expression toggles for new, unmodified physical key-downs, and a
/// manual clear for unmodified Escape/Space.
///
/// The gate mirrors the product contract: window focused, avatar ready, no
/// text edit / IME / popup / modal / file dialog owning the keyboard, no
/// modifiers, and no key repeat. Multiple key-downs are emitted in event
/// order; the avatar runtime's request system applies them in the same order.
pub(crate) fn expression_key_input(
    ctx: &egui::Context,
    vm: &UiViewModel,
    state: &mut UiState,
    dialog_active: bool,
) {
    if !vm.avatar.is_ready
        || vm.avatar.lifecycle != AvatarLifecycleState::Ready
        || dialog_active
        || vm.avatar_import_review.review.is_some()
    {
        return;
    }
    if ctx.egui_wants_keyboard_input() || ctx.any_popup_open() {
        return;
    }
    let Some(generation) = vm.expression.generation else {
        return;
    };
    let actions: Vec<UiAction> = ctx.input(|input| {
        if !input.focused {
            return Vec::new();
        }
        input
            .events
            .iter()
            .filter_map(|event| match event {
                egui::Event::Key {
                    physical_key: Some(key),
                    pressed: true,
                    repeat: false,
                    modifiers,
                    ..
                } if modifiers.is_none() => match key {
                    egui::Key::Escape | egui::Key::Space => {
                        Some(UiAction::ClearManualExpression { generation })
                    }
                    key => expression_key_from_egui(*key)
                        .map(|key| UiAction::ToggleExpressionKey { generation, key }),
                },
                _ => None,
            })
            .collect()
    });
    for action in actions {
        state.emit(action);
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_layout_keeps_navigation_and_a_persistent_monitor() {
        assert!(!use_sidebar(800.0));
        assert!(use_sidebar(1280.0));
        assert_eq!(NAVIGATION.len(), 7);
        assert!(NAVIGATION.contains(&Pane::Studio));
        assert!(NAVIGATION.contains(&Pane::Settings));
        assert!(!NAVIGATION.contains(&Pane::Preview));
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
                    None,
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
    fn every_destination_has_a_label_in_all_four_languages() {
        for pane in NAVIGATION {
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

    fn review_fixture() -> VrmLicenseReview {
        VrmLicenseReview {
            generation: crate::import::VrmGeneration::Vrm1,
            model_name: "Test model".to_string(),
            version: None,
            authors: vec!["Author".to_string()],
            contact_information: None,
            references: Vec::new(),
            avatar_permission: Some("everyone".to_string()),
            allow_violent_usage: Some("false".to_string()),
            allow_sexual_usage: Some("false".to_string()),
            commercial_usage: Some("personalNonProfit".to_string()),
            other_permission_url: None,
            modification_license: Some("prohibited".to_string()),
            license_name: None,
            license_url: None,
            other_license_url: None,
            source_path: std::path::PathBuf::from("model.vrm"),
        }
    }

    #[test]
    fn license_review_modal_escape_emits_cancel() {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let review = review_fixture();
        // Frame 1 registers the modal; frame 2 receives the Escape.
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            avatar_import_review_modal(ui.ctx(), &review, false, &mut state, UiLanguage::Ja);
        });
        let mut input = egui::RawInput::default();
        input.events.push(egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run_ui(input, |ui| {
            avatar_import_review_modal(ui.ctx(), &review, false, &mut state, UiLanguage::Ja);
        });
        assert!(
            state
                .pending_actions
                .contains(&UiAction::CancelAvatarImportReview)
        );
    }

    #[test]
    fn license_review_modal_does_not_import_while_unchecked() {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let review = review_fixture();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            avatar_import_review_modal(ui.ctx(), &review, false, &mut state, UiLanguage::Ja);
        });
        assert!(state.pending_actions.is_empty());
    }

    fn test_generation() -> vtuber_avatar::AvatarGeneration {
        vtuber_avatar::AvatarGeneration(1)
    }

    fn ready_expression_view_model() -> UiViewModel {
        let mut vm = UiViewModel::default();
        vm.avatar.is_ready = true;
        vm.avatar.lifecycle = AvatarLifecycleState::Ready;
        vm.expression.model_id = Some("model".into());
        vm.expression.generation = Some(test_generation());
        vm
    }

    fn expression_key_event(
        key: egui::Key,
        repeat: bool,
        modifiers: egui::Modifiers,
    ) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: Some(key),
            pressed: true,
            repeat,
            modifiers,
        }
    }

    fn run_expression_key_input(
        vm: &UiViewModel,
        state: &mut UiState,
        input: egui::RawInput,
        dialog_active: bool,
    ) {
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(input, |ctx| {
            expression_key_input(ctx, vm, state, dialog_active)
        });
    }

    fn focused_input(events: Vec<egui::Event>) -> egui::RawInput {
        egui::RawInput {
            focused: true,
            events,
            ..Default::default()
        }
    }

    #[test]
    fn expression_key_input_emits_toggle_even_in_avatar_only_mode() {
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        state.set_controls_open(false);
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            false,
        );
        assert_eq!(
            state.take_actions(),
            vec![UiAction::ToggleExpressionKey {
                generation: test_generation(),
                key: ExpressionKey::Digit1
            }]
        );
    }

    #[test]
    fn expression_key_input_escape_and_space_clear_the_manual_expression() {
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![
                expression_key_event(egui::Key::Escape, false, egui::Modifiers::NONE),
                expression_key_event(egui::Key::Space, false, egui::Modifiers::NONE),
            ]),
            false,
        );
        assert_eq!(
            state.take_actions(),
            vec![
                UiAction::ClearManualExpression {
                    generation: test_generation()
                },
                UiAction::ClearManualExpression {
                    generation: test_generation()
                },
            ]
        );
    }

    #[test]
    fn expression_key_input_ignores_modified_and_repeated_clear_keys() {
        let ctx = egui::Context::default();
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        // Frame 1: Escape goes down once and clears.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Escape,
                false,
                egui::Modifiers::NONE,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert_eq!(state.take_actions().len(), 1);

        // Frame 2: the OS repeats the held Escape. egui marks the event as a
        // repeat, which must not clear again.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Escape,
                true,
                egui::Modifiers::NONE,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert!(
            state.take_actions().is_empty(),
            "OS key repeat never clears again"
        );

        // A modified Space stays with the existing shortcut handling.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Space,
                false,
                egui::Modifiers::SHIFT,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert!(
            state.take_actions().is_empty(),
            "modifiers never clear the expression"
        );
    }

    #[test]
    fn expression_key_input_processes_multiple_keys_in_event_order() {
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![
                expression_key_event(egui::Key::Num1, false, egui::Modifiers::NONE),
                expression_key_event(egui::Key::Q, false, egui::Modifiers::NONE),
            ]),
            false,
        );
        assert_eq!(
            state.take_actions(),
            vec![
                UiAction::ToggleExpressionKey {
                    generation: test_generation(),
                    key: ExpressionKey::Digit1
                },
                UiAction::ToggleExpressionKey {
                    generation: test_generation(),
                    key: ExpressionKey::KeyQ
                },
            ]
        );
    }

    #[test]
    fn expression_key_input_ignores_os_key_repeat_and_key_up() {
        let ctx = egui::Context::default();
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        // Frame 1: the physical key goes down once.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert_eq!(state.take_actions().len(), 1);

        // Frame 2: the OS repeats the held key. egui marks the event as a
        // repeat, which must not toggle again.
        let _ = ctx.run_ui(
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert!(
            state.take_actions().is_empty(),
            "OS key repeat never toggles"
        );

        // Key-up alone never clears.
        let _ = ctx.run_ui(
            focused_input(vec![egui::Event::Key {
                key: egui::Key::Num1,
                physical_key: Some(egui::Key::Num1),
                pressed: false,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }]),
            |ctx| expression_key_input(ctx, &vm, &mut state, false),
        );
        assert!(state.take_actions().is_empty(), "key-up never clears");
    }

    #[test]
    fn expression_key_input_ignores_modifiers_and_unfocused_window() {
        let vm = ready_expression_view_model();

        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::SHIFT,
            )]),
            false,
        );
        assert!(
            state.take_actions().is_empty(),
            "Shift+1 must stay with the existing shortcut handling"
        );

        let mut state = UiState::default();
        let mut input = focused_input(vec![expression_key_event(
            egui::Key::Num1,
            false,
            egui::Modifiers::NONE,
        )]);
        input.focused = false;
        run_expression_key_input(&vm, &mut state, input, false);
        assert!(state.take_actions().is_empty(), "unfocused window is inert");
    }

    #[test]
    fn expression_key_input_ignores_dialog_and_license_modal() {
        let mut vm = ready_expression_view_model();
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            true,
        );
        assert!(state.take_actions().is_empty(), "file dialog owns input");

        vm.avatar_import_review.review = Some(review_fixture());
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            false,
        );
        assert!(state.take_actions().is_empty(), "license modal owns input");
    }

    #[test]
    fn expression_key_input_defers_to_a_focused_text_edit() {
        let ctx = egui::Context::default();
        let vm = ready_expression_view_model();
        let mut state = UiState::default();
        let mut text = String::new();
        // First frame registers and focuses the text edit.
        let _ = ctx.run_ui(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.text_edit_singleline(&mut text).request_focus();
            });
        });
        assert!(ctx.egui_wants_keyboard_input());
        let input = focused_input(vec![expression_key_event(
            egui::Key::Num1,
            false,
            egui::Modifiers::NONE,
        )]);
        let _ = ctx.run_ui(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.text_edit_singleline(&mut text);
            });
            expression_key_input(ctx, &vm, &mut state, false);
        });
        assert!(
            state.take_actions().is_empty(),
            "typing in a text field must not trigger expressions"
        );
    }

    #[test]
    fn expression_key_input_requires_a_ready_avatar() {
        let mut vm = UiViewModel::default();
        let mut state = UiState::default();
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            false,
        );
        assert!(state.take_actions().is_empty());

        vm.avatar.is_ready = true;
        vm.avatar.lifecycle = AvatarLifecycleState::Loading;
        run_expression_key_input(
            &vm,
            &mut state,
            focused_input(vec![expression_key_event(
                egui::Key::Num1,
                false,
                egui::Modifiers::NONE,
            )]),
            false,
        );
        assert!(state.take_actions().is_empty());
    }

    #[test]
    fn expression_ui_strings_exist_in_all_four_languages() {
        let entry = ExpressionEntryViewModel {
            id: "happy".into(),
            source_name: "happy".into(),
            kind: ExpressionKind::EmotionalPreset,
            availability: ExpressionAvailability::Ready,
        };
        let custom = ExpressionEntryViewModel {
            id: "笑顔".into(),
            source_name: "笑顔".into(),
            kind: ExpressionKind::Tracking,
            availability: ExpressionAvailability::Unsupported {
                reason: "test".into(),
            },
        };
        for lang in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            assert!(!expression_entry_name(&entry, lang).is_empty());
            let custom_label = expression_entry_name(&custom, lang);
            assert!(custom_label.contains("笑顔"), "custom names stay verbatim");
            assert!(!expression_availability_suffix(&custom.availability, lang).is_empty());
            assert!(
                expression_availability_suffix(&ExpressionAvailability::Ready, lang).is_empty()
            );
        }
    }

    #[test]
    fn rich_look_view_model_default_matches_the_look_settings() {
        let model = crate::ui_model::RichLookViewModel::default();
        let settings = vtuber_avatar::RichLookSettings::default();
        assert_eq!(model.enabled, settings.enabled);
        assert_eq!(model.strength, settings.strength);
    }

    #[test]
    fn render_rich_look_controls_renders_in_every_language() {
        for lang in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            let ctx = egui::Context::default();
            let mut state = UiState::default();
            let mut vm = UiViewModel::default();
            vm.look.enabled = true;
            vm.look.strength = 0.5;
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                render_rich_look_controls(ui, &vm, &mut state, lang);
            });
            let mut off = vm;
            off.look.enabled = false;
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                render_rich_look_controls(ui, &off, &mut state, lang);
            });
            assert!(state.take_actions().is_empty());
        }
    }

    #[test]
    fn material_role_controls_render_every_entry_and_stay_hidden_without_one() {
        use crate::ui_model::MaterialRoleEntryViewModel;
        for lang in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            let ctx = egui::Context::default();
            let mut state = UiState::default();
            let vm = UiViewModel {
                look_materials: vec![
                    MaterialRoleEntryViewModel {
                        material_index: 0,
                        name: "Face".into(),
                        selected: Some(vtuber_avatar::MaterialRole::Face),
                    },
                    MaterialRoleEntryViewModel {
                        material_index: 3,
                        name: String::new(),
                        selected: None,
                    },
                ],
                ..Default::default()
            };
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                render_material_role_controls(ui, &vm, &mut state, lang);
            });
            assert!(state.take_actions().is_empty());
        }
        // Without a loaded model's materials nothing renders and no section
        // appears on the two-operation normal screen.
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let vm = UiViewModel::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            render_material_role_controls(ui, &vm, &mut state, UiLanguage::Ja);
        });
        assert!(state.take_actions().is_empty());
    }

    #[test]
    fn material_role_labels_cover_every_role_in_four_languages() {
        use vtuber_avatar::MaterialRole;
        let roles = [
            None,
            Some(MaterialRole::General),
            Some(MaterialRole::Face),
            Some(MaterialRole::Skin),
            Some(MaterialRole::Hair),
            Some(MaterialRole::Fabric),
            Some(MaterialRole::Metal),
            Some(MaterialRole::Eye),
        ];
        for lang in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            for (index, role) in roles.iter().enumerate() {
                let label = material_role_label(lang, *role);
                for (other_index, other) in roles.iter().enumerate() {
                    if index != other_index {
                        assert_ne!(
                            label,
                            material_role_label(lang, *other),
                            "{lang:?}: {label} is ambiguous"
                        );
                    }
                }
            }
        }
        assert_eq!(material_role_label(UiLanguage::Ja, None), "自動");
        assert_eq!(
            material_role_label(UiLanguage::Ja, Some(MaterialRole::Face)),
            "顔"
        );
        assert_eq!(material_role_label(UiLanguage::En, None), "Auto");
        assert_eq!(material_role_label(UiLanguage::Zh, None), "自动");
        assert_eq!(material_role_label(UiLanguage::Ko, None), "자동");
    }

    #[test]
    fn expression_settings_section_renders_with_and_without_a_catalog() {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let mut vm = ready_expression_view_model();
        // Without a catalog: the load-model message renders.
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            settings_page(ui, &vm, &mut state, UiLanguage::Ja);
        });

        // With a catalog: a disabled unavailable candidate and a saved
        // unknown binding render without panicking.
        vm.expression.has_catalog = true;
        vm.expression.entries = vec![ExpressionEntryViewModel {
            id: "happy".into(),
            source_name: "happy".into(),
            kind: ExpressionKind::EmotionalPreset,
            availability: ExpressionAvailability::Empty,
        }];
        vm.expression.bindings = vec![crate::ui_model::ExpressionKeyBindingViewModel {
            key: ExpressionKey::Digit1,
            expression: Some("vanished".into()),
            selected: false,
        }];
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            settings_page(ui, &vm, &mut state, UiLanguage::Ja);
        });
    }

    #[test]
    fn expression_key_mapping_covers_the_fixed_36_keys() {
        for key in ExpressionKey::ALL {
            let egui_key = match key {
                ExpressionKey::Digit1 => egui::Key::Num1,
                ExpressionKey::Digit2 => egui::Key::Num2,
                ExpressionKey::Digit3 => egui::Key::Num3,
                ExpressionKey::Digit4 => egui::Key::Num4,
                ExpressionKey::Digit5 => egui::Key::Num5,
                ExpressionKey::Digit6 => egui::Key::Num6,
                ExpressionKey::Digit7 => egui::Key::Num7,
                ExpressionKey::Digit8 => egui::Key::Num8,
                ExpressionKey::Digit9 => egui::Key::Num9,
                ExpressionKey::Digit0 => egui::Key::Num0,
                ExpressionKey::KeyQ => egui::Key::Q,
                ExpressionKey::KeyW => egui::Key::W,
                ExpressionKey::KeyE => egui::Key::E,
                ExpressionKey::KeyR => egui::Key::R,
                ExpressionKey::KeyT => egui::Key::T,
                ExpressionKey::KeyY => egui::Key::Y,
                ExpressionKey::KeyU => egui::Key::U,
                ExpressionKey::KeyI => egui::Key::I,
                ExpressionKey::KeyO => egui::Key::O,
                ExpressionKey::KeyP => egui::Key::P,
                ExpressionKey::KeyA => egui::Key::A,
                ExpressionKey::KeyS => egui::Key::S,
                ExpressionKey::KeyD => egui::Key::D,
                ExpressionKey::KeyF => egui::Key::F,
                ExpressionKey::KeyG => egui::Key::G,
                ExpressionKey::KeyH => egui::Key::H,
                ExpressionKey::KeyJ => egui::Key::J,
                ExpressionKey::KeyK => egui::Key::K,
                ExpressionKey::KeyL => egui::Key::L,
                ExpressionKey::KeyZ => egui::Key::Z,
                ExpressionKey::KeyX => egui::Key::X,
                ExpressionKey::KeyC => egui::Key::C,
                ExpressionKey::KeyV => egui::Key::V,
                ExpressionKey::KeyB => egui::Key::B,
                ExpressionKey::KeyN => egui::Key::N,
                ExpressionKey::KeyM => egui::Key::M,
            };
            assert_eq!(expression_key_from_egui(egui_key), Some(key));
        }
        assert_eq!(expression_key_from_egui(egui::Key::Escape), None);
        assert_eq!(expression_key_from_egui(egui::Key::Space), None);
        assert_eq!(expression_key_from_egui(egui::Key::F1), None);
    }
}
