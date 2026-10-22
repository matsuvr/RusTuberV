//! Camera selection, preview consent and display settings.
use crate::actions::ActionQueue;

use super::super::privacy::{CameraPreviewConsent, CameraPreviewEvent};
use super::{RichText, Ui, UiLanguage, UiState, UiViewModel, egui};
use super::{section, setup_section};
use crate::{actions::UiAction, preview::PreviewState, preview_landmarks::PreviewLandmarkState};
use bevy_egui::egui::{Color32, CornerRadius, TextureId, vec2};
use vtuber_avatar::AvatarMotionMirror;
use vtuber_core::monotonic_now;

fn hand_tracking_switch(ui: &mut Ui, enabled: &mut bool, label: &str) -> egui::Response {
    let switch_size = vec2(36.0, 22.0);
    let gap = 10.0;
    let text_color = ui.visuals().text_color();
    let text = ui.painter().layout(
        label.to_owned(),
        egui::TextStyle::Body.resolve(ui.style()),
        text_color,
        (ui.available_width() - switch_size.x - gap).max(0.0),
    );
    let size = vec2(text.size().x + gap + switch_size.x, text.size().y.max(28.0));
    let (rect, mut response) = ui.allocate_exact_size(size, egui::Sense::click());
    if response.clicked() {
        *enabled = !*enabled;
        response.mark_changed();
    }
    response.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *enabled, label)
    });

    if ui.is_rect_visible(rect) {
        let progress = ui.ctx().animate_bool_with_time(response.id, *enabled, 0.16);
        let accent = Color32::from_rgb(0, 122, 255);
        let fill = Color32::from_rgb(174, 174, 178).lerp_to_gamma(accent, progress);
        let track = egui::Rect::from_center_size(
            egui::pos2(rect.right() - switch_size.x / 2.0, rect.center().y),
            switch_size,
        );
        let painter = ui.painter();
        painter.galley(
            egui::pos2(rect.left(), rect.center().y - text.size().y / 2.0),
            text,
            text_color,
        );
        painter.rect_filled(track, 11.0, fill);
        let center = egui::pos2(
            egui::lerp(track.left() + 11.0..=track.right() - 11.0, progress),
            track.center().y,
        );
        painter.add(
            egui::epaint::Shadow {
                offset: [0, 1],
                blur: 3,
                spread: 0,
                color: Color32::from_black_alpha(45),
            }
            .as_shape(egui::Rect::from_center_size(center, vec2(18.0, 18.0)), 9.0),
        );
        painter.circle_filled(center, 9.0, Color32::WHITE);
        if response.has_focus() {
            painter.rect_stroke(
                track.expand(3.0),
                14.0,
                egui::Stroke::new(2.0, accent),
                egui::StrokeKind::Outside,
            );
        }
    }
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

fn camera_controls(
    actions: &mut ActionQueue,
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    lang: UiLanguage,
) {
    let selected_label = vm
        .camera
        .selected_index
        .and_then(|index| vm.camera.available_cameras.get(index))
        .map(|camera| camera.name.as_str())
        .unwrap_or_else(|| lang.pick("カメラを選択", "Select camera", "选择摄像头", "카메라 선택"));
    egui::ComboBox::from_id_salt("studio_camera_device")
        .selected_text(selected_label)
        .width(ui.available_width().min(360.0))
        .show_ui(ui, |ui| {
            for (index, camera) in vm.camera.available_cameras.iter().enumerate() {
                if ui
                    .selectable_label(vm.camera.selected_index == Some(index), &camera.name)
                    .clicked()
                {
                    state.emit(actions, UiAction::SelectCamera { index });
                }
            }
        });
    if ui
        .button(lang.pick(
            "カメラを再検出",
            "Refresh cameras",
            "重新检测摄像头",
            "카메라 새로 고침",
        ))
        .clicked()
    {
        state.emit(actions, UiAction::RefreshCameras);
    }
    if vm.camera.selected_index.is_some() {
        ui.add_space(8.0);
        let mut enabled = vm.arm_tracking_enabled;
        if hand_tracking_switch(
            ui,
            &mut enabled,
            lang.pick(
                "手のトラッキングあり",
                "Hand tracking",
                "手部跟踪",
                "손 트래킹",
            ),
        )
        .changed()
        {
            state.emit(actions, UiAction::SetArmTrackingEnabled { enabled });
        }
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

pub(super) fn camera_select_page(
    actions: &mut ActionQueue,
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    dialog_active: bool,
    lang: UiLanguage,
) {
    setup_section(
        ui,
        lang.pick("入力カメラ", "Input camera", "输入摄像头", "입력 카메라"),
        vm.camera.selected_index.is_none(),
        !dialog_active && vm.avatar_import_review.review.is_none(),
        |ui| camera_controls(actions, ui, vm, state, lang),
    );
}

/// Camera preview consent and mirroring, moved from the camera page.
fn camera_preview_section(
    actions: &mut ActionQueue,
    ui: &mut Ui,
    state: &mut UiState,
    preview: &PreviewState,
    landmarks: &PreviewLandmarkState,
    texture: Option<TextureId>,
    lang: UiLanguage,
) {
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
                            ui.label(lang.pick(
                                "映像を待っています。VRMとカメラを選択してください。",
                                "Waiting for frames. Select a VRM and camera.",
                                "正在等待影像。请选择VRM和摄像头。",
                                "영상을 기다리고 있습니다. VRM과 카메라를 선택하세요.",
                            ));
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
                state.emit(actions, UiAction::ToggleMirror);
            }
        },
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the pane draws from one argument per widget input, so each widget reads exactly the state it is given"
)]
pub(super) fn camera_status_page(
    actions: &mut ActionQueue,
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    preview: &PreviewState,
    landmarks: &PreviewLandmarkState,
    texture: Option<TextureId>,
    mirror: AvatarMotionMirror,
    lang: UiLanguage,
) {
    camera_preview_section(actions, ui, state, preview, landmarks, texture, lang);
    display_framing_section(actions, ui, vm, state, mirror, lang);
}

/// Avatar mirroring, framing reset, and viewport help.
fn display_framing_section(
    actions: &mut ActionQueue,
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    mirror: AvatarMotionMirror,
    lang: UiLanguage,
) {
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
                state.emit(actions, UiAction::ToggleAvatarMotionMirror);
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
                state.emit(actions, UiAction::ResetAvatarCamera);
            }
        },
    );
}
