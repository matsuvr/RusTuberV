//! Camera selection, preview consent and display settings.
use crate::actions::ActionQueue;

use super::super::privacy::{CameraPreviewConsent, CameraPreviewEvent};
use super::{RichText, Ui, UiLanguage, UiState, UiViewModel, egui};
use super::{section, setup_section};
use crate::{actions::UiAction, preview::PreviewState, preview_landmarks::PreviewLandmarkState};
use bevy_egui::egui::{Color32, CornerRadius, TextureId, vec2};
use vtuber_avatar::AvatarMotionMirror;
use vtuber_core::monotonic_now;

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
            ui.label(lang.pick("F1でアバターのみを表示。左ドラッグで回転、右ドラッグで移動、ホイールでズームします。", "Use F1 for avatar-only view. Left-drag to orbit, right-drag to pan, and scroll to zoom.", "按F1仅显示虚拟形象。左键拖动旋转，右键拖动平移，滚轮缩放。", "F1로 아바타만 표시합니다. 왼쪽 드래그로 회전, 오른쪽 드래그로 이동, 휠로 확대·축소합니다."));
        },
    );
}
