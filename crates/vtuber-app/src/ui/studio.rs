//! Apple-style desktop workspace. The sidebar contains destinations only.
//! Camera controls, session commands, and state never live in that sidebar.

use bevy_egui::egui::{self, Color32, CornerRadius, Frame, Id, RichText, TextureId, Ui, vec2};
use crate::actions::UiAction;
use crate::diagnostics::DiagnosticsSnapshot;
use crate::error_presenter::ErrorPresentation;
use crate::preview::PreviewState;
use crate::preview_landmarks::PreviewLandmarkState;
use crate::settings::UiLanguage;
use crate::ui_model::{AppLifecycle, AvatarLifecycleState, NdiOutputUiState, Pane, TrackingState, UiViewModel};
use vtuber_avatar::{ArmPoseProfileOverride, AvatarMotionMirror};
use vtuber_core::{VideoOutputProfile, monotonic_now};
use super::privacy::{CameraPreviewConsent, CameraPreviewEvent};
use super::shell::UiState;

const NAVIGATION: [Pane; 7] = [Pane::Studio, Pane::Avatar, Pane::Camera, Pane::Calibration, Pane::NdiOutput, Pane::Settings, Pane::Diagnostics];

fn page_title(pane: Pane, lang: UiLanguage) -> &'static str {
    match pane {
        Pane::Studio => lang.pick("スタジオ", "Studio", "工作室", "스튜디오"),
        Pane::Avatar => lang.pick("アバター", "Avatar", "虚拟形象", "아바타"),
        Pane::Camera | Pane::Preview => lang.pick("カメラ", "Camera", "摄像头", "카메라"),
        Pane::Calibration => lang.pick("キャリブレーション", "Calibration", "校准", "캘리브레이션"),
        Pane::NdiOutput => lang.pick("出力", "Output", "输出", "출력"),
        Pane::Settings => lang.pick("設定", "Settings", "设置", "설정"),
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
        AvatarLifecycleState::None => lang.pick("未読み込み", "Not loaded", "未加载", "불러오지 않음"),
        AvatarLifecycleState::Loading => lang.pick("読み込み中…", "Loading…", "正在加载…", "불러오는 중…"),
        AvatarLifecycleState::Binding => lang.pick("準備中…", "Preparing…", "正在准备…", "준비 중…"),
        AvatarLifecycleState::Ready => lang.pick("準備完了", "Ready", "就绪", "준비 완료"),
        AvatarLifecycleState::Unloading => lang.pick("解除中…", "Unloading…", "正在卸载…", "해제 중…"),
        AvatarLifecycleState::Failed => lang.pick("読み込み失敗", "Load failed", "加载失败", "불러오기 실패"),
    }
}

fn panel_frame(gray: u8) -> Frame {
    Frame::new().fill(Color32::from_gray(gray)).inner_margin(egui::Margin::same(16))
}

fn section(ui: &mut Ui, title: &str, contents: impl FnOnce(&mut Ui)) {
    ui.add_space(12.0);
    Frame::new().fill(Color32::WHITE)
        .stroke(egui::Stroke::new(1.0, Color32::from_gray(220)))
        .corner_radius(CornerRadius::same(10)).inner_margin(egui::Margin::same(16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(title).size(17.0).strong());
            ui.add_space(10.0);
            contents(ui);
        });
}

fn primary_button(ui: &mut Ui, label: &str, enabled: bool) -> egui::Response {
    ui.add_enabled(enabled, egui::Button::new(RichText::new(label).color(Color32::WHITE))
        .fill(ui.visuals().hyperlink_color).min_size(vec2(0.0, 30.0)))
}

fn use_sidebar(width: f32) -> bool { width >= 1000.0 }

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
        if ui.add_sized([ui.available_width(), 30.0], egui::Button::selectable(selected, page_title(pane, lang))).clicked() {
            state.emit(UiAction::SwitchPane(pane));
        }
    }
}

fn compact_navigation(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    egui::ComboBox::from_id_salt("studio_navigation")
        .selected_text(page_title(vm.pane, lang))
        .show_ui(ui, |ui| {
            for pane in NAVIGATION {
                if ui.selectable_label(vm.pane == pane, page_title(pane, lang)).clicked() {
                    state.emit(UiAction::SwitchPane(pane));
                }
            }
        });
}

fn toolbar(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, sidebar: bool, lang: UiLanguage) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("RusTuberV").strong());
        if !sidebar { compact_navigation(ui, vm, state, lang); }
        ui.separator();
        ui.label(session_label(vm.lifecycle, lang));
        if vm.can_stop() {
            if ui.button(lang.pick("トラッキングを停止", "Stop tracking", "停止跟踪", "트래킹 중지")).clicked() {
                state.emit(UiAction::Stop);
            }
        } else if primary_button(ui, lang.pick("トラッキングを開始", "Start tracking", "开始跟踪", "트래킹 시작"), vm.can_start()).clicked() {
            state.emit(UiAction::Start);
        }
        if ui.button(lang.pick("アバターのみ (F1)", "Avatar only (F1)", "仅虚拟形象 (F1)", "아바타만 (F1)")).clicked() {
            state.set_controls_open(false);
        }
    });
}

fn avatar_monitor(
    ui: &mut Ui, vm: &UiViewModel, texture: Option<(TextureId, VideoOutputProfile)>,
    max_width: f32, lang: UiLanguage,
) {
    ui.label(RichText::new(lang.pick("アバタープレビュー", "Avatar preview", "虚拟形象预览", "아바타 미리 보기")).strong());
    ui.add_space(8.0);
    if vm.avatar.is_ready {
        if let Some((texture, profile)) = texture {
            let width = ui.available_width().min(max_width);
            let height = width * profile.height as f32 / profile.width as f32;
            Frame::new().fill(Color32::from_gray(228)).corner_radius(CornerRadius::same(8)).show(ui, |ui| {
                ui.add(egui::Image::from_texture((texture, vec2(width, height))).corner_radius(CornerRadius::same(8)));
            });
        } else {
            ui.spinner();
        }
    } else {
        ui.label(avatar_label(vm.avatar.lifecycle, lang));
    }
    ui.add_space(8.0);
    ui.label(RichText::new(lang.pick("NDIと同じアバター専用描画です。設定UIとカメラ映像は含みません。", "The same avatar-only render used by NDI. Settings and camera pixels are excluded.", "与NDI共用的虚拟形象画面，不含设置界面或摄像头影像。", "NDI와 동일한 아바타 전용 화면입니다. 설정과 카메라 영상은 포함되지 않습니다.")).small().weak());
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_studio(
    ctx: &egui::Context, vm: &UiViewModel, state: &mut UiState,
    diagnostics: &DiagnosticsSnapshot, error: Option<&ErrorPresentation>,
    preview: &PreviewState, landmarks: &PreviewLandmarkState,
    avatar_mirror: AvatarMotionMirror, camera_texture: Option<TextureId>,
    avatar_texture: Option<(TextureId, VideoOutputProfile)>, dialog_active: bool,
    font_error: Option<&str>, lang: UiLanguage,
) -> bool {
    if !state.controls_open {
        let handle = egui::Area::new(Id::new("open_studio"))
            .anchor(egui::Align2::LEFT_CENTER, vec2(0.0, 0.0)).movable(false)
            .show(ctx, |ui| {
                if ui.button(lang.pick("設定 (F1)", "Settings (F1)", "设置 (F1)", "설정 (F1)")).clicked() {
                    state.set_controls_open(true);
                }
            });
        return ctx.input(|input| input.pointer.interact_pos()).is_some_and(|pos| handle.response.rect.contains(pos));
    }
    let sidebar = use_sidebar(ctx.viewport_rect().width());
    let mut root = Ui::new(ctx.clone(), Id::new("studio_root"), egui::UiBuilder::new()
        .layer_id(egui::LayerId::background()).max_rect(ctx.viewport_rect()));
    egui::Panel::top("studio_toolbar").resizable(false).frame(panel_frame(250))
        .show(&mut root, |ui| toolbar(ui, vm, state, sidebar, lang));
    if sidebar {
        egui::Panel::left("studio_sidebar").exact_size(200.0).resizable(false).frame(panel_frame(242))
            .show(&mut root, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| navigation(ui, vm, state, lang));
            });
        egui::Panel::right("studio_monitor").exact_size(300.0).resizable(false).frame(panel_frame(250))
            .show(&mut root, |ui| avatar_monitor(ui, vm, avatar_texture, 268.0, lang));
    } else {
        // On smaller windows keep the monitor outside the settings scroll area.
        egui::Panel::top("studio_compact_monitor").resizable(false).frame(panel_frame(250))
            .show(&mut root, |ui| avatar_monitor(ui, vm, avatar_texture, 200.0, lang));
    }
    egui::CentralPanel::default().frame(panel_frame(247)).show(&mut root, |ui| {
        egui::ScrollArea::vertical().id_salt(("studio_detail", vm.pane as u8))
            .auto_shrink([false, false]).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(page_title(vm.pane, lang)).size(26.0).strong());
                if let Some(detail) = font_error {
                    section(ui, "Font could not be loaded", |ui| {
                        ui.label(detail);
                        ui.horizontal(|ui| {
                            if ui.button("Japanese").clicked() { state.emit(UiAction::SetLanguage(UiLanguage::Ja)); }
                            if ui.button("English").clicked() { state.emit(UiAction::SetLanguage(UiLanguage::En)); }
                        });
                    });
                }
                if let Some(error) = error {
                    section(ui, lang.pick("操作を確認してください", "Check this operation", "请检查此操作", "작업을 확인하세요"), |ui| {
                        ui.label(&error.user_message);
                        for action in &error.suggested_actions {
                            let label = match action {
                                UiAction::RefreshCameras => lang.pick("カメラを再検出", "Refresh cameras", "重新检测摄像头", "카메라 새로 고침"),
                                UiAction::RetryAfterError => lang.pick("再試行", "Retry", "重试", "다시 시도"),
                                UiAction::DismissError => lang.pick("閉じる", "Dismiss", "关闭", "닫기"),
                                _ => continue,
                            };
                            if ui.button(label).clicked() { state.emit(action.clone()); }
                        }
                    });
                }
                match vm.pane {
                    Pane::Studio => overview(ui, vm, state, dialog_active, lang),
                    Pane::Avatar => avatar_page(ui, vm, state, avatar_mirror, dialog_active, lang),
                    Pane::Camera | Pane::Preview => camera_page(ui, vm, state, preview, landmarks, camera_texture, lang),
                    Pane::Calibration => calibration_page(ui, vm, state, lang),
                    Pane::NdiOutput => output_page(ui, vm, state, lang),
                    Pane::Settings => settings_page(ui, state, lang),
                    Pane::Diagnostics => diagnostics_page(ui, vm, diagnostics, lang),
                }
                ui.add_space(16.0);
            });
    });
    ctx.input(|input| input.pointer.interact_pos()).is_some_and(|pos| ctx.viewport_rect().contains(pos))
}

fn import_controls(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, active: bool, lang: UiLanguage) {
    if let Some(model) = &vm.avatar.imported_model {
        ui.label(RichText::new(&model.name).strong());
        ui.label(match model.generation { crate::import::VrmGeneration::Vrm0 => "VRM 0.x", crate::import::VrmGeneration::Vrm1 => "VRM 1.0" });
    }
    ui.label(avatar_label(vm.avatar.lifecycle, lang));
    if primary_button(ui, lang.pick("VRMを読み込む…", "Choose VRM…", "选择VRM…", "VRM 불러오기…"), !active).clicked() {
        state.import_requested = true;
    }
    ui.label(RichText::new(lang.pick("VRM 0.x / 1.0。ファイルをウインドウにドロップしても読み込めます。", "VRM 0.x / 1.0. You can also drop a file into this window.", "支持VRM 0.x / 1.0，也可将文件拖入窗口。", "VRM 0.x / 1.0을 지원합니다. 파일을 창에 끌어 놓아도 됩니다.")).small().weak());
}

fn camera_controls(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    let selected_label = vm.camera.selected_index.and_then(|index| vm.camera.available_cameras.get(index))
        .map(|camera| camera.name.as_str()).unwrap_or_else(|| lang.pick("カメラを選択", "Select camera", "选择摄像头", "카메라 선택"));
    let can_change = matches!(vm.lifecycle, AppLifecycle::Idle | AppLifecycle::Failed);
    ui.add_enabled_ui(can_change, |ui| {
        egui::ComboBox::from_id_salt("studio_camera_device").selected_text(selected_label)
            .width(ui.available_width().min(360.0)).show_ui(ui, |ui| {
                for (index, camera) in vm.camera.available_cameras.iter().enumerate() {
                    if ui.selectable_label(vm.camera.selected_index == Some(index), &camera.name).clicked() {
                        state.emit(UiAction::SelectCamera { index });
                    }
                }
            });
    });
    if !can_change {
        ui.label(lang.pick("カメラを変更するには、トラッキングを停止してください。", "Stop tracking before changing the camera.", "更换摄像头前请停止跟踪。", "카메라를 변경하려면 트래킹을 중지하세요."));
    }
    if ui.button(lang.pick("カメラを再検出", "Refresh cameras", "重新检测摄像头", "카메라 새로 고침")).clicked() {
        state.emit(UiAction::RefreshCameras);
    }
    if vm.camera.available_cameras.is_empty() {
        ui.label(lang.pick("カメラが見つかりません。接続とOSのカメラ権限を確認してください。", "No camera found. Check the connection and OS camera permission.", "未找到摄像头。请检查连接及系统摄像头权限。", "카메라를 찾지 못했습니다. 연결과 OS 카메라 권한을 확인하세요."));
    }
}

fn overview(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, dialog_active: bool, lang: UiLanguage) {
    ui.label(lang.pick("アバターを読み込み、カメラを選んでトラッキングを開始します。", "Load an avatar, select a camera, then start tracking.", "加载虚拟形象、选择摄像头，然后开始跟踪。", "아바타를 불러오고 카메라를 선택한 뒤 트래킹을 시작하세요."));
    section(ui, lang.pick("1. アバターを読み込む", "1. Load an avatar", "1. 加载虚拟形象", "1. 아바타 불러오기"), |ui| import_controls(ui, vm, state, dialog_active, lang));
    section(ui, lang.pick("2. カメラを選ぶ", "2. Choose a camera", "2. 选择摄像头", "2. 카메라 선택"), |ui| {
        camera_controls(ui, vm, state, lang);
        ui.add_space(8.0);
        ui.label(lang.pick("カメラ映像は非表示です。メニューを開くだけでは表示されません。", "Camera preview is hidden. Opening a menu never reveals it.", "摄像头预览默认隐藏，打开菜单不会显示影像。", "카메라 미리 보기는 숨겨져 있습니다. 메뉴를 열어도 영상이 표시되지 않습니다."));
        if ui.button(lang.pick("カメラ調整へ", "Camera settings", "摄像头设置", "카메라 설정")).clicked() { state.emit(UiAction::SwitchPane(Pane::Camera)); }
    });
    section(ui, lang.pick("3. 動きと出力を確認する", "3. Check motion and output", "3. 检查动作与输出", "3. 움직임과 출력 확인"), |ui| {
        ui.label(lang.pick("上部でトラッキングを開始し、常時表示のアバタープレビューで動きを確認してください。", "Start tracking in the toolbar and check motion in the persistent avatar preview.", "在顶部开始跟踪，通过常驻的虚拟形象预览检查动作。", "상단에서 트래킹을 시작하고 항상 표시되는 아바타 미리 보기로 움직임을 확인하세요."));
        ui.horizontal_wrapped(|ui| {
            if ui.button(page_title(Pane::Calibration, lang)).clicked() { state.emit(UiAction::SwitchPane(Pane::Calibration)); }
            if ui.button(lang.pick("出力を設定", "Configure output", "设置输出", "출력 설정")).clicked() { state.emit(UiAction::SwitchPane(Pane::NdiOutput)); }
        });
    });
}

fn avatar_page(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, mirror: AvatarMotionMirror, dialog_active: bool, lang: UiLanguage) {
    section(ui, lang.pick("モデル", "Model", "模型", "모델"), |ui| {
        import_controls(ui, vm, state, dialog_active, lang);
        if vm.avatar.imported_model.is_some() && ui.button(lang.pick("アバターを解除", "Unload avatar", "卸载虚拟形象", "아바타 해제")).clicked() { state.emit(UiAction::UnloadAvatar); }
        if vm.avatar.load_failed && ui.button(lang.pick("再読み込み", "Retry load", "重新加载", "다시 불러오기")).clicked() { state.emit(UiAction::RetryAfterError); }
    });
    section(ui, lang.pick("表示と構図", "Display and framing", "显示与构图", "표시 및 구도"), |ui| {
        let mut reflected = mirror.is_enabled();
        if ui.checkbox(&mut reflected, lang.pick("アバターの動きを左右反転", "Mirror avatar motion", "镜像虚拟形象动作", "아바타 움직임 좌우 반전")).changed() { state.emit(UiAction::ToggleAvatarMotionMirror); }
        if ui.add_enabled(vm.can_reset_camera(), egui::Button::new(lang.pick("アバターの画角をリセット", "Reset avatar framing", "重置虚拟形象构图", "아바타 구도 초기화"))).clicked() { state.emit(UiAction::ResetAvatarCamera); }
        ui.label(lang.pick("F1でアバターのみを表示。左ドラッグで回転、右ドラッグで移動、ホイールでズームします。", "Use F1 for avatar-only view. Left-drag to orbit, right-drag to pan, and scroll to zoom.", "按F1仅显示虚拟形象。左键拖动旋转，右键拖动平移，滚轮缩放。", "F1로 아바타만 표시합니다. 왼쪽 드래그로 회전, 오른쪽 드래그로 이동, 휠로 확대·축소합니다."));
    });
    if vm.avatar.imported_model.is_some() {
        section(ui, lang.pick("腕の姿勢", "Arm pose", "手臂姿势", "팔 자세"), |ui| {
            ui.label(lang.pick("このモデルの設定として保存されます。", "Saved for this model.", "设置将为此模型保存。", "이 모델의 설정으로 저장됩니다."));
            let mut profile = vm.arm_pose.profile;
            let mut drop_degrees = profile.arm_drop_radians.to_degrees();
            let mut curl_degrees = profile.finger_curl_radians.to_degrees();
            let mut changed = ui.add(egui::Slider::new(&mut drop_degrees, 0.0..=90.0).text(lang.pick("腕下げ", "Arm drop", "手臂下垂", "팔 내리기"))).changed();
            changed |= ui.add(egui::Slider::new(&mut profile.reach_ratio, 0.01..=1.0).text(lang.pick("リーチ比", "Reach ratio", "伸展比例", "뻗기 비율"))).changed();
            changed |= ui.add(egui::Slider::new(&mut profile.forward_hand_offset_ratio, -1.0..=1.0).text(lang.pick("前方オフセット", "Forward offset", "前向偏移", "앞쪽 오프셋"))).changed();
            changed |= ui.add(egui::Slider::new(&mut profile.elbow_pole_offset_ratio, 0.0..=1.0).text(lang.pick("肘の位置", "Elbow pole", "肘部位置", "팔꿈치 위치"))).changed();
            changed |= ui.add(egui::Slider::new(&mut profile.shoulder_follow_weight, 0.0..=1.0).text(lang.pick("肩の追従", "Shoulder follow", "肩部跟随", "어깨 추종"))).changed();
            changed |= ui.add(egui::Slider::new(&mut curl_degrees, 0.0..=90.0).text(lang.pick("指の曲げ", "Finger curl", "手指弯曲", "손가락 굽힘"))).changed();
            if changed {
                profile.arm_drop_radians = drop_degrees.to_radians(); profile.finger_curl_radians = curl_degrees.to_radians();
                state.emit(UiAction::SetArmPoseProfile { profile: ArmPoseProfileOverride::from_profile(profile) });
            }
            if vm.arm_pose.has_override && ui.button(lang.pick("自動に戻す", "Reset to automatic", "恢复自动", "자동으로 되돌리기")).clicked() { state.emit(UiAction::ResetArmPoseProfile); }
        });
    }
}

fn camera_page(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, preview: &PreviewState, landmarks: &PreviewLandmarkState, texture: Option<TextureId>, lang: UiLanguage) {
    section(ui, lang.pick("入力カメラ", "Input camera", "输入摄像头", "입력 카메라"), |ui| camera_controls(ui, vm, state, lang));
    section(ui, lang.pick("カメラ映像の確認", "Check camera preview", "查看摄像头预览", "카메라 영상 확인"), |ui| {
        match state.camera_consent {
            CameraPreviewConsent::Hidden => {
                ui.label(lang.pick("カメラ映像は非表示です。トラッキングとは独立した表示設定です。", "Camera pixels are hidden. Preview visibility is independent of tracking.", "摄像头影像已隐藏。预览显示与跟踪功能相互独立。", "카메라 영상은 숨겨져 있습니다. 미리 보기 표시는 트래킹과 별개입니다."));
                if ui.button(lang.pick("カメラ映像を確認…", "Preview camera…", "查看摄像头影像…", "카메라 영상 확인…")).clicked() { state.preview_event(CameraPreviewEvent::Request); }
            }
            CameraPreviewConsent::Confirming => {
                ui.label(RichText::new(lang.pick("このウインドウに実際のカメラ映像を表示します。", "This will show the actual camera image in this window.", "此操作将在本窗口显示真实摄像头影像。", "이 창에 실제 카메라 영상이 표시됩니다.")).strong());
                ui.label(lang.pick("デスクトップ／ウインドウキャプチャで配信中の場合、顔や部屋も配信に映ります。NDI出力には含まれません。", "Desktop or window capture can broadcast your face and room. NDI output will not include them.", "若正在通过桌面或窗口捕获直播，您的脸部和房间也会被播出。NDI输出不包含这些影像。", "데스크톱이나 창 캡처로 방송 중이면 얼굴과 방도 방송에 보입니다. NDI 출력에는 포함되지 않습니다."));
                ui.horizontal_wrapped(|ui| {
                    if ui.button(lang.pick("キャンセル", "Cancel", "取消", "취소")).clicked() { state.preview_event(CameraPreviewEvent::Hide); }
                    if ui.button(lang.pick("確認して映像を表示", "Show camera image", "确认并显示影像", "확인 후 영상 표시")).clicked() { state.preview_event(CameraPreviewEvent::Confirm); }
                });
            }
            CameraPreviewConsent::Visible => {
                if ui.button(lang.pick("カメラ映像を隠す (Esc)", "Hide camera (Esc)", "隐藏摄像头影像 (Esc)", "카메라 영상 숨기기 (Esc)")).clicked() { state.preview_event(CameraPreviewEvent::Hide); }
                // Check after the Hide button: do not emit even one more image mesh.
                if state.camera_consent == CameraPreviewConsent::Visible {
                    if let Some(texture) = texture {
                        let width = ui.available_width().min(640.0);
                        let size = vec2(width, width * 9.0 / 16.0);
                        let uv = if preview.mirrored { egui::Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0)) } else { egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)) };
                        let rect = ui.add(egui::Image::from_texture((texture, size)).uv(uv).corner_radius(CornerRadius::same(8))).rect;
                        if let Some(snapshot) = landmarks.latest_fresh_at(monotonic_now()) {
                            let painter = ui.painter().with_clip_rect(rect);
                            for point in snapshot.landmarks.iter() {
                                if point.x.is_finite() && point.y.is_finite() && (0.0..=1.0).contains(&point.x) && (0.0..=1.0).contains(&point.y) {
                                    let x = if preview.mirrored { 1.0 - point.x } else { point.x };
                                    painter.circle_filled(egui::pos2(rect.left() + x * rect.width(), rect.top() + point.y * rect.height()), 1.5, Color32::YELLOW);
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
        if ui.checkbox(&mut mirrored, lang.pick("カメラプレビューを左右反転", "Mirror camera preview", "镜像摄像头预览", "카메라 미리 보기 좌우 반전")).changed() { state.emit(UiAction::ToggleMirror); }
        ui.label(RichText::new(lang.pick("別の画面への移動、設定を閉じる操作、Escで再び非表示になります。", "Changing pages, closing settings, or pressing Esc hides it again.", "切换页面、关闭设置或按Esc后，影像会再次隐藏。", "다른 화면으로 이동하거나 설정을 닫거나 Esc를 누르면 다시 숨겨집니다.")).small().weak());
    });
}

fn calibration_page(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    section(ui, lang.pick("中立姿勢", "Neutral pose", "自然姿势", "중립 자세"), |ui| {
        ui.label(lang.pick("カメラに向かい、自然な表情で開始してください。映像を表示する必要はありません。", "Face the camera with a relaxed expression. There is no need to reveal the camera image.", "面向摄像头，保持自然表情后开始。无需显示摄像头影像。", "카메라를 향해 편안한 표정으로 시작하세요. 카메라 영상을 표시할 필요는 없습니다."));
        if vm.calibration.is_calibrating {
            let value = if vm.calibration.samples_target > 0 { vm.calibration.samples_collected as f32 / vm.calibration.samples_target as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(value.clamp(0.0, 1.0)).text(format!("{} / {}", vm.calibration.samples_collected, vm.calibration.samples_target)));
            if ui.button(lang.pick("キャンセル", "Cancel", "取消", "취소")).clicked() { state.emit(UiAction::CancelCalibration); }
        } else if vm.calibration.is_complete {
            ui.label(lang.pick("キャリブレーション完了", "Calibrated", "校准完成", "캘리브레이션 완료"));
            if ui.button(lang.pick("やり直す", "Redo", "重新校准", "다시 하기")).clicked() { state.emit(UiAction::RetryCalibration); }
        } else {
            if primary_button(ui, lang.pick("キャリブレーション開始", "Begin calibration", "开始校准", "캘리브레이션 시작"), vm.can_calibrate()).clicked() { state.emit(UiAction::BeginCalibration); }
            if !vm.can_calibrate() { ui.label(lang.pick("先にトラッキングを開始してください。", "Start tracking first.", "请先开始跟踪。", "먼저 트래킹을 시작하세요.")); }
        }
        if let Some(score) = vm.calibration.quality_score { ui.label(format!("{}: {:.0}%", lang.pick("品質", "Quality", "质量", "품질"), score * 100.0)); }
        if let Some(reason) = &vm.calibration.last_reject_reason { ui.label(format!("{}: {reason}", lang.pick("詳細", "Details", "详情", "세부 정보"))); }
    });
}

fn output_page(ui: &mut Ui, vm: &UiViewModel, state: &mut UiState, lang: UiLanguage) {
    section(ui, lang.pick("NDI — アバターのみ", "NDI — avatar only", "NDI — 仅虚拟形象", "NDI — 아바타만"), |ui| {
        ui.label(lang.pick("常時プレビューと同じアバター専用描画を送信します。設定UI、カメラ映像、プレビュー枠は送信しません。", "Sends the same avatar-only render as the persistent preview. Settings, camera pixels, and preview chrome are never sent.", "发送与常驻预览相同的虚拟形象画面，不发送设置界面、摄像头影像或预览边框。", "항상 표시되는 미리 보기와 같은 아바타 전용 화면을 전송합니다. 설정, 카메라 영상, 미리 보기 테두리는 전송하지 않습니다."));
        let status = match vm.ndi_output.state {
            NdiOutputUiState::Off => lang.pick("停止", "Off", "已停止", "중지"),
            NdiOutputUiState::Starting => lang.pick("開始中…", "Starting…", "正在启动…", "시작 중…"),
            NdiOutputUiState::Live => lang.pick("送信中", "Sending", "发送中", "전송 중"),
            NdiOutputUiState::Error => lang.pick("エラー", "Error", "错误", "오류"),
        };
        ui.label(format!("{}: {status}", lang.pick("状態", "Status", "状态", "상태")));
        if let Some(name) = &vm.ndi_output.source_name { ui.label(format!("{}: {name}", lang.pick("ソース名", "Source name", "源名称", "소스 이름"))); }
        if let Some(count) = vm.ndi_output.connections { ui.label(format!("{}: {count}", lang.pick("受信数", "Receivers", "接收端数量", "수신 수"))); }
        if !vm.ndi_output.available {
            ui.label(lang.pick("このビルドにはNDI出力が含まれていません。", "This build does not include NDI output.", "此构建未包含NDI输出。", "이 빌드에는 NDI 출력이 포함되어 있지 않습니다."));
        } else if !vm.ndi_output.runtime_installed {
            ui.label(lang.pick("NDIランタイムが見つかりません。NDIを使う場合のみ必要です。", "NDI runtime not found. It is needed only when using NDI.", "未找到NDI运行库。仅使用NDI时需要它。", "NDI 런타임을 찾지 못했습니다. NDI를 사용할 때만 필요합니다."));
            ui.hyperlink_to(lang.pick("NDI公式サイト", "NDI website", "NDI官方网站", "NDI 공식 사이트"), "https://ndi.video");
        }
        ui.horizontal_wrapped(|ui| {
            if primary_button(ui, lang.pick("NDI送信を開始", "Start NDI", "开始NDI发送", "NDI 전송 시작"), vm.can_start_ndi_output() && vm.ndi_output.runtime_installed).clicked() { state.emit(UiAction::StartNdiOutput); }
            if ui.add_enabled(vm.can_stop_ndi_output(), egui::Button::new(lang.pick("NDI送信を停止", "Stop NDI", "停止NDI发送", "NDI 전송 중지"))).clicked() { state.emit(UiAction::StopNdiOutput); }
        });
        if let Some(code) = &vm.ndi_output.error_code { ui.label(format!("{}: {code}", lang.pick("エラーコード", "Error code", "错误代码", "오류 코드"))); }
        if let Some(detail) = &vm.ndi_output.error_message {
            egui::CollapsingHeader::new(lang.pick("技術情報（原文）", "Technical details (original)", "技术详情（原文）", "기술 정보 (원문)")).show(ui, |ui| { ui.label(detail); });
        }
    });
    section(ui, lang.pick("画面キャプチャで使う", "Use screen capture", "使用屏幕捕获", "화면 캡처 사용"), |ui| {
        ui.label(lang.pick("F1で設定を隠し、アバター表示に戻せます。ただし、デスクトップ／ウインドウキャプチャには、後から開いた設定や表示を許可したカメラ映像も映ります。アバターのみを確実に送る場合はNDIを選んでください。", "F1 hides settings and returns to the avatar. Desktop/window capture can still include settings opened later or a camera preview you explicitly reveal. Use NDI for an avatar-only feed.", "按F1隐藏设置并返回虚拟形象。桌面或窗口捕获仍会包含之后打开的设置以及您确认显示的摄像头影像。需要仅发送虚拟形象时，请使用NDI。", "F1로 설정을 숨기고 아바타 화면으로 돌아갑니다. 데스크톱이나 창 캡처에는 나중에 연 설정이나 직접 표시한 카메라 영상도 포함될 수 있습니다. 아바타만 전송하려면 NDI를 사용하세요."));
        if ui.button(lang.pick("アバターだけを表示 (F1)", "Show avatar only (F1)", "仅显示虚拟形象 (F1)", "아바타만 표시 (F1)")).clicked() { state.set_controls_open(false); }
    });
}

fn settings_page(ui: &mut Ui, state: &mut UiState, lang: UiLanguage) {
    section(ui, lang.pick("表示言語", "Display language", "显示语言", "표시 언어"), |ui| {
        let languages = [
            (UiLanguage::Ja, lang.pick("日本語", "Japanese", "日语", "일본어")),
            (UiLanguage::En, lang.pick("英語", "English", "英语", "영어")),
            (UiLanguage::Zh, lang.pick("中国語（簡体字）", "Chinese (Simplified)", "简体中文", "중국어 (간체)")),
            (UiLanguage::Ko, lang.pick("韓国語", "Korean", "韩语", "한국어")),
        ];
        for (language, label) in languages {
            if ui.radio(lang == language, label).clicked() && lang != language { state.emit(UiAction::SetLanguage(language)); }
        }
        ui.label(lang.pick("初期設定は日本語です。変更はすぐに反映され、再起動後も維持されます。", "Japanese is the initial language. Changes apply immediately and persist across restarts.", "初始语言为日语。更改立即生效，重启后仍会保留。", "초기 언어는 일본어입니다. 변경 사항은 즉시 적용되며 재시작 후에도 유지됩니다."));
    });
}

fn diagnostics_page(ui: &mut Ui, vm: &UiViewModel, snapshot: &DiagnosticsSnapshot, lang: UiLanguage) {
    section(ui, lang.pick("トラッキング", "Tracking", "跟踪", "트래킹"), |ui| {
        let tracking = match vm.tracking.state {
            TrackingState::Idle => lang.pick("停止中", "Idle", "已停止", "중지됨"),
            TrackingState::Initializing => lang.pick("初期化中", "Initializing", "初始化中", "초기화 중"),
            TrackingState::Tracking => lang.pick("追跡中", "Tracking", "跟踪中", "트래킹 중"),
            TrackingState::Lost => lang.pick("顔を検出できません", "Face lost", "未检测到脸部", "얼굴을 찾지 못함"),
        };
        ui.label(tracking);
        ui.label(format!("{}: {:.0}%", lang.pick("信頼度", "Confidence", "置信度", "신뢰도"), vm.tracking.confidence * 100.0));
        ui.label(avatar_label(vm.avatar.lifecycle, lang));
    });
    section(ui, lang.pick("詳細診断", "Detailed diagnostics", "详细诊断", "상세 진단"), |ui| {
        ui.label(lang.pick("技術的な識別子とバックエンドのメッセージは原文で表示します。カメラ映像は表示しません。", "Technical identifiers and backend messages retain their original text. No camera image is shown.", "技术标识符和后端消息保留原文，不显示摄像头影像。", "기술 식별자와 백엔드 메시지는 원문으로 표시합니다. 카메라 영상은 표시하지 않습니다."));
        egui::CollapsingHeader::new(lang.pick("技術情報を開く", "Open technical details", "打开技术详情", "기술 정보 열기")).show(ui, |ui| { ui.label(RichText::new(format!("{snapshot:#?}")).monospace()); });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_layout_keeps_navigation_and_a_persistent_monitor() {
        assert!(!use_sidebar(800.0)); assert!(use_sidebar(1280.0));
        assert_eq!(NAVIGATION.len(), 7);
        assert!(NAVIGATION.contains(&Pane::Studio)); assert!(NAVIGATION.contains(&Pane::Settings));
        assert!(!NAVIGATION.contains(&Pane::Preview));
    }
    #[test]
    fn every_destination_has_a_label_in_all_four_languages() {
        for pane in NAVIGATION {
            for lang in [UiLanguage::Ja, UiLanguage::En, UiLanguage::Zh, UiLanguage::Ko] {
                assert!(!page_title(pane, lang).is_empty());
            }
        }
    }
}
