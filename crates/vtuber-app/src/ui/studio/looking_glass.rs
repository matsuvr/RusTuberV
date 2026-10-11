//! Settings page for the shared Looking Glass renderer.

use super::*;
use crate::looking_glass::DisplayFormat;

pub(super) fn page(
    actions: &mut ActionQueue,
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    dialog_active: bool,
    lang: UiLanguage,
) {
    let output = &vm.looking_glass;
    let selection = state
        .looking_glass_selection
        .get_or_insert_with(|| output.preferences.clone());
    let mut start = false;
    let mut stop = false;
    let mut browse = false;
    section(
        ui,
        lang.pick("出力機種", "Display model", "显示器型号", "출력 모델"),
        |ui| {
            ui.label(if output.running {
                lang.pick("出力中", "Output active", "正在输出", "출력 중")
            } else {
                lang.pick("停止", "Stopped", "已停止", "중지")
            });
            ui.add_enabled_ui(!output.running && !dialog_active, |ui| {
                let selected = output
                    .devices
                    .iter()
                    .find(|device| device.id == selection.device_id);
                egui::ComboBox::from_id_salt("looking_glass_model")
                    .width(ui.available_width().min(420.0))
                    .selected_text(selected.map_or("—", |device| device.name.as_str()))
                    .show_ui(ui, |ui| {
                        for device in &output.devices {
                            ui.selectable_value(
                                &mut selection.device_id,
                                device.id.clone(),
                                &device.name,
                            );
                        }
                    });
            });
            if let Some(device) = output
                .devices
                .iter()
                .find(|device| device.id == selection.device_id)
            {
                let [width, height] = device.screen;
                ui.label(format!("{width} × {height} px · {:.4}:1", device.aspect));
                if device.legacy {
                    ui.label(lang.pick(
                        "旧モデル（既存の機器向け）",
                        "Legacy model (for existing owners)",
                        "旧型号（现有设备）",
                        "이전 모델 (기존 기기용)",
                    ));
                }
                if let Some(quilt) = &device.quilt {
                    ui.label(format!(
                        "Quilt: {} × {} px · {} × {} = {} views",
                        quilt.width,
                        quilt.height,
                        quilt.columns,
                        quilt.rows,
                        quilt.columns * quilt.rows
                    ));
                    ui.label(lang.pick(
                        "実機のキャリブレーション（visual.json）",
                        "This display's calibration (visual.json)",
                        "此显示器的校准文件（visual.json）",
                        "이 디스플레이의 보정 파일 (visual.json)",
                    ));
                    ui.horizontal_wrapped(|ui| {
                        ui.label(if selection.calibration_path.as_os_str().is_empty() {
                            "—".into()
                        } else {
                            selection.calibration_path.display().to_string()
                        });
                        browse = ui
                            .add_enabled(
                                !output.running && !dialog_active,
                                egui::Button::new(lang.pick(
                                    "ファイルを選択…",
                                    "Choose file…",
                                    "选择文件…",
                                    "파일 선택…",
                                )),
                            )
                            .clicked();
                    });
                    ui.label(lang.pick("この実機の LKG_calibration/visual.json を選択してください。機種名だけではレンズ補正値は決まりません。", "Select this unit's LKG_calibration/visual.json. Lens calibration is unique to each unit.", "请选择此设备的 LKG_calibration/visual.json。镜头校准值因设备而异。", "이 기기의 LKG_calibration/visual.json을 선택하세요. 렌즈 보정값은 기기마다 다릅니다."));
                } else if device.format == DisplayFormat::Hld {
                    ui.label(lang.pick(
                        "HLD：Quilt不要。縦向きの通常映像を出力します。",
                        "HLD: standard portrait video; no quilt needed.",
                        "HLD：输出普通竖向影像，无需Quilt。",
                        "HLD: Quilt 없이 일반 세로 영상을 출력합니다.",
                    ));
                } else {
                    ui.label(lang.pick("musubiは保存済み動画・写真専用です。外部モニター入力がないため、ライブ出力は利用できません。", "musubi plays stored photos and videos. It has no external monitor input for live output.", "musubi仅播放存储的照片和视频，没有实时显示器输入。", "musubi는 저장된 사진과 동영상 전용이며 라이브 모니터 입력이 없습니다."));
                }
                ui.horizontal(|ui| {
                    let can_start = device.format != DisplayFormat::Standalone
                        && (device.format != DisplayFormat::LightField
                            || !selection.calibration_path.as_os_str().is_empty());
                    start = ui
                        .add_enabled(
                            can_start && !output.running && !dialog_active,
                            egui::Button::new(lang.pick(
                                "出力開始",
                                "Start output",
                                "开始输出",
                                "출력 시작",
                            )),
                        )
                        .clicked();
                    stop = ui
                        .add_enabled(
                            output.running,
                            egui::Button::new(lang.pick(
                                "出力停止",
                                "Stop output",
                                "停止输出",
                                "출력 중지",
                            )),
                        )
                        .clicked();
                });
                ui.label(lang.pick("出力ウィンドウを対象ディスプレイへ移動し、F11で全画面表示。OSは拡張表示・上記の解像度に設定してください。F10：Quilt確認、Esc：出力停止。", "Move the output window to the display and press F11. Use extended desktop at the resolution above. F10: quilt preview; Esc: stop.", "将输出窗口移到显示器并按F11。系统设为扩展显示及上述分辨率。F10：Quilt预览，Esc：停止。", "출력 창을 디스플레이로 옮기고 F11을 누르세요. 위 해상도의 확장 화면을 사용하세요. F10: Quilt, Esc: 중지."));
                ui.small(lang.pick(
                    "Looking Glass出力は通常描画60fpsの保証対象外です。",
                    "Looking Glass output is outside the normal 60 fps rendering target.",
                    "Looking Glass输出不在通常60fps渲染目标内。",
                    "Looking Glass 출력은 일반 60fps 렌더링 목표에서 제외됩니다.",
                ));
            }
            if let Some(error) = &output.error {
                ui.colored_label(Color32::DARK_RED, error);
            }
        },
    );
    let preferences = selection.clone();
    state.calibration_requested |= browse;
    if start {
        state.emit(actions, UiAction::StartLookingGlass(preferences));
    }
    if stop {
        state.emit(actions, UiAction::StopLookingGlass);
    }
}
