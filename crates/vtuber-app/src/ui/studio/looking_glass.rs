//! Connected Looking Glass selection; factory calibration comes from Bridge.

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
    if !output.discovered {
        state.emit(actions, UiAction::RefreshLookingGlass);
    }
    let selection = state
        .looking_glass_selection
        .get_or_insert_with(|| output.preferences.clone());
    if output.discovered
        && !output.loading
        && !output
            .devices
            .iter()
            .any(|device| device.id == selection.device_id)
        && let Some(device) = output
            .devices
            .iter()
            .find(|device| device.format == DisplayFormat::LightField)
    {
        selection.device_id = device.id.clone();
    }
    let mut start = false;
    let mut stop = false;
    let mut refresh = false;
    section(
        ui,
        lang.pick("出力先", "Output display", "输出设备", "출력 디스플레이"),
        |ui| {
            ui.horizontal(|ui| {
                if output.loading {
                    ui.spinner();
                    ui.label(lang.pick(
                        "Bridgeから取得中…",
                        "Querying Bridge…",
                        "正在查询Bridge…",
                        "Bridge에서 가져오는 중…",
                    ));
                } else {
                    ui.label(if output.running {
                        lang.pick("出力中", "Output active", "正在输出", "출력 중")
                    } else {
                        lang.pick("停止", "Stopped", "已停止", "중지")
                    });
                }
                refresh = ui
                    .add_enabled(
                        !output.loading && !output.running && !dialog_active,
                        egui::Button::new(lang.pick(
                            "再検出",
                            "Refresh devices",
                            "重新检测",
                            "다시 검색",
                        )),
                    )
                    .clicked();
            });
            ui.add_enabled_ui(!output.running && !output.loading && !dialog_active, |ui| {
                let selected = output
                    .devices
                    .iter()
                    .find(|device| device.id == selection.device_id);
                egui::ComboBox::from_id_salt("looking_glass_device")
                    .width(ui.available_width().min(420.0))
                    .selected_text(selected.map_or(
                        lang.pick("機器を選択", "Select display", "选择设备", "기기 선택"),
                        |device| device.name.as_str(),
                    ))
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
                        "Bridgeから実機の工場校正値を自動取得します。",
                        "Factory calibration is retrieved automatically from Bridge.",
                        "通过Bridge自动获取此设备的出厂校准值。",
                        "Bridge에서 기기의 공장 보정 값을 자동으로 가져옵니다.",
                    ));
                } else if device.format == DisplayFormat::Hld {
                    ui.label(lang.pick(
                        "HLD：Quilt不要。縦向きの通常映像を出力します。",
                        "HLD: standard portrait video; no quilt needed.",
                        "HLD：输出普通竖向影像，无需Quilt。",
                        "HLD: Quilt 없이 일반 세로 영상을 출력합니다.",
                    ));
                }
            }
            ui.horizontal(|ui| {
                start = ui
                    .add_enabled(
                        !output.running
                            && !output.loading
                            && !dialog_active
                            && output
                                .devices
                                .iter()
                                .any(|device| device.id == selection.device_id),
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
                        output.running || output.loading,
                        egui::Button::new(lang.pick(
                            "出力停止",
                            "Stop output",
                            "停止输出",
                            "출력 중지",
                        )),
                    )
                    .clicked();
            });
            ui.label(lang.pick("出力ウィンドウを対象ディスプレイへ移動し、F11で全画面表示。OSは拡張表示・実機の解像度に設定してください。F10：Quilt確認、Esc：出力停止。", "Move the output window to the display and press F11. Use extended desktop at its native resolution. F10: quilt preview; Esc: stop.", "将输出窗口移到显示器并按F11。系统设为扩展显示及设备原生分辨率。F10：Quilt预览，Esc：停止。", "출력 창을 디스플레이로 옮기고 F11을 누르세요. 기기의 기본 해상도로 확장 화면을 사용하세요. F10: Quilt, Esc: 중지."));
            ui.small(lang.pick(
                "Looking Glass出力は通常描画60fpsの保証対象外です。",
                "Looking Glass output is outside the normal 60 fps rendering target.",
                "Looking Glass输出不在通常60fps渲染目标内。",
                "Looking Glass 출력은 일반 60fps 렌더링 목표에서 제외됩니다.",
            ));
            if let Some(error) = &output.error {
                ui.colored_label(Color32::DARK_RED, error);
            }
        },
    );
    let preferences = selection.clone();
    if refresh {
        state.emit(actions, UiAction::RefreshLookingGlass);
    }
    if start {
        state.emit(actions, UiAction::StartLookingGlass(preferences));
    }
    if stop {
        state.emit(actions, UiAction::StopLookingGlass);
    }
}
