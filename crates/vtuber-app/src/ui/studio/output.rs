//! NDI output controls.
use super::{primary_button, section};
use crate::actions::{ActionQueue, UiAction};
use crate::settings::UiLanguage;
use crate::ui::shell::UiState;
use crate::ui_model::{NdiOutputUiState, UiViewModel};
use bevy_egui::egui::{self, Ui};

pub(super) fn output_page(
    actions: &mut ActionQueue,
    ui: &mut Ui,
    vm: &UiViewModel,
    state: &mut UiState,
    lang: UiLanguage,
) {
    section(
        ui,
        lang.pick(
            "NDI — アバターのみ",
            "NDI — avatar only",
            "NDI — 仅虚拟形象",
            "NDI — 아바타만",
        ),
        |ui| {
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
            }
            ui.hyperlink_to(
                lang.pick(
                    "NDI®公式サイト",
                    "NDI® website",
                    "NDI®官方网站",
                    "NDI® 공식 사이트",
                ),
                "https://ndi.video",
            );
            ui.small("NDI® is a registered trademark of Vizrt NDI AB.");
            ui.horizontal_wrapped(|ui| {
                if primary_button(
                    ui,
                    lang.pick("NDI送信を開始", "Start NDI", "开始NDI发送", "NDI 전송 시작"),
                    vm.can_start_ndi_output() && vm.ndi_output.runtime_installed,
                )
                .clicked()
                {
                    state.emit(actions, UiAction::StartNdiOutput);
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
                    state.emit(actions, UiAction::StopNdiOutput);
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
}
