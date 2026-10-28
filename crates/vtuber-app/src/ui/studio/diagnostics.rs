//! Tracking status and detailed diagnostics.
use super::{avatar_label, section};
use crate::diagnostics::DiagnosticsSnapshot;
use crate::settings::UiLanguage;
use crate::ui_model::{TrackingState, UiViewModel};
use bevy_egui::egui::{self, RichText, Ui};

pub(super) fn diagnostics_page(
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
