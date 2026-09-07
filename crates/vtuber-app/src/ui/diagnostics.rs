//! Diagnostics pane — the single home for live tracking health and system
//! metrics, rendered as macOS-style grouped lists.
//!
//! Lifecycle / tracking / face status lives only here, so each piece of
//! information is shown in exactly one place across the whole UI.

use bevy_egui::egui::{Color32, RichText, Ui};

use crate::diagnostics::DiagnosticsSnapshot;
use crate::settings::UiLanguage;
use crate::ui_model::{TrackingState, UiViewModel};

use super::widgets::{
    ALERT_RED, INFO_BLUE, LABEL, OK_GREEN, SECONDARY, app_lifecycle_text, caption, group,
    info_row, row_separator, section_caption, status_text,
};

/// Status color and label for the tracking state.
fn tracking_state_color(
    state: TrackingState,
    lang: UiLanguage,
) -> (Color32, &'static str) {
    match state {
        TrackingState::Tracking => (OK_GREEN, lang.pick("トラッキング中", "Tracking")),
        TrackingState::Lost => (ALERT_RED, lang.pick("見失い", "Lost")),
        TrackingState::Initializing => (INFO_BLUE, lang.pick("初期化中", "Initializing")),
        TrackingState::Idle => (SECONDARY, lang.pick("待機", "Idle")),
    }
}

/// Localized value label for the capture worker state string.
fn capture_state_label(raw: &str, lang: UiLanguage) -> &str {
    match lang {
        UiLanguage::En => raw,
        UiLanguage::Ja => match raw {
            "Idle" => "待機",
            "Selected" => "選択済み",
            "Starting" => "開始中",
            "Running" => "実行中",
            "Reconnecting" => "再接続中",
            "BackOff" => "再試行待ち",
            "Stopping" => "停止中",
            other => other,
        },
    }
}

/// Localized value label for the inference worker state string.
fn inference_state_label(raw: &str, lang: UiLanguage) -> &str {
    match lang {
        UiLanguage::En => raw,
        UiLanguage::Ja => match raw {
            "Idle" => "待機",
            "LoadingModel" => "モデル読み込み中",
            "Running" => "実行中",
            "Stopping" => "停止中",
            "Failed" => "失敗",
            other => other,
        },
    }
}

/// Localized value label for the tracking state string.
fn tracking_state_label(raw: &str, lang: UiLanguage) -> &str {
    match lang {
        UiLanguage::En => raw,
        UiLanguage::Ja => match raw {
            "Idle" => "待機",
            "Starting" => "開始中",
            "Initializing" => "初期化中",
            "Tracking" => "トラッキング中",
            "Lost" => "見失い",
            other => other,
        },
    }
}

/// Localized value label for the auto-neutral state string.
fn auto_neutral_state_label(raw: &str, lang: UiLanguage) -> &str {
    match lang {
        UiLanguage::En => raw,
        UiLanguage::Ja => match raw {
            "WaitingForFace" => "顔待ち",
            "Ready" => "準備完了",
            other => other,
        },
    }
}

/// Localized placeholder for absent numeric or textual values.
fn none_value(lang: UiLanguage) -> &'static str {
    lang.pick("（なし）", "(none)")
}

/// Live tracking health: lifecycle, tracking state, face detection, and
/// confidence.
fn render_live_status_group(ui: &mut Ui, vm: &UiViewModel, lang: UiLanguage) {
    section_caption(ui, lang.pick("ライブステータス", "Live status"));
    group(ui, |ui| {
        let (lc, lc_label) = app_lifecycle_text(vm.lifecycle, lang);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(lang.pick("セッション", "Session"))
                    .size(13.0)
                    .color(SECONDARY),
            );
            ui.with_layout(bevy_egui::egui::Layout::right_to_left(
                bevy_egui::egui::Align::Center,
            ), |ui| {
                status_text(ui, lc, lc_label);
            });
        });
        row_separator(ui);
        let (tc, tlabel) = tracking_state_color(vm.tracking.state, lang);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(lang.pick("トラッキング", "Tracking"))
                    .size(13.0)
                    .color(SECONDARY),
            );
            ui.with_layout(bevy_egui::egui::Layout::right_to_left(
                bevy_egui::egui::Align::Center,
            ), |ui| {
                let face = if vm.tracking.face_detected {
                    lang.pick("顔検出", "Face detected")
                } else {
                    lang.pick("顔なし", "No face")
                };
                status_text(ui, tc, &format!("{tlabel} · {face}"));
            });
        });
        row_separator(ui);
        info_row(
            ui,
            lang.pick("信頼度", "Confidence"),
            &format!("{:.0}%", vm.tracking.confidence * 100.0),
            LABEL,
        );
        row_separator(ui);
        let cal = &vm.calibration;
        let (cc, clabel) = if cal.is_complete {
            (OK_GREEN, lang.pick("キャリブレーション完了", "Calibrated").to_string())
        } else if cal.is_calibrating {
            (
                INFO_BLUE,
                match lang {
                    UiLanguage::Ja => format!(
                        "キャリブレーション中 {}/{}",
                        cal.samples_collected, cal.samples_target
                    ),
                    UiLanguage::En => format!(
                        "Calibrating {}/{}",
                        cal.samples_collected, cal.samples_target
                    ),
                },
            )
        } else {
            (SECONDARY, lang.pick("未キャリブレーション", "Not calibrated").to_string())
        };
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(lang.pick("キャリブレーション", "Calibration"))
                    .size(13.0)
                    .color(SECONDARY),
            );
            ui.with_layout(bevy_egui::egui::Layout::right_to_left(
                bevy_egui::egui::Align::Center,
            ), |ui| {
                status_text(ui, cc, &clabel);
            });
        });
        if vm.tracking.state == TrackingState::Lost {
            ui.add_space(4.0);
            caption(
                ui,
                lang.pick(
                    "顔を見失いました — 照明とフレーミングを確認してください。",
                    "Face lost — attempting recovery. Check lighting and framing.",
                ),
            );
        }
    });
}

/// Render the Diagnostics pane.
pub fn render_diagnostics_pane(
    ui: &mut Ui,
    vm: &UiViewModel,
    diagnostics: &DiagnosticsSnapshot,
    lang: UiLanguage,
) {
    render_live_status_group(ui, vm, lang);

    section_caption(ui, lang.pick("パフォーマンス", "Performance"));
    group(ui, |ui| {
        info_row(
            ui,
            lang.pick("描画 FPS", "Render FPS"),
            &format!("{:.1}", diagnostics.render_fps),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("プロセス CPU", "Process CPU"),
            &diagnostics
                .process_cpu_usage
                .map(|cpu| format!("{cpu:.1}%"))
                .unwrap_or_else(|| none_value(lang).to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("プロセスメモリ", "Process memory"),
            &diagnostics
                .process_memory_gib
                .map(|mem| format!("{mem:.3} GiB"))
                .unwrap_or_else(|| none_value(lang).to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("キャプチャレート", "Capture rate"),
            &format!("{:.1} Hz", diagnostics.capture_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("推論レート", "Inference rate"),
            &format!("{:.1} Hz", diagnostics.inference_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("検出レート", "Detector rate"),
            &format!("{:.1} Hz", diagnostics.detector_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("ランドマークレート", "Landmark rate"),
            &format!("{:.1} Hz", diagnostics.landmark_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("トラッキングレート", "Tracking rate"),
            &format!("{:.1} Hz", diagnostics.tracking_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("顔なしフレーム", "No-face frames"),
            &diagnostics.inference_no_face_frames.to_string(),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("キャプチャワーカー", "Capture worker"),
            capture_state_label(&diagnostics.capture_state, lang),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("推論ワーカー", "Inference worker"),
            inference_state_label(&diagnostics.inference_state, lang),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("スロット上書き", "Slot overwrites"),
            &diagnostics.slot_overwrites.to_string(),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("適用されたアバターフレーム", "Avatar frames applied"),
            &diagnostics.avatar_frames_applied.to_string(),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("スキップされたアバターフレーム", "Avatar frames skipped"),
            &diagnostics.avatar_frames_skipped.to_string(),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("キャプチャ→適用 p50", "Capture→apply p50"),
            &diagnostics
                .capture_to_apply_p50_ms
                .map(|v| format!("{v:.2} ms"))
                .unwrap_or_else(|| none_value(lang).to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("キャプチャ→適用 p95", "Capture→apply p95"),
            &diagnostics
                .capture_to_apply_p95_ms
                .map(|v| format!("{v:.2} ms"))
                .unwrap_or_else(|| none_value(lang).to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("メトリクス出力", "Metrics export"),
            &diagnostics.metrics_export_status,
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("出力サンプル", "Export samples"),
            &format!("{} / 31", diagnostics.metrics_export_samples),
            LABEL,
        );
    });

    if !diagnostics.stage_timings.is_empty() {
        section_caption(ui, lang.pick("ステージ処理時間", "Stage timings"));
        group(ui, |ui| {
            for (index, (name, duration)) in diagnostics.stage_timings.iter().enumerate() {
                if index > 0 {
                    row_separator(ui);
                }
                info_row(ui, name, &format!("{duration:.2} ms"), LABEL);
            }
        });
    }

    if !diagnostics.stage_percentiles.is_empty() {
        section_caption(ui, lang.pick("ステージ百分位", "Stage percentiles"));
        group(ui, |ui| {
            for (index, (name, p50, p95)) in diagnostics.stage_percentiles.iter().enumerate() {
                if index > 0 {
                    row_separator(ui);
                }
                info_row(
                    ui,
                    name,
                    &format!("p50 {p50:.2} ms · p95 {p95:.2} ms"),
                    LABEL,
                );
            }
        });
    }

    section_caption(ui, lang.pick("モデルとカメラ", "Model & camera"));
    group(ui, |ui| {
        info_row(
            ui,
            lang.pick("モデルハッシュ", "Model hash"),
            diagnostics
                .model_hash
                .as_deref()
                .unwrap_or_else(|| none_value(lang)),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("パイプライン", "Pipeline"),
            diagnostics
                .pipeline_id
                .as_deref()
                .unwrap_or_else(|| none_value(lang)),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("ROI 状態", "ROI state"),
            diagnostics
                .roi_state
                .as_deref()
                .unwrap_or_else(|| none_value(lang)),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("検出信頼度", "Detector confidence"),
            &diagnostics
                .detector_confidence
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| none_value(lang).to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("カメラバックエンド", "Camera backend"),
            diagnostics
                .camera_backend
                .as_deref()
                .unwrap_or_else(|| none_value(lang)),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("トラッキングバックエンド", "Tracking backend"),
            diagnostics
                .tracking_backend
                .as_deref()
                .unwrap_or_else(|| none_value(lang)),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("トラッキング仕様", "Tracking contract"),
            diagnostics
                .tracking_contract
                .as_deref()
                .unwrap_or_else(|| none_value(lang)),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("アバター機能", "Avatar capabilities"),
            diagnostics
                .avatar_capabilities
                .as_deref()
                .unwrap_or_else(|| none_value(lang)),
            LABEL,
        );
    });

    section_caption(ui, lang.pick("トラッキング", "Tracking"));
    group(ui, |ui| {
        info_row(
            ui,
            lang.pick("状態", "State"),
            tracking_state_label(&diagnostics.tracking_state, lang),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("自動中立姿勢", "Auto-neutral"),
            diagnostics
                .auto_neutral_state
                .as_deref()
                .map(|raw| auto_neutral_state_label(raw, lang))
                .unwrap_or_else(|| none_value(lang)),
            LABEL,
        );
        if let Some(ready) = diagnostics.face_tracking_calibration_ready {
            row_separator(ui);
            info_row(
                ui,
                lang.pick("キャリブレーション準備", "Calibration ready"),
                if ready {
                    lang.pick("はい", "yes")
                } else {
                    lang.pick("いいえ", "no")
                },
                LABEL,
            );
        }
        row_separator(ui);
        info_row(
            ui,
            lang.pick("最新残差", "Latest residual"),
            &diagnostics
                .face_tracking_latest_residual
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| none_value(lang).to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            lang.pick("追加レイテンシ", "Added latency"),
            &diagnostics
                .face_tracking_added_latency_ms
                .map(|v| format!("{v:.1} ms"))
                .unwrap_or_else(|| none_value(lang).to_string()),
            LABEL,
        );
    });

    if diagnostics.last_error_code.is_some()
        || diagnostics.inference_failure_stage.is_some()
        || diagnostics.last_error.is_some()
    {
        section_caption(ui, lang.pick("エラー", "Errors"));
        group(ui, |ui| {
            if let Some(code) = &diagnostics.last_error_code {
                info_row(ui, lang.pick("エラーコード", "Error code"), code, ALERT_RED);
            }
            if let Some(stage) = &diagnostics.inference_failure_stage {
                if diagnostics.last_error_code.is_some() {
                    row_separator(ui);
                }
                info_row(ui, lang.pick("失敗ステージ", "Failure stage"), stage, LABEL);
            }
            if let Some(error) = &diagnostics.last_error {
                ui.add_space(4.0);
                caption(ui, error);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracking_state_colors_are_semantic() {
        assert_eq!(tracking_state_color(TrackingState::Tracking, UiLanguage::En).1, "Tracking");
        assert_eq!(tracking_state_color(TrackingState::Lost, UiLanguage::En).1, "Lost");
        assert_eq!(
            tracking_state_color(TrackingState::Initializing, UiLanguage::En).1,
            "Initializing"
        );
        assert_eq!(tracking_state_color(TrackingState::Idle, UiLanguage::En).1, "Idle");
    }

    #[test]
    fn japanese_labels_are_used_for_states_and_placeholders() {
        assert_eq!(tracking_state_color(TrackingState::Tracking, UiLanguage::Ja).1, "トラッキング中");
        assert_eq!(capture_state_label("Running", UiLanguage::Ja), "実行中");
        assert_eq!(inference_state_label("LoadingModel", UiLanguage::Ja), "モデル読み込み中");
        assert_eq!(tracking_state_label("Lost", UiLanguage::Ja), "見失い");
        assert_eq!(auto_neutral_state_label("WaitingForFace", UiLanguage::Ja), "顔待ち");
        assert_eq!(none_value(UiLanguage::Ja), "（なし）");
    }

    #[test]
    fn unknown_state_strings_pass_through_untranslated() {
        assert_eq!(capture_state_label("Mystery", UiLanguage::Ja), "Mystery");
        assert_eq!(inference_state_label("Mystery", UiLanguage::Ja), "Mystery");
    }
}
