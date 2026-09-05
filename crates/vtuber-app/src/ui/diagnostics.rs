//! Diagnostics pane — the single home for live tracking health and system
//! metrics, rendered as macOS-style grouped lists.
//!
//! Lifecycle / tracking / face status lives only here, so each piece of
//! information is shown in exactly one place across the whole UI.

use bevy_egui::egui::{Color32, RichText, Ui};

use crate::diagnostics::DiagnosticsSnapshot;
use crate::ui_model::{TrackingState, UiViewModel};

use super::widgets::{
    ALERT_RED, INFO_BLUE, LABEL, OK_GREEN, SECONDARY, app_lifecycle_text, caption, group,
    info_row, row_separator, section_caption, status_text,
};

/// Status color and label for the tracking state.
fn tracking_state_color(state: TrackingState) -> (Color32, &'static str) {
    match state {
        TrackingState::Tracking => (OK_GREEN, "Tracking"),
        TrackingState::Lost => (ALERT_RED, "Lost"),
        TrackingState::Initializing => (INFO_BLUE, "Initializing"),
        TrackingState::Idle => (SECONDARY, "Idle"),
    }
}

/// Live tracking health: lifecycle, tracking state, face detection, and
/// confidence.
fn render_live_status_group(ui: &mut Ui, vm: &UiViewModel) {
    section_caption(ui, "Live status");
    group(ui, |ui| {
        let (lc, lc_label) = app_lifecycle_text(vm.lifecycle);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Session").size(13.0).color(SECONDARY));
            ui.with_layout(bevy_egui::egui::Layout::right_to_left(
                bevy_egui::egui::Align::Center,
            ), |ui| {
                status_text(ui, lc, lc_label);
            });
        });
        row_separator(ui);
        let (tc, tlabel) = tracking_state_color(vm.tracking.state);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Tracking").size(13.0).color(SECONDARY));
            ui.with_layout(bevy_egui::egui::Layout::right_to_left(
                bevy_egui::egui::Align::Center,
            ), |ui| {
                let face = if vm.tracking.face_detected {
                    "Face detected"
                } else {
                    "No face"
                };
                status_text(ui, tc, &format!("{tlabel} · {face}"));
            });
        });
        row_separator(ui);
        info_row(
            ui,
            "Confidence",
            &format!("{:.0}%", vm.tracking.confidence * 100.0),
            LABEL,
        );
        row_separator(ui);
        let cal = &vm.calibration;
        let (cc, clabel) = if cal.is_complete {
            (OK_GREEN, "Calibrated".to_string())
        } else if cal.is_calibrating {
            (
                INFO_BLUE,
                format!("Calibrating {}/{}", cal.samples_collected, cal.samples_target),
            )
        } else {
            (SECONDARY, "Not calibrated".to_string())
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new("Calibration").size(13.0).color(SECONDARY));
            ui.with_layout(bevy_egui::egui::Layout::right_to_left(
                bevy_egui::egui::Align::Center,
            ), |ui| {
                status_text(ui, cc, &clabel);
            });
        });
        if vm.tracking.state == TrackingState::Lost {
            ui.add_space(4.0);
            caption(ui, "Face lost — attempting recovery. Check lighting and framing.");
        }
    });
}

/// Render the Diagnostics pane.
pub fn render_diagnostics_pane(ui: &mut Ui, vm: &UiViewModel, diagnostics: &DiagnosticsSnapshot) {
    render_live_status_group(ui, vm);

    section_caption(ui, "Performance");
    group(ui, |ui| {
        info_row(
            ui,
            "Render FPS",
            &format!("{:.1}", diagnostics.render_fps),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Process CPU",
            &diagnostics
                .process_cpu_usage
                .map(|cpu| format!("{cpu:.1}%"))
                .unwrap_or_else(|| "(none)".to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Process memory",
            &diagnostics
                .process_memory_gib
                .map(|mem| format!("{mem:.3} GiB"))
                .unwrap_or_else(|| "(none)".to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Capture rate",
            &format!("{:.1} Hz", diagnostics.capture_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Inference rate",
            &format!("{:.1} Hz", diagnostics.inference_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Detector rate",
            &format!("{:.1} Hz", diagnostics.detector_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Landmark rate",
            &format!("{:.1} Hz", diagnostics.landmark_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Tracking rate",
            &format!("{:.1} Hz", diagnostics.tracking_rate),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "No-face frames",
            &diagnostics.inference_no_face_frames.to_string(),
            LABEL,
        );
        row_separator(ui);
        info_row(ui, "Capture worker", &diagnostics.capture_state, LABEL);
        row_separator(ui);
        info_row(ui, "Inference worker", &diagnostics.inference_state, LABEL);
        row_separator(ui);
        info_row(
            ui,
            "Slot overwrites",
            &diagnostics.slot_overwrites.to_string(),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Avatar frames applied",
            &diagnostics.avatar_frames_applied.to_string(),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Avatar frames skipped",
            &diagnostics.avatar_frames_skipped.to_string(),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Capture→apply p50",
            &diagnostics
                .capture_to_apply_p50_ms
                .map(|v| format!("{v:.2} ms"))
                .unwrap_or_else(|| "(none)".to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Capture→apply p95",
            &diagnostics
                .capture_to_apply_p95_ms
                .map(|v| format!("{v:.2} ms"))
                .unwrap_or_else(|| "(none)".to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(ui, "Metrics export", &diagnostics.metrics_export_status, LABEL);
        row_separator(ui);
        info_row(
            ui,
            "Export samples",
            &format!("{} / 31", diagnostics.metrics_export_samples),
            LABEL,
        );
    });

    if !diagnostics.stage_timings.is_empty() {
        section_caption(ui, "Stage timings");
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
        section_caption(ui, "Stage percentiles");
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

    section_caption(ui, "Model & camera");
    group(ui, |ui| {
        info_row(
            ui,
            "Model hash",
            diagnostics.model_hash.as_deref().unwrap_or("(none)"),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Pipeline",
            diagnostics.pipeline_id.as_deref().unwrap_or("(none)"),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "ROI state",
            diagnostics.roi_state.as_deref().unwrap_or("(none)"),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Detector confidence",
            &diagnostics
                .detector_confidence
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "(none)".to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Camera backend",
            diagnostics.camera_backend.as_deref().unwrap_or("(none)"),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Tracking backend",
            diagnostics.tracking_backend.as_deref().unwrap_or("(none)"),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Tracking contract",
            diagnostics.tracking_contract.as_deref().unwrap_or("(none)"),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Avatar capabilities",
            diagnostics.avatar_capabilities.as_deref().unwrap_or("(none)"),
            LABEL,
        );
    });

    section_caption(ui, "Tracking");
    group(ui, |ui| {
        info_row(ui, "State", &diagnostics.tracking_state, LABEL);
        row_separator(ui);
        info_row(
            ui,
            "Auto-neutral",
            diagnostics.auto_neutral_state.as_deref().unwrap_or("(none)"),
            LABEL,
        );
        if let Some(ready) = diagnostics.face_tracking_calibration_ready {
            row_separator(ui);
            info_row(ui, "Calibration ready", if ready { "yes" } else { "no" }, LABEL);
        }
        row_separator(ui);
        info_row(
            ui,
            "Latest residual",
            &diagnostics
                .face_tracking_latest_residual
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "(none)".to_string()),
            LABEL,
        );
        row_separator(ui);
        info_row(
            ui,
            "Added latency",
            &diagnostics
                .face_tracking_added_latency_ms
                .map(|v| format!("{v:.1} ms"))
                .unwrap_or_else(|| "(none)".to_string()),
            LABEL,
        );
    });

    if diagnostics.last_error_code.is_some()
        || diagnostics.inference_failure_stage.is_some()
        || diagnostics.last_error.is_some()
    {
        section_caption(ui, "Errors");
        group(ui, |ui| {
            if let Some(code) = &diagnostics.last_error_code {
                info_row(ui, "Error code", code, ALERT_RED);
            }
            if let Some(stage) = &diagnostics.inference_failure_stage {
                if diagnostics.last_error_code.is_some() {
                    row_separator(ui);
                }
                info_row(ui, "Failure stage", stage, LABEL);
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
        assert_eq!(tracking_state_color(TrackingState::Tracking).1, "Tracking");
        assert_eq!(tracking_state_color(TrackingState::Lost).1, "Lost");
        assert_eq!(
            tracking_state_color(TrackingState::Initializing).1,
            "Initializing"
        );
        assert_eq!(tracking_state_color(TrackingState::Idle).1, "Idle");
    }
}
