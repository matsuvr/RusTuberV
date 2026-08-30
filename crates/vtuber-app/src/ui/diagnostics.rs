//! Diagnostics screen rendering.

use bevy_egui::egui::Ui;

use crate::diagnostics::DiagnosticsSnapshot;
use crate::ui_model::UiViewModel;

/// Render the Diagnostics screen.
pub fn render_diagnostics_screen(ui: &mut Ui, vm: &UiViewModel, diagnostics: &DiagnosticsSnapshot) {
    ui.heading("Diagnostics");
    ui.separator();

    // Performance metrics.
    ui.heading("Performance");
    bevy_egui::egui::Grid::new("perf_grid")
        .num_columns(2)
        .spacing([40.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label("Render FPS");
            ui.label(format!("{:.1}", diagnostics.render_fps));
            ui.end_row();

            ui.label("Process CPU");
            ui.label(
                diagnostics
                    .process_cpu_usage
                    .map(|value| format!("{value:.1}%"))
                    .unwrap_or_else(|| "(none)".to_string()),
            );
            ui.end_row();

            ui.label("Process memory");
            ui.label(
                diagnostics
                    .process_memory_gib
                    .map(|value| format!("{value:.3} GiB"))
                    .unwrap_or_else(|| "(none)".to_string()),
            );
            ui.end_row();

            ui.label("Capture rate");
            ui.label(format!("{:.1} Hz", diagnostics.capture_rate));
            ui.end_row();

            ui.label("Inference rate");
            ui.label(format!("{:.1} Hz", diagnostics.inference_rate));
            ui.end_row();

            ui.label("Detector rate");
            ui.label(format!("{:.1} Hz", diagnostics.detector_rate));
            ui.end_row();

            ui.label("Landmark rate");
            ui.label(format!("{:.1} Hz", diagnostics.landmark_rate));
            ui.end_row();

            ui.label("No-face frames");
            ui.label(format!("{}", diagnostics.inference_no_face_frames));
            ui.end_row();

            ui.label("Tracking rate");
            ui.label(format!("{:.1} Hz", diagnostics.tracking_rate));
            ui.end_row();

            ui.label("Capture worker");
            ui.label(&diagnostics.capture_state);
            ui.end_row();

            ui.label("Inference worker");
            ui.label(&diagnostics.inference_state);
            ui.end_row();

            ui.label("Slot overwrites");
            ui.label(format!("{}", diagnostics.slot_overwrites));
            ui.end_row();

            ui.label("Avatar frames applied");
            ui.label(format!("{}", diagnostics.avatar_frames_applied));
            ui.end_row();

            ui.label("Avatar frames skipped");
            ui.label(format!("{}", diagnostics.avatar_frames_skipped));
            ui.end_row();

            ui.label("Capture-to-apply p50");
            ui.label(
                diagnostics
                    .capture_to_apply_p50_ms
                    .map(|value| format!("{value:.2} ms"))
                    .unwrap_or_else(|| "(none)".to_string()),
            );
            ui.end_row();

            ui.label("Capture-to-apply p95");
            ui.label(
                diagnostics
                    .capture_to_apply_p95_ms
                    .map(|value| format!("{value:.2} ms"))
                    .unwrap_or_else(|| "(none)".to_string()),
            );
            ui.end_row();

            ui.label("Metrics export");
            ui.label(&diagnostics.metrics_export_status);
            ui.end_row();

            ui.label("Export samples");
            ui.label(format!("{} / 31", diagnostics.metrics_export_samples));
            ui.end_row();
        });
    ui.separator();

    // Stage timings.
    if !diagnostics.stage_timings.is_empty() {
        ui.heading("Stage Timings");
        bevy_egui::egui::Grid::new("timing_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                for (name, duration) in &diagnostics.stage_timings {
                    ui.label(name);
                    ui.label(format!("{:.2} ms", duration));
                    ui.end_row();
                }
            });
        ui.separator();
    }

    if !diagnostics.stage_percentiles.is_empty() {
        ui.heading("Stage Percentiles");
        bevy_egui::egui::Grid::new("percentile_grid")
            .num_columns(3)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Stage");
                ui.label("p50");
                ui.label("p95");
                ui.end_row();
                for (name, p50, p95) in &diagnostics.stage_percentiles {
                    ui.label(name);
                    ui.label(format!("{p50:.2} ms"));
                    ui.label(format!("{p95:.2} ms"));
                    ui.end_row();
                }
            });
        ui.separator();
    }

    // Model and camera info.
    ui.heading("Model & Camera");
    bevy_egui::egui::Grid::new("info_grid")
        .num_columns(2)
        .spacing([40.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label("Model hash");
            ui.label(diagnostics.model_hash.as_deref().unwrap_or("(none)"));
            ui.end_row();

            ui.label("Pipeline");
            ui.label(diagnostics.pipeline_id.as_deref().unwrap_or("(none)"));
            ui.end_row();

            ui.label("ROI state");
            ui.label(diagnostics.roi_state.as_deref().unwrap_or("(none)"));
            ui.end_row();

            ui.label("Detector confidence");
            ui.label(
                diagnostics
                    .detector_confidence
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "(none)".to_string()),
            );
            ui.end_row();

            ui.label("Camera backend");
            ui.label(diagnostics.camera_backend.as_deref().unwrap_or("(none)"));
            ui.end_row();

            ui.label("Tracking backend");
            ui.label(diagnostics.tracking_backend.as_deref().unwrap_or("(none)"));
            ui.end_row();

            ui.label("Tracking contract");
            ui.label(diagnostics.tracking_contract.as_deref().unwrap_or("(none)"));
            ui.end_row();

            ui.label("Avatar capabilities");
            ui.label(
                diagnostics
                    .avatar_capabilities
                    .as_deref()
                    .unwrap_or("(none)"),
            );
            ui.end_row();
        });
    ui.separator();

    // Tracking state.
    ui.heading("Tracking");
    ui.label(format!("State: {}", diagnostics.tracking_state));
    ui.label(format!(
        "Auto-neutral: {}",
        diagnostics
            .auto_neutral_state
            .as_deref()
            .unwrap_or("(none)")
    ));
    if let Some(ready) = diagnostics.face_tracking_calibration_ready {
        ui.label(format!(
            "Calibration ready: {}",
            if ready { "yes" } else { "no" }
        ));
    }
    ui.label(format!(
        "Requested backend: {}",
        diagnostics
            .face_tracking_requested
            .as_deref()
            .unwrap_or("(none)")
    ));
    ui.label(format!(
        "Output authority: {}",
        diagnostics
            .face_tracking_authority
            .as_deref()
            .unwrap_or("(none)")
    ));
    if let Some(reason) = diagnostics.face_tracking_fallback_reason.as_deref() {
        ui.colored_label(
            bevy_egui::egui::Color32::YELLOW,
            format!("Fallback reason: {reason}"),
        );
    }
    ui.label(format!(
        "Latest residual: {}",
        diagnostics
            .face_tracking_latest_residual
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "(none)".to_owned())
    ));
    ui.label(format!(
        "Added latency: {}",
        diagnostics
            .face_tracking_added_latency_ms
            .map(|value| format!("{value:.1} ms"))
            .unwrap_or_else(|| "(none)".to_owned())
    ));
    ui.separator();

    // Last error.
    if let Some(code) = &diagnostics.last_error_code {
        ui.label(format!("Error code: {code}"));
    }
    if let Some(stage) = &diagnostics.inference_failure_stage {
        ui.label(format!("Inference failure stage: {stage}"));
    }
    if let Some(error) = &diagnostics.last_error {
        ui.heading("Last Error");
        ui.colored_label(bevy_egui::egui::Color32::LIGHT_RED, error);
    }

    // Current app state summary.
    ui.separator();
    ui.heading("Current State");
    ui.label(format!("Screen: {:?}", vm.screen));
    ui.label(format!("Lifecycle: {:?}", vm.lifecycle));
    ui.label(format!(
        "Camera: {}",
        vm.camera
            .selected_index
            .map(|i| vm
                .camera
                .available_cameras
                .get(i)
                .map(|c| c.name.as_str())
                .unwrap_or("?"))
            .unwrap_or("none")
    ));
    ui.label(format!(
        "Avatar: {}",
        vm.avatar
            .imported_model
            .as_ref()
            .map(|m| m.name.as_str())
            .unwrap_or("none")
    ));
}
