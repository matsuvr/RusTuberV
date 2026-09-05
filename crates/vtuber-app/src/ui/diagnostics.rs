//! Diagnostics screen — now the single home for live tracking health and
//! system metrics that previously lived partially on the Live tab.
//!
//! Status (lifecycle, tracking, confidence, face detection) was moved here
//! from Live per the 2-tab consolidation, while Performance / Model /
//! Tracking / Errors remain. Cards give a consistent visual rhythm with the
//! Controls tab.

use bevy_egui::egui::{Color32, CornerRadius, Frame, Margin, ProgressBar, RichText, Stroke, Ui};

use crate::diagnostics::DiagnosticsSnapshot;
use crate::ui_model::{TrackingState, UiViewModel};

// ── style helpers ──────────────────────────────────────────────────────────

fn card_frame() -> Frame {
    Frame::new()
        .fill(Color32::from_rgba_unmultiplied(28, 31, 44, 220))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::same(12))
        .outer_margin(Margin {
            left: 0,
            right: 8,
            top: 4,
            bottom: 4,
        })
        .stroke(Stroke::new(
            1.0,
            Color32::from_rgba_unmultiplied(255, 255, 255, 14),
        ))
}

fn section_header(ui: &mut Ui, icon: &str, title: &str, subtitle: &str) {
    ui.horizontal(|ui| {
        let icon_bg = Frame::new()
            .fill(Color32::from_rgba_unmultiplied(99, 102, 241, 30))
            .corner_radius(CornerRadius::same(7))
            .inner_margin(Margin::symmetric(7, 4));
        icon_bg.show(ui, |ui| {
            ui.label(
                RichText::new(icon)
                    .size(13.0)
                    .color(Color32::from_rgb(165, 180, 252)),
            );
        });
        ui.vertical(|ui| {
            ui.label(
                RichText::new(title)
                    .size(13.0)
                    .strong()
                    .color(Color32::from_rgb(226, 232, 240)),
            );
            ui.label(
                RichText::new(subtitle)
                    .size(10.0)
                    .color(Color32::from_rgb(148, 163, 184)),
            );
        });
    });
    ui.add_space(8.0);
}

fn tracking_state_color(state: TrackingState) -> (Color32, &'static str) {
    match state {
        TrackingState::Tracking => (Color32::from_rgb(74, 222, 128), "Tracking"),
        TrackingState::Lost => (Color32::from_rgb(248, 113, 113), "Lost"),
        TrackingState::Initializing => (Color32::from_rgb(96, 165, 250), "Initializing"),
        TrackingState::Idle => (Color32::from_rgb(100, 116, 139), "Idle"),
    }
}

fn badge(ui: &mut Ui, text: &str, color: Color32) {
    Frame::new()
        .fill(Color32::from_rgba_unmultiplied(
            color.r(),
            color.g(),
            color.b(),
            32,
        ))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(8, 3))
        .stroke(Stroke::new(
            1.0,
            Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 70),
        ))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("●").size(9.0).color(color));
                ui.label(RichText::new(text).size(11.0).color(color).strong());
            });
        });
}

fn kv_row(ui: &mut Ui, key: &str, value: &str) {
    ui.add(
        bevy_egui::egui::Label::new(
            RichText::new(key)
                .size(10.5)
                .color(Color32::from_rgb(148, 163, 184)),
        )
        .wrap(),
    );
    ui.add(
        bevy_egui::egui::Label::new(
            RichText::new(value)
                .size(11.0)
                .color(Color32::from_rgb(226, 232, 240)),
        )
        .wrap(),
    );
    ui.end_row();
}

// ── live status card (migrated from Live tab) ─────────────────────────────

fn render_live_status_card(ui: &mut Ui, vm: &UiViewModel) {
    card_frame().show(ui, |ui| {
        section_header(
            ui,
            "◉",
            "Live Status",
            "Lifecycle, tracking, and face detection",
        );
        // row 1: lifecycle + tracking badges
        ui.horizontal_wrapped(|ui| {
            let lc = vm.lifecycle;
            let (lc_color, lc_label) = match lc {
                crate::ui_model::AppLifecycle::Idle => (Color32::from_rgb(100, 116, 139), "Idle"),
                crate::ui_model::AppLifecycle::Starting => {
                    (Color32::from_rgb(96, 165, 250), "Starting")
                }
                crate::ui_model::AppLifecycle::Running => {
                    (Color32::from_rgb(74, 222, 128), "Running")
                }
                crate::ui_model::AppLifecycle::Stopping => {
                    (Color32::from_rgb(251, 191, 36), "Stopping")
                }
                crate::ui_model::AppLifecycle::Failed => {
                    (Color32::from_rgb(248, 113, 113), "Failed")
                }
            };
            ui.label(
                RichText::new("Lifecycle")
                    .size(10.0)
                    .color(Color32::from_rgb(148, 163, 184)),
            );
            badge(ui, lc_label, lc_color);
            ui.add_space(10.0);
            let (tc, tlabel) = tracking_state_color(vm.tracking.state);
            ui.label(
                RichText::new("Tracking")
                    .size(10.0)
                    .color(Color32::from_rgb(148, 163, 184)),
            );
            badge(ui, tlabel, tc);
            ui.add_space(10.0);
            // Face detected badge
            let (fc, flabel) = if vm.tracking.face_detected {
                (Color32::from_rgb(74, 222, 128), "Face: yes")
            } else {
                (Color32::from_rgb(100, 116, 139), "Face: no")
            };
            badge(ui, flabel, fc);
        });
        ui.add_space(8.0);
        // Confidence bar
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Confidence")
                    .size(11.0)
                    .color(Color32::from_rgb(203, 213, 225)),
            );
            ui.add_space(8.0);
            let conf = vm.tracking.confidence.clamp(0.0, 1.0);
            let bar_color = if conf > 0.6 {
                Color32::from_rgb(74, 222, 128)
            } else if conf > 0.3 {
                Color32::from_rgb(251, 191, 36)
            } else {
                Color32::from_rgb(100, 116, 139)
            };
            ui.add(
                ProgressBar::new(conf)
                    .corner_radius(CornerRadius::same(6))
                    .desired_width(140.0)
                    .desired_height(8.0)
                    .fill(bar_color),
            );
            ui.label(
                RichText::new(format!("{:.0}%", conf * 100.0))
                    .size(11.0)
                    .color(Color32::from_rgb(226, 232, 240)),
            );
        });
        if vm.tracking.state == TrackingState::Lost {
            ui.add_space(6.0);
            Frame::new()
                .fill(Color32::from_rgba_unmultiplied(127, 29, 29, 60))
                .corner_radius(CornerRadius::same(7))
                .inner_margin(Margin::symmetric(8, 6))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(
                            "Face lost — attempting recovery. Check lighting and framing.",
                        )
                        .size(10.0)
                        .color(Color32::from_rgb(254, 202, 202)),
                    );
                });
        }
        // Calibration quick summary mirrors Setup card
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let cal = &vm.calibration;
            let (cc, clabel): (Color32, String) = if cal.is_complete {
                (Color32::from_rgb(74, 222, 128), "Calibrated".to_string())
            } else if cal.is_calibrating {
                (
                    Color32::from_rgb(96, 165, 250),
                    format!(
                        "Calibrating {}/{}",
                        cal.samples_collected, cal.samples_target
                    ),
                )
            } else {
                (
                    Color32::from_rgb(100, 116, 139),
                    "Not calibrated".to_string(),
                )
            };
            ui.label(
                RichText::new("Calibration")
                    .size(10.0)
                    .color(Color32::from_rgb(148, 163, 184)),
            );
            Frame::new()
                .fill(Color32::from_rgba_unmultiplied(cc.r(), cc.g(), cc.b(), 24))
                .corner_radius(CornerRadius::same(12))
                .inner_margin(Margin::symmetric(8, 3))
                .show(ui, |ui| {
                    ui.label(RichText::new(clabel).size(10.0).color(cc));
                });
        });
    });
}

// ── main render ────────────────────────────────────────────────────────────

/// Render the Diagnostics screen.
pub fn render_diagnostics_screen(ui: &mut Ui, vm: &UiViewModel, diagnostics: &DiagnosticsSnapshot) {
    ui.horizontal(|ui| {
        ui.heading(
            RichText::new("Diagnostics")
                .size(18.0)
                .strong()
                .color(Color32::from_rgb(248, 250, 252)),
        );
        ui.with_layout(
            bevy_egui::egui::Layout::right_to_left(bevy_egui::egui::Align::Center),
            |ui| {
                let fps = diagnostics.render_fps;
                let fps_color = if fps >= 55.0 {
                    Color32::from_rgb(74, 222, 128)
                } else if fps >= 30.0 {
                    Color32::from_rgb(251, 191, 36)
                } else {
                    Color32::from_rgb(248, 113, 113)
                };
                Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(
                        fps_color.r(),
                        fps_color.g(),
                        fps_color.b(),
                        28,
                    ))
                    .corner_radius(CornerRadius::same(12))
                    .inner_margin(Margin::symmetric(10, 4))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(format!("{fps:.0} FPS"))
                                .size(11.0)
                                .strong()
                                .color(fps_color),
                        );
                    });
            },
        );
    });
    ui.label(
        RichText::new("Live health  ·  Performance  ·  Model & Tracking  ·  Errors")
            .size(10.5)
            .color(Color32::from_rgb(148, 163, 184)),
    );
    ui.add_space(4.0);
    let rect = ui.available_rect_before_wrap();
    ui.painter().hline(
        rect.x_range(),
        rect.top(),
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(99, 102, 241, 40)),
    );
    ui.add_space(10.0);

    // Live status (migrated)
    render_live_status_card(ui, vm);

    // Performance
    card_frame().show(ui, |ui| {
        section_header(ui, "▦", "Performance", "Render, workers, and latency");
        bevy_egui::egui::Grid::new("perf_grid")
            .num_columns(2)
            .spacing([18.0, 5.0])
            .striped(false)
            .show(ui, |ui| {
                kv_row(ui, "Render FPS", &format!("{:.1}", diagnostics.render_fps));
                if let Some(cpu) = diagnostics.process_cpu_usage {
                    kv_row(ui, "Process CPU", &format!("{cpu:.1}%"));
                } else {
                    kv_row(ui, "Process CPU", "(none)");
                }
                if let Some(mem) = diagnostics.process_memory_gib {
                    kv_row(ui, "Process memory", &format!("{mem:.3} GiB"));
                } else {
                    kv_row(ui, "Process memory", "(none)");
                }
                kv_row(
                    ui,
                    "Capture rate",
                    &format!("{:.1} Hz", diagnostics.capture_rate),
                );
                kv_row(
                    ui,
                    "Inference rate",
                    &format!("{:.1} Hz", diagnostics.inference_rate),
                );
                kv_row(
                    ui,
                    "Detector rate",
                    &format!("{:.1} Hz", diagnostics.detector_rate),
                );
                kv_row(
                    ui,
                    "Landmark rate",
                    &format!("{:.1} Hz", diagnostics.landmark_rate),
                );
                kv_row(
                    ui,
                    "No-face frames",
                    &format!("{}", diagnostics.inference_no_face_frames),
                );
                kv_row(
                    ui,
                    "Tracking rate",
                    &format!("{:.1} Hz", diagnostics.tracking_rate),
                );
                kv_row(ui, "Capture worker", &diagnostics.capture_state);
                kv_row(ui, "Inference worker", &diagnostics.inference_state);
                kv_row(
                    ui,
                    "Slot overwrites",
                    &format!("{}", diagnostics.slot_overwrites),
                );
                kv_row(
                    ui,
                    "Avatar frames applied",
                    &format!("{}", diagnostics.avatar_frames_applied),
                );
                kv_row(
                    ui,
                    "Avatar frames skipped",
                    &format!("{}", diagnostics.avatar_frames_skipped),
                );
                kv_row(
                    ui,
                    "Capture→apply p50",
                    &diagnostics
                        .capture_to_apply_p50_ms
                        .map(|v| format!("{v:.2} ms"))
                        .unwrap_or_else(|| "(none)".to_string()),
                );
                kv_row(
                    ui,
                    "Capture→apply p95",
                    &diagnostics
                        .capture_to_apply_p95_ms
                        .map(|v| format!("{v:.2} ms"))
                        .unwrap_or_else(|| "(none)".to_string()),
                );
                kv_row(ui, "Metrics export", &diagnostics.metrics_export_status);
                kv_row(
                    ui,
                    "Export samples",
                    &format!("{} / 31", diagnostics.metrics_export_samples),
                );
            });
    });

    // Stage timings
    if !diagnostics.stage_timings.is_empty() {
        card_frame().show(ui, |ui| {
            section_header(ui, "◐", "Stage Timings", "Per-stage wall time");
            bevy_egui::egui::Grid::new("timing_grid")
                .num_columns(2)
                .spacing([18.0, 4.0])
                .striped(false)
                .show(ui, |ui| {
                    for (name, duration) in &diagnostics.stage_timings {
                        kv_row(ui, name, &format!("{duration:.2} ms"));
                    }
                });
        });
    }

    if !diagnostics.stage_percentiles.is_empty() {
        card_frame().show(ui, |ui| {
            section_header(ui, "◑", "Stage Percentiles", "p50 / p95 per stage");
            bevy_egui::egui::Grid::new("percentile_grid")
                .num_columns(3)
                .spacing([14.0, 4.0])
                .striped(false)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new("Stage")
                            .size(10.0)
                            .strong()
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    ui.label(
                        RichText::new("p50")
                            .size(10.0)
                            .strong()
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    ui.label(
                        RichText::new("p95")
                            .size(10.0)
                            .strong()
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    ui.end_row();
                    for (name, p50, p95) in &diagnostics.stage_percentiles {
                        ui.label(
                            RichText::new(name)
                                .size(10.5)
                                .color(Color32::from_rgb(203, 213, 225)),
                        );
                        ui.label(
                            RichText::new(format!("{p50:.2} ms"))
                                .size(10.5)
                                .color(Color32::from_rgb(226, 232, 240)),
                        );
                        ui.label(
                            RichText::new(format!("{p95:.2} ms"))
                                .size(10.5)
                                .color(Color32::from_rgb(226, 232, 240)),
                        );
                        ui.end_row();
                    }
                });
        });
    }

    // Model & Camera
    card_frame().show(ui, |ui| {
        section_header(ui, "⬢", "Model & Camera", "Asset identity and backend");
        bevy_egui::egui::Grid::new("info_grid")
            .num_columns(2)
            .spacing([18.0, 5.0])
            .striped(false)
            .show(ui, |ui| {
                kv_row(
                    ui,
                    "Model hash",
                    diagnostics.model_hash.as_deref().unwrap_or("(none)"),
                );
                kv_row(
                    ui,
                    "Pipeline",
                    diagnostics.pipeline_id.as_deref().unwrap_or("(none)"),
                );
                kv_row(
                    ui,
                    "ROI state",
                    diagnostics.roi_state.as_deref().unwrap_or("(none)"),
                );
                kv_row(
                    ui,
                    "Detector confidence",
                    &diagnostics
                        .detector_confidence
                        .map(|v| format!("{v:.3}"))
                        .unwrap_or_else(|| "(none)".to_string()),
                );
                kv_row(
                    ui,
                    "Camera backend",
                    diagnostics.camera_backend.as_deref().unwrap_or("(none)"),
                );
                kv_row(
                    ui,
                    "Tracking backend",
                    diagnostics.tracking_backend.as_deref().unwrap_or("(none)"),
                );
                kv_row(
                    ui,
                    "Tracking contract",
                    diagnostics.tracking_contract.as_deref().unwrap_or("(none)"),
                );
                kv_row(
                    ui,
                    "Avatar capabilities",
                    diagnostics
                        .avatar_capabilities
                        .as_deref()
                        .unwrap_or("(none)"),
                );
            });
    });

    // Tracking
    card_frame().show(ui, |ui| {
        section_header(
            ui,
            "◎",
            "Tracking",
            "Auto-neutral, backend authority, and residuals",
        );
        bevy_egui::egui::Grid::new("tracking_grid")
            .num_columns(2)
            .spacing([18.0, 5.0])
            .show(ui, |ui| {
                kv_row(ui, "State", &diagnostics.tracking_state);
                kv_row(
                    ui,
                    "Auto-neutral",
                    diagnostics
                        .auto_neutral_state
                        .as_deref()
                        .unwrap_or("(none)"),
                );
                if let Some(ready) = diagnostics.face_tracking_calibration_ready {
                    kv_row(ui, "Calibration ready", if ready { "yes" } else { "no" });
                }
                kv_row(
                    ui,
                    "Requested backend",
                    diagnostics
                        .face_tracking_requested
                        .as_deref()
                        .unwrap_or("(none)"),
                );
                kv_row(
                    ui,
                    "Output authority",
                    diagnostics
                        .face_tracking_authority
                        .as_deref()
                        .unwrap_or("(none)"),
                );
                if let Some(reason) = diagnostics.face_tracking_fallback_reason.as_deref() {
                    kv_row(ui, "Fallback reason", reason);
                }
                kv_row(
                    ui,
                    "Latest residual",
                    &diagnostics
                        .face_tracking_latest_residual
                        .map(|v| format!("{v:.4}"))
                        .unwrap_or_else(|| "(none)".to_string()),
                );
                kv_row(
                    ui,
                    "Added latency",
                    &diagnostics
                        .face_tracking_added_latency_ms
                        .map(|v| format!("{v:.1} ms"))
                        .unwrap_or_else(|| "(none)".to_string()),
                );
            });
        if let Some(reason) = diagnostics.face_tracking_fallback_reason.as_deref() {
            // highlight already in grid, also emphasize as warning bar
            ui.add_space(6.0);
            Frame::new()
                .fill(Color32::from_rgba_unmultiplied(120, 53, 15, 70))
                .corner_radius(CornerRadius::same(7))
                .inner_margin(Margin::symmetric(8, 6))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(format!("⚠ Fallback: {reason}"))
                            .size(10.0)
                            .color(Color32::from_rgb(253, 224, 71)),
                    );
                });
        }
    });

    // Last error
    if diagnostics.last_error_code.is_some()
        || diagnostics.inference_failure_stage.is_some()
        || diagnostics.last_error.is_some()
    {
        card_frame().show(ui, |ui| {
            section_header(ui, "⚠", "Errors", "Last failure and stage");
            if let Some(code) = &diagnostics.last_error_code {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Error code")
                            .size(10.5)
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    ui.label(
                        RichText::new(code)
                            .size(11.0)
                            .monospace()
                            .color(Color32::from_rgb(252, 165, 165)),
                    );
                });
            }
            if let Some(stage) = &diagnostics.inference_failure_stage {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Failure stage")
                            .size(10.5)
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    ui.label(
                        RichText::new(stage)
                            .size(11.0)
                            .color(Color32::from_rgb(226, 232, 240)),
                    );
                });
            }
            if let Some(error) = &diagnostics.last_error {
                ui.add_space(6.0);
                Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(127, 29, 29, 70))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(error)
                                .size(11.0)
                                .color(Color32::from_rgb(254, 202, 202)),
                        );
                    });
            }
        });
    }

    // Current State summary
    card_frame().show(ui, |ui| {
        section_header(ui, "≡", "Current State", "Mirrors Controls selection");
        bevy_egui::egui::Grid::new("current_state_grid")
            .num_columns(2)
            .spacing([18.0, 5.0])
            .show(ui, |ui| {
                kv_row(ui, "Screen", &format!("{:?}", vm.screen));
                kv_row(ui, "Lifecycle", &format!("{:?}", vm.lifecycle));
                kv_row(
                    ui,
                    "Camera",
                    vm.camera
                        .selected_index
                        .and_then(|i| vm.camera.available_cameras.get(i).map(|c| c.name.as_str()))
                        .unwrap_or("none"),
                );
                kv_row(
                    ui,
                    "Avatar",
                    vm.avatar
                        .imported_model
                        .as_ref()
                        .map(|m| m.name.as_str())
                        .unwrap_or("none"),
                );
                // extra: show confidence / face here too for quick glance
                kv_row(ui, "Confidence", &format!("{:.2}", vm.tracking.confidence));
                kv_row(
                    ui,
                    "Face",
                    if vm.tracking.face_detected {
                        "yes"
                    } else {
                        "no"
                    },
                );
            });
    });

    ui.add_space(10.0);
    ui.add(
        bevy_egui::egui::Label::new(
            RichText::new(
                "Tip: Return to Controls to adjust camera, calibration, preview, or NDI output.",
            )
            .size(10.0)
            .color(Color32::from_rgb(71, 85, 105))
            .italics(),
        )
        .wrap(),
    );
    ui.add_space(8.0);
}
