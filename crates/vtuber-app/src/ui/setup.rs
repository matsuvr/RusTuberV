//! Unified setup / control screen — the single operation hub that replaces
//! the former Setup + Live split.
//!
//! This screen consolidates all display-affecting controls into themed cards:
//! Session, Camera + View, Avatar, Calibration, Preview & Display, and NDI
//! Output.  Diagnostics keeps read-only health/status, so operators get a
//! clear split between "do things" (Setup/Controls) and "see health"
//! (Diagnostics).

use bevy_egui::egui::{
    Color32, CornerRadius, Frame, Margin, ProgressBar, Rect, RichText, Stroke, TextureId, Ui, vec2,
};

use crate::actions::UiAction;
use crate::error_presenter::ErrorPresentation;
use crate::preview::PreviewState;
use crate::preview_landmarks::PreviewLandmarkState;
use crate::ui_model::{AvatarLifecycleState, NdiOutputUiState, UiViewModel};
use vtuber_avatar::{ArmPoseProfileOverride, AvatarMotionMirror};
use vtuber_core::{FaceLandmark, MonoTimeNs, monotonic_now};

use super::error::render_error_panel;
use super::file_dialog::FileDialogState;

// ---------------------------------------------------------------------------
// Preview helpers — moved from the former `live.rs` so the preview stays
// adjacent to the controls that affect it.
// ---------------------------------------------------------------------------

fn preview_uv(mirrored: bool) -> Rect {
    if mirrored {
        Rect::from_min_max(
            bevy_egui::egui::pos2(1.0, 0.0),
            bevy_egui::egui::pos2(0.0, 1.0),
        )
    } else {
        Rect::from_min_max(
            bevy_egui::egui::pos2(0.0, 0.0),
            bevy_egui::egui::pos2(1.0, 1.0),
        )
    }
}

fn landmark_overlay_position(
    rect: Rect,
    landmark: &FaceLandmark,
    mirrored: bool,
) -> Option<bevy_egui::egui::Pos2> {
    if !landmark.x.is_finite()
        || !landmark.y.is_finite()
        || !(0.0..=1.0).contains(&landmark.x)
        || !(0.0..=1.0).contains(&landmark.y)
    {
        return None;
    }
    let x = if mirrored {
        1.0 - landmark.x
    } else {
        landmark.x
    };
    Some(bevy_egui::egui::pos2(
        rect.left() + x * rect.width(),
        rect.top() + landmark.y * rect.height(),
    ))
}

fn should_draw_landmark_overlay(
    preview_visible: bool,
    preview_texture: Option<TextureId>,
    landmarks: &PreviewLandmarkState,
    now: MonoTimeNs,
) -> bool {
    preview_visible && preview_texture.is_some() && landmarks.latest_fresh_at(now).is_some()
}

fn draw_landmark_overlay(
    ui: &Ui,
    rect: Rect,
    mirrored: bool,
    landmarks: &PreviewLandmarkState,
    now: MonoTimeNs,
) {
    let Some(snapshot) = landmarks.latest_fresh_at(now) else {
        return;
    };
    let painter = ui.painter().with_clip_rect(rect);
    for landmark in snapshot.landmarks.iter() {
        if let Some(position) = landmark_overlay_position(rect, landmark, mirrored) {
            painter.circle_filled(position, 1.5, Color32::from_rgb(255, 255, 0));
        }
    }
}

// ---------------------------------------------------------------------------
// NDI helpers — small view-model so the render fn stays pure over UiViewModel.
// ---------------------------------------------------------------------------

/// Official NDI site shown beside the output controls.
pub(crate) const NDI_OFFICIAL_URL: &str = "https://ndi.video";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NdiLiveSection {
    pub heading: &'static str,
    pub official_link: &'static str,
    pub source_name: String,
    pub status_text: String,
    pub start_enabled: bool,
    pub stop_enabled: bool,
    pub start_action: UiAction,
    pub stop_action: UiAction,
    pub unavailable_hint: Option<&'static str>,
    pub error_text: Option<String>,
}

pub(crate) fn ndi_live_section(vm: &UiViewModel) -> NdiLiveSection {
    let source_name = vm
        .ndi_output
        .source_name
        .clone()
        .unwrap_or_else(|| "RusTuberV".to_owned());
    let status_text = match vm.ndi_output.state {
        NdiOutputUiState::Off => "Off".to_string(),
        NdiOutputUiState::Starting => "Starting…".to_string(),
        NdiOutputUiState::Live => vm
            .ndi_output
            .connections
            .map(|count| format!("Live ({count} receiver(s))"))
            .unwrap_or_else(|| "Live".to_string()),
        NdiOutputUiState::Error => "Error".to_string(),
    };
    let error_text = match (
        vm.ndi_output.error_code.as_deref(),
        vm.ndi_output.error_message.as_deref(),
    ) {
        (Some(code), Some(message)) => Some(format!("{code}: {message}")),
        _ => None,
    };
    NdiLiveSection {
        heading: "NDI Output",
        official_link: NDI_OFFICIAL_URL,
        source_name,
        status_text,
        start_enabled: vm.can_start_ndi_output(),
        stop_enabled: vm.can_stop_ndi_output(),
        start_action: UiAction::StartNdiOutput,
        stop_action: UiAction::StopNdiOutput,
        unavailable_hint: (!vm.ndi_output.available)
            .then_some("NDI output is not included in this build."),
        error_text,
    }
}

// ---------------------------------------------------------------------------
// Visual primitives
// ---------------------------------------------------------------------------

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
            .fill(Color32::from_rgba_unmultiplied(99, 102, 241, 38))
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

fn status_dot(color: Color32) -> RichText {
    RichText::new("●").size(10.0).color(color)
}

fn lifecycle_color(state: AvatarLifecycleState) -> Color32 {
    match state {
        AvatarLifecycleState::Ready => Color32::from_rgb(74, 222, 128),
        AvatarLifecycleState::Loading | AvatarLifecycleState::Binding => {
            Color32::from_rgb(96, 165, 250)
        }
        AvatarLifecycleState::Failed => Color32::from_rgb(248, 113, 113),
        AvatarLifecycleState::Unloading => Color32::from_rgb(251, 191, 36),
        AvatarLifecycleState::None => Color32::from_rgb(100, 116, 139),
    }
}

fn app_lifecycle_color(lc: crate::ui_model::AppLifecycle) -> (Color32, &'static str) {
    use crate::ui_model::AppLifecycle as Lc;
    match lc {
        Lc::Idle => (Color32::from_rgb(100, 116, 139), "Idle"),
        Lc::Starting => (Color32::from_rgb(96, 165, 250), "Starting"),
        Lc::Running => (Color32::from_rgb(74, 222, 128), "Running"),
        Lc::Stopping => (Color32::from_rgb(251, 191, 36), "Stopping"),
        Lc::Failed => (Color32::from_rgb(248, 113, 113), "Failed"),
    }
}

// ---------------------------------------------------------------------------
// Public render entry
// ---------------------------------------------------------------------------

/// Render the unified Setup / Controls screen.
///
/// `preview`, `landmarks`, `avatar_motion_mirror`, and `preview_texture` are
/// the display-state resources that formerly lived only on the Live tab.
/// They now live alongside camera/avatar setup so the whole display pipeline
/// is controllable from one place.
#[allow(clippy::too_many_arguments)]
pub fn render_setup_screen(
    ui: &mut Ui,
    vm: &UiViewModel,
    ui_state: &mut super::UiState,
    preview: &PreviewState,
    landmarks: &PreviewLandmarkState,
    avatar_motion_mirror: AvatarMotionMirror,
    preview_texture: Option<TextureId>,
    face_backend: crate::face_backend::FaceTrackingBackendSelection,
    file_dialog: &mut FileDialogState,
    current_error: Option<&ErrorPresentation>,
) {
    // ── Screen title ────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.heading(
            RichText::new("Controls")
                .size(18.0)
                .strong()
                .color(Color32::from_rgb(248, 250, 252)),
        );
        ui.with_layout(
            bevy_egui::egui::Layout::right_to_left(bevy_egui::egui::Align::Center),
            |ui| {
                let (color, label) = app_lifecycle_color(vm.lifecycle);
                let badge = Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(
                        color.r(),
                        color.g(),
                        color.b(),
                        32,
                    ))
                    .corner_radius(CornerRadius::same(12))
                    .inner_margin(Margin::symmetric(10, 4))
                    .stroke(Stroke::new(
                        1.0,
                        Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 70),
                    ));
                badge.show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(status_dot(color));
                        ui.label(RichText::new(label).size(11.0).strong().color(color));
                    });
                });
            },
        );
    });
    ui.label(
        RichText::new("Camera  ·  Avatar  ·  Calibration  ·  Preview  ·  Output")
            .size(10.5)
            .color(Color32::from_rgb(148, 163, 184)),
    );
    ui.add_space(4.0);
    // thin accent line
    let rect = ui.available_rect_before_wrap();
    ui.painter().hline(
        rect.x_range(),
        rect.top(),
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(99, 102, 241, 40)),
    );
    ui.add_space(10.0);

    // ── Error banner (if any) ───────────────────────────────────────────
    if let Some(presentation) = current_error {
        let err_frame = Frame::new()
            .fill(Color32::from_rgba_unmultiplied(127, 29, 29, 90))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(12))
            .stroke(Stroke::new(
                1.0,
                Color32::from_rgba_unmultiplied(248, 113, 113, 90),
            ));
        err_frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("⚠")
                        .size(14.0)
                        .color(Color32::from_rgb(252, 165, 165)),
                );
                ui.label(
                    RichText::new("Current error")
                        .size(11.0)
                        .strong()
                        .color(Color32::from_rgb(254, 202, 202)),
                );
            });
            ui.add_space(4.0);
            render_error_panel(ui, presentation, ui_state);
        });
        ui.add_space(4.0);
    }

    // ── Session control ─────────────────────────────────────────────────
    card_frame().show(ui, |ui| {
        section_header(
            ui,
            "▶",
            "Session",
            "Start / stop the capture → tracking pipeline",
        );
        ui.horizontal(|ui| {
            // App lifecycle badge row
            let (ac, al) = app_lifecycle_color(vm.lifecycle);
            ui.label(
                RichText::new("App:")
                    .size(11.0)
                    .color(Color32::from_rgb(148, 163, 184)),
            );
            ui.label(status_dot(ac));
            ui.label(
                RichText::new(al)
                    .size(11.0)
                    .color(Color32::from_rgb(226, 232, 240)),
            );
            ui.add_space(12.0);
            let lc = lifecycle_color(vm.avatar.lifecycle);
            ui.label(
                RichText::new("Avatar:")
                    .size(11.0)
                    .color(Color32::from_rgb(148, 163, 184)),
            );
            ui.label(status_dot(lc));
            ui.label(
                RichText::new(format!("{:?}", vm.avatar.lifecycle))
                    .size(11.0)
                    .color(Color32::from_rgb(226, 232, 240)),
            );
        });
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            let start_enabled = vm.can_start();
            let stop_enabled = vm.can_stop();

            // Start — prominent green
            let start_btn =
                bevy_egui::egui::Button::new(RichText::new("●  Start").size(13.0).strong().color(
                    if start_enabled {
                        Color32::WHITE
                    } else {
                        Color32::from_rgb(100, 116, 139)
                    },
                ))
                .min_size(vec2(108.0, 36.0))
                .corner_radius(CornerRadius::same(9))
                .fill(if start_enabled {
                    Color32::from_rgb(22, 163, 74)
                } else {
                    Color32::from_rgba_unmultiplied(30, 41, 59, 200)
                })
                .stroke(Stroke::new(
                    1.0,
                    if start_enabled {
                        Color32::from_rgb(34, 197, 94)
                    } else {
                        Color32::from_rgba_unmultiplied(51, 65, 85, 180)
                    },
                ));
            if ui.add_enabled(start_enabled, start_btn).clicked() {
                ui_state.emit(UiAction::Start);
            }

            // Stop — prominent red
            let stop_btn =
                bevy_egui::egui::Button::new(RichText::new("■  Stop").size(13.0).strong().color(
                    if stop_enabled {
                        Color32::WHITE
                    } else {
                        Color32::from_rgb(100, 116, 139)
                    },
                ))
                .min_size(vec2(108.0, 36.0))
                .corner_radius(CornerRadius::same(9))
                .fill(if stop_enabled {
                    Color32::from_rgb(220, 38, 38)
                } else {
                    Color32::from_rgba_unmultiplied(30, 41, 59, 200)
                })
                .stroke(Stroke::new(
                    1.0,
                    if stop_enabled {
                        Color32::from_rgb(248, 113, 113)
                    } else {
                        Color32::from_rgba_unmultiplied(51, 65, 85, 180)
                    },
                ));
            if ui.add_enabled(stop_enabled, stop_btn).clicked() {
                ui_state.emit(UiAction::Stop);
            }

            if !start_enabled && !stop_enabled {
                ui.label(
                    RichText::new("Idle — configure camera & avatar to start")
                        .size(11.0)
                        .color(Color32::from_rgb(100, 116, 139))
                        .italics(),
                );
            } else if vm.lifecycle == crate::ui_model::AppLifecycle::Running {
                ui.label(
                    RichText::new("Running")
                        .size(11.0)
                        .color(Color32::from_rgb(74, 222, 128)),
                );
            }
        });
        if !vm.can_start() && vm.lifecycle == crate::ui_model::AppLifecycle::Idle {
            ui.add_space(6.0);
            let mut hints: Vec<String> = Vec::new();
            if vm.camera.selected_index.is_none() {
                hints.push("Select a camera".to_string());
            }
            if !vm.avatar.is_ready {
                hints.push("Import an avatar".to_string());
            }
            if !hints.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new("✦")
                            .size(11.0)
                            .color(Color32::from_rgb(251, 191, 36)),
                    );
                    ui.add(
                        bevy_egui::egui::Label::new(
                            RichText::new(hints.join("  ·  "))
                                .size(11.0)
                                .color(Color32::from_rgb(203, 213, 225)),
                        )
                        .wrap(),
                    );
                });
            }
        }
    });

    // ── Camera ──────────────────────────────────────────────────────────
    card_frame().show(ui, |ui| {
        section_header(
            ui,
            "◎",
            "Camera",
            "Device selection and viewport navigation",
        );

        // Device selection
        ui.label(
            RichText::new("Capture device")
                .size(11.0)
                .strong()
                .color(Color32::from_rgb(203, 213, 225)),
        );
        ui.add_space(4.0);
        if vm.camera.available_cameras.is_empty() {
            ui.horizontal(|ui| {
                Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(51, 65, 85, 120))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new("No cameras detected.")
                                .size(11.0)
                                .color(Color32::from_rgb(148, 163, 184)),
                        );
                    });
                if ui
                    .add(
                        bevy_egui::egui::Button::new(RichText::new("↻  Refresh").size(11.0))
                            .corner_radius(CornerRadius::same(8))
                            .min_size(vec2(92.0, 28.0)),
                    )
                    .clicked()
                {
                    ui_state.emit(UiAction::RefreshCameras);
                }
            });
        } else {
            ui.horizontal(|ui| {
                let selected = vm.camera.selected_index.unwrap_or(usize::MAX);
                let mut new_selected = selected;
                let combo_label = vm
                    .camera
                    .available_cameras
                    .get(selected)
                    .map(|c| c.name.as_str())
                    .unwrap_or("None");
                bevy_egui::egui::ComboBox::from_label("")
                    .selected_text(
                        RichText::new(combo_label)
                            .size(11.0)
                            .color(Color32::from_rgb(226, 232, 240)),
                    )
                    .width(190.0)
                    .show_ui(ui, |ui| {
                        for (i, cam) in vm.camera.available_cameras.iter().enumerate() {
                            ui.selectable_value(&mut new_selected, i, &cam.name);
                        }
                    });
                if new_selected != selected {
                    ui_state.emit(UiAction::SelectCamera {
                        index: new_selected,
                    });
                }
                if ui
                    .add(
                        bevy_egui::egui::Button::new(RichText::new("↻").size(12.0))
                            .corner_radius(CornerRadius::same(7))
                            .min_size(vec2(30.0, 24.0)),
                    )
                    .on_hover_text("Refresh cameras")
                    .clicked()
                {
                    ui_state.emit(UiAction::RefreshCameras);
                }
            });
            if let Some(idx) = vm.camera.selected_index
                && let Some(cam) = vm.camera.available_cameras.get(idx)
            {
                ui.label(
                    RichText::new(format!("Selected: {}", cam.name))
                        .size(10.0)
                        .color(Color32::from_rgb(100, 116, 139)),
                );
            }
        }

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);

        // View navigation
        ui.label(
            RichText::new("View")
                .size(11.0)
                .strong()
                .color(Color32::from_rgb(203, 213, 225)),
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let can_reset = vm.can_reset_camera();
            let btn = bevy_egui::egui::Button::new(RichText::new("⌖  Reset Camera").size(11.0))
                .corner_radius(CornerRadius::same(8))
                .min_size(vec2(132.0, 28.0));
            if ui.add_enabled(can_reset, btn).clicked() {
                ui_state.emit(UiAction::ResetAvatarCamera);
            }
            if !can_reset {
                ui.label(
                    RichText::new("— avatar not ready")
                        .size(10.0)
                        .color(Color32::from_rgb(100, 116, 139))
                        .italics(),
                );
            }
        });
        ui.label(
            RichText::new("Left drag: Orbit  ·  Right drag: Pan  ·  Wheel: Dolly")
                .size(10.0)
                .color(Color32::from_rgb(100, 116, 139))
                .italics(),
        );
    });

    // ── Avatar ──────────────────────────────────────────────────────────
    card_frame().show(ui, |ui| {
        section_header(ui, "◍", "Avatar", "VRM model, face tracking, and arm pose");

        if let Some(model) = &vm.avatar.imported_model {
            // Model summary chip
            Frame::new()
                .fill(Color32::from_rgba_unmultiplied(51, 65, 85, 90))
                .corner_radius(CornerRadius::same(10))
                .inner_margin(Margin::same(10))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("⬢")
                                .size(13.0)
                                .color(Color32::from_rgb(165, 180, 252)),
                        );
                        ui.vertical(|ui| {
                            ui.label(
                                RichText::new(&model.name)
                                    .size(12.0)
                                    .strong()
                                    .color(Color32::from_rgb(248, 250, 252)),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "{}  ·  {}  ·  {}",
                                    match model.generation {
                                        crate::import::VrmGeneration::Vrm0 => "VRM 0.x",
                                        crate::import::VrmGeneration::Vrm1 => "VRM 1.0",
                                    },
                                    &model.id[..8.min(model.id.len())],
                                    if model.has_required_bones {
                                        "bones ✓"
                                    } else {
                                        "bones ✗"
                                    }
                                ))
                                .size(10.0)
                                .color(Color32::from_rgb(148, 163, 184)),
                            );
                        });
                    });
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("Expressions: {}", model.expression_count))
                                .size(10.0)
                                .color(Color32::from_rgb(148, 163, 184)),
                        );
                        let lc = lifecycle_color(vm.avatar.lifecycle);
                        ui.with_layout(
                            bevy_egui::egui::Layout::right_to_left(bevy_egui::egui::Align::Center),
                            |ui| {
                                Frame::new()
                                    .fill(Color32::from_rgba_unmultiplied(
                                        lc.r(),
                                        lc.g(),
                                        lc.b(),
                                        30,
                                    ))
                                    .corner_radius(CornerRadius::same(12))
                                    .inner_margin(Margin::symmetric(8, 3))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(status_dot(lc));
                                            ui.label(
                                                RichText::new(format!("{:?}", vm.avatar.lifecycle))
                                                    .size(10.0)
                                                    .color(lc),
                                            );
                                        });
                                    });
                            },
                        );
                    });
                    if vm.avatar.lifecycle == AvatarLifecycleState::Failed
                        && ui
                            .add(
                                bevy_egui::egui::Button::new(
                                    RichText::new("↻  Retry Load").size(11.0),
                                )
                                .corner_radius(CornerRadius::same(8))
                                .min_size(vec2(110.0, 26.0)),
                            )
                            .clicked()
                    {
                        ui_state.emit(UiAction::RetryAfterError);
                    }
                });

            ui.add_space(8.0);

            // Settings — keep collapsings but padded
            render_face_tracking_backend_settings(ui, ui_state, face_backend);
            ui.add_space(4.0);
            render_arm_pose_settings(ui, vm, ui_state);

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .add(
                        bevy_egui::egui::Button::new(
                            RichText::new("Import VRM…")
                                .size(11.0)
                                .color(Color32::WHITE),
                        )
                        .corner_radius(CornerRadius::same(8))
                        .min_size(vec2(114.0, 30.0))
                        .fill(Color32::from_rgb(79, 70, 229)),
                    )
                    .clicked()
                    && !file_dialog.is_active()
                {
                    file_dialog.start();
                }
                if ui
                    .add(
                        bevy_egui::egui::Button::new(RichText::new("Unload").size(11.0))
                            .corner_radius(CornerRadius::same(8))
                            .min_size(vec2(78.0, 30.0)),
                    )
                    .clicked()
                {
                    ui_state.emit(UiAction::UnloadAvatar);
                }
            });
        } else {
            Frame::new()
                .fill(Color32::from_rgba_unmultiplied(51, 65, 85, 70))
                .corner_radius(CornerRadius::same(10))
                .inner_margin(Margin::same(14))
                .stroke(Stroke::new(
                    1.0,
                    Color32::from_rgba_unmultiplied(255, 255, 255, 8),
                ))
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(
                            RichText::new("◍")
                                .size(22.0)
                                .color(Color32::from_rgb(71, 85, 105)),
                        );
                        ui.label(
                            RichText::new("No avatar loaded.")
                                .size(12.0)
                                .color(Color32::from_rgb(148, 163, 184)),
                        );
                        ui.label(
                            RichText::new("Import a VRM to begin.")
                                .size(10.0)
                                .color(Color32::from_rgb(100, 116, 139)),
                        );
                        ui.add_space(8.0);
                        if ui
                            .add(
                                bevy_egui::egui::Button::new(
                                    RichText::new("＋  Import VRM…")
                                        .size(12.0)
                                        .strong()
                                        .color(Color32::WHITE),
                                )
                                .corner_radius(CornerRadius::same(9))
                                .min_size(vec2(150.0, 34.0))
                                .fill(Color32::from_rgb(79, 70, 229)),
                            )
                            .clicked()
                            && !file_dialog.is_active()
                        {
                            file_dialog.start();
                        }
                    });
                });

            // Still show backend selector even with no avatar? keep collapsed hidden — but show hint
            ui.add_space(6.0);
            ui.collapsing("Face Tracking", |ui| {
                ui.small("Selects the requested face tracking backend.");
                for selection in crate::face_backend::FaceTrackingBackendSelection::all() {
                    if ui
                        .radio(selection == face_backend, selection.label())
                        .clicked()
                    {
                        ui_state.emit(UiAction::SetFaceTrackingBackend(selection));
                    }
                }
            });
        }
    });

    // ── Calibration ─────────────────────────────────────────────────────
    card_frame().show(ui, |ui| {
        section_header(
            ui,
            "◎",
            "Calibration",
            "Neutral pose reference for tracking",
        );
        if vm.calibration.is_calibrating {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Calibrating…")
                        .size(11.0)
                        .strong()
                        .color(Color32::from_rgb(96, 165, 250)),
                );
                ui.with_layout(
                    bevy_egui::egui::Layout::right_to_left(bevy_egui::egui::Align::Center),
                    |ui| {
                        ui.label(
                            RichText::new(format!(
                                "{}/{}",
                                vm.calibration.samples_collected, vm.calibration.samples_target
                            ))
                            .size(11.0)
                            .color(Color32::from_rgb(203, 213, 225)),
                        );
                    },
                );
            });
            ui.add_space(4.0);
            let progress = if vm.calibration.samples_target > 0 {
                vm.calibration.samples_collected as f32 / vm.calibration.samples_target as f32
            } else {
                0.0
            };
            ui.add(
                ProgressBar::new(progress.clamp(0.0, 1.0))
                    .corner_radius(CornerRadius::same(6))
                    .desired_height(8.0),
            );
            if let Some(score) = vm.calibration.quality_score {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Quality")
                            .size(10.0)
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    ui.add(
                        ProgressBar::new(score.clamp(0.0, 1.0))
                            .corner_radius(CornerRadius::same(6))
                            .desired_width(90.0)
                            .desired_height(6.0),
                    );
                    ui.label(
                        RichText::new(format!("{:.0}%", score * 100.0))
                            .size(10.0)
                            .color(Color32::from_rgb(203, 213, 225)),
                    );
                });
            }
            if let Some(reason) = &vm.calibration.last_reject_reason {
                ui.add_space(4.0);
                Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(127, 29, 29, 70))
                    .corner_radius(CornerRadius::same(7))
                    .inner_margin(Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("✗")
                                    .size(11.0)
                                    .color(Color32::from_rgb(248, 113, 113)),
                            );
                            ui.label(
                                RichText::new(format!("Rejected: {reason}"))
                                    .size(10.0)
                                    .color(Color32::from_rgb(254, 202, 202)),
                            );
                        });
                    });
            }
            ui.add_space(8.0);
            if ui
                .add(
                    bevy_egui::egui::Button::new(RichText::new("Cancel").size(11.0))
                        .corner_radius(CornerRadius::same(8))
                        .min_size(vec2(86.0, 28.0)),
                )
                .clicked()
            {
                ui_state.emit(UiAction::CancelCalibration);
            }
        } else if vm.calibration.is_complete {
            Frame::new()
                .fill(Color32::from_rgba_unmultiplied(20, 83, 45, 70))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::symmetric(10, 8))
                .stroke(Stroke::new(
                    1.0,
                    Color32::from_rgba_unmultiplied(74, 222, 128, 60),
                ))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("✓")
                                .size(13.0)
                                .color(Color32::from_rgb(74, 222, 128)),
                        );
                        ui.label(
                            RichText::new("Calibration complete.")
                                .size(11.0)
                                .strong()
                                .color(Color32::from_rgb(187, 247, 208)),
                        );
                        if let Some(score) = vm.calibration.quality_score {
                            ui.with_layout(
                                bevy_egui::egui::Layout::right_to_left(
                                    bevy_egui::egui::Align::Center,
                                ),
                                |ui| {
                                    ui.label(
                                        RichText::new(format!("{:.0}%", score * 100.0))
                                            .size(11.0)
                                            .color(Color32::from_rgb(134, 239, 172)),
                                    );
                                },
                            );
                        }
                    });
                });
            ui.add_space(8.0);
            if ui
                .add(
                    bevy_egui::egui::Button::new(RichText::new("↻  Retry").size(11.0))
                        .corner_radius(CornerRadius::same(8))
                        .min_size(vec2(86.0, 28.0)),
                )
                .clicked()
            {
                ui_state.emit(UiAction::RetryCalibration);
            }
        } else if vm.can_calibrate() {
            ui.label(
                RichText::new("Ready to capture your neutral pose.")
                    .size(11.0)
                    .color(Color32::from_rgb(203, 213, 225)),
            );
            ui.label(
                RichText::new("Face the camera with a relaxed expression, then begin.")
                    .size(10.0)
                    .color(Color32::from_rgb(100, 116, 139)),
            );
            ui.add_space(8.0);
            if ui
                .add(
                    bevy_egui::egui::Button::new(
                        RichText::new("◎  Begin Calibration")
                            .size(11.0)
                            .strong()
                            .color(Color32::WHITE),
                    )
                    .corner_radius(CornerRadius::same(9))
                    .min_size(vec2(168.0, 32.0))
                    .fill(Color32::from_rgb(79, 70, 229)),
                )
                .clicked()
            {
                ui_state.emit(UiAction::BeginCalibration);
            }
        } else {
            ui.horizontal(|ui| {
                ui.add_enabled(
                    false,
                    bevy_egui::egui::Button::new(RichText::new("◎  Begin Calibration").size(11.0))
                        .corner_radius(CornerRadius::same(9))
                        .min_size(vec2(168.0, 32.0)),
                );
                ui.label(
                    RichText::new("—")
                        .size(11.0)
                        .color(Color32::from_rgb(71, 85, 105)),
                );
                ui.label(
                    RichText::new("Start tracking to calibrate.")
                        .size(10.0)
                        .color(Color32::from_rgb(100, 116, 139))
                        .italics(),
                );
            });
            if vm.lifecycle != crate::ui_model::AppLifecycle::Running {
                ui.add_space(4.0);
                Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(51, 65, 85, 70))
                    .corner_radius(CornerRadius::same(7))
                    .inner_margin(Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new("Pipeline is not running. Press Start above first.")
                                .size(10.0)
                                .color(Color32::from_rgb(148, 163, 184)),
                        );
                    });
            }
        }
    });

    // ── Preview & Display ───────────────────────────────────────────────
    card_frame().show(ui, |ui| {
        section_header(
            ui,
            "▣",
            "Preview & Display",
            "Camera feed, mirroring, and landmarks",
        );

        // Toggles row — styled as pill checkboxes
        ui.horizontal_wrapped(|ui| {
            let mut preview_visible = preview.visible;
            let toggle_preview = ui.checkbox(
                &mut preview_visible,
                RichText::new("Show Preview").size(11.0),
            );
            if toggle_preview.changed() {
                ui_state.emit(UiAction::TogglePreview);
            }
            ui.add_space(10.0);
            let mut mirror = preview.mirrored;
            let toggle_mirror =
                ui.checkbox(&mut mirror, RichText::new("Mirror Preview").size(11.0));
            if toggle_mirror.changed() {
                ui_state.emit(UiAction::ToggleMirror);
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let mut mirror_avatar_motion = avatar_motion_mirror.is_enabled();
            if ui
                .checkbox(
                    &mut mirror_avatar_motion,
                    RichText::new("Mirror Avatar Motion").size(11.0),
                )
                .changed()
            {
                ui_state.emit(UiAction::ToggleAvatarMotionMirror);
            }
            ui.label(
                RichText::new("(operator view)")
                    .size(10.0)
                    .color(Color32::from_rgb(100, 116, 139))
                    .italics(),
            );
        });

        ui.add_space(10.0);

        if preview.visible {
            if let Some(texture) = preview_texture {
                // Frame owns its own inner width — compute image size inside so
                // it never exceeds the frame's inner margin and causes a right
                // edge clip inside the 340px drawer.
                let frame = Frame::new()
                    .fill(Color32::from_rgb(2, 6, 23))
                    .corner_radius(CornerRadius::same(10))
                    .stroke(Stroke::new(
                        1.0,
                        Color32::from_rgba_unmultiplied(255, 255, 255, 10),
                    ))
                    .inner_margin(Margin::same(4));
                let inner = frame.show(ui, |ui| {
                    let w = ui.available_width().max(120.0);
                    let size = vec2(w, w * 9.0 / 16.0);
                    let response = ui.add(
                        bevy_egui::egui::Image::from_texture((texture, size))
                            .uv(preview_uv(preview.mirrored))
                            .corner_radius(CornerRadius::same(7)),
                    );
                    response.rect
                });
                // overlay landmarks inside the same clip
                let now = monotonic_now();
                if should_draw_landmark_overlay(preview.visible, Some(texture), landmarks, now) {
                    draw_landmark_overlay(ui, inner.inner, preview.mirrored, landmarks, now);
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("● LIVE PREVIEW")
                            .size(9.0)
                            .color(Color32::from_rgb(74, 222, 128))
                            .strong(),
                    );
                    if preview.mirrored {
                        ui.label(
                            RichText::new("· mirrored")
                                .size(9.0)
                                .color(Color32::from_rgb(100, 116, 139)),
                        );
                    }
                    ui.with_layout(
                        bevy_egui::egui::Layout::right_to_left(bevy_egui::egui::Align::Center),
                        |ui| {
                            ui.label(
                                RichText::new(format!("{} fps target", preview.target_fps))
                                    .size(9.0)
                                    .color(Color32::from_rgb(71, 85, 105)),
                            );
                        },
                    );
                });
            } else {
                Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(15, 23, 42, 180))
                    .corner_radius(CornerRadius::same(10))
                    .inner_margin(Margin::same(12))
                    .stroke(Stroke::new(
                        1.0,
                        Color32::from_rgba_unmultiplied(255, 255, 255, 8),
                    ))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.vertical_centered(|ui| {
                            ui.spinner();
                            ui.add_space(6.0);
                            ui.add(
                                bevy_egui::egui::Label::new(
                                    RichText::new("Waiting for camera frames…")
                                        .size(11.0)
                                        .color(Color32::from_rgb(148, 163, 184)),
                                )
                                .wrap(),
                            );
                            ui.add(
                                bevy_egui::egui::Label::new(
                                    RichText::new("Start the session and allow camera access.")
                                        .size(10.0)
                                        .color(Color32::from_rgb(100, 116, 139)),
                                )
                                .wrap(),
                            );
                        });
                    });
            }
        } else {
            Frame::new()
                .fill(Color32::from_rgba_unmultiplied(15, 23, 42, 120))
                .corner_radius(CornerRadius::same(10))
                .inner_margin(Margin::same(12))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(
                        bevy_egui::egui::Label::new(
                            RichText::new("⬚  Preview hidden — tracking still runs in background.")
                                .size(10.5)
                                .color(Color32::from_rgb(100, 116, 139)),
                        )
                        .wrap(),
                    );
                });
        }
    });

    // ── NDI Output ──────────────────────────────────────────────────────
    {
        let ndi = ndi_live_section(vm);
        card_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                section_header(ui, "⬡", ndi.heading, "Transparent avatar output");
                ui.with_layout(
                    bevy_egui::egui::Layout::right_to_left(bevy_egui::egui::Align::Center),
                    |ui| {
                        ui.hyperlink_to(
                            RichText::new("NDI® info ↗")
                                .size(10.0)
                                .color(Color32::from_rgb(147, 197, 253)),
                            ndi.official_link,
                        );
                    },
                );
            });

            // Source + status row
            Frame::new()
                .fill(Color32::from_rgba_unmultiplied(15, 23, 42, 140))
                .corner_radius(CornerRadius::same(9))
                .inner_margin(Margin::symmetric(10, 8))
                .show(ui, |ui| {
                    bevy_egui::egui::Grid::new("ndi_meta_grid")
                        .num_columns(2)
                        .spacing([14.0, 6.0])
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new("Source")
                                    .size(10.0)
                                    .color(Color32::from_rgb(148, 163, 184)),
                            );
                            ui.label(
                                RichText::new(&ndi.source_name)
                                    .size(11.0)
                                    .strong()
                                    .color(Color32::from_rgb(226, 232, 240))
                                    .monospace(),
                            );
                            ui.end_row();
                            ui.label(
                                RichText::new("Status")
                                    .size(10.0)
                                    .color(Color32::from_rgb(148, 163, 184)),
                            );
                            let (sc, stxt) = match vm.ndi_output.state {
                                NdiOutputUiState::Off => {
                                    (Color32::from_rgb(100, 116, 139), ndi.status_text.clone())
                                }
                                NdiOutputUiState::Starting => {
                                    (Color32::from_rgb(96, 165, 250), ndi.status_text.clone())
                                }
                                NdiOutputUiState::Live => {
                                    (Color32::from_rgb(74, 222, 128), ndi.status_text.clone())
                                }
                                NdiOutputUiState::Error => {
                                    (Color32::from_rgb(248, 113, 113), ndi.status_text.clone())
                                }
                            };
                            ui.horizontal(|ui| {
                                ui.label(status_dot(sc));
                                ui.label(RichText::new(stxt).size(11.0).color(sc).strong());
                            });
                            ui.end_row();
                        });
                });
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                let start_btn = bevy_egui::egui::Button::new(
                    RichText::new("▶  Start")
                        .size(11.0)
                        .strong()
                        .color(if ndi.start_enabled {
                            Color32::WHITE
                        } else {
                            Color32::from_rgb(100, 116, 139)
                        }),
                )
                .corner_radius(CornerRadius::same(8))
                .min_size(vec2(86.0, 30.0))
                .fill(if ndi.start_enabled {
                    Color32::from_rgb(79, 70, 229)
                } else {
                    Color32::from_rgba_unmultiplied(30, 41, 59, 200)
                });
                if ui.add_enabled(ndi.start_enabled, start_btn).clicked() {
                    ui_state.emit(ndi.start_action.clone());
                }
                let stop_btn = bevy_egui::egui::Button::new(
                    RichText::new("■  Stop")
                        .size(11.0)
                        .strong()
                        .color(if ndi.stop_enabled {
                            Color32::WHITE
                        } else {
                            Color32::from_rgb(100, 116, 139)
                        }),
                )
                .corner_radius(CornerRadius::same(8))
                .min_size(vec2(86.0, 30.0))
                .fill(if ndi.stop_enabled {
                    Color32::from_rgb(220, 38, 38)
                } else {
                    Color32::from_rgba_unmultiplied(30, 41, 59, 200)
                });
                if ui.add_enabled(ndi.stop_enabled, stop_btn).clicked() {
                    ui_state.emit(ndi.stop_action.clone());
                }
            });

            if let Some(hint) = ndi.unavailable_hint {
                ui.add_space(6.0);
                ui.add(
                    bevy_egui::egui::Label::new(
                        RichText::new(hint)
                            .size(10.0)
                            .color(Color32::from_rgb(251, 191, 36))
                            .italics(),
                    )
                    .wrap(),
                );
            }
            if let Some(error) = &ndi.error_text {
                ui.add_space(6.0);
                Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(127, 29, 29, 70))
                    .corner_radius(CornerRadius::same(7))
                    .inner_margin(Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.add(
                            bevy_egui::egui::Label::new(
                                RichText::new(error)
                                    .size(10.0)
                                    .color(Color32::from_rgb(254, 202, 202)),
                            )
                            .wrap(),
                        );
                    });
            }
            if vm.ndi_output.dropped_frames > 0 || vm.ndi_output.replaced_frames > 0 {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!(
                        "Mailbox: {} dropped, {} replaced",
                        vm.ndi_output.dropped_frames, vm.ndi_output.replaced_frames
                    ))
                    .size(10.0)
                    .color(Color32::from_rgb(100, 116, 139)),
                );
            }
        });
    }

    ui.add_space(14.0);
    ui.separator();
    ui.add_space(4.0);
    ui.add(
        bevy_egui::egui::Label::new(
            RichText::new(
                "Tip: Use Diagnostics tab to monitor tracking health, latency, and performance.",
            )
            .size(10.0)
            .color(Color32::from_rgb(71, 85, 105))
            .italics(),
        )
        .wrap(),
    );
    ui.add_space(8.0);
}

fn render_face_tracking_backend_settings(
    ui: &mut Ui,
    ui_state: &mut super::UiState,
    current: crate::face_backend::FaceTrackingBackendSelection,
) {
    ui.collapsing(
        RichText::new("Face Tracking")
            .size(11.0)
            .strong()
            .color(Color32::from_rgb(203, 213, 225)),
        |ui| {
            ui.label(
                RichText::new("Selects the requested face tracking backend.")
                    .size(10.0)
                    .color(Color32::from_rgb(100, 116, 139)),
            );
            ui.add_space(4.0);
            for selection in crate::face_backend::FaceTrackingBackendSelection::all() {
                if ui
                    .radio(
                        selection == current,
                        RichText::new(selection.label()).size(11.0),
                    )
                    .clicked()
                {
                    ui_state.emit(UiAction::SetFaceTrackingBackend(selection));
                }
            }
            if current != crate::face_backend::FaceTrackingBackendSelection::DirectMediaPipe {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Falls back to Direct automatically while the backend is unavailable.",
                    )
                    .size(10.0)
                    .color(Color32::from_rgb(100, 116, 139))
                    .italics(),
                );
            }
        },
    );
}

fn render_arm_pose_settings(ui: &mut Ui, vm: &UiViewModel, ui_state: &mut super::UiState) {
    ui.collapsing(
        RichText::new("Default arm pose")
            .size(11.0)
            .strong()
            .color(Color32::from_rgb(203, 213, 225)),
        |ui| {
            ui.label(
                RichText::new("Saved per model by its content hash.")
                    .size(10.0)
                    .color(Color32::from_rgb(100, 116, 139)),
            );
            ui.add_space(4.0);
            let mut profile = vm.arm_pose.profile;
            let mut arm_drop_degrees = profile.arm_drop_radians.to_degrees();
            let mut finger_curl_degrees = profile.finger_curl_radians.to_degrees();
            let mut changed = false;
            changed |= ui
                .add(
                    bevy_egui::egui::Slider::new(&mut arm_drop_degrees, 0.0..=90.0)
                        .text(RichText::new("Arm drop (deg)").size(10.0)),
                )
                .changed();
            changed |= ui
                .add(
                    bevy_egui::egui::Slider::new(&mut profile.reach_ratio, 0.01..=1.0)
                        .text(RichText::new("Reach ratio").size(10.0)),
                )
                .changed();
            changed |= ui
                .add(
                    bevy_egui::egui::Slider::new(
                        &mut profile.forward_hand_offset_ratio,
                        -1.0..=1.0,
                    )
                    .text(RichText::new("Forward offset").size(10.0)),
                )
                .changed();
            changed |= ui
                .add(
                    bevy_egui::egui::Slider::new(&mut profile.elbow_pole_offset_ratio, 0.0..=1.0)
                        .text(RichText::new("Elbow pole").size(10.0)),
                )
                .changed();
            changed |= ui
                .add(
                    bevy_egui::egui::Slider::new(&mut profile.shoulder_follow_weight, 0.0..=1.0)
                        .text(RichText::new("Shoulder follow").size(10.0)),
                )
                .changed();
            changed |= ui
                .add(
                    bevy_egui::egui::Slider::new(&mut finger_curl_degrees, 0.0..=90.0)
                        .text(RichText::new("Finger curl (deg)").size(10.0)),
                )
                .changed();

            profile.arm_drop_radians = arm_drop_degrees.to_radians();
            profile.finger_curl_radians = finger_curl_degrees.to_radians();
            if changed {
                ui_state.emit(UiAction::SetArmPoseProfile {
                    profile: ArmPoseProfileOverride::from_profile(profile),
                });
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if vm.arm_pose.has_override {
                    if ui
                        .add(
                            bevy_egui::egui::Button::new(
                                RichText::new("↺  Reset to automatic").size(11.0),
                            )
                            .corner_radius(CornerRadius::same(8))
                            .min_size(vec2(148.0, 28.0)),
                        )
                        .clicked()
                    {
                        ui_state.emit(UiAction::ResetArmPoseProfile);
                    }
                } else {
                    ui.label(
                        RichText::new("Using automatic geometry-derived defaults.")
                            .size(10.0)
                            .color(Color32::from_rgb(100, 116, 139))
                            .italics(),
                    );
                }
            });
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_preview_reverses_only_the_horizontal_uv_axis() {
        let normal = preview_uv(false);
        let mirrored = preview_uv(true);
        assert_eq!(normal.min, bevy_egui::egui::pos2(0.0, 0.0));
        assert_eq!(normal.max, bevy_egui::egui::pos2(1.0, 1.0));
        assert_eq!(mirrored.min, bevy_egui::egui::pos2(1.0, 0.0));
        assert_eq!(mirrored.max, bevy_egui::egui::pos2(0.0, 1.0));
    }

    #[test]
    fn landmark_overlay_maps_corners_and_center_to_preview_rect() {
        let rect = Rect::from_min_max(
            bevy_egui::egui::pos2(10.0, 20.0),
            bevy_egui::egui::pos2(110.0, 70.0),
        );
        assert_eq!(
            landmark_overlay_position(
                rect,
                &FaceLandmark {
                    x: 0.0,
                    y: 0.0,
                    ..FaceLandmark::default()
                },
                false,
            ),
            Some(bevy_egui::egui::pos2(10.0, 20.0))
        );
        assert_eq!(
            landmark_overlay_position(
                rect,
                &FaceLandmark {
                    x: 0.5,
                    y: 0.5,
                    ..FaceLandmark::default()
                },
                false,
            ),
            Some(bevy_egui::egui::pos2(60.0, 45.0))
        );
        assert_eq!(
            landmark_overlay_position(
                rect,
                &FaceLandmark {
                    x: 1.0,
                    y: 1.0,
                    ..FaceLandmark::default()
                },
                false,
            ),
            Some(bevy_egui::egui::pos2(110.0, 70.0))
        );
    }

    #[test]
    fn mirror_changes_x_only_for_landmark_overlay() {
        let rect = Rect::from_min_max(
            bevy_egui::egui::pos2(10.0, 20.0),
            bevy_egui::egui::pos2(110.0, 70.0),
        );
        let landmark = FaceLandmark {
            x: 0.25,
            y: 0.2,
            ..FaceLandmark::default()
        };
        assert_eq!(
            landmark_overlay_position(rect, &landmark, false),
            Some(bevy_egui::egui::pos2(35.0, 30.0))
        );
        assert_eq!(
            landmark_overlay_position(rect, &landmark, true),
            Some(bevy_egui::egui::pos2(85.0, 30.0))
        );
    }

    #[test]
    fn invalid_landmark_coordinates_are_skipped() {
        let rect = Rect::from_min_max(
            bevy_egui::egui::pos2(0.0, 0.0),
            bevy_egui::egui::pos2(100.0, 100.0),
        );
        for landmark in [
            FaceLandmark {
                x: f32::NAN,
                y: 0.5,
                ..FaceLandmark::default()
            },
            FaceLandmark {
                x: 0.5,
                y: f32::INFINITY,
                ..FaceLandmark::default()
            },
            FaceLandmark {
                x: -0.1,
                y: 0.5,
                ..FaceLandmark::default()
            },
            FaceLandmark {
                x: 0.5,
                y: 1.1,
                ..FaceLandmark::default()
            },
        ] {
            assert!(landmark_overlay_position(rect, &landmark, false).is_none());
        }
    }

    #[test]
    fn overlay_requires_visible_registered_preview_and_fresh_snapshot() {
        let state = PreviewLandmarkState::default();
        let texture = TextureId::User(1);
        let now = MonoTimeNs(1_000);
        assert!(!should_draw_landmark_overlay(
            false,
            Some(texture),
            &state,
            now
        ));
        assert!(!should_draw_landmark_overlay(true, None, &state, now));
        assert!(!should_draw_landmark_overlay(
            true,
            Some(texture),
            &state,
            now
        ));
    }

    #[test]
    fn valid_snapshot_contains_478_draw_candidates_without_repacking_landmarks() {
        let state = PreviewLandmarkState {
            latest: Some(crate::preview_landmarks::PreviewLandmarkSnapshot {
                source_seq: vtuber_core::FrameSeq(1),
                captured_at: MonoTimeNs(1),
                published_at: MonoTimeNs(1),
                landmarks: (0..vtuber_core::MEDIAPIPE_FACE_LANDMARK_COUNT)
                    .map(|index| FaceLandmark {
                        x: index as f32 / vtuber_core::MEDIAPIPE_FACE_LANDMARK_COUNT as f32,
                        y: 0.5,
                        ..FaceLandmark::default()
                    })
                    .collect::<Vec<_>>()
                    .into(),
            }),
        };
        let rect = Rect::from_min_max(
            bevy_egui::egui::pos2(0.0, 0.0),
            bevy_egui::egui::pos2(100.0, 100.0),
        );
        let snapshot = state
            .latest_fresh_at(MonoTimeNs(2))
            .expect("snapshot is fresh");
        let candidates = snapshot
            .landmarks
            .iter()
            .filter_map(|landmark| landmark_overlay_position(rect, landmark, false))
            .count();
        assert_eq!(candidates, vtuber_core::MEDIAPIPE_FACE_LANDMARK_COUNT);
    }

    fn ready_ndi_view_model() -> UiViewModel {
        let mut vm = UiViewModel::default();
        vm.avatar.is_ready = true;
        vm.avatar.lifecycle = crate::ui_model::AvatarLifecycleState::Ready;
        vm.ndi_output.available = true;
        vm.ndi_output.source_name = Some("RusTuberV".into());
        vm
    }

    #[test]
    fn live_ndi_section_exists_with_official_link_and_does_not_hold_a_sender() {
        let section = ndi_live_section(&ready_ndi_view_model());
        assert_eq!(section.heading, "NDI Output");
        assert_eq!(section.official_link, NDI_OFFICIAL_URL);
        assert!(section.official_link.starts_with("https://ndi.video"));
        assert_eq!(section.start_action, UiAction::StartNdiOutput);
        assert_eq!(section.stop_action, UiAction::StopNdiOutput);
        assert!(section.start_enabled);
        assert!(!section.stop_enabled);
        assert_eq!(section.status_text, "Off");
    }

    #[test]
    fn off_start_and_live_stop_each_emit_exactly_one_action() {
        let off = ndi_live_section(&ready_ndi_view_model());
        assert!(off.start_enabled);
        assert!(!off.stop_enabled);
        let mut ui_state = crate::ui::UiState::default();
        ui_state.emit(off.start_action.clone());
        ui_state.emit(off.start_action.clone());
        assert_eq!(ui_state.take_actions(), vec![UiAction::StartNdiOutput]);

        let mut live = ready_ndi_view_model();
        live.ndi_output.state = NdiOutputUiState::Live;
        live.ndi_output.connections = Some(1);
        let live_section = ndi_live_section(&live);
        assert!(!live_section.start_enabled);
        assert!(live_section.stop_enabled);
        assert_eq!(live_section.status_text, "Live (1 receiver(s))");
        ui_state.emit(live_section.stop_action.clone());
        ui_state.emit(live_section.stop_action.clone());
        assert_eq!(ui_state.take_actions(), vec![UiAction::StopNdiOutput]);
    }

    #[test]
    fn starting_state_does_not_emit_a_duplicate_start() {
        let mut vm = ready_ndi_view_model();
        vm.ndi_output.state = NdiOutputUiState::Starting;
        let section = ndi_live_section(&vm);
        assert!(!section.start_enabled);
        assert!(section.stop_enabled);
        assert_eq!(section.status_text, "Starting…");
    }

    #[test]
    fn unavailable_and_error_states_are_visible_and_start_is_disabled() {
        let mut vm = ready_ndi_view_model();
        vm.ndi_output.available = false;
        let unavailable = ndi_live_section(&vm);
        assert!(!unavailable.start_enabled);
        assert_eq!(
            unavailable.unavailable_hint,
            Some("NDI output is not included in this build.")
        );

        vm.ndi_output.available = true;
        vm.ndi_output.state = NdiOutputUiState::Error;
        vm.ndi_output.error_code = Some("NDI_RUNTIME_NOT_FOUND".into());
        vm.ndi_output.error_message = Some("NDI runtime could not be loaded.".into());
        let error = ndi_live_section(&vm);
        assert!(
            error.start_enabled,
            "Start is the Retry path after an NDI error"
        );
        assert_eq!(error.status_text, "Error");
        assert_eq!(
            error.error_text.as_deref(),
            Some("NDI_RUNTIME_NOT_FOUND: NDI runtime could not be loaded.")
        );
    }
}
