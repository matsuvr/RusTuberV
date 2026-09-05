//! Settings panes for the detail area.
//!
//! Follows the macOS System Settings model: each category renders as a
//! grouped list (label left, control right, hairline separators) and each
//! piece of information lives in exactly one place. Session Start/Stop is
//! owned by the sidebar; Diagnostics owns all health status.

use bevy_egui::egui::{
    self, Color32, CornerRadius, ProgressBar, Rect, RichText, TextureId, Ui, pos2, vec2,
};

use crate::actions::UiAction;
use crate::preview::PreviewState;
use crate::preview_landmarks::PreviewLandmarkState;
use crate::ui_model::{AvatarLifecycleState, NdiOutputUiState, UiViewModel};
use vtuber_avatar::{ArmPoseProfileOverride, AvatarMotionMirror};
use vtuber_core::{FaceLandmark, MonoTimeNs, monotonic_now};

use super::file_dialog::FileDialogState;
use super::widgets::{
    ACCENT, ALERT_RED, INFO_BLUE, LABEL, OK_GREEN, SECONDARY, WARNING_AMBER, caption,
    destructive_button, filled_button, group, info_row, plain_button, row_separator,
    section_caption, settings_row, status_text,
};

// ---------------------------------------------------------------------------
// Preview helpers — camera feed texture and landmark overlay
// ---------------------------------------------------------------------------

fn preview_uv(mirrored: bool) -> Rect {
    if mirrored {
        Rect::from_min_max(pos2(1.0, 0.0), pos2(0.0, 1.0))
    } else {
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0))
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
    Some(pos2(
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
    let status = match vm.ndi_output.state {
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
        official_link: NDI_OFFICIAL_URL,
        source_name,
        status_text: status,
        start_enabled: vm.can_start_ndi_output(),
        stop_enabled: vm.can_stop_ndi_output(),
        start_action: UiAction::StartNdiOutput,
        stop_action: UiAction::StopNdiOutput,
        unavailable_hint: (!vm.ndi_output.available)
            .then_some("NDI output is not included in this build."),
        error_text,
    }
}

/// Status color for the NDI sender state.
fn ndi_status_color(vm: &UiViewModel) -> Color32 {
    match vm.ndi_output.state {
        NdiOutputUiState::Off => SECONDARY,
        NdiOutputUiState::Starting => INFO_BLUE,
        NdiOutputUiState::Live => OK_GREEN,
        NdiOutputUiState::Error => ALERT_RED,
    }
}

// ---------------------------------------------------------------------------
// Camera pane
// ---------------------------------------------------------------------------

/// Camera capture device and viewport navigation.
pub fn render_camera_pane(ui: &mut Ui, vm: &UiViewModel, ui_state: &mut super::UiState) {
    section_caption(ui, "Capture device");
    group(ui, |ui| {
        if vm.camera.available_cameras.is_empty() {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("No cameras detected.")
                        .size(13.0)
                        .color(SECONDARY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if plain_button(ui, "Refresh", true).clicked() {
                        ui_state.emit(UiAction::RefreshCameras);
                    }
                });
            });
        } else {
            settings_row(ui, "Camera", |ui| {
                let selected = vm.camera.selected_index.unwrap_or(usize::MAX);
                let mut new_selected = selected;
                let combo_label = vm
                    .camera
                    .available_cameras
                    .get(selected)
                    .map(|c| c.name.as_str())
                    .unwrap_or("None");
                egui::ComboBox::from_id_salt("camera_select")
                    .selected_text(RichText::new(combo_label).size(13.0).color(LABEL))
                    .width(170.0)
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
                        egui::Button::new(RichText::new("↻").size(13.0))
                            .corner_radius(CornerRadius::same(6))
                            .min_size(vec2(28.0, 26.0)),
                    )
                    .on_hover_text("Refresh cameras")
                    .clicked()
                {
                    ui_state.emit(UiAction::RefreshCameras);
                }
            });
        }
    });

    section_caption(ui, "Viewport");
    group(ui, |ui| {
        settings_row(ui, "Camera pose", |ui| {
            let can_reset = vm.can_reset_camera();
            if plain_button(ui, "Reset", can_reset).clicked() {
                ui_state.emit(UiAction::ResetAvatarCamera);
            }
        });
    });
    caption(ui, "Left drag: Orbit · Right drag: Pan · Wheel: Dolly");
}

// ---------------------------------------------------------------------------
// Avatar pane
// ---------------------------------------------------------------------------

/// Lifecycle color from the macOS semantic palette.
fn lifecycle_color(state: AvatarLifecycleState) -> Color32 {
    match state {
        AvatarLifecycleState::Ready => OK_GREEN,
        AvatarLifecycleState::Loading | AvatarLifecycleState::Binding => INFO_BLUE,
        AvatarLifecycleState::Failed => ALERT_RED,
        AvatarLifecycleState::Unloading => WARNING_AMBER,
        AvatarLifecycleState::None => SECONDARY,
    }
}

/// VRM model import and arm pose settings.
pub fn render_avatar_pane(
    ui: &mut Ui,
    vm: &UiViewModel,
    ui_state: &mut super::UiState,
    file_dialog: &mut FileDialogState,
) {
    section_caption(ui, "Model");
    match &vm.avatar.imported_model {
        Some(model) => {
            group(ui, |ui| {
                info_row(ui, "Name", &model.name, LABEL);
                row_separator(ui);
                info_row(
                    ui,
                    "Format",
                    match model.generation {
                        crate::import::VrmGeneration::Vrm0 => "VRM 0.x",
                        crate::import::VrmGeneration::Vrm1 => "VRM 1.0",
                    },
                    LABEL,
                );
                row_separator(ui);
                info_row(ui, "Expressions", &model.expression_count.to_string(), LABEL);
                row_separator(ui);
                let lc = lifecycle_color(vm.avatar.lifecycle);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Status").size(13.0).color(SECONDARY));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        status_text(ui, lc, &format!("{:?}", vm.avatar.lifecycle));
                    });
                });
                row_separator(ui);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if destructive_button(ui, "Unload", true).clicked() {
                        ui_state.emit(UiAction::UnloadAvatar);
                    }
                    if filled_button(ui, "Import VRM…", true).clicked() && !file_dialog.is_active()
                    {
                        file_dialog.start();
                    }
                });
                if vm.avatar.lifecycle == AvatarLifecycleState::Failed {
                    row_separator(ui);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if plain_button(ui, "Retry Load", true).clicked() {
                            ui_state.emit(UiAction::RetryAfterError);
                        }
                    });
                }
            });
            section_caption(ui, "Arm pose");
            group(ui, |ui| {
                render_arm_pose_settings(ui, vm, ui_state);
            });
        }
        None => {
            group(ui, |ui| {
                caption(ui, "No avatar loaded. Import a VRM to begin.");
                ui.add_space(4.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if filled_button(ui, "Import VRM…", true).clicked() && !file_dialog.is_active()
                    {
                        file_dialog.start();
                    }
                });
            });
        }
    }
}

/// Arm pose sliders as grouped rows; emits a single action per change.
fn render_arm_pose_settings(ui: &mut Ui, vm: &UiViewModel, ui_state: &mut super::UiState) {
    caption(ui, "Saved per model by its content hash.");
    ui.add_space(4.0);
    let mut profile = vm.arm_pose.profile;
    let mut arm_drop_degrees = profile.arm_drop_radians.to_degrees();
    let mut finger_curl_degrees = profile.finger_curl_radians.to_degrees();
    let mut changed = false;

    settings_row(ui, "Arm drop", |ui| {
        changed |= ui
            .add(egui::Slider::new(&mut arm_drop_degrees, 0.0..=90.0))
            .changed();
    });
    row_separator(ui);
    settings_row(ui, "Reach ratio", |ui| {
        changed |= ui
            .add(
                egui::Slider::new(&mut profile.reach_ratio, 0.01..=1.0),
            )
            .changed();
    });
    row_separator(ui);
    settings_row(ui, "Forward offset", |ui| {
        changed |= ui
            .add(
                egui::Slider::new(&mut profile.forward_hand_offset_ratio, -1.0..=1.0)
                    ,
            )
            .changed();
    });
    row_separator(ui);
    settings_row(ui, "Elbow pole", |ui| {
        changed |= ui
            .add(
                egui::Slider::new(&mut profile.elbow_pole_offset_ratio, 0.0..=1.0)
                    ,
            )
            .changed();
    });
    row_separator(ui);
    settings_row(ui, "Shoulder follow", |ui| {
        changed |= ui
            .add(
                egui::Slider::new(&mut profile.shoulder_follow_weight, 0.0..=1.0)
                    ,
            )
            .changed();
    });
    row_separator(ui);
    settings_row(ui, "Finger curl", |ui| {
        changed |= ui
            .add(egui::Slider::new(&mut finger_curl_degrees, 0.0..=90.0))
            .changed();
    });

    profile.arm_drop_radians = arm_drop_degrees.to_radians();
    profile.finger_curl_radians = finger_curl_degrees.to_radians();
    if changed {
        ui_state.emit(UiAction::SetArmPoseProfile {
            profile: ArmPoseProfileOverride::from_profile(profile),
        });
    }

    if vm.arm_pose.has_override {
        row_separator(ui);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if plain_button(ui, "Reset to automatic", true).clicked() {
                ui_state.emit(UiAction::ResetArmPoseProfile);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Calibration pane
// ---------------------------------------------------------------------------

/// Neutral pose calibration.
pub fn render_calibration_pane(ui: &mut Ui, vm: &UiViewModel, ui_state: &mut super::UiState) {
    section_caption(ui, "Neutral pose");
    group(ui, |ui| {
        if vm.calibration.is_calibrating {
            settings_row(ui, "Calibrating", |ui| {
                ui.label(
                    RichText::new(format!(
                        "{}/{}",
                        vm.calibration.samples_collected, vm.calibration.samples_target
                    ))
                    .size(13.0)
                    .color(LABEL),
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
                    .corner_radius(CornerRadius::same(5))
                    .desired_height(8.0),
            );
            if let Some(score) = vm.calibration.quality_score {
                row_separator(ui);
                info_row(
                    ui,
                    "Quality",
                    &format!("{:.0}%", score * 100.0),
                    LABEL,
                );
            }
            if let Some(reason) = &vm.calibration.last_reject_reason {
                ui.add_space(4.0);
                caption(ui, &format!("Rejected: {reason}"));
            }
            row_separator(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if plain_button(ui, "Cancel", true).clicked() {
                    ui_state.emit(UiAction::CancelCalibration);
                }
            });
        } else if vm.calibration.is_complete {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let quality = vm
                    .calibration
                    .quality_score
                    .map(|score| format!("Calibrated · {:.0}%", score * 100.0))
                    .unwrap_or_else(|| "Calibrated".to_string());
                status_text(ui, OK_GREEN, &quality);
            });
            row_separator(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if plain_button(ui, "Redo", true).clicked() {
                    ui_state.emit(UiAction::RetryCalibration);
                }
            });
        } else if vm.can_calibrate() {
            caption(ui, "Face the camera with a relaxed expression, then begin.");
            ui.add_space(4.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if filled_button(ui, "Begin Calibration", true).clicked() {
                    ui_state.emit(UiAction::BeginCalibration);
                }
            });
        } else {
            caption(ui, "Start the session to calibrate.");
            ui.add_space(4.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                filled_button(ui, "Begin Calibration", false);
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Preview pane
// ---------------------------------------------------------------------------

/// Camera feed preview and display mirroring.
pub fn render_preview_pane(
    ui: &mut Ui,
    ui_state: &mut super::UiState,
    preview: &PreviewState,
    landmarks: &PreviewLandmarkState,
    avatar_motion_mirror: AvatarMotionMirror,
    preview_texture: Option<TextureId>,
) {
    section_caption(ui, "Display");
    group(ui, |ui| {
        let mut preview_visible = preview.visible;
        settings_row(ui, "Show preview", |ui| {
            if ui.checkbox(&mut preview_visible, "").changed() {
                ui_state.emit(UiAction::TogglePreview);
            }
        });
        row_separator(ui);
        let mut mirror = preview.mirrored;
        settings_row(ui, "Mirror preview", |ui| {
            if ui.checkbox(&mut mirror, "").changed() {
                ui_state.emit(UiAction::ToggleMirror);
            }
        });
        row_separator(ui);
        let mut mirror_avatar_motion = avatar_motion_mirror.is_enabled();
        settings_row(ui, "Mirror avatar motion", |ui| {
            if ui
                .checkbox(&mut mirror_avatar_motion, "")
                .on_hover_text("Reflect avatar motion for the operator view")
                .changed()
            {
                ui_state.emit(UiAction::ToggleAvatarMotionMirror);
            }
        });
    });

    if !preview.visible {
        caption(ui, "Preview hidden — tracking still runs in background.");
        return;
    }

    match preview_texture {
        Some(texture) => {
            let w = ui.available_width();
            let size = vec2(w, w * 9.0 / 16.0);
            let image_rect = ui
                .add(
                    bevy_egui::egui::Image::from_texture((texture, size))
                        .uv(preview_uv(preview.mirrored))
                        .corner_radius(CornerRadius::same(6)),
                )
                .rect;
            caption(
                ui,
                &format!("Camera feed · {} fps target", preview.target_fps),
            );
            let now = monotonic_now();
            if should_draw_landmark_overlay(preview.visible, Some(texture), landmarks, now) {
                draw_landmark_overlay(ui, image_rect, preview.mirrored, landmarks, now);
            }
        }
        None => {
            group(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.spinner();
                    ui.add_space(4.0);
                    caption(ui, "Waiting for camera frames…");
                });
            });
        }
    }
}

// ---------------------------------------------------------------------------
// NDI pane
// ---------------------------------------------------------------------------

/// NDI transparent avatar output.
pub fn render_ndi_pane(ui: &mut Ui, vm: &UiViewModel, ui_state: &mut super::UiState) {
    let ndi = ndi_live_section(vm);
    section_caption(ui, "NDI Output");
    group(ui, |ui| {
        info_row(ui, "Source", &ndi.source_name, LABEL);
        row_separator(ui);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Status").size(13.0).color(SECONDARY));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                status_text(ui, ndi_status_color(vm), &ndi.status_text);
            });
        });
        row_separator(ui);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if plain_button(ui, "Stop", ndi.stop_enabled).clicked() {
                ui_state.emit(ndi.stop_action.clone());
            }
            if filled_button(ui, "Start", ndi.start_enabled).clicked() {
                ui_state.emit(ndi.start_action.clone());
            }
        });
    });
    if let Some(hint) = ndi.unavailable_hint {
        caption(ui, hint);
    }
    if let Some(error) = &ndi.error_text {
        ui.add(
            bevy_egui::egui::Label::new(
                RichText::new(error).size(11.0).color(ALERT_RED),
            )
            .wrap(),
        );
    }
    ui.add_space(4.0);
    ui.hyperlink_to(
        RichText::new("NDI® info ↗").size(11.0).color(ACCENT),
        ndi.official_link,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preview_landmarks::PreviewLandmarkSnapshot;
    use crate::ui::UiState;

    #[test]
    fn mirror_preview_reverses_only_the_horizontal_uv_axis() {
        let normal = preview_uv(false);
        let mirrored = preview_uv(true);
        assert_eq!(normal.min, pos2(0.0, 0.0));
        assert_eq!(normal.max, pos2(1.0, 1.0));
        assert_eq!(mirrored.min, pos2(1.0, 0.0));
        assert_eq!(mirrored.max, pos2(0.0, 1.0));
    }

    #[test]
    fn landmark_overlay_maps_corners_and_center_to_preview_rect() {
        let rect = Rect::from_min_max(pos2(10.0, 20.0), pos2(110.0, 70.0));
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
            Some(pos2(10.0, 20.0))
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
            Some(pos2(60.0, 45.0))
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
            Some(pos2(110.0, 70.0))
        );
    }

    #[test]
    fn mirror_changes_x_only_for_landmark_overlay() {
        let rect = Rect::from_min_max(pos2(10.0, 20.0), pos2(110.0, 70.0));
        let landmark = FaceLandmark {
            x: 0.25,
            y: 0.2,
            ..FaceLandmark::default()
        };
        assert_eq!(
            landmark_overlay_position(rect, &landmark, false),
            Some(pos2(35.0, 30.0))
        );
        assert_eq!(
            landmark_overlay_position(rect, &landmark, true),
            Some(pos2(85.0, 30.0))
        );
    }

    #[test]
    fn invalid_landmark_coordinates_are_skipped() {
        let rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(100.0, 100.0));
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
            latest: Some(PreviewLandmarkSnapshot {
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
        let rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(100.0, 100.0));
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
        let mut ui_state = UiState::default();
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
