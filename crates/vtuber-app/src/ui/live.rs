//! Live screen rendering.

use bevy_egui::egui::{Color32, Rect, Ui};

use crate::actions::UiAction;
use crate::preview::PreviewState;
use crate::preview_landmarks::PreviewLandmarkState;
use crate::ui_model::{NdiOutputUiState, UiViewModel};
use vtuber_avatar::AvatarMotionMirror;
use vtuber_core::{FaceLandmark, MonoTimeNs, monotonic_now};

fn preview_uv(mirrored: bool) -> bevy_egui::egui::Rect {
    if mirrored {
        bevy_egui::egui::Rect::from_min_max(
            bevy_egui::egui::pos2(1.0, 0.0),
            bevy_egui::egui::pos2(0.0, 1.0),
        )
    } else {
        bevy_egui::egui::Rect::from_min_max(
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
    preview_texture: Option<bevy_egui::egui::TextureId>,
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

/// Official NDI site shown beside the Live output controls.
pub(crate) const NDI_OFFICIAL_URL: &str = "https://ndi.video";

/// Pure presentation of the Live NDI section.
///
/// The UI renderer reads this snapshot and emits the contained actions. It
/// never receives an NDI controller or sender handle.
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
        heading: "NDI® Output",
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

/// Render the Live screen.
pub fn render_live_screen(
    ui: &mut Ui,
    vm: &UiViewModel,
    ui_state: &mut super::UiState,
    preview: &PreviewState,
    landmarks: &PreviewLandmarkState,
    avatar_motion_mirror: AvatarMotionMirror,
    preview_texture: Option<bevy_egui::egui::TextureId>,
) {
    ui.heading("Live");
    ui.separator();

    // Lifecycle and tracking status.
    ui.heading("Status");
    ui.label(format!("Lifecycle: {:?}", vm.lifecycle));
    ui.label(format!("Tracking: {:?}", vm.tracking.state));
    ui.label(format!("Confidence: {:.2}", vm.tracking.confidence));
    ui.label(format!(
        "Face detected: {}",
        if vm.tracking.face_detected {
            "yes"
        } else {
            "no"
        }
    ));
    ui.separator();

    // Main viewport camera controls. The renderer emits an intent only; the
    // avatar domain owns the camera transform, target, and FOV state.
    ui.heading("Camera");
    if vm.can_reset_camera() {
        if ui.button("Reset Camera").clicked() {
            ui_state.emit(UiAction::ResetAvatarCamera);
        }
    } else {
        ui.add_enabled(false, bevy_egui::egui::Button::new("Reset Camera"));
    }
    ui.label("Left drag: Orbit · Right drag: Pan · Wheel: Dolly");
    ui.separator();

    // Calibration section.
    ui.heading("Calibration");
    if vm.calibration.is_calibrating {
        ui.label(format!(
            "Samples: {}/{}",
            vm.calibration.samples_collected, vm.calibration.samples_target
        ));
        if let Some(score) = vm.calibration.quality_score {
            ui.label(format!("Quality: {:.2}", score));
        }
        if let Some(reason) = &vm.calibration.last_reject_reason {
            ui.colored_label(
                bevy_egui::egui::Color32::LIGHT_RED,
                format!("Rejected: {reason}"),
            );
        }
        if ui.button("Cancel").clicked() {
            ui_state.emit(UiAction::CancelCalibration);
        }
    } else if vm.calibration.is_complete {
        ui.colored_label(
            bevy_egui::egui::Color32::LIGHT_GREEN,
            "Calibration complete.",
        );
        if ui.button("Retry").clicked() {
            ui_state.emit(UiAction::RetryCalibration);
        }
    } else if vm.can_calibrate() {
        if ui.button("Begin Calibration").clicked() {
            ui_state.emit(UiAction::BeginCalibration);
        }
    } else {
        ui.add_enabled(false, bevy_egui::egui::Button::new("Begin Calibration"));
        ui.label("Start tracking to calibrate.");
    }
    ui.separator();

    // Preview section.
    ui.heading("Preview");
    let mut preview_visible = preview.visible;
    if ui.checkbox(&mut preview_visible, "Show Preview").changed() {
        ui_state.emit(UiAction::TogglePreview);
    }
    let mut mirror = preview.mirrored;
    if ui.checkbox(&mut mirror, "Mirror Preview").changed() {
        ui_state.emit(UiAction::ToggleMirror);
    }
    let mut mirror_avatar_motion = avatar_motion_mirror.is_enabled();
    if ui
        .checkbox(&mut mirror_avatar_motion, "Mirror Avatar Motion")
        .changed()
    {
        ui_state.emit(UiAction::ToggleAvatarMotionMirror);
    }

    if preview.visible {
        if let Some(texture) = preview_texture {
            let available = ui.available_width().max(160.0);
            let size = bevy_egui::egui::vec2(available, available * 9.0 / 16.0);
            let response = ui.add(
                bevy_egui::egui::Image::from_texture((texture, size))
                    .uv(preview_uv(preview.mirrored)),
            );
            let now = monotonic_now();
            if should_draw_landmark_overlay(preview.visible, Some(texture), landmarks, now) {
                draw_landmark_overlay(ui, response.rect, preview.mirrored, landmarks, now);
            }
        } else {
            ui.label("Waiting for camera frames…");
        }
    }
    ui.separator();

    // Optional transparent NDI output. This section intentionally exposes
    // only the fixed application profile; source, resolution, FPS, codec,
    // network, and group configuration remain outside the Live UI contract.
    let ndi = ndi_live_section(vm);
    ui.heading(ndi.heading);
    ui.horizontal(|ui| {
        ui.label("Source:");
        ui.monospace(&ndi.source_name);
    });
    ui.label(format!("Status: {}", ndi.status_text));
    ui.horizontal(|ui| {
        if ui
            .add_enabled(ndi.start_enabled, bevy_egui::egui::Button::new("Start"))
            .clicked()
        {
            ui_state.emit(ndi.start_action.clone());
        }
        if ui
            .add_enabled(ndi.stop_enabled, bevy_egui::egui::Button::new("Stop"))
            .clicked()
        {
            ui_state.emit(ndi.stop_action.clone());
        }
    });
    if let Some(hint) = ndi.unavailable_hint {
        ui.small(hint);
    }
    if let Some(error) = &ndi.error_text {
        ui.colored_label(bevy_egui::egui::Color32::LIGHT_RED, error);
    }
    if vm.ndi_output.dropped_frames > 0 || vm.ndi_output.replaced_frames > 0 {
        ui.small(format!(
            "Mailbox: {} dropped, {} replaced",
            vm.ndi_output.dropped_frames, vm.ndi_output.replaced_frames
        ));
    }
    ui.hyperlink_to("NDI® SDK information", ndi.official_link);
    ui.separator();

    // Start/Stop buttons.
    if vm.can_stop() && ui.button("Stop").clicked() {
        ui_state.emit(UiAction::Stop);
    }
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
        let rect = bevy_egui::egui::Rect::from_min_max(
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
        let rect = bevy_egui::egui::Rect::from_min_max(
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
        let rect = bevy_egui::egui::Rect::from_min_max(
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
        let texture = bevy_egui::egui::TextureId::User(1);
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
        let rect = bevy_egui::egui::Rect::from_min_max(
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
        assert_eq!(section.heading, "NDI® Output");
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
