//! Setup screen rendering.

use bevy_egui::egui::Ui;

use crate::actions::UiAction;
use crate::error_presenter::ErrorPresentation;
use crate::ui_model::{AvatarLifecycleState, UiViewModel};
use vtuber_avatar::ArmPoseProfileOverride;

use super::error::render_error_panel;
use super::file_dialog::FileDialogState;

/// Render the Setup screen.
pub fn render_setup_screen(
    ui: &mut Ui,
    vm: &UiViewModel,
    ui_state: &mut super::UiState,
    face_backend: crate::face_backend::FaceTrackingBackendSelection,
    file_dialog: &mut FileDialogState,
    current_error: Option<&ErrorPresentation>,
) {
    ui.heading("Setup");
    ui.separator();

    // Camera section.
    ui.heading("Camera");
    if vm.camera.available_cameras.is_empty() {
        ui.label("No cameras detected.");
        if ui.button("Refresh Cameras").clicked() {
            ui_state.emit(UiAction::RefreshCameras);
        }
    } else {
        // Keep the unselected state distinct from camera index 0. Otherwise a
        // one-camera list renders camera 0 but selecting it produces no change
        // event, leaving Start disabled forever.
        let selected = vm.camera.selected_index.unwrap_or(usize::MAX);
        let mut new_selected = selected;
        bevy_egui::egui::ComboBox::from_label("Select camera")
            .selected_text(
                vm.camera
                    .available_cameras
                    .get(selected)
                    .map(|c| c.name.as_str())
                    .unwrap_or("None"),
            )
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
    }
    ui.separator();

    // Avatar section.
    ui.heading("Avatar");
    if let Some(presentation) = current_error {
        ui.heading("Current error");
        render_error_panel(ui, presentation, ui_state);
        ui.separator();
    }

    if let Some(model) = &vm.avatar.imported_model {
        ui.label(format!("Model: {}", model.name));
        let generation = match model.generation {
            crate::import::VrmGeneration::Vrm0 => "VRM 0.x",
            crate::import::VrmGeneration::Vrm1 => "VRM 1.0",
        };
        ui.label(format!("Generation: {generation}"));
        ui.label(format!("ID: {}", &model.id[..8.min(model.id.len())]));
        ui.label(format!(
            "Required bones: {}",
            if model.has_required_bones {
                "yes"
            } else {
                "no"
            }
        ));
        ui.label(format!("Expressions: {}", model.expression_count));

        render_face_tracking_backend_settings(ui, ui_state, face_backend);

        render_arm_pose_settings(ui, vm, ui_state);

        // Lifecycle status for the avatar.
        match vm.avatar.lifecycle {
            AvatarLifecycleState::None => {
                ui.label("Status: Waiting to load…");
            }
            AvatarLifecycleState::Loading => {
                ui.colored_label(
                    bevy_egui::egui::Color32::LIGHT_BLUE,
                    "Status: Loading model…",
                );
            }
            AvatarLifecycleState::Binding => {
                ui.colored_label(
                    bevy_egui::egui::Color32::LIGHT_YELLOW,
                    "Status: Binding bones…",
                );
            }
            AvatarLifecycleState::Ready => {
                ui.colored_label(bevy_egui::egui::Color32::LIGHT_GREEN, "Status: Ready");
            }
            AvatarLifecycleState::Unloading => {
                ui.label("Status: Unloading…");
            }
            AvatarLifecycleState::Failed => {
                ui.colored_label(bevy_egui::egui::Color32::LIGHT_RED, "Status: Load failed");
                if ui.button("Retry Load").clicked() {
                    ui_state.emit(UiAction::RetryAfterError);
                }
            }
        }

        if ui.button("Unload Avatar").clicked() {
            ui_state.emit(UiAction::UnloadAvatar);
        }
    } else {
        ui.label("No avatar loaded.");
    }

    if ui.button("Import VRM...").clicked() && !file_dialog.is_active() {
        file_dialog.start();
    }

    ui.separator();

    // Lifecycle display.
    ui.heading("Status");
    ui.label(format!("App lifecycle: {:?}", vm.lifecycle));
    ui.label(format!("Avatar lifecycle: {:?}", vm.avatar.lifecycle));

    ui.separator();

    // Start/Stop buttons.
    if vm.can_start() {
        if ui.button("Start").clicked() {
            ui_state.emit(UiAction::Start);
        }
    } else {
        ui.add_enabled(false, bevy_egui::egui::Button::new("Start"));
        if vm.camera.selected_index.is_none() {
            ui.label("Select a camera to start.");
        }
        if !vm.avatar.is_ready {
            ui.label("Import an avatar to start.");
        }
    }

    if vm.can_stop() && ui.button("Stop").clicked() {
        ui_state.emit(UiAction::Stop);
    }
}

fn render_face_tracking_backend_settings(
    ui: &mut Ui,
    ui_state: &mut super::UiState,
    current: crate::face_backend::FaceTrackingBackendSelection,
) {
    use crate::face_backend::FaceTrackingBackendSelection;
    ui.collapsing("Face Tracking", |ui| {
        ui.small("Selects the requested face tracking backend.");
        for selection in FaceTrackingBackendSelection::all() {
            if ui.radio(selection == current, selection.label()).clicked() {
                ui_state.emit(UiAction::SetFaceTrackingBackend(selection));
            }
        }
        if current != FaceTrackingBackendSelection::DirectMediaPipe {
            ui.small("Falls back to Direct automatically while the backend is unavailable.");
        }
    });
}

fn render_arm_pose_settings(ui: &mut Ui, vm: &UiViewModel, ui_state: &mut super::UiState) {
    ui.collapsing("Default arm pose", |ui| {
        ui.small("Saved per model by its content hash.");
        let mut profile = vm.arm_pose.profile;
        let mut arm_drop_degrees = profile.arm_drop_radians.to_degrees();
        let mut finger_curl_degrees = profile.finger_curl_radians.to_degrees();
        let mut changed = false;
        changed |= ui
            .add(
                bevy_egui::egui::Slider::new(&mut arm_drop_degrees, 0.0..=90.0)
                    .text("Arm drop (deg)"),
            )
            .changed();
        changed |= ui
            .add(
                bevy_egui::egui::Slider::new(&mut profile.reach_ratio, 0.01..=1.0)
                    .text("Reach ratio"),
            )
            .changed();
        changed |= ui
            .add(
                bevy_egui::egui::Slider::new(&mut profile.forward_hand_offset_ratio, -1.0..=1.0)
                    .text("Forward offset"),
            )
            .changed();
        changed |= ui
            .add(
                bevy_egui::egui::Slider::new(&mut profile.elbow_pole_offset_ratio, 0.0..=1.0)
                    .text("Elbow pole"),
            )
            .changed();
        changed |= ui
            .add(
                bevy_egui::egui::Slider::new(&mut profile.shoulder_follow_weight, 0.0..=1.0)
                    .text("Shoulder follow"),
            )
            .changed();
        changed |= ui
            .add(
                bevy_egui::egui::Slider::new(&mut finger_curl_degrees, 0.0..=90.0)
                    .text("Finger curl (deg)"),
            )
            .changed();

        profile.arm_drop_radians = arm_drop_degrees.to_radians();
        profile.finger_curl_radians = finger_curl_degrees.to_radians();
        if changed {
            ui_state.emit(UiAction::SetArmPoseProfile {
                profile: ArmPoseProfileOverride::from_profile(profile),
            });
        }
        if vm.arm_pose.has_override && ui.button("Reset to automatic").clicked() {
            ui_state.emit(UiAction::ResetArmPoseProfile);
        }
        if !vm.arm_pose.has_override {
            ui.small("Using automatic geometry-derived defaults.");
        }
    });
}
