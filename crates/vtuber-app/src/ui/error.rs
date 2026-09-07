//! Error panel rendering.

use bevy_egui::egui::Ui;

use crate::actions::UiAction;
use crate::error_presenter::ErrorPresentation;
use crate::settings::UiLanguage;

/// Render the error panel if there's a current error.
pub fn render_error_panel(
    ui: &mut Ui,
    presentation: &ErrorPresentation,
    ui_state: &mut super::UiState,
    lang: UiLanguage,
) {
    ui.horizontal(|ui| {
        ui.colored_label(bevy_egui::egui::Color32::LIGHT_RED, "⚠");
        ui.colored_label(
            bevy_egui::egui::Color32::LIGHT_RED,
            &presentation.user_message,
        );
    });
    ui.label(format!(
        "{}: {}",
        lang.pick("コード", "Code"),
        presentation.code
    ));

    if !presentation.suggested_actions.is_empty() {
        ui.horizontal(|ui| {
            for action in &presentation.suggested_actions {
                let label = match action {
                    UiAction::DismissError => lang.pick("閉じる", "Dismiss"),
                    UiAction::RetryAfterError => lang.pick("再試行", "Retry"),
                    UiAction::RefreshCameras => lang.pick("カメラを再検出", "Refresh Cameras"),
                    _ => continue,
                };
                if ui.button(label).clicked() {
                    ui_state.emit(action.clone());
                }
            }
        });
    }
}
