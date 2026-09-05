//! Shared macOS-style widgets for the sidebar and settings panes.
//!
//! Follows the macOS System Settings visual language: grouped lists with
//! hairline separators, a single accent color, and a compact type scale
//! (15pt pane titles, 13pt body, 11pt captions).

use bevy_egui::egui::{
    self, Button, Color32, CornerRadius, Frame, Margin, Response, RichText, Ui, vec2,
};

use crate::ui_model::AppLifecycle;

/// macOS systemBlue (dark mode) — the single accent color.
pub(crate) const ACCENT: Color32 = Color32::from_rgb(10, 132, 255);
/// macOS `secondaryLabelColor` (dark mode) for captions and row labels.
pub(crate) const SECONDARY: Color32 = Color32::from_rgb(152, 155, 163);
/// macOS `labelColor` (dark mode) for primary values and titles.
pub(crate) const LABEL: Color32 = Color32::from_rgb(235, 237, 240);
/// macOS `systemGreen` (dark mode) for healthy status text.
pub(crate) const OK_GREEN: Color32 = Color32::from_rgb(48, 209, 88);
/// macOS `systemRed` (dark mode) for failure status text.
pub(crate) const ALERT_RED: Color32 = Color32::from_rgb(255, 105, 97);
/// macOS `systemYellow` (dark mode) for in-progress status text.
pub(crate) const WARNING_AMBER: Color32 = Color32::from_rgb(255, 214, 10);
/// macOS `systemBlue` (dark mode) for in-flight status text.
pub(crate) const INFO_BLUE: Color32 = Color32::from_rgb(100, 210, 255);

/// Section caption above a group: 11pt secondary text.
pub(crate) fn section_caption(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).size(11.0).color(SECONDARY));
    ui.add_space(3.0);
}

/// Hairline separator between rows inside a group.
pub(crate) fn row_separator(ui: &mut Ui) {
    ui.add_space(2.0);
    ui.separator();
    ui.add_space(2.0);
}

/// A rounded group container in the macOS settings style.
pub(crate) fn group(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    Frame::new()
        .fill(Color32::from_rgba_unmultiplied(255, 255, 255, 14))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
    ui.add_space(10.0);
}

/// A settings row: 13pt label on the left, caller-drawn control on the right.
pub(crate) fn settings_row(ui: &mut Ui, label: &str, control: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(13.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), control);
    });
}

/// An informational row: secondary label on the left, value on the right.
pub(crate) fn info_row(ui: &mut Ui, label: &str, value: &str, value_color: Color32) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(13.0).color(SECONDARY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(value).size(13.0).color(value_color));
        });
    });
}

/// A small status indicator: colored dot plus colored text.
pub(crate) fn status_text(ui: &mut Ui, color: Color32, text: &str) {
    ui.label(RichText::new("●").size(10.0).color(color));
    ui.label(RichText::new(text).size(13.0).color(color));
}

/// Primary action: filled accent button, 28pt tall.
pub(crate) fn filled_button(ui: &mut Ui, text: &str, enabled: bool) -> Response {
    let button = Button::new(
        RichText::new(text)
            .size(13.0)
            .color(if enabled { Color32::WHITE } else { SECONDARY }),
    )
    .fill(if enabled {
        ACCENT
    } else {
        Color32::from_rgba_unmultiplied(255, 255, 255, 20)
    })
    .corner_radius(CornerRadius::same(6))
    .min_size(vec2(90.0, 28.0));
    ui.add_enabled(enabled, button)
}

/// Secondary action: plain bordered button, 28pt tall.
pub(crate) fn plain_button(ui: &mut Ui, text: &str, enabled: bool) -> Response {
    let button = Button::new(
        RichText::new(text)
            .size(13.0)
            .color(if enabled {
                ui.visuals().text_color()
            } else {
                SECONDARY
            }),
    )
    .corner_radius(CornerRadius::same(6))
    .min_size(vec2(80.0, 28.0));
    ui.add_enabled(enabled, button)
}

/// Destructive action: plain button with red text when enabled.
pub(crate) fn destructive_button(ui: &mut Ui, text: &str, enabled: bool) -> Response {
    let button = Button::new(
        RichText::new(text)
            .size(13.0)
            .color(if enabled { ALERT_RED } else { SECONDARY }),
    )
    .corner_radius(CornerRadius::same(6))
    .min_size(vec2(80.0, 28.0));
    ui.add_enabled(enabled, button)
}

/// 11pt secondary caption, wrapped.
pub(crate) fn caption(ui: &mut Ui, text: &str) {
    ui.add(
        egui::Label::new(RichText::new(text).size(11.0).color(SECONDARY)).wrap(),
    );
}

/// Lifecycle label and semantic color for the session status.
pub(crate) fn app_lifecycle_text(lifecycle: AppLifecycle) -> (Color32, &'static str) {
    match lifecycle {
        AppLifecycle::Idle => (SECONDARY, "Idle"),
        AppLifecycle::Starting => (INFO_BLUE, "Starting"),
        AppLifecycle::Running => (OK_GREEN, "Running"),
        AppLifecycle::Stopping => (WARNING_AMBER, "Stopping"),
        AppLifecycle::Failed => (ALERT_RED, "Failed"),
    }
}
