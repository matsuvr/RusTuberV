//! Desktop studio: navigation, settings, and an always-visible clean avatar preview.
//! UI rendering reads snapshots and emits existing application actions.

mod avatar_preview;
pub mod file_dialog;
mod fonts;
mod privacy;
pub mod shell;
mod studio;

pub use avatar_preview::{
    AvatarPreviewPlugin, AvatarPreviewTexture, paint_avatar_preview, paint_avatar_preview_at,
};
pub use shell::{UiShellPlugin, UiState};

#[cfg(test)]
fn run_headless_ui(
    context: &bevy_egui::egui::Context,
    input: bevy_egui::egui::RawInput,
    render: impl FnMut(&mut bevy_egui::egui::Ui),
) {
    // These tests inspect widgets and actions without a GPU renderer. Discard
    // the resulting texture uploads explicitly, as required by egui.
    let mut output = context.run_ui(input, render);
    output.textures_delta.clear();
}
