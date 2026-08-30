//! Async file dialog handling for VRM import.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use crate::actions::UiAction;

/// Resource managing the async file dialog state.
#[derive(Resource, Clone, Default)]
pub struct FileDialogState {
    /// Shared state for the async file dialog.
    inner: Arc<Mutex<FileDialogInner>>,
}

/// Internal file dialog state.
#[derive(Default)]
struct FileDialogInner {
    /// Whether a dialog is currently active.
    active: bool,
    /// Result from the last completed dialog.
    result: Option<Option<PathBuf>>,
}

impl FileDialogState {
    /// Check if a file dialog is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
    }

    /// Start a new file dialog.
    pub fn start(&mut self) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if inner.active {
            return;
        }
        inner.active = true;
        inner.result = None;

        // Spawn the async file dialog task.
        let state = self.inner.clone();
        std::thread::spawn(move || {
            // Use a simple blocking approach in a separate thread.
            // This is acceptable because it's in a separate thread.
            let result = std::panic::catch_unwind(|| {
                // Create a new runtime for the async file dialog.
                let rt = tokio::runtime::Runtime::new().ok()?;
                rt.block_on(async {
                    let handle = rfd::AsyncFileDialog::new()
                        .add_filter("VRM models", &["vrm"])
                        .set_title("Select VRM model (0.x or 1.0)")
                        .pick_file()
                        .await;
                    handle.map(|h| h.path().to_path_buf())
                })
            });

            let mut inner = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.active = false;
            inner.result = Some(result.ok().flatten());
        });
    }

    /// Take the result if available.
    pub fn take_result(&mut self) -> Option<Option<PathBuf>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .result
            .take()
    }
}

/// Poll the file dialog and emit import action if a file was selected.
pub fn poll_file_dialog(state: &mut FileDialogState, ui_state: &mut super::UiState) {
    if let Some(Some(path)) = state.take_result() {
        ui_state.emit(UiAction::ImportAvatar { path });
    }
}

/// Handle dropped files from egui context.
pub fn handle_dropped_files(ctx: &bevy_egui::egui::Context, ui_state: &mut super::UiState) {
    for event in ctx.input(|i| i.raw.dropped_files.clone()) {
        if let Some(path) = event.path {
            let path_buf = PathBuf::from(&path);
            if let Some(ext) = path_buf.extension()
                && ext.to_string_lossy().to_lowercase() == "vrm"
            {
                ui_state.emit(UiAction::ImportAvatar { path: path_buf });
                break; // Only accept the first .vrm file.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_dialog_default_is_inactive() {
        let state = FileDialogState::default();
        assert!(!state.is_active());
    }

    #[test]
    fn vrm_extension_check() {
        // Test the extension matching logic.
        let path = PathBuf::from("test.vrm");
        let ext = path.extension().unwrap().to_string_lossy().to_lowercase();
        assert_eq!(ext, "vrm");

        let path_upper = PathBuf::from("test.VRM");
        let ext_upper = path_upper
            .extension()
            .unwrap()
            .to_string_lossy()
            .to_lowercase();
        assert_eq!(ext_upper, "vrm");
    }
}
