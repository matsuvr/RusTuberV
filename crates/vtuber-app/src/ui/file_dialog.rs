//! Async file dialog handling for VRM import.

use crate::actions::UiAction;
use crate::settings::UiLanguage;
use bevy::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Resource managing the async file dialog state.
#[derive(Resource, Clone, Default)]
pub struct FileDialogState {
    inner: Arc<Mutex<FileDialogInner>>,
}
#[derive(Default)]
struct FileDialogInner {
    active: bool,
    result: Option<Option<PathBuf>>,
}

impl FileDialogState {
    /// Whether a dialog is already active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
    }

    /// Start a new dialog. Called by the shell's side-effect boundary, not by
    /// the snapshot-rendering functions. Native dialog buttons follow the OS.
    pub fn start(&mut self, lang: UiLanguage) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if inner.active {
            return;
        }
        inner.active = true;
        inner.result = None;
        let state = self.inner.clone();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(|| {
                let rt = tokio::runtime::Runtime::new().ok()?;
                rt.block_on(async {
                    let handle = rfd::AsyncFileDialog::new()
                        .add_filter(
                            lang.pick("VRM モデル", "VRM models", "VRM 模型", "VRM 모델"),
                            &["vrm"],
                        )
                        .set_title(lang.pick(
                            "VRM モデルを選択 (0.x / 1.0)",
                            "Select VRM model (0.x or 1.0)",
                            "选择VRM模型 (0.x / 1.0)",
                            "VRM 모델 선택 (0.x / 1.0)",
                        ))
                        .pick_file()
                        .await;
                    handle.map(|handle| handle.path().to_path_buf())
                })
            });
            let mut inner = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.active = false;
            inner.result = Some(result.ok().flatten());
        });
    }

    /// Take a completed dialog result.
    pub fn take_result(&mut self) -> Option<Option<PathBuf>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .result
            .take()
    }
}

/// Poll the dialog and emit the import command.
pub fn poll_file_dialog(state: &mut FileDialogState, ui_state: &mut super::UiState) {
    if let Some(Some(path)) = state.take_result() {
        ui_state.emit(UiAction::ImportAvatar { path });
    }
}

/// Accept the first dropped VRM file.
pub fn handle_dropped_files(ctx: &bevy_egui::egui::Context, ui_state: &mut super::UiState) {
    for event in ctx.input(|input| input.raw.dropped_files.clone()) {
        if let Some(path) = event.path {
            let path_buf = PathBuf::from(&path);
            if let Some(ext) = path_buf.extension()
                && ext.to_string_lossy().to_lowercase() == "vrm"
            {
                ui_state.emit(UiAction::ImportAvatar { path: path_buf });
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn file_dialog_default_is_inactive() {
        assert!(!FileDialogState::default().is_active());
    }
    #[test]
    fn vrm_extension_check() {
        for name in ["test.vrm", "test.VRM"] {
            let path = PathBuf::from(name);
            assert_eq!(
                path.extension().unwrap().to_string_lossy().to_lowercase(),
                "vrm"
            );
        }
    }
}
