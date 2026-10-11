//! Async file dialog handling for VRM import.

use crate::actions::{ActionQueue, UiAction};
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
        self.start_kind(lang, false);
    }

    /// Pick the raw calibration belonging to a physical Looking Glass.
    pub fn start_calibration(&mut self, lang: UiLanguage) {
        self.start_kind(lang, true);
    }

    fn start_kind(&mut self, lang: UiLanguage, calibration: bool) {
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
                    let dialog = rfd::AsyncFileDialog::new();
                    let dialog = if calibration {
                        dialog
                            .add_filter("Looking Glass calibration", &["json"])
                            .set_title(lang.pick(
                                "実機の visual.json を選択",
                                "Select this display's visual.json",
                                "选择此显示器的 visual.json",
                                "이 디스플레이의 visual.json 선택",
                            ))
                    } else {
                        dialog.add_filter("VRM", &["vrm"]).set_title(lang.pick(
                            "VRM モデルを選択 (0.x / 1.0)",
                            "Select VRM model (0.x or 1.0)",
                            "选择VRM模型 (0.x / 1.0)",
                            "VRM 모델 선택 (0.x / 1.0)",
                        ))
                    };
                    let handle = dialog.pick_file().await;
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

/// Poll the dialog and request a license review for the selection.
pub fn poll_file_dialog(
    actions: &mut ActionQueue,
    state: &mut FileDialogState,
    ui_state: &mut super::UiState,
) {
    if let Some(Some(path)) = state.take_result() {
        if ui_state.dialog_for_calibration {
            if let Some(selection) = &mut ui_state.looking_glass_selection {
                selection.calibration_path = path;
            }
        } else {
            ui_state.emit(actions, UiAction::RequestAvatarImportReview { path });
        }
    }
}

/// Accept the first dropped VRM file as a license review request.
pub fn handle_dropped_files(
    actions: &mut ActionQueue,
    ctx: &bevy_egui::egui::Context,
    ui_state: &mut super::UiState,
) {
    for event in ctx.input(|input| input.raw.dropped_files.clone()) {
        let path_buf = event.path().to_path_buf();
        if let Some(ext) = path_buf.extension()
            && ext.to_string_lossy().to_lowercase() == "vrm"
        {
            ui_state.emit(
                actions,
                UiAction::RequestAvatarImportReview { path: path_buf },
            );
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
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
