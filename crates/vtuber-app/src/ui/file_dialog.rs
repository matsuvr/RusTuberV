//! Async file dialog handling for VRM import.

use crate::actions::{ActionQueue, UiAction};
use crate::settings::UiLanguage;
use bevy::prelude::*;
use std::path::PathBuf;
use std::thread::JoinHandle;

/// Completion of a file selection, including failures outside user control.
#[derive(Debug, PartialEq, Eq)]
pub enum FileDialogOutcome {
    /// A file was selected.
    Selected(PathBuf),
    /// The user dismissed the dialog.
    Cancelled,
    /// The runtime, dialog, or worker failed.
    Failed(String),
}

/// Resource managing the async file dialog state.
#[derive(Resource, Default)]
pub struct FileDialogState {
    worker: Option<JoinHandle<Result<Option<PathBuf>, String>>>,
    result: Option<FileDialogOutcome>,
}

impl FileDialogState {
    /// Whether a dialog is already active or awaiting collection.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.worker.is_some()
    }

    /// Start a new dialog at the shell's side-effect boundary.
    pub fn start(&mut self, lang: UiLanguage) {
        if self.is_active() {
            return;
        }
        self.result = None;
        match std::thread::Builder::new()
            .name("file-dialog".into())
            .spawn(move || {
                let rt = tokio::runtime::Runtime::new().map_err(|error| {
                    format!("file dialog runtime initialization failed: {error}")
                })?;
                Ok(rt.block_on(async {
                    let dialog = rfd::AsyncFileDialog::new()
                        .add_filter("VRM", &["vrm"])
                        .set_title(lang.pick(
                            "VRM モデルを選択 (0.x / 1.0)",
                            "Select VRM model (0.x or 1.0)",
                            "选择VRM模型 (0.x / 1.0)",
                            "VRM 모델 선택 (0.x / 1.0)",
                        ));
                    dialog
                        .pick_file()
                        .await
                        .map(|handle| handle.path().to_path_buf())
                }))
            }) {
            Ok(worker) => self.worker = Some(worker),
            Err(error) => {
                self.result = Some(FileDialogOutcome::Failed(format!(
                    "file dialog worker could not start: {error}"
                )))
            }
        }
    }

    /// Take a completed dialog result without waiting for an open dialog.
    pub fn take_result(&mut self) -> Option<FileDialogOutcome> {
        if self.worker.as_ref().is_some_and(JoinHandle::is_finished) {
            let worker = self.worker.take()?;
            self.result = Some(dialog_outcome(worker.join()));
        }
        self.result.take()
    }
}

fn dialog_outcome(
    result: std::thread::Result<Result<Option<PathBuf>, String>>,
) -> FileDialogOutcome {
    match result {
        Ok(Ok(Some(path))) => FileDialogOutcome::Selected(path),
        Ok(Ok(None)) => FileDialogOutcome::Cancelled,
        Ok(Err(error)) => FileDialogOutcome::Failed(error),
        Err(payload) => {
            let detail = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic payload");
            FileDialogOutcome::Failed(format!("file dialog panicked: {detail}"))
        }
    }
}

/// Poll the dialog and request a license review or report its failure.
pub fn poll_file_dialog(
    actions: &mut ActionQueue,
    state: &mut FileDialogState,
    ui_state: &mut super::UiState,
) {
    match state.take_result() {
        Some(FileDialogOutcome::Selected(path)) => {
            ui_state.emit(actions, UiAction::RequestAvatarImportReview { path })
        }
        Some(FileDialogOutcome::Failed(error)) => {
            ui_state.emit(actions, UiAction::FileDialogFailed(error))
        }
        Some(FileDialogOutcome::Cancelled) | None => {}
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

    #[test]
    fn selection_cancellation_and_failures_remain_distinct() {
        let path = PathBuf::from("avatar.vrm");
        assert_eq!(
            dialog_outcome(Ok(Ok(Some(path.clone())))),
            FileDialogOutcome::Selected(path)
        );
        assert_eq!(dialog_outcome(Ok(Ok(None))), FileDialogOutcome::Cancelled);
        assert_eq!(
            dialog_outcome(Ok(Err("runtime failed".into()))),
            FileDialogOutcome::Failed("runtime failed".into())
        );
        for payload in [
            Box::new("dialog panic") as Box<dyn std::any::Any + Send>,
            Box::new(String::from("dialog panic")),
        ] {
            assert_eq!(
                dialog_outcome(Err(payload)),
                FileDialogOutcome::Failed("file dialog panicked: dialog panic".into())
            );
        }
    }

    #[test]
    fn cancellation_is_silent_but_failure_reaches_error_presenter_once() {
        let mut state = FileDialogState {
            worker: None,
            result: Some(FileDialogOutcome::Cancelled),
        };
        let mut actions = ActionQueue::default();
        let mut ui = super::super::UiState::default();
        poll_file_dialog(&mut actions, &mut state, &mut ui);
        assert!(actions.take_actions().is_empty());
        state.result = Some(FileDialogOutcome::Failed("runtime failed".into()));
        poll_file_dialog(&mut actions, &mut state, &mut ui);
        poll_file_dialog(&mut actions, &mut state, &mut ui);
        let mut orchestrator = crate::orchestrator::Orchestrator::default();
        let emitted = actions.take_actions();
        assert_eq!(emitted.len(), 1);
        orchestrator.process_action(&emitted[0]);
        let presentation = crate::error_presenter::present_error(
            orchestrator.last_error().unwrap(),
            UiLanguage::En,
        );
        assert_eq!(presentation.code, "FILE_DIALOG_FAILED");
        assert!(presentation.user_message.contains("runtime failed"));
    }
}
