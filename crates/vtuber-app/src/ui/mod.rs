//! UI rendering module using `bevy_egui`.
//!
//! The UI reads [`crate::ui_model::UiViewModel`] snapshots and emits
//! [`crate::actions::UiAction`] commands. It never accesses domain
//! services directly.

pub mod diagnostics;
pub mod error;
pub mod file_dialog;
pub mod panes;
pub mod shell;
pub(crate) mod widgets;

pub use shell::{UiShellPlugin, UiState};
