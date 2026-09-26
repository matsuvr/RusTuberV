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
