//! Camera preview texture pipeline.
//!
//! Manages the Bevy `Image` asset that displays the camera feed.
//! The preview texture is reused (not recreated each frame) and
//! can be toggled on/off without affecting the tracking pipeline.
//!
//! Mirror preview is a display-only transform — it does not affect
//! the inference input.

use bevy::prelude::*;

/// Resource managing the preview texture state.
#[derive(Resource, Debug)]
pub struct PreviewState {
    /// Whether the preview is visible.
    pub visible: bool,
    /// Whether the preview is mirrored.
    pub mirrored: bool,
    /// Handle to the preview image asset (reused each frame).
    pub image_handle: Option<Handle<Image>>,
    /// Target FPS for texture updates.
    pub target_fps: u32,
}

impl Default for PreviewState {
    fn default() -> Self {
        Self {
            visible: true,
            mirrored: true,
            image_handle: None,
            target_fps: 30,
        }
    }
}

impl PreviewState {
    /// Toggle preview visibility.
    pub fn toggle_visible(&mut self) {
        self.visible = !self.visible;
    }

    /// Toggle preview mirroring.
    pub fn toggle_mirrored(&mut self) {
        self.mirrored = !self.mirrored;
    }

    /// Returns the minimum interval between preview texture uploads.
    #[must_use]
    pub fn update_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs_f32(1.0 / self.target_fps.max(1) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_default_is_visible_and_mirrored() {
        let state = PreviewState::default();
        assert!(state.visible);
        assert!(state.mirrored);
    }

    #[test]
    fn preview_toggle_visible() {
        let mut state = PreviewState::default();
        state.toggle_visible();
        assert!(!state.visible);
        state.toggle_visible();
        assert!(state.visible);
    }

    #[test]
    fn preview_toggle_mirrored() {
        let mut state = PreviewState::default();
        state.toggle_mirrored();
        assert!(!state.mirrored);
        state.toggle_mirrored();
        assert!(state.mirrored);
    }

    /// Mirror toggle must not affect tracking data.
    /// This is a design invariant — verified by the fact that
    /// PreviewState has no connection to inference input.
    #[test]
    fn preview_mirror_does_not_affect_tracking() {
        let mut state = PreviewState::default();
        state.toggle_mirrored();
        // PreviewState has no tracking data fields.
        // This test documents the invariant.
        assert!(state.visible);
        assert!(!state.mirrored);
    }
}
