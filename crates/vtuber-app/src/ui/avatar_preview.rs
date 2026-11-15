//! The Bevy output image registered with egui for the studio preview.
use bevy_egui::egui;
use vtuber_core::VideoOutputProfile;

/// Texture registration and dimensions for the avatar monitor and full-screen preview.
#[derive(Clone, Copy)]
pub struct AvatarPreviewTexture {
    id: egui::TextureId,
    /// Profile used to preserve the camera aspect ratio.
    pub profile: VideoOutputProfile,
}

impl AvatarPreviewTexture {
    /// Associates the registered image with its output dimensions.
    pub const fn new(id: egui::TextureId, profile: VideoOutputProfile) -> Self {
        Self { id, profile }
    }

    pub(super) fn widget(self, size: egui::Vec2, radius: u8) -> egui::Image<'static> {
        egui::Image::new((self.id, size)).corner_radius(radius)
    }
}
