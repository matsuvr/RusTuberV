//! User-facing mirror policy for avatar control.
//!
//! Tracking keeps its canonical, unmirrored camera-coordinate contract. This
//! adapter-local resource selects how that contract is presented by the avatar.

use bevy::prelude::*;

/// Controls whether avatar motion is reflected for the person operating it.
///
/// The default is enabled: horizontal head and eye motion, head roll, and
/// side-specific blink expressions are reflected at the avatar boundary.
/// Pitch and non-directional expressions remain unchanged. This never changes
/// camera frames, inference input, calibration, or tracking values.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct AvatarMotionMirror {
    enabled: bool,
}

impl Default for AvatarMotionMirror {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl AvatarMotionMirror {
    /// Returns whether reflected avatar motion is enabled.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        self.enabled
    }

    /// Toggles reflected avatar motion.
    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avatar_motion_mirror_is_enabled_by_default_and_can_be_disabled() {
        let mut mirror = AvatarMotionMirror::default();
        assert!(mirror.is_enabled());

        mirror.toggle();
        assert!(!mirror.is_enabled());
    }
}
