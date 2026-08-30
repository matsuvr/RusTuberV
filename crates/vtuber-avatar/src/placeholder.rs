//! Placeholder module for the avatar subsystem.

/// Returns a short status string until the avatar subsystem is implemented.
#[must_use]
pub fn status() -> &'static str {
    "vtuber-avatar placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_status() {
        assert_eq!(status(), "vtuber-avatar placeholder");
    }

    #[test]
    fn plugin_default() {
        let _ = crate::VtuberAvatarPlugin;
    }
}
