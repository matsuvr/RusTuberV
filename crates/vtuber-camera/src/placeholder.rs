//! Placeholder module for the camera subsystem.

/// Returns a short status string until the camera subsystem is implemented.
#[must_use]
pub fn status() -> &'static str {
    "vtuber-camera placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_status() {
        assert_eq!(status(), "vtuber-camera placeholder");
    }
}
