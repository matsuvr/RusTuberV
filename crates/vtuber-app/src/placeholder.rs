//! Placeholder module for the app subsystem.

/// Returns a short status string until the app subsystem is implemented.
#[must_use]
pub fn status() -> &'static str {
    "vtuber-app placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_status() {
        assert_eq!(status(), "vtuber-app placeholder");
    }
}
