//! Placeholder module for the tracking subsystem.

/// Returns a short status string until the tracking subsystem is implemented.
#[must_use]
pub fn status() -> &'static str {
    "vtuber-tracking placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_status() {
        assert_eq!(status(), "vtuber-tracking placeholder");
    }
}
