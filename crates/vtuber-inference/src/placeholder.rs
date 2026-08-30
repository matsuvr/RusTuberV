//! Placeholder module for the inference subsystem.

/// Returns a short status string until the inference subsystem is implemented.
#[must_use]
pub fn status() -> &'static str {
    "vtuber-inference placeholder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_status() {
        assert_eq!(status(), "vtuber-inference placeholder");
    }
}
