//! Blink capability fallback mapping.
//!
//! Converts raw blink weights into expression preset names based on the
//! model's available capabilities:
//!
//! - Per-eye model (blinkLeft + blinkRight): separate weights for each eye
//! - Combined model (blink only): single weight applied to "blink"
//! - No blink capability: empty output

use crate::capabilities::BlinkMode;

/// Raw blink input from tracking.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawBlinkInput {
    /// Left eye blink weight [0, 1]. Used for per-eye mode.
    pub left: f32,
    /// Right eye blink weight [0, 1]. Used for per-eye mode.
    pub right: f32,
    /// Combined blink weight [0, 1]. Used for combined mode.
    pub combined: f32,
}

/// Map raw blink input to expression preset weights based on capability.
///
/// Returns a list of (expression_name, weight) pairs.
///
/// # Behavior
///
/// - `BlinkMode::PerEye`: outputs ("blinkLeft", left) and ("blinkRight", right)
/// - `BlinkMode::Combined`: outputs ("blink", combined)
/// - `BlinkMode::None`: outputs nothing
///
/// Weights are NOT clamped here — the expression command builder handles that.
#[must_use]
pub fn map_blink_to_expressions(input: &RawBlinkInput, mode: BlinkMode) -> Vec<(String, f32)> {
    match mode {
        BlinkMode::PerEye => {
            vec![
                ("blinkLeft".to_string(), input.left),
                ("blinkRight".to_string(), input.right),
            ]
        }
        BlinkMode::Combined => {
            vec![("blink".to_string(), input.combined)]
        }
        BlinkMode::None => Vec::new(),
    }
}

/// Map raw blink input using per-eye values with combined fallback.
///
/// If the model supports per-eye blink, uses left/right separately.
/// If only combined blink is available, uses the max of left/right as combined.
/// If no blink is available, returns empty.
#[must_use]
pub fn map_blink_with_fallback(input: &RawBlinkInput, mode: BlinkMode) -> Vec<(String, f32)> {
    match mode {
        BlinkMode::PerEye => map_blink_to_expressions(input, mode),
        BlinkMode::Combined => {
            // Use the max of left/right as the combined value if combined is not set.
            let combined = if input.combined > 0.0 {
                input.combined
            } else {
                input.left.max(input.right)
            };
            vec![("blink".to_string(), combined)]
        }
        BlinkMode::None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blink_mapping_per_eye_asymmetric() {
        let input = RawBlinkInput {
            left: 0.8,
            right: 0.2,
            combined: 0.0,
        };
        let result = map_blink_to_expressions(&input, BlinkMode::PerEye);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("blinkLeft".to_string(), 0.8));
        assert_eq!(result[1], ("blinkRight".to_string(), 0.2));
    }

    #[test]
    fn blink_mapping_per_eye_symmetric() {
        let input = RawBlinkInput {
            left: 0.5,
            right: 0.5,
            combined: 0.0,
        };
        let result = map_blink_to_expressions(&input, BlinkMode::PerEye);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, 0.5);
        assert_eq!(result[1].1, 0.5);
    }

    #[test]
    fn blink_mapping_combined_only() {
        let input = RawBlinkInput {
            left: 0.0,
            right: 0.0,
            combined: 0.7,
        };
        let result = map_blink_to_expressions(&input, BlinkMode::Combined);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("blink".to_string(), 0.7));
    }

    #[test]
    fn blink_mapping_none_produces_empty() {
        let input = RawBlinkInput {
            left: 0.5,
            right: 0.5,
            combined: 0.5,
        };
        let result = map_blink_to_expressions(&input, BlinkMode::None);

        assert!(result.is_empty());
    }

    #[test]
    fn blink_mapping_combined_fallback_uses_max() {
        let input = RawBlinkInput {
            left: 0.8,
            right: 0.3,
            combined: 0.0, // not set
        };
        let result = map_blink_with_fallback(&input, BlinkMode::Combined);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "blink");
        assert_eq!(result[0].1, 0.8, "should use max of left/right");
    }

    #[test]
    fn blink_mapping_combined_prefers_explicit_value() {
        let input = RawBlinkInput {
            left: 0.8,
            right: 0.3,
            combined: 0.5, // explicitly set
        };
        let result = map_blink_with_fallback(&input, BlinkMode::Combined);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 0.5, "should prefer explicit combined value");
    }

    #[test]
    fn blink_mapping_no_preset_no_panic() {
        let input = RawBlinkInput::default();
        let result = map_blink_to_expressions(&input, BlinkMode::None);
        assert!(result.is_empty());

        let result = map_blink_with_fallback(&input, BlinkMode::None);
        assert!(result.is_empty());
    }

    #[test]
    fn blink_mapping_zero_weights() {
        let input = RawBlinkInput {
            left: 0.0,
            right: 0.0,
            combined: 0.0,
        };
        let result = map_blink_to_expressions(&input, BlinkMode::PerEye);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, 0.0);
        assert_eq!(result[1].1, 0.0);
    }
}
