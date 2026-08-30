//! Mouth preset mapping with `aa` fallback.
//!
//! Converts raw mouth openness or vowel coefficients into VRM expression
//! preset names based on the model's available capabilities:
//!
//! - Full mouth (aa, ih, ou, ee, oh): use available vowel coefficients
//! - aa-only: assign mouth openness to "aa"
//! - No mouth presets: empty output

use crate::capabilities::MouthMode;

/// Raw mouth input from tracking.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawMouthInput {
    /// Overall mouth openness [0, 1]. Used for aa-only fallback.
    pub openness: f32,
    /// Vowel coefficients [0, 1] each. Used for full mouth mode.
    pub aa: f32,
    /// "ih" vowel coefficient [0, 1].
    pub ih: f32,
    /// "ou" vowel coefficient [0, 1].
    pub ou: f32,
    /// "ee" vowel coefficient [0, 1].
    pub ee: f32,
    /// "oh" vowel coefficient [0, 1].
    pub oh: f32,
}

/// Map raw mouth input to expression preset weights based on capability.
///
/// Returns a list of (expression_name, weight) pairs.
///
/// # Behavior
///
/// - `MouthMode::Full`: outputs available vowel presets with their coefficients
/// - `MouthMode::AaOnly`: outputs ("aa", openness)
/// - `MouthMode::None`: outputs nothing
///
/// Weights are NOT clamped here — the expression command builder handles that.
#[must_use]
pub fn map_mouth_to_expressions(input: &RawMouthInput, mode: MouthMode) -> Vec<(String, f32)> {
    match mode {
        MouthMode::Full => {
            let mut result = Vec::new();
            if input.aa > 0.0 {
                result.push(("aa".to_string(), input.aa));
            }
            if input.ih > 0.0 {
                result.push(("ih".to_string(), input.ih));
            }
            if input.ou > 0.0 {
                result.push(("ou".to_string(), input.ou));
            }
            if input.ee > 0.0 {
                result.push(("ee".to_string(), input.ee));
            }
            if input.oh > 0.0 {
                result.push(("oh".to_string(), input.oh));
            }
            result
        }
        MouthMode::AaOnly => {
            if input.openness > 0.0 {
                vec![("aa".to_string(), input.openness)]
            } else {
                Vec::new()
            }
        }
        MouthMode::None => Vec::new(),
    }
}

/// Map raw mouth input using openness as aa fallback.
///
/// If the model supports full mouth, uses vowel coefficients.
/// If only aa is available, uses openness as the aa weight.
/// If no mouth presets, returns empty.
#[must_use]
pub fn map_mouth_with_fallback(input: &RawMouthInput, mode: MouthMode) -> Vec<(String, f32)> {
    match mode {
        MouthMode::Full => map_mouth_to_expressions(input, mode),
        MouthMode::AaOnly => {
            // Use openness directly as aa weight.
            if input.openness > 0.0 {
                vec![("aa".to_string(), input.openness)]
            } else {
                // Fall back to max of vowel coefficients if openness is zero.
                let max_vowel = input
                    .aa
                    .max(input.ih)
                    .max(input.ou)
                    .max(input.ee)
                    .max(input.oh);
                if max_vowel > 0.0 {
                    vec![("aa".to_string(), max_vowel)]
                } else {
                    Vec::new()
                }
            }
        }
        MouthMode::None => Vec::new(),
    }
}

/// Check if a vowel name is a valid VRM mouth preset.
#[must_use]
pub fn is_valid_mouth_preset(name: &str) -> bool {
    matches!(name, "aa" | "ih" | "ou" | "ee" | "oh")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mouth_mapping_full_all_vowels() {
        let input = RawMouthInput {
            openness: 0.0,
            aa: 0.8,
            ih: 0.2,
            ou: 0.0,
            ee: 0.0,
            oh: 0.0,
        };
        let result = map_mouth_to_expressions(&input, MouthMode::Full);

        assert_eq!(result.len(), 2);
        assert!(result.iter().any(|(n, w)| n == "aa" && *w == 0.8));
        assert!(result.iter().any(|(n, w)| n == "ih" && *w == 0.2));
    }

    #[test]
    fn mouth_mapping_full_skips_zero_vowels() {
        let input = RawMouthInput {
            openness: 0.0,
            aa: 0.5,
            ih: 0.0,
            ou: 0.0,
            ee: 0.0,
            oh: 0.0,
        };
        let result = map_mouth_to_expressions(&input, MouthMode::Full);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "aa");
    }

    #[test]
    fn mouth_mapping_aa_only() {
        let input = RawMouthInput {
            openness: 0.7,
            aa: 0.0,
            ih: 0.0,
            ou: 0.0,
            ee: 0.0,
            oh: 0.0,
        };
        let result = map_mouth_to_expressions(&input, MouthMode::AaOnly);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("aa".to_string(), 0.7));
    }

    #[test]
    fn mouth_mapping_none_produces_empty() {
        let input = RawMouthInput {
            openness: 0.5,
            aa: 0.5,
            ih: 0.5,
            ou: 0.5,
            ee: 0.5,
            oh: 0.5,
        };
        let result = map_mouth_to_expressions(&input, MouthMode::None);

        assert!(result.is_empty());
    }

    #[test]
    fn mouth_mapping_aa_fallback_uses_openness() {
        let input = RawMouthInput {
            openness: 0.6,
            aa: 0.0,
            ih: 0.0,
            ou: 0.0,
            ee: 0.0,
            oh: 0.0,
        };
        let result = map_mouth_with_fallback(&input, MouthMode::AaOnly);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("aa".to_string(), 0.6));
    }

    #[test]
    fn mouth_mapping_aa_fallback_uses_max_vowel_when_openness_zero() {
        let input = RawMouthInput {
            openness: 0.0,
            aa: 0.3,
            ih: 0.5,
            ou: 0.2,
            ee: 0.0,
            oh: 0.0,
        };
        let result = map_mouth_with_fallback(&input, MouthMode::AaOnly);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "aa");
        assert_eq!(result[0].1, 0.5, "should use max vowel coefficient");
    }

    #[test]
    fn mouth_mapping_no_preset_no_panic() {
        let input = RawMouthInput::default();
        let result = map_mouth_to_expressions(&input, MouthMode::None);
        assert!(result.is_empty());

        let result = map_mouth_with_fallback(&input, MouthMode::None);
        assert!(result.is_empty());
    }

    #[test]
    fn mouth_mapping_does_not_send_unsupported_vowels() {
        // Even if input has non-zero values for all vowels,
        // MouthMode::AaOnly should only output "aa".
        let input = RawMouthInput {
            openness: 0.0,
            aa: 0.3,
            ih: 0.5,
            ou: 0.2,
            ee: 0.4,
            oh: 0.1,
        };
        let result = map_mouth_with_fallback(&input, MouthMode::AaOnly);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "aa");
        // ih, ou, ee, oh should NOT appear.
    }

    #[test]
    fn mouth_mapping_valid_preset_check() {
        assert!(is_valid_mouth_preset("aa"));
        assert!(is_valid_mouth_preset("ih"));
        assert!(is_valid_mouth_preset("ou"));
        assert!(is_valid_mouth_preset("ee"));
        assert!(is_valid_mouth_preset("oh"));
        assert!(!is_valid_mouth_preset("blink"));
        assert!(!is_valid_mouth_preset("happy"));
        assert!(!is_valid_mouth_preset(""));
    }
}
