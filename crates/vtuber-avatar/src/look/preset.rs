//! Look settings and the numeric interpolation shared by every look stage.

use serde::{Deserialize, Serialize};

/// The single look switch and strength shared by lighting, materials and the
/// finish.
///
/// The initial state is OFF with strength 1, matching the one-click UI
/// contract: switching the look ON starts at full strength, and a strength the
/// user adjusted is kept while the look is OFF.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RichLookSettings {
    /// Whether the rich look is applied at all.
    pub enabled: bool,
    /// The user-facing strength in `0..=1`.
    pub strength: f32,
}

impl Default for RichLookSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            strength: 1.0,
        }
    }
}

/// The strength every look stage must use: zero while the look is OFF.
#[must_use]
pub fn effective_look_strength(settings: RichLookSettings) -> f32 {
    if settings.enabled { settings.strength } else { 0.0 }
}

/// Interpolates between the original and rich value for one scalar.
///
/// `strength == 0` returns `original` and `strength == 1` returns `rich`
/// unchanged, so the endpoints keep their exact meaning.
#[must_use]
pub fn blend_look_scalar(original: f32, rich: f32, strength: f32) -> f32 {
    if strength == 0.0 {
        return original;
    }
    if strength == 1.0 {
        return rich;
    }
    original + (rich - original) * strength
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strength_is_zero_while_disabled_and_kept_while_enabled() {
        assert_eq!(
            effective_look_strength(RichLookSettings {
                enabled: false,
                strength: 1.0
            }),
            0.0
        );
        assert_eq!(
            effective_look_strength(RichLookSettings {
                enabled: true,
                strength: 0.25
            }),
            0.25
        );
    }

    #[test]
    fn blend_endpoints_are_exact() {
        assert_eq!(blend_look_scalar(0.4, 0.1, 0.0), 0.4);
        assert_eq!(blend_look_scalar(0.4, 0.1, 1.0), 0.1);
        assert_eq!(blend_look_scalar(0.4, 0.2, 0.5), 0.3);
    }

    #[test]
    fn default_settings_are_off_at_full_strength() {
        assert_eq!(
            RichLookSettings::default(),
            RichLookSettings {
                enabled: false,
                strength: 1.0
            }
        );
    }
}
