//! Validated look switch/strength shared by the UI, persistence and rendering.
//!
//! OFF selects the baseline MToon/Standard/unlit materials and front light.
//! The baseline MToon includes the approved transparent-BLEND discard correction.
//! ON keeps that front light and selects application Rich MToon/Standard paths,
//! adding key/rim lighting. Unlit materials remain unlit.
//! Strength scales the additional contribution, not the baseline light;
//! at zero the additional lights and effects contribute nothing. Switching OFF
//! preserves the stored strength for the next activation.

use std::fmt;

use serde::{Deserialize, Serialize};

/// The single look switch and strength shared by the settings UI.
///
/// Strength is always finite and in `0..=1`. The initial state is OFF with
/// strength 1; switching OFF retains the strength selected by the user.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RichLookSettingsDto", into = "RichLookSettingsDto")]
pub struct RichLookSettings {
    enabled: bool,
    strength: f32,
}

impl RichLookSettings {
    /// Constructs settings without clamping or replacing an invalid strength.
    ///
    /// # Errors
    /// Returns [`RichLookSettingsError`] for a non-finite strength or a value
    /// outside `0..=1`, including when the switch is OFF.
    pub fn try_new(enabled: bool, strength: f32) -> Result<Self, RichLookSettingsError> {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(RichLookSettingsError { strength });
        }
        Ok(Self { enabled, strength })
    }

    /// Whether the Rich material and additional contribution are selected.
    #[must_use]
    pub const fn enabled(self) -> bool {
        self.enabled
    }

    /// The configured strength, also retained while the switch is OFF.
    #[must_use]
    pub const fn strength(self) -> f32 {
        self.strength
    }

    /// Changes the switch without changing the configured strength.
    #[must_use]
    pub const fn with_enabled(self, enabled: bool) -> Self {
        Self { enabled, ..self }
    }

    /// Changes strength without changing the switch.
    ///
    /// # Errors
    /// Returns [`RichLookSettingsError`] for a non-finite or out-of-range
    /// strength. The original value is unchanged.
    pub fn with_strength(self, strength: f32) -> Result<Self, RichLookSettingsError> {
        Self::try_new(self.enabled, strength)
    }
}

impl Default for RichLookSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            strength: 1.0,
        }
    }
}

/// An invalid strength supplied when constructing or editing look settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RichLookSettingsError {
    strength: f32,
}

impl fmt::Display for RichLookSettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "look strength must be finite and in 0..=1, got {}",
            self.strength
        )
    }
}

impl std::error::Error for RichLookSettingsError {}

/// Persistence representation; deserialization must pass through validation.
#[derive(Serialize, Deserialize)]
struct RichLookSettingsDto {
    enabled: bool,
    strength: f32,
}

impl TryFrom<RichLookSettingsDto> for RichLookSettings {
    type Error = RichLookSettingsError;

    fn try_from(value: RichLookSettingsDto) -> Result<Self, Self::Error> {
        Self::try_new(value.enabled, value.strength)
    }
}

impl From<RichLookSettings> for RichLookSettingsDto {
    fn from(value: RichLookSettings) -> Self {
        Self {
            enabled: value.enabled,
            strength: value.strength,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_are_off_at_full_strength() {
        let settings = RichLookSettings::default();
        assert!(!settings.enabled());
        assert_eq!(settings.strength(), 1.0);
    }

    #[test]
    fn boundary_and_intermediate_strengths_round_trip_with_the_same_keys() {
        for enabled in [false, true] {
            for strength in [0.0, 0.5, 1.0] {
                let settings = RichLookSettings::try_new(enabled, strength).unwrap();
                let text = toml::to_string(&settings).unwrap();
                assert_eq!(toml::from_str::<RichLookSettings>(&text).unwrap(), settings);
                let value: toml::Value = toml::from_str(&text).unwrap();
                let table = value.as_table().unwrap();
                assert_eq!(table.len(), 2);
                assert_eq!(
                    table.get("enabled").and_then(toml::Value::as_bool),
                    Some(enabled)
                );
                assert_eq!(
                    table.get("strength").and_then(toml::Value::as_float),
                    Some(f64::from(strength))
                );
            }
        }
    }

    #[test]
    fn invalid_strength_is_rejected_even_while_off() {
        for strength in [-0.1, 1.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for enabled in [false, true] {
                assert!(RichLookSettings::try_new(enabled, strength).is_err());
            }
            let current = RichLookSettings::try_new(true, 0.5).unwrap();
            assert!(current.with_strength(strength).is_err());
            assert_eq!(current.strength(), 0.5);
            assert!(current.enabled());
        }
    }

    #[test]
    fn deserialization_uses_the_same_validation() {
        for strength in ["-0.1", "1.1", "nan", "inf", "-inf"] {
            let text = format!("enabled = false\nstrength = {strength}\n");
            assert!(toml::from_str::<RichLookSettings>(&text).is_err(), "{text}");
        }
    }

    #[test]
    fn switching_off_preserves_strength_including_zero() {
        for strength in [0.0, 0.5, 1.0] {
            let settings = RichLookSettings::try_new(true, strength).unwrap();
            let off = settings.with_enabled(false);
            assert!(!off.enabled());
            assert_eq!(off.strength(), strength);
            assert_eq!(off.with_enabled(true), settings);
        }
    }

    #[test]
    fn validation_error_implements_the_standard_error_contract() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<RichLookSettingsError>();
        let error = RichLookSettings::try_new(true, 2.0).unwrap_err();
        assert!(error.to_string().contains("0..=1"));
    }
}
