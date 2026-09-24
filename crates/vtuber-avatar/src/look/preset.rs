//! Look switch/strength shared UI state.
//!
//! This module keeps only the switch and strength the settings UI edits and
//! persists. `#93` drives the additional lights from these values; the
//! app-side Rich shaders follow in later issues.

use serde::{Deserialize, Serialize};

/// The single look switch and strength shared by the settings UI.
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

#[cfg(test)]
mod tests {
    use super::*;

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
