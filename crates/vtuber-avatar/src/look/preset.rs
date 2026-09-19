//! Look settings and the numeric interpolation shared by every look stage.

use bevy::prelude::{LinearRgba, Vec3};
use bevy_vrm1::prelude::MToonPortraitParams;
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

/// One studio light of the portrait preset, expressed in camera space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudioLightPreset {
    /// The direction the light travels, in camera space (the camera looks down
    /// its local `-Z`).
    pub direction: Vec3,
    /// The light color.
    pub color: LinearRgba,
    /// The illuminance in lux.
    pub illuminance: f32,
    /// Whether this light casts shadows.
    pub shadows_enabled: bool,
}

/// The portrait lighting preset: one key with shadows, a weak fill and a weak
/// rim, plus a small studio environment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudioPreset {
    /// The shadow-casting key light.
    pub key: StudioLightPreset,
    /// The weak fill light that keeps the shaded side readable.
    pub fill: StudioLightPreset,
    /// The weak rim light that separates the silhouette.
    pub rim: StudioLightPreset,
    /// The environment light intensity in cd/m².
    pub environment_intensity: f32,
}

/// The first version's only preset. These are adjustment starting points, not
/// measured optima: key about 40° to the camera's left and 30° up, fill from
/// the opposite side, rim from behind.
///
/// The key is intentionally dimmer than the scene's own 1500 lx light, so
/// switching the look on is a *portrait* rig (softer key plus fill and rim)
/// rather than more light on the same direction. That is also what the
/// strength slider interpolates: at 0 it is the scene's original light, at 1
/// the preset below.
pub const STUDIO_PRESET: StudioPreset = StudioPreset {
    key: StudioLightPreset {
        direction: Vec3::new(0.45, -0.40, -0.80),
        color: LinearRgba::new(1.0, 0.97, 0.93, 1.0),
        illuminance: 550.0,
        shadows_enabled: true,
    },
    fill: StudioLightPreset {
        direction: Vec3::new(-0.55, -0.15, -0.82),
        color: LinearRgba::new(0.85, 0.90, 1.0, 1.0),
        illuminance: 180.0,
        shadows_enabled: false,
    },
    rim: StudioLightPreset {
        direction: Vec3::new(-0.20, -0.35, 0.90),
        color: LinearRgba::new(1.0, 1.0, 1.0, 1.0),
        illuminance: 260.0,
        shadows_enabled: false,
    },
    environment_intensity: 160.0,
};

/// The added portrait terms' nominal values.
///
/// These are adjustment starting points, not measured optima: a modest
/// non-metal gloss, a moderate environment reflection and a weak rim. The
/// look's strength scales all of them at once in the shader.
pub const MTOON_PORTRAIT_PRESET: MToonPortraitParams = MToonPortraitParams {
    strength: 0.0,
    specular_gain: 1.20,
    perceptual_roughness: 0.45,
    environment_gain: 1.00,
    rim_gain: 0.60,
    rim_power: 3.0,
};

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



