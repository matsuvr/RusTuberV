//! The MToon side of the look switch and the added portrait terms.
//!
//! `MToonMaterial::portrait.strength` selects between the standard MToon
//! display (zero: the shading ramp folds the light into the authored base/shade
//! colors, exactly the plain display) and the rich look (above zero: each light
//! contributes its own color and radiance, the light rig applies, the
//! material's normal texture is evaluated, and a modest gloss, environment
//! reflection and rim are layered on top of the author's material).

use bevy::prelude::*;
use bevy_vrm1::prelude::{MToonMaterial, MToonPortraitParams, VrmMaterialBaseValues};

use crate::look::AvatarLookSettings;
use crate::look::preset::{MTOON_PORTRAIT_PRESET, RichLookSettings, effective_look_strength};

/// Resolves the extra portrait values for the current look settings.
///
/// The gains are the preset's nominal values; `strength` is the only value the
/// user can change and is applied once, in the shader.
#[must_use]
pub fn resolve_mtoon_portrait(settings: RichLookSettings) -> MToonPortraitParams {
    MToonPortraitParams {
        strength: effective_look_strength(settings),
        ..MTOON_PORTRAIT_PRESET
    }
}

/// Mirrors the resolved portrait values onto the avatar's MToon materials.
pub fn apply_mtoon_portrait_settings(
    settings: Res<AvatarLookSettings>,
    mut materials: ResMut<Assets<MToonMaterial>>,
    meshes: Query<&MeshMaterial3d<MToonMaterial>, With<VrmMaterialBaseValues>>,
) {
    let params = resolve_mtoon_portrait(settings.0);
    for handle in &meshes {
        let Some(mut material) = materials.get_mut(handle.id()) else {
            continue;
        };
        if material.portrait != params {
            material.portrait = params;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with_material() -> (App, Handle<MToonMaterial>) {
        let mut app = App::new();
        app.init_resource::<Assets<MToonMaterial>>()
            .init_resource::<AvatarLookSettings>()
            .add_systems(Update, apply_mtoon_portrait_settings);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<MToonMaterial>>()
            .add(MToonMaterial::default());
        app.world_mut().spawn((
            MeshMaterial3d(handle.clone()),
            VrmMaterialBaseValues::from_mtoon(&MToonMaterial::default()),
        ));
        (app, handle)
    }

    fn portrait(app: &App, handle: &Handle<MToonMaterial>) -> MToonPortraitParams {
        app.world()
            .resource::<Assets<MToonMaterial>>()
            .get(handle)
            .expect("material")
            .portrait
    }

    #[test]
    fn default_material_has_no_extra_terms() {
        assert_eq!(MToonMaterial::default().portrait.strength, 0.0);
        assert_eq!(
            MToonMaterial::default().portrait,
            MToonPortraitParams::default()
        );
    }

    #[test]
    fn strength_comes_from_the_switch_and_the_other_gains_are_nominal() {
        let on = resolve_mtoon_portrait(RichLookSettings {
            enabled: true,
            strength: 0.5,
        });
        assert_eq!(on.strength, 0.5);
        assert_eq!(on.specular_gain, MTOON_PORTRAIT_PRESET.specular_gain);
        assert_eq!(on.rim_power, MTOON_PORTRAIT_PRESET.rim_power);

        let off = resolve_mtoon_portrait(RichLookSettings {
            enabled: false,
            strength: 1.0,
        });
        assert_eq!(off.strength, 0.0);
        assert_eq!(
            off,
            MToonPortraitParams {
                strength: 0.0,
                ..MTOON_PORTRAIT_PRESET
            }
        );
    }

    #[test]
    fn the_switch_and_the_strength_reach_the_material() {
        let (mut app, handle) = app_with_material();
        app.update();
        assert_eq!(portrait(&app, &handle).strength, 0.0);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();
        assert_eq!(portrait(&app, &handle).strength, 1.0);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 0.0,
        };
        app.update();
        assert_eq!(portrait(&app, &handle).strength, 0.0);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: false,
            strength: 1.0,
        };
        app.update();
        assert_eq!(
            portrait(&app, &handle),
            resolve_mtoon_portrait(RichLookSettings {
                enabled: false,
                strength: 1.0
            })
        );
        assert_eq!(portrait(&app, &handle).strength, 0.0);
    }
}
