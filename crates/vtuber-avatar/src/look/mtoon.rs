//! The MToon side of the look switch and the added portrait terms.
//!
//! [`MToonShadingMode`] selects the display path: `Native` is the fixed
//! upstream MToon display, `Rich` is the path that composes the look's effects
//! over the Native result. The switch and the effect amount are separate:
//! `enabled` picks the path (Rich even at strength 0), while `strength` is the
//! amount of added effect and is applied once, in the shader's
//! `compose_rich_mtoon`.

use bevy::prelude::*;
use bevy_vrm1::prelude::{
    MToonMaterial, MToonPortraitParams, MToonShadingMode, VrmMaterialBaseValues,
};

use crate::look::AvatarLookSettings;
use crate::look::preset::{MTOON_PORTRAIT_PRESET, RichLookSettings, effective_look_strength};

/// Resolves the MToon display path for the current look settings.
///
/// The switch alone decides the path: OFF is the fixed Native display and ON
/// is Rich even at strength 0, so the zero-effect case proves the Rich path's
/// identity against Native instead of falling back to the Native pipeline.
/// The mode must not be derived from the strength.
#[must_use]
pub fn resolve_mtoon_shading_mode(settings: RichLookSettings) -> MToonShadingMode {
    if settings.enabled {
        MToonShadingMode::Rich
    } else {
        MToonShadingMode::Native
    }
}

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
    let mode = resolve_mtoon_shading_mode(settings.0);
    let params = resolve_mtoon_portrait(settings.0);
    for handle in &meshes {
        let Some(mut material) = materials.get_mut(handle.id()) else {
            continue;
        };
        if material.shading_mode != mode {
            material.shading_mode = mode;
        }
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

    fn shading_mode(app: &App, handle: &Handle<MToonMaterial>) -> MToonShadingMode {
        app.world()
            .resource::<Assets<MToonMaterial>>()
            .get(handle)
            .expect("material")
            .shading_mode
    }

    #[test]
    fn default_material_has_no_extra_terms_and_stays_native() {
        assert_eq!(MToonMaterial::default().portrait.strength, 0.0);
        assert_eq!(
            MToonMaterial::default().portrait,
            MToonPortraitParams::default()
        );
        assert_eq!(
            MToonMaterial::default().shading_mode,
            MToonShadingMode::Native
        );
    }

    #[test]
    fn the_switch_selects_the_path_and_the_strength_only_asks_for_effects() {
        assert_eq!(
            resolve_mtoon_shading_mode(RichLookSettings {
                enabled: false,
                strength: 1.0,
            }),
            MToonShadingMode::Native
        );
        assert_eq!(
            resolve_mtoon_shading_mode(RichLookSettings {
                enabled: true,
                strength: 0.0,
            }),
            MToonShadingMode::Rich
        );
        assert_eq!(
            resolve_mtoon_shading_mode(RichLookSettings {
                enabled: true,
                strength: 1.0,
            }),
            MToonShadingMode::Rich
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
        assert_eq!(shading_mode(&app, &handle), MToonShadingMode::Native);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();
        assert_eq!(portrait(&app, &handle).strength, 1.0);
        assert_eq!(shading_mode(&app, &handle), MToonShadingMode::Rich);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 0.0,
        };
        app.update();
        assert_eq!(portrait(&app, &handle).strength, 0.0);
        assert_eq!(shading_mode(&app, &handle), MToonShadingMode::Rich);

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
        assert_eq!(shading_mode(&app, &handle), MToonShadingMode::Native);
    }
}
