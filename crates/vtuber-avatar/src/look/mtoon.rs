//! The MToon side of the look switch.
//!
//! `MToonMaterial::look_strength` selects between the standard MToon display
//! (zero: the shading ramp folds the light into the authored base/shade colors,
//! exactly the plain display) and the rich look (above zero: each light
//! contributes its own color and radiance, the light rig applies, and the
//! material's normal texture is evaluated).

use bevy::prelude::*;
use bevy_vrm1::prelude::{MToonMaterial, VrmMaterialBaseValues};

use crate::look::AvatarLookSettings;
use crate::look::preset::effective_look_strength;

/// Mirrors the effective look strength onto the avatar's MToon materials.
pub fn apply_mtoon_look_strength(
    settings: Res<AvatarLookSettings>,
    mut materials: ResMut<Assets<MToonMaterial>>,
    meshes: Query<&MeshMaterial3d<MToonMaterial>, With<VrmMaterialBaseValues>>,
) {
    let strength = effective_look_strength(settings.0);
    for handle in &meshes {
        let Some(mut material) = materials.get_mut(handle.id()) else {
            continue;
        };
        if material.look_strength != strength {
            material.look_strength = strength;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::look::RichLookSettings;

    fn app_with_material() -> (App, Handle<MToonMaterial>) {
        let mut app = App::new();
        app.init_resource::<Assets<MToonMaterial>>()
            .init_resource::<AvatarLookSettings>()
            .add_systems(Update, apply_mtoon_look_strength);
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

    fn strength(app: &App, handle: &Handle<MToonMaterial>) -> f32 {
        app.world()
            .resource::<Assets<MToonMaterial>>()
            .get(handle)
            .expect("material")
            .look_strength
    }

    #[test]
    fn default_material_is_the_standard_display() {
        assert_eq!(MToonMaterial::default().look_strength, 0.0);
    }

    #[test]
    fn the_switch_and_the_strength_reach_the_material() {
        let (mut app, handle) = app_with_material();
        app.update();
        assert_eq!(strength(&app, &handle), 0.0);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();
        assert_eq!(strength(&app, &handle), 1.0);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 0.0,
        };
        app.update();
        assert_eq!(strength(&app, &handle), 0.0);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: false,
            strength: 1.0,
        };
        app.update();
        assert_eq!(strength(&app, &handle), 0.0);
    }
}
