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
    MToonMaterial, MToonPortraitParams, MToonShadingMode, VrmMaterialBaseValues, VrmMaterialIndex,
};

use crate::binding::AvatarBinding;
use crate::look::AvatarLookSettings;
use crate::look::material::{AvatarMaterialRoles, MaterialRole};
use crate::look::preset::{RichLookSettings, resolve_mtoon_role_params};

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
    resolve_mtoon_role_params(settings, MaterialRole::General)
}

/// The world unit vector of the rendered head's face front.
///
/// The head bone's own `+Z` axis is the canonical front shared by the
/// normalization and the LookAt convention; reading the rendered
/// [`GlobalTransform`] keeps the steering on the same head pose the frame is
/// drawn with.
#[must_use]
pub fn head_world_forward(head: &GlobalTransform) -> Vec3 {
    head.rotation() * Vec3::Z
}

/// Mirrors the resolved role values onto the avatar's MToon materials.
///
/// Each material receives the gains of its effective role. `Face` materials
/// additionally carry the rendered head's forward and a steering amount that
/// scales with the look strength, so at strength 0 the authored normal is kept
/// exactly.
#[allow(clippy::type_complexity)]
pub fn apply_mtoon_portrait_settings(
    settings: Res<AvatarLookSettings>,
    roles: Res<AvatarMaterialRoles>,
    lifecycle: Res<crate::lifecycle::AvatarLifecycle>,
    bindings: Query<&AvatarBinding>,
    transforms: Query<&GlobalTransform>,
    mut materials: ResMut<Assets<MToonMaterial>>,
    meshes: Query<(&MeshMaterial3d<MToonMaterial>, &VrmMaterialIndex), With<VrmMaterialBaseValues>>,
) {
    let mode = resolve_mtoon_shading_mode(settings.0);
    let face_forward = lifecycle
        .active_root()
        .and_then(|root| bindings.get(root).ok())
        .and_then(|binding| transforms.get(binding.head).ok())
        .map(head_world_forward);
    for (handle, index) in &meshes {
        let role = roles.role(index.0);
        let mut params = resolve_mtoon_role_params(settings.0, role);
        match face_forward {
            Some(forward) => params.face_forward = forward,
            // Without the rendered head pose there is no face forward to steer
            // toward, so the authored normal is kept.
            None => params.face_normal_amount = 0.0,
        }
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
    use crate::lifecycle::AvatarLifecycle;
    use crate::look::MaterialRoleOverride;
    use crate::look::preset::{MTOON_PORTRAIT_PRESET, face_lighting_normal};

    fn app_with_material() -> (App, Handle<MToonMaterial>) {
        let mut app = App::new();
        app.init_resource::<Assets<MToonMaterial>>()
            .init_resource::<AvatarLookSettings>()
            .init_resource::<AvatarMaterialRoles>()
            .init_resource::<AvatarLifecycle>()
            .add_systems(Update, apply_mtoon_portrait_settings);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<MToonMaterial>>()
            .add(MToonMaterial::default());
        app.world_mut().spawn((
            MeshMaterial3d(handle.clone()),
            VrmMaterialBaseValues::from_mtoon(&MToonMaterial::default()),
            VrmMaterialIndex(0),
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

    #[test]
    fn role_gains_differ_and_face_carries_the_normal_amount() {
        let on = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        let general = resolve_mtoon_role_params(on, MaterialRole::General);
        assert_eq!(general.specular_gain, MTOON_PORTRAIT_PRESET.specular_gain);
        assert_eq!(general.face_normal_amount, 0.0);
        let face = resolve_mtoon_role_params(on, MaterialRole::Face);
        assert!(face.specular_gain < general.specular_gain);
        assert_eq!(face.face_normal_amount, 1.0);
        let skin = resolve_mtoon_role_params(on, MaterialRole::Skin);
        assert!(skin.specular_gain < face.specular_gain);
        let hair = resolve_mtoon_role_params(on, MaterialRole::Hair);
        assert!(hair.specular_gain > general.specular_gain);
        for role in [MaterialRole::Fabric, MaterialRole::Metal, MaterialRole::Eye] {
            assert_eq!(
                resolve_mtoon_role_params(on, role).face_normal_amount,
                0.0,
                "role {role:?} never steers the diffuse normal"
            );
        }
        let zero = resolve_mtoon_role_params(
            RichLookSettings {
                enabled: true,
                strength: 0.0,
            },
            MaterialRole::Face,
        );
        assert_eq!(zero.strength, 0.0);
        assert_eq!(zero.face_normal_amount, 0.0);
    }

    #[test]
    fn face_lighting_normal_matches_the_wgsl_formula() {
        let mesh = Vec3::new(0.0, 0.0, 1.0).normalize();
        let forward = Vec3::X;
        assert_eq!(
            face_lighting_normal(mesh, forward, 0.0),
            mesh,
            "zero amount keeps the mesh normal exactly"
        );
        let full = face_lighting_normal(mesh, forward, 1.0);
        let expected = mesh.lerp(forward, 0.25).normalize();
        assert!(full.distance(expected) < 1e-6);
        let half = face_lighting_normal(mesh, forward, 0.5);
        assert!(half.distance(mesh.lerp(forward, 0.125).normalize()) < 1e-6);
        // The steering stays strictly between the mesh normal and the forward.
        assert!(full.dot(mesh) < 1.0 && full.dot(forward) > 0.0);
    }

    #[test]
    fn the_head_pose_drives_the_face_normal_and_the_strength_gates_it() {
        let (mut app, handle) = app_with_material();
        app.world_mut()
            .resource_mut::<AvatarMaterialRoles>()
            .replace_overrides(
                [MaterialRoleOverride {
                    material_index: 0,
                    selected: Some(MaterialRole::Face),
                }]
                .into_iter(),
            );
        let head = app
            .world_mut()
            .spawn(GlobalTransform::from_rotation(Quat::from_rotation_y(
                std::f32::consts::FRAC_PI_2,
            )))
            .id();
        let root = app.world_mut().spawn_empty().id();
        let generation = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .unwrap();
        app.world_mut()
            .entity_mut(root)
            .insert(AvatarBinding::head_only(root, head, generation));
        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();
        let resolved = portrait(&app, &handle);
        assert_eq!(resolved.face_normal_amount, 1.0);
        let expected = head_world_forward(app.world().get::<GlobalTransform>(head).unwrap());
        assert!(resolved.face_forward.distance(expected) < 1e-6);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: false,
            strength: 1.0,
        };
        app.update();
        let resolved = portrait(&app, &handle);
        assert_eq!(resolved.face_normal_amount, 0.0);
        assert_eq!(resolved.strength, 0.0);
    }
}
