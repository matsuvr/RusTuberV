//! Captured base values of the `StandardMaterial` fields the look owns.
//!
//! The look only changes four scalar fields of a `StandardMaterial`
//! (`perceptual_roughness`, `reflectance`, `clearcoat` and
//! `clearcoat_perceptual_roughness`). Everything else (base color, emission,
//! UV transform, metallic and the texture handles) stays owned by the author
//! data and the existing expression writer.

use std::collections::HashMap;

use bevy::asset::AssetId;
use bevy::prelude::*;
use bevy_vrm1::prelude::VrmMaterialBaseValues;

use crate::lifecycle::{AvatarLifecycle, AvatarLifecycleState};
use crate::look::AvatarLookSettings;
use crate::look::preset::{RichLookSettings, effective_look_strength};

/// The relative roughness the rich look starts from. This is an adjustment
/// starting point, not a measured optimum: it keeps the author's roughness
/// texture variation while making the surface react a little more to the
/// studio lights.
const RICH_ROUGHNESS_SCALE: f32 = 0.95;

/// Resolves the look-owned `StandardMaterial` values for one material.
///
/// The look keeps the author reflectance, clearcoat and clearcoat roughness in
/// the general preset, so only the perceptual roughness moves. Unlit materials
/// and a zero strength return the captured values unchanged.
#[must_use]
pub fn resolve_standard_portrait(
    original: StandardLookBase,
    settings: RichLookSettings,
    unlit: bool,
) -> StandardLookBase {
    let strength = effective_look_strength(settings);
    if unlit || strength == 0.0 {
        return original;
    }
    StandardLookBase {
        perceptual_roughness: original.perceptual_roughness * RICH_ROUGHNESS_SCALE,
        ..original
    }
}

/// Writes the look-owned values back into the material.
fn write_standard_look(material: &mut StandardMaterial, values: StandardLookBase) {
    if material.perceptual_roughness != values.perceptual_roughness {
        material.perceptual_roughness = values.perceptual_roughness;
    }
    if material.reflectance != values.reflectance {
        material.reflectance = values.reflectance;
    }
    if material.clearcoat != values.clearcoat {
        material.clearcoat = values.clearcoat;
    }
    if material.clearcoat_perceptual_roughness != values.clearcoat_perceptual_roughness {
        material.clearcoat_perceptual_roughness = values.clearcoat_perceptual_roughness;
    }
}

/// Applies the resolved values to the avatar's `StandardMaterial` assets.
///
/// The material type, its handle, the mesh's `MeshMaterial3d` and every other
/// field (base color, emission, UV, metallic, alpha and all texture handles)
/// are left alone; only the four captured scalars are touched, and only for
/// materials that were captured for this avatar.
pub fn apply_standard_portrait_settings(
    settings: Res<AvatarLookSettings>,
    bases: Res<StandardLookBases>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    meshes: Query<&MeshMaterial3d<StandardMaterial>, With<VrmMaterialBaseValues>>,
) {
    for handle in &meshes {
        let Some(original) = bases.get(handle.id()) else {
            continue;
        };
        let Some(material) = materials.get(handle.id()) else {
            continue;
        };
        let values = resolve_standard_portrait(original, settings.0, material.unlit);
        if capture_standard_look_base(material) == values {
            continue;
        }
        let Some(mut material) = materials.get_mut(handle.id()) else {
            continue;
        };
        write_standard_look(&mut material, values);
    }
}

/// Snapshot of the four `StandardMaterial` fields the look may rewrite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StandardLookBase {
    /// Author perceptual roughness.
    pub perceptual_roughness: f32,
    /// Author reflectance.
    pub reflectance: f32,
    /// Author clearcoat strength.
    pub clearcoat: f32,
    /// Author clearcoat perceptual roughness.
    pub clearcoat_perceptual_roughness: f32,
}

/// Reads the look-owned fields out of a `StandardMaterial`.
#[must_use]
pub fn capture_standard_look_base(material: &StandardMaterial) -> StandardLookBase {
    StandardLookBase {
        perceptual_roughness: material.perceptual_roughness,
        reflectance: material.reflectance,
        clearcoat: material.clearcoat,
        clearcoat_perceptual_roughness: material.clearcoat_perceptual_roughness,
    }
}

/// Base values keyed by material handle.
///
/// Meshes that share one `StandardMaterial` handle share one entry, and only
/// materials that belong to a VRM scene (they carry
/// [`VrmMaterialBaseValues`](bevy_vrm1::prelude::VrmMaterialBaseValues)) are
/// recorded, so the ground plane and UI materials are never captured.
#[derive(Resource, Default, Debug)]
pub struct StandardLookBases {
    bases: HashMap<AssetId<StandardMaterial>, StandardLookBase>,
}

impl StandardLookBases {
    /// Records a base value once. Returns `true` when it was newly recorded.
    pub fn record(
        &mut self,
        id: AssetId<StandardMaterial>,
        base: StandardLookBase,
    ) -> bool {
        if self.bases.contains_key(&id) {
            return false;
        }
        self.bases.insert(id, base);
        true
    }

    /// The captured base for one material handle.
    #[must_use]
    pub fn get(&self, id: AssetId<StandardMaterial>) -> Option<StandardLookBase> {
        self.bases.get(&id).copied()
    }

    /// Number of captured materials.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bases.len()
    }

    /// Whether no material has been captured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    /// Drops every captured material.
    pub fn clear(&mut self) {
        self.bases.clear();
    }
}

/// Captures the look-owned `StandardMaterial` fields once, after the VRM
/// material setup has finalized a mesh material.
///
/// `Added<VrmMaterialBaseValues>` fires exactly once per finalized material,
/// so repeating load/ON/OFF cycles can never promote a modified value back to
/// the base.
pub fn initialize_look_materials(
    mut bases: ResMut<StandardLookBases>,
    finalized: Query<
        (&MeshMaterial3d<StandardMaterial>,),
        Added<VrmMaterialBaseValues>,
    >,
    materials: Res<Assets<StandardMaterial>>,
) {
    for (handle,) in &finalized {
        let Some(material) = materials.get(handle.id()) else {
            continue;
        };
        bases.record(handle.id(), capture_standard_look_base(material));
    }
}

/// Releases the captured materials when the active avatar leaves the ready
/// state (unload, replacement or failed load).
pub fn clear_look_materials_on_unload(
    lifecycle: Res<AvatarLifecycle>,
    mut bases: ResMut<StandardLookBases>,
) {
    if lifecycle.is_changed() && lifecycle.state() != AvatarLifecycleState::Ready {
        bases.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::AvatarLifecycle;

    fn test_app() -> App {
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<StandardLookBases>()
            .init_resource::<AvatarLifecycle>()
            .add_systems(Update, (initialize_look_materials, clear_look_materials_on_unload));
        app
    }

    fn add_material(app: &mut App, material: StandardMaterial) -> Handle<StandardMaterial> {
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(material)
    }

    #[test]
    fn resolve_keeps_the_endpoints_and_the_general_fields() {
        let original = StandardLookBase {
            perceptual_roughness: 0.5,
            reflectance: 0.3,
            clearcoat: 0.2,
            clearcoat_perceptual_roughness: 0.4,
        };
        let off = RichLookSettings {
            enabled: false,
            strength: 1.0,
        };
        assert_eq!(resolve_standard_portrait(original, off, false), original);
        let zero = RichLookSettings {
            enabled: true,
            strength: 0.0,
        };
        assert_eq!(resolve_standard_portrait(original, zero, false), original);
        assert_eq!(resolve_standard_portrait(original, off, true), original);

        let on = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        let rich = resolve_standard_portrait(original, on, false);
        assert!((rich.perceptual_roughness - 0.475).abs() < 1e-6);
        assert_eq!(rich.reflectance, original.reflectance);
        assert_eq!(rich.clearcoat, original.clearcoat);
        assert_eq!(
            rich.clearcoat_perceptual_roughness,
            original.clearcoat_perceptual_roughness
        );
        assert_eq!(resolve_standard_portrait(original, on, true), original);
    }

    fn standard_material() -> StandardMaterial {
        StandardMaterial {
            base_color: Color::srgb(0.4, 0.5, 0.6),
            emissive: LinearRgba::new(0.1, 0.2, 0.3, 1.0),
            perceptual_roughness: 0.6,
            reflectance: 0.35,
            clearcoat: 0.25,
            clearcoat_perceptual_roughness: 0.45,
            metallic: 0.7,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            unlit: false,
            ..default()
        }
    }

    fn portrait_app() -> (App, Handle<StandardMaterial>) {
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<StandardLookBases>()
            .init_resource::<AvatarLookSettings>()
            .add_systems(
                Update,
                (initialize_look_materials, apply_standard_portrait_settings),
            );
        let handle = add_material(&mut app, standard_material());
        let base = VrmMaterialBaseValues::from_standard(
            app.world().resource::<Assets<StandardMaterial>>().get(&handle).unwrap(),
        );
        app.world_mut().spawn((MeshMaterial3d(handle.clone()), base));
        (app, handle)
    }

    fn owned_values(app: &App, handle: &Handle<StandardMaterial>) -> StandardLookBase {
        capture_standard_look_base(
            app.world().resource::<Assets<StandardMaterial>>().get(handle).unwrap(),
        )
    }

    #[test]
    fn portrait_round_trip_only_touches_the_owned_fields() {
        let (mut app, handle) = portrait_app();
        app.update();

        let before = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(&handle)
            .unwrap()
            .clone();

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();

        let after = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(&handle)
            .unwrap()
            .clone();
        assert!((after.perceptual_roughness - before.perceptual_roughness * 0.95).abs() < 1e-6);
        assert_eq!(after.reflectance, before.reflectance);
        assert_eq!(after.clearcoat, before.clearcoat);
        assert_eq!(
            after.clearcoat_perceptual_roughness,
            before.clearcoat_perceptual_roughness
        );
        assert_eq!(after.base_color, before.base_color);
        assert_eq!(after.emissive, before.emissive);
        assert_eq!(after.metallic, before.metallic);
        assert_eq!(after.alpha_mode, before.alpha_mode);
        assert_eq!(after.uv_transform, before.uv_transform);
        assert_eq!(after.base_color_texture, before.base_color_texture);
        assert_eq!(after.normal_map_texture, before.normal_map_texture);
        assert_eq!(after.double_sided, before.double_sided);

        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: false,
            strength: 1.0,
        };
        app.update();
        assert_eq!(owned_values(&app, &handle), capture_standard_look_base(&before));
    }

    #[test]
    fn unlit_and_repeated_frames_leave_the_material_alone() {
        let (mut app, handle) = portrait_app();
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&handle)
            .unwrap()
            .unlit = true;
        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();
        assert_eq!(owned_values(&app, &handle).perceptual_roughness, 0.6);

        // Even with a stale snapshot, a fixed value must not be rewritten.
        let revision = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(&handle)
            .unwrap()
            .perceptual_roughness;
        app.update();
        assert_eq!(owned_values(&app, &handle).perceptual_roughness, revision);
    }

    #[test]
    fn portrait_never_reaches_mtoon_materials() {
        use bevy_vrm1::prelude::MToonMaterial;
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<MToonMaterial>>()
            .init_resource::<StandardLookBases>()
            .init_resource::<AvatarLookSettings>()
            .add_systems(
                Update,
                (initialize_look_materials, apply_standard_portrait_settings),
            );
        let mtoon = app
            .world_mut()
            .resource_mut::<Assets<MToonMaterial>>()
            .add(MToonMaterial::default());
        app.world_mut()
            .spawn((MeshMaterial3d(mtoon), VrmMaterialBaseValues::from_mtoon(
                &MToonMaterial::default(),
            )));
        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();

        assert!(app.world().resource::<StandardLookBases>().is_empty());
    }

    #[test]
    fn capture_reads_only_the_owned_fields() {
        let material = StandardMaterial {
            perceptual_roughness: 0.2,
            reflectance: 0.3,
            clearcoat: 0.4,
            clearcoat_perceptual_roughness: 0.5,
            ..default()
        };
        assert_eq!(
            capture_standard_look_base(&material),
            StandardLookBase {
                perceptual_roughness: 0.2,
                reflectance: 0.3,
                clearcoat: 0.4,
                clearcoat_perceptual_roughness: 0.5,
            }
        );
    }

    #[test]
    fn finalized_materials_are_captured_once_and_do_not_drift() {
        let mut app = test_app();
        let handle = add_material(
            &mut app,
            StandardMaterial {
                perceptual_roughness: 0.5,
                ..default()
            },
        );
        let base = VrmMaterialBaseValues::from_standard(
            app.world().resource::<Assets<StandardMaterial>>().get(&handle).unwrap(),
        );
        let entity = app
            .world_mut()
            .spawn((MeshMaterial3d(handle.clone()), base))
            .id();

        app.update();
        let captured = app
            .world()
            .resource::<StandardLookBases>()
            .get(handle.id())
            .expect("base captured");
        assert_eq!(captured.perceptual_roughness, 0.5);

        // A look stage rewrites the material; the base must not follow.
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&handle)
            .unwrap()
            .perceptual_roughness = 0.05;
        app.update();

        assert_eq!(
            app.world()
                .resource::<StandardLookBases>()
                .get(handle.id())
                .expect("base still present")
                .perceptual_roughness,
            0.5
        );
        assert_eq!(app.world().resource::<StandardLookBases>().len(), 1);

        // The entity is not required by the capture; it exists so the component
        // is attached to a mesh like the real scene.
        assert!(app.world().get_entity(entity).is_ok());
    }

    #[test]
    fn shared_material_handle_is_recorded_once() {
        let mut app = test_app();
        let handle = add_material(&mut app, StandardMaterial::default());
        let component = VrmMaterialBaseValues::from_standard(
            app.world().resource::<Assets<StandardMaterial>>().get(&handle).unwrap(),
        );
        for _ in 0..3 {
            app.world_mut()
                .spawn((MeshMaterial3d(handle.clone()), component));
        }

        app.update();

        assert_eq!(app.world().resource::<StandardLookBases>().len(), 1);
    }

    #[test]
    fn materials_without_a_vrm_index_are_ignored() {
        let mut app = test_app();
        let handle = add_material(&mut app, StandardMaterial::default());
        // Ground/UI material: a `StandardMaterial` mesh with no VRM identity.
        app.world_mut().spawn(MeshMaterial3d(handle));

        app.update();

        assert!(app.world().resource::<StandardLookBases>().is_empty());
    }

    #[test]
    fn leaving_the_ready_state_releases_the_captured_materials() {
        let mut app = test_app();
        let handle = add_material(&mut app, StandardMaterial::default());
        let component = VrmMaterialBaseValues::from_standard(
            app.world().resource::<Assets<StandardMaterial>>().get(&handle).unwrap(),
        );
        app.world_mut()
            .spawn((MeshMaterial3d(handle), component));
        app.update();
        assert_eq!(app.world().resource::<StandardLookBases>().len(), 1);

        let root = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .expect("load from no avatar");
        app.update();

        assert!(app.world().resource::<StandardLookBases>().is_empty());
    }
}
