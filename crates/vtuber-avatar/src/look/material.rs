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
