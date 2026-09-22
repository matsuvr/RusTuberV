//! Captured base values of the `StandardMaterial` fields the look owns, plus
//! the per-material role presets.
//!
//! The look only changes four scalar fields of a `StandardMaterial`
//! (`perceptual_roughness`, `reflectance`, `clearcoat` and
//! `clearcoat_perceptual_roughness`). Everything else (base color, emission,
//! UV transform, metallic and the texture handles) stays owned by the author
//! data and the existing expression writer.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use bevy::asset::AssetId;
use bevy::prelude::*;
use bevy_vrm1::prelude::{
    MToonMaterial, VrmMaterialBaseValues, VrmMaterialIndex, VrmcMaterialRegistry,
};
use serde::{Deserialize, Serialize};

use crate::lifecycle::{AvatarGeneration, AvatarLifecycle, AvatarLifecycleState};
use crate::look::AvatarLookSettings;
use crate::look::preset::{RichLookSettings, blend_look_scalar, effective_look_strength};

/// The material's display role: which role preset tunes the added look terms.
///
/// A role never converts the material's shader kind: an unlit material stays
/// unlit and an MToon material stays MToon. Unknown materials resolve to
/// [`MaterialRole::General`], the regular shared preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaterialRole {
    /// The first version's modest common enhancement.
    General,
    /// The author's colors, expression shading and shade colors dominate, with
    /// a weak gloss.
    Face,
    /// A broad, weak gloss that never applies the head's face normal treatment.
    Skin,
    /// A clearer highlight than the face, without directional flow data.
    Hair,
    /// Restrained gloss that keeps the author's roughness variation.
    Fabric,
    /// The authored metallic/roughness surface and IBL stay in charge.
    Metal,
    /// The drawn eyes, MatCap and alpha/UV expression stay in charge.
    Eye,
}

/// The metallic threshold of the PBR-based Metal hint.
///
/// A Standard material with no role keyword in its name and an authored
/// metallic at or above this value resolves to [`MaterialRole::Metal`]. MToon
/// materials never reach this hint: their glTF-compatible PBR values are not
/// authored metal.
const METALLIC_HINT_THRESHOLD: f32 = 0.5;

/// Infers the material role from the material name and the properties the
/// renderer actually sees.
///
/// The rule set is a small deterministic table over lowercase substring
/// matches. It runs once per material at load time, never per frame. A name
/// that matches no keyword resolves to `General`, except a true Standard
/// material whose authored metallic reaches [`METALLIC_HINT_THRESHOLD`], which
/// resolves to `Metal`. `is_mtoon` only disables that metallic hint: MToon
/// keeps its glTF-compatible PBR values and must never read them as metal.
#[must_use]
pub fn infer_material_role(
    name: &str,
    is_mtoon: bool,
    standard_metallic: Option<f32>,
) -> MaterialRole {
    let name = name.to_lowercase();
    // `skin/body/肌` is tested before `face/顔` so a compound name can never
    // turn the body skin into the Face role, and overlapping words resolve by
    // the first matching row.
    let table: [(&str, MaterialRole); 10] = [
        ("skin", MaterialRole::Skin),
        ("body", MaterialRole::Skin),
        ("肌", MaterialRole::Skin),
        ("face", MaterialRole::Face),
        ("顔", MaterialRole::Face),
        ("hair", MaterialRole::Hair),
        ("髪", MaterialRole::Hair),
        ("eye", MaterialRole::Eye),
        ("瞳", MaterialRole::Eye),
        ("目", MaterialRole::Eye),
    ];
    for (keyword, role) in table {
        if name.contains(keyword) {
            return role;
        }
    }
    let clothing: [(&str, MaterialRole); 5] = [
        ("fabric", MaterialRole::Fabric),
        ("cloth", MaterialRole::Fabric),
        ("金属", MaterialRole::Metal),
        ("metal", MaterialRole::Metal),
        ("服", MaterialRole::Fabric),
    ];
    for (keyword, role) in clothing {
        if name.contains(keyword) {
            return role;
        }
    }
    if !is_mtoon && standard_metallic.is_some_and(|metallic| metallic >= METALLIC_HINT_THRESHOLD) {
        return MaterialRole::Metal;
    }
    MaterialRole::General
}

/// Resolves the effective role: a user selection wins, otherwise the inferred
/// role applies. `selected == None` is the UI's "Auto".
#[must_use]
pub fn resolve_material_role(
    inferred: MaterialRole,
    selected: Option<MaterialRole>,
) -> MaterialRole {
    selected.unwrap_or(inferred)
}

/// One persisted per-material role selection. `selected == None` is "Auto".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialRoleOverride {
    /// The glTF material index the selection applies to.
    pub material_index: usize,
    /// The user's role selection; `None` is the UI's "Auto".
    pub selected: Option<MaterialRole>,
}

/// Per-material roles and names for the active avatar, keyed by
/// [`VrmMaterialIndex`].
///
/// The inference happens once per finalized material at load; user overrides
/// are replaced from the persisted settings. Shared materials share one index
/// and therefore one role, so a shared-material edit acts on the whole
/// material.
#[derive(Resource, Default, Debug)]
pub struct AvatarMaterialRoles {
    inferred: BTreeMap<usize, MaterialRole>,
    selected: BTreeMap<usize, MaterialRole>,
    names: BTreeMap<usize, String>,
}

impl AvatarMaterialRoles {
    /// Records a material's inferred role and name once. Returns `true` when
    /// it was newly recorded.
    pub fn record(
        &mut self,
        index: usize,
        name: &str,
        is_mtoon: bool,
        standard_metallic: Option<f32>,
    ) -> bool {
        if self.inferred.contains_key(&index) {
            return false;
        }
        self.inferred.insert(
            index,
            infer_material_role(name, is_mtoon, standard_metallic),
        );
        self.names.insert(index, name.to_string());
        true
    }

    /// Replaces every user selection. `None` selections remove the override
    /// and return the material to "Auto".
    pub fn replace_overrides(&mut self, overrides: impl Iterator<Item = MaterialRoleOverride>) {
        self.selected.clear();
        for MaterialRoleOverride {
            material_index,
            selected,
        } in overrides
        {
            match selected {
                Some(role) => {
                    self.selected.insert(material_index, role);
                }
                None => {
                    self.selected.remove(&material_index);
                }
            }
        }
    }

    /// The effective role of one material: the user selection, else the
    /// inferred role, else `General`.
    #[must_use]
    pub fn role(&self, index: usize) -> MaterialRole {
        let inferred = self
            .inferred
            .get(&index)
            .copied()
            .unwrap_or(MaterialRole::General);
        resolve_material_role(inferred, self.selected.get(&index).copied())
    }

    /// The material name recorded for one index, if the glTF source named it.
    #[must_use]
    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(&index).map(String::as_str)
    }

    /// The recorded materials as `(index, name, user selection)` triples in
    /// ascending index order.
    #[must_use]
    pub fn entries(&self) -> Vec<(usize, &str, Option<MaterialRole>)> {
        self.inferred
            .keys()
            .chain(self.selected.keys())
            .copied()
            .collect::<BTreeSet<usize>>()
            .into_iter()
            .map(|index| {
                (
                    index,
                    self.name(index).unwrap_or(""),
                    self.selected.get(&index).copied(),
                )
            })
            .collect()
    }

    /// Drops every recorded material and selection.
    pub fn clear(&mut self) {
        self.inferred.clear();
        self.selected.clear();
        self.names.clear();
    }
}

/// Requests that [`AvatarMaterialRoles`] replace its user selections.
#[derive(Message, Clone, Debug, PartialEq)]
pub struct MaterialRoleOverridesChanged(pub Vec<MaterialRoleOverride>);

/// Copies queued role-override changes into [`AvatarMaterialRoles`].
///
/// This system only updates the resource: it never touches materials, files or
/// the inference.
pub fn apply_material_role_overrides(
    mut changes: MessageReader<MaterialRoleOverridesChanged>,
    mut roles: ResMut<AvatarMaterialRoles>,
) {
    for change in changes.read() {
        roles.replace_overrides(change.0.iter().copied());
    }
}

/// The relative roughness the rich look starts from. This is an adjustment
/// starting point, not a measured optimum: it keeps the author's roughness
/// texture variation while making the surface react a little more to the
/// studio lights.
const RICH_ROUGHNESS_SCALE: f32 = 0.95;

/// The per-role relative roughness the Standard look starts from.
///
/// These are adjustment starting points, not measured optima. `General` is the
/// first version's modest common enhancement. `Face` and `Skin` soften the
/// roughness slightly more for their broader, weaker gloss and `Hair` the most
/// for its clearer highlight. `Fabric` keeps the author's roughness variation,
/// and `Metal` and `Eye` keep the authored surface entirely.
fn role_roughness_scale(role: MaterialRole) -> f32 {
    match role {
        MaterialRole::General => RICH_ROUGHNESS_SCALE,
        MaterialRole::Face => 0.92,
        MaterialRole::Skin => 0.85,
        MaterialRole::Hair => 0.75,
        MaterialRole::Fabric | MaterialRole::Metal | MaterialRole::Eye => 1.0,
    }
}

/// Resolves the look-owned `StandardMaterial` values for one material role.
///
/// The look keeps the author reflectance, clearcoat and clearcoat roughness in
/// every role, so only the perceptual roughness moves, and roles that keep the
/// authored surface (`Fabric`, `Metal`, `Eye`) return the original unchanged.
/// Unlit materials and a zero strength return the captured values unchanged.
#[must_use]
pub fn resolve_standard_role_params(
    original: StandardLookBase,
    settings: RichLookSettings,
    role: MaterialRole,
    unlit: bool,
) -> StandardLookBase {
    let strength = effective_look_strength(settings);
    if unlit || strength == 0.0 {
        return original;
    }
    let scale = role_roughness_scale(role);
    if scale == 1.0 {
        return original;
    }
    StandardLookBase {
        perceptual_roughness: blend_look_scalar(
            original.perceptual_roughness,
            original.perceptual_roughness * scale,
            strength,
        ),
        ..original
    }
}

/// Resolves the look-owned `StandardMaterial` values for one material with the
/// first version's general preset.
#[must_use]
pub fn resolve_standard_portrait(
    original: StandardLookBase,
    settings: RichLookSettings,
    unlit: bool,
) -> StandardLookBase {
    resolve_standard_role_params(original, settings, MaterialRole::General, unlit)
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
    roles: Res<AvatarMaterialRoles>,
    bases: Res<StandardLookBases>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    meshes: Query<
        (&MeshMaterial3d<StandardMaterial>, &VrmMaterialIndex),
        With<VrmMaterialBaseValues>,
    >,
) {
    for (handle, index) in &meshes {
        let Some(original) = bases.get(handle.id()) else {
            continue;
        };
        let Some(material) = materials.get(handle.id()) else {
            continue;
        };
        let values =
            resolve_standard_role_params(original, settings.0, roles.role(index.0), material.unlit);
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
    pub fn record(&mut self, id: AssetId<StandardMaterial>, base: StandardLookBase) -> bool {
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

/// Captures the look-owned `StandardMaterial` fields and the material role
/// once, after the VRM material setup has finalized a mesh material.
///
/// `Added<VrmMaterialBaseValues>` fires exactly once per finalized material,
/// so repeating load/ON/OFF cycles can never promote a modified value back to
/// the base.
#[allow(clippy::type_complexity)]
pub fn initialize_look_materials(
    mut bases: ResMut<StandardLookBases>,
    mut roles: ResMut<AvatarMaterialRoles>,
    finalized: Query<
        (Entity, &MeshMaterial3d<StandardMaterial>, &VrmMaterialIndex),
        Added<VrmMaterialBaseValues>,
    >,
    materials: Res<Assets<StandardMaterial>>,
    registries: Query<&VrmcMaterialRegistry>,
    parents: Query<&ChildOf>,
) {
    for (entity, handle, index) in &finalized {
        let Some(material) = materials.get(handle.id()) else {
            continue;
        };
        bases.record(handle.id(), capture_standard_look_base(material));
        let name = gltf_material_name(&registries, &parents, entity, index.0);
        roles.record(index.0, &name, false, Some(material.metallic));
    }
}

/// Captures the material role of each finalized MToon material.
///
/// MToon materials keep their glTF-compatible PBR values, so no Standard
/// metallic is handed to the inference.
pub fn initialize_mtoon_look_materials(
    mut roles: ResMut<AvatarMaterialRoles>,
    finalized: Query<
        (Entity, &MeshMaterial3d<MToonMaterial>, &VrmMaterialIndex),
        Added<VrmMaterialBaseValues>,
    >,
    registries: Query<&VrmcMaterialRegistry>,
    parents: Query<&ChildOf>,
) {
    for (entity, _, index) in &finalized {
        let name = gltf_material_name(&registries, &parents, entity, index.0);
        roles.record(index.0, &name, true, None);
    }
}

/// The glTF material name recorded for one finalized material entity.
fn gltf_material_name(
    registries: &Query<&VrmcMaterialRegistry>,
    parents: &Query<&ChildOf>,
    entity: Entity,
    index: usize,
) -> String {
    let root = parents.root_ancestor(entity);
    registries
        .get(root)
        .ok()
        .and_then(|registry| registry.names.get(&index).cloned())
        .unwrap_or_default()
}

/// Releases the captured materials and roles when the old avatar goes away,
/// while keeping one model's `Loading -> Binding -> Ready` progression intact.
///
/// Old-model destruction (`Unloading`, `NoAvatar`, `Failed`) always clears.
/// `Loading`/`Binding` clears only when the lifecycle generation changed, so a
/// new model drops the previous model's data while the same model's binding
/// keeps restored selections and captured bases. `Ready` never clears.
///
/// This system must run before [`apply_material_role_overrides`],
/// [`initialize_look_materials`] and [`initialize_mtoon_look_materials`] in
/// the same frame, so a new model's restore and capture land after the old
/// data is gone instead of being wiped by it.
pub fn clear_look_materials_on_unload(
    lifecycle: Res<AvatarLifecycle>,
    mut bases: ResMut<StandardLookBases>,
    mut roles: ResMut<AvatarMaterialRoles>,
    mut last_generation: Local<Option<AvatarGeneration>>,
) {
    let state = lifecycle.state();
    let generation = lifecycle.current_generation();
    let generation_changed = *last_generation != Some(generation);
    let should_clear = match state {
        AvatarLifecycleState::Failed
        | AvatarLifecycleState::NoAvatar
        | AvatarLifecycleState::Unloading => true,
        AvatarLifecycleState::Loading | AvatarLifecycleState::Binding => generation_changed,
        AvatarLifecycleState::Ready => false,
    };
    if should_clear && (lifecycle.is_changed() || generation_changed) {
        bases.clear();
        roles.clear();
    }
    *last_generation = Some(generation);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::AvatarLifecycle;

    fn test_app() -> App {
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<StandardLookBases>()
            .init_resource::<AvatarMaterialRoles>()
            .init_resource::<AvatarLifecycle>()
            .add_systems(
                Update,
                (
                    clear_look_materials_on_unload,
                    initialize_look_materials.after(clear_look_materials_on_unload),
                ),
            );
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

    #[test]
    fn resolve_interpolates_roughness_at_intermediate_strengths() {
        let original = StandardLookBase {
            perceptual_roughness: 0.5,
            reflectance: 0.3,
            clearcoat: 0.2,
            clearcoat_perceptual_roughness: 0.4,
        };
        for (strength, expected) in [(0.0, 0.5), (0.01, 0.49975), (0.5, 0.4875), (1.0, 0.475)] {
            let settings = RichLookSettings {
                enabled: true,
                strength,
            };
            let resolved = resolve_standard_portrait(original, settings, false);
            assert!((resolved.perceptual_roughness - expected).abs() < 1e-6);
            assert_eq!(
                StandardLookBase {
                    perceptual_roughness: original.perceptual_roughness,
                    ..resolved
                },
                original
            );
            assert_eq!(
                resolve_standard_portrait(original, settings, true),
                original
            );
            assert_eq!(
                resolve_standard_portrait(
                    original,
                    RichLookSettings {
                        enabled: false,
                        ..settings
                    },
                    false
                ),
                original
            );
        }
    }

    fn standard_material() -> StandardMaterial {
        StandardMaterial {
            base_color: Color::srgb(0.4, 0.5, 0.6),
            emissive: LinearRgba::new(0.1, 0.2, 0.3, 1.0),
            perceptual_roughness: 0.6,
            reflectance: 0.35,
            clearcoat: 0.25,
            clearcoat_perceptual_roughness: 0.45,
            // Not metallic: an unnamed metallic material would infer the Metal
            // role, which keeps the authored surface untouched.
            metallic: 0.0,
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
            .init_resource::<AvatarMaterialRoles>()
            .init_resource::<AvatarLookSettings>()
            .add_systems(
                Update,
                (initialize_look_materials, apply_standard_portrait_settings),
            );
        let handle = add_material(&mut app, standard_material());
        let base = VrmMaterialBaseValues::from_standard(
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .unwrap(),
        );
        app.world_mut()
            .spawn((MeshMaterial3d(handle.clone()), base, VrmMaterialIndex(0)));
        (app, handle)
    }

    fn owned_values(app: &App, handle: &Handle<StandardMaterial>) -> StandardLookBase {
        capture_standard_look_base(
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(handle)
                .unwrap(),
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
        assert_eq!(
            owned_values(&app, &handle),
            capture_standard_look_base(&before)
        );
    }

    #[test]
    fn strength_changes_always_use_the_captured_original() {
        let (mut app, handle) = portrait_app();
        app.update();
        let original = owned_values(&app, &handle);
        for _ in 0..3 {
            for (enabled, strength, expected) in [
                (true, 0.0, 0.6),
                (true, 1.0, 0.57),
                (true, 0.5, 0.585),
                (false, 0.5, 0.6),
                (true, 0.5, 0.585),
                (true, 0.0, 0.6),
            ] {
                app.world_mut().resource_mut::<AvatarLookSettings>().0 =
                    RichLookSettings { enabled, strength };
                app.update();
                let actual = owned_values(&app, &handle);
                assert!((actual.perceptual_roughness - expected).abs() < 1e-6);
                assert_eq!(
                    StandardLookBase {
                        perceptual_roughness: original.perceptual_roughness,
                        ..actual
                    },
                    original
                );
                assert_eq!(
                    app.world().resource::<StandardLookBases>().get(handle.id()),
                    Some(original)
                );
                if !enabled || strength == 0.0 {
                    assert_eq!(actual, original);
                }
            }
        }
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
            .init_resource::<AvatarMaterialRoles>()
            .init_resource::<AvatarLookSettings>()
            .add_systems(
                Update,
                (initialize_look_materials, apply_standard_portrait_settings),
            );
        let mtoon = app
            .world_mut()
            .resource_mut::<Assets<MToonMaterial>>()
            .add(MToonMaterial::default());
        app.world_mut().spawn((
            MeshMaterial3d(mtoon),
            VrmMaterialBaseValues::from_mtoon(&MToonMaterial::default()),
        ));
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
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .unwrap(),
        );
        let entity = app
            .world_mut()
            .spawn((MeshMaterial3d(handle.clone()), base, VrmMaterialIndex(0)))
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
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .unwrap(),
        );
        for _ in 0..3 {
            app.world_mut().spawn((
                MeshMaterial3d(handle.clone()),
                component,
                VrmMaterialIndex(0),
            ));
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
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .unwrap(),
        );
        app.world_mut()
            .spawn((MeshMaterial3d(handle), component, VrmMaterialIndex(0)));
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

    #[test]
    fn role_names_resolve_to_distinct_roles() {
        for (name, expected) in [
            ("Face", MaterialRole::Face),
            ("face_01", MaterialRole::Face),
            ("顔", MaterialRole::Face),
            ("Skin", MaterialRole::Skin),
            ("body", MaterialRole::Skin),
            ("肌", MaterialRole::Skin),
            ("Hair_Front", MaterialRole::Hair),
            ("髪", MaterialRole::Hair),
            ("Eye_L", MaterialRole::Eye),
            ("eyewhite", MaterialRole::Eye),
            ("瞳", MaterialRole::Eye),
            ("Fabric", MaterialRole::Fabric),
            ("cloth_01", MaterialRole::Fabric),
            ("服", MaterialRole::Fabric),
            ("Metal", MaterialRole::Metal),
            ("金属", MaterialRole::Metal),
            ("UnknownMaterial", MaterialRole::General),
            ("衣装_01", MaterialRole::General),
        ] {
            assert_eq!(
                infer_material_role(name, false, None),
                expected,
                "name {name:?}"
            );
        }
    }

    #[test]
    fn body_is_skin_even_when_face_like_words_follow() {
        assert_eq!(
            infer_material_role("body_face", false, None),
            MaterialRole::Skin
        );
        // A face material is never mistaken for the body skin.
        assert_eq!(infer_material_role("Face", false, None), MaterialRole::Face);
    }

    /// Real material names from `tests/fixtures/vrm`: multi-word and
    /// `(Instance)`-suffixed names resolve by the first matching table row.
    /// `Face_00_SKIN` carries both words and resolves to Skin (the skin row
    /// runs first), which is the recorded behavior for this ambiguity.
    #[test]
    fn fixture_model_material_names_resolve_deterministically() {
        for (name, expected) in [
            (
                "N00_000_00_FaceMouth_00_FACE (Instance)",
                MaterialRole::Face,
            ),
            ("N00_000_00_Face_00_SKIN (Instance)", MaterialRole::Skin),
            ("N00_000_00_Body_00_SKIN (Instance)", MaterialRole::Skin),
            ("N00_000_00_EyeIris_00_EYE (Instance)", MaterialRole::Eye),
            ("N00_000_00_FaceBrow_00_FACE (Instance)", MaterialRole::Face),
            ("N00_000_00_HairBack_00_HAIR (Instance)", MaterialRole::Hair),
            ("N00_002_01_Tops_01_CLOTH (Instance)", MaterialRole::Fabric),
            (
                "N00_001_02_Accessory_Tie_01_CLOTH (Instance)",
                MaterialRole::Fabric,
            ),
        ] {
            assert_eq!(
                infer_material_role(name, true, None),
                expected,
                "name {name:?}"
            );
        }
    }

    #[test]
    fn the_metallic_hint_applies_only_to_true_standard_materials() {
        assert_eq!(
            infer_material_role("UnknownMaterial", false, Some(0.8)),
            MaterialRole::Metal
        );
        assert_eq!(
            infer_material_role("UnknownMaterial", false, Some(0.5)),
            MaterialRole::Metal
        );
        // Below the threshold, or MToon's glTF-compatible PBR values, the hint
        // never fires.
        assert_eq!(
            infer_material_role("UnknownMaterial", false, Some(0.49)),
            MaterialRole::General
        );
        assert_eq!(
            infer_material_role("UnknownMaterial", true, Some(0.8)),
            MaterialRole::General
        );
        // A named material keeps its name role over the metallic hint.
        assert_eq!(
            infer_material_role("Hair", false, Some(0.9)),
            MaterialRole::Hair
        );
    }

    #[test]
    fn a_user_selection_wins_and_auto_keeps_the_inference() {
        assert_eq!(
            resolve_material_role(MaterialRole::Skin, None),
            MaterialRole::Skin
        );
        assert_eq!(
            resolve_material_role(MaterialRole::Skin, Some(MaterialRole::Face)),
            MaterialRole::Face
        );
    }

    #[test]
    fn standard_role_params_differ_per_role_and_keep_the_author_fields() {
        let original = StandardLookBase {
            perceptual_roughness: 0.5,
            reflectance: 0.3,
            clearcoat: 0.2,
            clearcoat_perceptual_roughness: 0.4,
        };
        let on = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        let general = resolve_standard_role_params(original, on, MaterialRole::General, false);
        assert!((general.perceptual_roughness - 0.475).abs() < 1e-6);
        let face = resolve_standard_role_params(original, on, MaterialRole::Face, false);
        assert!((face.perceptual_roughness - 0.46).abs() < 1e-6);
        let skin = resolve_standard_role_params(original, on, MaterialRole::Skin, false);
        assert!((skin.perceptual_roughness - 0.425).abs() < 1e-6);
        let hair = resolve_standard_role_params(original, on, MaterialRole::Hair, false);
        assert!((hair.perceptual_roughness - 0.375).abs() < 1e-6);
        for role in [MaterialRole::Fabric, MaterialRole::Metal, MaterialRole::Eye] {
            assert_eq!(
                resolve_standard_role_params(original, on, role, false),
                original,
                "role {role:?} keeps the authored surface"
            );
        }
        for role in [
            MaterialRole::General,
            MaterialRole::Face,
            MaterialRole::Skin,
            MaterialRole::Hair,
            MaterialRole::Fabric,
            MaterialRole::Metal,
            MaterialRole::Eye,
        ] {
            let resolved = resolve_standard_role_params(original, on, role, false);
            assert_eq!(resolved.reflectance, original.reflectance);
            assert_eq!(resolved.clearcoat, original.clearcoat);
            assert_eq!(
                resolved.clearcoat_perceptual_roughness,
                original.clearcoat_perceptual_roughness
            );
            assert_eq!(
                resolve_standard_role_params(original, on, role, true),
                original,
                "unlit role {role:?}"
            );
        }
    }

    #[test]
    fn role_overrides_replace_and_clear_by_index() {
        let mut roles = AvatarMaterialRoles::default();
        roles.record(0, "Face", true, None);
        roles.record(1, "body", true, None);
        assert_eq!(roles.role(0), MaterialRole::Face);
        assert_eq!(roles.role(1), MaterialRole::Skin);
        roles.replace_overrides(
            [
                MaterialRoleOverride {
                    material_index: 0,
                    selected: Some(MaterialRole::Hair),
                },
                MaterialRoleOverride {
                    material_index: 1,
                    selected: None,
                },
            ]
            .into_iter(),
        );
        assert_eq!(roles.role(0), MaterialRole::Hair);
        assert_eq!(roles.role(1), MaterialRole::Skin);
        assert_eq!(roles.name(0), Some("Face"));
        assert_eq!(
            roles.entries(),
            vec![(0, "Face", Some(MaterialRole::Hair)), (1, "body", None),]
        );
        roles.clear();
        assert_eq!(roles.role(0), MaterialRole::General);
        assert!(roles.entries().is_empty());
    }

    #[test]
    fn override_messages_replace_the_resource_selections() {
        let mut app = App::new();
        app.init_resource::<AvatarMaterialRoles>()
            .add_message::<MaterialRoleOverridesChanged>()
            .add_systems(Update, apply_material_role_overrides);
        app.world_mut()
            .resource_mut::<Messages<MaterialRoleOverridesChanged>>()
            .write(MaterialRoleOverridesChanged(vec![MaterialRoleOverride {
                material_index: 2,
                selected: Some(MaterialRole::Eye),
            }]));
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(2),
            MaterialRole::Eye
        );
    }

    #[test]
    fn mtoon_roles_are_captured_without_the_metallic_hint() {
        let mut app = App::new();
        app.init_resource::<Assets<MToonMaterial>>()
            .init_resource::<AvatarMaterialRoles>()
            .add_systems(Update, initialize_mtoon_look_materials);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<MToonMaterial>>()
            .add(MToonMaterial::default());
        let root = app
            .world_mut()
            .spawn(VrmcMaterialRegistry {
                names: HashMap::from([(0, "Face".to_string())]),
                ..default()
            })
            .id();
        let entity = app
            .world_mut()
            .spawn((
                MeshMaterial3d(handle),
                VrmMaterialBaseValues::from_mtoon(&MToonMaterial::default()),
                VrmMaterialIndex(0),
                ChildOf(root),
            ))
            .id();
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::Face
        );
        app.world_mut().despawn(entity);
    }

    fn full_look_app() -> App {
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<MToonMaterial>>()
            .init_resource::<StandardLookBases>()
            .init_resource::<AvatarMaterialRoles>()
            .init_resource::<AvatarLookSettings>()
            .init_resource::<AvatarLifecycle>()
            .add_message::<MaterialRoleOverridesChanged>()
            .add_systems(
                Update,
                (
                    clear_look_materials_on_unload,
                    apply_material_role_overrides.after(clear_look_materials_on_unload),
                    initialize_look_materials.after(clear_look_materials_on_unload),
                    initialize_mtoon_look_materials.after(clear_look_materials_on_unload),
                    apply_standard_portrait_settings
                        .after(initialize_look_materials)
                        .after(apply_material_role_overrides),
                )
                    .chain(),
            );
        app
    }

    fn spawn_standard_material_entity(
        app: &mut App,
        handle: Handle<StandardMaterial>,
        index: usize,
    ) {
        let base = VrmMaterialBaseValues::from_standard(
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .unwrap(),
        );
        app.world_mut()
            .spawn((MeshMaterial3d(handle), base, VrmMaterialIndex(index)));
    }

    fn restore_roles(app: &mut App, overrides: Vec<MaterialRoleOverride>) {
        app.world_mut()
            .resource_mut::<Messages<MaterialRoleOverridesChanged>>()
            .write(MaterialRoleOverridesChanged(overrides));
    }

    fn spawn_named_standard_material_entity(
        app: &mut App,
        handle: Handle<StandardMaterial>,
        index: usize,
        name: &str,
    ) {
        use std::collections::HashMap;
        let registry_root = app
            .world_mut()
            .spawn(bevy_vrm1::prelude::VrmcMaterialRegistry {
                names: HashMap::from([(index, name.to_string())]),
                ..default()
            })
            .id();
        let base = VrmMaterialBaseValues::from_standard(
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .unwrap(),
        );
        app.world_mut().spawn((
            MeshMaterial3d(handle),
            base,
            VrmMaterialIndex(index),
            ChildOf(registry_root),
        ));
    }

    #[test]
    fn saved_manual_survives_loading_to_ready() {
        let mut app = full_look_app();
        let root = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .unwrap();
        let handle = add_material(&mut app, StandardMaterial::default());
        spawn_standard_material_entity(&mut app, handle.clone(), 0);
        restore_roles(
            &mut app,
            vec![MaterialRoleOverride {
                material_index: 0,
                selected: Some(MaterialRole::Face),
            }],
        );
        // Loading -> Binding -> Ready keeps the restored selection and capture.
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(root);
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::Face
        );
        assert_eq!(app.world().resource::<StandardLookBases>().len(), 1);
        assert!(
            !app.world()
                .resource::<AvatarMaterialRoles>()
                .entries()
                .is_empty()
        );
    }

    #[test]
    fn inferred_roles_survive_binding_without_selection() {
        let mut app = full_look_app();
        let root = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .unwrap();
        let handle = add_material(&mut app, StandardMaterial::default());
        spawn_named_standard_material_entity(&mut app, handle, 0, "body");
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::Skin,
            "Auto resolves through the captured inference"
        );
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(root);
        for _ in 0..5 {
            app.update();
        }
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::Skin,
            "Binding frames must not drop the captured inference"
        );
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().name(0),
            Some("body")
        );
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::Skin
        );
    }

    #[test]
    fn binding_multiple_frames_retain_roles_names_and_bases() {
        let mut app = full_look_app();
        let root = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .unwrap();
        let handle = add_material(&mut app, StandardMaterial::default());
        spawn_standard_material_entity(&mut app, handle.clone(), 0);
        restore_roles(
            &mut app,
            vec![MaterialRoleOverride {
                material_index: 0,
                selected: Some(MaterialRole::Hair),
            }],
        );
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(root);
        for _ in 0..5 {
            app.update();
        }
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::Hair
        );
        assert_eq!(app.world().resource::<StandardLookBases>().len(), 1);
        assert_eq!(
            app.world()
                .resource::<AvatarMaterialRoles>()
                .entries()
                .len(),
            1
        );
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::Hair
        );
    }

    #[test]
    fn model_switch_restores_each_without_mixing() {
        let mut app = full_look_app();
        // Model A.
        let root_a = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root_a)
            .unwrap();
        let handle_a = add_material(&mut app, StandardMaterial::default());
        spawn_standard_material_entity(&mut app, handle_a.clone(), 0);
        restore_roles(
            &mut app,
            vec![MaterialRoleOverride {
                material_index: 0,
                selected: Some(MaterialRole::Face),
            }],
        );
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(root_a);
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::Face
        );

        // Model A -> B replaces the generation and clears A's data first.
        let root_b = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_replace(root_b)
            .unwrap();
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_unload();
        app.update();
        assert!(app.world().resource::<StandardLookBases>().is_empty());
        assert!(
            app.world()
                .resource::<AvatarMaterialRoles>()
                .entries()
                .is_empty()
        );

        let handle_b = add_material(&mut app, StandardMaterial::default());
        spawn_standard_material_entity(&mut app, handle_b.clone(), 1);
        restore_roles(
            &mut app,
            vec![MaterialRoleOverride {
                material_index: 1,
                selected: Some(MaterialRole::Hair),
            }],
        );
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(root_b);
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(1),
            MaterialRole::Hair
        );
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::General,
            "model B must not carry model A's role"
        );

        // Model B -> A again restores A's selection without B leaking in.
        let root_a2 = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_replace(root_a2)
            .unwrap();
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_unload();
        app.update();
        assert!(
            app.world()
                .resource::<AvatarMaterialRoles>()
                .entries()
                .is_empty()
        );
        let handle_a2 = add_material(&mut app, StandardMaterial::default());
        spawn_standard_material_entity(&mut app, handle_a2.clone(), 0);
        restore_roles(
            &mut app,
            vec![MaterialRoleOverride {
                material_index: 0,
                selected: Some(MaterialRole::Face),
            }],
        );
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(root_a2);
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(0),
            MaterialRole::Face
        );
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(1),
            MaterialRole::General,
            "model A must not carry model B's role"
        );
    }

    #[test]
    fn unload_and_failure_drop_old_lists() {
        let mut app = full_look_app();
        let root = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .unwrap();
        let handle = add_material(&mut app, StandardMaterial::default());
        spawn_standard_material_entity(&mut app, handle, 0);
        restore_roles(
            &mut app,
            vec![MaterialRoleOverride {
                material_index: 0,
                selected: Some(MaterialRole::Face),
            }],
        );
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(root);
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();
        app.update();
        assert!(
            !app.world()
                .resource::<AvatarMaterialRoles>()
                .entries()
                .is_empty()
        );

        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_unload()
            .unwrap();
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_unload();
        app.update();
        assert!(app.world().resource::<StandardLookBases>().is_empty());
        assert!(
            app.world()
                .resource::<AvatarMaterialRoles>()
                .entries()
                .is_empty()
        );

        let root_b = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root_b)
            .unwrap();
        let handle_b = add_material(&mut app, StandardMaterial::default());
        spawn_standard_material_entity(&mut app, handle_b, 0);
        restore_roles(
            &mut app,
            vec![MaterialRoleOverride {
                material_index: 0,
                selected: Some(MaterialRole::Hair),
            }],
        );
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(root_b);
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .fail(crate::lifecycle::AvatarLifecycleFailure::AssetLoadFailed);
        app.update();
        assert!(app.world().resource::<StandardLookBases>().is_empty());
        assert!(
            app.world()
                .resource::<AvatarMaterialRoles>()
                .entries()
                .is_empty()
        );
    }

    #[test]
    fn shared_index_meshes_share_one_role_application() {
        let mut app = full_look_app();
        let root = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .unwrap();
        let handle = add_material(
            &mut app,
            StandardMaterial {
                perceptual_roughness: 0.5,
                ..default()
            },
        );
        // Two meshes share one material handle and one glTF index.
        for _ in 0..2 {
            spawn_standard_material_entity(&mut app, handle.clone(), 3);
        }
        restore_roles(
            &mut app,
            vec![MaterialRoleOverride {
                material_index: 3,
                selected: Some(MaterialRole::Hair),
            }],
        );
        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(root);
        app.update();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(3),
            MaterialRole::Hair
        );
        assert!(
            (owned_values(&app, &handle).perceptual_roughness - 0.375).abs() < 1e-6,
            "shared meshes resolve through their shared index"
        );
        assert_eq!(app.world().resource::<StandardLookBases>().len(), 1);
    }

    #[test]
    fn a_material_role_override_switches_the_applied_standard_preset() {
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<StandardLookBases>()
            .init_resource::<AvatarMaterialRoles>()
            .init_resource::<AvatarLookSettings>()
            .add_message::<MaterialRoleOverridesChanged>()
            .add_systems(
                Update,
                (
                    initialize_look_materials,
                    apply_material_role_overrides,
                    apply_standard_portrait_settings,
                )
                    .chain(),
            );
        let handle = add_material(
            &mut app,
            StandardMaterial {
                perceptual_roughness: 0.5,
                ..default()
            },
        );
        let base = VrmMaterialBaseValues::from_standard(
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .unwrap(),
        );
        app.world_mut()
            .spawn((MeshMaterial3d(handle.clone()), base, VrmMaterialIndex(0)));
        app.world_mut()
            .resource_mut::<Messages<MaterialRoleOverridesChanged>>()
            .write(MaterialRoleOverridesChanged(vec![MaterialRoleOverride {
                material_index: 0,
                selected: Some(MaterialRole::Hair),
            }]));
        app.world_mut().resource_mut::<AvatarLookSettings>().0 = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        app.update();
        assert!(
            (owned_values(&app, &handle).perceptual_roughness - 0.375).abs() < 1e-6,
            "the Hair preset applies through the runtime override"
        );
    }
}
