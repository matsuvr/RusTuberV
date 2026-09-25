//! App-side material color / UV expression binds (see #91, PR #98 R3).
//!
//! The unmodified upstream runtime applies morph binds only; `materialColorBinds`
//! and `textureTransformBinds` are not part of its expression contract, and the
//! VRM 0.x `materialValues` equivalent was dropped by the first port of this
//! PR. This module restores that non-Rich feature on top of upstream's public
//! APIs without touching upstream shaders or duplicating the morph update:
//!
//! - A public glTF loader extension hook
//!   ([`bevy::gltf::GltfExtensionHandlers`]) tags every spawned mesh entity
//!   with its glTF material index (`VrmMaterialIndex`) at load time.
//! - Binding inserts `ExpressionMaterialBinds` (parsed source facts) on each
//!   expression entity and `AvatarMaterialExpressionState` on the avatar root.
//! - [`apply_expression_materials`] evaluates, per frame, the same effective
//!   weight the upstream morph pass uses (raw override/transform weight,
//!   binary snapping, category suppression) and writes `base + Σ (target -
//!   base) · weight` into each resolved scene material asset. The base is the
//!   author's expression-free value captured once per asset before the first
//!   write, so a weight of 0 restores it exactly and nothing accumulates frame
//!   to frame. Each asset is evaluated and written at most once per frame, and
//!   unchanged values are never written to `Assets`.

use std::collections::{HashMap, HashSet};

use bevy::asset::{AssetId, Assets};
use bevy::color::LinearRgba;
use bevy::gltf::extensions::GltfExtensionHandlers;
use bevy::math::{Affine2, Vec2};
use bevy::prelude::*;

use bevy_vrm1::prelude::{
    BinaryExpression, ExpressionEntityMap, ExpressionOverride, ExpressionOverrideSettings,
    MToonMaterial,
};

use crate::expression::source::{MaterialColorTarget, SourceExpressions};
use crate::lifecycle::{AvatarLifecycle, AvatarLifecycleState};

/// glTF material index of a mesh entity spawned by the glTF loader.
///
/// Expression material binds reference materials by this stable index; the
/// glTF spec does not require material names to be unique, so the index is
/// the only identity. Inserted into the load-time scene by
/// [`GltfMaterialIndexHandler`], so it survives per-instance scene cloning
/// (the type must stay registered).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component)]
pub struct VrmMaterialIndex(pub usize);

/// Expression-free base values captured once per material asset.
///
/// Capture happens before the first expression write, so the values are the
/// author's; a zero weight always evaluates back to them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VrmMaterialBaseValues {
    /// Expression-free base color.
    pub base_color: LinearRgba,
    /// Expression-free emissive color.
    pub emissive: LinearRgba,
    /// Expression-free MToon shade color (`BLACK` for standard materials).
    pub shade_color: LinearRgba,
    /// Expression-free MToon parametric rim color (`BLACK` for standard).
    pub rim_color: LinearRgba,
    /// Expression-free MToon outline color (`BLACK` for standard).
    pub outline_color: LinearRgba,
    /// Base UV transform; texture binds blend toward their scale/offset.
    pub uv_transform: Affine2,
}

/// Expression state owned by one concrete material asset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialExpressionState {
    /// glTF material index the asset was spawned for.
    pub material_index: usize,
    /// Expression-free values captured before the first write.
    pub base: VrmMaterialBaseValues,
    /// Last values written to the asset; unchanged evaluations are skipped.
    pub applied: VrmMaterialBaseValues,
}

/// Per-avatar expression state keyed by concrete material asset ID.
///
/// Mesh entities can share one material asset, so the base and last-applied
/// values are owned by the asset, not the mesh. The component lives on the
/// avatar root and is dropped with it by the existing unload lifecycle.
#[derive(Component, Debug, Clone, Default)]
pub struct AvatarMaterialExpressionState {
    /// MToon material state by asset ID.
    pub mtoon: HashMap<AssetId<MToonMaterial>, MaterialExpressionState>,
    /// Standard material state by asset ID.
    pub standard: HashMap<AssetId<StandardMaterial>, MaterialExpressionState>,
}

/// One material color bind resolved onto an expression entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpressionMaterialColorBind {
    /// glTF material index the bind targets.
    pub material_index: usize,
    /// Target property on that material.
    pub target: MaterialColorTarget,
    /// Linear RGBA target value from the source.
    pub target_value: LinearRgba,
}

/// One texture transform bind resolved onto an expression entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpressionTextureTransformBind {
    /// glTF material index the bind targets.
    pub material_index: usize,
    /// UV scale target; `[1, 1]` when the source omitted it.
    pub scale: Vec2,
    /// UV offset target; `[0, 0]` when the source omitted it.
    pub offset: Vec2,
}

/// Material binds declared by one expression, kept on its expression entity.
#[derive(Component, Debug, Clone, Default, PartialEq)]
pub struct ExpressionMaterialBinds {
    /// Declared color binds with a supported target type.
    pub colors: Vec<ExpressionMaterialColorBind>,
    /// Declared texture transform binds.
    pub transforms: Vec<ExpressionTextureTransformBind>,
}

impl ExpressionMaterialBinds {
    /// Converts parsed source facts; `None` target types are dropped here and
    /// already counted as unsupported in the bind status.
    #[must_use]
    pub fn from_source(expressions: &SourceExpressions, name: &str) -> Self {
        let Some(entry) = expressions.entry(name) else {
            return Self::default();
        };
        Self {
            colors: entry
                .material_color_binds
                .iter()
                .filter_map(|bind| {
                    Some(ExpressionMaterialColorBind {
                        material_index: bind.material,
                        target: bind.target?,
                        target_value: LinearRgba::new(
                            bind.target_value[0],
                            bind.target_value[1],
                            bind.target_value[2],
                            bind.target_value[3],
                        ),
                    })
                })
                .collect(),
            transforms: entry
                .texture_transform_binds
                .iter()
                .map(|bind| ExpressionTextureTransformBind {
                    material_index: bind.material,
                    scale: bind.scale.map_or(Vec2::ONE, Vec2::from),
                    offset: bind.offset.map_or(Vec2::ZERO, Vec2::from),
                })
                .collect(),
        }
    }

    /// Returns `true` when neither color nor texture-transform binds exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty() && self.transforms.is_empty()
    }
}

impl VrmMaterialBaseValues {
    fn from_mtoon(material: &MToonMaterial) -> Self {
        Self {
            base_color: material.base_color.to_linear(),
            emissive: material.emissive,
            shade_color: material.shade.color,
            rim_color: material.rim_lighting.color,
            outline_color: material.outline.color,
            uv_transform: material.uv_transform,
        }
    }

    fn from_standard(material: &StandardMaterial) -> Self {
        Self {
            base_color: material.base_color.to_linear(),
            emissive: material.emissive,
            shade_color: LinearRgba::BLACK,
            rim_color: LinearRgba::BLACK,
            outline_color: LinearRgba::BLACK,
            uv_transform: material.uv_transform,
        }
    }
}

/// glTF loader extension hook that records the glTF material index of every
/// spawned mesh-and-material entity.
///
/// The loader invokes [`bevy::gltf::extensions::GltfExtensionHandler::
/// on_spawn_mesh_and_material`] while the load-time scene is being built,
/// before the upstream MToon pass can replace
/// `MeshMaterial3d<StandardMaterial>`, so the mapping is race-free and
/// survives per-instance scene cloning.
#[derive(Default)]
pub struct GltfMaterialIndexHandler;

impl bevy::gltf::extensions::GltfExtensionHandler for GltfMaterialIndexHandler {
    fn dyn_clone(&self) -> Box<dyn bevy::gltf::extensions::ErasedGltfExtensionHandler> {
        Box::new(Self)
    }

    fn on_spawn_mesh_and_material(
        &mut self,
        _load_context: &mut bevy::asset::LoadContext<'_>,
        _primitive: &bevy::gltf::gltf::Primitive,
        _mesh: &bevy::gltf::gltf::Mesh,
        material: &bevy::gltf::gltf::Material,
        entity: &mut EntityWorldMut,
        _material_label: &str,
    ) {
        if let Some(index) = material.index() {
            entity.insert(VrmMaterialIndex(index));
        }
    }
}

/// Registers the handler with the glTF loader. Must run before the loader
/// plugin's `finish` snapshots the handler list.
pub fn register_gltf_material_index_handler(app: &mut App) {
    app.init_resource::<GltfExtensionHandlers>();
    let handlers = app.world_mut().resource_mut::<GltfExtensionHandlers>();
    handlers
        .0
        .write_blocking()
        .push(Box::new(GltfMaterialIndexHandler));
    app.register_type::<VrmMaterialIndex>();
}

/// How a name-driven expression is classified for suppression, matching the
/// upstream category table (`ExpressionCategory::from_preset_name`) exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpressionCategory {
    Mouth,
    Blink,
    LookAt,
    Other,
}

fn category_of(name: &str) -> ExpressionCategory {
    match name {
        "aa" | "ih" | "ou" | "ee" | "oh" => ExpressionCategory::Mouth,
        "blink" | "blinkLeft" | "blinkRight" => ExpressionCategory::Blink,
        "lookUp" | "lookDown" | "lookLeft" | "lookRight" => ExpressionCategory::LookAt,
        _ => ExpressionCategory::Other,
    }
}

fn output_weight(raw_weight: f32, is_binary: bool) -> f32 {
    if is_binary {
        if raw_weight > 0.5 { 1.0 } else { 0.0 }
    } else {
        raw_weight.clamp(0.0, 1.0)
    }
}

/// Final expression weight for one expression, in the same computation order
/// as the adopted upstream `bind_expressions`: the raw weight is binary-snapped
/// or clamped first, and a binary expression is fully suppressed (weight 0)
/// whenever its category multiplier is below 1. Non-binary expressions keep
/// the ordinary `output * multiplier` product.
fn effective_expression_weight(raw_weight: f32, is_binary: bool, category_multiplier: f32) -> f32 {
    let output_weight = output_weight(raw_weight, is_binary);
    if is_binary && category_multiplier < 1.0 {
        0.0
    } else {
        output_weight * category_multiplier
    }
}

/// Applies material binds for the active avatar root each frame.
///
/// Runs in `PostUpdate` after the upstream `Expressions` set so the weights
/// read here are the ones the morph pass consumed. Each concrete material
/// asset is evaluated at most once per frame; meshes sharing one asset share
/// one base/last-applied record.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn apply_expression_materials(
    lifecycle: Res<AvatarLifecycle>,
    mut roots: Query<(&ExpressionEntityMap, &mut AvatarMaterialExpressionState)>,
    expressions: Query<(
        &Name,
        &Transform,
        &ExpressionOverrideSettings,
        Option<&ExpressionOverride>,
        Option<&BinaryExpression>,
        Option<&ExpressionMaterialBinds>,
    )>,
    parents: Query<&ChildOf>,
    meshes: Query<(
        Entity,
        &VrmMaterialIndex,
        Option<&MeshMaterial3d<MToonMaterial>>,
        Option<&MeshMaterial3d<StandardMaterial>>,
        Option<&crate::look::RichMtoonSwap>,
        Option<&crate::look::RichStandardSwap>,
    )>,
    mtoon_assets: Option<ResMut<Assets<MToonMaterial>>>,
    standard_assets: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let (Some(mut mtoon_assets), Some(mut standard_assets)) = (mtoon_assets, standard_assets)
    else {
        return;
    };
    if lifecycle.state() != AvatarLifecycleState::Ready {
        return;
    }
    let Some(root) = lifecycle.active_root() else {
        return;
    };
    let Ok((map, mut state)) = roots.get_mut(root) else {
        return;
    };
    let is_descendant = |entity: Entity| crate::binding::is_descendant(entity, root, &parents);

    struct ExpressionEntry<'a> {
        raw_weight: f32,
        is_binary: bool,
        category: ExpressionCategory,
        binds: Option<&'a ExpressionMaterialBinds>,
    }
    let mut entries: Vec<ExpressionEntry<'_>> = Vec::with_capacity(map.0.len());
    let mut mouth_rate = 0.0_f32;
    let mut blink_rate = 0.0_f32;
    let mut look_at_rate = 0.0_f32;
    for (_name, entity) in map.0.iter() {
        let Ok((expression_name, transform, settings, override_weight, binary, binds)) =
            expressions.get(*entity)
        else {
            continue;
        };
        let raw_weight = override_weight.map_or(transform.translation.x, |w| w.0);
        let is_binary = binary.is_some();
        let output_weight = output_weight(raw_weight, is_binary);
        let category = category_of(expression_name.as_str());
        mouth_rate += settings.override_mouth.rate(output_weight);
        blink_rate += settings.override_blink.rate(output_weight);
        look_at_rate += settings.override_look_at.rate(output_weight);
        entries.push(ExpressionEntry {
            raw_weight,
            is_binary,
            category,
            binds: binds.filter(|binds| !binds.is_empty()),
        });
    }
    let mouth_mul = 1.0 - mouth_rate.clamp(0.0, 1.0);
    let blink_mul = 1.0 - blink_rate.clamp(0.0, 1.0);
    let look_at_mul = 1.0 - look_at_rate.clamp(0.0, 1.0);

    // One weighted bind set per expression. `evaluate_material_values`
    // selects the binds that target the material it is evaluating, so a set
    // must be listed exactly once; listing it once per bind in the set would
    // apply the whole set once per bind.
    let mut weighted: Vec<(&ExpressionMaterialBinds, f32)> = Vec::new();
    for entry in &entries {
        let Some(binds) = entry.binds else {
            continue;
        };
        let multiplier = match entry.category {
            ExpressionCategory::Mouth => mouth_mul,
            ExpressionCategory::Blink => blink_mul,
            ExpressionCategory::LookAt => look_at_mul,
            ExpressionCategory::Other => 1.0,
        };
        let final_weight =
            effective_expression_weight(entry.raw_weight, entry.is_binary, multiplier);
        if final_weight > 0.0 {
            weighted.push((binds, final_weight));
        }
    }

    let mut visited_mtoon: HashSet<AssetId<MToonMaterial>> = HashSet::new();
    let mut visited_standard: HashSet<AssetId<StandardMaterial>> = HashSet::new();
    for (entity, index, mtoon_handle, standard_handle, rich_mtoon_swap, rich_standard_swap) in
        meshes.iter()
    {
        if !is_descendant(entity) {
            continue;
        }
        // While the look is ON the mesh renders the app-side Rich material,
        // so only the swap component still carries the native handle. The
        // native asset stays the owner of the author/current expression
        // values, so the writer follows that handle too.
        let mtoon_id = mtoon_handle
            .map(|handle| handle.id())
            .or_else(|| rich_mtoon_swap.map(|swap| swap.native.id()));
        if let Some(id) = mtoon_id {
            if !visited_mtoon.insert(id) {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(slot) = state.mtoon.entry(id) {
                // First sight of this asset: capture the expression-free base
                // before any write. The MToon pass runs during loading, well
                // before expressions can apply, so the asset is pristine.
                let Some(base) = mtoon_assets.get(id).map(VrmMaterialBaseValues::from_mtoon) else {
                    continue;
                };
                slot.insert(MaterialExpressionState {
                    material_index: index.0,
                    base,
                    applied: base,
                });
            }
            let Some(material_state) = state.mtoon.get_mut(&id) else {
                continue;
            };
            let evaluated = evaluate_material_values(
                &material_state.base,
                material_state.material_index,
                &weighted,
            );
            if material_state.applied == evaluated {
                continue;
            }
            if let Some(mut material) = mtoon_assets.get_mut(id) {
                material.base_color = Color::LinearRgba(evaluated.base_color);
                material.emissive = evaluated.emissive;
                material.shade.color = evaluated.shade_color;
                material.rim_lighting.color = evaluated.rim_color;
                material.outline.color = evaluated.outline_color;
                material.uv_transform = evaluated.uv_transform;
            }
            material_state.applied = evaluated;
        } else {
            let standard_id = standard_handle
                .map(|handle| handle.id())
                .or_else(|| rich_standard_swap.map(|swap| swap.native.id()));
            let Some(id) = standard_id else {
                continue;
            };
            if !visited_standard.insert(id) {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(slot) = state.standard.entry(id) {
                let Some(base) = standard_assets
                    .get(id)
                    .map(VrmMaterialBaseValues::from_standard)
                else {
                    continue;
                };
                slot.insert(MaterialExpressionState {
                    material_index: index.0,
                    base,
                    applied: base,
                });
            }
            let Some(material_state) = state.standard.get_mut(&id) else {
                continue;
            };
            let evaluated = evaluate_material_values(
                &material_state.base,
                material_state.material_index,
                &weighted,
            );
            if material_state.applied == evaluated {
                continue;
            }
            if let Some(mut material) = standard_assets.get_mut(id) {
                material.base_color = Color::LinearRgba(evaluated.base_color);
                material.emissive = evaluated.emissive;
                material.uv_transform = evaluated.uv_transform;
            }
            material_state.applied = evaluated;
        }
    }
}

/// Restores expression-written material fields while the active avatar is
/// unloading, before the existing despawn drops the state.
///
/// The base and last-applied values are owned by the avatar root, but the
/// expression values were written into shared `Assets`: a strong handle held
/// elsewhere keeps the changed values alive after the root is gone, and a
/// later root reusing that asset would capture them as its expression-free
/// base. This system restores only the assets the writer actually changed
/// (`applied != base`), using the recorded base; it performs no lifecycle
/// transition, no despawn, and no file IO.
pub(crate) fn restore_expression_materials_on_unload(
    lifecycle: Res<AvatarLifecycle>,
    states: Query<&AvatarMaterialExpressionState>,
    mut mtoon_assets: ResMut<Assets<MToonMaterial>>,
    mut standard_assets: ResMut<Assets<StandardMaterial>>,
) {
    if lifecycle.state() != AvatarLifecycleState::Unloading {
        return;
    }
    let Some(root) = lifecycle.active_root() else {
        return;
    };
    let Ok(state) = states.get(root) else {
        return;
    };
    for (id, material_state) in &state.mtoon {
        if material_state.applied == material_state.base {
            continue;
        }
        if let Some(mut material) = mtoon_assets.get_mut(*id) {
            material.base_color = Color::LinearRgba(material_state.base.base_color);
            material.emissive = material_state.base.emissive;
            material.shade.color = material_state.base.shade_color;
            material.rim_lighting.color = material_state.base.rim_color;
            material.outline.color = material_state.base.outline_color;
            material.uv_transform = material_state.base.uv_transform;
        }
    }
    for (id, material_state) in &state.standard {
        if material_state.applied == material_state.base {
            continue;
        }
        if let Some(mut material) = standard_assets.get_mut(*id) {
            material.base_color = Color::LinearRgba(material_state.base.base_color);
            material.emissive = material_state.base.emissive;
            material.uv_transform = material_state.base.uv_transform;
        }
    }
}

/// Evaluates one material's base plus every weighted bind targeting
/// `material_index`. Pure numeric function: `base + Σ (target − base) · weight`.
fn evaluate_material_values(
    base: &VrmMaterialBaseValues,
    material_index: usize,
    weighted: &[(&ExpressionMaterialBinds, f32)],
) -> VrmMaterialBaseValues {
    let mut result = *base;
    let mut weighted_transforms: Vec<(&ExpressionTextureTransformBind, f32)> = Vec::new();
    for (binds, weight) in weighted {
        for bind in &binds.colors {
            if bind.material_index != material_index {
                continue;
            }
            match bind.target {
                MaterialColorTarget::Color => {
                    result.base_color = accumulate_linear(
                        result.base_color,
                        base.base_color,
                        bind.target_value,
                        *weight,
                    );
                }
                MaterialColorTarget::EmissionColor => {
                    result.emissive = accumulate_linear(
                        result.emissive,
                        base.emissive,
                        bind.target_value,
                        *weight,
                    );
                }
                MaterialColorTarget::ShadeColor => {
                    result.shade_color = accumulate_linear(
                        result.shade_color,
                        base.shade_color,
                        bind.target_value,
                        *weight,
                    );
                }
                MaterialColorTarget::RimColor => {
                    result.rim_color = accumulate_linear(
                        result.rim_color,
                        base.rim_color,
                        bind.target_value,
                        *weight,
                    );
                }
                MaterialColorTarget::OutlineColor => {
                    result.outline_color = accumulate_linear(
                        result.outline_color,
                        base.outline_color,
                        bind.target_value,
                        *weight,
                    );
                }
            }
        }
        for bind in &binds.transforms {
            if bind.material_index == material_index {
                weighted_transforms.push((bind, *weight));
            }
        }
    }
    result.uv_transform = blend_expression_uv(base.uv_transform, &weighted_transforms);
    result
}

fn accumulate_linear(
    current: LinearRgba,
    base: LinearRgba,
    target: LinearRgba,
    weight: f32,
) -> LinearRgba {
    LinearRgba::new(
        current.red + (target.red - base.red) * weight,
        current.green + (target.green - base.green) * weight,
        current.blue + (target.blue - base.blue) * weight,
        current.alpha + (target.alpha - base.alpha) * weight,
    )
}

/// Interpolates the base UV transform toward each bind's scale/offset by its
/// weight, preserving the base rotation.
fn blend_expression_uv(
    base: Affine2,
    weighted_binds: &[(&ExpressionTextureTransformBind, f32)],
) -> Affine2 {
    let x_axis = base.matrix2.x_axis;
    let rotation = x_axis.y.atan2(x_axis.x);
    let determinant_sign = if base.matrix2.determinant() < 0.0 {
        -1.0
    } else {
        1.0
    };
    let base_scale = Vec2::new(
        x_axis.length(),
        base.matrix2.y_axis.length() * determinant_sign,
    );
    let base_offset = base.translation;
    let mut scale = base_scale;
    let mut offset = base_offset;
    for (bind, weight) in weighted_binds {
        scale += (bind.scale - base_scale) * *weight;
        offset += (bind.offset - base_offset) * *weight;
    }
    Affine2::from_scale_angle_translation(scale, rotation, offset)
}

#[cfg(test)]
mod tests {
    // Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

    use super::*;
    use crate::expression::source::SourceExpressionEntry;

    fn base() -> VrmMaterialBaseValues {
        VrmMaterialBaseValues {
            base_color: LinearRgba::new(0.2, 0.3, 0.4, 1.0),
            emissive: LinearRgba::BLACK,
            shade_color: LinearRgba::new(0.5, 0.5, 0.5, 1.0),
            rim_color: LinearRgba::BLACK,
            outline_color: LinearRgba::new(0.1, 0.1, 0.1, 1.0),
            uv_transform: Affine2::from_scale_angle_translation(Vec2::ONE, 0.0, Vec2::ZERO),
        }
    }

    fn color_bind(
        index: usize,
        target: MaterialColorTarget,
        value: [f32; 4],
    ) -> ExpressionMaterialColorBind {
        ExpressionMaterialColorBind {
            material_index: index,
            target,
            target_value: LinearRgba::new(value[0], value[1], value[2], value[3]),
        }
    }

    #[test]
    fn full_weight_reaches_the_target_color_exactly() {
        let binds = ExpressionMaterialBinds {
            colors: vec![color_bind(
                0,
                MaterialColorTarget::Color,
                [0.8, 0.2, 0.2, 1.0],
            )],
            transforms: Vec::new(),
        };
        let evaluated = evaluate_material_values(&base(), 0, &[(&binds, 1.0)]);
        assert_eq!(evaluated.base_color, LinearRgba::new(0.8, 0.2, 0.2, 1.0));
        // Non-target channels stay at base.
        assert_eq!(evaluated.shade_color, base().shade_color);
        assert_eq!(evaluated.uv_transform, base().uv_transform);
    }

    #[test]
    fn zero_weight_restores_the_base_exactly() {
        let binds = ExpressionMaterialBinds {
            colors: vec![color_bind(
                0,
                MaterialColorTarget::Color,
                [0.8, 0.2, 0.2, 1.0],
            )],
            transforms: Vec::new(),
        };
        let full = evaluate_material_values(&base(), 0, &[(&binds, 1.0)]);
        let released = evaluate_material_values(&base(), 0, &[(&binds, 0.0)]);
        assert_eq!(released, base());
        // Released evaluation is not "full weight minus something": it is
        // recomputed from base, so repeated 1.0 -> 0.0 cycles never drift.
        let cycled = evaluate_material_values(&base(), 0, &[(&binds, 0.0)]);
        assert_eq!(cycled, released);
        assert_ne!(full, released);
    }

    #[test]
    fn partial_weight_interpolates_between_base_and_target() {
        let binds = ExpressionMaterialBinds {
            colors: vec![color_bind(
                0,
                MaterialColorTarget::ShadeColor,
                [1.0, 0.0, 0.0, 1.0],
            )],
            transforms: Vec::new(),
        };
        let evaluated = evaluate_material_values(&base(), 0, &[(&binds, 0.25)]);
        assert!((evaluated.shade_color.red - 0.625).abs() < 1.0e-6);
    }

    #[test]
    fn one_bind_set_is_applied_once_even_with_color_shade_and_uv_binds() {
        // A single expression changing color, shade and UV of the same
        // material must be applied once, not once per bind in the set:
        // base red 0.2 -> target 0.8 at weight 0.5 is 0.5 (not 0.8).
        let binds = ExpressionMaterialBinds {
            colors: vec![
                color_bind(0, MaterialColorTarget::Color, [0.8, 0.2, 0.2, 1.0]),
                color_bind(0, MaterialColorTarget::ShadeColor, [0.8, 0.5, 0.5, 1.0]),
            ],
            transforms: vec![ExpressionTextureTransformBind {
                material_index: 0,
                scale: Vec2::new(2.0, 2.0),
                offset: Vec2::new(0.25, 0.0),
            }],
        };
        let evaluated = evaluate_material_values(&base(), 0, &[(&binds, 0.5)]);
        assert!((evaluated.base_color.red - 0.5).abs() < 1.0e-6);
        assert!((evaluated.shade_color.red - 0.65).abs() < 1.0e-6);
        let scale = evaluated.uv_transform.matrix2.x_axis.length();
        assert!((scale - 1.5).abs() < 1.0e-6);
        assert!((evaluated.uv_transform.translation.x - 0.125).abs() < 1.0e-6);
    }

    #[test]
    fn binds_for_other_materials_do_not_leak() {
        let binds = ExpressionMaterialBinds {
            colors: vec![color_bind(
                1,
                MaterialColorTarget::Color,
                [1.0, 1.0, 1.0, 1.0],
            )],
            transforms: vec![ExpressionTextureTransformBind {
                material_index: 1,
                scale: Vec2::new(3.0, 3.0),
                offset: Vec2::ONE,
            }],
        };
        let evaluated = evaluate_material_values(&base(), 0, &[(&binds, 1.0)]);
        assert_eq!(evaluated, base());
    }

    #[test]
    fn texture_transform_blend_scales_linearly_and_restores() {
        let bind = ExpressionTextureTransformBind {
            material_index: 0,
            scale: Vec2::new(2.0, 2.0),
            offset: Vec2::new(0.25, 0.5),
        };
        let binds = ExpressionMaterialBinds {
            colors: Vec::new(),
            transforms: vec![bind],
        };
        let half = evaluate_material_values(&base(), 0, &[(&binds, 0.5)]);
        let scale = half.uv_transform.matrix2.x_axis.length();
        assert!((scale - 1.5).abs() < 1.0e-6);
        assert!((half.uv_transform.translation.x - 0.125).abs() < 1.0e-6);
        let released = evaluate_material_values(&base(), 0, &[(&binds, 0.0)]);
        assert_eq!(released.uv_transform, base().uv_transform);
    }

    #[test]
    fn binary_weights_snap_at_one_half() {
        assert_eq!(output_weight(0.3, true), 0.0);
        assert_eq!(output_weight(0.7, true), 1.0);
        assert_eq!(output_weight(0.3, false), 0.3);
    }

    #[test]
    fn effective_weight_matches_the_upstream_binary_and_suppression_order() {
        // Binary entries snap at 0.5 and are fully suppressed by any category
        // multiplier below 1, exactly like upstream `bind_expressions`:
        // blink=1 with a 0.25 suppression rate yields 0 for the morph pass
        // and therefore 0 for the material/UV pass here.
        assert_eq!(effective_expression_weight(1.0, true, 0.25), 0.0);
        assert_eq!(effective_expression_weight(0.4, true, 1.0), 0.0);
        assert_eq!(effective_expression_weight(0.6, true, 1.0), 1.0);
        // Non-binary entries keep the ordinary multiplication.
        assert_eq!(effective_expression_weight(1.0, false, 0.25), 0.25);
        assert_eq!(effective_expression_weight(0.5, false, 1.0), 0.5);
        assert_eq!(effective_expression_weight(1.5, false, 1.0), 1.0);
    }

    #[test]
    fn categories_match_the_upstream_name_table() {
        assert_eq!(category_of("aa"), ExpressionCategory::Mouth);
        assert_eq!(category_of("blinkRight"), ExpressionCategory::Blink);
        assert_eq!(category_of("lookUp"), ExpressionCategory::LookAt);
        assert_eq!(category_of("happy"), ExpressionCategory::Other);
        assert_eq!(category_of("custom_smile"), ExpressionCategory::Other);
    }

    #[test]
    fn binds_are_built_from_source_facts_by_name() {
        let facts = SourceExpressions {
            entries: vec![SourceExpressionEntry {
                name: "cheek".into(),
                declared_as_preset: false,
                is_binary: false,
                override_mouth: "none".into(),
                override_blink: "none".into(),
                override_look_at: "none".into(),
                morph_binds: Vec::new(),
                material_color_binds: vec![crate::expression::source::SourceMaterialColorBind {
                    material: 2,
                    target: Some(MaterialColorTarget::Color),
                    target_value: [0.9, 0.4, 0.4, 1.0],
                }],
                texture_transform_binds: Vec::new(),
            }],
        };
        let binds = ExpressionMaterialBinds::from_source(&facts, "cheek");
        assert_eq!(binds.colors.len(), 1);
        assert_eq!(binds.colors[0].material_index, 2);
        assert!(ExpressionMaterialBinds::from_source(&facts, "missing").is_empty());
    }

    use bevy::platform::collections::HashMap as BevyHashMap;

    use bevy_vrm1::prelude::{ExpressionOverrideType, VrmExpression};

    fn writer_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
            .init_asset::<MToonMaterial>()
            .init_asset::<StandardMaterial>()
            .init_resource::<AvatarLifecycle>()
            .add_systems(
                Update,
                (
                    apply_expression_materials,
                    restore_expression_materials_on_unload,
                    crate::unload::despawn_unloading_avatar,
                )
                    .chain(),
            );
        app
    }

    fn none_settings() -> ExpressionOverrideSettings {
        ExpressionOverrideSettings {
            override_mouth: ExpressionOverrideType::None,
            override_blink: ExpressionOverrideType::None,
            override_look_at: ExpressionOverrideType::None,
        }
    }

    fn spawn_expression(
        app: &mut App,
        name: &'static str,
        settings: ExpressionOverrideSettings,
        binds: ExpressionMaterialBinds,
    ) -> Entity {
        app.world_mut()
            .spawn((Name::new(name), Transform::default(), settings, binds))
            .id()
    }

    fn enter_ready(app: &mut App, root: Entity) {
        let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
    }

    fn expression_root(app: &mut App, entries: &[(&str, Entity)]) -> Entity {
        let map = BevyHashMap::from_iter(
            entries
                .iter()
                .map(|(name, entity)| (VrmExpression::from(*name), *entity)),
        );
        app.world_mut()
            .spawn((
                ExpressionEntityMap(map),
                AvatarMaterialExpressionState::default(),
            ))
            .id()
    }

    #[test]
    fn writer_applies_one_expressions_color_shade_and_uv_binds_once_each() {
        let mut app = writer_app();
        let material_handle = {
            let mut materials = app.world_mut().resource_mut::<Assets<MToonMaterial>>();
            materials.add(MToonMaterial {
                base_color: Color::LinearRgba(LinearRgba::new(0.2, 0.3, 0.4, 1.0)),
                shade: bevy_vrm1::prelude::Shade {
                    color: LinearRgba::new(0.5, 0.5, 0.5, 1.0),
                    ..bevy_vrm1::prelude::Shade::default()
                },
                ..MToonMaterial::default()
            })
        };
        let expression = spawn_expression(
            &mut app,
            "cheek",
            none_settings(),
            ExpressionMaterialBinds {
                colors: vec![
                    color_bind(0, MaterialColorTarget::Color, [0.8, 0.2, 0.2, 1.0]),
                    color_bind(0, MaterialColorTarget::ShadeColor, [0.8, 0.5, 0.5, 1.0]),
                ],
                transforms: vec![ExpressionTextureTransformBind {
                    material_index: 0,
                    scale: Vec2::new(2.0, 2.0),
                    offset: Vec2::new(0.25, 0.0),
                }],
            },
        );
        let root = expression_root(&mut app, &[("cheek", expression)]);
        enter_ready(&mut app, root);
        app.world_mut().spawn((
            VrmMaterialIndex(0),
            ChildOf(root),
            MeshMaterial3d(material_handle.clone()),
        ));

        // Half weight: the color component is exactly halfway (0.5), not at
        // the target 0.8 or a double-applied value.
        app.world_mut()
            .entity_mut(expression)
            .insert(ExpressionOverride(0.5));
        app.update();
        let written = app
            .world()
            .resource::<Assets<MToonMaterial>>()
            .get(material_handle.id())
            .unwrap();
        assert!(
            (written.base_color.to_linear().red - 0.5).abs() < 1.0e-6,
            "color bind must be applied once, got {:?}",
            written.base_color
        );
        assert!((written.shade.color.red - 0.65).abs() < 1.0e-6);
        let scale = written.uv_transform.matrix2.x_axis.length();
        assert!((scale - 1.5).abs() < 1.0e-6);
        assert!((written.uv_transform.translation.x - 0.125).abs() < 1.0e-6);

        // Releasing the expression restores the author's base exactly.
        app.world_mut()
            .entity_mut(expression)
            .insert(ExpressionOverride(0.0));
        app.update();
        app.update();
        let restored = app
            .world()
            .resource::<Assets<MToonMaterial>>()
            .get(material_handle.id())
            .unwrap();
        assert_eq!(
            restored.base_color.to_linear(),
            LinearRgba::new(0.2, 0.3, 0.4, 1.0)
        );
        assert_eq!(restored.shade.color, LinearRgba::new(0.5, 0.5, 0.5, 1.0));
        assert_eq!(
            restored.uv_transform,
            Affine2::from_scale_angle_translation(Vec2::ONE, 0.0, Vec2::ZERO)
        );

        // The base and last-applied values are owned by the asset ID on the
        // avatar root.
        let state = app
            .world()
            .get::<AvatarMaterialExpressionState>(root)
            .unwrap();
        let material_state = state
            .mtoon
            .get(&material_handle.id())
            .expect("state captured on first apply");
        assert_eq!(material_state.material_index, 0);
        assert_eq!(
            material_state.base.base_color,
            LinearRgba::new(0.2, 0.3, 0.4, 1.0)
        );
        assert_eq!(material_state.applied, material_state.base);
    }

    #[test]
    fn writer_shares_one_base_per_standard_material_asset_across_meshes() {
        let mut app = writer_app();
        let material_handle = {
            let mut materials = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
            materials.add(StandardMaterial {
                base_color: Color::LinearRgba(LinearRgba::new(0.2, 0.3, 0.4, 1.0)),
                ..StandardMaterial::default()
            })
        };
        let expression = spawn_expression(
            &mut app,
            "cheek",
            none_settings(),
            ExpressionMaterialBinds {
                colors: vec![color_bind(
                    0,
                    MaterialColorTarget::Color,
                    [0.8, 0.2, 0.2, 1.0],
                )],
                transforms: Vec::new(),
            },
        );
        let root = expression_root(&mut app, &[("cheek", expression)]);
        enter_ready(&mut app, root);
        let mesh_a = app
            .world_mut()
            .spawn((
                VrmMaterialIndex(0),
                ChildOf(root),
                MeshMaterial3d(material_handle.clone()),
            ))
            .id();
        let mesh_b = app
            .world_mut()
            .spawn((
                VrmMaterialIndex(0),
                ChildOf(root),
                MeshMaterial3d(material_handle.clone()),
            ))
            .id();
        assert_ne!(mesh_a, mesh_b);

        // Start already at half weight: a per-mesh base would make the second
        // mesh capture the first mesh's 0.5 write as its own base and write
        // 0.65.
        app.world_mut()
            .entity_mut(expression)
            .insert(ExpressionOverride(0.5));
        app.update();
        let written = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(material_handle.id())
            .unwrap()
            .base_color
            .to_linear();
        assert!(
            (written.red - 0.5).abs() < 1.0e-6,
            "shared asset must be evaluated once from the author base, got {written:?}"
        );

        // Both meshes return to the author's base once the weight is 0.
        app.world_mut()
            .entity_mut(expression)
            .insert(ExpressionOverride(0.0));
        app.update();
        app.update();
        let restored = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(material_handle.id())
            .unwrap()
            .base_color
            .to_linear();
        assert_eq!(restored, LinearRgba::new(0.2, 0.3, 0.4, 1.0));

        let state = app
            .world()
            .get::<AvatarMaterialExpressionState>(root)
            .unwrap();
        assert_eq!(state.standard.len(), 1, "one state per asset, not per mesh");
        let material_state = state.standard.get(&material_handle.id()).unwrap();
        assert_eq!(
            material_state.base.base_color,
            LinearRgba::new(0.2, 0.3, 0.4, 1.0)
        );
        assert_eq!(material_state.applied, material_state.base);
    }

    #[test]
    fn writer_restores_a_shared_material_on_unload_before_a_new_root_reuses_it() {
        let mut app = writer_app();
        let material_handle = {
            let mut materials = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
            materials.add(StandardMaterial {
                base_color: Color::LinearRgba(LinearRgba::new(0.2, 0.3, 0.4, 1.0)),
                ..StandardMaterial::default()
            })
        };
        let expression = spawn_expression(
            &mut app,
            "cheek",
            none_settings(),
            ExpressionMaterialBinds {
                colors: vec![color_bind(
                    0,
                    MaterialColorTarget::Color,
                    [0.8, 0.2, 0.2, 1.0],
                )],
                transforms: vec![ExpressionTextureTransformBind {
                    material_index: 0,
                    scale: Vec2::new(2.0, 2.0),
                    offset: Vec2::new(0.25, 0.0),
                }],
            },
        );
        let root = expression_root(&mut app, &[("cheek", expression)]);
        enter_ready(&mut app, root);
        app.world_mut().spawn((
            VrmMaterialIndex(0),
            ChildOf(root),
            MeshMaterial3d(material_handle.clone()),
        ));

        // Half weight is applied and never released before the unload; the
        // strong handle held here outlives the root.
        app.world_mut()
            .entity_mut(expression)
            .insert(ExpressionOverride(0.5));
        app.update();
        let applied = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(material_handle.id())
            .unwrap();
        assert!((applied.base_color.to_linear().red - 0.5).abs() < 1.0e-6);
        let applied_scale = applied.uv_transform.matrix2.x_axis.length();
        assert!((applied_scale - 1.5).abs() < 1.0e-6);

        // Existing unload path: the restore system returns the asset to the
        // author's base before the despawn drops the root state.
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_unload()
            .unwrap();
        app.update();

        assert!(
            !app.world().entities().contains(root),
            "the old root is despawned by the existing unload"
        );
        let restored = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(material_handle.id())
            .unwrap();
        assert_eq!(
            restored.base_color.to_linear(),
            LinearRgba::new(0.2, 0.3, 0.4, 1.0),
            "unload must return the shared asset to the author's base"
        );
        assert_eq!(
            restored.uv_transform,
            Affine2::from_scale_angle_translation(Vec2::ONE, 0.0, Vec2::ZERO)
        );

        // Reuse the same asset in a new root at weight 0: the captured base
        // must be the author's value, not the stale 0.5 write.
        let expression_b = spawn_expression(
            &mut app,
            "cheek",
            none_settings(),
            ExpressionMaterialBinds {
                colors: vec![color_bind(
                    0,
                    MaterialColorTarget::Color,
                    [0.8, 0.2, 0.2, 1.0],
                )],
                transforms: vec![ExpressionTextureTransformBind {
                    material_index: 0,
                    scale: Vec2::new(2.0, 2.0),
                    offset: Vec2::new(0.25, 0.0),
                }],
            },
        );
        let root_b = expression_root(&mut app, &[("cheek", expression_b)]);
        enter_ready(&mut app, root_b);
        app.world_mut().spawn((
            VrmMaterialIndex(0),
            ChildOf(root_b),
            MeshMaterial3d(material_handle.clone()),
        ));
        app.update();

        let reused = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(material_handle.id())
            .unwrap();
        assert_eq!(
            reused.base_color.to_linear(),
            LinearRgba::new(0.2, 0.3, 0.4, 1.0)
        );
        assert_eq!(
            reused.uv_transform,
            Affine2::from_scale_angle_translation(Vec2::ONE, 0.0, Vec2::ZERO)
        );
        let state = app
            .world()
            .get::<AvatarMaterialExpressionState>(root_b)
            .unwrap();
        let captured_base = state.standard.get(&material_handle.id()).unwrap().base;
        assert_eq!(
            captured_base.base_color,
            LinearRgba::new(0.2, 0.3, 0.4, 1.0),
            "the new root must capture the author's base, not the stale write"
        );
        assert_eq!(
            captured_base.uv_transform,
            Affine2::from_scale_angle_translation(Vec2::ONE, 0.0, Vec2::ZERO)
        );
    }

    #[test]
    fn writer_suppresses_a_binary_expressions_material_and_uv_binds_like_upstream() {
        let mut app = writer_app();
        let material_handle = {
            let mut materials = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
            materials.add(StandardMaterial {
                base_color: Color::LinearRgba(LinearRgba::new(0.2, 0.3, 0.4, 1.0)),
                ..StandardMaterial::default()
            })
        };
        let blink = spawn_expression(
            &mut app,
            "blink",
            none_settings(),
            ExpressionMaterialBinds {
                colors: vec![color_bind(
                    0,
                    MaterialColorTarget::Color,
                    [0.8, 0.2, 0.2, 1.0],
                )],
                transforms: vec![ExpressionTextureTransformBind {
                    material_index: 0,
                    scale: Vec2::new(2.0, 2.0),
                    offset: Vec2::ZERO,
                }],
            },
        );
        app.world_mut().entity_mut(blink).insert(BinaryExpression);
        let mut blend_settings = none_settings();
        blend_settings.override_blink = ExpressionOverrideType::Blend;
        let happy = spawn_expression(
            &mut app,
            "happy",
            blend_settings,
            ExpressionMaterialBinds::default(),
        );
        let root = expression_root(&mut app, &[("blink", blink), ("happy", happy)]);
        enter_ready(&mut app, root);
        app.world_mut().spawn((
            VrmMaterialIndex(0),
            ChildOf(root),
            MeshMaterial3d(material_handle.clone()),
        ));

        // binary blink=1 with a 0.25 blink suppression rate: upstream's morph
        // pass yields 0, and the material/UV pass must yield 0 as well.
        app.world_mut()
            .entity_mut(blink)
            .insert(ExpressionOverride(1.0));
        app.world_mut()
            .entity_mut(happy)
            .insert(ExpressionOverride(0.25));
        app.update();

        let written = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(material_handle.id())
            .unwrap();
        assert_eq!(
            written.base_color.to_linear(),
            LinearRgba::new(0.2, 0.3, 0.4, 1.0),
            "a suppressed binary expression must not write its binds"
        );
        assert_eq!(
            written.uv_transform,
            Affine2::from_scale_angle_translation(Vec2::ONE, 0.0, Vec2::ZERO)
        );
    }
}
