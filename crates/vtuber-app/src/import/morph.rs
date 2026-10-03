//! Morph target reduction and reference remapping for over-limit meshes.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use vtuber_avatar::glb::Glb;

use super::{MAX_MORPH_TARGETS, ModelImportError};

/// Rewrites a GLB so that no mesh carries more than [`MAX_MORPH_TARGETS`]
/// morph targets, which is the hard limit the Bevy runtime imposes at load.
///
/// Morph targets with no VRM expression bind or nonzero default weight are dropped from
/// meshes above the limit, and bind indices are remapped to the reduced
/// target arrays. Returns `Ok(None)` when the bytes need no rewrite or cannot be
/// rewritten safely: GLBs without an excess mesh, models
/// that animate morph weights, and meshes whose referenced bind set alone
/// exceeds the limit together with the nonzero defaults. Invalid containers return `Err`.
pub fn normalize_vrm_morph_targets(bytes: &[u8]) -> Result<Option<Vec<u8>>, ModelImportError> {
    let mut glb =
        Glb::parse(bytes).map_err(|error| ModelImportError::GlbParse(error.to_string()))?;
    if has_morph_weight_animation(&glb.document) {
        return Ok(None);
    }
    let references = collect_morph_references(&glb.document);
    let Some(plan) = plan_morph_reduction(&glb.document, &references) else {
        return Ok(None);
    };
    apply_morph_reduction(&mut glb.document, &plan);
    glb.to_vec()
        .map(Some)
        .map_err(|error| ModelImportError::GlbParse(error.to_string()))
}

/// One over-limit mesh's keep set, keyed by glTF mesh index.
struct MorphReductionPlan {
    /// Target count before reduction, used to detect coherent name/weight arrays.
    target_count: usize,
    /// Old morph target index to reduced index.
    remap: BTreeMap<usize, usize>,
}

fn has_morph_weight_animation(root: &Value) -> bool {
    root.get("animations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|animation| {
            animation
                .get("channels")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .any(|channel| {
            channel
                .get("target")
                .and_then(|target| target.get("path"))
                .and_then(Value::as_str)
                == Some("weights")
        })
}

/// Collects morph targets needed by VRM expression binds and authored defaults,
/// keyed by glTF mesh index. VRM 0.x binds are mesh-indexed; VRM 1.0 binds
/// are node-indexed and resolved through the node's mesh.
fn collect_morph_references(root: &Value) -> BTreeMap<usize, BTreeSet<usize>> {
    let mut references: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut record = |mesh: usize, index: usize| {
        references.entry(mesh).or_default().insert(index);
    };

    if let Some(groups) = root
        .get("extensions")
        .and_then(|extensions| extensions.get("VRM"))
        .and_then(|vrm| vrm.get("blendShapeMaster"))
        .and_then(|master| master.get("blendShapeGroups"))
        .and_then(Value::as_array)
    {
        for bind in groups
            .iter()
            .filter_map(|group| group.get("binds"))
            .filter_map(Value::as_array)
            .flatten()
        {
            let mesh = bind.get("mesh").and_then(Value::as_u64);
            let index = bind.get("index").and_then(Value::as_u64);
            if let (Some(mesh), Some(index)) = (mesh, index)
                && let (Ok(mesh), Ok(index)) = (usize::try_from(mesh), usize::try_from(index))
            {
                record(mesh, index);
            }
        }
    }

    for bind in vrm1_morph_target_binds(root) {
        let node = bind.get("node").and_then(Value::as_u64);
        let index = bind.get("index").and_then(Value::as_u64);
        if let (Some(node), Some(index)) = (node, index)
            && let (Ok(node), Ok(index)) = (usize::try_from(node), usize::try_from(index))
            && let Some(mesh) = node_mesh_index(root, node)
        {
            record(mesh, index);
        }
    }
    // Nonzero authored defaults affect the neutral shape even without a VRM bind.
    if let Some(meshes) = root.get("meshes").and_then(Value::as_array) {
        for (mesh, value) in meshes.iter().enumerate() {
            for index in morph_default_indices(value) {
                record(mesh, index);
            }
        }
    }
    if let Some(nodes) = root.get("nodes").and_then(Value::as_array) {
        for node in nodes {
            if let Some(mesh) = node
                .get("mesh")
                .and_then(Value::as_u64)
                .and_then(|mesh| usize::try_from(mesh).ok())
            {
                for index in morph_default_indices(node) {
                    record(mesh, index);
                }
            }
        }
    }
    references
}

fn morph_default_indices(value: &Value) -> impl Iterator<Item = usize> + '_ {
    value
        .get("weights")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter(|(_, weight)| weight.as_f64().is_some_and(|weight| weight != 0.0))
        .map(|(index, _)| index)
}

fn node_mesh_index(root: &Value, node: usize) -> Option<usize> {
    let mesh = root
        .get("nodes")?
        .as_array()?
        .get(node)?
        .get("mesh")?
        .as_u64()?;
    usize::try_from(mesh).ok()
}

/// Iterates VRM 1.0 `morphTargetBinds` entries across the preset and custom
/// expression sections.
fn vrm1_morph_target_binds(root: &Value) -> impl Iterator<Item = &Value> {
    let expressions = root
        .get("extensions")
        .and_then(|extensions| extensions.get("VRMC_vrm"))
        .and_then(|vrm| vrm.get("expressions"));
    ["preset", "custom"]
        .into_iter()
        .filter_map(move |section| expressions.and_then(|value| value.get(section)))
        .filter_map(Value::as_object)
        .flat_map(|section| section.values())
        .filter_map(|expression| expression.get("morphTargetBinds"))
        .filter_map(Value::as_array)
        .flatten()
}

fn plan_morph_reduction(
    root: &Value,
    references: &BTreeMap<usize, BTreeSet<usize>>,
) -> Option<BTreeMap<usize, MorphReductionPlan>> {
    let meshes = root.get("meshes")?.as_array()?;
    let mut plan = BTreeMap::new();
    for (mesh_index, mesh) in meshes.iter().enumerate() {
        let target_count = morph_target_count(mesh);
        if target_count <= MAX_MORPH_TARGETS {
            continue;
        }
        let keep: Vec<usize> = references
            .get(&mesh_index)
            .into_iter()
            .flatten()
            .copied()
            .filter(|index| *index < target_count)
            .collect();
        if keep.len() > MAX_MORPH_TARGETS {
            return None;
        }
        let remap: BTreeMap<usize, usize> = keep
            .iter()
            .enumerate()
            .map(|(reduced, original)| (*original, reduced))
            .collect();
        plan.insert(
            mesh_index,
            MorphReductionPlan {
                target_count,
                remap,
            },
        );
    }
    (!plan.is_empty()).then_some(plan)
}

fn morph_target_count(mesh: &Value) -> usize {
    mesh.get("primitives")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|primitive| primitive.get("targets").and_then(Value::as_array))
        .map(Vec::len)
        .max()
        .unwrap_or(0)
}

pub(super) fn over_limit_morph_target_count(bytes: &[u8]) -> Option<usize> {
    let root = Glb::parse(bytes).ok()?.document;
    root.get("meshes")?
        .as_array()?
        .iter()
        .map(morph_target_count)
        .max()
        .filter(|count| *count > MAX_MORPH_TARGETS)
}

fn apply_morph_reduction(root: &mut Value, plan: &BTreeMap<usize, MorphReductionPlan>) {
    reduce_meshes(root, plan);
    if let Some(nodes) = root.get_mut("nodes").and_then(Value::as_array_mut) {
        for node in nodes {
            let reduction = node
                .get("mesh")
                .and_then(Value::as_u64)
                .and_then(|mesh| usize::try_from(mesh).ok())
                .and_then(|mesh| plan.get(&mesh));
            if let Some(reduction) = reduction
                && let Some(weights) = node.get_mut("weights").and_then(Value::as_array_mut)
            {
                *weights = reduction
                    .remap
                    .keys()
                    .filter_map(|index| weights.get(*index).cloned())
                    .collect();
            }
        }
    }
    remap_legacy_binds(root, plan);
    remap_vrm1_binds(root, plan);
}

fn reduce_meshes(root: &mut Value, plan: &BTreeMap<usize, MorphReductionPlan>) {
    let Some(meshes) = root.get_mut("meshes").and_then(Value::as_array_mut) else {
        return;
    };
    for (mesh_index, mesh) in meshes.iter_mut().enumerate() {
        let Some(reduction) = plan.get(&mesh_index) else {
            continue;
        };
        let Some(object) = mesh.as_object_mut() else {
            continue;
        };
        if let Some(primitives) = object.get_mut("primitives").and_then(Value::as_array_mut) {
            for primitive in primitives {
                if let Some(targets) = primitive.get_mut("targets").and_then(Value::as_array_mut) {
                    *targets = reduction
                        .remap
                        .keys()
                        .filter_map(|index| targets.get(*index).cloned())
                        .collect();
                }
            }
        }
        if let Some(extras) = object.get_mut("extras").and_then(Value::as_object_mut) {
            match extras.get("targetNames").and_then(Value::as_array) {
                Some(names) if names.len() == reduction.target_count => {
                    let reduced = reduction
                        .remap
                        .keys()
                        .filter_map(|index| names.get(*index).cloned())
                        .collect();
                    extras.insert("targetNames".into(), Value::Array(reduced));
                }
                Some(_) => {
                    extras.remove("targetNames");
                }
                None => {}
            }
        }
        match object.get("weights").and_then(Value::as_array) {
            Some(weights) if weights.len() == reduction.target_count => {
                let reduced = reduction
                    .remap
                    .keys()
                    .filter_map(|index| weights.get(*index).cloned())
                    .collect();
                object.insert("weights".into(), Value::Array(reduced));
            }
            Some(_) => {
                object.remove("weights");
            }
            None => {}
        }
    }
}

fn remap_legacy_binds(root: &mut Value, plan: &BTreeMap<usize, MorphReductionPlan>) {
    let Some(groups) = root
        .get_mut("extensions")
        .and_then(|extensions| extensions.get_mut("VRM"))
        .and_then(|vrm| vrm.get_mut("blendShapeMaster"))
        .and_then(|master| master.get_mut("blendShapeGroups"))
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for group in groups {
        let Some(binds) = group.get_mut("binds").and_then(Value::as_array_mut) else {
            continue;
        };
        binds.retain_mut(|bind| {
            remap_bind(bind, plan, |bind| bind.get("mesh").and_then(Value::as_u64)).unwrap_or(true)
        });
    }
}

fn remap_vrm1_binds(root: &mut Value, plan: &BTreeMap<usize, MorphReductionPlan>) {
    let node_meshes: BTreeMap<usize, usize> = root
        .get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(node, value)| {
            let mesh = value.get("mesh").and_then(Value::as_u64)?;
            Some((node, usize::try_from(mesh).ok()?))
        })
        .collect();
    let Some(sections) = root
        .get_mut("extensions")
        .and_then(|extensions| extensions.get_mut("VRMC_vrm"))
        .and_then(|vrm| vrm.get_mut("expressions"))
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    for section in ["preset", "custom"] {
        let Some(section) = sections.get_mut(section).and_then(Value::as_object_mut) else {
            continue;
        };
        for expression in section.values_mut() {
            let Some(binds) = expression
                .get_mut("morphTargetBinds")
                .and_then(Value::as_array_mut)
            else {
                continue;
            };
            binds.retain_mut(|bind| {
                remap_bind(bind, plan, |bind| {
                    let node = bind.get("node").and_then(Value::as_u64)?;
                    let node = usize::try_from(node).ok()?;
                    let mesh = node_meshes.get(&node)?;
                    u64::try_from(*mesh).ok()
                })
                .unwrap_or(true)
            });
        }
    }
}

/// Remaps one bind's morph target index. Returns `None` when the bind does
/// not participate in a reduced mesh; `Some(false)` when the bind referenced
/// a dropped morph target and must be removed.
fn remap_bind(
    bind: &mut Value,
    plan: &BTreeMap<usize, MorphReductionPlan>,
    mesh_of: impl Fn(&Value) -> Option<u64>,
) -> Option<bool> {
    let index = bind.get("index").and_then(Value::as_u64)?;
    let mesh = mesh_of(bind)?;
    let mesh = usize::try_from(mesh).ok()?;
    let reduction = plan.get(&mesh)?;
    let index = usize::try_from(index).ok()?;
    let Some(reduced) = reduction.remap.get(&index) else {
        return Some(false);
    };
    if let Some(object) = bind.as_object_mut() {
        object.insert("index".into(), Value::from(*reduced as u64));
    }
    Some(true)
}
