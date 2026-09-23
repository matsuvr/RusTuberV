//! VRM 0.x extension normalization for import-time conversion.
//!
//! Moved from the removed vendored bevy_vrm1 patch (see #91) and retargeted
//! at the unmodified upstream v0.9.3 contract: normalized extensions are
//! serialized back into the managed glTF JSON, so the upstream runtime loads
//! the converted model without any vendored loader or ECS patch.
//!
//! Differences from the vendored shape are upstream-driven, not behavioral:
//! upstream `Expressions` carries a single `preset` map (standard and custom
//! entries share it, standards win on collision) and upstream `VrmPreset`
//! carries morph binds only. Legacy material/texture expression binds cannot
//! be represented upstream and are dropped here; the import diagnostics
//! already report them as unsupported.
use std::collections::BTreeSet;

use anyhow::Context;
use bevy::platform::collections::HashMap;
use bevy_vrm1::prelude::{
    Collider, ColliderGroup, ColliderShape, Expressions, FirstPerson, FirstPersonFlag, Humanoid,
    LookAtProperties, LookAtType, MeshAnnotation, Meta, MorphTargetBind, RangeMap, Sphere, Spring,
    SpringJoint, VRMCSpringBone, VrmNode, VrmPreset, VrmcVrm,
};
use serde_json::Value;

use super::descriptor::{
    VrmCompatibilityWarning, VrmFirstPersonFlag, VrmLookAtType, VrmRuntimeDescriptor,
};

type AppResult<T> = anyhow::Result<T>;
pub(crate) fn normalized_legacy_vrm(
    descriptor: &VrmRuntimeDescriptor,
    legacy: &Value,
    root: &Value,
) -> AppResult<VrmcVrm> {
    let human_bones = descriptor
        .humanoid
        .human_bones
        .iter()
        .map(|(name, node)| (name.clone(), VrmNode { node: *node }))
        .collect::<HashMap<_, _>>();
    let first_person = descriptor
        .first_person
        .as_ref()
        .map(|first_person| FirstPerson {
            mesh_annotations: first_person
                .mesh_annotations
                .iter()
                .map(|annotation| MeshAnnotation {
                    node: annotation.node,
                    first_person_flag: match annotation.flag {
                        VrmFirstPersonFlag::Auto => FirstPersonFlag::Auto,
                        VrmFirstPersonFlag::Both => FirstPersonFlag::Both,
                        VrmFirstPersonFlag::ThirdPersonOnly => FirstPersonFlag::ThirdPersonOnly,
                        VrmFirstPersonFlag::FirstPersonOnly => FirstPersonFlag::FirstPersonOnly,
                    },
                })
                .collect(),
        });
    let look_at = descriptor.look_at.as_ref().map(|look_at| LookAtProperties {
        offset_from_head_bone: look_at.offset_from_head_bone,
        range_map_horizontal_inner: RangeMap {
            input_max_value: look_at.range_map_horizontal_inner.input_max_value,
            output_scale: look_at.range_map_horizontal_inner.output_scale,
        },
        range_map_horizontal_outer: RangeMap {
            input_max_value: look_at.range_map_horizontal_outer.input_max_value,
            output_scale: look_at.range_map_horizontal_outer.output_scale,
        },
        range_map_vertical_down: RangeMap {
            input_max_value: look_at.range_map_vertical_down.input_max_value,
            output_scale: look_at.range_map_vertical_down.output_scale,
        },
        range_map_vertical_up: RangeMap {
            input_max_value: look_at.range_map_vertical_up.input_max_value,
            output_scale: look_at.range_map_vertical_up.output_scale,
        },
        r#type: match look_at.r#type {
            VrmLookAtType::Bone => LookAtType::Bone,
            VrmLookAtType::Expression => LookAtType::Expression,
        },
    });

    Ok(VrmcVrm {
        expressions: normalized_legacy_expressions(
            legacy,
            root,
            &descriptor.compatibility_warnings,
        )?,
        first_person,
        humanoid: Humanoid { human_bones },
        look_at,
        meta: Some(Meta {
            allow_antisocial_or_hate_usage: true,
            allow_excessively_sexual_usage: true,
            allow_excessively_violent_usage: true,
            allow_political_or_religious_usage: true,
            allow_redistribution: true,
            authors: descriptor.meta.authors.clone(),
            avatar_permission: None,
            commercial_usage: None,
            credit_notation: None,
            license_url: descriptor.meta.license_url.clone(),
            modification: None,
            name: descriptor.meta.name.clone(),
            other_license_url: None,
            thumbnail_image: None,
            version: None,
        }),
        spec_version: "1.0".into(),
    })
}
fn normalized_legacy_expressions(
    legacy: &Value,
    root: &Value,
    _compatibility_warnings: &[VrmCompatibilityWarning],
) -> AppResult<Option<Expressions>> {
    let groups = legacy
        .get("blendShapeMaster")
        .and_then(|master| master.get("blendShapeGroups"))
        .and_then(Value::as_array);
    let Some(groups) = groups else {
        return Ok(None);
    };
    // The upstream contract carries a single `preset` map. Standards are
    // collected first so they win the rare collision where a custom author
    // name spells a standard runtime ID; only known VRM 0.x semantics enter
    // as standards, everything else keeps the author name.
    let mut standards = Vec::new();
    let mut customs = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        let Some(name) = normalized_legacy_expression_name(group, group_index) else {
            continue;
        };
        if legacy_expression_is_standard(group) {
            standards.push((name, group_index));
        } else {
            customs.push((name, group_index));
        }
    }
    let mut preset = HashMap::default();
    // Bounds are guaranteed by construction: every stored index comes from
    // enumerating this same `groups` array above. See the AGENTS.md
    // production panic policy.
    #[allow(clippy::indexing_slicing)]
    for (name, group_index) in standards.into_iter().chain(customs.into_iter()) {
        let group = &groups[group_index];
        if preset.contains_key(&name) {
            continue;
        }
        let morph_target_binds = group
            .get("binds")
            .and_then(Value::as_array)
            .map(|binds| {
                let mut seen = BTreeSet::new();
                binds
                    .iter()
                    .enumerate()
                    .map(|(bind_index, bind)| {
                        let path = format!(
                            "VRM.blendShapeMaster.blendShapeGroups[{group_index}].binds[{bind_index}]"
                        );
                        let mesh = required_index(bind, "mesh", &format!("{path}.mesh"))?;
                        let morph_index = required_index(
                            bind,
                            "index",
                            &format!("{path}.index"),
                        )?;
                        validate_mesh_index(root, mesh, &format!("{path}.mesh"))?;
                        validate_morph_target_index(
                            root,
                            mesh,
                            morph_index,
                            &format!("{path}.index"),
                        )?;
                        let weight = required_f32(bind, "weight", &format!("{path}.weight"))?;
                        if !(0.0..=100.0).contains(&weight) {
                            return Err(anyhow::anyhow!(
                                "invalid field {path}.weight: expected 0..=100"
                            ));
                        }
                        let nodes = mesh_instance_nodes(root, mesh, &format!("{path}.mesh"))?;
                        let normalized = nodes
                            .into_iter()
                            .filter_map(|node| {
                                seen.insert((node, morph_index)).then_some(MorphTargetBind {
                                    index: morph_index,
                                    node,
                                    weight: weight / 100.0,
                                })
                            })
                            .collect::<Vec<_>>();
                        Ok(normalized.into_iter())
                    })
                    .collect::<AppResult<Vec<_>>>()
                    .map(|binds| binds.into_iter().flatten().collect())
            })
            .transpose()?;
        // Legacy material/texture binds cannot be represented by the
        // upstream contract and are dropped here; the import diagnostics
        // already report them as unsupported.
        preset.insert(
            name,
            VrmPreset {
                is_binary: group
                    .get("isBinary")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                morph_target_binds,
                override_blink: "none".into(),
                override_look_at: "none".into(),
                override_mouth: "none".into(),
            },
        );
    }
    Ok(Some(Expressions { preset }))
}
/// Maps a known VRM 0.x `presetName` to the VRM 1.0 runtime ID.
///
/// This is the only standard-semantic table. It is applied exclusively to the
/// source `presetName`; author names are never translated, so a custom
/// expression that happens to be named `joy` or `A` keeps its own ID.
pub(crate) fn vrm0_preset_runtime_name(preset_name: &str) -> Option<&'static str> {
    Some(match preset_name {
        "A" | "a" => "aa",
        "I" | "i" => "ih",
        "U" | "u" => "ou",
        "E" | "e" => "ee",
        "O" | "o" => "oh",
        "Blink" | "blink" => "blink",
        "Blink_L" | "blink_l" => "blinkLeft",
        "Blink_R" | "blink_r" => "blinkRight",
        "LookUp" | "lookup" => "lookUp",
        "LookDown" | "lookdown" => "lookDown",
        "LookLeft" | "lookleft" => "lookLeft",
        "LookRight" | "lookright" => "lookRight",
        "Joy" | "joy" => "happy",
        "Angry" | "angry" => "angry",
        "Sorrow" | "sorrow" => "sad",
        "Fun" | "fun" => "relaxed",
        "Neutral" | "neutral" => "neutral",
        _ => return None,
    })
}

/// Returns `true` when a legacy group's `presetName` is a known VRM 0.x
/// semantic. `Unknown`/missing preset names stay custom even when the author
/// name spells a standard preset name.
fn legacy_expression_is_standard(group: &Value) -> bool {
    group
        .get("presetName")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("unknown"))
        .and_then(vrm0_preset_runtime_name)
        .is_some()
}
fn normalized_legacy_expression_name(group: &Value, group_index: usize) -> Option<String> {
    let preset = group
        .get("presetName")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("unknown"));
    if let Some(runtime_name) = preset.and_then(vrm0_preset_runtime_name) {
        return Some(runtime_name.into());
    }
    let name = group
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    Some(
        name.map(str::to_owned)
            .unwrap_or_else(|| format!("custom_{group_index}")),
    )
}
pub(crate) fn normalized_legacy_spring_bone(
    root: &Value,
    legacy: &Value,
) -> AppResult<Option<VRMCSpringBone>> {
    let Some(secondary) = legacy.get("secondaryAnimation") else {
        return Ok(None);
    };
    let mut colliders = Vec::new();
    let mut collider_groups = Vec::new();

    for group in secondary
        .get("colliderGroups")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let group_index = collider_groups.len();
        let node = required_index(
            group,
            "node",
            &format!("VRM.secondaryAnimation.colliderGroups[{group_index}].node"),
        )?;
        validate_node_index(
            root,
            node,
            &format!("VRM.secondaryAnimation.colliderGroups[{group_index}].node"),
        )?;
        let group_colliders = group
            .get("colliders")
            .and_then(Value::as_array)
            .context("VRM.secondaryAnimation.colliderGroups[].colliders must be an array")?
            .iter()
            .enumerate()
            .map(|(collider_index, collider)| {
                let path = format!(
                    "VRM.secondaryAnimation.colliderGroups[{group_index}].colliders[{collider_index}]"
                );
                let offset = required_vector3(collider, "offset", &format!("{path}.offset"))?;
                let radius = required_f32(collider, "radius", &format!("{path}.radius"))?;
                if radius < 0.0 {
                    return Err(anyhow::anyhow!(
                        "invalid field {path}.radius: expected a non-negative number"
                    ));
                }
                let index = colliders.len() as u64;
                colliders.push(Collider {
                    node,
                    shape: ColliderShape::Sphere(
                        Sphere {
                            // VRM 0.x collider offsets are already local to
                            // the target node. The normalized scene is placed
                            // below one Y=pi basis root, so converting this
                            // local value would apply the basis twice. Gravity
                            // is handled separately as an external/world
                            // vector below.
                            offset,
                            radius,
                        },
                    ),
                });
                Ok(index)
            })
            .collect::<AppResult<Vec<_>>>()?;
        collider_groups.push(ColliderGroup {
            name: None,
            colliders: group_colliders,
        });
    }

    let springs = secondary
        .get("boneGroups")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, group)| {
            let roots = group
                .get("bones")
                .and_then(Value::as_array)
                .context(format!(
                    "VRM.secondaryAnimation.boneGroups[{index}].bones must be an array"
                ))?;
            let mut paths = Vec::new();
            for (root_index, root_value) in roots.iter().enumerate() {
                let root_node = root_value.as_u64().and_then(|value| value.try_into().ok())
                    .ok_or_else(|| anyhow::anyhow!(
                        "invalid field VRM.secondaryAnimation.boneGroups[{index}].bones[{root_index}]"
                    ))?;
                validate_node_index(
                    root,
                    root_node,
                    &format!("VRM.secondaryAnimation.boneGroups[{index}].bones[{root_index}]"),
                )?;
                let mut visiting = BTreeSet::new();
                collect_spring_paths(root, root_node, &mut visiting, &mut Vec::new(), &mut paths)?;
            }
            let collider_groups = group
                .get("colliderGroups")
                .and_then(Value::as_array)
                .map(|groups| {
                    groups
                        .iter()
                        .enumerate()
                        .map(|(group_index, value)| {
                            let collider_group = value.as_u64()
                                .and_then(|value| value.try_into().ok())
                                .ok_or_else(|| anyhow::anyhow!(
                                    "invalid field VRM.secondaryAnimation.boneGroups[{index}].colliderGroups[{group_index}]"
                                ))?;
                            (collider_group < collider_groups.len())
                                .then_some(collider_group)
                                .ok_or_else(|| anyhow::anyhow!(
                                    "invalid index VRM.secondaryAnimation.boneGroups[{index}].colliderGroups[{group_index}]={collider_group}"
                                ))
                        })
                        .collect::<AppResult<Vec<_>>>()
                })
                .transpose()?;
            let center = match group.get("center") {
                None => None,
                Some(value) if value.as_i64() == Some(-1) => None,
                Some(value) => {
                    let center = value
                        .as_u64()
                        .and_then(|value| value.try_into().ok())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "invalid field VRM.secondaryAnimation.boneGroups[{index}].center"
                            )
                        })?;
                    validate_node_index(
                        root,
                        center,
                        &format!("VRM.secondaryAnimation.boneGroups[{index}].center"),
                    )?;
                    Some(center)
                }
            };
            let gravity_dir = match group.get("gravityDir") {
                None => [0.0, -1.0, 0.0],
                Some(_) => legacy_gravity_direction(required_vector3(
                    group,
                    "gravityDir",
                    &format!(
                        "VRM.secondaryAnimation.boneGroups[{index}].gravityDir"
                    ),
                )?),
            };
            let mut springs = Vec::new();
            for (path_index, path) in paths.into_iter().enumerate() {
                let joints = path
                    .into_iter()
                    .map(|node| -> AppResult<SpringJoint> { Ok(SpringJoint {
                        node,
                        drag_force: Some(clamped_f32(group, "dragForce", 0.0, 1.0)?),
                        gravity_dir: Some(gravity_dir),
                        gravity_power: Some(non_negative_f32(group, "gravityPower")?),
                        hit_radius: Some(non_negative_f32(group, "hitRadius")?),
                        stiffness: Some(non_negative_f32(
                            group,
                            if group.get("stiffiness").is_some() { "stiffiness" } else { "stiffness" },
                        )?),
                    }) })
                    .collect::<AppResult<Vec<_>>>()?;
                springs.push((path_index, Spring {
                    name: format!("legacy-spring-{index}-{path_index}"),
                    joints,
                    collider_groups: collider_groups.clone(),
                    center,
                }));
            }
            Ok(springs)
        })
        .collect::<AppResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let mut claimed_nodes = BTreeSet::<usize>::new();
    let springs = springs
        .into_iter()
        .filter_map(|(_, mut spring)| {
            spring
                .joints
                .retain(|joint| claimed_nodes.insert(joint.node));
            (!spring.joints.is_empty()).then_some(spring)
        })
        .collect();
    Ok(Some(VRMCSpringBone {
        spec_version: "1.0".into(),
        colliders,
        collider_groups,
        springs,
    }))
}
fn finite_f32(value: &Value) -> Option<f32> {
    value
        .as_f64()
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
}

fn vector3(value: &Value) -> Option<[f32; 3]> {
    let object = value.as_object()?;
    Some([
        finite_f32(object.get("x")?)?,
        finite_f32(object.get("y")?)?,
        finite_f32(object.get("z")?)?,
    ])
}

fn legacy_gravity_direction([x, y, z]: [f32; 3]) -> [f32; 3] {
    // VRM 0.x faces -Z while the normalized runtime basis faces +Z. The
    // scene basis rotates node transforms; gravity is a world-space vector,
    // so it receives the same Y=pi conversion exactly once here.
    [-x, y, -z]
}
fn required_index(value: &Value, field: &str, path: &str) -> AppResult<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| anyhow::anyhow!("invalid field {path}: expected a non-negative integer"))
}
fn required_f32(value: &Value, field: &str, path: &str) -> AppResult<f32> {
    value
        .get(field)
        .and_then(finite_f32)
        .ok_or_else(|| anyhow::anyhow!("invalid field {path}: expected a finite number"))
}
fn required_vector3(value: &Value, field: &str, path: &str) -> AppResult<[f32; 3]> {
    value
        .get(field)
        .and_then(vector3)
        .ok_or_else(|| anyhow::anyhow!("invalid field {path}: expected x, y, z numbers"))
}
fn clamped_f32(value: &Value, field: &str, min: f32, max: f32) -> AppResult<f32> {
    let number = value
        .get(field)
        .map(|_| {
            required_f32(
                value,
                field,
                &format!("VRM.secondaryAnimation.boneGroups[].{field}"),
            )
        })
        .transpose()?
        .unwrap_or(min);
    if !(min..=max).contains(&number) {
        return Err(anyhow::anyhow!(
            "invalid field VRM.secondaryAnimation.boneGroups[].{field}: expected {min}..={max}"
        ));
    }
    Ok(number)
}
fn non_negative_f32(value: &Value, field: &str) -> AppResult<f32> {
    let number = value
        .get(field)
        .map(|_| {
            required_f32(
                value,
                field,
                &format!("VRM.secondaryAnimation.boneGroups[].{field}"),
            )
        })
        .transpose()?
        .unwrap_or(0.0);
    if number < 0.0 {
        return Err(anyhow::anyhow!(
            "invalid field VRM.secondaryAnimation.boneGroups[].{field}: expected a non-negative number"
        ));
    }
    Ok(number)
}
fn validate_node_index(root: &Value, index: usize, path: &str) -> AppResult<()> {
    let count = root
        .get("nodes")
        .and_then(Value::as_array)
        .context("glTF nodes array is required for legacy VRM normalization")?
        .len();
    if index >= count {
        return Err(anyhow::anyhow!("invalid index {path}={index}"));
    }
    Ok(())
}
fn validate_mesh_index(root: &Value, index: usize, path: &str) -> AppResult<()> {
    let count = root
        .get("meshes")
        .and_then(Value::as_array)
        .context("glTF meshes array is required for legacy VRM normalization")?
        .len();
    if index >= count {
        return Err(anyhow::anyhow!("invalid index {path}={index}"));
    }
    Ok(())
}
fn mesh_instance_nodes(root: &Value, mesh: usize, path: &str) -> AppResult<Vec<usize>> {
    validate_mesh_index(root, mesh, path)?;
    let nodes = root
        .get("nodes")
        .and_then(Value::as_array)
        .context("glTF nodes array is required for legacy VRM normalization")?;
    let instances = nodes
        .iter()
        .enumerate()
        .filter_map(|(node, value)| {
            (value.get("mesh").and_then(Value::as_u64) == Some(mesh as u64)).then_some(node)
        })
        .collect::<Vec<_>>();
    Ok(instances)
}
fn validate_morph_target_index(
    root: &Value,
    mesh: usize,
    morph_index: usize,
    path: &str,
) -> AppResult<()> {
    validate_mesh_index(root, mesh, path)?;
    let mesh_value = root
        .get("meshes")
        .and_then(Value::as_array)
        .and_then(|meshes| meshes.get(mesh))
        .context("glTF mesh is unavailable")?;
    let count = mesh_value
        .get("primitives")
        .and_then(Value::as_array)
        .map(|primitives| {
            primitives
                .iter()
                .filter_map(|primitive| primitive.get("targets").and_then(Value::as_array))
                .map(Vec::len)
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);
    if morph_index >= count {
        return Err(anyhow::anyhow!(
            "invalid index {path}={morph_index}; mesh {mesh} has {count} morph targets"
        ));
    }
    Ok(())
}
fn collect_spring_paths(
    root: &Value,
    node: usize,
    visiting: &mut BTreeSet<usize>,
    current: &mut Vec<usize>,
    paths: &mut Vec<Vec<usize>>,
) -> AppResult<()> {
    if !visiting.insert(node) {
        return Err(anyhow::anyhow!(
            "cycle in glTF node hierarchy at node {node}"
        ));
    }
    current.push(node);
    let children = root
        .get("nodes")
        .and_then(Value::as_array)
        .and_then(|nodes| nodes.get(node))
        .and_then(|value| value.get("children"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if children.is_empty() {
        paths.push(current.clone());
    } else {
        for (child_index, child) in children.iter().enumerate() {
            let child = child
                .as_u64()
                .and_then(|value| value.try_into().ok())
                .ok_or_else(|| {
                    anyhow::anyhow!("invalid child index at glTF node {node}, child {child_index}")
                })?;
            validate_node_index(
                root,
                child,
                &format!("nodes[{node}].children[{child_index}]"),
            )?;
            collect_spring_paths(root, child, visiting, current, paths)?;
        }
    }
    current.pop();
    visiting.remove(&node);
    Ok(())
}
