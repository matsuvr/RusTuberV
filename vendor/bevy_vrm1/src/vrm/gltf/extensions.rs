pub mod runtime_descriptor;
pub mod vrmc_node_constraint;
pub mod vrmc_spring_bone;
pub mod vrmc_vrm;

pub use runtime_descriptor::{
    classify_legacy_shader, collect_legacy_compatibility_warnings, CoordinateBasis,
    LegacyShaderKind, VrmFirstPerson, VrmFirstPersonFlag, VrmGeneration, VrmHumanoid, VrmLookAt,
    Vrm0MetaDiagnostics, VrmCompatibilityWarning, VrmCompatibilityWarningCode, VrmLookAtType,
    VrmMeshAnnotation, VrmMeta, VrmParseError, VrmRangeMap, VrmRuntimeDescriptor,
    parse_runtime_descriptor,
};

use crate::error::AppResult;
use crate::vrm::gltf::extensions::vrmc_spring_bone::{
    Collider, ColliderGroup, Sphere, Spring, SpringJoint, VRMCSpringBone,
};
use crate::vrm::gltf::extensions::vrmc_vrm::{
    Expressions, FirstPerson, FirstPersonFlag, Humanoid, LookAtProperties, LookAtType, Meta,
    MorphTargetBind, RangeMap, VrmPreset, VrmcVrm,
};
use anyhow::Context;
use bevy::gltf::Gltf;
use bevy::platform::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeSet;

#[derive(Serialize, Deserialize)]
pub struct VrmExtensions {
    /// Generation-independent core descriptor used by the adapter boundary.
    pub runtime_descriptor: VrmRuntimeDescriptor,
    #[serde(rename = "VRMC_vrm")]
    pub vrmc_vrm: VrmcVrm,

    #[serde(rename = "VRMC_springBone")]
    pub vrmc_spring_bone: Option<VRMCSpringBone>,
}

impl VrmExtensions {
    pub fn new(json: &serde_json::map::Map<String, serde_json::Value>) -> AppResult<Self> {
        let mut root = Value::Object(Map::new());
        if let Value::Object(root_object) = &mut root {
            root_object.insert("extensions".into(), Value::Object(json.clone()));
        }
        Self::from_root(&root)
    }

    /// Creates a normalized descriptor from the complete glTF JSON document.
    ///
    /// The complete document is required for legacy VRM because 0.x stores
    /// mesh and morph references as glTF indices rather than scene entity
    /// names.
    pub fn from_root(root: &Value) -> AppResult<Self> {
        let runtime_descriptor = parse_runtime_descriptor(root)?;
        let extensions = root
            .get("extensions")
            .and_then(Value::as_object)
            .context("Not found glTF extensions")?;
        let vrmc_vrm = match extensions.get("VRMC_vrm") {
            Some(vrmc) => serde_json::from_value(vrmc.clone())?,
            None => normalized_legacy_vrm(
                &runtime_descriptor,
                extensions.get("VRM").context("Not found VRM extension")?,
                root,
            )?,
        };
        let vrmc_spring_bone = match extensions.get("VRMC_springBone") {
            Some(value) => Some(serde_json::from_value(value.clone())?),
            None => match extensions.get("VRM") {
                Some(legacy) => normalized_legacy_spring_bone(root, legacy)?,
                None => None,
            },
        };
        Ok(Self {
            runtime_descriptor,
            vrmc_vrm,
            vrmc_spring_bone,
        })
    }

    /// Creates a new [`VrmExtensions`] from the glTF asset.
    pub fn from_gltf(gltf: &Gltf) -> AppResult<Self> {
        let source = gltf.source.as_ref().context("glTF source is unavailable")?;
        let root = serde_json::to_value(source.as_json())?;
        Self::from_root(&root)
    }

    /// Gets the name of the VRM avatar.
    ///
    /// Returns `None` if the name does not exist in the meta information.
    pub fn name(&self) -> Option<String> {
        self.runtime_descriptor.meta.name.clone()
    }
}

fn normalized_legacy_vrm(
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
                .map(
                    |annotation| crate::vrm::gltf::extensions::vrmc_vrm::MeshAnnotation {
                        node: annotation.node,
                        first_person_flag: match annotation.flag {
                            VrmFirstPersonFlag::Auto => FirstPersonFlag::Auto,
                            VrmFirstPersonFlag::Both => FirstPersonFlag::Both,
                            VrmFirstPersonFlag::ThirdPersonOnly => FirstPersonFlag::ThirdPersonOnly,
                            VrmFirstPersonFlag::FirstPersonOnly => FirstPersonFlag::FirstPersonOnly,
                        },
                    },
                )
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
    let mut preset = HashMap::default();
    for (group_index, group) in groups.iter().enumerate() {
        let Some(name) = normalized_legacy_expression_name(group, group_index) else {
            continue;
        };
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

fn normalized_legacy_expression_name(
    group: &Value,
    group_index: usize,
) -> Option<String> {
    let preset = group
        .get("presetName")
        .and_then(Value::as_str)
        .map(str::trim);
    let name = group
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim);
    let source = preset
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("unknown"))
        .or(name)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("custom_{group_index}"));
    Some(
        match source.as_str() {
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
            other => other,
        }
        .into(),
    )
}

fn normalized_legacy_spring_bone(
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
                    shape: crate::vrm::gltf::extensions::vrmc_spring_bone::ColliderShape::Sphere(
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
                    terminal_length: Some(0.07),
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

fn required_index(
    value: &Value,
    field: &str,
    path: &str,
) -> AppResult<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| anyhow::anyhow!("invalid field {path}: expected a non-negative integer"))
}

fn required_f32(
    value: &Value,
    field: &str,
    path: &str,
) -> AppResult<f32> {
    value
        .get(field)
        .and_then(finite_f32)
        .ok_or_else(|| anyhow::anyhow!("invalid field {path}: expected a finite number"))
}

fn required_vector3(
    value: &Value,
    field: &str,
    path: &str,
) -> AppResult<[f32; 3]> {
    value
        .get(field)
        .and_then(vector3)
        .ok_or_else(|| anyhow::anyhow!("invalid field {path}: expected x, y, z numbers"))
}

fn clamped_f32(
    value: &Value,
    field: &str,
    min: f32,
    max: f32,
) -> AppResult<f32> {
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

fn non_negative_f32(
    value: &Value,
    field: &str,
) -> AppResult<f32> {
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

fn validate_node_index(
    root: &Value,
    index: usize,
    path: &str,
) -> AppResult<()> {
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

fn validate_mesh_index(
    root: &Value,
    index: usize,
    path: &str,
) -> AppResult<()> {
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

fn mesh_instance_nodes(
    root: &Value,
    mesh: usize,
    path: &str,
) -> AppResult<Vec<usize>> {
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

fn legacy_vector([x, y, z]: [f32; 3]) -> [f32; 3] {
    [-x, y, -z]
}

/// Represents a node in the glTF file.
#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub struct VrmNode {
    /// The index of the node in the glTF file.
    pub node: usize,
}

pub(crate) fn obtain_extensions(
    gltf: &Gltf
) -> AppResult<&serde_json::map::Map<String, serde_json::Value>> {
    gltf.source
        .as_ref()
        .and_then(|source| source.extensions())
        .context("Not found gltf extensions")
}

pub(crate) fn obtain_vrmc_vrm(
    json: &serde_json::map::Map<String, serde_json::Value>
) -> AppResult<serde_json::Value> {
    Ok(json
        .get("VRMC_vrm")
        .or_else(|| json.get("VRMC_vrm_animation"))
        .context("Not found VRMC_vrm or VRMC_vrm_animation")?
        .clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::{Quat, Vec3};
    use crate::vrm::gltf::extensions::vrmc_spring_bone::ColliderShape;
    use serde_json::json;

    #[test]
    fn normalizes_legacy_core_into_existing_vrm1_contract() {
        let root = json!({
            "nodes": [{}, {}, {"mesh": 2}, {"mesh": 3}, {"mesh": 3}],
            "meshes": [{}, {}, {"primitives": [{}]}, {"primitives": [{"targets": [{}, {}, {}, {}, {}]}]}],
            "extensions": {"VRM": {
                "meta": {"title": "legacy", "author": "author"},
                "humanoid": {"humanBones": [
                    {"bone": "hips", "node": 0}, {"bone": "head", "node": 1}
                ]},
                "blendShapeMaster": {"blendShapeGroups": [
                    {"name": "vowel-a", "presetName": "A", "binds": [
                        {"mesh": 3, "index": 4, "weight": 50.0}
                    ]},
                    {"name": "left-blink", "presetName": "Blink_L", "isBinary": true}
                ]},
                "firstPerson": {
                    "meshAnnotations": [{"mesh": 2, "firstPersonFlag": "Both"}],
                    "lookAtTypeName": "Bone",
                    "firstPersonBoneOffset": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "lookAtHorizontalInner": {"xRange": 90.0, "yRange": 10.0},
                    "lookAtHorizontalOuter": {"xRange": 90.0, "yRange": 10.0},
                    "lookAtVerticalDown": {"xRange": 90.0, "yRange": 10.0},
                    "lookAtVerticalUp": {"xRange": 90.0, "yRange": 10.0}
                }
            }}
        });
        let normalized = VrmExtensions::from_root(&root).expect("legacy core should normalize");

        assert_eq!(
            normalized.runtime_descriptor.generation,
            VrmGeneration::Vrm0
        );
        assert_eq!(normalized.name().as_deref(), Some("legacy"));
        assert_eq!(normalized.vrmc_vrm.spec_version, "1.0");
        assert_eq!(normalized.vrmc_vrm.humanoid.human_bones["head"].node, 1);
        let expressions = normalized.vrmc_vrm.expressions.as_ref().unwrap();
        assert!(expressions.preset.contains_key("aa"));
        assert!(expressions.preset.contains_key("blinkLeft"));
        let bind = &expressions.preset["aa"]
            .morph_target_binds
            .as_ref()
            .unwrap()[0];
        assert_eq!(bind.node, 3);
        assert_eq!(bind.index, 4);
        assert_eq!(bind.weight, 0.5);
        assert_eq!(
            expressions.preset["aa"]
                .morph_target_binds
                .as_ref()
                .unwrap()
                .iter()
                .map(|bind| bind.node)
                .collect::<Vec<_>>(),
            [3, 4]
        );
        assert!(expressions.preset["blinkLeft"].is_binary);
        assert_eq!(
            normalized
                .vrmc_vrm
                .first_person
                .as_ref()
                .unwrap()
                .mesh_annotations[0]
                .first_person_flag,
            FirstPersonFlag::Both
        );
        assert_eq!(
            normalized.vrmc_vrm.look_at.as_ref().unwrap().r#type,
            LookAtType::Bone
        );
    }

    #[test]
    fn canonicalizes_legacy_expression_preset_names() {
        let groups = json!([
            {"name": "a", "presetName": "a"},
            {"name": "i", "presetName": "I"},
            {"name": "u", "presetName": "U"},
            {"name": "e", "presetName": "E"},
            {"name": "o", "presetName": "O"},
            {"name": "blink", "presetName": "blink"},
            {"name": "joy", "presetName": "joy"},
            {"name": "look-up", "presetName": "lookup"},
            {"name": "custom", "presetName": "Unknown"}
        ]);
        let legacy = json!({"blendShapeMaster": {"blendShapeGroups": groups}});
        let root = json!({"nodes": [], "meshes": []});
        let expressions = normalized_legacy_expressions(&legacy, &root, &[])
            .expect("groups should parse")
            .expect("expressions should exist");
        for name in [
            "aa", "ih", "ou", "ee", "oh", "blink", "happy", "lookUp", "custom",
        ] {
            assert!(expressions.preset.contains_key(name), "missing {name}");
        }
    }

    #[test]
    fn normalizes_legacy_secondary_animation_with_terminal_and_gravity_basis() {
        let root = json!({
            "nodes": [{}, {}, {}, {}, {"children": [5]}, {}],
            "extensions": {}
        });
        let legacy = json!({
            "secondaryAnimation": {
                "colliderGroups": [{
                    "node": 2,
                    "colliders": [{
                        "offset": {"x": 0.1, "y": 0.2, "z": 0.3},
                        "radius": 0.04
                    }]
                }],
                "boneGroups": [{
                    "stiffiness": 2.8,
                    "gravityPower": 0.5,
                    "gravityDir": {"x": 1.0, "y": -1.0, "z": 0.25},
                    "dragForce": 0.2,
                    "center": -1,
                    "hitRadius": 0.02,
                    "bones": [4],
                    "colliderGroups": [0]
                }]
            }
        });
        let spring = normalized_legacy_spring_bone(&root, &legacy)
            .expect("spring should parse")
            .expect("spring should exist");
        assert_eq!(spring.springs.len(), 1);
        assert_eq!(spring.springs[0].joints.len(), 2);
        assert_eq!(spring.springs[0].terminal_length, Some(0.07));
        assert_eq!(spring.springs[0].center, None);
        assert_eq!(spring.springs[0].joints[0].stiffness, Some(2.8));
        assert_eq!(
            spring.springs[0].joints[0].gravity_dir,
            Some([-1.0, -1.0, -0.25])
        );
        assert_eq!(spring.colliders.len(), 1);
        assert_eq!(
            spring.colliders[0].shape,
            ColliderShape::Sphere(Sphere {
                offset: [0.1, 0.2, 0.3],
                radius: 0.04,
            })
        );
        assert_eq!(spring.spring_colliders(&[0, 9]).len(), 1);
    }

    #[test]
    fn legacy_collider_offset_gets_basis_once_at_world_boundary() {
        let source_offset = [0.25, 0.5, -0.75];
        let root = json!({
            "nodes": [{}, {}, {}],
            "extensions": {}
        });
        let legacy = json!({
            "secondaryAnimation": {
                "colliderGroups": [{
                    "node": 1,
                    "colliders": [{
                        "offset": {"x": 0.25, "y": 0.5, "z": -0.75},
                        "radius": 0.1
                    }]
                }]
            }
        });
        let spring = normalized_legacy_spring_bone(&root, &legacy)
            .expect("collider should parse")
            .expect("collider should exist");
        let sphere = match spring.colliders[0].shape {
            ColliderShape::Sphere(sphere) => sphere,
            ColliderShape::Capsule(_) => panic!("legacy collider must normalize to a sphere"),
        };
        assert_eq!(sphere.offset, source_offset);

        let basis = Quat::from_rotation_y(std::f32::consts::PI);
        let world = basis * Vec3::from(source_offset);
        assert!((world - Vec3::new(-0.25, 0.5, 0.75)).length() < 1.0e-5);
        let double_converted_world = basis * Vec3::from([
            -source_offset[0],
            source_offset[1],
            -source_offset[2],
        ]);
        assert!((double_converted_world - Vec3::from(source_offset)).length() < 1.0e-5);
    }

    #[test]
    fn legacy_gravity_basis_is_applied_once_for_vertical_and_horizontal_vectors() {
        assert_eq!(legacy_gravity_direction([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]);
        assert_eq!(legacy_gravity_direction([1.0, 0.0, 0.25]), [-1.0, 0.0, -0.25]);
    }

    #[test]
    fn drops_duplicate_legacy_joint_writers() {
        let root = json!({
            "nodes": [{}, {}, {}, {}, {"children": [5]}, {}, {}],
            "extensions": {}
        });
        let legacy = json!({
            "secondaryAnimation": {
                "boneGroups": [
                    {"bones": [4, 5]},
                    {"bones": [5, 6]}
                ]
            }
        });
        let spring = normalized_legacy_spring_bone(&root, &legacy)
            .expect("spring should parse")
            .expect("spring should exist");
        assert_eq!(
            spring.springs[0]
                .joints
                .iter()
                .map(|joint| joint.node)
                .collect::<Vec<_>>(),
            [4, 5]
        );
        assert_eq!(
            spring.springs[1]
                .joints
                .iter()
                .map(|joint| joint.node)
                .collect::<Vec<_>>(),
            [6]
        );
    }

}
