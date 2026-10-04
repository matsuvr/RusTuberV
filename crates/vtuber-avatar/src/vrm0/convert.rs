//! VRM 0.x to VRM 1.0 conversion, used only at the import boundary.
//! Coordinates, bones, expressions, materials and springs are converted here;
//! all later preparation and runtime systems consume the resulting VRM 1.0.

use super::descriptor::{LegacyShaderKind, classify_legacy_shader, parse_vrm0};
use super::materials::{
    LegacyAlphaMode, convert_legacy_material_properties_with_render_queue_offset,
    legacy_alpha_mode, plan_legacy_render_queue_offsets,
};
use super::normalize::{
    normalized_legacy_expressions, normalized_legacy_spring_bone, normalized_legacy_vrm,
};
use crate::vrm::{VrmGeneration, VrmParseError, VrmPrepareError, VrmSourceInfo};
use anyhow::Context;
use serde_json::{Map, Value};

pub(crate) fn convert_vrm0_document(
    document: &mut Value,
) -> Result<VrmSourceInfo, VrmPrepareError> {
    let legacy =
        document
            .pointer("/extensions/VRM")
            .cloned()
            .ok_or(VrmPrepareError::Descriptor(
                VrmParseError::MissingGeneration,
            ))?;
    let descriptor = parse_vrm0(document, &legacy).map_err(VrmPrepareError::Descriptor)?;
    let vrmc = normalized_legacy_vrm(&descriptor);
    let spring = normalized_legacy_spring_bone(document, &legacy).map_err(invalid_field)?;

    convert_materials(document, &legacy)?;
    bake_y_pi_basis(document);
    let mut vrmc_value =
        serde_json::to_value(&vrmc).map_err(|error| VrmPrepareError::InvalidField {
            path: "extensions.VRMC_vrm".to_string(),
            reason: error.to_string(),
        })?;
    // The upstream `Expressions` type cannot carry material color binds or
    // the custom-origin record, so the expression section is built as raw
    // JSON and injected over the (null) placeholder.
    let expressions = normalized_legacy_expressions(&legacy, document).map_err(invalid_field)?;
    if let Some(expressions) = expressions
        && let Some(slot) = vrmc_value.get_mut("expressions")
    {
        *slot = expressions;
    }
    inject_extension(document, "VRMC_vrm", vrmc_value);
    if let Some(spring) = spring {
        inject_extension(
            document,
            "VRMC_springBone",
            serde_json::to_value(&spring).map_err(|error| VrmPrepareError::InvalidField {
                path: "extensions.VRMC_springBone".to_string(),
                reason: error.to_string(),
            })?,
        );
    }
    remove_root_extension(document, "VRM");

    Ok(VrmSourceInfo {
        generation: VrmGeneration::Vrm0,
        exporter_version: descriptor
            .legacy_meta
            .and_then(|meta| meta.exporter_version),
        compatibility_warnings: descriptor.compatibility_warnings,
    })
}

/// Repairs thumb names emitted by the old converter in existing managed copies.
pub(crate) fn repair_converted_thumb_names(document: &mut Value) -> bool {
    let mut changed = false;
    if let Some(bones) = document
        .pointer_mut("/extensions/VRMC_vrm/humanoid/humanBones")
        .and_then(Value::as_object_mut)
    {
        for (metacarpal, proximal, intermediate) in [
            (
                "leftThumbMetacarpal",
                "leftThumbProximal",
                "leftThumbIntermediate",
            ),
            (
                "rightThumbMetacarpal",
                "rightThumbProximal",
                "rightThumbIntermediate",
            ),
        ] {
            if let Some(mcp) = bones.remove(intermediate) {
                if let Some(cmc) = bones.remove(proximal) {
                    bones.insert(metacarpal.to_owned(), cmc);
                }
                bones.insert(proximal.to_owned(), mcp);
                changed = true;
            }
        }
    }
    changed
}

fn invalid_field(error: anyhow::Error) -> VrmPrepareError {
    match error.downcast::<VrmParseError>() {
        Ok(error) => VrmPrepareError::Descriptor(error),
        Err(error) => VrmPrepareError::InvalidField {
            path: "extensions.VRM".to_string(),
            reason: error.to_string(),
        },
    }
}

/// VRM 1.0 extension keys of the intermediate material property map.
///
/// Every other key is either a `legacy*` fact the converter maps onto
/// standard glTF material fields or a runtime-only detail the upstream
/// contract cannot represent.
const VRM1_MATERIAL_KEYS: [&str; 25] = [
    "specVersion",
    "matcapFactor",
    "matcapTexture",
    "parametricRimFresnelPowerFactor",
    "rimMultiplyTexture",
    "outlineColorFactor",
    "outlineLightingMixFactor",
    "outlineWidthFactor",
    "outlineWidthMultiplyTexture",
    "outlineWidthMode",
    "parametricRimColorFactor",
    "parametricRimLiftFactor",
    "rimLightingMixFactor",
    "shadeColorFactor",
    "shadeMultiplyTexture",
    "renderQueueOffsetNumber",
    "shadingShiftFactor",
    "shadingShiftTexture",
    "shadingToonyFactor",
    "transparentWithZWrite",
    "uvAnimationMaskTexture",
    "uvAnimationRotationSpeedFactor",
    "uvAnimationScrollXSpeedFactor",
    "uvAnimationScrollYSpeedFactor",
    "giEqualizationFactor",
];

fn convert_materials(document: &mut Value, legacy: &Value) -> Result<(), VrmPrepareError> {
    let Some(entries) = legacy.get("materialProperties").and_then(Value::as_array) else {
        return Ok(());
    };
    let texture_count = document
        .get("textures")
        .and_then(Value::as_array)
        .map(Vec::len);
    let materials = document
        .get_mut("materials")
        .and_then(Value::as_array_mut)
        .context("glTF materials array is required for legacy material migration")
        .map_err(invalid_field)?;
    let offsets = plan_legacy_render_queue_offsets(entries, materials.len());
    let mut used_mtoon = false;
    let mut used_unlit = false;

    for material in materials.iter_mut() {
        let Some(name) = material.get("name").and_then(Value::as_str) else {
            continue;
        };
        let Some((entry_index, entry)) = entries
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.get("name").and_then(Value::as_str) == Some(name))
        else {
            continue;
        };
        let shader = entry
            .get("shader")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match classify_legacy_shader(shader) {
            LegacyShaderKind::MToon => {
                let Some(map) = convert_legacy_material_properties_with_render_queue_offset(
                    entry,
                    texture_count,
                    offsets.get(entry_index).copied().flatten(),
                ) else {
                    continue;
                };
                apply_legacy_standard_fields(material, &map, texture_count);
                if let Some(mode) = legacy_alpha_mode(entry) {
                    apply_legacy_alpha(material, mode);
                }
                let extension = map
                    .into_iter()
                    .filter(|(key, _)| VRM1_MATERIAL_KEYS.contains(&key.as_str()))
                    .collect::<Map<String, Value>>();
                material_extension(material, "VRMC_materials_mtoon", Value::Object(extension));
                used_mtoon = true;
            }
            LegacyShaderKind::SupportedUnlit => {
                let map = convert_legacy_material_properties_with_render_queue_offset(
                    entry,
                    texture_count,
                    None,
                )
                .unwrap_or_default();
                apply_legacy_standard_fields(material, &map, texture_count);
                if let Some(mode) = legacy_alpha_mode(entry) {
                    apply_legacy_alpha(material, mode);
                }
                material_extension(material, "KHR_materials_unlit", Value::Object(Map::new()));
                used_unlit = true;
            }
            LegacyShaderKind::Passthrough | LegacyShaderKind::Unknown => {}
        }
    }

    if used_mtoon {
        mark_extension_used(document, "VRMC_materials_mtoon");
    }
    if used_unlit {
        mark_extension_used(document, "KHR_materials_unlit");
    }
    Ok(())
}

/// Maps the `legacy*` migration facts onto standard glTF material fields.
///
/// Values taken from the legacy entry win over the authored standard fields,
/// matching the vendored runtime preference for the same inputs.
fn apply_legacy_standard_fields(
    material: &mut Value,
    map: &Map<String, Value>,
    texture_count: Option<usize>,
) {
    if let Some(color) = map
        .get("legacyBaseColor")
        .and_then(Value::as_array)
        .map(|color| {
            color
                .iter()
                .filter_map(Value::as_f64)
                .map(|component| component as f32)
                .collect::<Vec<_>>()
        })
        .filter(|color| color.len() >= 4)
    {
        set_pbr_field(
            material,
            "baseColorFactor",
            Value::Array(color.into_iter().map(Value::from).collect()),
        );
    }
    if let Some(index) = map
        .get("legacyBaseTexture")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .filter(|index| texture_count.is_none_or(|count| *index < count))
    {
        let transform = map
            .get("legacyUvTransform")
            .and_then(uv_transform_extension);
        set_pbr_field(material, "baseColorTexture", texture_info(index, transform));
    }
    if let Some(emissive) = map
        .get("legacyEmissive")
        .and_then(Value::as_array)
        .map(|color| {
            color
                .iter()
                .filter_map(Value::as_f64)
                .map(|component| component as f32)
                .collect::<Vec<_>>()
        })
        .filter(|color| color.len() >= 3)
    {
        set_material_field(
            material,
            "emissiveFactor",
            Value::Array(emissive.into_iter().map(Value::from).collect()),
        );
    }
    if let Some(index) = map
        .get("legacyEmissiveTexture")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .filter(|index| texture_count.is_none_or(|count| *index < count))
    {
        let transform = map
            .get("legacyUvTransform")
            .and_then(uv_transform_extension);
        set_material_field(material, "emissiveTexture", texture_info(index, transform));
    }
    if let Some(double_sided) = map.get("legacyDoubleSided").and_then(Value::as_bool) {
        set_material_field(material, "doubleSided", Value::Bool(double_sided));
    }
}

/// Builds a texture info object with an optional `KHR_texture_transform`.
fn texture_info(index: usize, transform: Option<Value>) -> Value {
    let mut info = Map::new();
    info.insert("index".to_string(), Value::from(index));
    if let Some(transform) = transform {
        info.insert(
            "extensions".to_string(),
            Value::Object(Map::from_iter([(
                "KHR_texture_transform".to_string(),
                transform,
            )])),
        );
    }
    Value::Object(info)
}

/// Converts a migrated `[sx, sy, ox, oy]` UV transform into the texture
/// transform extension, or `None` when the entry carries no transform.
fn uv_transform_extension(value: &Value) -> Option<Value> {
    let transform = value.as_array()?;
    if transform.len() < 4 {
        return None;
    }
    let numbers = transform
        .iter()
        .filter_map(Value::as_f64)
        .map(|component| component as f32)
        .collect::<Vec<_>>();
    if numbers.len() < 4 || !numbers.iter().all(|value| value.is_finite()) {
        return None;
    }
    #[expect(
        clippy::indexing_slicing,
        reason = "the length check above rejects fewer than four numbers, so the four reads are in bounds"
    )]
    let (sx, sy, ox, oy) = (numbers[0], numbers[1], numbers[2], numbers[3]);
    let mut extension = Map::new();
    extension.insert(
        "offset".to_string(),
        Value::Array(vec![Value::from(ox), Value::from(oy)]),
    );
    extension.insert(
        "scale".to_string(),
        Value::Array(vec![Value::from(sx), Value::from(sy)]),
    );
    Some(Value::Object(extension))
}

/// Sets one field of the `pbrMetallicRoughness` object, creating it when the
/// material shape allows. Non-object materials are left untouched; the
/// normalization layer reports malformed fields.
fn set_pbr_field(material: &mut Value, key: &str, value: Value) {
    let Some(object) = material.as_object_mut() else {
        return;
    };
    let pbr = object
        .entry("pbrMetallicRoughness")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(pbr) = pbr.as_object_mut() {
        pbr.insert(key.to_string(), value);
    }
}

/// Sets one top-level material field. Non-object materials are left
/// untouched; the normalization layer reports malformed fields.
fn set_material_field(material: &mut Value, key: &str, value: Value) {
    if let Some(object) = material.as_object_mut() {
        object.insert(key.to_string(), value);
    }
}

/// Applies one legacy alpha mode to the standard `alphaMode` fields.
fn apply_legacy_alpha(material: &mut Value, mode: LegacyAlphaMode) {
    match mode {
        LegacyAlphaMode::Opaque => {
            set_material_field(material, "alphaMode", Value::String("OPAQUE".to_string()));
        }
        LegacyAlphaMode::Mask(cutoff) => {
            set_material_field(material, "alphaMode", Value::String("MASK".to_string()));
            set_material_field(material, "alphaCutoff", Value::from(cutoff));
        }
        LegacyAlphaMode::Blend => {
            set_material_field(material, "alphaMode", Value::String("BLEND".to_string()));
        }
    }
}

/// Inserts or replaces one material-level extension object.
fn material_extension(material: &mut Value, key: &str, extension: Value) {
    let Some(object) = material.as_object_mut() else {
        return;
    };
    let extensions = object
        .entry("extensions")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(object) = extensions.as_object_mut() {
        object.insert(key.to_string(), extension);
    }
}

/// Injects one root-level extension and marks it used.
fn inject_extension(document: &mut Value, key: &str, extension: Value) {
    let Some(object) = document.as_object_mut() else {
        return;
    };
    let extensions = object
        .entry("extensions")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(object) = extensions.as_object_mut() {
        object.insert(key.to_string(), extension);
    }
    mark_extension_used(document, key);
}

/// Removes one root-level extension object and its `extensionsUsed` mark.
fn remove_root_extension(document: &mut Value, key: &str) {
    if let Some(extensions) = document
        .get_mut("extensions")
        .and_then(Value::as_object_mut)
    {
        extensions.remove(key);
    }
    for list_key in ["extensionsUsed", "extensionsRequired"] {
        if let Some(used) = document.get_mut(list_key).and_then(Value::as_array_mut) {
            used.retain(|entry| entry.as_str() != Some(key));
        }
    }
}

/// Adds one entry to `extensionsUsed` unless already present.
fn mark_extension_used(document: &mut Value, key: &str) {
    let Some(object) = document.as_object_mut() else {
        return;
    };
    let used = object
        .entry("extensionsUsed")
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Some(used) = used.as_array_mut()
        && !used.iter().any(|entry| entry.as_str() == Some(key))
    {
        used.push(Value::String(key.to_string()));
    }
}

/// Bakes the VRM 0.x `Y = pi` basis into every scene root node.
///
/// VRM 0.x faces `-Z`; premultiplying each root rotation by a half-turn about
/// `+Y` faces the stored model toward the canonical `+Z` direction with no
/// runtime basis entity. Child-relative data (bone indices, morph binds,
/// collider offsets) is unaffected because the whole subtree rotates as one.
fn bake_y_pi_basis(document: &mut Value) {
    let root_nodes = document
        .get("scenes")
        .and_then(Value::as_array)
        .map(|scenes| {
            scenes
                .iter()
                .filter_map(|scene| scene.get("nodes").and_then(Value::as_array))
                .flat_map(|nodes| {
                    nodes
                        .iter()
                        .filter_map(Value::as_u64)
                        .filter_map(|index| usize::try_from(index).ok())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let Some(nodes) = document.get_mut("nodes").and_then(Value::as_array_mut) else {
        return;
    };
    for index in root_nodes {
        let Some(node) = nodes.get_mut(index) else {
            continue;
        };
        let current = node
            .get("rotation")
            .and_then(Value::as_array)
            .and_then(|rotation| {
                rotation
                    .iter()
                    .filter_map(Value::as_f64)
                    .map(|component| component as f32)
                    .collect::<Vec<_>>()
                    .try_into()
                    .ok()
            })
            .map(|rotation: [f32; 4]| {
                let length = rotation.iter().map(|c| c * c).sum::<f32>().sqrt();
                if length.is_finite() && length > f32::EPSILON {
                    [
                        rotation[0] / length,
                        rotation[1] / length,
                        rotation[2] / length,
                        rotation[3] / length,
                    ]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            })
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        // Half-turn about +Y is (0, 1, 0, 0); premultiply the stored rotation.
        let (x, y, z, w) = (current[0], current[1], current[2], current[3]);
        node["rotation"] = Value::Array(vec![
            Value::from(z),
            Value::from(w),
            Value::from(-x),
            Value::from(-y),
        ]);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

    use crate::{glb::Glb, prepare_managed_vrm_bytes};
    use serde_json::json;

    #[test]
    fn legacy_collider_handedness_and_scene_facing_are_distinct_transforms() {
        use bevy::prelude::{Quat, Vec3};
        let node_rotation = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let source = json!({
            "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"children": [1]}, {"rotation": node_rotation.to_array()}],
            "extensions": {"VRM": {
                "humanoid": {"humanBones": [{"bone": "hips", "node": 0}, {"bone": "head", "node": 1}]},
                "secondaryAnimation": {"boneGroups": [], "colliderGroups": [{"node": 1,
                    "colliders": [{"offset": {"x": 1.0, "y": 2.0, "z": 3.0}, "radius": 0.2}]}]}
            }}
        });
        let bytes = Glb::new(source, None).to_vec().unwrap();
        let converted = prepare_managed_vrm_bytes(&bytes).unwrap().unwrap();
        let document = Glb::parse(&converted).unwrap().document;
        let offset: [f32; 3] = serde_json::from_value(
            document["extensions"]["VRMC_springBone"]["colliders"][0]["shape"]["sphere"]["offset"]
                .clone(),
        )
        .unwrap();
        let root: [f32; 4] =
            serde_json::from_value(document["nodes"][0]["rotation"].clone()).unwrap();
        let world = Quat::from_array(root) * node_rotation * Vec3::from_array(offset);
        assert!(world.distance(Vec3::new(-1.0, 3.0, -2.0)) < 1.0e-6);
        assert!(prepare_managed_vrm_bytes(&converted).unwrap().is_none());
    }

    #[test]
    fn vrm0_thumb_nodes_bind_to_the_same_anatomical_joints_in_vrm1() {
        let source = json!({
            "asset": {"version": "2.0"},
            "nodes": [{}, {}, {}, {}, {}, {}, {}, {}],
            "extensions": {"VRM": {"humanoid": {"humanBones": [
                {"bone": "hips", "node": 0}, {"bone": "head", "node": 1},
                {"bone": "leftThumbProximal", "node": 2},
                {"bone": "leftThumbIntermediate", "node": 3},
                {"bone": "leftThumbDistal", "node": 4},
                {"bone": "rightThumbProximal", "node": 5},
                {"bone": "rightThumbIntermediate", "node": 6},
                {"bone": "rightThumbDistal", "node": 7}
            ]}}}
        });
        let bytes = Glb::new(source, None).to_vec().unwrap();
        let converted = prepare_managed_vrm_bytes(&bytes).unwrap().unwrap();
        let document = Glb::parse(&converted).unwrap().document;
        let bones = &document["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"];
        for (side, base) in [("left", 2), ("right", 5)] {
            assert_eq!(bones[format!("{side}ThumbMetacarpal")]["node"], base);
            assert_eq!(bones[format!("{side}ThumbProximal")]["node"], base + 1);
            assert_eq!(bones[format!("{side}ThumbDistal")]["node"], base + 2);
            assert!(bones.get(format!("{side}ThumbIntermediate")).is_none());
        }
    }

    #[test]
    fn loading_an_existing_managed_copy_repairs_legacy_thumb_names_once() {
        let source = json!({
            "asset": {"version": "2.0"},
            "extensions": {"VRMC_vrm": {"specVersion": "1.0", "humanoid": {"humanBones": {
                "leftThumbProximal": {"node": 77},
                "leftThumbIntermediate": {"node": 78},
                "leftThumbDistal": {"node": 79},
                "rightThumbMetacarpal": {"node": 101},
                "rightThumbProximal": {"node": 102},
                "rightThumbDistal": {"node": 103}
            }}}}
        });
        let bin = vec![1, 2, 3, 4];
        let bytes = Glb::new(source, Some(&bin)).to_vec().unwrap();
        let converted = prepare_managed_vrm_bytes(&bytes).unwrap().unwrap();
        let glb = Glb::parse(&converted).unwrap();
        let document = glb.document;
        let converted_bin = glb.bin;
        let bones = &document["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"];
        assert_eq!(bones["leftThumbMetacarpal"]["node"], 77);
        assert_eq!(bones["leftThumbProximal"]["node"], 78);
        assert_eq!(bones["leftThumbDistal"]["node"], 79);
        assert!(bones.get("leftThumbIntermediate").is_none());
        assert_eq!(bones["rightThumbMetacarpal"]["node"], 101);
        assert_eq!(bones["rightThumbProximal"]["node"], 102);
        assert_eq!(bones["rightThumbDistal"]["node"], 103);
        assert_eq!(converted_bin, Some(bin.as_slice()));
        assert!(prepare_managed_vrm_bytes(&converted).unwrap().is_none());
    }
}
