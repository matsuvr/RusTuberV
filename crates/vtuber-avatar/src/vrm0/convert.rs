//! VRM 0.x managed-copy conversion for import-time use.
//!
//! Converts a VRM 0.x source file into VRM 1.0-shaped bytes so the unmodified
//! upstream runtime can load it without any vendored loader or ECS patch.
//! The source file is never overwritten: the caller stores the converted
//! bytes as the managed copy (the same slot the morph-target normalization
//! already rewrites).
//!
//! Conversion outline:
//! - parse the GLB and require the root `VRM` extension without `VRMC_vrm`;
//! - normalize humanoid, expressions, look-at, first-person, and spring bone
//!   through the ported descriptor layers and inject them as `VRMC_vrm` and
//!   `VRMC_springBone`;
//! - migrate each `VRM/MToon` `materialProperties` entry into a
//!   `VRMC_materials_mtoon` extension and map the legacy base/alpha/sided/UV
//!   facts onto standard glTF material fields; tag known unlit shaders with
//!   `KHR_materials_unlit`;
//! - bake the VRM 0.x `Y = pi` basis into the scene root nodes so the stored
//!   model faces the canonical direction with no runtime basis entity;
//! - drop the root `VRM` extension object so the managed copy is unambiguous.
//!
//! Legacy material/texture expression binds and normal-map (`_BumpMap`)
//! details cannot be represented by the upstream contract and are dropped;
//! the import diagnostics already report the unsupported properties.

use anyhow::Context;
use serde_json::{Map, Value};

use super::descriptor::{LegacyShaderKind, classify_legacy_shader, parse_runtime_descriptor};
use super::materials::{
    LegacyAlphaMode, convert_legacy_material_properties_with_render_queue_offset,
    legacy_alpha_mode, plan_legacy_render_queue_offsets,
};
use super::normalize::{
    normalized_legacy_expressions, normalized_legacy_spring_bone, normalized_legacy_vrm,
};

/// Errors that can occur while converting a VRM 0.x source file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Vrm0ConvertError {
    /// The input is not a binary glTF container.
    NotGlb,
    /// The GLB JSON chunk is not valid JSON.
    InvalidJson(String),
    /// The input has no root `VRM` extension or already carries `VRMC_vrm`.
    NotVrm0,
    /// A VRM field cannot be normalized into the VRM 1.0 shape.
    InvalidField {
        /// JSON field path.
        path: String,
        /// Stable validation reason.
        reason: String,
    },
}

impl std::fmt::Display for Vrm0ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotGlb => write!(f, "file is not a binary glTF container"),
            Self::InvalidJson(reason) => write!(f, "glTF JSON is invalid: {reason}"),
            Self::NotVrm0 => write!(f, "file is not a VRM 0.x model"),
            Self::InvalidField { path, reason } => {
                write!(f, "invalid VRM field {path}: {reason}")
            }
        }
    }
}

impl std::error::Error for Vrm0ConvertError {}

/// Converts VRM 0.x `bytes` into VRM 1.0-shaped bytes.
///
/// Returns `Ok(None)` when the input carries no root `VRM` extension (not a
/// VRM 0.x source). Returns `Err` when the input claims to be VRM 0.x but
/// cannot be normalized.
pub fn convert_vrm0_to_vrm1(bytes: &[u8]) -> Result<Option<Vec<u8>>, Vrm0ConvertError> {
    let (mut document, bin) = parse_glb(bytes)?;
    let extensions = document
        .get("extensions")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let Some(legacy) = extensions.get("VRM").cloned() else {
        return Ok(None);
    };
    if extensions.contains_key("VRMC_vrm") {
        return Err(Vrm0ConvertError::NotVrm0);
    }

    let descriptor =
        parse_runtime_descriptor(&document).map_err(|error| Vrm0ConvertError::InvalidField {
            path: "extensions.VRM".to_string(),
            reason: error.to_string(),
        })?;
    let vrmc = normalized_legacy_vrm(&descriptor).map_err(invalid_field)?;
    let spring = normalized_legacy_spring_bone(&document, &legacy).map_err(invalid_field)?;

    convert_materials(&mut document, &legacy)?;
    bake_y_pi_basis(&mut document);
    let mut vrmc_value =
        serde_json::to_value(&vrmc).map_err(|error| Vrm0ConvertError::InvalidField {
            path: "extensions.VRMC_vrm".to_string(),
            reason: error.to_string(),
        })?;
    // The upstream `Expressions` type cannot carry material color binds or
    // the custom-origin record, so the expression section is built as raw
    // JSON and injected over the (null) placeholder.
    let expressions = normalized_legacy_expressions(&legacy, &document).map_err(|error| {
        Vrm0ConvertError::InvalidField {
            path: "extensions.VRM.blendShapeMaster".to_string(),
            reason: error.to_string(),
        }
    })?;
    if let Some(expressions) = expressions
        && let Some(slot) = vrmc_value.get_mut("expressions")
    {
        *slot = expressions;
    }
    inject_extension(&mut document, "VRMC_vrm", vrmc_value);
    if let Some(spring) = spring {
        inject_extension(
            &mut document,
            "VRMC_springBone",
            serde_json::to_value(&spring).map_err(|error| Vrm0ConvertError::InvalidField {
                path: "extensions.VRMC_springBone".to_string(),
                reason: error.to_string(),
            })?,
        );
    }
    remove_root_extension(&mut document, "VRM");

    let json = serde_json::to_vec(&document).map_err(|error| Vrm0ConvertError::InvalidField {
        path: "$".to_string(),
        reason: error.to_string(),
    })?;
    Ok(Some(repack_glb(&json, bin)))
}

/// Prepares a managed model copy for the unmodified upstream runtime.
///
/// The adaptation is selected from the root extension actually present:
/// - root `VRM` without `VRMC_vrm`: VRM 0.x → VRM 1.0 conversion;
/// - root `VRMC_vrm`: VRM 1.0 expression adaptation
///   ([`crate::vrm1::adapt_vrm1_expressions`]).
///
/// Returns `Ok(None)` when neither shape is present or the input already
/// satisfies the runtime contract. The managed copy alone carries everything
/// the adaptation needs; the original source file is not required.
pub fn prepare_managed_vrm_bytes(bytes: &[u8]) -> Result<Option<Vec<u8>>, Vrm0ConvertError> {
    let (document, _) = parse_glb(bytes)?;
    let extensions = document.get("extensions").and_then(Value::as_object);
    let has_legacy = extensions.is_some_and(|extensions| extensions.contains_key("VRM"));
    let has_modern = extensions.is_some_and(|extensions| extensions.contains_key("VRMC_vrm"));
    match (has_legacy, has_modern) {
        (true, false) => convert_vrm0_to_vrm1(bytes),
        (false, true) => crate::vrm1::adapt_vrm1_expressions(bytes),
        _ => Ok(None),
    }
}

fn invalid_field(error: anyhow::Error) -> Vrm0ConvertError {
    Vrm0ConvertError::InvalidField {
        path: "extensions.VRM".to_string(),
        reason: error.to_string(),
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

fn convert_materials(document: &mut Value, legacy: &Value) -> Result<(), Vrm0ConvertError> {
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

/// Splits GLB bytes into the JSON document and the binary chunk.
pub(crate) fn parse_glb(bytes: &[u8]) -> Result<(Value, Option<Vec<u8>>), Vrm0ConvertError> {
    if bytes.get(0..4) != Some(b"glTF".as_slice()) {
        return Err(Vrm0ConvertError::NotGlb);
    }
    let json_len = bytes
        .get(12..16)
        .and_then(|chunk| chunk.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or(Vrm0ConvertError::NotGlb)? as usize;
    if bytes.get(16..20) != Some(b"JSON".as_slice()) {
        return Err(Vrm0ConvertError::NotGlb);
    }
    let Some(json_chunk) = bytes.get(20..20 + json_len) else {
        return Err(Vrm0ConvertError::NotGlb);
    };
    let document: Value = serde_json::from_slice(json_chunk)
        .map_err(|error| Vrm0ConvertError::InvalidJson(error.to_string()))?;
    let rest = bytes.get(20 + json_len..).unwrap_or_default();
    let bin = if rest.len() >= 8 && rest.get(4..8) == Some(b"BIN\0".as_slice()) {
        let bin_len = rest
            .get(0..4)
            .and_then(|chunk| chunk.try_into().ok())
            .map(u32::from_le_bytes)
            .ok_or(Vrm0ConvertError::NotGlb)? as usize;
        let Some(chunk) = rest.get(8..8 + bin_len) else {
            return Err(Vrm0ConvertError::NotGlb);
        };
        Some(chunk.to_vec())
    } else {
        None
    };
    Ok((document, bin))
}

/// Repacks a JSON document and an optional binary chunk into GLB bytes.
pub(crate) fn repack_glb(json: &[u8], bin: Option<Vec<u8>>) -> Vec<u8> {
    let json_padded = pad_chunk(json, 0x20);
    let mut out = Vec::with_capacity(28 + json_padded.len() + bin.as_ref().map_or(0, Vec::len));
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    let bin_padded = bin.map(|bin| pad_chunk(&bin, 0));
    let total = 12 + 8 + json_padded.len() + bin_padded.as_ref().map_or(0, |bin| 8 + bin.len());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_padded.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_padded);
    if let Some(bin) = bin_padded {
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin);
    }
    return out;

    fn pad_chunk(chunk: &[u8], pad: u8) -> Vec<u8> {
        let mut padded = chunk.to_vec();
        while padded.len() % 4 != 0 {
            padded.push(pad);
        }
        padded
    }
}
