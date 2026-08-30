use bevy::color::LinearRgba;
use bevy::prelude::Reflect;
use bevy::render::render_resource::ShaderType;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::vrm::gltf::extensions::{classify_legacy_shader, LegacyShaderKind};

#[derive(Serialize, Deserialize, Reflect, Debug, Clone)]
pub struct VrmcMaterialsExtensitions {
    /// Indicates the version number of `VRMC_materials_mtoon` extension.
    ///
    /// The value is fixed to "1.0".
    #[serde(rename = "specVersion")]
    pub spec_version: String,
    #[serde(rename = "matcapFactor")]
    pub matcap_factor: [f32; 3],
    #[serde(rename = "matcapTexture")]
    pub matcap_texture: Option<MatcapTexture>,
    #[serde(
        rename = "parametricRimFresnelPowerFactor",
        default = "default_parametric_rim_fresnel_power"
    )]
    pub parametric_rim_fresnel_power: f32,
    #[serde(rename = "rimMultiplyTexture")]
    pub rim_multiply_texture: Option<RimMultiplyTexture>,
    #[serde(rename = "outlineColorFactor")]
    pub outline_color_factor: [f32; 3],
    #[serde(rename = "outlineLightingMixFactor")]
    pub outline_lighting_mix_factor: f32,
    #[serde(rename = "outlineWidthFactor")]
    pub outline_width_factor: Option<f32>,
    #[serde(rename = "outlineWidthMultiplyTexture")]
    pub outline_width_multiply_texture: Option<OutlineWidthMultiplyTexture>,
    #[serde(rename = "outlineWidthMode")]
    pub outline_width_mode: String,
    #[serde(rename = "parametricRimColorFactor")]
    pub parametric_rim_color_factor: [f32; 3],
    #[serde(rename = "parametricRimLiftFactor")]
    pub parametric_rim_lift_factor: f32,
    #[serde(rename = "rimLightingMixFactor")]
    pub rim_lighting_mix_factor: f32,
    /// The shade color.
    /// The value is evaluated in linear color space.
    #[serde(rename = "shadeColorFactor")]
    pub shade_color_factor: [f32; 3],
    #[serde(rename = "shadeMultiplyTexture")]
    pub shade_multiply_texture: Option<VrmTexture>,
    #[serde(rename = "renderQueueOffsetNumber")]
    pub render_queue_offset_number: f32,
    #[serde(rename = "shadingShiftFactor")]
    pub shading_shift_factor: f32,
    #[serde(rename = "shadingShiftTexture")]
    pub shading_shift_texture: Option<ShadingShiftTexture>,
    #[serde(rename = "shadingToonyFactor")]
    pub shading_toony_factor: f32,
    #[serde(rename = "transparentWithZWrite")]
    pub transparent_with_z_write: bool,
    #[serde(rename = "uvAnimationMaskTexture")]
    pub uv_animation_mask_texture: Option<UVAnimationMaskTexture>,
    #[serde(rename = "uvAnimationRotationSpeedFactor")]
    pub uv_animation_rotation_speed_factor: f32,
    #[serde(rename = "uvAnimationScrollXSpeedFactor")]
    pub uv_animation_scroll_x_speed_factor: f32,
    #[serde(rename = "uvAnimationScrollYSpeedFactor")]
    pub uv_animation_scroll_y_speed_factor: f32,
    #[serde(rename = "giEqualizationFactor")]
    pub gi_equalization_factor: f32,
    /// Alpha behavior carried by the legacy shader/tag contract.
    ///
    /// This is intentionally not serialized: it is a compatibility hint used
    /// while constructing the existing renderer material.
    #[serde(skip)]
    #[reflect(ignore)]
    pub legacy_alpha_mode: Option<LegacyAlphaMode>,
    /// Legacy `_Color` override in linear RGBA, if present.
    #[serde(skip)]
    #[reflect(ignore)]
    pub legacy_base_color: Option<[f32; 4]>,
    /// Legacy `_MainTex` image index, if present.
    #[serde(skip)]
    #[reflect(ignore)]
    pub legacy_base_texture: Option<usize>,
    /// Legacy emission color in linear RGB, if present.
    #[serde(skip)]
    #[reflect(ignore)]
    pub legacy_emissive: Option<[f32; 3]>,
    /// Legacy emission texture image index, if present.
    #[serde(skip)]
    #[reflect(ignore)]
    pub legacy_emissive_texture: Option<usize>,
    /// Legacy `_Cull` override. `false` means double-sided.
    #[serde(skip)]
    #[reflect(ignore)]
    pub legacy_double_sided: Option<bool>,
    /// Legacy `_MainTex` converted to Bevy's
    /// `[scale_x, scale_y, offset_x, offset_y]`; `_MainTex_ST` is a separate
    /// compatibility alias that is already in this target ordering.
    #[serde(skip)]
    #[reflect(ignore)]
    pub legacy_uv_transform: Option<[f32; 4]>,
    /// True when this entry is an intentional existing `StandardMaterial`
    /// fallback for one of the supported VRM/Unlit shaders.
    #[serde(skip)]
    #[reflect(ignore)]
    pub legacy_standard_fallback: bool,
    /// True when the source requested transparent Z-write, which Bevy's
    /// `StandardMaterial` cannot express.
    #[serde(skip)]
    #[reflect(ignore)]
    pub legacy_z_write_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LegacyAlphaMode {
    Opaque,
    Mask(f32),
    Blend,
}

fn default_parametric_rim_fresnel_power() -> f32 {
    5.0
}

impl VrmcMaterialsExtensitions {
    pub fn shade_color(&self) -> LinearRgba {
        let c = self.shade_color_factor;
        LinearRgba::rgb(c[0], c[1], c[2])
    }

    pub fn parametric_rim_color(&self) -> LinearRgba {
        let c = self.parametric_rim_color_factor;
        LinearRgba::rgb(c[0], c[1], c[2])
    }

    pub fn matcap_color(&self) -> LinearRgba {
        let c = self.matcap_factor;
        LinearRgba::rgb(c[0], c[1], c[2])
    }
}

/// Converts one VRM 0.x materialProperties entry into the existing `MToon`
/// material contract. Legacy values are normalized by name and texture index;
/// the renderer remains the same `MToon` renderer used for VRM 1.0.
pub fn convert_legacy_material_properties(value: &Value) -> Option<VrmcMaterialsExtensitions> {
    let planned = plan_legacy_render_queue_offsets(std::slice::from_ref(value), 1);
    convert_legacy_material_properties_with_render_queue_offset(
        value,
        None,
        planned.first().copied().flatten(),
    )
}

/// Converts a legacy material while validating all referenced image indices.
pub fn convert_legacy_material_properties_with_texture_count(
    value: &Value,
    texture_count: Option<usize>,
) -> Option<VrmcMaterialsExtensitions> {
    let planned = plan_legacy_render_queue_offsets(std::slice::from_ref(value), 1);
    convert_legacy_material_properties_with_render_queue_offset(
        value,
        texture_count,
        planned.first().copied().flatten(),
    )
}

/// Plans the VRM 0.x render-queue offsets using the fixed `UniVRM` migration.
///
/// Transparent materials are ordered from the largest source queue toward
/// zero, while transparent-with-Z-write materials are ordered from zero
/// upward. Opaque and cutout materials always use their `MToon` 1.0 default.
#[must_use]
pub fn plan_legacy_render_queue_offsets(
    values: &[Value],
    gltf_material_count: usize,
) -> Vec<Option<i32>> {
    let mut modes = Vec::with_capacity(values.len());
    let mut transparent = std::collections::BTreeSet::new();
    let mut transparent_z_write = std::collections::BTreeSet::new();

    for (index, value) in values.iter().enumerate() {
        let is_mtoon = value
            .get("shader")
            .and_then(Value::as_str)
            .is_some_and(|shader| classify_legacy_shader(shader) == LegacyShaderKind::MToon);
        let mode = is_mtoon
            .then(|| (index < gltf_material_count).then(|| legacy_render_mode(value)))
            .flatten()
            .flatten();
        let Some(mode) = mode else {
            modes.push(None);
            continue;
        };
        let source_offset = legacy_source_render_queue_offset(value, mode);
        modes.push(Some((mode, source_offset)));
        match mode {
            2 => {
                transparent.insert(source_offset);
            }
            3 => {
                transparent_z_write.insert(source_offset);
            }
            _ => {}
        }
    }

    let transparent_map = transparent
        .into_iter()
        .rev()
        .enumerate()
        .map(|(index, source)| (source, -(index as i32).min(9)))
        .collect::<std::collections::BTreeMap<_, _>>();
    let transparent_z_write_map = transparent_z_write
        .into_iter()
        .enumerate()
        .map(|(index, source)| (source, (index as i32).min(9)))
        .collect::<std::collections::BTreeMap<_, _>>();

    modes
        .into_iter()
        .map(|entry| {
            entry.map(|(mode, source)| match mode {
                2 => transparent_map.get(&source).copied().unwrap_or(0),
                3 => transparent_z_write_map.get(&source).copied().unwrap_or(0),
                _ => 0,
            })
        })
        .collect()
}

pub(crate) fn convert_legacy_material_properties_with_render_queue_offset(
    value: &Value,
    texture_count: Option<usize>,
    render_queue_offset: Option<i32>,
) -> Option<VrmcMaterialsExtensitions> {
    let mut properties = json!({
        "specVersion": "1.0",
        "matcapFactor": [1.0, 1.0, 1.0],
        "matcapTexture": null,
        "parametricRimFresnelPowerFactor": 5.0,
        "rimMultiplyTexture": null,
        "outlineColorFactor": [0.0, 0.0, 0.0],
        "outlineLightingMixFactor": 0.0,
        "outlineWidthFactor": null,
        "outlineWidthMultiplyTexture": null,
        "outlineWidthMode": "none",
        "parametricRimColorFactor": [0.0, 0.0, 0.0],
        "parametricRimLiftFactor": 0.0,
        "rimLightingMixFactor": 1.0,
        "shadeColorFactor": [0.0, 0.0, 0.0],
        "shadeMultiplyTexture": null,
        "renderQueueOffsetNumber": 0.0,
        "shadingShiftFactor": 0.0,
        "shadingShiftTexture": null,
        "shadingToonyFactor": 0.9,
        "transparentWithZWrite": false,
        "uvAnimationMaskTexture": null,
        "uvAnimationRotationSpeedFactor": 0.0,
        "uvAnimationScrollXSpeedFactor": 0.0,
        "uvAnimationScrollYSpeedFactor": 0.0,
        "giEqualizationFactor": 0.9
    })
    .as_object_mut()?
    .clone();

    let floats = value
        .get("floatProperties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let shade_toony = floats.get("_ShadeToony").and_then(finite_f32);
    let shade_shift = floats.get("_ShadeShift").and_then(finite_f32);
    let blend_mode = floats
        .get("_BlendMode")
        .and_then(finite_f32)
        .and_then(|value| {
            let mode = value as i32;
            (value == mode as f32 && (0..=3).contains(&mode)).then_some(mode)
        });
    if let Some(blend_mode) = blend_mode {
        properties.insert(
            "transparentWithZWrite".into(),
            json!(blend_mode == 3),
        );
    } else if !floats.contains_key("_BlendMode")
        && let Some(z_write) = floats.get("_ZWrite").and_then(finite_f32)
    {
        properties.insert("transparentWithZWrite".into(), json!(z_write > 0.5));
    }
    if let Some(render_queue_offset) = render_queue_offset {
        properties.insert(
            "renderQueueOffsetNumber".into(),
            json!(render_queue_offset),
        );
    }
    for (name, value) in &floats {
        let Some(value) = finite_f32(value) else {
            continue;
        };
        match name.as_str() {
            "_OutlineLightingMix" => {
                properties.insert("outlineLightingMixFactor".into(), json!(value));
            }
            "_RimFresnelPower" => {
                properties.insert("parametricRimFresnelPowerFactor".into(), json!(value));
            }
            "_RimLift" => {
                properties.insert("parametricRimLiftFactor".into(), json!(value));
            }
            "_GIEqualizationFactor" => {
                properties.insert("giEqualizationFactor".into(), json!(value));
            }
            "_IndirectLightIntensity" => {
                properties.insert("giEqualizationFactor".into(), json!((1.0 - value).clamp(0.0, 1.0)));
            }
            "_UvAnimRotation" | "_UV_Animation_RotationSpeed" => {
                properties.insert(
                    "uvAnimationRotationSpeedFactor".into(),
                    json!(value * std::f32::consts::TAU),
                );
            }
            "_UvAnimScrollX" | "_UV_Animation_ScrollX" => {
                properties.insert("uvAnimationScrollXSpeedFactor".into(), json!(value));
            }
            "_UvAnimScrollY" | "_UV_Animation_ScrollY" => {
                properties.insert("uvAnimationScrollYSpeedFactor".into(), json!(-value));
            }
            "_Cull" => {
                properties.insert("legacyDoubleSided".into(), json!(value <= 0.5));
            }
            "_CullMode" => {
                // Official MToon 0.x values are Off=0, Front=1, Back=2.
                // glTF cannot express front-face-only culling, so Off and
                // Front both become double-sided.
                properties.insert("legacyDoubleSided".into(), json!(value < 1.5));
            }
            "_ShadingShiftTextureScale" => {
                properties.insert("shadingShiftTextureScale".into(), json!(value));
            }
            _ => {}
        }
    }

    let shade_toony = shade_toony.unwrap_or(0.9);
    let shade_shift = shade_shift.unwrap_or(0.0);
    let range_min = shade_shift;
    let range_max = 1.0 + (shade_shift - 1.0) * shade_toony;
    let migrated_shading_toony = ((2.0 - (range_max - range_min)) * 0.5).clamp(0.0, 1.0);
    let migrated_shading_shift = (-(range_max + range_min) * 0.5).clamp(-1.0, 1.0);
    properties.insert(
        "shadingToonyFactor".into(),
        json!(migrated_shading_toony),
    );
    properties.insert(
        "shadingShiftFactor".into(),
        json!(migrated_shading_shift),
    );

    let outline_width = floats.get("_OutlineWidth").and_then(finite_f32).unwrap_or(0.0);
    let outline_width_mode = floats
        .get("_OutlineWidthMode")
        .and_then(finite_f32)
        .unwrap_or(if outline_width > 0.0 { 1.0 } else { 0.0 });
    match outline_width_mode as i32 {
        1 => {
            properties.insert("outlineWidthMode".into(), json!("worldCoordinates"));
            properties.insert("outlineWidthFactor".into(), json!(outline_width.max(0.0) * 0.01));
        }
        2 => {
            properties.insert("outlineWidthMode".into(), json!("screenCoordinates"));
            properties.insert(
                "outlineWidthFactor".into(),
                json!(outline_width.max(0.0) * 0.01 * 0.5),
            );
        }
        _ => {
            properties.insert("outlineWidthMode".into(), json!("none"));
            properties.insert("outlineWidthFactor".into(), Value::Null);
        }
    }
    if let Some(outline_color_mode) = floats.get("_OutlineColorMode").and_then(finite_f32) {
        properties.insert(
            "outlineLightingMixFactor".into(),
            json!(if outline_color_mode as i32 == 0 {
                0.0
            } else {
                floats
                    .get("_OutlineLightingMix")
                    .and_then(finite_f32)
                    .unwrap_or(0.0)
            }),
        );
    }

    let vectors = value
        .get("vectorProperties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (name, value) in &vectors {
        let Some(vector) = finite_array(value) else {
            continue;
        };
        match name.as_str() {
            "_Color" | "_MainColor" => {
                if vector.len() >= 4 {
                    properties.insert("legacyBaseColor".into(), json!(unity_color(&vector)));
                }
            }
            "_EmissionColor" | "_Emission" => {
                if vector.len() >= 3 {
                    // UniVRM's official exporter stores emission as linear
                    // floats, unlike the sRGB base/shade/rim/outline colors.
                    properties.insert("legacyEmissive".into(), json!(&vector[..3]));
                }
            }
            "_ShadeColor" => {
                if vector.len() >= 3 {
                    let color = unity_rgb(&vector);
                    properties.insert("shadeColorFactor".into(), json!(color));
                }
            }
            "_RimColor" => {
                if vector.len() >= 3 {
                    let color = unity_rgb(&vector);
                    properties.insert("parametricRimColorFactor".into(), json!(color));
                }
            }
            "_OutlineColor" => {
                if vector.len() >= 3 {
                    let color = unity_rgb(&vector);
                    properties.insert("outlineColorFactor".into(), json!(color));
                }
            }
            "_MatcapColor" | "_MatCapColor" if vector.len() >= 3 => {
                let color = unity_rgb(&vector);
                properties.insert("matcapFactor".into(), json!(color));
            }
            _ => {}
        }
    }

    let textures = value
        .get("textureProperties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let main_texture_index = textures
        .get("_MainTex")
        .or_else(|| textures.get("_MainTexture"))
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
        .filter(|index| texture_count.is_none_or(|count| *index < count));
    let main_texture_transform = main_texture_index.and_then(|_| {
        vectors
            .get("_MainTex")
            .and_then(finite_array)
            .and_then(|values| {
                (values.len() >= 4).then(|| legacy_main_texture_transform(&values))
            })
            .or_else(|| {
                vectors
                    .get("_MainTex_ST")
                    .and_then(finite_array)
                    .and_then(|values| {
                        (values.len() >= 4).then(|| [values[0], values[1], values[2], values[3]])
                    })
            })
    });
    for (name, value) in &textures {
        let Some(index) = value.as_u64().and_then(|value| usize::try_from(value).ok()) else {
            continue;
        };
        let texture = texture_with_transform(index, main_texture_transform);
        if texture_count.is_some_and(|count| index >= count) {
            #[cfg(feature = "log")]
            bevy::log::warn!(
                "Ignoring legacy material property {name}: texture index {index} is out of range"
            );
            continue;
        }
        match name.as_str() {
            "_MainTex" | "_MainTexture" => {
                properties.insert("legacyBaseTexture".into(), json!(index));
            }
            "_EmissionMap" | "_EmissionTexture" => {
                properties.insert("legacyEmissiveTexture".into(), json!(index));
            }
            "_ShadeTexture" => {
                properties.insert("shadeMultiplyTexture".into(), texture);
            }
            "_RimTexture" | "_RimMultiplyTexture" => {
                properties.insert("rimMultiplyTexture".into(), texture);
            }
            "_SphereAdd" | "_MatcapTexture" | "_MatCapTex" => {
                properties.insert("matcapTexture".into(), json!({"index": index}));
            }
            "_ShadingShiftTexture" => {
                let scale = floats
                    .get("_ShadingShiftTextureScale")
                    .and_then(finite_f32)
                    .unwrap_or(1.0);
                properties.insert(
                    "shadingShiftTexture".into(),
                    json!({"index": index, "texCoord": 0.0, "scale": scale}),
                );
            }
            "_OutlineWidthTexture" => {
                properties.insert(
                    "outlineWidthMultiplyTexture".into(),
                    json!({"index": index}),
                );
            }
            "_UvAnimMaskTex" | "_UvAnimMaskTexture" => {
                properties.insert("uvAnimationMaskTexture".into(), json!({"index": index}));
            }
            _ => {}
        }
    }
    if !textures.contains_key("_ShadeTexture")
        && let Some(index) = main_texture_index
    {
        properties.insert(
            "shadeMultiplyTexture".into(),
            texture_with_transform(index, main_texture_transform),
        );
    }

    let mut converted: VrmcMaterialsExtensitions =
        serde_json::from_value(Value::Object(properties)).ok()?;
    converted.legacy_alpha_mode = legacy_alpha_mode(value);
    converted.legacy_base_color = value
        .get("vectorProperties")
        .and_then(Value::as_object)
        .and_then(|vectors| vectors.get("_Color").or_else(|| vectors.get("_MainColor")))
        .and_then(finite_array)
        .and_then(|values| (values.len() >= 4).then(|| unity_color(&values)));
    converted.legacy_base_texture = main_texture_index;
    converted.legacy_emissive = value
        .get("vectorProperties")
        .and_then(Value::as_object)
        .and_then(|vectors| {
            vectors
                .get("_EmissionColor")
                .or_else(|| vectors.get("_Emission"))
        })
        .and_then(finite_array)
        .and_then(|values| (values.len() >= 3).then(|| [values[0], values[1], values[2]]));
    converted.legacy_emissive_texture = value
        .get("textureProperties")
        .and_then(Value::as_object)
        .and_then(|textures| {
            textures
                .get("_EmissionMap")
                .or_else(|| textures.get("_EmissionTexture"))
        })
        .and_then(|index| index.as_u64())
        .and_then(|index| index.try_into().ok())
        .filter(|index| texture_count.is_none_or(|count| *index < count));
    converted.legacy_double_sided = value
        .get("floatProperties")
        .and_then(Value::as_object)
        .and_then(|floats| {
            floats
                .get("_CullMode")
                .and_then(finite_f32)
                .map(|cull| cull < 1.5)
                .or_else(|| floats.get("_Cull").and_then(finite_f32).map(|cull| cull <= 0.5))
        });
    converted.legacy_uv_transform = main_texture_transform;
    for name in ["_BumpMap", "_BumpScale"] {
        if value
            .get("textureProperties")
            .and_then(Value::as_object)
            .is_some_and(|properties| properties.contains_key(name))
            || value
                .get("floatProperties")
                .and_then(Value::as_object)
                .is_some_and(|properties| properties.contains_key(name))
        {
            #[cfg(feature = "log")]
            bevy::log::warn!(
                "Legacy material property {name} is retained by glTF fallback because the existing MToon contract has no normal-map slot"
            );
        }
    }
    Some(converted)
}

fn legacy_render_mode(value: &Value) -> Option<i32> {
    let value = value
        .get("floatProperties")
        .and_then(Value::as_object)
        .and_then(|floats| floats.get("_BlendMode"))
        .and_then(finite_f32);
    let Some(value) = value else {
        return Some(0);
    };
    let mode = value as i32;
    (value == mode as f32 && (0..=3).contains(&mode)).then_some(mode)
}

fn legacy_source_render_queue_offset(value: &Value, mode: i32) -> i32 {
    let Some(render_queue) = value.get("renderQueue").and_then(finite_i32) else {
        return 0;
    };
    render_queue.saturating_sub(match mode {
        0 => -1,
        1 => 2450,
        2 => 3000,
        3 => 2501,
        _ => 0,
    })
}

fn legacy_alpha_mode(value: &Value) -> Option<LegacyAlphaMode> {
    let render_type = value
        .get("tagMap")
        .and_then(Value::as_object)
        .and_then(|tags| tags.get("RenderType"))
        .and_then(Value::as_str);
    let shader = value.get("shader").and_then(Value::as_str);
    let cutoff = value
        .get("floatProperties")
        .and_then(Value::as_object)
        .and_then(|floats| floats.get("_Cutoff"))
        .and_then(finite_f32)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    if let Some(blend_mode) = value
        .get("floatProperties")
        .and_then(Value::as_object)
        .and_then(|floats| floats.get("_BlendMode"))
        .and_then(finite_f32)
    {
        return match blend_mode as i32 {
            0 => Some(LegacyAlphaMode::Opaque),
            1 => Some(LegacyAlphaMode::Mask(cutoff)),
            2 | 3 => Some(LegacyAlphaMode::Blend),
            _ => None,
        };
    }
    let name = render_type.or(shader)?;
    match name {
        "TransparentCutout" | "Cutout" | "VRM/UnlitCutout" => {
            Some(LegacyAlphaMode::Mask(cutoff))
        }
        "Transparent" | "VRM/UnlitTransparent" | "VRM/UnlitTransparentZWrite" => {
            Some(LegacyAlphaMode::Blend)
        }
        "Opaque"
        | "VRM/MToon"
        | "VRM/UnlitTexture"
        | "VRM_USE_GLTFSHADER"
        | "Standard"
        | "UniGLTF/UniUnlit" => Some(LegacyAlphaMode::Opaque),
        _ => None,
    }
}

fn finite_f32(value: &Value) -> Option<f32> {
    value
        .as_f64()
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
}

fn finite_i32(value: &Value) -> Option<i32> {
    let value = value.as_i64()?;
    i32::try_from(value).ok()
}

fn finite_array(value: &Value) -> Option<Vec<f32>> {
    value.as_array()?.iter().map(finite_f32).collect()
}

fn unity_rgb(vector: &[f32]) -> [f32; 3] {
    [
        srgb_to_linear(vector[0]),
        srgb_to_linear(vector[1]),
        srgb_to_linear(vector[2]),
    ]
}

fn unity_color(vector: &[f32]) -> [f32; 4] {
    [
        srgb_to_linear(vector[0]),
        srgb_to_linear(vector[1]),
        srgb_to_linear(vector[2]),
        vector[3].clamp(0.0, 1.0),
    ]
}

fn srgb_to_linear(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn texture_with_transform(index: usize, transform: Option<[f32; 4]>) -> Value {
    let transform = transform.unwrap_or([1.0, 1.0, 0.0, 0.0]);
    json!({
        "index": index,
        "extensions": {
            "KHR_texture_transform": {
                "offset": [transform[2], transform[3]],
                "scale": [transform[0], transform[1]]
            }
        }
    })
}

fn legacy_main_texture_transform(values: &[f32]) -> [f32; 4] {
    [values[2], values[3], values[0], 1.0 - values[1] - values[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_legacy_mtoon_properties_to_existing_fields() {
        let value = json!({
            "name": "Body",
            "shader": "VRM/MToon",
            "renderQueue": 2500,
            "floatProperties": {
                "_ShadeToony": 0.7,
                "_OutlineWidth": 0.02,
                "_ZWrite": 1.0,
                "_GIEqualizationFactor": 0.8,
                "_Cutoff": 0.35
            },
            "vectorProperties": {
                "_Color": [0.5, 0.25, 1.0, 0.75],
                "_MainTex_ST": [0.8, 0.9, 0.1, 0.2],
                "_ShadeColor": [0.1, 0.2, 0.3, 1.0],
                "_OutlineColor": [0.4, 0.5, 0.6, 1.0]
            },
            "textureProperties": {
                "_MainTex": 2,
                "_ShadeTexture": 3,
                "_MatCapTex": 4
            },
            "tagMap": {"RenderType": "TransparentCutout"}
        });
        let converted = convert_legacy_material_properties(&value).expect("material should parse");
        assert_eq!(converted.shading_toony_factor, 0.85);
        assert_eq!(converted.shading_shift_factor, -0.15);
        assert_eq!(converted.outline_width_factor, Some(0.0002));
        assert_eq!(converted.render_queue_offset_number, 0.0);
        assert!(converted.transparent_with_z_write);
        assert_eq!(
            converted.shade_color_factor,
            [
                srgb_to_linear(0.1),
                srgb_to_linear(0.2),
                srgb_to_linear(0.3)
            ]
        );
        assert_eq!(
            converted.outline_color_factor,
            [
                srgb_to_linear(0.4),
                srgb_to_linear(0.5),
                srgb_to_linear(0.6)
            ]
        );
        assert_eq!(converted.shade_multiply_texture.unwrap().index, 3);
        assert_eq!(converted.matcap_texture.unwrap().index, 4);
        assert_eq!(converted.legacy_base_texture, Some(2));
        assert_eq!(converted.legacy_base_color.unwrap()[3], 0.75);
        assert_eq!(converted.legacy_uv_transform, Some([0.8, 0.9, 0.1, 0.2]));
        assert_eq!(
            converted.legacy_alpha_mode,
            Some(LegacyAlphaMode::Mask(0.35))
        );
    }

    #[test]
    fn rejects_non_finite_legacy_values_without_panicking() {
        let value = json!({
            "floatProperties": {"_ShadeToony": null},
            "vectorProperties": {"_ShadeColor": [0.1, null, 0.3]}
        });
        let converted = convert_legacy_material_properties(&value).expect("defaults should parse");
        assert_eq!(converted.shading_toony_factor, 0.95);
        assert_eq!(converted.shade_color_factor, [0.0, 0.0, 0.0]);
        assert_eq!(converted.legacy_alpha_mode, None);
    }

    #[test]
    fn supports_all_vrm_unlit_alpha_contracts_with_standard_material_inputs() {
        for (shader, expected) in [
            ("VRM/UnlitTexture", LegacyAlphaMode::Opaque),
            ("VRM/UnlitCutout", LegacyAlphaMode::Mask(0.35)),
            ("VRM/UnlitTransparent", LegacyAlphaMode::Blend),
            ("VRM/UnlitTransparentZWrite", LegacyAlphaMode::Blend),
        ] {
            let value = json!({
                "shader": shader,
                "floatProperties": {"_Cutoff": 0.35, "_Cull": 0.0},
                "vectorProperties": {
                    "_Color": [0.5, 0.25, 1.0, 0.75],
                    "_MainTex_ST": [0.8, 0.9, 0.1, 0.2]
                },
                "textureProperties": {"_MainTex": 0}
            });
            let converted = convert_legacy_material_properties_with_texture_count(&value, Some(1))
                .expect("supported unlit shader should use the standard fallback contract");
            assert_eq!(converted.legacy_alpha_mode, Some(expected));
            assert_eq!(converted.legacy_base_texture, Some(0));
            assert_eq!(converted.legacy_double_sided, Some(true));
            assert_eq!(converted.legacy_uv_transform, Some([0.8, 0.9, 0.1, 0.2]));
        }
    }

    #[test]
    fn migrates_official_vrm0_mtoon_source_shape() {
        let value = json!({
            "shader": "VRM/MToon",
            "renderQueue": 2501,
            "floatProperties": {
                "_BlendMode": 3.0,
                "_CullMode": 2.0,
                "_Cutoff": 0.35,
                "_ShadeToony": 0.8,
                "_ShadeShift": 0.1,
                "_IndirectLightIntensity": 0.25,
                "_OutlineWidth": 2.0,
                "_OutlineWidthMode": 2.0,
                "_OutlineColorMode": 1.0,
                "_OutlineLightingMix": 0.4,
                "_UvAnimRotation": 0.25,
                "_UvAnimScrollX": 0.3,
                "_UvAnimScrollY": 0.4
            },
            "vectorProperties": {
                "_Color": [0.8, 0.7, 0.6, 1.0],
                "_ShadeColor": [0.4, 0.3, 0.2, 1.0],
                "_MainTex": [0.1, 0.2, 0.8, 0.7],
                "_EmissionColor": [0.2, 0.3, 0.4, 1.0]
            },
            "textureProperties": {"_MainTex": 1, "_SphereAdd": 2}
        });
        let converted = convert_legacy_material_properties_with_texture_count(&value, Some(3))
            .expect("official VRM 0.x source shape should parse");

        assert_eq!(converted.legacy_alpha_mode, Some(LegacyAlphaMode::Blend));
        assert_eq!(converted.legacy_double_sided, Some(false));
        assert!(converted.transparent_with_z_write);
        assert_eq!(converted.gi_equalization_factor, 0.75);
        assert!((converted.shading_toony_factor - 0.91).abs() < 1.0e-6);
        assert!((converted.shading_shift_factor + 0.19).abs() < 1.0e-6);
        assert_eq!(converted.outline_width_mode, "screenCoordinates");
        assert_eq!(converted.outline_width_factor, Some(0.01));
        assert_eq!(converted.outline_lighting_mix_factor, 0.4);
        assert_eq!(converted.rim_lighting_mix_factor, 1.0);
        assert_eq!(
            converted.uv_animation_rotation_speed_factor,
            0.25 * std::f32::consts::TAU
        );
        assert_eq!(converted.uv_animation_scroll_x_speed_factor, 0.3);
        assert_eq!(converted.uv_animation_scroll_y_speed_factor, -0.4);
        let uv = converted.legacy_uv_transform.expect("official MainTex transform");
        assert!(uv
            .iter()
            .zip([0.8, 0.7, 0.1, 0.1])
            .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-6));
        assert_eq!(converted.matcap_texture.unwrap().index, 2);
        assert_eq!(converted.legacy_emissive, Some([0.2, 0.3, 0.4]));
    }

    #[test]
    fn covers_legacy_rim_matcap_emission_uv_and_texture_validation_contract() {
        let value = json!({
            "shader": "VRM/MToon",
            "renderQueue": 2450,
            "floatProperties": {
                "_RimFresnelPower": 3.0,
                "_RimLift": 0.2,
                "_RimLightingMix": 0.4,
                "_UvAnimRotation": 0.5,
                "_UvAnimScrollX": 0.6,
                "_UvAnimScrollY": 0.7,
                "_Cull": 0.0,
                "_ZWrite": 1.0,
                "_ShadingShiftTextureScale": 1.5
            },
            "vectorProperties": {
                "_RimColor": [0.1, 0.2, 0.3, 1.0],
                "_MatcapColor": [0.4, 0.5, 0.6, 1.0],
                "_EmissionColor": [0.7, 0.8, 0.9, 1.0],
                "_MainTex_ST": [0.8, 0.9, 0.1, 0.2]
            },
            "textureProperties": {
                "_RimTexture": 1,
                "_ShadingShiftTexture": 2,
                "_UvAnimMaskTexture": 3,
                "_MatcapTexture": 99,
                "_OutlineWidthTexture": 99,
                "_EmissionMap": 1
            },
            "tagMap": {"RenderType": "Transparent"}
        });
        let converted = convert_legacy_material_properties_with_texture_count(&value, Some(4))
            .expect("material should parse");

        assert_eq!(converted.render_queue_offset_number, 0.0);
        assert_eq!(converted.parametric_rim_fresnel_power, 3.0);
        assert_eq!(converted.parametric_rim_lift_factor, 0.2);
        assert_eq!(converted.rim_lighting_mix_factor, 1.0);
        assert_eq!(
            converted.uv_animation_rotation_speed_factor,
            0.5 * std::f32::consts::TAU
        );
        assert_eq!(converted.uv_animation_scroll_x_speed_factor, 0.6);
        assert_eq!(converted.uv_animation_scroll_y_speed_factor, -0.7);
        assert!(converted.transparent_with_z_write);
        assert_eq!(converted.legacy_double_sided, Some(true));
        assert_eq!(converted.legacy_emissive_texture, Some(1));
        assert!(converted.legacy_uv_transform.is_none());
        assert_eq!(converted.rim_multiply_texture.unwrap().index, 1);
        assert_eq!(converted.shading_shift_texture.unwrap().index, 2);
        assert_eq!(converted.uv_animation_mask_texture.unwrap().index, 3);
        assert!(converted.matcap_texture.is_none());
        assert!(converted.outline_width_multiply_texture.is_none());
        assert_eq!(
            converted.legacy_alpha_mode,
            Some(LegacyAlphaMode::Blend)
        );
    }

    #[test]
    fn plans_render_queue_offsets_like_fixed_univrm_migration() {
        let values = vec![
            json!({"shader": "VRM/MToon", "renderQueue": 1999, "floatProperties": {"_BlendMode": 0.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 2450, "floatProperties": {"_BlendMode": 1.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 3020, "floatProperties": {"_BlendMode": 2.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 3005, "floatProperties": {"_BlendMode": 2.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 3020, "floatProperties": {"_BlendMode": 2.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 2507, "floatProperties": {"_BlendMode": 3.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 2501, "floatProperties": {"_BlendMode": 3.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 2507, "floatProperties": {"_BlendMode": 3.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 9999, "floatProperties": {"_BlendMode": 99.0}}),
        ];
        assert_eq!(
            plan_legacy_render_queue_offsets(&values, values.len()),
            vec![
                Some(0),
                Some(0),
                Some(0),
                Some(-1),
                Some(0),
                Some(1),
                Some(0),
                Some(1),
                None
            ]
        );
        assert!(plan_legacy_render_queue_offsets(&values, values.len())
            .into_iter()
            .flatten()
            .all(|offset| (-9..=9).contains(&offset)));
    }

    #[test]
    fn render_queue_planning_ignores_non_mtoon_materials() {
        let values = vec![
            json!({"shader": "VRM/MToon", "renderQueue": 3020, "floatProperties": {"_BlendMode": 2.0}}),
            json!({"shader": "Custom/MToon", "renderQueue": 3010, "floatProperties": {"_BlendMode": 2.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 3005, "floatProperties": {"_BlendMode": 2.0}}),
            json!({"shader": "VRM/UnlitTransparent", "renderQueue": 3001, "floatProperties": {"_BlendMode": 2.0}}),
            json!({"shader": "VRM_USE_GLTFSHADER", "renderQueue": 2999, "floatProperties": {"_BlendMode": 2.0}}),
            json!({"shader": "Standard", "renderQueue": 2998, "floatProperties": {"_BlendMode": 2.0}}),
            json!({"shader": "VRM/MToon", "renderQueue": 2997, "floatProperties": {"_BlendMode": 2.0}}),
        ];
        assert_eq!(
            plan_legacy_render_queue_offsets(&values, 3),
            vec![Some(0), None, Some(-1), None, None, None, None]
        );
    }

    #[test]
    fn main_texture_falls_back_to_shade_like_univrm() {
        let main_only = json!({
            "shader": "VRM/MToon",
            "textureProperties": {"_MainTex": 2}
        });
        let explicit_shade = json!({
            "shader": "VRM/MToon",
            "textureProperties": {"_MainTex": 2, "_ShadeTexture": 1}
        });
        let invalid_main = json!({
            "shader": "VRM/MToon",
            "textureProperties": {"_MainTex": 3}
        });

        assert_eq!(
            convert_legacy_material_properties_with_texture_count(&main_only, Some(3))
                .unwrap()
                .shade_multiply_texture
                .unwrap()
                .index,
            2
        );
        assert_eq!(
            convert_legacy_material_properties_with_texture_count(&explicit_shade, Some(3))
                .unwrap()
                .shade_multiply_texture
                .unwrap()
                .index,
            1
        );
        assert!(
            convert_legacy_material_properties_with_texture_count(&invalid_main, Some(3))
                .unwrap()
                .shade_multiply_texture
                .is_none()
        );
    }

    #[test]
    fn main_texture_transform_requires_main_texture() {
        let vector_only = json!({
            "shader": "VRM/MToon",
            "vectorProperties": {"_MainTex": [0.1, 0.2, 0.8, 0.7]}
        });
        let official = json!({
            "shader": "VRM/MToon",
            "vectorProperties": {
                "_MainTex": [0.1, 0.2, 0.8, 0.7],
                "_MainTex_ST": [0.2, 0.3, 0.4, 0.5]
            },
            "textureProperties": {"_MainTex": 0}
        });
        let alias = json!({
            "shader": "VRM/MToon",
            "vectorProperties": {"_MainTex_ST": [0.2, 0.3, 0.4, 0.5]},
            "textureProperties": {"_MainTex": 0}
        });

        assert!(
            convert_legacy_material_properties(&vector_only)
                .unwrap()
                .legacy_uv_transform
                .is_none()
        );
        let official_transform =
            convert_legacy_material_properties_with_texture_count(&official, Some(1))
                .unwrap()
                .legacy_uv_transform
                .expect("valid MainTex should activate the official transform");
        assert!(official_transform
            .iter()
            .zip([0.8, 0.7, 0.1, 0.1])
            .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-6));
        assert_eq!(
            convert_legacy_material_properties_with_texture_count(&alias, Some(1))
                .unwrap()
                .legacy_uv_transform,
            Some([0.2, 0.3, 0.4, 0.5])
        );
    }

    #[test]
    fn legacy_rim_lighting_mix_migrates_to_safe_one() {
        for value in [0.0, 0.4, 1.0] {
            let source = json!({
                "shader": "VRM/MToon",
                "floatProperties": {"_RimLightingMix": value}
            });
            assert_eq!(
                convert_legacy_material_properties(&source)
                    .unwrap()
                    .rim_lighting_mix_factor,
                1.0
            );
        }
    }

    #[test]
    fn blend_mode_owns_transparent_zwrite() {
        for floats in [
            json!({"_ZWrite": 1.0, "_BlendMode": 0.0}),
            json!({"_BlendMode": 3.0, "_ZWrite": 0.0}),
        ] {
            let value = json!({"shader": "VRM/MToon", "floatProperties": floats});
            let converted = convert_legacy_material_properties(&value).unwrap();
            assert_eq!(converted.transparent_with_z_write, floats["_BlendMode"] == 3.0);
        }
        let legacy_fallback = json!({
            "shader": "VRM/MToon",
            "floatProperties": {"_ZWrite": 1.0}
        });
        assert!(
            convert_legacy_material_properties(&legacy_fallback)
                .unwrap()
                .transparent_with_z_write
        );
    }

    #[test]
    fn malformed_legacy_vector_is_not_compacted() {
        let value = json!({
            "shader": "VRM/MToon",
            "vectorProperties": {
                "_Color": [0.1, null, 0.3, 1.0],
                "_MainTex": [0.1, null, 0.8, 0.7],
                "_EmissionColor": [0.2, null, 0.4, 1.0]
            },
            "textureProperties": {"_MainTex": 0, "_EmissionMap": 0}
        });
        let converted =
            convert_legacy_material_properties_with_texture_count(&value, Some(1))
                .expect("malformed optional properties must not reject the material");
        assert!(converted.legacy_base_color.is_none());
        assert!(converted.legacy_uv_transform.is_none());
        assert!(converted.legacy_emissive.is_none());
    }

    #[test]
    fn custom_mtoon_shader_uses_no_legacy_mtoon_conversion() {
        for shader in ["Custom/MToon", "MyMToonShader", "VRM/MToonExtra"] {
            assert_eq!(classify_legacy_shader(shader), LegacyShaderKind::Unknown);
        }
    }
}

#[derive(Serialize, Deserialize, Reflect, Debug, Clone, Copy)]
pub struct MatcapTexture {
    pub index: usize,
}

#[derive(Serialize, Deserialize, Reflect, Debug, Clone, Copy)]
pub struct RimMultiplyTexture {
    pub index: usize,
}

#[derive(Serialize, Deserialize, Reflect, Debug, Clone, Copy)]
pub struct OutlineWidthMultiplyTexture {
    pub index: usize,
}

#[derive(Serialize, Deserialize, Reflect, Debug, Clone, Copy)]
pub struct UVAnimationMaskTexture {
    pub index: usize,
}

#[derive(Serialize, Deserialize, Reflect, Debug, Clone, Copy)]
pub struct ShadingShiftTexture {
    pub index: usize,
    #[serde(rename = "texCoord")]
    pub tex_coord: f32,
    pub scale: f32,
}

#[derive(Serialize, Deserialize, Reflect, Debug, Clone, Copy)]
pub struct VrmTexture {
    pub extensions: VrmTextureExtensions,
    pub index: usize,
}

#[derive(Serialize, Deserialize, Reflect, Debug, Clone, Copy)]
pub struct VrmTextureExtensions {
    #[serde(rename = "KHR_texture_transform")]
    pub khr_texture_transform: KhrTextureTransform,
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Reflect, Debug, Clone, PartialEq, Copy, ShaderType)]
pub struct KhrTextureTransform {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
}

impl Default for KhrTextureTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
        }
    }
}
