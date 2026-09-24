//! Legacy VRM 0.x material migration for import-time conversion.
//!
//! Moved from the removed vendored bevy_vrm1 patch (see #91). The migration
//! builds the intermediate property map for one legacy `materialProperties`
//! entry: VRM 1.0 `VRMC_materials_mtoon` fields plus `legacy*` keys the file
//! converter maps onto standard glTF material fields. Unknown Unity shader
//! properties are dropped; the import diagnostics already report them.
use serde_json::{Map, Value, json};

use super::descriptor::{LegacyShaderKind, classify_legacy_shader};

/// Alpha behavior migrated from one legacy material entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LegacyAlphaMode {
    Opaque,
    Mask(f32),
    Blend,
}

pub(crate) fn plan_legacy_render_queue_offsets(
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
) -> Option<Map<String, Value>> {
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
        properties.insert("transparentWithZWrite".into(), json!(blend_mode == 3));
    } else if !floats.contains_key("_BlendMode")
        && let Some(z_write) = floats.get("_ZWrite").and_then(finite_f32)
    {
        properties.insert("transparentWithZWrite".into(), json!(z_write > 0.5));
    }
    if let Some(render_queue_offset) = render_queue_offset {
        properties.insert("renderQueueOffsetNumber".into(), json!(render_queue_offset));
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
                properties.insert(
                    "giEqualizationFactor".into(),
                    json!((1.0 - value).clamp(0.0, 1.0)),
                );
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
    properties.insert("shadingToonyFactor".into(), json!(migrated_shading_toony));
    properties.insert("shadingShiftFactor".into(), json!(migrated_shading_shift));

    let outline_width = floats
        .get("_OutlineWidth")
        .and_then(finite_f32)
        .unwrap_or(0.0);
    let outline_width_mode = floats
        .get("_OutlineWidthMode")
        .and_then(finite_f32)
        .unwrap_or(if outline_width > 0.0 { 1.0 } else { 0.0 });
    match outline_width_mode as i32 {
        1 => {
            properties.insert("outlineWidthMode".into(), json!("worldCoordinates"));
            properties.insert(
                "outlineWidthFactor".into(),
                json!(outline_width.max(0.0) * 0.01),
            );
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
                    let linear = vector.iter().take(3).copied().collect::<Vec<_>>();
                    properties.insert("legacyEmissive".into(), json!(linear));
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
            .and_then(|values| (values.len() >= 4).then(|| legacy_main_texture_transform(&values)))
            .or_else(|| {
                vectors
                    .get("_MainTex_ST")
                    .and_then(finite_array)
                    .and_then(|values| {
                        (values.len() >= 4).then(|| {
                            let at = |index: usize| values.get(index).copied().unwrap_or(0.0);
                            [at(0), at(1), at(2), at(3)]
                        })
                    })
            })
    });
    for (name, value) in &textures {
        let Some(index) = value.as_u64().and_then(|value| usize::try_from(value).ok()) else {
            continue;
        };
        let texture = texture_with_transform(index, main_texture_transform);
        if texture_count.is_some_and(|count| index >= count) {
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
    properties.insert(
        "legacyUvTransform".into(),
        main_texture_transform
            .map(|transform| json!([transform[0], transform[1], transform[2], transform[3]]))
            .unwrap_or(Value::Null),
    );
    Some(properties)
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

pub(crate) fn legacy_alpha_mode(value: &Value) -> Option<LegacyAlphaMode> {
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
        "TransparentCutout" | "Cutout" | "VRM/UnlitCutout" => Some(LegacyAlphaMode::Mask(cutoff)),
        "Transparent" | "VRM/UnlitTransparent" | "VRM/UnlitTransparentZWrite" => {
            Some(LegacyAlphaMode::Blend)
        }
        "Opaque" | "VRM/MToon" | "VRM/UnlitTexture" | "VRM_USE_GLTFSHADER" | "Standard"
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
    // Callers only pass vectors with at least three components; the fallback
    // below is unreachable. See the AGENTS.md production panic policy.
    let at = |index: usize| vector.get(index).copied().unwrap_or(0.0);
    [
        srgb_to_linear(at(0)),
        srgb_to_linear(at(1)),
        srgb_to_linear(at(2)),
    ]
}

fn unity_color(vector: &[f32]) -> [f32; 4] {
    // Callers only pass vectors with at least four components; the fallbacks
    // below are unreachable. See the AGENTS.md production panic policy.
    let at = |index: usize| vector.get(index).copied().unwrap_or(0.0);
    [
        srgb_to_linear(at(0)),
        srgb_to_linear(at(1)),
        srgb_to_linear(at(2)),
        vector.get(3).copied().unwrap_or(1.0).clamp(0.0, 1.0),
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
    // Callers only pass vectors with at least four components; the fallbacks
    // below are unreachable. See the AGENTS.md production panic policy.
    let at = |index: usize| values.get(index).copied().unwrap_or(0.0);
    [at(2), at(3), at(0), 1.0 - at(1) - at(3)]
}
