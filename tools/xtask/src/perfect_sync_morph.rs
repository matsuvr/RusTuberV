//! Loads the sparse non-tongue morph response of a local VRM model.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use vtuber_core::ArkitBlendshape;
use vtuber_tracking::{AvatarMorphDelta, AvatarVertexKey, PerfectSyncMorphResponse};

use crate::teacher_replay::sha256_hex;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Vrm0Extension {
    blend_shape_master: Vrm0Master,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Vrm0Master {
    blend_shape_groups: Vec<Vrm0Group>,
}

#[derive(Deserialize)]
struct Vrm0Group {
    name: String,
    #[serde(default)]
    binds: Vec<Vrm0Bind>,
}

#[derive(Deserialize)]
struct Vrm0Bind {
    mesh: usize,
    index: usize,
    weight: f32,
}

#[derive(Deserialize)]
struct Vrm1Extension {
    expressions: Vrm1Expressions,
}

#[derive(Deserialize)]
struct Vrm1Expressions {
    #[serde(default)]
    preset: BTreeMap<String, Vrm1Expression>,
    #[serde(default)]
    custom: BTreeMap<String, Vrm1Expression>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Vrm1Expression {
    #[serde(default)]
    morph_target_binds: Vec<Vrm1Bind>,
}

#[derive(Deserialize)]
struct Vrm1Bind {
    node: usize,
    index: usize,
    weight: f32,
}

struct MorphBind {
    channel: ArkitBlendshape,
    mesh: usize,
    target: usize,
    weight: f32,
}

fn parse_channel(name: &str) -> Option<ArkitBlendshape> {
    ArkitBlendshape::from_name(name).or_else(|| {
        let mut characters = name.chars();
        let first = characters.next()?;
        let canonical = first.to_uppercase().chain(characters).collect::<String>();
        ArkitBlendshape::from_name(&canonical)
    })
}

fn vrm0_binds(value: &serde_json::Value) -> Result<Vec<MorphBind>, String> {
    let extension: Vrm0Extension = serde_json::from_value(value.clone())
        .map_err(|error| format!("parse VRM 0.x expression extension: {error}"))?;
    Ok(extension
        .blend_shape_master
        .blend_shape_groups
        .into_iter()
        .filter_map(|group| parse_channel(&group.name).map(|channel| (channel, group.binds)))
        .filter(|(channel, _)| *channel != ArkitBlendshape::TongueOut)
        .flat_map(|(channel, binds)| {
            binds.into_iter().map(move |bind| MorphBind {
                channel,
                mesh: bind.mesh,
                target: bind.index,
                weight: bind.weight / 100.0,
            })
        })
        .collect())
}

fn vrm1_binds(
    document: &gltf::Document,
    value: &serde_json::Value,
) -> Result<Vec<MorphBind>, String> {
    let extension: Vrm1Extension = serde_json::from_value(value.clone())
        .map_err(|error| format!("parse VRM 1.0 expression extension: {error}"))?;
    let mut expressions = extension.expressions.preset;
    expressions.extend(extension.expressions.custom);
    let mut binds = Vec::new();
    for (name, expression) in expressions {
        let Some(channel) = parse_channel(&name) else {
            continue;
        };
        if channel == ArkitBlendshape::TongueOut {
            continue;
        }
        for bind in expression.morph_target_binds {
            let mesh = document
                .nodes()
                .nth(bind.node)
                .and_then(|node| node.mesh())
                .ok_or_else(|| format!("VRM 1.0 expression {name} references node without mesh"))?;
            binds.push(MorphBind {
                channel,
                mesh: mesh.index(),
                target: bind.index,
                weight: bind.weight,
            });
        }
    }
    Ok(binds)
}

fn target_positions(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    mesh_index: usize,
    target_index: usize,
) -> Result<Vec<(usize, [f32; 3])>, String> {
    let mesh = document
        .meshes()
        .nth(mesh_index)
        .ok_or_else(|| format!("expression references missing mesh {mesh_index}"))?;
    let mut vertex_base = 0;
    let mut deltas = Vec::new();
    for primitive in mesh.primitives() {
        let reader =
            primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
        let vertex_count = reader
            .read_positions()
            .ok_or_else(|| format!("mesh {mesh_index} primitive has no positions"))?
            .count();
        let target = reader
            .read_morph_targets()
            .nth(target_index)
            .ok_or_else(|| format!("mesh {mesh_index} has no morph target {target_index}"))?;
        let positions = target.0.ok_or_else(|| {
            format!("mesh {mesh_index} morph target {target_index} has no position deltas")
        })?;
        deltas.extend(
            positions
                .enumerate()
                .map(|(index, delta)| (vertex_base + index, delta)),
        );
        vertex_base += vertex_count;
    }
    Ok(deltas)
}

/// Extracts a sparse morph response through the same glTF loader used by the app.
pub(crate) fn load_perfect_sync_morph(path: &Path) -> Result<PerfectSyncMorphResponse, String> {
    let (document, buffers, _) =
        gltf::import(path).map_err(|error| format!("load VRM {}: {error}", path.display()))?;
    let binds = if let Some(extension) = document.extension_value("VRM") {
        vrm0_binds(extension)?
    } else if let Some(extension) = document.extension_value("VRMC_vrm") {
        vrm1_binds(&document, extension)?
    } else {
        return Err(format!("{} has no VRM extension", path.display()));
    };
    let mut response: BTreeMap<usize, BTreeMap<AvatarVertexKey, [f32; 3]>> = BTreeMap::new();
    for bind in binds {
        for (vertex_index, delta) in target_positions(&document, &buffers, bind.mesh, bind.target)?
        {
            let entry = response
                .entry(bind.channel.index())
                .or_default()
                .entry(AvatarVertexKey {
                    mesh_index: bind.mesh,
                    vertex_index,
                })
                .or_insert([0.0; 3]);
            for (output, value) in entry.iter_mut().zip(delta) {
                *output += bind.weight * value;
            }
        }
    }
    let channels = response
        .into_iter()
        .map(|(channel_index, vertices)| {
            let channel = ArkitBlendshape::ALL
                .get(channel_index)
                .copied()
                .ok_or_else(|| "invalid ARKit channel index".to_owned())?;
            let deltas = vertices
                .into_iter()
                .map(|(vertex, delta_xyz)| AvatarMorphDelta { vertex, delta_xyz })
                .collect();
            Ok((channel, deltas))
        })
        .collect::<Result<Vec<_>, String>>()?;
    if channels.is_empty() {
        return Err(format!(
            "{} has no recognized non-tongue Perfect Sync morphs",
            path.display()
        ));
    }
    Ok(PerfectSyncMorphResponse {
        model_sha256: sha256_hex(path)?,
        channels,
    })
}
