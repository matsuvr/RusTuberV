//! VRM generation detection, metadata inspection, and source validation.

use serde_json::Value;
use std::{fs, path::Path};
use vtuber_avatar::glb::Glb;
use vtuber_avatar::vrm::prepare_vrm_document;

use super::{HumanoidNodes, ModelImportError, VrmGeneration, VrmInspectionSummary};

/// Inspects a VRM file without copying it.
pub fn inspect_vrm<P: AsRef<Path>>(path: P) -> Result<VrmInspectionSummary, ModelImportError> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    prepare_and_inspect_vrm(path, &bytes).map(|(summary, _)| summary)
}

/// Inspects the same converted document that will be stored and loaded.
pub(super) fn prepare_and_inspect_vrm(
    path: &Path,
    bytes: &[u8],
) -> Result<(VrmInspectionSummary, Vec<u8>), ModelImportError> {
    let mut glb =
        Glb::parse(bytes).map_err(|error| ModelImportError::GlbParse(error.to_string()))?;
    let (source, changed) = prepare_vrm_document(&mut glb.document)?;
    let root = serde_json::from_value(glb.document.clone())
        .map_err(|error| ModelImportError::GlbParse(error.to_string()))?;
    let document = gltf::Document::from_json(root)
        .map_err(|error| ModelImportError::GlbParse(error.to_string()))?;
    let blob = glb.bin.map(<[u8]>::to_vec);
    check_external_uris(&document)?;
    let buffers = gltf::import_buffers(&document, path.parent(), blob)
        .map_err(|e| ModelImportError::GlbParse(e.to_string()))?;
    // gltf's image decoder slices buffer views without checking their bounds.
    // Validate against the actual bytes before handing any view to the decoder.
    for view in document.views() {
        let length = buffers
            .get(view.buffer().index())
            .map(|buffer| buffer.0.len());
        if view
            .offset()
            .checked_add(view.length())
            .zip(length)
            .is_none_or(|(end, length)| end > length)
        {
            return Err(ModelImportError::InvalidVrmField {
                path: format!("bufferViews[{}]", view.index()),
                reason: format!(
                    "byte range {} + {} exceeds buffer {} ({} bytes)",
                    view.offset(),
                    view.length(),
                    view.buffer().index(),
                    length.unwrap_or(0)
                ),
            });
        }
    }
    gltf::import_images(&document, path.parent(), &buffers)
        .map_err(|e| ModelImportError::GlbParse(e.to_string()))?;

    let root = &glb.document;
    let vrmc = root
        .pointer("/extensions/VRMC_vrm")
        .ok_or_else(|| ModelImportError::NotVrm {
            reason: "missing converted VRMC_vrm".into(),
        })?;
    let mut summary = inspect_vrm1(&document, vrmc)?;
    let (mtoon, unlit, fallback) = material_counts(root);
    summary.mtoon_material_count = mtoon;
    summary.unlit_material_count = unlit;
    summary.fallback_material_count = fallback;
    let (chains, joints, colliders, centers) = spring_counts(root);
    summary.spring_chain_count = chains;
    summary.spring_joint_count = joints;
    summary.spring_collider_count = colliders;
    summary.spring_center_count = centers;
    summary.has_node_constraint = root
        .get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|node| node.pointer("/extensions/VRMC_node_constraint").is_some());
    summary.has_mtoon_materials = mtoon > 0;
    // Format and exporter describe the original file only. All capabilities
    // above come from the VRM 1.0 document consumed by the runtime.
    summary.generation = source.generation;
    if source.generation == VrmGeneration::Vrm0 {
        summary.spec_version = "0.x".into();
    }
    summary.exporter_version = source.exporter_version;
    summary.compatibility_warnings = source.compatibility_warnings;
    let prepared = if changed {
        glb.to_vec()
            .map_err(|error| ModelImportError::GlbParse(error.to_string()))?
    } else {
        bytes.to_vec()
    };
    Ok((summary, prepared))
}

fn material_counts(root: &Value) -> (usize, usize, usize) {
    let mut mtoon = 0;
    let mut unlit = 0;
    let mut fallback = 0;
    for material in root
        .get("materials")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if material
            .pointer("/extensions/VRMC_materials_mtoon")
            .is_some()
        {
            mtoon += 1;
        } else if material
            .pointer("/extensions/KHR_materials_unlit")
            .is_some()
        {
            unlit += 1;
        } else {
            fallback += 1;
        }
    }
    (mtoon, unlit, fallback)
}

fn spring_counts(root: &Value) -> (usize, usize, usize, usize) {
    let Some(extension) = root.pointer("/extensions/VRMC_springBone") else {
        return (0, 0, 0, 0);
    };
    let springs = extension.get("springs").and_then(Value::as_array);
    let chains = springs.map_or(0, Vec::len);
    let joints = springs
        .into_iter()
        .flatten()
        .filter_map(|spring| spring.get("joints").and_then(Value::as_array))
        .map(Vec::len)
        .sum();
    let colliders = extension
        .get("colliders")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let centers = springs
        .into_iter()
        .flatten()
        .filter(|spring| spring.get("center").is_some_and(|center| !center.is_null()))
        .count();
    (chains, joints, colliders, centers)
}

fn inspect_vrm1(
    document: &gltf::Document,
    vrmc: &serde_json::Value,
) -> Result<VrmInspectionSummary, ModelImportError> {
    let meta = vrmc
        .get("meta")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();
    let name = meta
        .get("name")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let authors = meta
        .get("authors")
        .and_then(|authors| authors.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let license_url = meta
        .get("licenseUrl")
        .and_then(|value| value.as_str())
        .map(String::from);

    let human_bones = vrmc
        .get("humanoid")
        .and_then(|humanoid| humanoid.get("humanBones"))
        .and_then(|bones| bones.as_object())
        .ok_or_else(|| ModelImportError::GlbParse("missing humanoid.humanBones".into()))?;
    let node_count = document.nodes().len();
    let hips = required_bone_index(human_bones, "hips", node_count)?;
    let head = required_bone_index(human_bones, "head", node_count)?;
    let neck = optional_bone_index(human_bones, "neck", node_count)?;

    // VRM 1.0 keeps standard presets and author-defined custom expressions in
    // separate maps. The inspection summary lists both so custom-only models
    // do not look expression-less before the runtime catalog is built.
    let expressions = vrmc.get("expressions");
    let mut expression_presets = ["preset", "custom"]
        .into_iter()
        .filter_map(|section| {
            expressions
                .and_then(|expressions| expressions.get(section))
                .and_then(|section| section.as_object())
        })
        .flat_map(|section| section.keys().cloned())
        .collect::<Vec<_>>();
    expression_presets.sort();
    expression_presets.dedup();

    let look_at_type = vrmc
        .get("lookAt")
        .and_then(|look_at| look_at.get("type"))
        .and_then(|value| value.as_str())
        .map(String::from);

    Ok(VrmInspectionSummary {
        generation: VrmGeneration::Vrm1,
        spec_version: "1.0".into(),
        name,
        authors,
        license_url,
        expression_presets,
        look_at_type,
        has_spring_bone: document
            .as_json()
            .extensions
            .as_ref()
            .is_some_and(|ext| ext.others.contains_key("VRMC_springBone")),
        has_node_constraint: false,
        has_first_person: vrmc
            .get("firstPerson")
            .is_some_and(|value| !value.is_null()),
        has_mtoon_materials: false,
        humanoid_nodes: HumanoidNodes { hips, head, neck },
        ..Default::default()
    })
}

fn required_bone_index(
    human_bones: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    node_count: usize,
) -> Result<usize, ModelImportError> {
    let index = human_bones
        .get(name)
        .and_then(|b| b.as_object())
        .and_then(|b| b.get("node"))
        .and_then(|n| n.as_u64())
        .map(|n| n as usize)
        .ok_or_else(|| ModelImportError::MissingRequiredBone(name.to_string()))?;
    if index >= node_count {
        return Err(ModelImportError::InvalidNodeIndex { index });
    }
    Ok(index)
}

fn optional_bone_index(
    human_bones: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    node_count: usize,
) -> Result<Option<usize>, ModelImportError> {
    match required_bone_index(human_bones, name, node_count) {
        Ok(index) => Ok(Some(index)),
        Err(ModelImportError::MissingRequiredBone(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn check_external_uris(document: &gltf::Document) -> Result<(), ModelImportError> {
    for buffer in document.buffers() {
        if let gltf::buffer::Source::Uri(uri) = buffer.source() {
            return Err(ModelImportError::ExternalUri(uri.to_string()));
        }
    }
    for image in document.images() {
        if let gltf::image::Source::Uri { uri, .. } = image.source() {
            return Err(ModelImportError::ExternalUri(uri.to_string()));
        }
    }
    Ok(())
}
