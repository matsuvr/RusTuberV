//! VRM generation detection, metadata inspection, and source validation.

use serde_json::Value;
use std::{fs, path::Path};
use vtuber_avatar::glb::Glb;
use vtuber_avatar::vrm0::descriptor::{VrmLookAtType, parse_runtime_descriptor};

use super::{HumanoidNodes, ModelImportError, VrmGeneration, VrmInspectionSummary};

/// Inspects a VRM file without copying it.
pub fn inspect_vrm<P: AsRef<Path>>(path: P) -> Result<VrmInspectionSummary, ModelImportError> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let glb = Glb::parse(&bytes).map_err(|error| ModelImportError::GlbParse(error.to_string()))?;
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

    let json = document.as_json().clone();
    let extensions = json.extensions.as_ref().map(|ext| &ext.others);
    let legacy = extensions.and_then(|ext| ext.get("VRM"));
    let modern = extensions.and_then(|ext| ext.get("VRMC_vrm"));

    let mut summary = match (legacy, modern) {
        (Some(_), Some(_)) => {
            return Err(ModelImportError::AmbiguousVrmVersion {
                reason: "both VRM and VRMC_vrm extensions are present".into(),
            });
        }
        (Some(vrm), None) => inspect_vrm0(&document, &glb.document, vrm)?,
        (None, Some(vrmc)) => inspect_vrm1(&document, vrmc)?,
        (None, None) => {
            return Err(ModelImportError::NotVrm {
                reason: "missing VRM or VRMC_vrm extension".into(),
            });
        }
    };

    let material_root = serde_json::to_value(&json).map_err(|error| {
        ModelImportError::GlbParse(format!("failed to inspect materials: {error}"))
    })?;
    let (mtoon_material_count, unlit_material_count, fallback_material_count) =
        material_counts(&material_root, summary.generation, legacy);
    summary.mtoon_material_count = mtoon_material_count;
    summary.unlit_material_count = unlit_material_count;
    summary.fallback_material_count = fallback_material_count;
    let (spring_chain_count, spring_joint_count, spring_collider_count, spring_center_count) =
        spring_counts(&material_root, summary.generation, legacy);
    summary.spring_chain_count = spring_chain_count;
    summary.spring_joint_count = spring_joint_count;
    summary.spring_collider_count = spring_collider_count;
    summary.spring_center_count = spring_center_count;

    summary.has_node_constraint =
        extensions.is_some_and(|ext| ext.contains_key("VRMC_node_constraint"));
    summary.has_mtoon_materials = match summary.generation {
        VrmGeneration::Vrm0 => legacy
            .and_then(|vrm| vrm.get("materialProperties"))
            .is_some(),
        VrmGeneration::Vrm1 => json
            .extensions_used
            .iter()
            .any(|name| name == "VRMC_materials_mtoon"),
    };

    Ok(summary)
}

fn material_counts(
    root: &serde_json::Value,
    generation: VrmGeneration,
    legacy: Option<&serde_json::Value>,
) -> (usize, usize, usize) {
    let materials = root
        .get("materials")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten();
    let legacy_properties = legacy
        .and_then(|value| value.get("materialProperties"))
        .and_then(serde_json::Value::as_array);
    let mut mtoon = 0;
    let mut unlit = 0;
    let mut fallback = 0;

    for (index, material) in materials.enumerate() {
        let shader = match generation {
            VrmGeneration::Vrm0 => legacy_properties
                .and_then(|properties| properties.get(index))
                .and_then(|property| property.get("shader"))
                .and_then(serde_json::Value::as_str),
            VrmGeneration::Vrm1 => None,
        };
        let extensions = material
            .get("extensions")
            .and_then(serde_json::Value::as_object);
        if shader.is_some_and(|shader| {
            vtuber_avatar::classify_legacy_shader(shader) == vtuber_avatar::LegacyShaderKind::MToon
        }) || extensions
            .is_some_and(|extensions| extensions.contains_key("VRMC_materials_mtoon"))
        {
            mtoon += 1;
        } else if shader.is_some_and(|shader| {
            vtuber_avatar::classify_legacy_shader(shader)
                == vtuber_avatar::LegacyShaderKind::SupportedUnlit
        }) || extensions
            .is_some_and(|extensions| extensions.contains_key("KHR_materials_unlit"))
        {
            unlit += 1;
        } else {
            fallback += 1;
        }
    }
    (mtoon, unlit, fallback)
}

fn spring_counts(
    root: &serde_json::Value,
    generation: VrmGeneration,
    legacy: Option<&serde_json::Value>,
) -> (usize, usize, usize, usize) {
    let Some(extension) = (match generation {
        VrmGeneration::Vrm0 => legacy.and_then(|value| value.get("secondaryAnimation")),
        VrmGeneration::Vrm1 => root
            .get("extensions")
            .and_then(serde_json::Value::as_object)
            .and_then(|extensions| extensions.get("VRMC_springBone")),
    }) else {
        return (0, 0, 0, 0);
    };

    match generation {
        VrmGeneration::Vrm0 => {
            let groups = extension
                .get("boneGroups")
                .and_then(serde_json::Value::as_array);
            let chains = groups.map_or(0, Vec::len);
            let joints = groups
                .into_iter()
                .flatten()
                .filter_map(|group| group.get("bones"))
                .filter_map(serde_json::Value::as_array)
                .map(Vec::len)
                .sum();
            let colliders = extension
                .get("colliderGroups")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|group| group.get("colliders"))
                .filter_map(serde_json::Value::as_array)
                .map(Vec::len)
                .sum();
            let centers = groups
                .into_iter()
                .flatten()
                .filter(|group| {
                    group
                        .get("center")
                        .is_some_and(|center| center.as_i64() != Some(-1))
                })
                .count();
            (chains, joints, colliders, centers)
        }
        VrmGeneration::Vrm1 => {
            let springs = extension
                .get("springs")
                .and_then(serde_json::Value::as_array);
            let chains = springs.map_or(0, Vec::len);
            let joints = springs
                .into_iter()
                .flatten()
                .filter_map(|spring| spring.get("joints"))
                .filter_map(serde_json::Value::as_array)
                .map(Vec::len)
                .sum();
            let colliders = extension
                .get("colliders")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            let centers = springs
                .into_iter()
                .flatten()
                .filter(|spring| spring.get("center").is_some_and(|center| !center.is_null()))
                .count();
            (chains, joints, colliders, centers)
        }
    }
}

fn inspect_vrm0(
    document: &gltf::Document,
    root: &Value,
    vrm: &serde_json::Value,
) -> Result<VrmInspectionSummary, ModelImportError> {
    let descriptor = parse_runtime_descriptor(root)?;
    let indexed_bones = &descriptor.humanoid.human_bones;
    let hips = indexed_bones
        .get("hips")
        .copied()
        .ok_or_else(|| ModelImportError::MissingRequiredBone("hips".into()))?;
    let head = indexed_bones
        .get("head")
        .copied()
        .ok_or_else(|| ModelImportError::MissingRequiredBone("head".into()))?;
    let neck = indexed_bones.get("neck").copied();
    validate_vrm0_expression_binds(document, vrm)?;

    let mut expression_presets = vrm
        .get("blendShapeMaster")
        .and_then(|master| master.get("blendShapeGroups"))
        .and_then(|groups| groups.as_array())
        .map(|groups| {
            groups
                .iter()
                .enumerate()
                .map(|(index, group)| {
                    vtuber_avatar::vrm0::expression_id::resolve_legacy_expression_id(group, index)
                        .name
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    expression_presets.sort();
    expression_presets.dedup();

    let look_at_type = descriptor
        .look_at
        .as_ref()
        .map(|look_at| match look_at.r#type {
            VrmLookAtType::Bone => "bone".into(),
            VrmLookAtType::Expression => "expression".into(),
        });

    Ok(VrmInspectionSummary {
        generation: VrmGeneration::Vrm0,
        spec_version: descriptor.spec_version,
        exporter_version: vrm
            .get("exporterVersion")
            .or_else(|| vrm.get("meta").and_then(|meta| meta.get("exporterVersion")))
            .and_then(|value| value.as_str())
            .map(String::from),
        name: descriptor.meta.name.unwrap_or_default(),
        authors: descriptor.meta.authors,
        license_url: descriptor.meta.license_url,
        expression_presets,
        look_at_type,
        has_spring_bone: vrm.get("secondaryAnimation").is_some(),
        has_node_constraint: false,
        has_first_person: descriptor.first_person.is_some(),
        has_mtoon_materials: false,
        compatibility_warnings: descriptor.compatibility_warnings,
        humanoid_nodes: HumanoidNodes { hips, head, neck },
        ..Default::default()
    })
}

fn inspect_vrm1(
    document: &gltf::Document,
    vrmc: &serde_json::Value,
) -> Result<VrmInspectionSummary, ModelImportError> {
    let spec_version = vrmc
        .get("specVersion")
        .and_then(|value| value.as_str())
        .map(String::from)
        .ok_or_else(|| ModelImportError::GlbParse("missing specVersion".into()))?;
    if spec_version != "1.0" {
        return Err(ModelImportError::UnsupportedVersion(spec_version));
    }

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
        spec_version,
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
        has_first_person: vrmc.get("firstPerson").is_some(),
        has_mtoon_materials: false,
        humanoid_nodes: HumanoidNodes { hips, head, neck },
        ..Default::default()
    })
}

fn validate_vrm0_expression_binds(
    document: &gltf::Document,
    vrm: &serde_json::Value,
) -> Result<(), ModelImportError> {
    let Some(groups) = vrm
        .get("blendShapeMaster")
        .and_then(|master| master.get("blendShapeGroups"))
        .and_then(|groups| groups.as_array())
    else {
        return Ok(());
    };
    let root = serde_json::to_value(document.as_json())
        .map_err(|error| ModelImportError::GlbParse(error.to_string()))?;
    for group in groups {
        let Some(binds) = group.get("binds").and_then(|binds| binds.as_array()) else {
            continue;
        };
        for bind in binds {
            let mesh = bind
                .get("mesh")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .ok_or_else(|| ModelImportError::InvalidVrmField {
                    path: "VRM.blendShapeMaster.blendShapeGroups[].binds[].mesh".into(),
                    reason: "expected a non-negative integer".into(),
                })?;
            let index = bind
                .get("index")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .ok_or_else(|| ModelImportError::InvalidVrmField {
                    path: "VRM.blendShapeMaster.blendShapeGroups[].binds[].index".into(),
                    reason: "expected a non-negative integer".into(),
                })?;
            let weight = bind
                .get("weight")
                .and_then(|value| value.as_f64())
                .ok_or_else(|| ModelImportError::InvalidVrmField {
                    path: "VRM.blendShapeMaster.blendShapeGroups[].binds[].weight".into(),
                    reason: "expected a finite number in 0..=100".into(),
                })?;
            if !weight.is_finite() || !(0.0..=100.0).contains(&weight) {
                return Err(ModelImportError::InvalidVrmField {
                    path: "VRM.blendShapeMaster.blendShapeGroups[].binds[].weight".into(),
                    reason: "expected a finite number in 0..=100".into(),
                });
            }
            if mesh >= document.meshes().len() {
                return Err(ModelImportError::InvalidMeshIndex { index: mesh });
            }
            let count = root
                .get("meshes")
                .and_then(|meshes| meshes.as_array())
                .and_then(|meshes| meshes.get(mesh))
                .and_then(|mesh| mesh.get("primitives"))
                .and_then(|primitives| primitives.as_array())
                .into_iter()
                .flatten()
                .filter_map(|primitive| {
                    primitive
                        .get("targets")
                        .and_then(|targets| targets.as_array())
                })
                .map(Vec::len)
                .max()
                .unwrap_or(0);
            if index >= count {
                return Err(ModelImportError::InvalidMorphTargetIndex { mesh, index });
            }
        }
    }
    Ok(())
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
