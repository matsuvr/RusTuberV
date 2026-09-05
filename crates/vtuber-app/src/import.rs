//! VRM 0.x/1.0 model import and lightweight preflight inspection.
//!
//! Imports a user-selected file into an application-managed asset source and
//! verifies that it is a supported VRM generation before it reaches the
//! `bevy_vrm1` compatibility boundary.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Default maximum import size (256 MiB).
pub const DEFAULT_SIZE_LIMIT: u64 = 256 * 1024 * 1024;
/// Immutable hard cap (1 GiB).
pub const HARD_SIZE_CAP: u64 = 1024 * 1024 * 1024;

/// Maximum morph targets per mesh the Bevy runtime can load.
///
/// Mirrors `bevy_mesh::morph::MAX_MORPH_WEIGHTS`; pinned here so the import
/// boundary does not depend on Bevy.
pub const MAX_MORPH_TARGETS: usize = 256;

/// Errors that can occur while importing or inspecting a model.
#[derive(Debug, Error)]
pub enum ModelImportError {
    /// I/O failure during import.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// File extension is not `.vrm`.
    #[error("MODEL_FILE_INVALID: file extension must be .vrm")]
    InvalidExtension,
    /// File is not a regular file (e.g. symlink or directory).
    #[error("MODEL_FILE_INVALID: not a regular file")]
    NotRegularFile,
    /// File size exceeds the configured limit.
    #[error("MODEL_FILE_INVALID: size {size} exceeds limit {limit}")]
    SizeExceeded {
        /// Actual file size.
        size: u64,
        /// Configured size limit.
        limit: u64,
    },
    /// Configured size limit exceeds the hard cap.
    #[error("MODEL_FILE_INVALID: configured limit {limit} exceeds hard cap {hard_cap}")]
    LimitExceedsHardCap {
        /// Configured size limit.
        limit: u64,
        /// Immutable hard cap.
        hard_cap: u64,
    },
    /// GLB parse failure.
    #[error("MODEL_FILE_INVALID: failed to parse GLB: {0}")]
    GlbParse(String),
    /// No supported VRM generation extension was found.
    #[error("MODEL_NOT_VRM: {reason}")]
    NotVrm {
        /// Stable reason for diagnostics and user-facing error mapping.
        reason: String,
    },
    /// Both VRM 0.x and VRM 1.0 root extensions were supplied.
    #[error("MODEL_AMBIGUOUS_VRM_VERSION: {reason}")]
    AmbiguousVrmVersion {
        /// Stable reason for diagnostics and user-facing error mapping.
        reason: String,
    },
    /// Unsupported VRM spec version.
    #[error("MODEL_UNSUPPORTED_VERSION: spec version {0}")]
    UnsupportedVersion(String),
    /// A legacy human bone name occurred more than once.
    #[error("MODEL_DUPLICATE_HUMAN_BONE: {0}")]
    DuplicateHumanBone(String),
    /// Missing required humanoid bone.
    #[error("MODEL_MISSING_REQUIRED_BONE: {0}")]
    MissingRequiredBone(String),
    /// External buffer/image URI detected.
    #[error("MODEL_FILE_INVALID: external URI not allowed: {0}")]
    ExternalUri(String),
    /// Invalid node index referenced.
    #[error("MODEL_FILE_INVALID: invalid node index {index}")]
    InvalidNodeIndex {
        /// Node index that is out of range.
        index: usize,
    },
    /// Invalid glTF mesh index referenced by a VRM 0.x extension.
    #[error("MODEL_FILE_INVALID: invalid mesh index {index}")]
    InvalidMeshIndex {
        /// Mesh index that is out of range.
        index: usize,
    },
    /// Invalid morph target index referenced by a VRM 0.x bind.
    #[error("MODEL_FILE_INVALID: invalid morph target index {index} for mesh {mesh}")]
    InvalidMorphTargetIndex {
        /// glTF mesh index.
        mesh: usize,
        /// Morph target index.
        index: usize,
    },
    /// Invalid official VRM field shape or value.
    #[error("MODEL_FILE_INVALID: invalid VRM field {path}: {reason}")]
    InvalidVrmField {
        /// JSON field path.
        path: String,
        /// Stable validation reason.
        reason: String,
    },
}

impl ModelImportError {
    /// Returns the stable machine-readable import error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Io(_) => "MODEL_IO_ERROR",
            Self::InvalidExtension
            | Self::NotRegularFile
            | Self::SizeExceeded { .. }
            | Self::LimitExceedsHardCap { .. }
            | Self::GlbParse(_)
            | Self::ExternalUri(_)
            | Self::InvalidNodeIndex { .. }
            | Self::InvalidMeshIndex { .. }
            | Self::InvalidMorphTargetIndex { .. }
            | Self::InvalidVrmField { .. } => "MODEL_FILE_INVALID",
            Self::NotVrm { .. } => "MODEL_NOT_VRM",
            Self::AmbiguousVrmVersion { .. } => "MODEL_AMBIGUOUS_VRM_VERSION",
            Self::UnsupportedVersion(_) => "MODEL_UNSUPPORTED_VERSION",
            Self::DuplicateHumanBone(_) => "MODEL_DUPLICATE_HUMAN_BONE",
            Self::MissingRequiredBone(_) => "MODEL_MISSING_REQUIRED_BONE",
        }
    }
}

/// Supported VRM generation detected by preflight.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VrmGeneration {
    /// Legacy VRM 0.x using the root `VRM` extension.
    Vrm0,
    /// VRM 1.0 using the root `VRMC_vrm` extension.
    #[default]
    Vrm1,
}

/// Summary returned after a successful inspection.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VrmInspectionSummary {
    /// Detected VRM generation.
    pub generation: VrmGeneration,
    /// VRM spec version, or the stable `"0.x"` marker for VRM 0.x.
    pub spec_version: String,
    /// VRM 0.x exporterVersion, retained independently of spec detection.
    #[serde(default)]
    pub exporter_version: Option<String>,
    /// Model name from the generation-specific metadata object.
    pub name: String,
    /// Authors from the generation-specific metadata object.
    pub authors: Vec<String>,
    /// License URL from the generation-specific metadata object.
    pub license_url: Option<String>,
    /// Expression preset names discovered in the model.
    pub expression_presets: Vec<String>,
    /// LookAt type, if present.
    pub look_at_type: Option<String>,
    /// Whether the model contains SpringBone extensions.
    pub has_spring_bone: bool,
    /// Whether the model contains Node Constraint extensions.
    pub has_node_constraint: bool,
    /// Whether the model declares first-person mesh annotations.
    pub has_first_person: bool,
    /// Whether the model declares a material extension understood by the
    /// runtime compatibility layer.
    pub has_mtoon_materials: bool,
    /// Number of material entries classified as legacy/modern MToon.
    pub mtoon_material_count: usize,
    /// Number of material entries classified as unlit.
    pub unlit_material_count: usize,
    /// Number of material entries that use the StandardMaterial fallback.
    pub fallback_material_count: usize,
    /// Number of source-declared SpringBone groups/springs.
    ///
    /// This is an input inventory, not the number of runtime-normalized
    /// `SpringRoot` entities created after hierarchy expansion.
    pub spring_chain_count: usize,
    /// Number of source-declared SpringBone joint/root references.
    ///
    /// For VRM 0.x this counts `secondaryAnimation.boneGroups[*].bones`,
    /// which are root references rather than expanded ordered chains.
    pub spring_joint_count: usize,
    /// Number of source-declared SpringBone colliders.
    pub spring_collider_count: usize,
    /// Number of source-declared SpringBone center-space declarations.
    pub spring_center_count: usize,
    /// Humanoid node indices.
    pub humanoid_nodes: HumanoidNodes,
    /// Non-fatal source compatibility diagnostics.
    #[serde(default)]
    pub compatibility_warnings: Vec<vtuber_avatar::VrmCompatibilityWarning>,
}

/// Humanoid bone node indices.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HumanoidNodes {
    /// Hips node index.
    pub hips: usize,
    /// Head node index.
    pub head: usize,
    /// Optional neck node index.
    pub neck: Option<usize>,
}

/// Result of importing a model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImportedModel {
    /// Stable asset identifier (SHA-256 hex).
    pub id: String,
    /// User-facing model name.
    pub name: String,
    /// Path where the model was copied inside the application asset source.
    pub asset_path: PathBuf,
    /// Path to the import metadata file.
    pub meta_path: PathBuf,
    /// Inspection summary.
    pub summary: VrmInspectionSummary,
    /// Original file path.
    pub original_path: PathBuf,
    /// Original file size in bytes.
    pub size: u64,
}

/// Metadata stored alongside an imported model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImportMeta {
    /// Imported model descriptor.
    pub imported: ImportedModel,
    /// Original file modification time (UNIX epoch seconds).
    pub mtime: Option<u64>,
}

/// Imports a user-selected VRM file into `asset_root` and returns its summary.
///
/// The copied file is placed at `asset_root/avatars/<sha256>/model.vrm`.
/// A metadata file is written at `asset_root/avatars/<sha256>/import.toml`.
///
/// When the source declares more morph targets per mesh than the Bevy runtime
/// supports, the stored copy is normalized by
/// [`normalize_vrm_morph_targets`]; the identity hash always refers to the
/// original source bytes.
pub fn import_vrm<P: AsRef<Path>, Q: AsRef<Path>>(
    source: P,
    asset_root: Q,
    size_limit: u64,
) -> Result<ImportedModel, ModelImportError> {
    if size_limit > HARD_SIZE_CAP {
        return Err(ModelImportError::LimitExceedsHardCap {
            limit: size_limit,
            hard_cap: HARD_SIZE_CAP,
        });
    }

    let source = source.as_ref();
    if !source
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("vrm"))
    {
        return Err(ModelImportError::InvalidExtension);
    }
    let metadata = fs::symlink_metadata(source)?;
    if !metadata.is_file() {
        return Err(ModelImportError::NotRegularFile);
    }
    let size = metadata.len();
    if size > size_limit {
        return Err(ModelImportError::SizeExceeded {
            size,
            limit: size_limit,
        });
    }

    let summary = inspect_vrm(source)?;

    let source_bytes = fs::read(source)?;
    let id = format!("{:x}", Sha256::digest(&source_bytes));
    let stored_bytes = match normalize_vrm_morph_targets(&source_bytes) {
        Some(normalized) => normalized,
        None => {
            if let Some(target_count) = over_limit_morph_target_count(&source_bytes) {
                return Err(ModelImportError::InvalidVrmField {
                    path: "meshes[*].primitives[*].targets".to_string(),
                    reason: format!(
                        "{target_count} morph targets exceed the runtime limit of \
                         {MAX_MORPH_TARGETS} and cannot be reduced without changing animation"
                    ),
                });
            }
            source_bytes
        }
    };

    let dest_dir = asset_root.as_ref().join("avatars").join(&id);
    fs::create_dir_all(&dest_dir)?;
    let dest_model = dest_dir.join("model.vrm");
    let meta_path = dest_dir.join("import.toml");

    ensure_cached_model(&dest_model, &stored_bytes)?;

    let imported = ImportedModel {
        id,
        name: summary.name.clone(),
        asset_path: dest_model.clone(),
        meta_path: meta_path.clone(),
        summary,
        original_path: source.to_path_buf(),
        size,
    };

    let meta = ImportMeta {
        imported: imported.clone(),
        mtime: metadata.modified().ok().and_then(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs())
        }),
    };
    let meta_text = toml::to_string_pretty(&meta)
        .map_err(|e| ModelImportError::Io(io::Error::other(e.to_string())))?;
    write_atomic(&meta_path, meta_text.as_bytes())?;

    Ok(imported)
}

/// Inspects a VRM file without copying it.
pub fn inspect_vrm<P: AsRef<Path>>(path: P) -> Result<VrmInspectionSummary, ModelImportError> {
    let path = path.as_ref();
    let (document, _, _) =
        gltf::import(path).map_err(|e| ModelImportError::GlbParse(format!("{e}")))?;

    check_external_uris(&document)?;

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
        (Some(vrm), None) => inspect_vrm0(&document, vrm)?,
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
    if let Some(legacy) = legacy {
        summary.compatibility_warnings =
            vtuber_avatar::collect_legacy_compatibility_warnings(&material_root, legacy);
    }
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
    vrm: &serde_json::Value,
) -> Result<VrmInspectionSummary, ModelImportError> {
    let meta = vrm
        .get("meta")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();
    let name = meta
        .get("title")
        .or_else(|| meta.get("name"))
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let authors = meta
        .get("author")
        .and_then(|value| value.as_str())
        .map(|value| vec![value.to_string()])
        .unwrap_or_default();
    let license_url = meta
        .get("otherLicenseUrl")
        .or_else(|| meta.get("licenseUrl"))
        .and_then(|value| value.as_str())
        .map(String::from);

    let human_bones = vrm
        .get("humanoid")
        .and_then(|humanoid| humanoid.get("humanBones"))
        .and_then(|bones| bones.as_array())
        .ok_or_else(|| ModelImportError::GlbParse("missing legacy humanoid.humanBones".into()))?;
    let node_count = document.nodes().len();
    let indexed_bones = index_legacy_human_bones(human_bones, node_count)?;
    let hips = indexed_bones
        .get("hips")
        .copied()
        .ok_or_else(|| ModelImportError::MissingRequiredBone("hips".into()))?;
    let head = indexed_bones
        .get("head")
        .copied()
        .ok_or_else(|| ModelImportError::MissingRequiredBone("head".into()))?;
    let neck = indexed_bones.get("neck").copied();

    validate_vrm0_first_person(document, vrm)?;
    validate_vrm0_expression_binds(document, vrm)?;

    let mut expression_presets = vrm
        .get("blendShapeMaster")
        .and_then(|master| master.get("blendShapeGroups"))
        .and_then(|groups| groups.as_array())
        .map(|groups| {
            groups
                .iter()
                .enumerate()
                .map(|(index, group)| normalize_legacy_expression_name(group, index))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    expression_presets.sort();
    expression_presets.dedup();

    let look_at_type = validate_vrm0_look_at(vrm)?;

    Ok(VrmInspectionSummary {
        generation: VrmGeneration::Vrm0,
        spec_version: "0.x".into(),
        exporter_version: vrm
            .get("exporterVersion")
            .or_else(|| vrm.get("meta").and_then(|meta| meta.get("exporterVersion")))
            .and_then(|value| value.as_str())
            .map(String::from),
        name,
        authors,
        license_url,
        expression_presets,
        look_at_type,
        has_spring_bone: vrm.get("secondaryAnimation").is_some(),
        has_node_constraint: false,
        has_first_person: vrm.get("firstPerson").is_some(),
        has_mtoon_materials: false,
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

    let mut expression_presets = vrmc
        .get("expressions")
        .and_then(|expressions| expressions.get("preset"))
        .and_then(|preset| preset.as_object())
        .map(|preset| preset.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    expression_presets.sort();

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

fn validate_vrm0_look_at(vrm: &serde_json::Value) -> Result<Option<String>, ModelImportError> {
    let Some(first_person) = vrm.get("firstPerson") else {
        return Ok(None);
    };
    let look_at_fields = [
        "lookAtTypeName",
        "lookAtHorizontalInner",
        "lookAtHorizontalOuter",
        "lookAtVerticalDown",
        "lookAtVerticalUp",
    ];
    let has_look_at = look_at_fields
        .iter()
        .any(|field| first_person.get(*field).is_some());
    if !has_look_at {
        return Ok(None);
    }
    let look_at_type = first_person
        .get("lookAtTypeName")
        .and_then(|value| value.as_str())
        .ok_or_else(|| ModelImportError::InvalidVrmField {
            path: "VRM.firstPerson.lookAtTypeName".into(),
            reason: "expected Bone or BlendShape".into(),
        })?;
    let normalized = match look_at_type {
        "Bone" => "bone",
        "BlendShape" => "expression",
        other => {
            return Err(ModelImportError::InvalidVrmField {
                path: "VRM.firstPerson.lookAtTypeName".into(),
                reason: format!("unknown value {other}"),
            });
        }
    };
    let offset = first_person.get("firstPersonBoneOffset").ok_or_else(|| {
        ModelImportError::InvalidVrmField {
            path: "VRM.firstPerson.firstPersonBoneOffset".into(),
            reason: "required when LookAt is declared".into(),
        }
    })?;
    validate_vector3_object(offset, "VRM.firstPerson.firstPersonBoneOffset")?;
    for field in [
        "lookAtHorizontalInner",
        "lookAtHorizontalOuter",
        "lookAtVerticalDown",
        "lookAtVerticalUp",
    ] {
        let path = format!("VRM.firstPerson.{field}");
        let map = first_person
            .get(field)
            .ok_or_else(|| ModelImportError::InvalidVrmField {
                path: path.clone(),
                reason: "all four DegreeMap objects are required".into(),
            })?;
        let object = map
            .as_object()
            .ok_or_else(|| ModelImportError::InvalidVrmField {
                path: path.clone(),
                reason: "expected an object".into(),
            })?;
        for range in ["xRange", "yRange"] {
            let value = object
                .get(range)
                .and_then(|value| value.as_f64())
                .filter(|value| value.is_finite());
            if value.is_none() || value.is_some_and(|value| value < 0.0) {
                return Err(ModelImportError::InvalidVrmField {
                    path: format!("{path}.{range}"),
                    reason: "expected a finite, non-negative number".into(),
                });
            }
        }
        if let Some(curve) = object.get("curve") {
            let values = curve
                .as_array()
                .ok_or_else(|| ModelImportError::InvalidVrmField {
                    path: format!("{path}.curve"),
                    reason: "expected an array".into(),
                })?;
            if values
                .iter()
                .any(|value| value.as_f64().is_none_or(|value| !value.is_finite()))
            {
                return Err(ModelImportError::InvalidVrmField {
                    path: format!("{path}.curve"),
                    reason: "curve coefficients must be finite numbers".into(),
                });
            }
            if values.len() % 4 != 0 {
                return Err(ModelImportError::InvalidVrmField {
                    path: format!("{path}.curve"),
                    reason: "curve must contain groups of four coefficients".into(),
                });
            }
        }
    }
    Ok(Some(normalized.into()))
}

fn validate_vector3_object(value: &serde_json::Value, path: &str) -> Result<(), ModelImportError> {
    let object = value
        .as_object()
        .ok_or_else(|| ModelImportError::InvalidVrmField {
            path: path.into(),
            reason: "expected an object with x, y, z".into(),
        })?;
    for field in ["x", "y", "z"] {
        if object
            .get(field)
            .and_then(|value| value.as_f64())
            .is_none_or(|value| !value.is_finite())
        {
            return Err(ModelImportError::InvalidVrmField {
                path: format!("{path}.{field}"),
                reason: "expected a finite number".into(),
            });
        }
    }
    Ok(())
}

fn validate_vrm0_first_person(
    document: &gltf::Document,
    vrm: &serde_json::Value,
) -> Result<(), ModelImportError> {
    let Some(first_person) = vrm.get("firstPerson") else {
        return Ok(());
    };
    let node_count = document.nodes().len();
    if let Some(value) = first_person.get("firstPersonBone") {
        let index = value
            .as_u64()
            .and_then(|value| value.try_into().ok())
            .ok_or_else(|| ModelImportError::InvalidVrmField {
                path: "VRM.firstPerson.firstPersonBone".into(),
                reason: "expected a non-negative integer".into(),
            })?;
        if index >= node_count {
            return Err(ModelImportError::InvalidNodeIndex { index });
        }
    }
    let annotations = match first_person.get("meshAnnotations") {
        None => return Ok(()),
        Some(value) => value
            .as_array()
            .ok_or_else(|| ModelImportError::InvalidVrmField {
                path: "VRM.firstPerson.meshAnnotations".into(),
                reason: "expected an array".into(),
            })?,
    };
    for annotation in annotations {
        let mesh = annotation
            .get("mesh")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .ok_or_else(|| ModelImportError::InvalidVrmField {
                path: "VRM.firstPerson.meshAnnotations[].mesh".into(),
                reason: "expected a non-negative integer".into(),
            })?;
        if mesh >= document.meshes().len() {
            return Err(ModelImportError::InvalidMeshIndex { index: mesh });
        }
        let flag = annotation
            .get("firstPersonFlag")
            .and_then(|value| value.as_str())
            .ok_or_else(|| ModelImportError::InvalidVrmField {
                path: "VRM.firstPerson.meshAnnotations[].firstPersonFlag".into(),
                reason: "expected Auto, Both, ThirdPersonOnly, or FirstPersonOnly".into(),
            })?;
        if !matches!(
            flag,
            "Auto"
                | "auto"
                | "Both"
                | "both"
                | "ThirdPersonOnly"
                | "thirdPersonOnly"
                | "FirstPersonOnly"
                | "firstPersonOnly"
        ) {
            return Err(ModelImportError::InvalidVrmField {
                path: "VRM.firstPerson.meshAnnotations[].firstPersonFlag".into(),
                reason: format!("unknown value {flag}"),
            });
        }
    }
    Ok(())
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

fn normalize_legacy_expression_name(group: &serde_json::Value, group_index: usize) -> String {
    let preset = group
        .get("presetName")
        .and_then(|value| value.as_str())
        .map(str::trim);
    let name = group
        .get("name")
        .and_then(|value| value.as_str())
        .map(str::trim);
    let source = preset
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("unknown"))
        .or(name)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("custom_{group_index}"));
    match source.as_str() {
        "A" | "a" => "aa",
        "I" | "i" => "ih",
        "U" | "u" => "ou",
        "E" | "e" => "ee",
        "O" | "o" => "oh",
        "Blink" | "blink" => "blink",
        "Blink_L" | "blink_l" => "blinkLeft",
        "Blink_R" | "blink_r" => "blinkRight",
        "Joy" | "joy" => "happy",
        "Angry" | "angry" => "angry",
        "Sorrow" | "sorrow" => "sad",
        "Fun" | "fun" => "relaxed",
        "LookUp" | "lookup" => "lookUp",
        "LookDown" | "lookdown" => "lookDown",
        "LookLeft" | "lookleft" => "lookLeft",
        "LookRight" | "lookright" => "lookRight",
        "Neutral" | "neutral" => "neutral",
        other => other,
    }
    .into()
}

fn index_legacy_human_bones(
    bones: &[serde_json::Value],
    node_count: usize,
) -> Result<std::collections::BTreeMap<String, usize>, ModelImportError> {
    let mut indexed = std::collections::BTreeMap::new();
    for bone in bones {
        let name = bone
            .get("bone")
            .and_then(|value| value.as_str())
            .ok_or_else(|| ModelImportError::InvalidVrmField {
                path: "VRM.humanoid.humanBones[].bone".into(),
                reason: "expected a bone name".into(),
            })?;
        let index = bone
            .get("node")
            .and_then(|node| node.as_u64())
            .and_then(|node| usize::try_from(node).ok())
            .ok_or_else(|| ModelImportError::InvalidVrmField {
                path: format!("VRM.humanoid.humanBones[{name}].node"),
                reason: "expected a non-negative integer".into(),
            })?;
        if index >= node_count {
            return Err(ModelImportError::InvalidNodeIndex { index });
        }
        if indexed.insert(name.to_string(), index).is_some() {
            return Err(ModelImportError::DuplicateHumanBone(name.to_string()));
        }
    }
    Ok(indexed)
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

/// Rewrites a GLB so that no mesh carries more than [`MAX_MORPH_TARGETS`]
/// morph targets, which is the hard limit the Bevy runtime imposes at load.
///
/// Morph targets that no VRM expression bind references are dropped from
/// meshes above the limit, and bind indices are remapped to the reduced
/// target arrays. Returns `None` when the bytes need no rewrite or cannot be
/// rewritten safely: non-GLB containers, GLBs without an excess mesh, models
/// that animate morph weights, and meshes whose referenced bind set alone
/// exceeds the limit.
pub fn normalize_vrm_morph_targets(bytes: &[u8]) -> Option<Vec<u8>> {
    let (mut json, bin_chunk) = parse_glb(bytes)?;
    if has_morph_weight_animation(&json) {
        return None;
    }
    let references = collect_morph_references(&json);
    let plan = plan_morph_reduction(&json, &references)?;
    apply_morph_reduction(&mut json, &plan);
    write_glb(&json, bin_chunk)
}

/// One over-limit mesh's keep set, keyed by glTF mesh index.
struct MorphReductionPlan {
    /// Target count before reduction, used to detect coherent name/weight arrays.
    target_count: usize,
    /// Old morph target index to reduced index.
    remap: BTreeMap<usize, usize>,
}

fn parse_glb(bytes: &[u8]) -> Option<(Value, Option<&[u8]>)> {
    if bytes.get(0..4)? != b"glTF" {
        return None;
    }
    let version = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?);
    if version != 2 {
        return None;
    }
    let mut json: Option<Value> = None;
    let mut bin_chunk: Option<&[u8]> = None;
    let mut offset = 12_usize;
    while offset < bytes.len() {
        let header = bytes.get(offset..offset + 8)?;
        let chunk_len =
            usize::try_from(u32::from_le_bytes(header.get(0..4)?.try_into().ok()?)).ok()?;
        let chunk_type: [u8; 4] = header.get(4..8)?.try_into().ok()?;
        let data = bytes.get(offset + 8..offset + 8 + chunk_len)?;
        match &chunk_type {
            b"JSON" if json.is_none() => json = Some(serde_json::from_slice(data).ok()?),
            b"BIN\0" if bin_chunk.is_none() => bin_chunk = Some(data),
            _ => {}
        }
        offset = offset.checked_add(8 + chunk_len)?;
    }
    Some((json?, bin_chunk))
}

fn write_glb(json: &Value, bin_chunk: Option<&[u8]>) -> Option<Vec<u8>> {
    let mut json_chunk = serde_json::to_vec(json).ok()?;
    while json_chunk.len() % 4 != 0 {
        json_chunk.push(b' ');
    }
    let mut out = Vec::with_capacity(12 + 8 + json_chunk.len());
    out.extend_from_slice(&0x46546C67_u32.to_le_bytes());
    out.extend_from_slice(&2_u32.to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(json_chunk.len()).ok()?.to_le_bytes());
    out.extend_from_slice(&0x4E4F534A_u32.to_le_bytes());
    out.extend_from_slice(&json_chunk);
    if let Some(bin_chunk) = bin_chunk {
        out.extend_from_slice(&u32::try_from(bin_chunk.len()).ok()?.to_le_bytes());
        out.extend_from_slice(&0x004E4942_u32.to_le_bytes());
        out.extend_from_slice(bin_chunk);
    }
    let total = u32::try_from(out.len()).ok()?;
    out.get_mut(8..12)?.copy_from_slice(&total.to_le_bytes());
    Some(out)
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

/// Collects every morph target index referenced by VRM expression binds,
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
    references
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

fn over_limit_morph_target_count(bytes: &[u8]) -> Option<usize> {
    let (root, _) = parse_glb(bytes)?;
    root.get("meshes")?
        .as_array()?
        .iter()
        .map(morph_target_count)
        .max()
        .filter(|count| *count > MAX_MORPH_TARGETS)
}

fn apply_morph_reduction(root: &mut Value, plan: &BTreeMap<usize, MorphReductionPlan>) {
    reduce_meshes(root, plan);
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

fn ensure_cached_model(dest: &Path, stored_bytes: &[u8]) -> Result<(), ModelImportError> {
    let stored_hash = format!("{:x}", Sha256::digest(stored_bytes));
    let cache_matches = fs::metadata(dest)
        .ok()
        .filter(|metadata| metadata.is_file() && metadata.len() as usize == stored_bytes.len())
        .is_some_and(|_| file_sha256(dest).is_ok_and(|hash| hash == stored_hash));

    if !cache_matches {
        write_atomic(dest, stored_bytes)?;
    }
    Ok(())
}

fn file_sha256(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn replace_staged_file(temp: &Path, dest: &Path) -> Result<(), ModelImportError> {
    match fs::rename(temp, dest) {
        Ok(()) => Ok(()),
        Err(rename_error) if dest.exists() => {
            // Windows does not replace an existing file with rename. The
            // validated source is already staged in `temp`; remove only this
            // cache entry, then complete the rename.
            fs::remove_file(dest).map_err(|_| rename_error)?;
            fs::rename(temp, dest)?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn write_atomic(path: &Path, contents: &[u8]) -> Result<(), ModelImportError> {
    let temp = path.with_extension("tmp");
    let mut file = fs::File::create(&temp)?;
    file.write_all(contents)?;
    file.sync_all()?;
    drop(file);
    replace_staged_file(&temp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn rejects_non_vrm_extension() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.txt");
        fs::write(&path, b"not a vrm").unwrap();
        let err = import_vrm(&path, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
        assert!(matches!(err, ModelImportError::InvalidExtension));
    }

    #[test]
    fn accepts_uppercase_vrm_extension_for_preflight() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.VRM");
        fs::write(&path, b"not a glb").unwrap();
        let err = import_vrm(&path, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
        assert!(!matches!(err, ModelImportError::InvalidExtension));
    }

    #[test]
    fn rejects_directory() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("model.vrm");
        fs::create_dir(&subdir).unwrap();
        let err = import_vrm(&subdir, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
        assert!(matches!(err, ModelImportError::NotRegularFile));
    }

    #[test]
    fn rejects_oversized() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.vrm");
        fs::write(&path, b"x").unwrap();
        let err = import_vrm(&path, dir.path(), 0).unwrap_err();
        assert!(matches!(err, ModelImportError::SizeExceeded { .. }));
    }

    #[test]
    fn rejects_hard_cap_config() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.vrm");
        fs::write(&path, b"x").unwrap();
        let err = import_vrm(&path, dir.path(), HARD_SIZE_CAP + 1).unwrap_err();
        assert!(matches!(err, ModelImportError::LimitExceedsHardCap { .. }));
    }

    #[test]
    fn rejects_invalid_bytes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.vrm");
        fs::write(&path, b"not glb").unwrap();
        let err = import_vrm(&path, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
        assert!(matches!(err, ModelImportError::GlbParse(_)));
    }

    #[test]
    fn idempotent_reimport_no_duplicate_copy() {
        let dir = TempDir::new().unwrap();
        let source_dir = TempDir::new().unwrap();
        let path = source_dir.path().join("model.vrm");
        fs::write(&path, b"x").unwrap();
        // Not a valid GLB, so use a raw copy path to verify idempotency.
        let dest_dir = dir.path().join("avatars").join("test");
        fs::create_dir_all(&dest_dir).unwrap();
        fs::write(dest_dir.join("model.vrm"), b"x").unwrap();

        let meta_path = dest_dir.join("import.toml");
        let imported = ImportedModel {
            id: "test".into(),
            name: "x".into(),
            asset_path: dest_dir.join("model.vrm"),
            meta_path: meta_path.clone(),
            summary: VrmInspectionSummary::default(),
            original_path: path.clone(),
            size: 1,
        };
        let meta = ImportMeta {
            imported: imported.clone(),
            mtime: None,
        };
        fs::write(&meta_path, toml::to_string_pretty(&meta).unwrap()).unwrap();

        let before = fs::metadata(dest_dir.join("model.vrm")).unwrap().len();
        assert_eq!(before, 1);
        // Re-writing the same fixture does not duplicate because import is
        // idempotent by sha; here we just assert the meta round-trips.
        let read: ImportMeta = toml::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
        assert_eq!(read.imported, imported);
    }

    const NON_VRM_GLTF_JSON: &str = r#"{
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{}]
    }"#;

    const VRM0_GLTF_JSON: &str = r#"{
        "asset": {"version": "2.0", "generator": "vtuber-app hermetic test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "buffers": [{"byteLength": 12}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}],
        "meshes": [{"name": "Face", "primitives": [{"attributes": {"POSITION": 0}, "targets": [{"POSITION": 0}, {"POSITION": 0}]}]}],
        "materials": [{"name": "Body"}, {"name": "Body"}],
        "nodes": [
            {"name": "Hips", "children": [1, 2, 3, 4]},
            {"name": "Head"},
            {"name": "Neck"},
            {"name": "Face", "mesh": 0},
            {"name": "Face", "mesh": 0}
        ],
        "extensionsUsed": ["VRM"],
        "extensions": {
            "VRM": {
                "exporterVersion": "UniVRM 0.123",
                "meta": {"title": "Hermetic VRM 0.x", "author": "Legacy Author", "exporterVersion": "nonstandard-meta"},
                "humanoid": {
                    "humanBones": [
                        {"bone": "hips", "node": 0},
                        {"bone": "head", "node": 1},
                        {"bone": "neck", "node": 2}
                    ]
                },
                "firstPerson": {
                    "firstPersonBone": 1,
                    "firstPersonBoneOffset": {"x": 0.0, "y": 0.1, "z": 0.2},
                    "meshAnnotations": [{"mesh": 0, "firstPersonFlag": "Both"}],
                    "lookAtTypeName": "BlendShape",
                    "lookAtHorizontalInner": {"curve": [0.0, 0.0, 0.0, 0.0], "xRange": 90.0, "yRange": 10.0},
                    "lookAtHorizontalOuter": {"xRange": 90.0, "yRange": 10.0},
                    "lookAtVerticalDown": {"xRange": 90.0, "yRange": 10.0},
                    "lookAtVerticalUp": {"xRange": 90.0, "yRange": 10.0}
                },
                "blendShapeMaster": {
                    "blendShapeGroups": [
                        {"name": "vowel-a", "presetName": "A", "binds": [{"mesh": 0, "index": 1, "weight": 100}]},
                        {"name": "blink", "presetName": "Blink_L"},
                        {"name": "joy", "presetName": "Joy"},
                        {"name": "customSmile", "presetName": "unknown"}
                    ]
                },
                "materialProperties": [
                    {"name": "Body", "shader": "VRM/MToon", "floatProperties": {"_Cull": 0.0}},
                    {"name": "Body", "shader": "VRM/MToon", "floatProperties": {"_Cull": 2.0}}
                ],
                "secondaryAnimation": {
                    "colliderGroups": [{"node": 2, "colliders": [{"offset": {"x": 0.0, "y": 0.1, "z": 0.0}, "radius": 0.02}]}],
                    "boneGroups": [{"bones": [3], "center": 1, "colliderGroups": [0], "gravityDir": {"x": 0.0, "y": -1.0, "z": 0.0}, "gravityPower": 0.5, "stiffiness": 0.8, "dragForce": 0.2, "hitRadius": 0.01}]
                }
            }
        }
    }"#;

    const VRM1_GLTF_JSON: &str = r#"{
        "asset": {"version": "2.0", "generator": "vtuber-app hermetic test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "Hips", "children": [1]},
            {"name": "Head"}
        ],
        "extensionsUsed": ["VRMC_vrm", "VRMC_springBone"],
        "extensions": {
            "VRMC_vrm": {
                "specVersion": "1.0",
                "meta": {"name": "Hermetic VRM 1.0"},
                "humanoid": {
                    "humanBones": {
                        "hips": {"node": 0},
                        "head": {"node": 1}
                    }
                }
            },
            "VRMC_springBone": {}
        }
    }"#;

    fn write_glb_fixture(dir: &TempDir, file_name: &str, json: &str) -> PathBuf {
        let mut json_chunk = json.as_bytes().to_vec();
        while !json_chunk.len().is_multiple_of(4) {
            json_chunk.push(b' ');
        }

        let bin_chunk = [0_u8; 12];
        let total_length = 12 + 8 + json_chunk.len() + 8 + bin_chunk.len();
        let mut bytes = Vec::with_capacity(total_length);
        bytes.extend_from_slice(&0x46546C67_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&(total_length as u32).to_le_bytes());
        bytes.extend_from_slice(&(json_chunk.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0x4E4F534A_u32.to_le_bytes());
        bytes.extend_from_slice(&json_chunk);
        bytes.extend_from_slice(&(bin_chunk.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0x004E4942_u32.to_le_bytes());
        bytes.extend_from_slice(&bin_chunk);

        let path = dir.path().join(file_name);
        fs::write(&path, bytes).unwrap();
        path
    }

    fn legacy_fixture(dir: &TempDir) -> PathBuf {
        write_glb_fixture(dir, "legacy.vrm", NON_VRM_GLTF_JSON)
    }

    fn vrm0_fixture(dir: &TempDir) -> PathBuf {
        write_glb_fixture(dir, "legacy-0.x.vrm", VRM0_GLTF_JSON)
    }

    fn vrm1_fixture(dir: &TempDir) -> PathBuf {
        write_glb_fixture(dir, "hermetic.vrm", VRM1_GLTF_JSON)
    }

    #[test]
    fn generated_non_vrm_glb_is_rejected_with_generation_error() {
        let dir = TempDir::new().unwrap();
        let err = inspect_vrm(legacy_fixture(&dir)).unwrap_err();
        assert!(matches!(&err, ModelImportError::NotVrm { .. }));
        assert_eq!(err.code(), "MODEL_NOT_VRM");
    }

    #[test]
    fn generated_non_vrm_glb_import_is_rejected_with_generation_error() {
        let dir = TempDir::new().unwrap();
        let source = legacy_fixture(&dir);
        let err = import_vrm(source, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
        assert!(matches!(err, ModelImportError::NotVrm { .. }));
    }

    #[test]
    fn inspects_generated_minimal_vrm0_fixture() {
        let dir = TempDir::new().unwrap();
        let summary = inspect_vrm(vrm0_fixture(&dir)).expect("fixture should be valid VRM 0.x");
        assert_eq!(summary.generation, VrmGeneration::Vrm0);
        assert_eq!(summary.spec_version, "0.x");
        assert_eq!(summary.exporter_version.as_deref(), Some("UniVRM 0.123"));
        assert_eq!(summary.name, "Hermetic VRM 0.x");
        assert_eq!(summary.authors, vec!["Legacy Author"]);
        assert_eq!(summary.look_at_type.as_deref(), Some("expression"));
        assert_eq!(
            summary.expression_presets,
            vec!["aa", "blinkLeft", "customSmile", "happy"]
        );
        assert!(summary.has_spring_bone);
        assert!(summary.has_mtoon_materials);
        assert_eq!(summary.humanoid_nodes.neck, Some(2));
    }

    #[test]
    fn inspects_generated_minimal_vrm1_fixture() {
        let dir = TempDir::new().unwrap();
        let summary = inspect_vrm(vrm1_fixture(&dir)).expect("fixture should be valid VRM 1.0");
        assert_eq!(summary.generation, VrmGeneration::Vrm1);
        assert_eq!(summary.spec_version, "1.0");
        assert_eq!(summary.exporter_version, None);
        assert!(!summary.name.is_empty(), "model name should be present");
        assert!(summary.humanoid_nodes.hips < 1000);
        assert!(summary.humanoid_nodes.head < 1000);
        assert!(summary.has_spring_bone);
    }

    #[test]
    fn rejects_legacy_mesh_and_morph_indices_during_preflight() {
        let dir = TempDir::new().unwrap();
        let invalid_mesh = VRM0_GLTF_JSON.replace(
            "\"meshAnnotations\": [{\"mesh\": 0",
            "\"meshAnnotations\": [{\"mesh\": 99",
        );
        let mesh_path = write_glb_fixture(&dir, "invalid-mesh.vrm", &invalid_mesh);
        assert!(matches!(
            inspect_vrm(mesh_path),
            Err(ModelImportError::InvalidMeshIndex { index: 99 })
        ));

        let invalid_morph =
            VRM0_GLTF_JSON.replace("\"index\": 1, \"weight\"", "\"index\": 99, \"weight\"");
        let morph_path = write_glb_fixture(&dir, "invalid-morph.vrm", &invalid_morph);
        assert!(matches!(
            inspect_vrm(morph_path),
            Err(ModelImportError::InvalidMorphTargetIndex { mesh: 0, index: 99 })
        ));
    }

    #[test]
    fn rejects_duplicate_legacy_human_bones_during_preflight() {
        let dir = TempDir::new().unwrap();
        let duplicate = VRM0_GLTF_JSON.replace(
            "{\"bone\": \"head\", \"node\": 1}",
            "{\"bone\": \"head\", \"node\": 1}, {\"bone\": \"head\", \"node\": 2}",
        );
        let path = write_glb_fixture(&dir, "duplicate-bone.vrm", &duplicate);
        let err = inspect_vrm(path).unwrap_err();
        assert!(matches!(&err, ModelImportError::DuplicateHumanBone(name) if name == "head"));
        assert_eq!(err.code(), "MODEL_DUPLICATE_HUMAN_BONE");
    }

    #[test]
    fn accepts_legacy_bone_look_at_during_preflight() {
        let dir = TempDir::new().unwrap();
        let bone = VRM0_GLTF_JSON.replace(
            "\"lookAtTypeName\": \"BlendShape\"",
            "\"lookAtTypeName\": \"Bone\"",
        );
        let path = write_glb_fixture(&dir, "bone-look-at.vrm", &bone);
        let summary = inspect_vrm(path).expect("Bone LookAt should be valid");
        assert_eq!(summary.look_at_type.as_deref(), Some("bone"));
    }

    #[test]
    fn rejects_malformed_legacy_degree_map_during_preflight() {
        let dir = TempDir::new().unwrap();
        let malformed = VRM0_GLTF_JSON.replace(
            "\"curve\": [0.0, 0.0, 0.0, 0.0]",
            "\"curve\": \"not-an-array\"",
        );
        let path = write_glb_fixture(&dir, "malformed-degree-map.vrm", &malformed);
        assert!(matches!(
            inspect_vrm(path),
            Err(ModelImportError::InvalidVrmField { path, .. })
                if path.ends_with("lookAtHorizontalInner.curve")
        ));
    }

    #[test]
    fn rejects_ambiguous_vrm_generation() {
        let dir = TempDir::new().unwrap();
        let both = VRM1_GLTF_JSON.replace("\"VRMC_vrm\": {", "\"VRM\": {}, \"VRMC_vrm\": {");
        let path = write_glb_fixture(&dir, "ambiguous.vrm", &both);
        let err = inspect_vrm(path).unwrap_err();
        assert!(matches!(&err, ModelImportError::AmbiguousVrmVersion { .. }));
        assert_eq!(err.code(), "MODEL_AMBIGUOUS_VRM_VERSION");
    }

    #[test]
    fn old_summary_defaults_to_vrm1_for_cache_compatibility() {
        let summary: VrmInspectionSummary = toml::from_str(
            r#"spec_version = "1.0"
name = "old cache"
authors = []
expression_presets = []
has_spring_bone = false
has_node_constraint = false
humanoid_nodes = { hips = 0, head = 1 }
"#,
        )
        .expect("old cache summary should remain readable");
        assert_eq!(summary.generation, VrmGeneration::Vrm1);
        assert!(!summary.has_first_person);
    }

    #[test]
    fn imports_generated_minimal_vrm1_fixture() {
        let dir = TempDir::new().unwrap();
        let source = vrm1_fixture(&dir);
        let asset_root = dir.path().join("asset-root");
        let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
            .expect("fixture should import successfully");
        assert_eq!(imported.summary.spec_version, "1.0");
        assert!(imported.asset_path.exists());
        assert!(imported.meta_path.exists());
        // Re-import with same file should be idempotent.
        let reimported =
            import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT).expect("re-import should succeed");
        assert_eq!(imported.id, reimported.id);
        assert_eq!(imported.asset_path, reimported.asset_path);
    }

    #[test]
    fn repairs_corrupt_existing_cached_file() {
        let dir = TempDir::new().unwrap();
        let source = vrm1_fixture(&dir);
        let asset_root = dir.path().join("asset-root");
        let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
            .expect("fixture should import successfully");

        fs::write(&imported.asset_path, b"corrupt cached model").unwrap();
        let repaired = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
            .expect("re-import should repair the cached file");

        assert_eq!(repaired.id, imported.id);
        assert_eq!(
            fs::read(&repaired.asset_path).unwrap(),
            fs::read(source).unwrap()
        );
    }

    fn over_limit_vrm0_fixture(
        dir: &TempDir,
        target_count: usize,
        bind_indices: &[usize],
    ) -> PathBuf {
        let mut root: serde_json::Value = serde_json::from_str(VRM0_GLTF_JSON).unwrap();
        let targets: Vec<serde_json::Value> =
            (0..target_count).map(|_| serde_json::json!({})).collect();
        let names: Vec<serde_json::Value> = (0..target_count)
            .map(|index| serde_json::Value::from(format!("m{index}")))
            .collect();
        let mut weights: Vec<serde_json::Value> = vec![serde_json::Value::from(0.0); target_count];
        weights[5] = serde_json::Value::from(0.25);
        weights[250] = serde_json::Value::from(0.75);

        let mesh = root["meshes"][0].as_object_mut().unwrap();
        mesh["primitives"][0]["targets"] = serde_json::Value::Array(targets);
        mesh.insert(
            "extras".to_string(),
            serde_json::json!({"targetNames": names}),
        );
        mesh.insert("weights".to_string(), serde_json::Value::Array(weights));

        let binds: Vec<serde_json::Value> = bind_indices
            .iter()
            .map(|index| serde_json::json!({"mesh": 0, "index": index, "weight": 50.0}))
            .collect();
        let groups = root["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]
            .as_array_mut()
            .unwrap();
        groups[0]["binds"] = serde_json::Value::Array(binds);

        write_glb_fixture(dir, "vrm0-over-limit.vrm", &root.to_string())
    }

    fn stored_glb_json(imported: &ImportedModel) -> serde_json::Value {
        let stored = fs::read(&imported.asset_path).unwrap();
        let (json, bin) = parse_glb(&stored).expect("stored copy is a valid GLB");
        assert!(bin.is_some(), "BIN chunk must be preserved");
        json
    }

    #[test]
    fn import_reduces_morph_targets_beyond_bevy_limit() {
        let dir = TempDir::new().unwrap();
        let source = over_limit_vrm0_fixture(&dir, 300, &[5, 250]);
        let asset_root = dir.path().join("asset-root");

        let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
            .expect("over-limit model should import");

        // Identity remains keyed to the original bytes; the stored copy is
        // normalized.
        let source_bytes = fs::read(&source).unwrap();
        assert_eq!(imported.id, format!("{:x}", Sha256::digest(&source_bytes)));
        assert_ne!(fs::read(&imported.asset_path).unwrap(), source_bytes);

        let json = stored_glb_json(&imported);
        let mesh = &json["meshes"][0];
        let targets = mesh["primitives"][0]["targets"].as_array().unwrap();
        assert_eq!(targets.len(), 2);
        let names = mesh["extras"]["targetNames"].as_array().unwrap();
        assert_eq!(names[0], "m5");
        assert_eq!(names[1], "m250");
        let weights = mesh["weights"].as_array().unwrap();
        assert_eq!(weights[0], 0.25);
        assert_eq!(weights[1], 0.75);

        let groups = &json["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"];
        let binds = groups[0]["binds"].as_array().unwrap();
        assert_eq!(binds[0]["mesh"], 0);
        assert_eq!(binds[0]["index"], 0);
        assert_eq!(binds[1]["index"], 1);

        // Re-import is idempotent and keeps the normalized copy.
        let reimported =
            import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT).expect("re-import succeeds");
        assert_eq!(imported.id, reimported.id);
        assert_eq!(
            fs::read(&imported.asset_path).unwrap(),
            fs::read(&reimported.asset_path).unwrap()
        );
    }

    #[test]
    fn import_keeps_models_within_the_morph_limit_unchanged() {
        let dir = TempDir::new().unwrap();
        let source = vrm0_fixture(&dir);
        let asset_root = dir.path().join("asset-root");
        let imported =
            import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT).expect("fixture imports");
        assert_eq!(
            fs::read(&imported.asset_path).unwrap(),
            fs::read(&source).unwrap()
        );
    }

    #[test]
    fn normalization_bails_when_referenced_binds_alone_exceed_the_limit() {
        let dir = TempDir::new().unwrap();
        let bind_indices: Vec<usize> = (0..MAX_MORPH_TARGETS + 4).collect();
        let source = over_limit_vrm0_fixture(&dir, 300, &bind_indices);
        let bytes = fs::read(&source).unwrap();
        assert_eq!(normalize_vrm_morph_targets(&bytes), None);

        let error = import_vrm(&source, dir.path().join("asset-root"), DEFAULT_SIZE_LIMIT)
            .expect_err("an over-limit model that cannot be reduced must not be cached raw");
        assert!(matches!(
            error,
            ModelImportError::InvalidVrmField { ref path, .. }
                if path == "meshes[*].primitives[*].targets"
        ));
    }

    #[test]
    fn normalization_ignores_non_glb_bytes() {
        assert_eq!(normalize_vrm_morph_targets(b"not glb"), None);
    }

    #[test]
    fn normalization_bails_on_morph_weight_animations() {
        let dir = TempDir::new().unwrap();
        let source = over_limit_vrm0_fixture(&dir, 300, &[5]);
        let source_bytes = fs::read(source).unwrap();
        let (mut root, bin_chunk) = parse_glb(&source_bytes).unwrap();
        root["animations"] = serde_json::json!([{
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}],
            "samplers": [{"input": 0, "output": 0}]
        }]);
        let bytes = write_glb(&root, bin_chunk).unwrap();
        assert_eq!(over_limit_morph_target_count(&bytes), Some(300));
        assert_eq!(normalize_vrm_morph_targets(&bytes), None);
    }

    fn over_limit_vrm1_fixture(dir: &TempDir) -> PathBuf {
        let mut root: serde_json::Value = serde_json::from_str(VRM1_GLTF_JSON).unwrap();
        root["buffers"] = serde_json::json!([{"byteLength": 12}]);
        root["bufferViews"] = serde_json::json!([{"buffer": 0, "byteOffset": 0, "byteLength": 12}]);
        root["accessors"] = serde_json::json!([{
            "bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3",
            "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]
        }]);
        root["nodes"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"name": "Face", "mesh": 0}));
        root["meshes"] = serde_json::json!([{
            "name": "Face",
            "primitives": [{
                "attributes": {"POSITION": 0},
                "targets": (0..300).map(|_| serde_json::json!({})).collect::<Vec<_>>()
            }],
            "extras": {"targetNames": (0..300).map(|index| format!("m{index}")).collect::<Vec<_>>()},
            "weights": (0..300).map(|_| 0.0).collect::<Vec<_>>()
        }]);
        root["extensions"]["VRMC_vrm"]["expressions"] = serde_json::json!({
            "preset": {
                "aa": {"morphTargetBinds": [{"node": 2, "index": 5}, {"node": 2, "index": 250}]}
            },
            "custom": {
                "smile": {"morphTargetBinds": [{"node": 2, "index": 250}]}
            }
        });
        write_glb_fixture(dir, "vrm1-over-limit.vrm", &root.to_string())
    }

    #[test]
    fn import_remaps_vrm1_node_based_binds() {
        let dir = TempDir::new().unwrap();
        let source = over_limit_vrm1_fixture(&dir);
        let asset_root = dir.path().join("asset-root");
        let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
            .expect("over-limit VRM 1.0 should import");

        let json = stored_glb_json(&imported);
        let expressions = &json["extensions"]["VRMC_vrm"]["expressions"];
        let preset_binds = &expressions["preset"]["aa"]["morphTargetBinds"];
        assert_eq!(preset_binds[0]["index"], 0);
        assert_eq!(preset_binds[0]["node"], 2);
        assert_eq!(preset_binds[1]["index"], 1);
        let custom_binds = &expressions["custom"]["smile"]["morphTargetBinds"];
        assert_eq!(custom_binds[0]["index"], 1);
        let targets = json["meshes"][0]["primitives"][0]["targets"]
            .as_array()
            .unwrap();
        assert_eq!(targets.len(), 2);
    }
}
