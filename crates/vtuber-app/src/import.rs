//! VRM 0.x/1.0 model import and lightweight preflight inspection.
//!
//! Imports a user-selected file into an application-managed asset source and
//! verifies that it is a supported VRM generation before it reaches the
//! `bevy_vrm1` compatibility boundary.
//!
//! This module owns file validation and managed storage. `inspection` reads
//! format metadata, `runtime` prepares loadable bytes, and `morph` reduces
//! over-limit morph target arrays. The public import API remains here.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
pub use vtuber_avatar::vrm::VrmGeneration;
use vtuber_avatar::vrm::{VrmParseError, VrmPrepareError};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

mod inspection;
mod morph;
mod runtime;

pub use inspection::inspect_vrm;
use morph::normalize_vrm_morph_targets;
use morph::over_limit_morph_target_count;
pub use runtime::{
    ensure_managed_model_ready, parse_runtime_source_facts, read_runtime_expression_facts,
    read_runtime_source_facts, runtime_ready_source_bytes,
};

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

impl From<VrmParseError> for ModelImportError {
    fn from(error: VrmParseError) -> Self {
        match error {
            VrmParseError::InvalidMorphTargetIndex { mesh, index } => {
                Self::InvalidMorphTargetIndex { mesh, index }
            }
            VrmParseError::DuplicateBone(name) => Self::DuplicateHumanBone(name),
            VrmParseError::MissingField(path) => {
                if let Some(name) = path.strip_prefix("humanoid.humanBones.") {
                    Self::MissingRequiredBone(name.into())
                } else {
                    Self::InvalidVrmField {
                        path,
                        reason: "required field is missing".into(),
                    }
                }
            }
            VrmParseError::InvalidField { path, reason } => Self::InvalidVrmField { path, reason },
            VrmParseError::InvalidIndex { path, index } if path.ends_with(".mesh") => {
                Self::InvalidMeshIndex { index }
            }
            VrmParseError::InvalidIndex { index, .. } => Self::InvalidNodeIndex { index },
            VrmParseError::MissingGeneration => Self::NotVrm {
                reason: error.to_string(),
            },
            VrmParseError::AmbiguousGeneration => Self::AmbiguousVrmVersion {
                reason: error.to_string(),
            },
            VrmParseError::UnsupportedVersion(version) => Self::UnsupportedVersion(version),
        }
    }
}

impl From<VrmPrepareError> for ModelImportError {
    fn from(error: VrmPrepareError) -> Self {
        match error {
            VrmPrepareError::Descriptor(error) => error.into(),
            VrmPrepareError::Glb(error) => Self::GlbParse(error.to_string()),
            VrmPrepareError::InvalidField { path, reason } => {
                Self::InvalidVrmField { path, reason }
            }
        }
    }
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

/// Source provenance and capabilities inspected from the prepared VRM 1.0 document.
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
    /// Number of prepared VRM 1.0 MToon materials.
    pub mtoon_material_count: usize,
    /// Number of material entries classified as unlit.
    pub unlit_material_count: usize,
    /// Number of material entries that use the StandardMaterial fallback.
    pub fallback_material_count: usize,
    /// Number of springs in the prepared VRMC_springBone extension.
    /// This is document inventory; runtime entities are reported separately.
    pub spring_chain_count: usize,
    /// Number of ordered joint references after legacy hierarchy expansion.
    pub spring_joint_count: usize,
    /// Number of colliders in the prepared VRMC_springBone extension.
    pub spring_collider_count: usize,
    /// Number of prepared springs declaring a center node.
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
/// VRM 0.x sources are converted into VRM 1.0-shaped bytes by
/// `vtuber_avatar::vrm::prepare_vrm_document` before inspection and storage, so the
/// upstream runtime loads the managed copy directly. When the source declares
/// more morph targets per mesh than the Bevy runtime supports, the stored
/// copy is additionally normalized to the runtime morph-target limit; the
/// identity hash always refers to the original source bytes.
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

    let source_bytes = fs::read(source)?;
    let id = format!("{:x}", Sha256::digest(&source_bytes));
    let (summary, runtime_bytes) = inspection::prepare_and_inspect_vrm(source, &source_bytes)?;
    let stored_bytes = runtime::finish_runtime_bytes(runtime_bytes)?;

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
    crate::file_io::replace_file(&meta_path, meta_text.as_bytes())?;

    Ok(imported)
}

fn ensure_cached_model(dest: &Path, stored_bytes: &[u8]) -> Result<(), ModelImportError> {
    let stored_hash = format!("{:x}", Sha256::digest(stored_bytes));
    let cache_matches = fs::metadata(dest)
        .ok()
        .filter(|metadata| metadata.is_file() && metadata.len() as usize == stored_bytes.len())
        .is_some_and(|_| file_sha256(dest).is_ok_and(|hash| hash == stored_hash));

    if !cache_matches {
        crate::file_io::replace_file(dest, stored_bytes)?;
    }
    Ok(())
}

fn file_sha256(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests;
