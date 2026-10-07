//! Conversion and preparation of the managed VRM copy for the avatar runtime.

use std::{fs, io, path::Path};
use vtuber_avatar::glb::Glb;

use super::{
    MAX_MORPH_TARGETS, ModelImportError, normalize_vrm_morph_targets, over_limit_morph_target_count,
};

/// Converts VRM 0.x first, then prepares either source through the same VRM 1.0 path.
/// Format classification is read from the bytes, never supplied by the caller.
pub fn runtime_ready_source_bytes(source_bytes: &[u8]) -> Result<Vec<u8>, ModelImportError> {
    let prepared = vtuber_avatar::prepare_managed_vrm_bytes(source_bytes)?
        .unwrap_or_else(|| source_bytes.to_vec());
    finish_runtime_bytes(prepared)
}

/// Applies the common VRM 1.0 morph limit after format conversion.
pub(super) fn finish_runtime_bytes(bytes: Vec<u8>) -> Result<Vec<u8>, ModelImportError> {
    if let Some(normalized) = normalize_vrm_morph_targets(&bytes)? {
        return Ok(normalized);
    }
    if let Some(target_count) = over_limit_morph_target_count(&bytes) {
        return Err(ModelImportError::InvalidVrmField {
            path: "meshes[*].primitives[*].targets".into(),
            reason: format!(
                "{target_count} morph targets exceed the runtime limit of {MAX_MORPH_TARGETS} and cannot be reduced without changing the default shape or animation"
            ),
        });
    }
    Ok(bytes)
}

/// Upgrades a managed copy stored by an older application version.
///
/// Reads the managed copy and applies the shared format adaptation
/// (`vtuber_avatar::prepare_managed_vrm_bytes`): an unconverted VRM 0.x copy
/// (raw root `VRM` extension) is converted into the VRM 1.0 shape, and a
/// VRM 1.0 copy gets the expression adaptation (custom merge and omitted
/// spec defaults). Morph-target normalization is re-applied on top, matching
/// the import pipeline. The managed copy alone carries everything the
/// adaptation needs, so a moved or deleted original file does not block it,
/// and the user's source file is never read or rewritten.
///
/// The managed copy is a cache. Returns `true` when the file was rewritten.
pub fn ensure_managed_model_ready(managed_path: &Path) -> Result<bool, ModelImportError> {
    let current = match fs::read(managed_path) {
        Ok(current) => current,
        // There is no managed copy to adapt; the runtime asset load surfaces
        // the missing file as an avatar load failure.
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let stored_bytes = runtime_ready_source_bytes(&current)?;
    if stored_bytes != current {
        crate::file_io::replace_file(managed_path, &stored_bytes)?;
        return Ok(true);
    }
    Ok(false)
}

/// Reads the runtime expression facts from a managed model.
///
/// The managed copy is VRM 1.0-shaped (VRM 0.x sources are converted at
/// import or by [`ensure_managed_model_ready`]), so the `VRMC_vrm`
/// expression section — including the app-retained `custom` origin record
/// and material bind entries — is the single source of truth for binding.
///
/// Read and container failures propagate as [`ModelImportError`].
pub fn read_runtime_expression_facts(
    managed_path: &Path,
) -> Result<vtuber_avatar::SourceExpressions, ModelImportError> {
    let bytes = fs::read(managed_path)?;
    let glb = Glb::parse(&bytes).map_err(|error| ModelImportError::GlbParse(error.to_string()))?;
    let json = glb.document;
    Ok(vtuber_avatar::parse_source_expressions(&json))
}

/// Reads expression facts and every helper constraint in one managed-model read.
///
/// Read, container and invalid-constraint errors propagate as [`ModelImportError`].
pub fn read_runtime_source_facts(
    managed_path: &Path,
) -> Result<
    (
        vtuber_avatar::SourceExpressions,
        vtuber_avatar::node_constraints::SourceNodeConstraints,
    ),
    ModelImportError,
> {
    let bytes = fs::read(managed_path)?;
    let glb = Glb::parse(&bytes).map_err(|error| ModelImportError::GlbParse(error.to_string()))?;
    let constraints = vtuber_avatar::node_constraints::parse_source_node_constraints(&glb.document)
        .map_err(|index| ModelImportError::InvalidVrmField {
            path: format!("nodes[{index}].extensions.VRMC_node_constraint"),
            reason: "invalid constraint source, axis or weight".into(),
        })?;
    Ok((
        vtuber_avatar::parse_source_expressions(&glb.document),
        constraints,
    ))
}
