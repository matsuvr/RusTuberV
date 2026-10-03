//! Conversion and preparation of the managed VRM copy for the avatar runtime.

use std::{borrow::Cow, fs, io, path::Path};
use vtuber_avatar::glb::Glb;

use super::{ModelImportError, VrmGeneration, normalize_vrm_morph_targets};

/// Converts a VRM 0.x source into VRM 1.0-shaped bytes for the runtime and
/// adapts VRM 1.0 sources to the upstream expression contract.
///
/// `generation` is the preflight classification; the adaptation itself is
/// selected from the root extension by
/// `vtuber_avatar::prepare_managed_vrm_bytes`. VRM 0.x sources are
/// normalized into the VRM 1.0 shape, and VRM 1.0 sources get the app-side
/// expression adaptation: author-defined `custom` expressions are merged
/// into the `preset` map the unmodified upstream runtime reads, and the
/// optional `isBinary` / `override*` fields the upstream serde requires are
/// filled with their specification defaults. Conversion failures become
/// [`ModelImportError`]s; the source file itself is never modified.
pub fn runtime_ready_source_bytes(
    source_bytes: &[u8],
    generation: VrmGeneration,
) -> Result<Vec<u8>, ModelImportError> {
    match vtuber_avatar::prepare_managed_vrm_bytes(source_bytes) {
        Ok(Some(prepared)) => Ok(prepared),
        Ok(None) => match generation {
            VrmGeneration::Vrm0 => Err(ModelImportError::NotVrm {
                reason:
                    "preflight classified the source as VRM 0.x but it carries no VRM extension"
                        .to_string(),
            }),
            VrmGeneration::Vrm1 => Ok(source_bytes.to_vec()),
        },
        Err(error) => Err(vrm0_convert_error(error)),
    }
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
    let runtime_bytes = match vtuber_avatar::prepare_managed_vrm_bytes(&current) {
        Ok(Some(prepared)) => Cow::Owned(prepared),
        Ok(None) => Cow::Borrowed(current.as_slice()),
        Err(error) => return Err(vrm0_convert_error(error)),
    };
    let stored_bytes = normalize_vrm_morph_targets(&runtime_bytes)?
        .map(Cow::Owned)
        .unwrap_or(runtime_bytes);
    if stored_bytes.as_ref() != current {
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

fn vrm0_convert_error(error: vtuber_avatar::Vrm0ConvertError) -> ModelImportError {
    use vtuber_avatar::Vrm0ConvertError as ConvertError;
    match error {
        ConvertError::Descriptor(error) => error.into(),
        ConvertError::Glb(error) => ModelImportError::GlbParse(error.to_string()),
        ConvertError::NotVrm0 => ModelImportError::NotVrm {
            reason: "no VRM 0.x extension to convert".to_string(),
        },
        ConvertError::InvalidField { path, reason } => {
            ModelImportError::InvalidVrmField { path, reason }
        }
    }
}
