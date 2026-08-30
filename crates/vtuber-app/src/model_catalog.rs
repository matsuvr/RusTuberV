//! Legacy manifest-driven composite pipeline descriptors for research tools.
//!
//! The desktop runtime does not call this module. It remains available because
//! the old detector/crop artifacts are useful for historical replay and
//! evaluation, but callers must opt into the `legacy-face-stack` inference
//! feature through an explicitly named research command.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};
use vtuber_inference::{
    ChannelOrder, CropInterpolation, CropOutsideFill, DetectorPostprocessConfig, FaceCropConfig,
    FacePipelineDescriptor, InputValueDomain, ModelArtifactDescriptor, ModelRole,
    NormalizationContract, OutputTensorContract, TensorContract, TensorLayout,
};

/// Errors while reading or validating the face pipeline manifest.
#[derive(Debug, thiserror::Error)]
pub enum ModelCatalogError {
    /// The manifest could not be read.
    #[error("cannot read model manifest {path}: {source}")]
    Io {
        /// File that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The manifest is syntactically invalid.
    #[error("cannot parse model manifest: {0}")]
    Parse(#[from] toml::de::Error),
    /// A required manifest field is absent or malformed.
    #[error("invalid model manifest field `{field}`: {reason}")]
    Invalid {
        /// Manifest field path.
        field: String,
        /// Validation explanation.
        reason: String,
    },
    /// Two model entries use the same stable ID.
    #[error("duplicate model ID `{id}`")]
    DuplicateModelId {
        /// Duplicated stable ID.
        id: String,
    },
    /// More than one artifact claims the same pipeline role.
    #[error("duplicate model role `{role}`")]
    DuplicateRole {
        /// Duplicated role name.
        role: String,
    },
    /// A pipeline model reference does not resolve to an artifact.
    #[error("pipeline `{pipeline_id}` references missing model `{model_id}`")]
    PipelineReferenceMissing {
        /// Legacy research pipeline ID.
        pipeline_id: String,
        /// Missing model ID.
        model_id: String,
    },
    /// A pipeline reference resolves to the wrong model role.
    #[error(
        "pipeline `{pipeline_id}` model `{model_id}` has role `{actual_role}`, expected `{expected_role}`"
    )]
    PipelineReferenceRoleMismatch {
        /// Legacy research pipeline ID.
        pipeline_id: String,
        /// Referenced model ID.
        model_id: String,
        /// Actual role in the manifest.
        actual_role: String,
        /// Required role.
        expected_role: String,
    },
    /// A manifest-listed artifact is not present on disk.
    #[error("model artifact `{model_id}` is missing: {path}")]
    ArtifactMissing {
        /// Stable model ID.
        model_id: String,
        /// Resolved artifact path.
        path: PathBuf,
    },
    /// A model artifact has a different byte size than the manifest.
    #[error(
        "model artifact `{model_id}` size mismatch: expected {expected} bytes, got {actual} bytes"
    )]
    ArtifactSizeMismatch {
        /// Stable model ID.
        model_id: String,
        /// Manifest size.
        expected: u64,
        /// Actual size.
        actual: u64,
    },
    /// A model artifact has a different SHA-256 than the manifest.
    #[error("model artifact `{model_id}` SHA-256 mismatch: expected {expected}, got {actual}")]
    ArtifactHashMismatch {
        /// Stable model ID.
        model_id: String,
        /// Manifest digest.
        expected: String,
        /// Actual digest.
        actual: String,
    },
}

/// Loads the legacy research pipeline from the workspace model manifest.
pub fn load_research_pipeline(
    project_root: &Path,
) -> Result<FacePipelineDescriptor, ModelCatalogError> {
    load_pipeline_from_manifest(
        &project_root
            .join("assets")
            .join("models")
            .join("manifest.toml"),
    )
}

/// Returns the directory containing artifacts referenced by the legacy
/// research pipeline descriptor.
#[must_use]
pub fn research_artifact_root(project_root: &Path) -> PathBuf {
    project_root.join("assets").join("models")
}

/// Loads a legacy research pipeline from an explicit manifest path.
pub fn load_pipeline_from_manifest(
    manifest_path: &Path,
) -> Result<FacePipelineDescriptor, ModelCatalogError> {
    let text = std::fs::read_to_string(manifest_path).map_err(|source| ModelCatalogError::Io {
        path: manifest_path.to_path_buf(),
        source,
    })?;
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    parse_pipeline_manifest(&text, manifest_dir)
}

/// Verifies both legacy research artifacts against size and SHA-256.
pub fn verify_research_pipeline_artifacts(
    manifest_path: &Path,
) -> Result<FacePipelineDescriptor, ModelCatalogError> {
    let pipeline = load_pipeline_from_manifest(manifest_path)?;
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    verify_artifact(manifest_dir, &pipeline.detector)?;
    verify_artifact(manifest_dir, &pipeline.landmarks)?;
    Ok(pipeline)
}

fn parse_pipeline_manifest(
    text: &str,
    _manifest_dir: &Path,
) -> Result<FacePipelineDescriptor, ModelCatalogError> {
    let value: toml::Value = toml::from_str(text)?;
    let root = value
        .as_table()
        .ok_or_else(|| invalid("root", "manifest root must be a table"))?;
    let pipeline = required_table(root, "legacy_research_pipeline")?;
    let pipeline_id = required_string(pipeline, "id", "legacy_research_pipeline")?;
    let detector_id = required_string(pipeline, "detector_model", "legacy_research_pipeline")?;
    let landmark_id = required_string(pipeline, "landmark_model", "legacy_research_pipeline")?;
    let model_values = root
        .get("models")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| invalid("models", "an array of model artifacts is required"))?;

    let mut models = HashMap::with_capacity(model_values.len());
    let mut roles = HashMap::new();
    for (index, model_value) in model_values.iter().enumerate() {
        let model = model_value
            .as_table()
            .ok_or_else(|| invalid(format!("models[{index}]"), "expected a table"))?;
        let artifact = parse_artifact(model, index)?;
        if models
            .insert(artifact.id.clone(), artifact.clone())
            .is_some()
        {
            return Err(ModelCatalogError::DuplicateModelId { id: artifact.id });
        }
        let role = role_name(artifact.role);
        if roles.insert(role.clone(), artifact.id.clone()).is_some() {
            return Err(ModelCatalogError::DuplicateRole { role });
        }
    }

    let detector = resolve_model(&models, &pipeline_id, &detector_id, ModelRole::FaceDetector)?;
    let landmarks = resolve_model(
        &models,
        &pipeline_id,
        &landmark_id,
        ModelRole::FaceLandmarks,
    )?;
    let detector_postprocess = parse_detector_postprocess(pipeline)?;
    let crop = parse_crop_config(pipeline)?;

    Ok(FacePipelineDescriptor {
        id: pipeline_id,
        detector,
        landmarks,
        detector_postprocess,
        crop,
    })
}

fn parse_artifact(
    model: &toml::map::Map<String, toml::Value>,
    index: usize,
) -> Result<ModelArtifactDescriptor, ModelCatalogError> {
    let prefix = format!("models[{index}]");
    let id = required_string(model, "id", &prefix)?;
    let role = parse_role(&required_string(model, "role", &prefix)?, &prefix)?;
    let file_text = required_string(model, "file", &prefix)?;
    let file = PathBuf::from(&file_text);
    if file.is_absolute()
        || file.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(invalid(
            format!("{prefix}.file"),
            "artifact path must be relative to the manifest directory",
        ));
    }
    let byte_size = required_u64(model, "byte_size", &prefix)?;
    if byte_size == 0 {
        return Err(invalid(
            format!("{prefix}.byte_size"),
            "artifact size must be positive",
        ));
    }
    let sha256 = required_string(model, "sha256", &prefix)?.to_ascii_lowercase();
    if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(
            format!("{prefix}.sha256"),
            "expected a 64-character hexadecimal SHA-256 digest",
        ));
    }
    let input_name = required_string(model, "input_name", &prefix)?;
    let source = required_string(model, "source", &prefix)?;
    let upstream = required_string(model, "upstream", &prefix)?;
    let license = required_string(model, "license", &prefix)?;
    let license_url = optional_string(model, "license_url", &prefix)?;
    let input = parse_tensor_contract(required_table(model, "input")?, &format!("{prefix}.input"))?;
    let outputs = parse_outputs(required_array(model, "outputs", &prefix)?, &prefix)?;
    let requires_crop = model
        .get("requires_crop")
        .and_then(toml::Value::as_bool)
        .ok_or_else(|| invalid(format!("{prefix}.requires_crop"), "expected a boolean"))?;
    let schema = optional_string(model, "schema", &prefix)?;
    let landmark_coordinate_encoding =
        optional_string(model, "landmark_coordinate_encoding", &prefix)?;
    let pose_method = optional_string(model, "pose_method", &prefix)?;
    let representative_indices = match model.get("representative_indices") {
        Some(value) => parse_usize_array(value, &format!("{prefix}.representative_indices"))?,
        None => Vec::new(),
    };

    if matches!(role, ModelRole::FaceLandmarks)
        && (schema.is_none()
            || !requires_crop
            || landmark_coordinate_encoding.is_none()
            || pose_method.is_none()
            || representative_indices.is_empty())
    {
        return Err(invalid(
            prefix,
            "landmark artifacts require schema, crop, coordinate encoding, pose method, and representative indices",
        ));
    }

    Ok(ModelArtifactDescriptor {
        id,
        role,
        file,
        byte_size,
        sha256,
        input_name,
        source,
        upstream,
        license,
        license_url,
        input,
        outputs,
        requires_crop,
        schema,
        landmark_coordinate_encoding,
        pose_method,
        representative_indices,
    })
}

fn parse_tensor_contract(
    table: &toml::map::Map<String, toml::Value>,
    prefix: &str,
) -> Result<TensorContract, ModelCatalogError> {
    let shape_value = table
        .get("shape")
        .ok_or_else(|| invalid(format!("{prefix}.shape"), "field is required"))?;
    let shape = parse_usize_array(shape_value, &format!("{prefix}.shape"))?;
    if shape.is_empty() {
        return Err(invalid(
            format!("{prefix}.shape"),
            "shape must not be empty",
        ));
    }
    let dtype = required_string(table, "dtype", prefix)?;
    let layout_text = required_string(table, "layout", prefix)?;
    let layout = match layout_text.as_str() {
        "NCHW" => TensorLayout::Nchw,
        "NHWC" => TensorLayout::Nhwc,
        _ => return Err(invalid(format!("{prefix}.layout"), "expected NCHW or NHWC")),
    };
    let channel_order = match required_string(table, "channel_order", prefix)?.as_str() {
        "RGB" => ChannelOrder::Rgb,
        "BGR" => ChannelOrder::Bgr,
        "RGBA" => ChannelOrder::Rgba,
        "BGRA" => ChannelOrder::Bgra,
        "GRAY" => ChannelOrder::Gray,
        _ => {
            return Err(invalid(
                format!("{prefix}.channel_order"),
                "unknown channel order",
            ));
        }
    };
    let value_domain = match required_string(table, "value_domain", prefix)?.as_str() {
        "raw_u8" => InputValueDomain::RawU8,
        "unit_float" => InputValueDomain::UnitFloat,
        _ => {
            return Err(invalid(
                format!("{prefix}.value_domain"),
                "unknown value domain",
            ));
        }
    };
    let normalization = required_table(table, "normalization")?;
    let mean_value = normalization
        .get("mean")
        .ok_or_else(|| invalid(format!("{prefix}.normalization.mean"), "field is required"))?;
    let scale_value = normalization
        .get("scale")
        .ok_or_else(|| invalid(format!("{prefix}.normalization.scale"), "field is required"))?;
    let mean = parse_float3(mean_value, &format!("{prefix}.normalization.mean"))?;
    let scale = parse_float3(scale_value, &format!("{prefix}.normalization.scale"))?;
    if scale.iter().any(|value| *value <= 0.0) {
        return Err(invalid(
            format!("{prefix}.normalization.scale"),
            "scale values must be positive",
        ));
    }
    Ok(TensorContract {
        shape,
        dtype,
        layout,
        channel_order,
        value_domain,
        normalization: NormalizationContract { mean, scale },
    })
}

fn parse_outputs(
    values: &[toml::Value],
    prefix: &str,
) -> Result<Vec<OutputTensorContract>, ModelCatalogError> {
    if values.is_empty() {
        return Err(invalid(
            format!("{prefix}.outputs"),
            "at least one output contract is required",
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let table = value
                .as_table()
                .ok_or_else(|| invalid(format!("{prefix}.outputs[{index}]"), "expected a table"))?;
            let output_prefix = format!("{prefix}.outputs[{index}]");
            Ok(OutputTensorContract {
                name: required_string(table, "name", &output_prefix)?,
                shape: parse_usize_array(
                    table.get("shape").ok_or_else(|| {
                        invalid(format!("{output_prefix}.shape"), "field is required")
                    })?,
                    &format!("{output_prefix}.shape"),
                )?,
                dtype: required_string(table, "dtype", &output_prefix)?,
                description: required_string(table, "description", &output_prefix)?,
            })
        })
        .collect()
}

fn parse_detector_postprocess(
    pipeline: &toml::map::Map<String, toml::Value>,
) -> Result<DetectorPostprocessConfig, ModelCatalogError> {
    let table = required_table(pipeline, "detector_postprocess")?;
    let score_threshold =
        parse_bounded_float(table, "score_threshold", 0.0, 1.0, "detector_postprocess")?;
    let nms_iou = parse_bounded_float(table, "nms_iou", 0.0, 1.0, "detector_postprocess")?;
    let max_pre_nms_candidates =
        required_usize(table, "max_pre_nms_candidates", "detector_postprocess")?;
    let max_post_nms_detections =
        required_usize(table, "max_post_nms_detections", "detector_postprocess")?;
    if max_pre_nms_candidates == 0 || max_post_nms_detections == 0 {
        return Err(invalid(
            "legacy_research_pipeline.detector_postprocess",
            "candidate limits must be positive",
        ));
    }
    Ok(DetectorPostprocessConfig {
        score_threshold,
        nms_iou,
        max_pre_nms_candidates,
        max_post_nms_detections,
    })
}

fn parse_crop_config(
    pipeline: &toml::map::Map<String, toml::Value>,
) -> Result<FaceCropConfig, ModelCatalogError> {
    let table = required_table(pipeline, "crop")?;
    let square_scale = parse_positive_float(table, "square_scale", "crop")?;
    let center_y_offset_fraction = parse_float_field(table, "center_y_offset_fraction", "crop")?;
    let output_size_values = parse_usize_array(
        table.get("output_size").ok_or_else(|| {
            invalid(
                "legacy_research_pipeline.crop.output_size",
                "field is required",
            )
        })?,
        "legacy_research_pipeline.crop.output_size",
    )?;
    let output_size: [usize; 2] = output_size_values.try_into().map_err(|_| {
        invalid(
            "legacy_research_pipeline.crop.output_size",
            "expected exactly width and height",
        )
    })?;
    if output_size.contains(&0) {
        return Err(invalid(
            "legacy_research_pipeline.crop.output_size",
            "dimensions must be positive",
        ));
    }
    let interpolation = match required_string(table, "interpolation", "crop")?.as_str() {
        "bilinear" => CropInterpolation::Bilinear,
        _ => {
            return Err(invalid(
                "legacy_research_pipeline.crop.interpolation",
                "unsupported mode",
            ));
        }
    };
    let outside_fill = match required_string(table, "outside_fill", "crop")?.as_str() {
        "normalization_mean" => CropOutsideFill::NormalizationMean,
        _ => {
            return Err(invalid(
                "legacy_research_pipeline.crop.outside_fill",
                "unsupported mode",
            ));
        }
    };
    Ok(FaceCropConfig {
        square_scale,
        center_y_offset_fraction,
        output_size,
        interpolation,
        outside_fill,
    })
}

fn resolve_model(
    models: &HashMap<String, ModelArtifactDescriptor>,
    pipeline_id: &str,
    model_id: &str,
    expected_role: ModelRole,
) -> Result<ModelArtifactDescriptor, ModelCatalogError> {
    let artifact =
        models
            .get(model_id)
            .ok_or_else(|| ModelCatalogError::PipelineReferenceMissing {
                pipeline_id: pipeline_id.to_owned(),
                model_id: model_id.to_owned(),
            })?;
    if artifact.role != expected_role {
        return Err(ModelCatalogError::PipelineReferenceRoleMismatch {
            pipeline_id: pipeline_id.to_owned(),
            model_id: model_id.to_owned(),
            actual_role: role_name(artifact.role),
            expected_role: role_name(expected_role),
        });
    }
    Ok(artifact.clone())
}

fn verify_artifact(
    manifest_dir: &Path,
    artifact: &ModelArtifactDescriptor,
) -> Result<(), ModelCatalogError> {
    let path = manifest_dir.join(&artifact.file);
    let metadata = std::fs::metadata(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ModelCatalogError::ArtifactMissing {
                model_id: artifact.id.clone(),
                path: path.clone(),
            }
        } else {
            ModelCatalogError::Io {
                path: path.clone(),
                source: error,
            }
        }
    })?;
    if !metadata.is_file() {
        return Err(ModelCatalogError::ArtifactMissing {
            model_id: artifact.id.clone(),
            path,
        });
    }
    if metadata.len() != artifact.byte_size {
        return Err(ModelCatalogError::ArtifactSizeMismatch {
            model_id: artifact.id.clone(),
            expected: artifact.byte_size,
            actual: metadata.len(),
        });
    }
    let bytes = std::fs::read(&path).map_err(|source| ModelCatalogError::Io {
        path: path.clone(),
        source,
    })?;
    let actual = sha256_hex(&Sha256::digest(&bytes));
    if actual != artifact.sha256 {
        return Err(ModelCatalogError::ArtifactHashMismatch {
            model_id: artifact.id.clone(),
            expected: artifact.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn sha256_hex(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(digest.len() * 2);
    for byte in digest {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

fn parse_role(value: &str, prefix: &str) -> Result<ModelRole, ModelCatalogError> {
    match value {
        "face_detector" => Ok(ModelRole::FaceDetector),
        "face_landmarks" => Ok(ModelRole::FaceLandmarks),
        _ => Err(invalid(format!("{prefix}.role"), "unknown model role")),
    }
}

fn role_name(role: ModelRole) -> String {
    match role {
        ModelRole::FaceDetector => "face_detector".to_owned(),
        ModelRole::FaceLandmarks => "face_landmarks".to_owned(),
    }
}

fn required_table<'a>(
    table: &'a toml::map::Map<String, toml::Value>,
    key: &str,
) -> Result<&'a toml::map::Map<String, toml::Value>, ModelCatalogError> {
    table
        .get(key)
        .and_then(toml::Value::as_table)
        .ok_or_else(|| invalid(key, "a table is required"))
}

fn required_array<'a>(
    table: &'a toml::map::Map<String, toml::Value>,
    key: &str,
    prefix: &str,
) -> Result<&'a [toml::Value], ModelCatalogError> {
    table
        .get(key)
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("{prefix}.{key}"), "an array is required"))
}

fn required_string(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
    prefix: &str,
) -> Result<String, ModelCatalogError> {
    let value = table
        .get(key)
        .and_then(toml::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid(format!("{prefix}.{key}"), "a non-empty string is required"))?;
    Ok(value.to_owned())
}

fn optional_string(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
    prefix: &str,
) -> Result<Option<String>, ModelCatalogError> {
    match table.get(key) {
        None => Ok(None),
        Some(value) => value
            .as_str()
            .filter(|value| !value.is_empty())
            .map(|value| Some(value.to_owned()))
            .ok_or_else(|| invalid(format!("{prefix}.{key}"), "expected a non-empty string")),
    }
}

fn required_u64(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
    prefix: &str,
) -> Result<u64, ModelCatalogError> {
    let value = table
        .get(key)
        .and_then(toml::Value::as_integer)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| invalid(format!("{prefix}.{key}"), "expected a non-negative integer"))?;
    Ok(value)
}

fn required_usize(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
    prefix: &str,
) -> Result<usize, ModelCatalogError> {
    required_u64(table, key, prefix).and_then(|value| {
        usize::try_from(value)
            .map_err(|_| invalid(format!("{prefix}.{key}"), "integer does not fit usize"))
    })
}

fn parse_usize_array(value: &toml::Value, field: &str) -> Result<Vec<usize>, ModelCatalogError> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid(field, "expected an array of positive integers"))?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let parsed = value
                .as_integer()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    invalid(format!("{field}[{index}]"), "expected a positive integer")
                })?;
            if parsed == 0 {
                return Err(invalid(
                    format!("{field}[{index}]"),
                    "value must be positive",
                ));
            }
            Ok(parsed)
        })
        .collect()
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn parse_float3(value: &toml::Value, field: &str) -> Result<[f32; 3], ModelCatalogError> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid(field, "expected three finite numbers"))?;
    if values.len() != 3 {
        return Err(invalid(field, "expected exactly three values"));
    }
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_float()
            .or_else(|| value.as_integer().map(|value| value as f64))
            .filter(|value| value.is_finite())
            .map(|value| value as f32)
            .ok_or_else(|| invalid(format!("{field}[{index}]"), "expected a finite number"))?;
    }
    Ok(result)
}

fn parse_float_field(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
    prefix: &str,
) -> Result<f32, ModelCatalogError> {
    table
        .get(key)
        .and_then(|value| {
            value
                .as_float()
                .or_else(|| value.as_integer().map(|value| value as f64))
        })
        .filter(|value| value.is_finite())
        .map(|value| value as f32)
        .ok_or_else(|| invalid(format!("{prefix}.{key}"), "expected a finite number"))
}

fn parse_positive_float(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
    prefix: &str,
) -> Result<f32, ModelCatalogError> {
    let value = parse_float_field(table, key, prefix)?;
    if value <= 0.0 {
        return Err(invalid(format!("{prefix}.{key}"), "value must be positive"));
    }
    Ok(value)
}

fn parse_bounded_float(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
    min: f32,
    max: f32,
    prefix: &str,
) -> Result<f32, ModelCatalogError> {
    let value = parse_float_field(table, key, prefix)?;
    if !(min..=max).contains(&value) {
        return Err(invalid(
            format!("{prefix}.{key}"),
            format!("value must be between {min} and {max}"),
        ));
    }
    Ok(value)
}

fn invalid(field: impl Into<String>, reason: impl Into<String>) -> ModelCatalogError {
    ModelCatalogError::Invalid {
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repository_manifest_text() -> String {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate is nested beneath workspace root");
        std::fs::read_to_string(root.join("assets/models/manifest.toml"))
            .expect("repository manifest should be readable")
    }

    #[test]
    fn repository_manifest_resolves_stable_pipeline_ids() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate is nested beneath workspace root");
        let pipeline = load_research_pipeline(root).expect("pipeline should parse");
        assert_eq!(pipeline.id, "ultraface-rfb-320-peppapig-98");
        assert_eq!(pipeline.detector.id, "ultraface-rfb-320");
        assert_eq!(pipeline.landmarks.id, "peppapig-98");
        assert_eq!(pipeline.detector_postprocess.max_pre_nms_candidates, 256);
        assert_eq!(pipeline.crop.output_size, [256, 256]);
    }

    #[test]
    fn duplicate_role_is_rejected_before_pipeline_resolution() {
        let text = repository_manifest_text()
            .replace(r#"role = "face_landmarks""#, r#"role = "face_detector""#);
        let error = parse_pipeline_manifest(&text, Path::new("assets/models"))
            .expect_err("duplicate role should fail");
        assert!(matches!(error, ModelCatalogError::DuplicateRole { .. }));
    }

    #[test]
    fn missing_artifact_reports_stable_model_id() {
        let text = repository_manifest_text().replace(
            r#"file = "version-RFB-320.onnx""#,
            r#"file = "missing-version-RFB-320.onnx""#,
        );
        let directory = tempfile::tempdir().expect("temporary directory");
        let manifest = directory.path().join("manifest.toml");
        std::fs::write(&manifest, text).expect("manifest should be written");
        let error =
            verify_research_pipeline_artifacts(&manifest).expect_err("missing model should fail");
        assert!(matches!(
            error,
            ModelCatalogError::ArtifactMissing { ref model_id, .. }
                if model_id == "ultraface-rfb-320"
        ));
    }
}
