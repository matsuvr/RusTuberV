//! Single import boundary: VRM 0.x conversion, then shared VRM 1.0 preparation.

use crate::glb::{Glb, GlbError};
use serde_json::Value;
use std::fmt;

/// Original file format, retained only for import metadata and license review.
#[derive(Clone, Copy, Debug, Default, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VrmGeneration {
    /// VRM 0.x source, converted before runtime inspection and loading.
    Vrm0,
    /// Native VRM 1.0 source.
    #[default]
    Vrm1,
}

/// Source-only facts retained before conversion removes the legacy extension.
#[derive(Clone, Debug, Default)]
pub struct VrmSourceInfo {
    /// Original file format; never used to choose runtime behavior.
    pub generation: VrmGeneration,
    /// Legacy exporter identifier, if present.
    pub exporter_version: Option<String>,
    /// Non-fatal conversion diagnostics for import UI.
    pub compatibility_warnings: Vec<crate::VrmCompatibilityWarning>,
}

/// Errors returned by the pure core descriptor parser.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VrmParseError {
    /// No supported root extension was found.
    MissingGeneration,
    /// Both generation roots were supplied.
    AmbiguousGeneration,
    /// VRM 1.0 declared an unsupported version.
    UnsupportedVersion(String),
    /// A required field was not present.
    MissingField(String),
    /// A field had an invalid JSON type or value.
    InvalidField {
        /// JSON field path.
        path: String,
        /// Stable validation reason.
        reason: String,
    },
    /// A glTF index was outside the referenced array.
    InvalidIndex {
        /// JSON field path.
        path: String,
        /// Index that was out of range.
        index: usize,
    },
    /// A morph index exceeds its mesh's target array.
    InvalidMorphTargetIndex {
        /// Referenced glTF mesh.
        mesh: usize,
        /// Out-of-range morph index.
        index: usize,
    },
    /// A legacy human bone was declared more than once.
    DuplicateBone(String),
}

impl fmt::Display for VrmParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingGeneration => write!(f, "missing VRM or VRMC_vrm extension"),
            Self::AmbiguousGeneration => {
                write!(f, "both VRM and VRMC_vrm extensions are present")
            }
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported VRMC_vrm specVersion {version}")
            }
            Self::MissingField(path) => write!(f, "missing required field {path}"),
            Self::InvalidField { path, reason } => write!(f, "invalid field {path}: {reason}"),
            Self::InvalidIndex { path, index } => {
                write!(f, "invalid index {path}={index}")
            }
            Self::InvalidMorphTargetIndex { mesh, index } => {
                write!(f, "invalid morph target index {index} for mesh {mesh}")
            }
            Self::DuplicateBone(name) => write!(f, "duplicate Humanoid bone {name}"),
        }
    }
}

impl std::error::Error for VrmParseError {}

/// Errors that can occur while preparing a model for the VRM 1.0 runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VrmPrepareError {
    /// GLB container or JSON failure.
    Glb(GlbError),
    /// VRM descriptor validation failure.
    Descriptor(VrmParseError),
    /// A VRM field cannot be normalized into the VRM 1.0 shape.
    InvalidField {
        /// JSON field path.
        path: String,
        /// Stable validation reason.
        reason: String,
    },
}

impl std::fmt::Display for VrmPrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Glb(error) => error.fmt(f),
            Self::Descriptor(error) => error.fmt(f),
            Self::InvalidField { path, reason } => {
                write!(f, "invalid VRM field {path}: {reason}")
            }
        }
    }
}

impl std::error::Error for VrmPrepareError {}

impl From<GlbError> for VrmPrepareError {
    fn from(error: GlbError) -> Self {
        Self::Glb(error)
    }
}

/// Converts legacy JSON first, then applies the same VRM 1.0 runtime adaptation.
/// Returns source metadata and whether the managed document changed.
pub fn prepare_vrm_document(
    document: &mut Value,
) -> Result<(VrmSourceInfo, bool), VrmPrepareError> {
    let extensions = document.get("extensions").and_then(Value::as_object);
    let legacy = extensions.is_some_and(|extensions| extensions.contains_key("VRM"));
    let modern = extensions.is_some_and(|extensions| extensions.contains_key("VRMC_vrm"));
    let (source, mut changed) = match (legacy, modern) {
        (true, true) => {
            return Err(VrmPrepareError::Descriptor(
                VrmParseError::AmbiguousGeneration,
            ));
        }
        (false, false) => {
            return Err(VrmPrepareError::Descriptor(
                VrmParseError::MissingGeneration,
            ));
        }
        (true, false) => (crate::vrm0::convert::convert_vrm0_document(document)?, true),
        (false, true) => {
            let version = document
                .pointer("/extensions/VRMC_vrm/specVersion")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    VrmPrepareError::Descriptor(VrmParseError::MissingField(
                        "VRMC_vrm.specVersion".into(),
                    ))
                })?;
            if version != "1.0" {
                return Err(VrmPrepareError::Descriptor(
                    VrmParseError::UnsupportedVersion(version.into()),
                ));
            }
            (
                VrmSourceInfo::default(),
                crate::vrm0::convert::repair_converted_thumb_names(document),
            )
        }
    };
    changed |= crate::vrm1::adapt_vrm1_expressions(document);
    Ok((source, changed))
}

/// Prepares a managed copy for the VRM 1.0 runtime without modifying its source.
/// None means the supported VRM 1.0 document already satisfies the contract.
/// Missing, ambiguous, and unsupported formats are errors on every load path.
pub fn prepare_managed_vrm_bytes(bytes: &[u8]) -> Result<Option<Vec<u8>>, VrmPrepareError> {
    let mut glb = Glb::parse(bytes)?;
    let (_, changed) = prepare_vrm_document(&mut glb.document)?;
    if changed {
        Ok(Some(glb.to_vec()?))
    } else {
        Ok(None)
    }
}
