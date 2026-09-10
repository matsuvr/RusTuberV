//! VRM license review extracted before an avatar import.
//!
//! The displayed fields mirror the metadata the VRM consortium sample viewer
//! exposes for VRM 0.x and VRM 1.0 (`VRM10Viewer/UIToolkit/MetaView.uxml`
//! and `VRM_Samples/SimpleViewer/ViewerUI.cs`). Extraction is a pure
//! function over the selected file's bytes so that the import path can fail
//! closed before any asset is copied.

use std::path::{Path, PathBuf};

use serde_json::Value;
use thiserror::Error;

use crate::import::VrmGeneration;

/// Errors that prevent a license review from being built.
#[derive(Debug, Error)]
pub enum VrmLicenseReviewError {
    /// The file could not be read.
    #[error("license review I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The file exceeds the import size limit.
    #[error("license review: size {size} exceeds limit {limit}")]
    SizeExceeded {
        /// Actual file size.
        size: u64,
        /// Configured size limit.
        limit: u64,
    },
    /// The file is not a parseable GLB.
    #[error("license review: failed to parse GLB: {0}")]
    GlbParse(String),
    /// No VRM generation extension was found.
    #[error("license review: missing VRM or VRMC_vrm extension")]
    NotVrm,
    /// Both generation extensions were present.
    #[error("license review: both VRM and VRMC_vrm extensions are present")]
    AmbiguousVrmVersion,
    /// VRM 1.0 declared an unsupported spec version.
    #[error("license review: unsupported VRM spec version {0}")]
    UnsupportedVersion(String),
    /// The metadata carries no model name to review.
    #[error("license review: model metadata has no name")]
    MissingName,
}

/// Reviewable license and usage metadata for a selected VRM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VrmLicenseReview {
    /// Detected VRM generation.
    pub generation: VrmGeneration,
    /// Model name.
    pub model_name: String,
    /// Model version, if declared.
    pub version: Option<String>,
    /// Author names, in source order.
    pub authors: Vec<String>,
    /// Contact information, if declared.
    pub contact_information: Option<String>,
    /// Reference texts or URLs, in source order.
    pub references: Vec<String>,
    /// Avatar usage permission token (`onlyAuthor`, `Everyone`, ...).
    pub avatar_permission: Option<String>,
    /// Excessive violent expression permission.
    pub allow_violent_usage: Option<String>,
    /// Excessive sexual expression permission.
    pub allow_sexual_usage: Option<String>,
    /// Commercial usage token.
    pub commercial_usage: Option<String>,
    /// Other permission URL.
    pub other_permission_url: Option<String>,
    /// Modification/distribution license token.
    pub modification_license: Option<String>,
    /// Named license (VRM 0.x `licenseName`), if declared.
    pub license_name: Option<String>,
    /// Standard license URL.
    pub license_url: Option<String>,
    /// Additional license URL.
    pub other_license_url: Option<String>,
    /// The file this review was extracted from.
    pub source_path: PathBuf,
}

/// Extracts the license review for a VRM file without importing it.
///
/// # Errors
///
/// Returns an error when the bytes are not a GLB, do not declare exactly one
/// VRM generation, declare an unsupported VRM 1.0 version, or carry no model
/// name. The caller must not import the model when review extraction fails.
pub fn extract_vrm_license_review(
    source_path: &Path,
    bytes: &[u8],
) -> Result<VrmLicenseReview, VrmLicenseReviewError> {
    // Only the JSON chunk is needed; `Gltf::from_slice` skips buffer and image
    // decoding so the review does not pay the full import cost.
    let gltf = gltf::Gltf::from_slice(bytes)
        .map_err(|error| VrmLicenseReviewError::GlbParse(error.to_string()))?;
    let json = gltf.document.as_json();
    let extensions = json
        .extensions
        .as_ref()
        .map(|extensions| &extensions.others);
    let legacy = extensions.and_then(|extensions| extensions.get("VRM"));
    let modern = extensions.and_then(|extensions| extensions.get("VRMC_vrm"));

    match (legacy, modern) {
        (Some(_), Some(_)) => Err(VrmLicenseReviewError::AmbiguousVrmVersion),
        (Some(vrm), None) => review_vrm0(source_path, vrm),
        (None, Some(vrmc)) => review_vrm1(source_path, vrmc),
        (None, None) => Err(VrmLicenseReviewError::NotVrm),
    }
}

fn review_vrm0(source_path: &Path, vrm: &Value) -> Result<VrmLicenseReview, VrmLicenseReviewError> {
    let meta = vrm.get("meta").and_then(Value::as_object);
    let string = |field: &str| {
        meta.and_then(|meta| meta.get(field))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(String::from)
    };
    let model_name = string("title")
        .or_else(|| string("name"))
        .ok_or(VrmLicenseReviewError::MissingName)?;
    Ok(VrmLicenseReview {
        generation: VrmGeneration::Vrm0,
        model_name,
        version: string("version"),
        authors: string("author").into_iter().collect(),
        contact_information: string("contactInformation"),
        references: string("reference").into_iter().collect(),
        avatar_permission: string("allowedUserName"),
        // VRM 0.x spells these fields with `Ussage`; the sample exporter and
        // the official schema both use that spelling.
        allow_violent_usage: string("violentUssageName"),
        allow_sexual_usage: string("sexualUssageName"),
        commercial_usage: string("commercialUssageName"),
        other_permission_url: string("otherPermissionUrl"),
        modification_license: None,
        license_name: string("licenseName"),
        license_url: string("licenseUrl"),
        other_license_url: string("otherLicenseUrl"),
        source_path: source_path.to_path_buf(),
    })
}

fn review_vrm1(
    source_path: &Path,
    vrmc: &Value,
) -> Result<VrmLicenseReview, VrmLicenseReviewError> {
    let spec_version = vrmc
        .get("specVersion")
        .and_then(Value::as_str)
        .ok_or_else(|| VrmLicenseReviewError::GlbParse("missing specVersion".into()))?;
    if spec_version != "1.0" {
        return Err(VrmLicenseReviewError::UnsupportedVersion(
            spec_version.to_string(),
        ));
    }
    let meta = vrmc.get("meta").and_then(Value::as_object);
    let string = |field: &str| {
        meta.and_then(|meta| meta.get(field))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(String::from)
    };
    let strings = |field: &str| {
        meta.and_then(|meta| meta.get(field))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    };
    let boolean = |field: &str| {
        meta.and_then(|meta| meta.get(field))
            .and_then(Value::as_bool)
            .map(|value| if value { "true" } else { "false" }.to_string())
    };
    let model_name = string("name").ok_or(VrmLicenseReviewError::MissingName)?;
    Ok(VrmLicenseReview {
        generation: VrmGeneration::Vrm1,
        model_name,
        version: string("version"),
        authors: strings("authors"),
        contact_information: string("contactInformation"),
        references: strings("references"),
        avatar_permission: string("avatarPermission"),
        allow_violent_usage: boolean("allowExcessivelyViolentUsage"),
        allow_sexual_usage: boolean("allowExcessivelySexualUsage"),
        commercial_usage: string("commercialUsage"),
        other_permission_url: None,
        modification_license: string("modification"),
        license_name: None,
        license_url: string("licenseUrl"),
        other_license_url: string("otherLicenseUrl"),
        source_path: source_path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const VRM0_REVIEW_JSON: &str = r#"{
        "asset": {"version": "2.0"},
        "extensions": {
            "VRM": {
                "meta": {
                    "title": "Legacy Model",
                    "version": "1.00",
                    "author": "Legacy Author",
                    "contactInformation": "https://example.test/contact",
                    "reference": "https://example.test/reference",
                    "allowedUserName": "Everyone",
                    "violentUssageName": "Allow",
                    "sexualUssageName": "Disallow",
                    "commercialUssageName": "Allow",
                    "otherPermissionUrl": "https://example.test/permission",
                    "licenseName": "Other",
                    "licenseUrl": "https://example.test/license",
                    "otherLicenseUrl": "https://example.test/other-license"
                }
            }
        }
    }"#;

    const VRM1_REVIEW_JSON: &str = r#"{
        "asset": {"version": "2.0"},
        "extensions": {
            "VRMC_vrm": {
                "specVersion": "1.0",
                "meta": {
                    "name": "Modern Model",
                    "version": "2.0",
                    "authors": ["Author A", "Author B"],
                    "contactInformation": "https://example.test/contact",
                    "references": ["https://example.test/ref-a", "https://example.test/ref-b"],
                    "avatarPermission": "onlyAuthor",
                    "allowExcessivelyViolentUsage": true,
                    "allowExcessivelySexualUsage": false,
                    "commercialUsage": "personalNonProfit",
                    "modification": "prohibited",
                    "licenseUrl": "https://vrm.dev/licenses/1.0/",
                    "otherLicenseUrl": "https://example.test/other-license"
                }
            }
        }
    }"#;

    fn glb_bytes(json: &str) -> Vec<u8> {
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
        bytes
    }

    fn extract(json: &str) -> Result<VrmLicenseReview, VrmLicenseReviewError> {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("review.vrm");
        extract_vrm_license_review(&path, &glb_bytes(json))
    }

    #[test]
    fn extracts_vrm0_license_review() {
        let review = extract(VRM0_REVIEW_JSON).expect("VRM 0.x review");
        assert_eq!(review.generation, VrmGeneration::Vrm0);
        assert_eq!(review.model_name, "Legacy Model");
        assert_eq!(review.version.as_deref(), Some("1.00"));
        assert_eq!(review.authors, vec!["Legacy Author"]);
        assert_eq!(
            review.contact_information.as_deref(),
            Some("https://example.test/contact")
        );
        assert_eq!(review.references, vec!["https://example.test/reference"]);
        assert_eq!(review.avatar_permission.as_deref(), Some("Everyone"));
        assert_eq!(review.allow_violent_usage.as_deref(), Some("Allow"));
        assert_eq!(review.allow_sexual_usage.as_deref(), Some("Disallow"));
        assert_eq!(review.commercial_usage.as_deref(), Some("Allow"));
        assert_eq!(
            review.other_permission_url.as_deref(),
            Some("https://example.test/permission")
        );
        assert_eq!(review.license_name.as_deref(), Some("Other"));
        assert_eq!(
            review.other_license_url.as_deref(),
            Some("https://example.test/other-license")
        );
        assert!(review.modification_license.is_none());
    }

    #[test]
    fn extracts_vrm1_license_review() {
        let review = extract(VRM1_REVIEW_JSON).expect("VRM 1.0 review");
        assert_eq!(review.generation, VrmGeneration::Vrm1);
        assert_eq!(review.model_name, "Modern Model");
        assert_eq!(review.version.as_deref(), Some("2.0"));
        assert_eq!(review.authors, vec!["Author A", "Author B"]);
        assert_eq!(review.references.len(), 2);
        assert_eq!(review.avatar_permission.as_deref(), Some("onlyAuthor"));
        assert_eq!(review.allow_violent_usage.as_deref(), Some("true"));
        assert_eq!(review.allow_sexual_usage.as_deref(), Some("false"));
        assert_eq!(
            review.commercial_usage.as_deref(),
            Some("personalNonProfit")
        );
        assert_eq!(review.modification_license.as_deref(), Some("prohibited"));
        assert!(review.license_name.is_none());
        assert!(review.other_permission_url.is_none());
    }

    #[test]
    fn keeps_the_source_path_in_the_review() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("source path.vrm");
        let review =
            extract_vrm_license_review(&path, &glb_bytes(VRM1_REVIEW_JSON)).expect("review");
        assert_eq!(review.source_path, path);
    }

    #[test]
    fn rejects_metadata_without_a_model_name() {
        let missing = VRM1_REVIEW_JSON.replace("\"name\": \"Modern Model\",", "");
        assert!(matches!(
            extract(&missing),
            Err(VrmLicenseReviewError::MissingName)
        ));
    }

    #[test]
    fn rejects_model_without_a_vrm_extension() {
        let json = r#"{"asset": {"version": "2.0"}}"#;
        assert!(matches!(extract(json), Err(VrmLicenseReviewError::NotVrm)));
    }

    #[test]
    fn rejects_model_with_both_generation_extensions() {
        let json = VRM1_REVIEW_JSON.replace("\"VRMC_vrm\"", "\"VRM\": {}, \"VRMC_vrm\"");
        assert!(matches!(
            extract(&json),
            Err(VrmLicenseReviewError::AmbiguousVrmVersion)
        ));
    }

    #[test]
    fn rejects_unsupported_vrm1_spec_version() {
        let json = VRM1_REVIEW_JSON.replace("\"specVersion\": \"1.0\"", "\"specVersion\": \"2.0\"");
        assert!(matches!(
            extract(&json),
            Err(VrmLicenseReviewError::UnsupportedVersion(version)) if version == "2.0"
        ));
    }

    #[test]
    fn rejects_non_glb_bytes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("broken.vrm");
        assert!(matches!(
            extract_vrm_license_review(&path, b"not a glb"),
            Err(VrmLicenseReviewError::GlbParse(_))
        ));
    }

    #[test]
    fn drops_blank_meta_values() {
        let json = VRM1_REVIEW_JSON
            .replace("\"version\": \"2.0\",", "\"version\": \"  \",")
            .replace(
                "\"authors\": [\"Author A\", \"Author B\"],",
                "\"authors\": [],",
            );
        let review = extract(&json).expect("review");
        assert!(review.version.is_none());
        assert!(review.authors.is_empty());
    }
}
