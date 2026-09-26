//! Versioned explicit inputs for the eye-closure analysis (Issue #51).
//!
//! The input list is deliberately tiny and fully explicit: every take, its
//! storage form, and any read-time pixel correction are named by the caller.
//! The tool never scans disks for captures and never guesses a rotation.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Schema version of the `inputs.json` document.
pub const INPUTS_SCHEMA_VERSION: u32 = 1;

/// How one take's data is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputKind {
    /// A completed ARKit teacher capture directory (`COMPLETED`, `frames.jsonl`, ...).
    Raw,
    /// A derived trace whose `replay-metadata.json` is schema version 1.
    TraceV1,
    /// A derived trace whose `replay-metadata.json` is schema version 2.
    TraceV2,
}

/// One explicit local input.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputTake {
    /// Stable identifier used in every output row.
    pub take_id: String,
    /// Storage form of this take.
    pub kind: InputKind,
    /// Directory containing the capture or derived trace.
    pub path: PathBuf,
    /// Session identity, when known independently of the files.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Grouping key shared by a raw and derived view of the same recording.
    ///
    /// Two entries with the same origin describe one recording and must not
    /// be counted twice, so loading rejects the duplicate.
    #[serde(default)]
    pub origin: Option<String>,
    /// Additional pixel rotation applied only at read time.
    ///
    /// The source bytes are never modified. `None` means no correction.
    #[serde(default)]
    pub rotation_degrees: Option<i32>,
    /// Whether the stored pixels are horizontally mirrored.
    #[serde(default)]
    pub mirrored: Option<bool>,
    /// Raw pixel directory used to resolve `rgb_reference` paths for review
    /// images when `kind` is a trace and the original `.bin` files remain.
    #[serde(default)]
    pub raw_frames_root: Option<PathBuf>,
}

/// Root document of `inputs.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputsDocument {
    /// Schema version of this document.
    pub schema_version: u32,
    /// Explicit inputs.
    pub inputs: Vec<InputTake>,
}

impl InputsDocument {
    /// Loads and validates `inputs.json`.
    ///
    /// # Errors
    ///
    /// Returns a message naming the offending file for read/parse failures,
    /// an unsupported schema version, empty input lists, duplicate `take_id`
    /// values, and duplicate `origin` groupings.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let document: Self = serde_json::from_str(&text)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        if document.schema_version != INPUTS_SCHEMA_VERSION {
            return Err(format!(
                "{}: unsupported inputs schema version {} (expected {})",
                path.display(),
                document.schema_version,
                INPUTS_SCHEMA_VERSION
            ));
        }
        if document.inputs.is_empty() {
            return Err(format!("{}: inputs must not be empty", path.display()));
        }
        let mut take_ids = BTreeSet::new();
        let mut origins = BTreeSet::new();
        for input in &document.inputs {
            if input.take_id.trim().is_empty() {
                return Err(format!("{}: take_id must not be empty", path.display()));
            }
            if !take_ids.insert(input.take_id.clone()) {
                return Err(format!(
                    "{}: duplicate take_id {}",
                    path.display(),
                    input.take_id
                ));
            }
            if let Some(origin) = &input.origin
                && !origins.insert(origin.clone())
            {
                return Err(format!(
                    "{}: origin {origin} appears more than once; the raw and derived views of one recording must not be counted twice",
                    path.display()
                ));
            }
        }
        Ok(document)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;

    fn document(inputs: Vec<InputTake>) -> InputsDocument {
        InputsDocument {
            schema_version: INPUTS_SCHEMA_VERSION,
            inputs,
        }
    }

    fn take(take_id: &str) -> InputTake {
        InputTake {
            take_id: take_id.into(),
            kind: InputKind::TraceV2,
            path: PathBuf::from("data/x"),
            session_id: None,
            origin: None,
            rotation_degrees: None,
            mirrored: None,
            raw_frames_root: None,
        }
    }

    #[test]
    fn duplicate_origin_is_rejected_before_double_counting() {
        let mut raw = take("raw");
        raw.kind = InputKind::Raw;
        raw.origin = Some("take_01".into());
        let mut trace = take("trace");
        trace.origin = Some("take_01".into());
        let path = std::env::temp_dir().join("eye_closure_duplicate_origin.json");
        std::fs::write(
            &path,
            serde_json::to_string(&document(vec![raw, trace])).unwrap(),
        )
        .unwrap();
        let error = InputsDocument::load(&path).unwrap_err();
        assert!(error.contains("origin take_01"), "{error}");
    }

    #[test]
    fn unknown_field_is_rejected() {
        let path = std::env::temp_dir().join("eye_closure_unknown_field.json");
        std::fs::write(
            &path,
            br#"{"schema_version":1,"inputs":[{"take_id":"t","kind":"trace_v2","path":"x","surprise":1}]}"#,
        )
        .unwrap();
        assert!(InputsDocument::load(&path).is_err());
    }

    #[test]
    fn unsupported_schema_is_rejected() {
        let path = std::env::temp_dir().join("eye_closure_schema.json");
        std::fs::write(&path, br#"{"schema_version":9,"inputs":[]}"#).unwrap();
        assert!(InputsDocument::load(&path).is_err());
    }
}
