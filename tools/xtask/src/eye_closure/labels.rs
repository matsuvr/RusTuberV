//! Extracted-frame loading, label CSV handling, and split validation (#52).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use vtuber_tracking::EyeSide;

/// One frame row as written by the #51/#65 `extract` command.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct FrameRow {
    pub take_id: String,
    pub frame_seq: u64,
    pub timestamp_micros: u64,
    #[serde(default)]
    pub openness_left: Option<f32>,
    #[serde(default)]
    pub openness_right: Option<f32>,
    #[serde(default)]
    pub mp_blink_left: Option<f32>,
    #[serde(default)]
    pub mp_blink_right: Option<f32>,
    #[serde(default)]
    pub lid_gap_left: Option<f32>,
    #[serde(default)]
    pub lid_gap_right: Option<f32>,
    #[serde(default)]
    pub lid_points_left: Option<[[f32; 2]; 8]>,
    #[serde(default)]
    pub lid_points_right: Option<[[f32; 2]; 8]>,
    #[serde(default)]
    pub inference_width: Option<u32>,
    #[serde(default)]
    pub inference_height: Option<u32>,
    #[serde(default)]
    pub arkit_blink_left: Option<f32>,
    #[serde(default)]
    pub arkit_blink_right: Option<f32>,
    #[serde(default)]
    pub seq_gap_before: bool,
    #[serde(default)]
    pub time_gap_before: bool,
    #[serde(default)]
    pub rgb_reference: Option<String>,
    #[serde(default)]
    pub rgb_width_px: Option<u32>,
    #[serde(default)]
    pub rgb_height_px: Option<u32>,
    #[serde(default)]
    pub rgb_pixel_format: Option<String>,
}

/// Per-take provenance needed to decode review images.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct TakeReportFile {
    pub take_id: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub raw_frames_root: Option<String>,
    pub pixel_rotation_degrees: i32,
    pub mirrored: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ExtractionMetadataFile {
    pub schema_version: u32,
    pub feature: String,
    #[serde(default)]
    pub inputs: Vec<TakeReportFile>,
}

/// Loaded extraction: frame rows plus per-take provenance.
#[derive(Clone, Debug)]
pub(crate) struct ExtractedData {
    pub takes: BTreeMap<String, TakeReportFile>,
    pub frames: Vec<FrameRow>,
}

impl ExtractedData {
    /// Loads `eye-frames.jsonl` and `extraction-metadata.json`.
    ///
    /// # Errors
    ///
    /// Reports malformed files with a concrete `file:line` and rejects an
    /// unsupported extraction schema or feature.
    pub fn load(directory: &Path) -> Result<Self, String> {
        let metadata_path = directory.join("extraction-metadata.json");
        let metadata: ExtractionMetadataFile = read_json(&metadata_path)?;
        if metadata.schema_version != 2 {
            return Err(format!(
                "{}: unsupported extraction schema version {}",
                metadata_path.display(),
                metadata.schema_version
            ));
        }
        if metadata.feature != vtuber_tracking::EYE_CLOSURE_FEATURE {
            return Err(format!(
                "{}: unexpected feature {:?}",
                metadata_path.display(),
                metadata.feature
            ));
        }
        let takes = metadata
            .inputs
            .into_iter()
            .map(|take| (take.take_id.clone(), take))
            .collect();
        let frames_path = directory.join("eye-frames.jsonl");
        let text = std::fs::read_to_string(&frames_path)
            .map_err(|error| format!("failed to read {}: {error}", frames_path.display()))?;
        let mut frames = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let frame: FrameRow = serde_json::from_str(line)
                .map_err(|error| format!("{}:{}: {error}", frames_path.display(), index + 1))?;
            frames.push(frame);
        }
        Ok(Self { takes, frames })
    }

    pub(crate) fn take(&self, take_id: &str) -> Option<&TakeReportFile> {
        self.takes.get(take_id)
    }
}

/// Ground-truth label for one eye at one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LabelValue {
    FullyClosed,
    NotClosed,
    Uncertain,
    Unobservable,
}

/// Where a label came from. Proxy labels never verify physical closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LabelSource {
    VisualReview,
    ArkitProxy,
}

/// One validated label row.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LabelRow {
    pub take_id: String,
    pub frame_seq: u64,
    pub eye: EyeSide,
    pub label: LabelValue,
    pub source: LabelSource,
    pub tag: Option<String>,
    pub event_id: String,
}

/// Labels keyed by identity, with the file hash that produced them.
#[derive(Clone, Debug, Default)]
pub(crate) struct Labels {
    rows: BTreeMap<(String, u64, EyeSide), LabelRow>,
    pub sha256: String,
}

impl Labels {
    /// Loads and validates a label CSV.
    ///
    /// # Errors
    ///
    /// Reports the offending line for unknown enum values, missing columns,
    /// and duplicate `(take_id, frame_seq, eye)` keys (conflicting or not).
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let sha256 = crate::eye_closure::hash_bytes(&bytes);
        let text = String::from_utf8(bytes)
            .map_err(|error| format!("{}: label CSV is not UTF-8: {error}", path.display()))?;
        let mut rows = BTreeMap::new();
        for (index, fields) in parse_csv(&text).into_iter().enumerate() {
            let line_no = index + 1;
            if line_no == 1 {
                continue;
            }
            if fields.iter().all(|field| field.trim().is_empty()) {
                continue;
            }
            let row = parse_label_row(&fields, path, line_no)?;
            let key = (row.take_id.clone(), row.frame_seq, row.eye);
            if rows.insert(key, row).is_some() {
                return Err(format!(
                    "{}:{line_no}: duplicate label for the same take/frame/eye",
                    path.display()
                ));
            }
        }
        Ok(Self { rows, sha256 })
    }

    pub(crate) fn get(&self, take_id: &str, frame_seq: u64, eye: EyeSide) -> Option<&LabelRow> {
        self.rows.get(&(take_id.to_owned(), frame_seq, eye))
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &LabelRow> {
        self.rows.values()
    }
}

fn parse_label_row(fields: &[String], path: &Path, line_no: usize) -> Result<LabelRow, String> {
    let column = |name: &str| -> Result<&str, String> {
        let header = [
            "take_id",
            "frame_seq",
            "eye",
            "label",
            "label_source",
            "tag",
            "event_id",
        ];
        let index = header
            .iter()
            .position(|candidate| *candidate == name)
            .ok_or_else(|| format!("{}:{line_no}: missing {name} column", path.display()))?;
        fields.get(index).map(String::as_str).ok_or_else(|| {
            format!(
                "{}:{line_no}: row is missing the {name} column",
                path.display()
            )
        })
    };
    let take_id = column("take_id")?.trim().to_owned();
    if take_id.is_empty() {
        return Err(format!(
            "{}:{line_no}: take_id must not be empty",
            path.display()
        ));
    }
    let frame_seq = column("frame_seq")?
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("{}:{line_no}: invalid frame_seq: {error}", path.display()))?;
    let eye = match column("eye")?.trim() {
        "left" => EyeSide::Left,
        "right" => EyeSide::Right,
        other => {
            return Err(format!(
                "{}:{line_no}: eye must be left or right, found {other:?}",
                path.display()
            ));
        }
    };
    let label = match column("label")?.trim() {
        "fully_closed" => LabelValue::FullyClosed,
        "not_closed" => LabelValue::NotClosed,
        "uncertain" => LabelValue::Uncertain,
        "unobservable" => LabelValue::Unobservable,
        other => {
            return Err(format!(
                "{}:{line_no}: unknown label {other:?}",
                path.display()
            ));
        }
    };
    let source = match column("label_source")?.trim() {
        "visual_review" => LabelSource::VisualReview,
        "arkit_proxy" => LabelSource::ArkitProxy,
        other => {
            return Err(format!(
                "{}:{line_no}: unknown label_source {other:?}",
                path.display()
            ));
        }
    };
    let tag = match column("tag")?.trim() {
        "" => None,
        other => Some(other.to_owned()),
    };
    let event_id = column("event_id")?.trim().to_owned();
    // An empty event_id means "this frame is not part of a reviewed closure
    // event"; it is allowed on any label.
    Ok(LabelRow {
        take_id,
        frame_seq,
        eye,
        label,
        source,
        tag,
        event_id,
    })
}

/// Train/validation/test split over take IDs.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct SplitFile {
    #[serde(default)]
    pub train: Vec<String>,
    #[serde(default)]
    pub validation: Vec<String>,
    #[serde(default)]
    pub test: Vec<String>,
}

impl SplitFile {
    /// Loads and validates split membership against the loaded takes.
    ///
    /// # Errors
    ///
    /// Rejects unknown take IDs, takes appearing in more than one split, and
    /// takes of one capture session that are spread across splits.
    pub fn load(path: &Path, data: &ExtractedData) -> Result<Self, String> {
        let split: Self = read_json(path)?;
        let mut seen: BTreeMap<&str, &'static str> = BTreeMap::new();
        for (name, takes) in [
            ("train", &split.train),
            ("validation", &split.validation),
            ("test", &split.test),
        ] {
            for take in takes {
                if !data.takes.contains_key(take) {
                    return Err(format!(
                        "{}: split {name} references unknown take {take}",
                        path.display()
                    ));
                }
                if let Some(previous) = seen.insert(take.as_str(), name) {
                    return Err(format!(
                        "{}: take {take} appears in both {previous} and {name}",
                        path.display()
                    ));
                }
            }
        }
        // Session grouping: takes that share a session must share a split.
        let mut session_split: BTreeMap<&str, &'static str> = BTreeMap::new();
        for (name, takes) in [
            ("train", &split.train),
            ("validation", &split.validation),
            ("test", &split.test),
        ] {
            for take in takes {
                let Some(session) = data
                    .takes
                    .get(take)
                    .and_then(|report| report.session_id.as_deref())
                else {
                    continue;
                };
                if let Some(previous) = session_split.insert(session, name)
                    && previous != name
                {
                    return Err(format!(
                        "{}: session {session} is split across {previous} and {name}",
                        path.display()
                    ));
                }
            }
        }
        Ok(split)
    }

    pub(crate) fn takes_for(&self, split: Split) -> &[String] {
        match split {
            Split::Train => &self.train,
            Split::Validation => &self.validation,
            Split::Test => &self.test,
        }
    }
}

/// Which split is being read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Split {
    Train,
    Validation,
    Test,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

/// Minimal RFC-4180-ish CSV parser with quoted-field support.
pub(crate) fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        if quoted {
            if character == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    let _ = chars.next();
                } else {
                    quoted = false;
                }
            } else {
                field.push(character);
            }
        } else {
            match character {
                '"' => quoted = true,
                ',' => fields.push(std::mem::take(&mut field)),
                '\r' => {}
                '\n' => {
                    fields.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut fields));
                }
                other => field.push(other),
            }
        }
    }
    if !field.is_empty() || !fields.is_empty() {
        fields.push(field);
        rows.push(fields);
    }
    rows
}

/// Quotes one CSV field when it contains a delimiter, quote, or newline.
pub(crate) fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

/// Returns every (take_id, frame_seq) present, for cross-checking labels.
#[allow(dead_code)]
pub(crate) fn frame_identities(frames: &[FrameRow]) -> BTreeSet<(String, u64)> {
    frames
        .iter()
        .map(|frame| (frame.take_id.clone(), frame.frame_seq))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_parser_handles_quotes_and_commas() {
        let rows = parse_csv("a,b\n\"x,y\",\"he said \"\"hi\"\"\"\n");
        assert_eq!(rows[1][0], "x,y");
        assert_eq!(rows[1][1], "he said \"hi\"");
    }

    #[test]
    fn duplicate_labels_are_rejected() -> Result<(), String> {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("labels.csv");
        std::fs::write(
            &path,
            "take_id,frame_seq,eye,label,label_source,tag,event_id\nt,1,left,fully_closed,visual_review,,e1\nt,1,left,not_closed,visual_review,,e2\n",
        )
        .unwrap();
        assert!(Labels::load(&path).is_err());
        Ok(())
    }

    #[test]
    fn conflicting_unknown_and_missing_values_are_reported_with_line() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("labels.csv");
        std::fs::write(
            &path,
            "take_id,frame_seq,eye,label,label_source,tag,event_id\nt,1,middle,fully_closed,visual_review,,e1\n",
        )
        .unwrap();
        let error = Labels::load(&path).unwrap_err();
        assert!(error.contains(":2:"), "{error}");
    }
}
