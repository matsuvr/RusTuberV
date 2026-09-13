//! Reading, validation, and normalization of eye-closure analysis inputs.
//!
//! Two storage forms are supported. Raw ARKit captures carry teacher
//! coefficients plus RGB references and are re-inferred with the current
//! MediaPipe runtime. Derived traces already carry paired MediaPipe
//! observations and teacher coefficients; they are reused verbatim without
//! describing them as current-runtime results.
//!
//! All reads are fail-closed: duplicate or regressed sequences, paired
//! timestamp disagreements, unknown schemas, wrong array lengths, and
//! non-finite coefficients are reported with a concrete `file:line`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use vtuber_core::{
    ArkitBlendshape, FaceTrackingOutcome, FrameSeq, MediaPipeBlendshape, MonoTimeNs, PixelFormat,
    VideoFrame,
};
use vtuber_inference::FaceTrackingInference;
use vtuber_inference::backend::mediapipe::{
    MediaPipeRuntime, TASK_BUNDLE_FILE, TASK_BUNDLE_SHA256,
};

use super::Options;
use super::decode::decode_rgb_bin;
use super::input::{InputKind, InputTake, InputsDocument};

/// Feature identity shared with the runtime judgement (Issue #52).
pub(crate) const FEATURE_ID: &str = "mediapipe_raw_eye_blink_openness_v1";
const INVENTORY_SCHEMA_VERSION: u32 = 1;
const EXTRACTION_SCHEMA_VERSION: u32 = 1;
const ARKIT52_COUNT: usize = 52;

/// Parsed MediaPipe blink pair plus optional landmark-presence quality.
type BlinkObservation = (Option<(f32, f32)>, Option<f32>);

// -----------------------------------------------------------------------------
// Inventory
// -----------------------------------------------------------------------------

#[derive(Serialize)]
struct InventoryDocument {
    schema_version: u32,
    feature: &'static str,
    inputs: Vec<InputInventory>,
}

#[derive(Serialize)]
struct InputInventory {
    take_id: String,
    kind: InputKind,
    path: String,
    detected_schema_version: Option<u32>,
    session_id: Option<String>,
    completed: bool,
    frame_count: Option<u64>,
    teacher_count: Option<u64>,
    rgb_count: Option<u64>,
    mediapipe_observation_count: Option<u64>,
    manifest_counts: Option<PairCounts>,
    task_bundle_sha256: Option<String>,
    pixel_rotation_degrees: Option<i32>,
    has_raw_frames: bool,
    notes: Vec<String>,
}

/// Runs `eye-closure inspect`.
pub(crate) fn run_inspect(options: &Options) -> Result<(), String> {
    let inputs_path = options
        .inputs
        .as_deref()
        .ok_or("missing required option --inputs")?;
    let document = InputsDocument::load(inputs_path)?;
    let mut inputs = Vec::new();
    for input in &document.inputs {
        inputs.push(inspect_input(input)?);
    }
    let report = InventoryDocument {
        schema_version: INVENTORY_SCHEMA_VERSION,
        feature: FEATURE_ID,
        inputs,
    };
    write_json(&options.output, &report)?;
    println!("wrote {}", options.output.display());
    Ok(())
}

fn inspect_input(input: &InputTake) -> Result<InputInventory, String> {
    match input.kind {
        InputKind::Raw => inspect_raw(input),
        InputKind::TraceV1 | InputKind::TraceV2 => inspect_trace(input),
    }
}

fn inspect_raw(input: &InputTake) -> Result<InputInventory, String> {
    let dir = &input.path;
    let completed = dir.join("COMPLETED").is_file();
    let mut notes = Vec::new();
    if !completed {
        notes.push("missing COMPLETED marker; this capture is not valid input".into());
    }
    let session: Option<RawSession> = read_json_optional(&dir.join("session.json"))?;
    let manifest: Option<RawManifest> = read_json_optional(&dir.join("manifest.json"))?;
    let teacher_count = count_jsonl(&dir.join("frames.jsonl"))?;
    let rgb_count = count_jsonl(&dir.join("rgb.jsonl"))?;
    let has_raw_frames = dir.join("frames").is_dir();
    Ok(InputInventory {
        take_id: input.take_id.clone(),
        kind: input.kind,
        path: input.path.display().to_string(),
        detected_schema_version: session.as_ref().map(|session| session.schema_version),
        session_id: session.as_ref().map(|session| session.session_id.clone()),
        completed,
        frame_count: teacher_count,
        teacher_count,
        rgb_count,
        mediapipe_observation_count: None,
        manifest_counts: manifest.map(|manifest| manifest.counts.into()),
        task_bundle_sha256: None,
        pixel_rotation_degrees: input.rotation_degrees,
        has_raw_frames,
        notes,
    })
}

fn inspect_trace(input: &InputTake) -> Result<InputInventory, String> {
    let dir = &input.path;
    let metadata: ReplayMetadata = read_json(&dir.join("replay-metadata.json"))?;
    let trace_path = dir.join("derived-trace.jsonl");
    let mut teacher_count = 0_u64;
    let mut mediapipe_count = 0_u64;
    let mut line_count = 0_u64;
    for (index, line) in read_lines(&trace_path)?.into_iter().enumerate() {
        let _ = index;
        line_count += 1;
        let parsed: TraceLine = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{}: {error}", trace_path.display(), index + 1))?;
        if parsed.teacher.is_some() {
            teacher_count += 1;
        }
        if !parsed.mediapipe_observation.is_null() {
            mediapipe_count += 1;
        }
    }
    let mut notes = Vec::new();
    if let Some(rotation) = metadata
        .config
        .as_ref()
        .and_then(|c| c.pixel_rotation_degrees)
    {
        notes.push(format!("recorded replay pixel_rotation_degrees={rotation}"));
    }
    Ok(InputInventory {
        take_id: input.take_id.clone(),
        kind: input.kind,
        path: input.path.display().to_string(),
        detected_schema_version: Some(metadata.schema_version),
        session_id: metadata
            .source_dataset
            .as_ref()
            .and_then(|dataset| dataset.session_id.clone()),
        completed: trace_path.is_file(),
        frame_count: metadata
            .source_dataset
            .as_ref()
            .and_then(|dataset| dataset.frame_count)
            .or(Some(line_count)),
        teacher_count: Some(teacher_count),
        rgb_count: None,
        mediapipe_observation_count: Some(mediapipe_count),
        manifest_counts: None,
        task_bundle_sha256: metadata
            .config
            .as_ref()
            .and_then(|config| config.task_bundle_sha256.clone()),
        pixel_rotation_degrees: input.rotation_degrees.or(metadata
            .config
            .as_ref()
            .and_then(|c| c.pixel_rotation_degrees)),
        has_raw_frames: input.raw_frames_root.is_some(),
        notes,
    })
}

// -----------------------------------------------------------------------------
// Extraction
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
struct PairCounts {
    paired: usize,
    unpaired_teacher: usize,
    unpaired_rgb: usize,
    dropped_sequences: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct TakeCounts {
    frames: usize,
    face_observed: usize,
    teacher_present: usize,
    mediapipe_present: usize,
    both_present: usize,
    mediapipe_missing: usize,
    teacher_missing: usize,
    gaps: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ExtractedFrame {
    take_id: String,
    session_id: Option<String>,
    frame_seq: u64,
    timestamp_micros: u64,
    mp_blink_left: Option<f32>,
    mp_blink_right: Option<f32>,
    openness_left: Option<f32>,
    openness_right: Option<f32>,
    arkit_blink_left: Option<f32>,
    arkit_blink_right: Option<f32>,
    face_observed: bool,
    gap_before: bool,
    mp_landmark_presence_median: Option<f32>,
    rgb_reference: Option<String>,
    rgb_width_px: Option<u32>,
    rgb_height_px: Option<u32>,
    rgb_pixel_format: Option<String>,
    rgb_declared_orientation_degrees: Option<i32>,
    rgb_declared_mirrored: Option<bool>,
}

#[derive(Debug, Serialize)]
struct TakeReport {
    take_id: String,
    kind: InputKind,
    path: String,
    take_root: String,
    raw_frames_root: Option<String>,
    session_id: Option<String>,
    pixel_rotation_degrees: i32,
    mirrored: bool,
    declared_orientation_degrees: Option<i32>,
    declared_mirrored: Option<bool>,
    input_hashes: BTreeMap<String, String>,
    trace_sha256: Option<String>,
    task_bundle_sha256: Option<String>,
    mediapipe_observation_source: &'static str,
    counts: TakeCounts,
    excluded: PairCounts,
    notes: Vec<String>,
}

#[derive(Serialize)]
struct ExtractionMetadata {
    schema_version: u32,
    feature: &'static str,
    xtask_version: &'static str,
    tool_commit: Option<String>,
    task_bundle_sha256_current: &'static str,
    totals: Totals,
    inputs: Vec<TakeReport>,
}

#[derive(Serialize, Default)]
struct Totals {
    takes: usize,
    frames: usize,
    face_observed: usize,
    teacher_present: usize,
    gaps: usize,
}

#[derive(Debug)]
struct TakeExtraction {
    frames: Vec<ExtractedFrame>,
    report: TakeReport,
}

/// Runs `eye-closure extract`.
pub(crate) fn run_extract(options: &Options) -> Result<(), String> {
    let inputs_path = options
        .inputs
        .as_deref()
        .ok_or("missing required option --inputs")?;
    let document = InputsDocument::load(inputs_path)?;
    let task_path = options
        .project_root
        .join("assets")
        .join("models")
        .join(TASK_BUNDLE_FILE);
    let mut frames = Vec::new();
    let mut reports = Vec::new();
    for input in &document.inputs {
        let extraction = match input.kind {
            InputKind::Raw => extract_raw(input, &task_path)?,
            InputKind::TraceV1 | InputKind::TraceV2 => extract_trace(input)?,
        };
        frames.extend(extraction.frames);
        reports.push(extraction.report);
    }
    let totals = Totals {
        takes: reports.len(),
        frames: frames.len(),
        face_observed: frames.iter().filter(|frame| frame.face_observed).count(),
        teacher_present: frames
            .iter()
            .filter(|frame| frame.arkit_blink_left.is_some())
            .count(),
        gaps: frames.iter().filter(|frame| frame.gap_before).count(),
    };
    let metadata = ExtractionMetadata {
        schema_version: EXTRACTION_SCHEMA_VERSION,
        feature: FEATURE_ID,
        xtask_version: env!("CARGO_PKG_VERSION"),
        tool_commit: current_git_commit(),
        task_bundle_sha256_current: TASK_BUNDLE_SHA256,
        totals,
        inputs: reports,
    };
    std::fs::create_dir_all(&options.output)
        .map_err(|error| format!("failed to create {}: {error}", options.output.display()))?;
    write_jsonl(&options.output.join("eye-frames.jsonl"), &frames)?;
    write_csv(&options.output.join("eye-frames.csv"), &frames)?;
    write_json(&options.output.join("extraction-metadata.json"), &metadata)?;
    println!(
        "wrote {} ({} frames from {} takes)",
        options.output.display(),
        frames.len(),
        metadata.totals.takes
    );
    Ok(())
}

// -----------------------------------------------------------------------------
// Derived trace input (B)
// -----------------------------------------------------------------------------

#[derive(Deserialize)]
struct ReplayMetadata {
    schema_version: u32,
    #[serde(default)]
    source_dataset: Option<SourceDataset>,
    #[serde(default)]
    config: Option<MetadataConfig>,
    #[serde(default)]
    trace_sha256: Option<String>,
}
#[derive(Deserialize)]
struct SourceDataset {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    take_id: Option<String>,
    #[serde(default)]
    frame_count: Option<u64>,
    #[serde(default)]
    input_hashes: BTreeMap<String, String>,
}
#[derive(Deserialize)]
struct MetadataConfig {
    #[serde(default)]
    task_bundle_sha256: Option<String>,
    #[serde(default)]
    pixel_rotation_degrees: Option<i32>,
}
#[derive(Deserialize)]
struct TraceLine {
    frame_seq: u64,
    timestamp_micros: u64,
    #[serde(default)]
    mediapipe_observation: Value,
    #[serde(default)]
    teacher: Option<TraceTeacher>,
    #[serde(default)]
    rgb_reference: Option<TraceRgbReference>,
}
#[derive(Deserialize)]
struct TraceTeacher {
    coefficients: Vec<f32>,
}
#[derive(Deserialize)]
struct TraceRgbReference {
    reference_path: String,
    #[serde(default)]
    width_px: Option<u32>,
    #[serde(default)]
    height_px: Option<u32>,
    #[serde(default)]
    pixel_format: Option<String>,
    #[serde(default)]
    orientation_degrees: Option<i32>,
    #[serde(default)]
    mirrored: Option<bool>,
}

fn extract_trace(input: &InputTake) -> Result<TakeExtraction, String> {
    let dir = &input.path;
    let metadata_path = dir.join("replay-metadata.json");
    let metadata: ReplayMetadata = read_json(&metadata_path)?;
    match (input.kind, metadata.schema_version) {
        (InputKind::TraceV1, 1) | (InputKind::TraceV2, 2) => {}
        (kind, version) => {
            return Err(format!(
                "{}: inputs kind {kind:?} does not match replay schema_version {version}",
                metadata_path.display()
            ));
        }
    }
    let trace_path = dir.join("derived-trace.jsonl");
    let trace_bytes = std::fs::read(&trace_path)
        .map_err(|error| format!("failed to read {}: {error}", trace_path.display()))?;
    let trace_sha256 = hex_sha256(&trace_bytes);
    if let Some(expected) = &metadata.trace_sha256
        && !expected.eq_ignore_ascii_case(&trace_sha256)
    {
        return Err(format!(
            "{}: trace_sha256 {} does not match the computed {}",
            metadata_path.display(),
            expected,
            trace_sha256
        ));
    }
    let session_id = metadata
        .source_dataset
        .as_ref()
        .and_then(|dataset| dataset.session_id.clone());
    if let (Some(left), Some(right)) = (&input.session_id, &session_id)
        && left != right
    {
        return Err(format!(
            "{}: input session_id {left} differs from metadata session_id {right}",
            metadata_path.display()
        ));
    }
    let schema_version = metadata.schema_version;
    let mut frames = Vec::new();
    let mut counts = TakeCounts::default();
    let mut previous: Option<(u64, u64)> = None;
    for (index, line) in read_lines(&trace_path)?.into_iter().enumerate() {
        let line_no = index + 1;
        let parsed: TraceLine = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{line_no}: {error}", trace_path.display()))?;
        let gap_before = previous.is_some_and(|(seq, _)| parsed.frame_seq != seq + 1);
        validate_ordering(&trace_path, line_no, &mut previous, &parsed)?;
        let arkit = parse_teacher(&parsed.teacher, &trace_path, line_no)?;
        let (mp, quality) = match schema_version {
            1 => parse_v1_observation(&parsed.mediapipe_observation, &trace_path, line_no)?,
            _ => parse_v2_observation(&parsed.mediapipe_observation, &trace_path, line_no)?,
        };
        let reference = parsed.rgb_reference.as_ref();
        let frame = make_frame(
            &input.take_id,
            session_id.as_deref(),
            parsed.frame_seq,
            parsed.timestamp_micros,
            mp,
            arkit,
            quality,
            reference.map(|reference| reference.reference_path.clone()),
            reference.and_then(|reference| reference.width_px),
            reference.and_then(|reference| reference.height_px),
            reference.and_then(|reference| reference.pixel_format.clone()),
            reference.and_then(|reference| reference.orientation_degrees),
            reference.and_then(|reference| reference.mirrored),
            gap_before,
        );
        accumulate(&mut counts, &frame);
        frames.push(frame);
    }
    if let Some(expected) = metadata
        .source_dataset
        .as_ref()
        .and_then(|dataset| dataset.frame_count)
        && frames.len() as u64 != expected
    {
        return Err(format!(
            "{}: parsed {} frames but source_dataset.frame_count is {expected}",
            trace_path.display(),
            frames.len()
        ));
    }
    let mut notes = vec![
        "trace observations are reused; verify compatibility with the current runtime before treating them as current conditions"
            .to_owned(),
    ];
    if let Some(metadata_take_id) = metadata
        .source_dataset
        .as_ref()
        .and_then(|dataset| dataset.take_id.as_deref())
        && metadata_take_id != input.take_id
    {
        notes.push(format!(
            "metadata take_id {metadata_take_id} differs from the input take_id {}",
            input.take_id
        ));
    }
    let report = TakeReport {
        take_id: input.take_id.clone(),
        kind: input.kind,
        path: input.path.display().to_string(),
        take_root: input.path.display().to_string(),
        raw_frames_root: input
            .raw_frames_root
            .as_ref()
            .map(|root| root.display().to_string()),
        session_id,
        pixel_rotation_degrees: input
            .rotation_degrees
            .or(metadata
                .config
                .as_ref()
                .and_then(|c| c.pixel_rotation_degrees))
            .unwrap_or(0),
        mirrored: input.mirrored.unwrap_or(false),
        declared_orientation_degrees: None,
        declared_mirrored: None,
        input_hashes: metadata
            .source_dataset
            .as_ref()
            .map(|dataset| dataset.input_hashes.clone())
            .unwrap_or_default(),
        trace_sha256: Some(trace_sha256),
        task_bundle_sha256: metadata
            .config
            .as_ref()
            .and_then(|config| config.task_bundle_sha256.clone()),
        mediapipe_observation_source: "derived_trace",
        counts,
        excluded: PairCounts::default(),
        notes,
    };
    Ok(TakeExtraction { frames, report })
}

fn validate_ordering(
    path: &Path,
    line_no: usize,
    previous: &mut Option<(u64, u64)>,
    parsed: &TraceLine,
) -> Result<(), String> {
    if let Some((prev_seq, prev_ts)) = *previous {
        if parsed.frame_seq == prev_seq {
            return Err(format!(
                "{}:{line_no}: duplicate frame_seq {}",
                path.display(),
                parsed.frame_seq
            ));
        }
        if parsed.frame_seq < prev_seq {
            return Err(format!(
                "{}:{line_no}: frame_seq {} regressed below {prev_seq}",
                path.display(),
                parsed.frame_seq
            ));
        }
        if parsed.timestamp_micros <= prev_ts {
            return Err(format!(
                "{}:{line_no}: timestamp {} is not greater than the previous {prev_ts}",
                path.display(),
                parsed.timestamp_micros
            ));
        }
    }
    *previous = Some((parsed.frame_seq, parsed.timestamp_micros));
    Ok(())
}

fn parse_teacher(
    teacher: &Option<TraceTeacher>,
    path: &Path,
    line_no: usize,
) -> Result<Option<(f32, f32)>, String> {
    let Some(teacher) = teacher else {
        return Ok(None);
    };
    if teacher.coefficients.len() != ARKIT52_COUNT {
        return Err(format!(
            "{}:{line_no}: teacher coefficients have {} entries, expected {ARKIT52_COUNT}",
            path.display(),
            teacher.coefficients.len()
        ));
    }
    Ok(Some((
        coefficient_at(
            &teacher.coefficients,
            ArkitBlendshape::EyeBlinkLeft,
            path,
            line_no,
        )?,
        coefficient_at(
            &teacher.coefficients,
            ArkitBlendshape::EyeBlinkRight,
            path,
            line_no,
        )?,
    )))
}

fn parse_v1_observation(
    value: &Value,
    path: &Path,
    line_no: usize,
) -> Result<BlinkObservation, String> {
    match value {
        Value::Null => Ok((None, None)),
        Value::Array(items) => {
            if items.len() != ARKIT52_COUNT {
                return Err(format!(
                    "{}:{line_no}: mediapipe_observation has {} entries, expected {ARKIT52_COUNT}",
                    path.display(),
                    items.len()
                ));
            }
            Ok((
                Some((
                    value_coefficient(items, ArkitBlendshape::EyeBlinkLeft, path, line_no)?,
                    value_coefficient(items, ArkitBlendshape::EyeBlinkRight, path, line_no)?,
                )),
                None,
            ))
        }
        other => Err(format!(
            "{}:{line_no}: schema v1 mediapipe_observation must be an array or null, found {other}",
            path.display()
        )),
    }
}

fn parse_v2_observation(
    value: &Value,
    path: &Path,
    line_no: usize,
) -> Result<BlinkObservation, String> {
    match value {
        Value::Null => Ok((None, None)),
        Value::Object(map) => {
            let direct = map
                .get("direct_coefficients")
                .ok_or_else(|| {
                    format!(
                        "{}:{line_no}: schema v2 mediapipe_observation has no direct_coefficients",
                        path.display()
                    )
                })?
                .as_array()
                .ok_or_else(|| {
                    format!(
                        "{}:{line_no}: schema v2 direct_coefficients is not an array",
                        path.display()
                    )
                })?;
            if direct.len() != ARKIT52_COUNT {
                return Err(format!(
                    "{}:{line_no}: direct_coefficients has {} entries, expected {ARKIT52_COUNT}",
                    path.display(),
                    direct.len()
                ));
            }
            let quality = map
                .get("landmark_presence_median")
                .and_then(Value::as_f64)
                .map(|value| value as f32);
            Ok((
                Some((
                    value_coefficient(direct, ArkitBlendshape::EyeBlinkLeft, path, line_no)?,
                    value_coefficient(direct, ArkitBlendshape::EyeBlinkRight, path, line_no)?,
                )),
                quality,
            ))
        }
        other => Err(format!(
            "{}:{line_no}: schema v2 mediapipe_observation must be an object or null, found {other}",
            path.display()
        )),
    }
}

// -----------------------------------------------------------------------------
// Raw capture input (A)
// -----------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawSession {
    schema_version: u32,
    session_id: String,
    timestamp_domain: String,
}
#[derive(Deserialize)]
struct RawManifest {
    schema_version: u32,
    counts: RawCounts,
}
#[derive(Deserialize)]
struct RawCounts {
    paired: usize,
    unpaired_teacher: usize,
    unpaired_rgb: usize,
    dropped_sequences: u64,
}
impl From<RawCounts> for PairCounts {
    fn from(counts: RawCounts) -> Self {
        Self {
            paired: counts.paired,
            unpaired_teacher: counts.unpaired_teacher,
            unpaired_rgb: counts.unpaired_rgb,
            dropped_sequences: counts.dropped_sequences,
        }
    }
}
#[derive(Deserialize)]
struct RawFrameRecord {
    frame_seq: u64,
    timestamp_micros: u64,
    kind: String,
    payload: RawTeacherPayload,
}
#[derive(Deserialize)]
struct RawTeacherPayload {
    coefficients_canonical: Vec<f32>,
}
#[derive(Deserialize)]
struct RawCapture {
    #[serde(default)]
    stored_orientation_degrees: Option<i32>,
    #[serde(default)]
    stored_mirrored: Option<bool>,
}
#[derive(Clone, Debug, Deserialize)]
struct RawRgbRecord {
    frame_seq: u64,
    timestamp_micros: u64,
    reference_path: String,
    width_px: u32,
    height_px: u32,
    pixel_format: String,
    orientation_degrees: i32,
    mirrored: bool,
}

struct RawTeacherRow {
    record: RawFrameRecord,
    arkit: (f32, f32),
}
struct RawRgbRow {
    record: RawRgbRecord,
}
#[derive(Debug)]
struct RawPair {
    frame_seq: u64,
    timestamp_micros: u64,
    arkit: (f32, f32),
    rgb: RawRgbRecord,
}
#[derive(Debug)]
struct RawTake {
    pairs: Vec<RawPair>,
    counts: PairCounts,
    session_id: String,
    declared_orientation_degrees: Option<i32>,
    declared_mirrored: Option<bool>,
    input_hashes: BTreeMap<String, String>,
}

fn read_raw_pairs(input: &InputTake) -> Result<RawTake, String> {
    let dir = &input.path;
    if !dir.join("COMPLETED").is_file() {
        return Err(format!(
            "{}: missing COMPLETED marker; a partial capture is not valid input",
            dir.display()
        ));
    }
    let session: RawSession = read_json(&dir.join("session.json"))?;
    if session.schema_version != 1 {
        return Err(format!(
            "{}: unsupported capture schema_version {}",
            dir.join("session.json").display(),
            session.schema_version
        ));
    }
    if session.timestamp_domain != "monotonic-micros-since-session-start" {
        return Err(format!(
            "{}: unknown timestamp_domain {:?}",
            dir.join("session.json").display(),
            session.timestamp_domain
        ));
    }
    let manifest: RawManifest = read_json(&dir.join("manifest.json"))?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "{}: unsupported manifest schema_version {}",
            dir.join("manifest.json").display(),
            manifest.schema_version
        ));
    }
    let capture: Option<RawCapture> = read_json_optional(&dir.join("capture.json"))?;
    let frames_path = dir.join("frames.jsonl");
    let rgb_path = dir.join("rgb.jsonl");
    let teacher = parse_raw_frames(&frames_path)?;
    let rgb = parse_raw_rgb(&rgb_path)?;
    let (pairs, counts) = pair_raw_records(&teacher, &rgb, &frames_path, &rgb_path)?;
    let expected: PairCounts = manifest.counts.into();
    if counts != expected {
        return Err(format!(
            "{}: pairing counts {:?} disagree with manifest.json {:?}",
            dir.display(),
            counts,
            expected
        ));
    }
    let mut input_hashes = BTreeMap::new();
    for name in [
        "capture.json",
        "frames.jsonl",
        "manifest.json",
        "rgb.jsonl",
        "session.json",
    ] {
        let path = dir.join(name);
        if path.is_file() {
            let bytes = std::fs::read(&path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            input_hashes.insert(name.to_owned(), hex_sha256(&bytes));
        }
    }
    Ok(RawTake {
        pairs,
        counts,
        session_id: session.session_id,
        declared_orientation_degrees: capture
            .as_ref()
            .and_then(|capture| capture.stored_orientation_degrees),
        declared_mirrored: capture.as_ref().and_then(|capture| capture.stored_mirrored),
        input_hashes,
    })
}

fn parse_raw_frames(path: &Path) -> Result<Vec<RawTeacherRow>, String> {
    let mut rows = Vec::new();
    let mut previous: Option<(u64, u64)> = None;
    for (index, line) in read_lines(path)?.into_iter().enumerate() {
        let line_no = index + 1;
        let record: RawFrameRecord = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{line_no}: {error}", path.display()))?;
        if record.kind != "arkit_teacher" {
            return Err(format!(
                "{}:{line_no}: unexpected record kind {:?}",
                path.display(),
                record.kind
            ));
        }
        if record.payload.coefficients_canonical.len() != ARKIT52_COUNT {
            return Err(format!(
                "{}:{line_no}: coefficients_canonical has {} entries, expected {ARKIT52_COUNT}",
                path.display(),
                record.payload.coefficients_canonical.len()
            ));
        }
        let arkit = (
            coefficient_at(
                &record.payload.coefficients_canonical,
                ArkitBlendshape::EyeBlinkLeft,
                path,
                line_no,
            )?,
            coefficient_at(
                &record.payload.coefficients_canonical,
                ArkitBlendshape::EyeBlinkRight,
                path,
                line_no,
            )?,
        );
        check_identity(
            path,
            line_no,
            &mut previous,
            record.frame_seq,
            record.timestamp_micros,
        )?;
        rows.push(RawTeacherRow { record, arkit });
    }
    Ok(rows)
}

fn parse_raw_rgb(path: &Path) -> Result<Vec<RawRgbRow>, String> {
    let mut rows = Vec::new();
    let mut previous: Option<(u64, u64)> = None;
    for (index, line) in read_lines(path)?.into_iter().enumerate() {
        let line_no = index + 1;
        let record: RawRgbRecord = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{line_no}: {error}", path.display()))?;
        check_identity(
            path,
            line_no,
            &mut previous,
            record.frame_seq,
            record.timestamp_micros,
        )?;
        rows.push(RawRgbRow { record });
    }
    Ok(rows)
}

fn check_identity(
    path: &Path,
    line_no: usize,
    previous: &mut Option<(u64, u64)>,
    frame_seq: u64,
    timestamp_micros: u64,
) -> Result<(), String> {
    if let Some((prev_seq, prev_ts)) = *previous {
        if frame_seq == prev_seq {
            return Err(format!(
                "{}:{line_no}: duplicate frame_seq {frame_seq}",
                path.display()
            ));
        }
        if frame_seq < prev_seq {
            return Err(format!(
                "{}:{line_no}: frame_seq {frame_seq} regressed below {prev_seq}",
                path.display()
            ));
        }
        if timestamp_micros <= prev_ts {
            return Err(format!(
                "{}:{line_no}: timestamp {timestamp_micros} is not greater than the previous {prev_ts}",
                path.display()
            ));
        }
    }
    *previous = Some((frame_seq, timestamp_micros));
    Ok(())
}

fn pair_raw_records(
    teacher: &[RawTeacherRow],
    rgb: &[RawRgbRow],
    frames_path: &Path,
    rgb_path: &Path,
) -> Result<(Vec<RawPair>, PairCounts), String> {
    let mut counts = PairCounts::default();
    let mut pairs = Vec::new();
    let mut rgb_index = 0_usize;
    for row in teacher {
        while let Some(candidate) = rgb.get(rgb_index) {
            if candidate.record.frame_seq >= row.record.frame_seq {
                break;
            }
            rgb_index += 1;
            counts.unpaired_rgb += 1;
        }
        match rgb.get(rgb_index) {
            Some(candidate) if candidate.record.frame_seq == row.record.frame_seq => {
                if candidate.record.timestamp_micros != row.record.timestamp_micros {
                    return Err(format!(
                        "{}: paired frame_seq {} has teacher timestamp {} but {} declares {}",
                        frames_path.display(),
                        row.record.frame_seq,
                        row.record.timestamp_micros,
                        rgb_path.display(),
                        candidate.record.timestamp_micros
                    ));
                }
                counts.paired += 1;
                pairs.push(RawPair {
                    frame_seq: row.record.frame_seq,
                    timestamp_micros: row.record.timestamp_micros,
                    arkit: row.arkit,
                    rgb: candidate.record.clone(),
                });
                rgb_index += 1;
            }
            _ => counts.unpaired_teacher += 1,
        }
    }
    counts.unpaired_rgb += rgb.len().saturating_sub(rgb_index);
    let mut present = std::collections::BTreeSet::new();
    for row in teacher {
        present.insert(row.record.frame_seq);
    }
    for row in rgb {
        present.insert(row.record.frame_seq);
    }
    if let (Some(min), Some(max)) = (present.iter().next(), present.iter().next_back()) {
        counts.dropped_sequences =
            (*min..=*max).filter(|seq| !present.contains(seq)).count() as u64;
    }
    Ok((pairs, counts))
}

fn extract_raw(input: &InputTake, task_path: &Path) -> Result<TakeExtraction, String> {
    let raw = read_raw_pairs(input)?;
    let mut runtime = MediaPipeRuntime::from_task_path(task_path)
        .map_err(|error| format!("MediaPipe runtime init failed: {error}"))?;
    let rotation = input.rotation_degrees.unwrap_or(0);
    let mirrored = input.mirrored.unwrap_or(false);
    let mut frames = Vec::new();
    let mut previous_seq = None;
    for pair in &raw.pairs {
        let frame_path = input.path.join(&pair.rgb.reference_path);
        let image = decode_rgb_bin(
            &frame_path,
            pair.rgb.width_px,
            pair.rgb.height_px,
            &pair.rgb.pixel_format,
            rotation,
            mirrored,
        )?;
        let mp = infer_blink(&mut runtime, &image, pair.frame_seq, pair.timestamp_micros)?;
        let gap_before = previous_seq.is_some_and(|seq| pair.frame_seq != seq + 1);
        previous_seq = Some(pair.frame_seq);
        frames.push(make_frame(
            &input.take_id,
            Some(&raw.session_id),
            pair.frame_seq,
            pair.timestamp_micros,
            mp,
            Some(pair.arkit),
            None,
            Some(pair.rgb.reference_path.clone()),
            Some(pair.rgb.width_px),
            Some(pair.rgb.height_px),
            Some(pair.rgb.pixel_format.clone()),
            Some(pair.rgb.orientation_degrees),
            Some(pair.rgb.mirrored),
            gap_before,
        ));
    }
    let mut counts = TakeCounts::default();
    for frame in &frames {
        accumulate(&mut counts, frame);
    }
    let report = TakeReport {
        take_id: input.take_id.clone(),
        kind: input.kind,
        path: input.path.display().to_string(),
        take_root: input.path.display().to_string(),
        raw_frames_root: Some(input.path.display().to_string()),
        session_id: Some(raw.session_id),
        pixel_rotation_degrees: rotation,
        mirrored,
        declared_orientation_degrees: raw.declared_orientation_degrees,
        declared_mirrored: raw.declared_mirrored,
        input_hashes: raw.input_hashes,
        trace_sha256: None,
        task_bundle_sha256: Some(TASK_BUNDLE_SHA256.to_owned()),
        mediapipe_observation_source: "current_mediapipe",
        counts,
        excluded: raw.counts,
        notes: Vec::new(),
    };
    Ok(TakeExtraction { frames, report })
}

fn infer_blink(
    runtime: &mut MediaPipeRuntime,
    image: &image::RgbImage,
    frame_seq: u64,
    timestamp_micros: u64,
) -> Result<Option<(f32, f32)>, String> {
    let mut rgba = Vec::with_capacity((image.width() * image.height() * 4) as usize);
    for pixel in image.pixels() {
        rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
    }
    let frame = VideoFrame {
        seq: FrameSeq(frame_seq),
        captured_at: MonoTimeNs(timestamp_micros.saturating_mul(1000)),
        width: image.width(),
        height: image.height(),
        stride_bytes: (image.width() * 4) as usize,
        format: PixelFormat::Rgba8,
        data: Arc::from(rgba.into_boxed_slice()),
    };
    match runtime.infer_face_tracking(&frame) {
        Ok(FaceTrackingOutcome::Face(sample)) => Ok(Some((
            sample.blendshapes.get(MediaPipeBlendshape::EyeBlinkLeft),
            sample.blendshapes.get(MediaPipeBlendshape::EyeBlinkRight),
        ))),
        Ok(FaceTrackingOutcome::NoFace { .. }) => Ok(None),
        Err(error) => Err(format!(
            "MediaPipe inference failed for frame_seq {frame_seq}: {error}"
        )),
    }
}

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn make_frame(
    take_id: &str,
    session_id: Option<&str>,
    frame_seq: u64,
    timestamp_micros: u64,
    mp: Option<(f32, f32)>,
    arkit: Option<(f32, f32)>,
    quality: Option<f32>,
    rgb_reference: Option<String>,
    rgb_width_px: Option<u32>,
    rgb_height_px: Option<u32>,
    rgb_pixel_format: Option<String>,
    rgb_declared_orientation_degrees: Option<i32>,
    rgb_declared_mirrored: Option<bool>,
    gap_before: bool,
) -> ExtractedFrame {
    ExtractedFrame {
        take_id: take_id.to_owned(),
        session_id: session_id.map(str::to_owned),
        frame_seq,
        timestamp_micros,
        mp_blink_left: mp.map(|(left, _)| left),
        mp_blink_right: mp.map(|(_, right)| right),
        openness_left: mp.map(|(left, _)| 1.0 - left),
        openness_right: mp.map(|(_, right)| 1.0 - right),
        arkit_blink_left: arkit.map(|(left, _)| left),
        arkit_blink_right: arkit.map(|(_, right)| right),
        face_observed: mp.is_some(),
        gap_before,
        mp_landmark_presence_median: quality,
        rgb_reference,
        rgb_width_px,
        rgb_height_px,
        rgb_pixel_format,
        rgb_declared_orientation_degrees,
        rgb_declared_mirrored,
    }
}

fn accumulate(counts: &mut TakeCounts, frame: &ExtractedFrame) {
    counts.frames += 1;
    if frame.face_observed {
        counts.face_observed += 1;
        counts.mediapipe_present += 1;
    } else {
        counts.mediapipe_missing += 1;
    }
    if frame.arkit_blink_left.is_some() {
        counts.teacher_present += 1;
    } else {
        counts.teacher_missing += 1;
    }
    if frame.face_observed && frame.arkit_blink_left.is_some() {
        counts.both_present += 1;
    }
    if frame.gap_before {
        counts.gaps += 1;
    }
}

fn coefficient_at(
    values: &[f32],
    channel: ArkitBlendshape,
    path: &Path,
    line_no: usize,
) -> Result<f32, String> {
    let index = channel.index();
    let value = values.get(index).copied().ok_or_else(|| {
        format!(
            "{}:{line_no}: coefficient array is missing index {index}",
            path.display()
        )
    })?;
    validate_coefficient(value, path, line_no, index)
}

fn value_coefficient(
    values: &[Value],
    channel: ArkitBlendshape,
    path: &Path,
    line_no: usize,
) -> Result<f32, String> {
    let index = channel.index();
    let value = values.get(index).and_then(Value::as_f64).ok_or_else(|| {
        format!(
            "{}:{line_no}: coefficient array is missing numeric index {index}",
            path.display()
        )
    })? as f32;
    validate_coefficient(value, path, line_no, index)
}

fn validate_coefficient(
    value: f32,
    path: &Path,
    line_no: usize,
    index: usize,
) -> Result<f32, String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!(
            "{}:{line_no}: non-finite or out-of-range coefficient {value} at index {index}",
            path.display()
        ));
    }
    Ok(value)
}

fn read_lines(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    Ok(text.lines().map(str::to_owned).collect())
}

fn count_jsonl(path: &Path) -> Result<Option<u64>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(read_lines(path)?.len() as u64))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn read_json_optional<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    read_json(path).map(Some)
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:X}", hasher.finalize())
}

fn current_git_commit() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|text| text.trim().to_owned())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
    std::fs::write(path, text)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn write_jsonl<T: Serialize>(path: &Path, values: &[T]) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    for value in values {
        let line = serde_json::to_string(value)
            .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
        writeln!(writer, "{line}")
            .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    }
    Ok(())
}

fn write_csv(path: &Path, frames: &[ExtractedFrame]) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    writeln!(
        writer,
        "take_id,session_id,frame_seq,timestamp_micros,mp_blink_left,mp_blink_right,openness_left,openness_right,arkit_blink_left,arkit_blink_right,face_observed,gap_before,mp_landmark_presence_median,rgb_reference,rgb_width_px,rgb_height_px,rgb_pixel_format,rgb_orientation_degrees,rgb_mirrored"
    )
    .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    for frame in frames {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            csv_field(&frame.take_id),
            csv_field(frame.session_id.as_deref().unwrap_or("")),
            frame.frame_seq,
            frame.timestamp_micros,
            opt_f32(frame.mp_blink_left),
            opt_f32(frame.mp_blink_right),
            opt_f32(frame.openness_left),
            opt_f32(frame.openness_right),
            opt_f32(frame.arkit_blink_left),
            opt_f32(frame.arkit_blink_right),
            frame.face_observed,
            frame.gap_before,
            opt_f32(frame.mp_landmark_presence_median),
            csv_field(frame.rgb_reference.as_deref().unwrap_or("")),
            opt_u32(frame.rgb_width_px),
            opt_u32(frame.rgb_height_px),
            csv_field(frame.rgb_pixel_format.as_deref().unwrap_or("")),
            opt_i32(frame.rgb_declared_orientation_degrees),
            opt_bool(frame.rgb_declared_mirrored),
        )
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    }
    Ok(())
}

fn opt_f32(value: Option<f32>) -> String {
    value.map_or_else(String::new, |value| format!("{value}"))
}

fn opt_u32(value: Option<u32>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn opt_i32(value: Option<i32>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn opt_bool(value: Option<bool>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn coefficients(left: f32, right: f32) -> Vec<f32> {
        let mut values = vec![0.01_f32; ARKIT52_COUNT];
        values[ArkitBlendshape::EyeBlinkLeft.index()] = left;
        values[ArkitBlendshape::EyeBlinkRight.index()] = right;
        values
    }

    fn trace_input(directory: &Path, kind: InputKind) -> InputTake {
        InputTake {
            take_id: "take_x".into(),
            kind,
            path: directory.to_path_buf(),
            session_id: Some("session".into()),
            origin: None,
            rotation_degrees: None,
            mirrored: None,
            raw_frames_root: None,
        }
    }

    fn write_trace(directory: &Path, schema_version: u32, rows: &[Value]) -> Value {
        let mut trace = String::new();
        for row in rows {
            trace.push_str(&serde_json::to_string(row).unwrap());
            trace.push('\n');
        }
        std::fs::write(directory.join("derived-trace.jsonl"), &trace).unwrap();
        let metadata = json!({
            "schema_version": schema_version,
            "source_dataset": {
                "session_id": "session",
                "take_id": "take_x",
                "frame_count": rows.len(),
                "input_hashes": { "frames.jsonl": "AA" }
            },
            "config": {
                "task_bundle_sha256": "64184E22",
                "pixel_rotation_degrees": 180
            },
            "counts": { "paired": rows.len() },
            "trace_sha256": hex_sha256(trace.as_bytes())
        });
        std::fs::write(
            directory.join("replay-metadata.json"),
            serde_json::to_string_pretty(&metadata).unwrap(),
        )
        .unwrap();
        metadata
    }

    fn v1_row(seq: u64, timestamp: u64, mp: Vec<f32>, arkit: Vec<f32>) -> Value {
        json!({
            "frame_seq": seq,
            "timestamp_micros": timestamp,
            "mediapipe_observation": mp,
            "teacher": { "coefficients": arkit },
            "rgb_reference": {
                "reference_path": format!("frames/frame_{seq:010}.bin"),
                "width_px": 480,
                "height_px": 640,
                "pixel_format": "jpeg-rgb8-srgb-upright-nonmirrored",
                "orientation_degrees": 0,
                "mirrored": false
            }
        })
    }

    fn v2_row(seq: u64, timestamp: u64, mp: Vec<f32>, arkit: Vec<f32>) -> Value {
        json!({
            "frame_seq": seq,
            "timestamp_micros": timestamp,
            "mediapipe_observation": {
                "direct_coefficients": mp,
                "landmark_presence_median": 0.87
            },
            "teacher": { "coefficients": arkit },
            "rgb_reference": {
                "reference_path": format!("frames/frame_{seq:010}.bin"),
                "width_px": 480,
                "height_px": 640,
                "pixel_format": "jpeg-rgb8-srgb-upright-nonmirrored",
                "orientation_degrees": 0,
                "mirrored": false
            }
        })
    }

    #[test]
    fn v1_v2_and_raw_teacher_extract_the_same_blink_values() {
        let directory = tempfile::tempdir().unwrap();
        let mp = coefficients(0.72, 0.18);
        let arkit = coefficients(0.91, 0.05);
        write_trace(
            directory.path(),
            1,
            &[v1_row(0, 0, mp.clone(), arkit.clone())],
        );
        let v1 = extract_trace(&trace_input(directory.path(), InputKind::TraceV1)).unwrap();

        let directory_v2 = tempfile::tempdir().unwrap();
        write_trace(
            directory_v2.path(),
            2,
            &[v2_row(0, 0, mp.clone(), arkit.clone())],
        );
        let v2 = extract_trace(&trace_input(directory_v2.path(), InputKind::TraceV2)).unwrap();

        for extraction in [&v1, &v2] {
            let frame = &extraction.frames[0];
            assert_eq!(frame.mp_blink_left, Some(0.72));
            assert_eq!(frame.mp_blink_right, Some(0.18));
            assert_eq!(frame.arkit_blink_left, Some(0.91));
            assert_eq!(frame.arkit_blink_right, Some(0.05));
            assert_eq!(frame.openness_left, Some(1.0 - 0.72));
            assert!(frame.face_observed);
        }
        // v1 has no landmark quality and must not fabricate one.
        assert_eq!(v1.frames[0].mp_landmark_presence_median, None);
        assert_eq!(v2.frames[0].mp_landmark_presence_median, Some(0.87));

        // The raw teacher record carries the same ARKit blink values.
        let raw_directory = tempfile::tempdir().unwrap();
        write_raw_take(
            raw_directory.path(),
            &[(0, 0, arkit.clone())],
            &[(0, 0)],
            PairCounts {
                paired: 1,
                unpaired_teacher: 0,
                unpaired_rgb: 0,
                dropped_sequences: 0,
            },
        );
        let raw = read_raw_pairs(&raw_input(raw_directory.path())).unwrap();
        assert_eq!(raw.pairs[0].arkit, (0.91, 0.05));
    }

    #[test]
    fn null_mediapipe_observation_stays_unknown_not_zero() {
        let directory = tempfile::tempdir().unwrap();
        let mut row = v2_row(0, 0, coefficients(0.5, 0.5), coefficients(0.2, 0.2));
        row["mediapipe_observation"] = Value::Null;
        write_trace(directory.path(), 2, &[row]);
        let extraction = extract_trace(&trace_input(directory.path(), InputKind::TraceV2)).unwrap();
        let frame = &extraction.frames[0];
        assert_eq!(frame.mp_blink_left, None);
        assert_eq!(frame.openness_left, None);
        assert!(!frame.face_observed);
        assert_eq!(frame.arkit_blink_left, Some(0.2));
    }

    #[test]
    fn duplicate_and_regressed_sequences_are_rejected_with_file_line() {
        let directory = tempfile::tempdir().unwrap();
        let row = v2_row(0, 0, coefficients(0.1, 0.1), coefficients(0.1, 0.1));
        write_trace(directory.path(), 2, &[row.clone(), row]);
        let error = extract_trace(&trace_input(directory.path(), InputKind::TraceV2)).unwrap_err();
        assert!(error.contains(":2:"), "{error}");
        assert!(error.contains("duplicate frame_seq 0"), "{error}");

        let directory = tempfile::tempdir().unwrap();
        write_trace(
            directory.path(),
            2,
            &[
                v2_row(1, 0, coefficients(0.1, 0.1), coefficients(0.1, 0.1)),
                v2_row(0, 1, coefficients(0.1, 0.1), coefficients(0.1, 0.1)),
            ],
        );
        let error = extract_trace(&trace_input(directory.path(), InputKind::TraceV2)).unwrap_err();
        assert!(error.contains(":2:"), "{error}");
        assert!(error.contains("regressed"), "{error}");
    }

    #[test]
    fn wrong_array_length_and_non_finite_are_reported_with_file_line() {
        let directory = tempfile::tempdir().unwrap();
        let mut row = v2_row(0, 0, coefficients(0.1, 0.1), coefficients(0.1, 0.1));
        row["mediapipe_observation"]["direct_coefficients"] = json!([0.1, 0.2]);
        write_trace(directory.path(), 2, &[row]);
        let error = extract_trace(&trace_input(directory.path(), InputKind::TraceV2)).unwrap_err();
        assert!(error.contains(":1:"), "{error}");
        assert!(error.contains("expected 52"), "{error}");

        let directory = tempfile::tempdir().unwrap();
        let mut bad = coefficients(0.1, 0.1);
        bad[8] = 1.5;
        write_trace(
            directory.path(),
            2,
            &[v2_row(0, 0, bad, coefficients(0.1, 0.1))],
        );
        let error = extract_trace(&trace_input(directory.path(), InputKind::TraceV2)).unwrap_err();
        assert!(error.contains(":1:"), "{error}");
        assert!(error.contains("non-finite or out-of-range"), "{error}");
    }

    #[test]
    fn trace_schema_kind_mismatch_and_unknown_version_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        write_trace(directory.path(), 1, &[]);
        let error = extract_trace(&trace_input(directory.path(), InputKind::TraceV2)).unwrap_err();
        assert!(
            error.contains("does not match replay schema_version 1"),
            "{error}"
        );

        let directory = tempfile::tempdir().unwrap();
        write_trace(directory.path(), 7, &[]);
        let error = extract_trace(&trace_input(directory.path(), InputKind::TraceV1)).unwrap_err();
        assert!(
            error.contains("does not match replay schema_version 7"),
            "{error}"
        );
    }

    #[test]
    fn trace_frame_count_mismatch_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        write_trace(
            directory.path(),
            2,
            &[v2_row(0, 0, coefficients(0.1, 0.1), coefficients(0.1, 0.1))],
        );
        let metadata_path = directory.path().join("replay-metadata.json");
        let text = std::fs::read_to_string(&metadata_path).unwrap();
        let mut metadata: Value = serde_json::from_str(&text).unwrap();
        metadata["source_dataset"]["frame_count"] = json!(5);
        std::fs::write(&metadata_path, serde_json::to_string(&metadata).unwrap()).unwrap();
        let error = extract_trace(&trace_input(directory.path(), InputKind::TraceV2)).unwrap_err();
        assert!(error.contains("frame_count is 5"), "{error}");
    }

    #[test]
    fn trace_sha256_mismatch_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        write_trace(
            directory.path(),
            2,
            &[v2_row(0, 0, coefficients(0.1, 0.1), coefficients(0.1, 0.1))],
        );
        std::fs::write(directory.path().join("derived-trace.jsonl"), b"tampered\n").unwrap();
        let error = extract_trace(&trace_input(directory.path(), InputKind::TraceV2)).unwrap_err();
        assert!(error.contains("trace_sha256"), "{error}");
    }

    #[test]
    fn gap_before_marks_missing_sequences_and_keeps_frames() {
        let directory = tempfile::tempdir().unwrap();
        write_trace(
            directory.path(),
            2,
            &[
                v2_row(0, 0, coefficients(0.1, 0.1), coefficients(0.1, 0.1)),
                v2_row(1, 33_000, coefficients(0.1, 0.1), coefficients(0.1, 0.1)),
                v2_row(3, 99_000, coefficients(0.1, 0.1), coefficients(0.1, 0.1)),
            ],
        );
        let extraction = extract_trace(&trace_input(directory.path(), InputKind::TraceV2)).unwrap();
        assert_eq!(extraction.frames.len(), 3, "gaps are kept, not filled");
        assert!(!extraction.frames[1].gap_before);
        assert!(extraction.frames[2].gap_before);
        assert_eq!(extraction.report.counts.gaps, 1);
    }

    #[test]
    fn distinct_takes_keep_their_own_seq_zero_apart() {
        let first = tempfile::tempdir().unwrap();
        write_trace(
            first.path(),
            2,
            &[v2_row(0, 0, coefficients(0.1, 0.1), coefficients(0.1, 0.1))],
        );
        let second = tempfile::tempdir().unwrap();
        write_trace(
            second.path(),
            2,
            &[v2_row(0, 0, coefficients(0.2, 0.2), coefficients(0.2, 0.2))],
        );
        let mut second_input = trace_input(second.path(), InputKind::TraceV2);
        second_input.take_id = "take_y".into();
        let a = extract_trace(&trace_input(first.path(), InputKind::TraceV2)).unwrap();
        let b = extract_trace(&second_input).unwrap();
        assert_eq!(a.frames[0].take_id, "take_x");
        assert_eq!(b.frames[0].take_id, "take_y");
        assert_ne!(a.frames[0].mp_blink_left, b.frames[0].mp_blink_left);
    }

    fn raw_input(directory: &Path) -> InputTake {
        InputTake {
            take_id: "take_x".into(),
            kind: InputKind::Raw,
            path: directory.to_path_buf(),
            session_id: Some("session".into()),
            origin: None,
            rotation_degrees: None,
            mirrored: None,
            raw_frames_root: None,
        }
    }

    fn write_raw_take(
        directory: &Path,
        teacher: &[(u64, u64, Vec<f32>)],
        rgb: &[(u64, u64)],
        counts: PairCounts,
    ) {
        std::fs::write(
            directory.join("session.json"),
            serde_json::to_string(&json!({
                "schema_version": 1,
                "session_id": "session",
                "timestamp_domain": "monotonic-micros-since-session-start"
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            directory.join("capture.json"),
            serde_json::to_string(&json!({
                "stored_orientation_degrees": 180,
                "stored_mirrored": false
            }))
            .unwrap(),
        )
        .unwrap();
        let mut frames = String::new();
        for (seq, timestamp, coefficients) in teacher {
            frames.push_str(
                &serde_json::to_string(&json!({
                    "frame_seq": seq,
                    "timestamp_micros": timestamp,
                    "kind": "arkit_teacher",
                    "payload": { "coefficients_canonical": coefficients }
                }))
                .unwrap(),
            );
            frames.push('\n');
        }
        std::fs::write(directory.join("frames.jsonl"), frames).unwrap();
        let mut rgb_text = String::new();
        for (seq, timestamp) in rgb {
            rgb_text.push_str(
                &serde_json::to_string(&json!({
                    "frame_seq": seq,
                    "timestamp_micros": timestamp,
                    "reference_path": format!("frames/frame_{seq:010}.bin"),
                    "width_px": 480,
                    "height_px": 640,
                    "pixel_format": "jpeg-rgb8-srgb-upright-nonmirrored",
                    "orientation_degrees": 0,
                    "mirrored": false
                }))
                .unwrap(),
            );
            rgb_text.push('\n');
        }
        std::fs::write(directory.join("rgb.jsonl"), rgb_text).unwrap();
        std::fs::write(
            directory.join("manifest.json"),
            serde_json::to_string(&json!({
                "schema_version": 1,
                "counts": {
                    "paired": counts.paired,
                    "unpaired_teacher": counts.unpaired_teacher,
                    "unpaired_rgb": counts.unpaired_rgb,
                    "dropped_sequences": counts.dropped_sequences
                },
                "skews": []
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(directory.join("COMPLETED"), b"completed\n").unwrap();
    }

    #[test]
    fn raw_pairing_counts_gaps_and_preserves_declared_orientation() {
        let directory = tempfile::tempdir().unwrap();
        write_raw_take(
            directory.path(),
            &[
                (0, 0, coefficients(0.0, 0.0)),
                (1, 16_000, coefficients(0.5, 0.5)),
                (3, 48_000, coefficients(1.0, 1.0)),
            ],
            &[(0, 0), (1, 16_000), (3, 48_000)],
            PairCounts {
                paired: 3,
                unpaired_teacher: 0,
                unpaired_rgb: 0,
                dropped_sequences: 1,
            },
        );
        let raw = read_raw_pairs(&raw_input(directory.path())).unwrap();
        assert_eq!(raw.counts.paired, 3);
        assert_eq!(raw.counts.dropped_sequences, 1);
        assert_eq!(raw.declared_orientation_degrees, Some(180));
        assert_eq!(raw.pairs[2].arkit, (1.0, 1.0));
        // The declared orientation is recorded, never auto-applied.
        assert_eq!(raw_input(directory.path()).rotation_degrees, None);
    }

    #[test]
    fn raw_pairing_rejects_a_paired_timestamp_mismatch() {
        let directory = tempfile::tempdir().unwrap();
        write_raw_take(
            directory.path(),
            &[(1, 10, coefficients(0.1, 0.1))],
            &[(1, 20)],
            PairCounts::default(),
        );
        let error = read_raw_pairs(&raw_input(directory.path())).unwrap_err();
        assert!(error.contains("teacher timestamp 10"), "{error}");
        assert!(error.contains("rgb.jsonl"), "{error}");
    }

    #[test]
    fn raw_manifest_count_mismatch_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        write_raw_take(
            directory.path(),
            &[(0, 0, coefficients(0.1, 0.1))],
            &[(0, 0)],
            PairCounts {
                paired: 2,
                unpaired_teacher: 0,
                unpaired_rgb: 0,
                dropped_sequences: 0,
            },
        );
        let error = read_raw_pairs(&raw_input(directory.path())).unwrap_err();
        assert!(error.contains("disagree with manifest.json"), "{error}");
    }

    #[test]
    fn raw_duplicate_and_regressed_sequences_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        write_raw_take(
            directory.path(),
            &[
                (1, 0, coefficients(0.1, 0.1)),
                (1, 1, coefficients(0.1, 0.1)),
            ],
            &[],
            PairCounts::default(),
        );
        let error = read_raw_pairs(&raw_input(directory.path())).unwrap_err();
        assert!(error.contains("frames.jsonl:2"), "{error}");
        assert!(error.contains("duplicate frame_seq 1"), "{error}");
    }

    #[test]
    fn missing_completion_marker_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        write_raw_take(
            directory.path(),
            &[(0, 0, coefficients(0.1, 0.1))],
            &[(0, 0)],
            PairCounts {
                paired: 1,
                unpaired_teacher: 0,
                unpaired_rgb: 0,
                dropped_sequences: 0,
            },
        );
        std::fs::remove_file(directory.path().join("COMPLETED")).unwrap();
        let error = read_raw_pairs(&raw_input(directory.path())).unwrap_err();
        assert!(error.contains("missing COMPLETED"), "{error}");
    }

    #[test]
    fn extract_writes_jsonl_csv_and_metadata() {
        let directory = tempfile::tempdir().unwrap();
        write_trace(
            directory.path(),
            2,
            &[
                v2_row(0, 0, coefficients(0.2, 0.3), coefficients(0.4, 0.5)),
                v2_row(1, 33_000, coefficients(0.6, 0.7), coefficients(0.8, 0.9)),
            ],
        );
        let inputs_path = directory.path().join("inputs.json");
        let inputs = json!({
            "schema_version": 1,
            "inputs": [{
                "take_id": "take_x",
                "kind": "trace_v2",
                "path": directory.path(),
                "session_id": "session"
            }]
        });
        std::fs::write(&inputs_path, serde_json::to_string(&inputs).unwrap()).unwrap();
        let output = directory.path().join("extracted");
        let options = Options {
            inputs: Some(inputs_path),
            data: None,
            labels: None,
            split: None,
            profile: None,
            output: output.clone(),
            project_root: directory.path().to_path_buf(),
        };
        run_extract(&options).unwrap();

        let jsonl = std::fs::read_to_string(output.join("eye-frames.jsonl")).unwrap();
        assert_eq!(jsonl.lines().count(), 2);
        let csv = std::fs::read_to_string(output.join("eye-frames.csv")).unwrap();
        assert_eq!(csv.lines().count(), 3, "header plus two rows");
        assert!(csv.lines().next().unwrap().contains("openness_left"));
        let metadata: Value = serde_json::from_str(
            &std::fs::read_to_string(output.join("extraction-metadata.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["totals"]["frames"], json!(2));
        assert_eq!(
            metadata["inputs"][0]["task_bundle_sha256"],
            json!("64184E22")
        );
    }
}
