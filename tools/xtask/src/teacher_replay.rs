//! Offline replay of a paired ARKit teacher capture dataset through the
//! existing MediaPipe face pipeline and the deterministic GNM baseline
//! (GNM #68.3 / Issue #110).
//!
//! The tool reads a completed capture dataset (GNM #68.2 layout), re-runs the
//! exact production MediaPipe face-landmarker path over the stored RGB
//! frames, computes the deterministic cold-start GNM projection beside it,
//! and writes a derived `PairedTemporalSample` trace:
//!
//! - RGB references and teacher records are joined by exact `frame_seq` and
//!   identical timestamps only; missing, duplicate, or mismatched frames are
//!   rejected instead of nearest-repaired.
//! - The trace stores the MediaPipe-derived ARKit52 observation, the
//!   deterministic GNM state (projected ARKit52 + solver residual), and the
//!   baseline output production would have published.
//! - Every input file and the derived trace bytes are hashed (SHA-256), so a
//!   re-run over the same inputs/config regenerates byte-identical outputs.
//! - Raw RGB payloads stay outside version control (GNM #68.1); the trace
//!   stores references and derived numbers only. See `docs/teacher-replay.md`
//!   for the post-replay raw deletion workflow.
//!
//! Determinism notes: the fit is a stateless cold-start per frame (no
//! warm-start lifecycle), the identity is the fixed model-neutral identity,
//! and no auxiliary objective term is enabled. `tongue_out` has no MediaPipe
//! channel and is pinned to `0.0` in the direct observation. The baseline
//! output equals the direct MediaPipe observation because the production
//! baseline applies no temporal prior.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::ImageReader;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vtuber_core::face_tracking::{FaceTrackingOutcome, MediaPipeBlendshape};
use vtuber_core::types::{FrameSeq, MonoTimeNs, PixelFormat, VideoFrame};
use vtuber_core::{Arkit52Coefficients, ArkitBlendshape};
use vtuber_gnm::{
    DenseCorrespondenceSet, DenseCoveragePolicy, DenseObservationStatus, DenseProjection,
    DenseRegionGroups, FixedGnmIdentity, GnmIdentityCalibration, GnmJointState, GnmModel,
    GnmSparseVertices, IdentityFitDiagnostics, NeutralPoseDiversity, SingleFrameFitConfig,
    compute_gnm_facial_features, fit_single_frame_cold_start, fitting_projection, load_gnm_head_v3,
    normalization_scales_from_mapping, repository_dense_mapping,
};
use vtuber_inference::backend::mediapipe::MediaPipeRuntime;
use vtuber_inference::runtime::FaceTrackingInference;
use vtuber_tracking::arkit_teacher::HEAD_TRANSFORM_CONVENTION;
use vtuber_tracking::{
    ArkitTeacherFrame, DeterministicGnmState, HeadTransform, PairedTemporalSample,
    RgbFrameReference, decode_gnm_arkit52, validate_paired_samples,
};

/// `tools/teacher-capture` `TIMESTAMP_DOMAIN` mirror (standalone crate).
const TIMESTAMP_DOMAIN: &str = "monotonic-micros-since-session-start";

const TEACHER_KIND: &str = "arkit_teacher";
const RGB_PIXEL_PREFIX: &str = "frames/frame_";
const COVERAGE_MIN_VALID_POINTS: usize = 2;
const COVERAGE_DEGRADED_FRACTION: f32 = 0.75;

/// Parsed CLI options for `teacher-replay`.
pub struct Options {
    dataset: PathBuf,
    output: PathBuf,
    task: PathBuf,
    gnm_model: PathBuf,
    fit_max_iterations: usize,
    fit_tolerance: f32,
    pixel_rotation_degrees: u16,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut dataset = None;
        let mut output = None;
        let mut task = None;
        let mut gnm_model = None;
        let mut fit_max_iterations = 64_usize;
        let mut fit_tolerance = 1.0e-6_f32;
        let mut pixel_rotation_degrees = 0_u16;
        let mut index = 0;
        while index < args.len() {
            let next = |index: &mut usize, flag: &str| -> Result<String, String> {
                *index += 1;
                args.get(*index)
                    .cloned()
                    .ok_or_else(|| format!("{flag} requires a value"))
            };
            // Bounds are proven by the loop condition; see AGENTS.md panic policy.
            #[allow(clippy::indexing_slicing)]
            match args[index].as_str() {
                "--dataset" => dataset = Some(next(&mut index, "--dataset")?),
                "--output" => output = Some(next(&mut index, "--output")?),
                "--task" => task = Some(next(&mut index, "--task")?),
                "--gnm-model" => gnm_model = Some(next(&mut index, "--gnm-model")?),
                "--fit-max-iterations" => {
                    fit_max_iterations = next(&mut index, "--fit-max-iterations")?
                        .parse()
                        .map_err(|_| "--fit-max-iterations must be a positive integer")?;
                }
                "--fit-tolerance" => {
                    fit_tolerance = next(&mut index, "--fit-tolerance")?
                        .parse()
                        .map_err(|_| "--fit-tolerance must be a float")?;
                }
                "--pixel-rotation" => {
                    pixel_rotation_degrees = next(&mut index, "--pixel-rotation")?
                        .parse()
                        .map_err(|_| "--pixel-rotation must be 0, 90, 180, or 270")?;
                    if !matches!(pixel_rotation_degrees, 0 | 90 | 180 | 270) {
                        return Err("--pixel-rotation must be 0, 90, 180, or 270".to_owned());
                    }
                }
                other => return Err(format!("unknown option {other}")),
            }
            index += 1;
        }
        let dataset = dataset
            .map(PathBuf::from)
            .ok_or("--dataset <capture-dataset-dir> is required")?;
        let take_id = dataset
            .file_name()
            .map(|name| name.to_owned())
            .ok_or("--dataset must point at the capture dataset directory")?;
        let output = output.map_or_else(
            || PathBuf::from("data/datasets").join(take_id),
            PathBuf::from,
        );
        Ok(Self {
            dataset,
            output,
            task: task.map_or_else(
                || PathBuf::from("assets/models/face_landmarker.task"),
                PathBuf::from,
            ),
            gnm_model: gnm_model.map_or_else(
                || PathBuf::from("assets/models/gnm_head.npz"),
                PathBuf::from,
            ),
            fit_max_iterations,
            fit_tolerance,
            pixel_rotation_degrees,
        })
    }
}

/// Prints command help.
pub fn print_help() {
    println!(
        "  teacher-replay --dataset <capture-dataset-dir> [--output <dir>]\n\
         *                            [--task <face_landmarker.task>] [--gnm-model <gnm_head.npz>]\n\
         *                            [--pixel-rotation <0|90|180|270>]\n\
         *                            [--fit-max-iterations <n>] [--fit-tolerance <f>]\n\
         *   Offline replay of a completed ARKit teacher capture (GNM #68.3): re-runs\n\
         *   MediaPipe + deterministic GNM over the stored RGB frames and writes a\n\
         *   derived PairedTemporalSample trace with hashes for reproducibility.\n\
         *   --pixel-rotation corrects a mis-declared capture orientation without\n\
         *   touching the stored dataset bytes (recorded in replay-metadata.json)."
    );
}

// ---------------------------------------------------------------------------
// Capture dataset records (deserialization mirrors of the #68.2 layout)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CaptureSessionFile {
    schema_version: u32,
    session_id: String,
    timestamp_domain: String,
}

#[derive(Deserialize)]
struct TeacherRecord {
    frame_seq: u64,
    timestamp_micros: u64,
    kind: String,
    payload: TeacherPayload,
}

#[derive(Deserialize)]
struct TeacherPayload {
    coefficients_canonical: Vec<f32>,
    rotation_quaternion_wxyz: [f32; 4],
    translation_meters: [f32; 3],
    head_transform_convention: String,
}

#[derive(Deserialize)]
struct RgbRecord {
    frame_seq: u64,
    timestamp_micros: u64,
    reference_path: String,
    width_px: u32,
    height_px: u32,
    pixel_format: String,
    orientation_degrees: u16,
    mirrored: bool,
}

#[derive(Deserialize)]
struct CaptureManifestFile {
    schema_version: u32,
    counts: CaptureManifestCounts,
}

#[derive(Debug, Deserialize)]
struct CaptureManifestCounts {
    paired: usize,
    unpaired_teacher: usize,
    unpaired_rgb: usize,
    dropped_sequences: u64,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut records = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<T>(line)
            .map_err(|error| format!("{} line {}: {error}", path.display(), line_index + 1))?;
        records.push(record);
    }
    Ok(records)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn sha256_hex(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("hash {}: {error}", path.display()))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect())
}

// ---------------------------------------------------------------------------
// Exact-identity pairing (fail-closed, no nearest repair)
// ---------------------------------------------------------------------------

/// Exact-identity classification of both record streams.
struct PairedSets {
    /// `(teacher index, rgb index)` pairs with identical `frame_seq` and
    /// timestamp, in strictly increasing sequence order.
    paired: Vec<(usize, usize)>,
    /// Sequences present only in the teacher stream.
    teacher_only: Vec<u64>,
    /// Sequences present only in the RGB stream.
    rgb_only: Vec<u64>,
    /// Sequences inside the observed range present on neither side.
    dropped: Vec<u64>,
}

/// Classifies both streams by exact identity.
///
/// Per-side duplicates/regressions and any paired frame whose timestamps
/// disagree are hard errors. Unpaired and dropped sequences are *reported*,
/// never nearest-repaired; the caller must cross-check them against the
/// capture manifest before proceeding.
fn classify_frames(teacher: &[TeacherRecord], rgb: &[RgbRecord]) -> Result<PairedSets, String> {
    check_strict_identity(
        teacher
            .iter()
            .map(|record| (record.frame_seq, record.timestamp_micros)),
        "teacher",
    )?;
    check_strict_identity(
        rgb.iter()
            .map(|record| (record.frame_seq, record.timestamp_micros)),
        "rgb",
    )?;

    let mut paired = Vec::new();
    let mut teacher_only = Vec::new();
    let mut rgb_only = Vec::new();
    let mut rgb_index = 0_usize;
    for (teacher_index, teacher_record) in teacher.iter().enumerate() {
        while rgb
            .get(rgb_index)
            .is_some_and(|record| record.frame_seq < teacher_record.frame_seq)
        {
            // Bounds: guarded by the `is_some_and` check above.
            #[allow(clippy::indexing_slicing)]
            {
                rgb_only.push(rgb[rgb_index].frame_seq);
            }
            rgb_index += 1;
        }
        match rgb.get(rgb_index) {
            Some(rgb_record) if rgb_record.frame_seq == teacher_record.frame_seq => {
                if rgb_record.timestamp_micros != teacher_record.timestamp_micros {
                    return Err(format!(
                        "identity mismatch at seq {}: teacher time {} vs rgb time {}",
                        teacher_record.frame_seq,
                        teacher_record.timestamp_micros,
                        rgb_record.timestamp_micros
                    ));
                }
                paired.push((teacher_index, rgb_index));
                rgb_index += 1;
            }
            _ => teacher_only.push(teacher_record.frame_seq),
        }
    }
    while rgb_index < rgb.len() {
        // Bounds: guarded by the loop condition above.
        #[allow(clippy::indexing_slicing)]
        {
            rgb_only.push(rgb[rgb_index].frame_seq);
        }
        rgb_index += 1;
    }

    let mut all_seqs: Vec<u64> = teacher
        .iter()
        .map(|record| record.frame_seq)
        .chain(rgb.iter().map(|record| record.frame_seq))
        .collect();
    all_seqs.sort_unstable();
    all_seqs.dedup();
    let dropped = match (all_seqs.first(), all_seqs.last()) {
        (Some(&first), Some(&last)) => {
            let present: std::collections::BTreeSet<u64> = all_seqs.into_iter().collect();
            (first..=last)
                .filter(|seq| !present.contains(seq))
                .collect()
        }
        _ => Vec::new(),
    };

    Ok(PairedSets {
        paired,
        teacher_only,
        rgb_only,
        dropped,
    })
}

fn check_strict_identity<I: Iterator<Item = (u64, u64)>>(
    records: I,
    side: &str,
) -> Result<(), String> {
    let mut last: Option<(u64, u64)> = None;
    for (frame_seq, timestamp_micros) in records {
        if let Some((previous_seq, previous_time)) = last {
            if frame_seq == previous_seq {
                return Err(format!("{side}: duplicate frame_seq {frame_seq}"));
            }
            if frame_seq < previous_seq {
                return Err(format!(
                    "{side}: frame_seq regressed {previous_seq} -> {frame_seq}"
                ));
            }
            if timestamp_micros <= previous_time {
                return Err(format!(
                    "{side}: non-monotonic timestamp at seq {frame_seq} ({previous_time} -> {timestamp_micros})"
                ));
            }
        }
        last = Some((frame_seq, timestamp_micros));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Derived trace row format (shared with `teacher-fit-prior`)
// ---------------------------------------------------------------------------

/// One derived `PairedTemporalSample` row as serialized in the trace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceRow {
    pub frame_seq: u64,
    pub timestamp_micros: u64,
    pub mediapipe_observation: Option<Vec<f32>>,
    pub gnm_state: Option<TraceGnmState>,
    pub baseline_output: Vec<f32>,
    pub teacher: Option<TraceTeacher>,
    pub rgb_reference: Option<TraceRgbReference>,
}

/// Serialized `DeterministicGnmState`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceGnmState {
    pub projected_coefficients: Vec<f32>,
    pub residual: f32,
}

/// Serialized `ArkitTeacherFrame`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceTeacher {
    pub frame_seq: u64,
    pub timestamp_micros: u64,
    pub coefficients: Vec<f32>,
    pub head_transform: TraceHeadTransform,
}

/// Serialized `HeadTransform`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceHeadTransform {
    pub rotation_unit_quaternion_wxyz: [f32; 4],
    pub translation_meters: [f32; 3],
}

/// Serialized `RgbFrameReference`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceRgbReference {
    pub reference_path: String,
    pub width_px: u32,
    pub height_px: u32,
    pub pixel_format: String,
    pub orientation_degrees: u16,
    pub mirrored: bool,
}

/// Converts a validated sample into its trace representation.
#[must_use]
pub fn trace_row(sample: &PairedTemporalSample) -> TraceRow {
    TraceRow {
        frame_seq: sample.frame_seq,
        timestamp_micros: sample.timestamp_micros,
        mediapipe_observation: sample
            .mediapipe_observation
            .as_ref()
            .map(|values| values.as_array().to_vec()),
        gnm_state: sample.gnm_state.as_ref().map(|state| TraceGnmState {
            projected_coefficients: state.projected_coefficients.as_array().to_vec(),
            residual: state.residual,
        }),
        baseline_output: sample.baseline_output.as_array().to_vec(),
        teacher: sample.teacher.as_ref().map(|teacher| TraceTeacher {
            frame_seq: teacher.frame_seq,
            timestamp_micros: teacher.timestamp_micros,
            coefficients: teacher.coefficients.as_array().to_vec(),
            head_transform: TraceHeadTransform {
                rotation_unit_quaternion_wxyz: teacher.head_transform.rotation_unit_quaternion_wxyz,
                translation_meters: teacher.head_transform.translation_meters,
            },
        }),
        rgb_reference: sample.rgb_reference.as_ref().map(|rgb| TraceRgbReference {
            reference_path: rgb.reference_path.clone(),
            width_px: rgb.width_px,
            height_px: rgb.height_px,
            pixel_format: rgb.pixel_format.clone(),
            orientation_degrees: rgb.orientation_degrees,
            mirrored: rgb.mirrored,
        }),
    }
}

/// Rebuilds a sample from its trace row; used by downstream tooling.
///
/// # Errors
///
/// Fails closed on dimension, range, or name-contract violations.
pub fn sample_from_row(row: &TraceRow) -> Result<PairedTemporalSample, String> {
    Ok(PairedTemporalSample {
        frame_seq: row.frame_seq,
        timestamp_micros: row.timestamp_micros,
        mediapipe_observation: coefficients_from_row(&row.mediapipe_observation)?,
        gnm_state: match &row.gnm_state {
            Some(state) => Some(DeterministicGnmState {
                projected_coefficients: coefficients_from_slice(
                    &state.projected_coefficients,
                    "gnm projected_coefficients",
                )?,
                residual: state.residual,
            }),
            None => None,
        },
        baseline_output: coefficients_from_slice(&row.baseline_output, "baseline_output")?,
        teacher: match &row.teacher {
            Some(teacher) => Some(ArkitTeacherFrame {
                frame_seq: teacher.frame_seq,
                timestamp_micros: teacher.timestamp_micros,
                coefficients: coefficients_from_slice(
                    &teacher.coefficients,
                    "teacher coefficients",
                )?,
                head_transform: HeadTransform {
                    rotation_unit_quaternion_wxyz: teacher
                        .head_transform
                        .rotation_unit_quaternion_wxyz,
                    translation_meters: teacher.head_transform.translation_meters,
                },
            }),
            None => None,
        },
        rgb_reference: row.rgb_reference.as_ref().map(|rgb| RgbFrameReference {
            reference_path: rgb.reference_path.clone(),
            width_px: rgb.width_px,
            height_px: rgb.height_px,
            pixel_format: rgb.pixel_format.clone(),
            orientation_degrees: rgb.orientation_degrees,
            mirrored: rgb.mirrored,
        }),
    })
}

fn coefficients_from_row(values: &Option<Vec<f32>>) -> Result<Option<Arkit52Coefficients>, String> {
    values
        .as_ref()
        .map(|values| coefficients_from_slice(values, "mediapipe_observation"))
        .transpose()
}

fn coefficients_from_slice(values: &[f32], field: &str) -> Result<Arkit52Coefficients, String> {
    let array: [f32; 52] = values
        .try_into()
        .map_err(|_| format!("{field}: expected 52 channels, got {}", values.len()))?;
    Arkit52Coefficients::try_from_array(array).map_err(|error| format!("{field}: {error:?}"))
}

/// Direct MediaPipe blendshape -> ARKit52 observation.
///
/// `tongue_out` has no MediaPipe category and is pinned to `0.0`; every other
/// canonical channel must map by name exactly once.
pub fn mediapipe_to_arkit52(
    set: &vtuber_core::face_tracking::FaceBlendshapeSet,
) -> Result<Arkit52Coefficients, String> {
    let mut values = [0.0_f32; 52];
    let mut mapped = 0_usize;
    for category in MediaPipeBlendshape::ALL {
        if category == MediaPipeBlendshape::Neutral {
            continue;
        }
        let name = category.as_str();
        let Some(channel) = ArkitBlendshape::from_name(name) else {
            return Err(format!("MediaPipe category {name} has no ARKit52 channel"));
        };
        // Invariant: `ArkitBlendshape::index()` < 52 (vtuber-core contract).
        #[allow(clippy::indexing_slicing)]
        {
            values[channel.index()] = set.get(category);
        }
        mapped += 1;
    }
    if mapped != 51 {
        return Err(format!(
            "expected exactly 51 named MediaPipe->ARKit52 channels, mapped {mapped}"
        ));
    }
    // Invariant: `ArkitBlendshape::TongueOut.index()` < 52.
    #[allow(clippy::indexing_slicing)]
    {
        values[ArkitBlendshape::TongueOut.index()] = 0.0;
    }
    Arkit52Coefficients::try_from_array(values).map_err(|error| format!("observation: {error:?}"))
}

// ---------------------------------------------------------------------------
// GNM deterministic baseline context
// ---------------------------------------------------------------------------

struct GnmContext {
    model: GnmModel,
    mapping: DenseCorrespondenceSet,
    groups: DenseRegionGroups,
    calibration: GnmIdentityCalibration,
    identity: FixedGnmIdentity,
    neutral_projection: DenseProjection,
    fit_config: SingleFrameFitConfig,
    coverage: DenseCoveragePolicy,
    model_sha256: String,
}

fn build_gnm_context(
    model_path: &Path,
    fit_config: SingleFrameFitConfig,
) -> Result<GnmContext, String> {
    let model = load_gnm_head_v3(model_path)
        .map_err(|error| format!("load {}: {error}", model_path.display()))?;
    let mapping = repository_dense_mapping()
        .bind(&model)
        .map_err(|error| format!("bind dense mapping: {error}"))?;
    let identity = FixedGnmIdentity::new(model.neutral_identity(), &model)
        .map_err(|error| error.to_string())?;
    let neutral_expression = model.neutral_expression();
    let mut surface = GnmSparseVertices::with_len(mapping.len());
    mapping
        .evaluate_surface(
            &model,
            identity.state(),
            &neutral_expression,
            &GnmJointState::neutral(model.joint_count()),
            &mut surface,
        )
        .map_err(|error| format!("evaluate neutral surface: {error}"))?;
    let neutral_surface = surface.values().to_vec();
    let scales = normalization_scales_from_mapping(&mapping, &neutral_surface);
    let calibration = GnmIdentityCalibration::new(
        &model,
        mapping.version(),
        identity.clone(),
        neutral_expression,
        neutral_surface,
        scales,
        IdentityFitDiagnostics {
            accepted_samples: 0,
            rejected_samples: 0,
            reprojection_rms: 0.0,
            // The fixed neutral identity solves no dimensions; the whole
            // identity vector is carried as-is (validation requires the
            // reported span to stay inside the model dimension).
            active_identity_dimension: model.identity_dimension(),
            condition_number: None,
            pose_diversity: NeutralPoseDiversity {
                yaw_span_radians: 0.0,
                pitch_span_radians: 0.0,
                near_duplicate_fraction: 0.0,
            },
        },
    )
    .map_err(|error| error.to_string())?;
    let groups = DenseRegionGroups::from_set(&mapping)
        .map_err(|error| format!("partition dense regions: {error}"))?;
    let neutral_projection =
        fitting_projection(surface.values(), [0.0; 3]).map_err(|error| error.to_string())?;
    let coverage = DenseCoveragePolicy::new(COVERAGE_MIN_VALID_POINTS, COVERAGE_DEGRADED_FRACTION)
        .map_err(|error| error.to_string())?;
    let model_sha256 = sha256_hex(model_path)?;
    Ok(GnmContext {
        model,
        mapping,
        groups,
        calibration,
        identity,
        neutral_projection,
        fit_config,
        coverage,
        model_sha256,
    })
}

enum GnmFrameOutcome {
    Solved(DeterministicGnmState),
    InsufficientCoverage,
    FitRejected(String),
}

fn fit_frame_gnm(
    context: &GnmContext,
    frame_seq: u64,
    timestamp_micros: u64,
    sample: &vtuber_core::face_tracking::FaceTrackingSample,
) -> Result<GnmFrameOutcome, String> {
    let points: Vec<[f32; 2]> = sample.landmarks.iter().map(|p| [p.x, p.y]).collect();
    let observation = vtuber_gnm::GnmDenseObservation::from_mediapipe_xy(
        frame_seq,
        timestamp_micros,
        &points,
        &context.mapping,
        context.coverage,
    )
    .map_err(|error| format!("frame {frame_seq}: dense observation: {error}"))?;
    if observation.coverage().status == DenseObservationStatus::Insufficient {
        return Ok(GnmFrameOutcome::InsufficientCoverage);
    }
    // Recover the rigid pose/camera projection from this frame's observation
    // against the neutral state first (same initialization as the production
    // reinitialize-dynamic-state path); a raw neutral projection makes the
    // cold-start crawl and miss its convergence budget.
    //
    // A recovery failure here is a per-frame numeric degeneracy (the config is
    // fixed at compile time and coverage was already checked), not a data
    // integrity problem: record it as a per-frame fit rejection so the rest of
    // the take still produces a trace, mirroring the invalid cold-start outcome
    // path below. Manifest/timestamp contradictions keep aborting fail-closed
    // at the dataset validation layer.
    let recovered = match vtuber_gnm::recover_rigid_projection(
        &context.model,
        context.identity.state(),
        &context.model.neutral_expression(),
        &GnmJointState::neutral(context.model.joint_count()),
        &context.mapping,
        &observation,
        context.neutral_projection,
        vtuber_gnm::RigidRecoveryConfig::default(),
    ) {
        Ok(recovered) => recovered,
        Err(error) => {
            return Ok(GnmFrameOutcome::FitRejected(format!(
                "frame {frame_seq}: rigid recovery diverged: {error}"
            )));
        }
    };
    let outcome = fit_single_frame_cold_start(
        &context.model,
        context.identity.state(),
        &context.model.neutral_expression(),
        &GnmJointState::neutral(context.model.joint_count()),
        &context.mapping,
        &observation,
        &recovered.projection,
        context.fit_config,
        None,
    )
    .map_err(|error| format!("frame {frame_seq}: GNM fit: {error}"))?;
    if !outcome.valid() {
        return Ok(GnmFrameOutcome::FitRejected(format!(
            "frame {frame_seq}: fit status {:?} after {} iterations, objective {}",
            outcome.status(),
            outcome.iterations(),
            outcome.objective()
        )));
    }
    let features = compute_gnm_facial_features(
        &context.model,
        context.identity.state(),
        outcome.expression(),
        outcome.joints(),
        &context.mapping,
        &context.groups,
        &context.calibration,
    )
    .map_err(|error| format!("frame {frame_seq}: facial features: {error}"))?;
    let decoded = decode_gnm_arkit52(&features)
        .map_err(|error| format!("frame {frame_seq}: ARKit52 decode: {error:?}"))?;
    Ok(GnmFrameOutcome::Solved(DeterministicGnmState {
        projected_coefficients: decoded.coefficients,
        residual: outcome.objective(),
    }))
}

// ---------------------------------------------------------------------------
// Replay execution
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ReplayMetadata {
    schema_version: u32,
    tool: &'static str,
    xtask_version: String,
    source_dataset: SourceDatasetMetadata,
    config: ReplayConfigMetadata,
    counts: ReplayCountsMetadata,
    trace_sha256: String,
}

#[derive(Serialize)]
struct SourceDatasetMetadata {
    session_id: String,
    take_id: String,
    frame_count: usize,
    input_hashes: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct ReplayConfigMetadata {
    task_bundle_sha256: String,
    gnm_model_sha256: String,
    dense_mapping_schema_revision: u32,
    fit_max_iterations: usize,
    fit_tolerance: f32,
    coverage_min_valid_points: usize,
    coverage_degraded_below_fraction: f32,
    identity: &'static str,
    auxiliary: &'static str,
    tongue_out_policy: &'static str,
    baseline_output_policy: &'static str,
    paced_to_capture_cadence: bool,
    pixel_rotation_degrees: u16,
}

#[derive(Serialize)]
struct ReplayCountsMetadata {
    paired: usize,
    solved: usize,
    no_face: usize,
    observation_insufficient: usize,
    fit_rejected: usize,
    excluded_unpaired_teacher: usize,
    excluded_unpaired_rgb: usize,
}

/// Runs the replay; see the module documentation and `docs/teacher-replay.md`.
///
/// # Errors
///
/// Fails closed on dataset contract violations, MediaPipe/GNM failures that
/// invalidate the deterministic contract, and output I/O errors.
#[allow(clippy::too_many_lines)]
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let dataset = &options.dataset;

    if !dataset.join("COMPLETED").is_file() {
        return Err(format!(
            "{} has no COMPLETED marker; refusing to replay a partial capture",
            dataset.display()
        ));
    }
    let session: CaptureSessionFile = read_json(&dataset.join("session.json"))?;
    if session.schema_version != 1 {
        return Err(format!(
            "unsupported dataset schema {}",
            session.schema_version
        ));
    }
    if session.timestamp_domain != TIMESTAMP_DOMAIN {
        return Err(format!(
            "unexpected timestamp domain {:?} (expected {:?})",
            session.timestamp_domain, TIMESTAMP_DOMAIN
        ));
    }
    let manifest: CaptureManifestFile = read_json(&dataset.join("manifest.json"))?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported manifest schema {}",
            manifest.schema_version
        ));
    }

    let teacher: Vec<TeacherRecord> = read_jsonl(&dataset.join("frames.jsonl"))?;
    let rgb: Vec<RgbRecord> = read_jsonl(&dataset.join("rgb.jsonl"))?;
    if teacher.is_empty() {
        return Err("frames.jsonl contains no teacher records".to_owned());
    }
    let paired_sets = classify_frames(&teacher, &rgb)?;
    // The capture manifest must account for every unpaired/dropped sequence
    // exactly; an undeclared disagreement means a broken capture and aborts.
    if manifest.counts.paired != paired_sets.paired.len()
        || manifest.counts.unpaired_teacher != paired_sets.teacher_only.len()
        || manifest.counts.unpaired_rgb != paired_sets.rgb_only.len()
        || manifest.counts.dropped_sequences != paired_sets.dropped.len() as u64
    {
        return Err(format!(
            "manifest counts {:?} disagree with the exact-identity classification (paired {}, teacher-only {}, rgb-only {}, dropped {})",
            manifest.counts,
            paired_sets.paired.len(),
            paired_sets.teacher_only.len(),
            paired_sets.rgb_only.len(),
            paired_sets.dropped.len()
        ));
    }
    if paired_sets.paired.is_empty() {
        return Err("no exactly-paired frames remain after identity classification".to_owned());
    }

    // Per-record teacher/rgb validation and payload presence (paired frames
    // only; unpaired frames are excluded from the trace).
    for (teacher_index, rgb_index) in &paired_sets.paired {
        // Bounds: both indices come from classify_frames over these slices.
        #[allow(clippy::indexing_slicing)]
        let (teacher_record, rgb_record) = (&teacher[*teacher_index], &rgb[*rgb_index]);
        if teacher_record.kind != TEACHER_KIND {
            return Err(format!(
                "frame {}: unexpected record kind {:?}",
                teacher_record.frame_seq, teacher_record.kind
            ));
        }
        let payload = &teacher_record.payload;
        if payload.head_transform_convention != HEAD_TRANSFORM_CONVENTION {
            return Err(format!(
                "frame {}: head transform convention {:?} does not match the contract",
                teacher_record.frame_seq, payload.head_transform_convention
            ));
        }
        if payload.coefficients_canonical.len() != 52 {
            return Err(format!(
                "frame {}: expected 52 canonical coefficients, got {}",
                teacher_record.frame_seq,
                payload.coefficients_canonical.len()
            ));
        }
        if !matches!(rgb_record.orientation_degrees, 0 | 90 | 180 | 270) {
            return Err(format!(
                "frame {}: invalid rgb orientation {}",
                rgb_record.frame_seq, rgb_record.orientation_degrees
            ));
        }
        if rgb_record.width_px == 0 || rgb_record.height_px == 0 {
            return Err(format!(
                "frame {}: zero rgb dimensions",
                rgb_record.frame_seq
            ));
        }
        let expected_path = format!("{RGB_PIXEL_PREFIX}{:010}.bin", rgb_record.frame_seq);
        if rgb_record.reference_path != expected_path {
            return Err(format!(
                "frame {}: reference path {:?} does not match the canonical layout {:?}",
                rgb_record.frame_seq, rgb_record.reference_path, expected_path
            ));
        }
        let payload_path = dataset.join(&rgb_record.reference_path);
        if !payload_path.is_file() {
            return Err(format!(
                "frame {}: rgb payload {} is missing",
                rgb_record.frame_seq,
                payload_path.display()
            ));
        }
    }

    // Input hashes for reproducibility metadata.
    let mut input_hashes = BTreeMap::new();
    for name in ["session.json", "frames.jsonl", "rgb.jsonl", "manifest.json"] {
        input_hashes.insert(name.to_owned(), sha256_hex(&dataset.join(name))?);
    }
    let capture_path = dataset.join("capture.json");
    if capture_path.is_file() {
        input_hashes.insert("capture.json".to_owned(), sha256_hex(&capture_path)?);
    }

    // MediaPipe runtime (production contract: SHA-verified task bundle).
    let mut runtime = MediaPipeRuntime::from_task_path(&options.task)
        .map_err(|error| format!("mediapipe runtime: {error}"))?;
    let task_sha256 = sha256_hex(&options.task)?;
    let fit_config = SingleFrameFitConfig::new(
        vtuber_gnm::DenseRigidStepConfig::default(),
        vtuber_gnm::DenseExpressionJointStepConfig::default(),
        options.fit_max_iterations,
        options.fit_tolerance,
    )
    .map_err(|error| format!("fit config: {error}"))?;
    let gnm = build_gnm_context(&options.gnm_model, fit_config)?;

    // The replay is paced to the capture cadence: each frame's capture
    // timestamp sits on the session timeline (micros since session start,
    // rebased 1:1 onto the process monotonic clock) and the loop blocks until
    // the real clock reaches that offset before calling inference. MediaPipe
    // therefore sees the exact capture-time cadence, and the
    // captured <= started sample contract holds by construction.
    // (Rebasing onto a later epoch makes MediaPipe packet timestamps run
    // ahead of its internal wall-clock references and breaks detection.)

    // Deterministic replay over the exact-paired frames.
    let mut samples: Vec<PairedTemporalSample> = Vec::with_capacity(paired_sets.paired.len());
    let mut first_face: Option<u64> = None;
    let mut first_no_face: Option<u64> = None;
    let mut counts = ReplayCountsMetadata {
        paired: paired_sets.paired.len(),
        solved: 0,
        no_face: 0,
        observation_insufficient: 0,
        fit_rejected: 0,
        excluded_unpaired_teacher: paired_sets.teacher_only.len(),
        excluded_unpaired_rgb: paired_sets.rgb_only.len(),
    };
    for (teacher_index, rgb_index) in &paired_sets.paired {
        // Bounds: both indices come from classify_frames over these slices.
        #[allow(clippy::indexing_slicing)]
        let (teacher_record, rgb_record) = (&teacher[*teacher_index], &rgb[*rgb_index]);
        let frame_seq = teacher_record.frame_seq;
        let timestamp_micros = teacher_record.timestamp_micros;
        let captured_at_ns = timestamp_micros.saturating_mul(1_000);
        let frame = decode_video_frame(
            dataset,
            rgb_record,
            captured_at_ns,
            options.pixel_rotation_degrees,
        )?;
        wait_until_capture_time(MonoTimeNs(captured_at_ns));
        let outcome = runtime
            .infer_face_tracking(&frame)
            .map_err(|error| format!("frame {frame_seq}: mediapipe: {error}"))?;
        let (mediapipe_observation, gnm_state) = match outcome {
            FaceTrackingOutcome::Face(sample) => {
                if first_face.is_none() {
                    first_face = Some(frame_seq);
                }
                let observation = mediapipe_to_arkit52(&sample.blendshapes)
                    .map_err(|error| format!("frame {frame_seq}: {error}"))?;
                match fit_frame_gnm(&gnm, frame_seq, timestamp_micros, &sample)? {
                    GnmFrameOutcome::Solved(state) => {
                        counts.solved += 1;
                        (Some(observation), Some(state))
                    }
                    GnmFrameOutcome::InsufficientCoverage => {
                        counts.observation_insufficient += 1;
                        (Some(observation), None)
                    }
                    GnmFrameOutcome::FitRejected(reason) => {
                        counts.fit_rejected += 1;
                        eprintln!("warning: {reason}");
                        (Some(observation), None)
                    }
                }
            }
            FaceTrackingOutcome::NoFace { .. } => {
                if first_no_face.is_none() {
                    first_no_face = Some(frame_seq);
                }
                counts.no_face += 1;
                (None, None)
            }
        };
        samples.push(PairedTemporalSample {
            frame_seq,
            timestamp_micros,
            baseline_output: mediapipe_observation.unwrap_or_default(),
            mediapipe_observation,
            gnm_state,
            teacher: Some(
                teacher_frame(teacher_record)
                    .map_err(|error| format!("frame {frame_seq}: {error}"))?,
            ),
            rgb_reference: Some(RgbFrameReference {
                reference_path: rgb_record.reference_path.clone(),
                width_px: rgb_record.width_px,
                height_px: rgb_record.height_px,
                pixel_format: rgb_record.pixel_format.clone(),
                orientation_degrees: rgb_record.orientation_degrees,
                mirrored: rgb_record.mirrored,
            }),
        });
    }
    validate_paired_samples(&samples)
        .map_err(|error| format!("derived trace invalid: {error:?}"))?;
    if counts.solved == 0 {
        return Err(format!(
            "no frame produced a deterministic GNM state (solved {}, no_face {}, insufficient {}, fit_rejected {}; first face at seq {first_face:?}, first no-face at seq {first_no_face:?})",
            counts.solved, counts.no_face, counts.observation_insufficient, counts.fit_rejected
        ));
    }

    let take_id = dataset
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or("dataset path has no directory name")?;
    write_trace(
        &options.output,
        &take_id,
        &session.session_id,
        &samples,
        &counts,
        &input_hashes,
        &task_sha256,
        &gnm,
        options.pixel_rotation_degrees,
    )?;
    println!(
        "teacher-replay: {} frames -> {} solved, {} no-face, {} insufficient, {} fit-rejected; excluded unpaired (teacher {}, rgb {})",
        counts.paired,
        counts.solved,
        counts.no_face,
        counts.observation_insufficient,
        counts.fit_rejected,
        counts.excluded_unpaired_teacher,
        counts.excluded_unpaired_rgb
    );
    println!(
        "trace: {}",
        options.output.join("derived-trace.jsonl").display()
    );
    println!(
        "metadata: {}",
        options.output.join("replay-metadata.json").display()
    );
    Ok(())
}

fn teacher_frame(record: &TeacherRecord) -> Result<ArkitTeacherFrame, String> {
    let payload = &record.payload;
    let mut values = [0.0_f32; 52];
    for (slot, value) in values.iter_mut().zip(payload.coefficients_canonical.iter()) {
        *slot = *value;
    }
    let coefficients = Arkit52Coefficients::try_from_array(values)
        .map_err(|error| format!("teacher coefficients: {error:?}"))?;
    Ok(ArkitTeacherFrame {
        frame_seq: record.frame_seq,
        timestamp_micros: record.timestamp_micros,
        coefficients,
        head_transform: HeadTransform {
            rotation_unit_quaternion_wxyz: payload.rotation_quaternion_wxyz,
            translation_meters: payload.translation_meters,
        },
    })
}

fn wait_until_capture_time(target: MonoTimeNs) {
    while vtuber_core::monotonic_now().0 < target.0 {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn decode_video_frame(
    dataset: &Path,
    rgb_record: &RgbRecord,
    captured_at_ns: u64,
    pixel_rotation_degrees: u16,
) -> Result<VideoFrame, String> {
    let payload_path = dataset.join(&rgb_record.reference_path);
    let reader = ImageReader::open(&payload_path)
        .map_err(|error| format!("open {}: {error}", payload_path.display()))?;
    let mut decoded = reader
        .with_guessed_format()
        .map_err(|error| format!("probe {}: {error}", payload_path.display()))?
        .decode()
        .map_err(|error| format!("decode {}: {error}", payload_path.display()))?;
    // Explicit capture-orientation correction (recorded in the replay
    // metadata): some capture apps store the raw sensor pixels with a
    // mis-declared orientation token. The stored dataset bytes are never
    // modified.
    decoded = match pixel_rotation_degrees {
        0 => decoded,
        90 => decoded.rotate90(),
        180 => decoded.rotate180(),
        270 => decoded.rotate270(),
        other => return Err(format!("invalid pixel rotation {other}")),
    };
    let rgb = decoded.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    if width != rgb_record.width_px || height != rgb_record.height_px {
        return Err(format!(
            "frame {}: decoded {}x{} disagrees with the record {}x{}",
            rgb_record.frame_seq, width, height, rgb_record.width_px, rgb_record.height_px
        ));
    }
    Ok(VideoFrame {
        seq: FrameSeq(rgb_record.frame_seq),
        // Capture timeline rebased onto the process monotonic epoch.
        captured_at: MonoTimeNs(captured_at_ns),
        width,
        height,
        stride_bytes: width as usize * 3,
        format: PixelFormat::Rgb8,
        data: Arc::from(rgb.into_raw()),
    })
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect()
}

// Metadata assembly groups every reproducibility input; the count is bounded
// by the fields of the metadata document itself.
#[allow(clippy::too_many_arguments)]
fn write_trace(
    output: &Path,
    take_id: &str,
    session_id: &str,
    samples: &[PairedTemporalSample],
    counts: &ReplayCountsMetadata,
    input_hashes: &BTreeMap<String, String>,
    task_sha256: &str,
    gnm: &GnmContext,
    pixel_rotation_degrees: u16,
) -> Result<(), String> {
    fs::create_dir_all(output).map_err(|error| format!("create {}: {error}", output.display()))?;

    let mut trace_text = String::new();
    for sample in samples {
        let row = trace_row(sample);
        let line = serde_json::to_string(&row)
            .map_err(|error| format!("encode trace row {}: {error}", sample.frame_seq))?;
        trace_text.push_str(&line);
        trace_text.push('\n');
    }
    let trace_sha256 = sha256_hex_bytes(trace_text.as_bytes());
    let trace_path = output.join("derived-trace.jsonl");
    fs::write(&trace_path, trace_text.as_bytes())
        .map_err(|error| format!("write {}: {error}", trace_path.display()))?;

    let metadata = ReplayMetadata {
        schema_version: 1,
        tool: "xtask teacher-replay",
        xtask_version: env!("CARGO_PKG_VERSION").to_owned(),
        source_dataset: SourceDatasetMetadata {
            session_id: session_id.to_owned(),
            take_id: take_id.to_owned(),
            frame_count: samples.len(),
            input_hashes: input_hashes.clone(),
        },
        config: ReplayConfigMetadata {
            task_bundle_sha256: task_sha256.to_owned(),
            gnm_model_sha256: gnm.model_sha256.clone(),
            dense_mapping_schema_revision: gnm.mapping.version().schema_revision,
            fit_max_iterations: gnm.fit_config.max_iterations,
            fit_tolerance: gnm.fit_config.tolerance,
            coverage_min_valid_points: gnm.coverage.min_valid_points(),
            coverage_degraded_below_fraction: gnm.coverage.degraded_below_fraction(),
            identity: "fixed-model-neutral",
            auxiliary: "none",
            tongue_out_policy: "fixed-zero (MediaPipe has no tongue channel)",
            baseline_output_policy: "equals mediapipe_observation (no temporal prior in baseline)",
            paced_to_capture_cadence: true,
            pixel_rotation_degrees,
        },
        counts: ReplayCountsMetadata {
            paired: counts.paired,
            solved: counts.solved,
            no_face: counts.no_face,
            observation_insufficient: counts.observation_insufficient,
            fit_rejected: counts.fit_rejected,
            excluded_unpaired_teacher: counts.excluded_unpaired_teacher,
            excluded_unpaired_rgb: counts.excluded_unpaired_rgb,
        },
        trace_sha256,
    };
    let metadata_path = output.join("replay-metadata.json");
    let json = serde_json::to_string_pretty(&metadata)
        .map_err(|error| format!("encode metadata: {error}"))?;
    fs::write(&metadata_path, json.as_bytes())
        .map_err(|error| format!("write {}: {error}", metadata_path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::face_tracking::FaceBlendshapeSet;

    fn blendshape_set(value_at: impl Fn(usize) -> f32) -> FaceBlendshapeSet {
        let pairs: Vec<(&str, f32)> = MediaPipeBlendshape::ALL
            .iter()
            .map(|category| (category.as_str(), value_at(category.index())))
            .collect();
        FaceBlendshapeSet::from_pairs(&pairs).expect("valid blendshape set")
    }

    #[test]
    fn mediapipe_to_arkit52_maps_known_channels_and_pins_tongue_to_zero() {
        let set = blendshape_set(|index| 0.01 + index as f32 * 0.01);
        let coefficients = mediapipe_to_arkit52(&set).expect("maps");
        let jaw = ArkitBlendshape::JawOpen;
        let expected = 0.01 + MediaPipeBlendshape::JawOpen.index() as f32 * 0.01;
        assert!((coefficients.as_array()[jaw.index()] - expected).abs() < 1e-6);
        assert_eq!(
            coefficients.as_array()[ArkitBlendshape::TongueOut.index()],
            0.0
        );
    }

    fn teacher(seq: u64) -> TeacherRecord {
        TeacherRecord {
            frame_seq: seq,
            timestamp_micros: seq * 33_000,
            kind: TEACHER_KIND.to_owned(),
            payload: TeacherPayload {
                coefficients_canonical: vec![0.0; 52],
                rotation_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
                translation_meters: [0.0; 3],
                head_transform_convention: HEAD_TRANSFORM_CONVENTION.to_owned(),
            },
        }
    }

    fn rgb(seq: u64) -> RgbRecord {
        RgbRecord {
            frame_seq: seq,
            timestamp_micros: seq * 33_000,
            reference_path: format!("frames/frame_{seq:010}.bin"),
            width_px: 480,
            height_px: 640,
            pixel_format: "jpeg-rgb8-srgb-upright-nonmirrored".to_owned(),
            orientation_degrees: 0,
            mirrored: false,
        }
    }

    #[test]
    fn classify_frames_accepts_exact_identity() {
        let teachers = vec![teacher(0), teacher(1)];
        let rgbs = vec![rgb(0), rgb(1)];
        let sets = classify_frames(&teachers, &rgbs).expect("classifies");
        assert_eq!(sets.paired.len(), 2);
        assert!(
            sets.teacher_only.is_empty() && sets.rgb_only.is_empty() && sets.dropped.is_empty()
        );
    }

    #[test]
    fn classify_frames_rejects_duplicates_regressions_and_mismatches() {
        let teachers = vec![teacher(0), teacher(0)];
        let rgbs = vec![rgb(0), rgb(0)];
        assert!(classify_frames(&teachers, &rgbs).is_err());

        let teachers = vec![teacher(1), teacher(0)];
        let rgbs = vec![rgb(1), rgb(0)];
        assert!(classify_frames(&teachers, &rgbs).is_err());

        let teachers = vec![teacher(0)];
        let mut shifted = rgb(0);
        shifted.timestamp_micros += 1;
        assert!(classify_frames(&teachers, &[shifted]).is_err());
    }

    #[test]
    fn classify_frames_reports_unpaired_instead_of_repairing() {
        // RGB has an extra trailing sequence the teacher never saw.
        let teachers = vec![teacher(0), teacher(1)];
        let rgbs = vec![rgb(0), rgb(1), rgb(2)];
        let sets = classify_frames(&teachers, &rgbs).expect("classifies");
        assert_eq!(sets.paired.len(), 2);
        assert!(sets.teacher_only.is_empty());
        assert_eq!(sets.rgb_only, vec![2]);

        // A sequence missing on both sides inside the range is "dropped".
        let teachers = vec![teacher(0), teacher(2)];
        let rgbs = vec![rgb(0), rgb(2)];
        let sets = classify_frames(&teachers, &rgbs).expect("classifies");
        assert_eq!(sets.paired.len(), 2);
        assert_eq!(sets.dropped, vec![1]);
    }

    #[test]
    fn trace_rows_round_trip_through_json() {
        let mut coefficients = [0.0_f32; 52];
        coefficients[ArkitBlendshape::JawOpen.index()] = 0.4;
        let sample = PairedTemporalSample {
            frame_seq: 7,
            timestamp_micros: 231_000,
            mediapipe_observation: Some(
                Arkit52Coefficients::try_from_array(coefficients).expect("valid"),
            ),
            gnm_state: Some(DeterministicGnmState {
                projected_coefficients: Arkit52Coefficients::try_from_array(coefficients)
                    .expect("valid"),
                residual: 0.125,
            }),
            baseline_output: Arkit52Coefficients::default(),
            teacher: Some(ArkitTeacherFrame {
                frame_seq: 7,
                timestamp_micros: 231_000,
                coefficients: Arkit52Coefficients::try_from_array(coefficients).expect("valid"),
                head_transform: HeadTransform {
                    rotation_unit_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
                    translation_meters: [0.0, 0.0, 0.5],
                },
            }),
            rgb_reference: Some(RgbFrameReference {
                reference_path: "frames/frame_0000000007.bin".to_owned(),
                width_px: 480,
                height_px: 640,
                pixel_format: "jpeg-rgb8-srgb-upright-nonmirrored".to_owned(),
                orientation_degrees: 0,
                mirrored: false,
            }),
        };
        let row = trace_row(&sample);
        let json = serde_json::to_string(&row).expect("serialize");
        let parsed: TraceRow = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, row);
        let rebuilt = sample_from_row(&parsed).expect("rebuild");
        assert_eq!(rebuilt, sample);
    }
}
