//! Sequential full-versus-reduced GNM solver benchmark (Issue #20).

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use vtuber_gnm::{
    DenseCoveragePolicy, DenseReprojectionConfig, GNM_HEAD_V3_IRIS_EXPRESSION_INDEX,
    GNM_HEAD_V3_TONGUE_EXPRESSION_RANGE, GnmDenseObservation, GnmExpressionState, GnmJointState,
    GnmReducedExpressionState, GnmSparseVertices, SingleFrameFitConfig,
    evaluate_dense_reprojection, fit_single_frame_cold_start, fit_single_frame_reduced,
    fitting_projection, load_gnm_head_v3, repository_dense_mapping,
};
use vtuber_tracking::{
    TeacherAlignedGnmBasisArtifact, load_reduced_gnm_basis, seed_gnm_projection_rotation,
};

use crate::teacher_fit_prior::load_trace;
use crate::teacher_replay::sha256_hex;

struct Options {
    basis: PathBuf,
    trace: PathBuf,
    take: String,
    max_frames: usize,
    output: PathBuf,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut basis = None;
        let mut trace = None;
        let mut take = None;
        let mut max_frames = None;
        let mut output = None;
        let mut index = 0;
        while index < args.len() {
            let next = |index: &mut usize, flag: &str| -> Result<String, String> {
                *index += 1;
                args.get(*index)
                    .cloned()
                    .ok_or_else(|| format!("{flag} requires a value"))
            };
            #[allow(clippy::indexing_slicing)]
            match args[index].as_str() {
                "--basis" => basis = Some(PathBuf::from(next(&mut index, "--basis")?)),
                "--trace" => trace = Some(PathBuf::from(next(&mut index, "--trace")?)),
                "--take" => take = Some(next(&mut index, "--take")?),
                "--max-frames" => {
                    max_frames = Some(
                        next(&mut index, "--max-frames")?
                            .parse()
                            .map_err(|_| "--max-frames must be a positive integer")?,
                    );
                }
                "--output" => output = Some(PathBuf::from(next(&mut index, "--output")?)),
                other => return Err(format!("unknown option {other}")),
            }
            index += 1;
        }
        let max_frames = max_frames.ok_or("--max-frames <n> is required")?;
        if max_frames == 0 {
            return Err("--max-frames must be positive".to_owned());
        }
        Ok(Self {
            basis: basis.ok_or("--basis <gnm-basis.json> is required")?,
            trace: trace.ok_or("--trace <trace-v2-dir> is required")?,
            take: take.ok_or("--take <id> is required")?,
            max_frames,
            output: output.ok_or("--output <report.json> is required")?,
        })
    }
}

pub fn print_help() {
    println!(
        "  teacher-benchmark-reduced-gnm --basis <gnm-basis.json> --trace <trace-v2-dir>\n\
         *       --take <id> --max-frames <n> --output <report.json>"
    );
}

#[derive(Serialize)]
struct Distribution {
    p50: f64,
    p95: f64,
    max: f64,
}

impl Distribution {
    fn from_values(values: &[f64]) -> Result<Self, String> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err("benchmark distribution is empty or non-finite".to_owned());
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let percentile = |numerator: usize| {
            let index = ((sorted.len() - 1) * numerator + 50) / 100;
            // `sorted` is non-empty and `index <= len - 1` by construction.
            #[allow(clippy::indexing_slicing)]
            sorted[index]
        };
        Ok(Self {
            p50: percentile(50),
            p95: percentile(95),
            max: sorted.last().copied().unwrap_or_default(),
        })
    }
}

#[derive(Serialize)]
struct SolverMetrics {
    valid_frames: usize,
    rejected_frames: usize,
    wall_milliseconds: Distribution,
    iterations: Distribution,
    weighted_rms: Distribution,
    final_objective: Distribution,
}

#[derive(Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    take_id: String,
    trace_sha256: String,
    model_sha256: String,
    mapping_schema_revision: u32,
    basis_content_hash: u64,
    basis_rank: usize,
    requested_max_frames: usize,
    benchmarked_frames: usize,
    full: SolverMetrics,
    reduced: SolverMetrics,
    reconstruction_expression_difference: Distribution,
    tongue_max_abs: f64,
}

#[derive(Default)]
struct Samples {
    wall_ms: Vec<f64>,
    iterations: Vec<f64>,
    weighted_rms: Vec<f64>,
    objective: Vec<f64>,
    valid: usize,
}

impl Samples {
    fn metrics(self, frame_count: usize) -> Result<SolverMetrics, String> {
        Ok(SolverMetrics {
            valid_frames: self.valid,
            rejected_frames: frame_count - self.valid,
            wall_milliseconds: Distribution::from_values(&self.wall_ms)?,
            iterations: Distribution::from_values(&self.iterations)?,
            weighted_rms: Distribution::from_values(&self.weighted_rms)?,
            final_objective: Distribution::from_values(&self.objective)?,
        })
    }
}

fn non_tongue_difference(left: &GnmExpressionState, right: &GnmExpressionState) -> f64 {
    let paired = left
        .values()
        .iter()
        .zip(right.values())
        .enumerate()
        .filter(|(index, _)| {
            !GNM_HEAD_V3_TONGUE_EXPRESSION_RANGE.contains(index)
                || *index == GNM_HEAD_V3_IRIS_EXPRESSION_INDEX
        });
    let (squared, count) = paired.fold((0.0, 0_usize), |(sum, count), (_, (a, b))| {
        let difference = f64::from(*a) - f64::from(*b);
        (sum + difference * difference, count + 1)
    });
    (squared / count as f64).sqrt()
}

fn tongue_max(expression: &GnmExpressionState) -> f64 {
    expression
        .values()
        .iter()
        .enumerate()
        .filter(|(index, _)| GNM_HEAD_V3_TONGUE_EXPRESSION_RANGE.contains(index))
        .map(|(_, value)| f64::from(value.abs()))
        .fold(0.0, f64::max)
}

/// Runs the sequential benchmark and writes its JSON report.
pub fn run(args: &[String]) -> Result<(), String> {
    if args.iter().any(|argument| argument == "--help") {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let trace = load_trace(&options.trace)?;
    if trace.take_id != options.take {
        return Err(format!(
            "--take {} does not match trace take {}",
            options.take, trace.take_id
        ));
    }
    let model_path = PathBuf::from("assets/models/gnm_head.npz");
    let model_sha256 = sha256_hex(&model_path)?;
    if model_sha256 != trace.model_sha256 {
        return Err("trace GNM model SHA-256 does not match repository model".to_owned());
    }
    let basis_text = fs::read_to_string(&options.basis)
        .map_err(|error| format!("read {}: {error}", options.basis.display()))?;
    let artifact: TeacherAlignedGnmBasisArtifact = serde_json::from_str(&basis_text)
        .map_err(|error| format!("parse {}: {error}", options.basis.display()))?;
    let reduced_basis =
        load_reduced_gnm_basis(&artifact, &model_sha256, trace.mapping_schema_revision)
            .map_err(|error| format!("load reduced basis: {error}"))?;
    let model = load_gnm_head_v3(&model_path).map_err(|error| error.to_string())?;
    let mapping = repository_dense_mapping()
        .bind(&model)
        .map_err(|error| error.to_string())?;
    let identity = model.neutral_identity();
    let neutral_expression = model.neutral_expression();
    let neutral_joints = GnmJointState::neutral(model.joint_count());
    let mut surface = GnmSparseVertices::with_len(mapping.len());
    mapping
        .evaluate_surface(
            &model,
            &identity,
            &neutral_expression,
            &neutral_joints,
            &mut surface,
        )
        .map_err(|error| error.to_string())?;
    let base_projection =
        fitting_projection(surface.values(), [0.0; 3]).map_err(|error| error.to_string())?;
    let coverage = DenseCoveragePolicy::new(2, 0.75).map_err(|error| error.to_string())?;
    let defaults = SingleFrameFitConfig::default();
    let config = SingleFrameFitConfig::new(defaults.rigid, defaults.expression_joint, 64, 1.0e-4)
        .map_err(|error| error.to_string())?;

    let mut full_expression = neutral_expression;
    let mut full_joints = neutral_joints.clone();
    let mut reduced_state = GnmReducedExpressionState::neutral(reduced_basis.rank());
    let mut reduced_joints = neutral_joints;
    let mut full_samples = Samples::default();
    let mut reduced_samples = Samples::default();
    let mut differences = Vec::new();
    let mut tongue_max_abs = 0.0_f64;

    for sample in trace.samples.iter().take(options.max_frames) {
        let observation = sample
            .mediapipe_observation
            .as_ref()
            .ok_or_else(|| format!("frame {} has no MediaPipe observation", sample.frame_seq))?;
        let dense = GnmDenseObservation::from_mediapipe_xy(
            sample.frame_seq,
            sample.timestamp_micros,
            &observation.landmarks_xy,
            &mapping,
            coverage,
        )
        .map_err(|error| format!("frame {} observation: {error}", sample.frame_seq))?;
        let projection =
            seed_gnm_projection_rotation(&observation.camera_to_face, &base_projection)
                .map_err(|error| format!("frame {} pose seed: {error}", sample.frame_seq))?;

        let started = Instant::now();
        let full = fit_single_frame_cold_start(
            &model,
            &identity,
            &full_expression,
            &full_joints,
            &mapping,
            &dense,
            &projection,
            config,
            None,
        )
        .map_err(|error| format!("frame {} full fit: {error}", sample.frame_seq))?;
        full_samples
            .wall_ms
            .push(started.elapsed().as_secs_f64() * 1_000.0);
        let full_report = evaluate_dense_reprojection(
            &model,
            &identity,
            full.expression(),
            full.joints(),
            &mapping,
            &dense,
            full.projection(),
            DenseReprojectionConfig::default(),
        )
        .map_err(|error| format!("frame {} full report: {error}", sample.frame_seq))?;
        full_samples.iterations.push(full.iterations() as f64);
        full_samples
            .weighted_rms
            .push(f64::from(full_report.weighted_rms()));
        full_samples.objective.push(f64::from(full.objective()));
        if full.valid() {
            full_samples.valid += 1;
            full_expression = full.expression().clone();
            full_joints = full.joints().clone();
        }

        let started = Instant::now();
        let reduced = fit_single_frame_reduced(
            &model,
            &identity,
            &reduced_basis,
            &reduced_state,
            &reduced_joints,
            &mapping,
            &dense,
            &projection,
            config,
            None,
        )
        .map_err(|error| format!("frame {} reduced fit: {error}", sample.frame_seq))?;
        reduced_samples
            .wall_ms
            .push(started.elapsed().as_secs_f64() * 1_000.0);
        reduced_samples.iterations.push(reduced.iterations() as f64);
        reduced_samples
            .weighted_rms
            .push(f64::from(reduced.final_report().weighted_rms()));
        reduced_samples
            .objective
            .push(f64::from(reduced.objective()));
        if reduced.valid() {
            reduced_samples.valid += 1;
            reduced_state = reduced.reduced_expression().clone();
            reduced_joints = reduced.joints().clone();
        }
        differences.push(non_tongue_difference(
            full.expression(),
            reduced.expression(),
        ));
        tongue_max_abs = tongue_max_abs
            .max(tongue_max(full.expression()))
            .max(tongue_max(reduced.expression()));
    }
    let frame_count = differences.len();
    if frame_count == 0 {
        return Err("trace contains no benchmark frames".to_owned());
    }
    let report = BenchmarkReport {
        schema_version: 1,
        take_id: trace.take_id,
        trace_sha256: trace.trace_sha256,
        model_sha256,
        mapping_schema_revision: mapping.version().schema_revision,
        basis_content_hash: artifact.content_hash,
        basis_rank: reduced_basis.rank(),
        requested_max_frames: options.max_frames,
        benchmarked_frames: frame_count,
        full: full_samples.metrics(frame_count)?,
        reduced: reduced_samples.metrics(frame_count)?,
        reconstruction_expression_difference: Distribution::from_values(&differences)?,
        tongue_max_abs,
    };
    if let Some(parent) = options.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize benchmark: {error}"))?;
    fs::write(&options.output, format!("{json}\n"))
        .map_err(|error| format!("write {}: {error}", options.output.display()))?;
    println!(
        "benchmarked {frame_count} frame(s) at rank {} -> {}",
        reduced_basis.rank(),
        options.output.display()
    );
    Ok(())
}
