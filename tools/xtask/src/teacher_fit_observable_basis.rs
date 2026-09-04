//! Fits a geometric observable GNM expression basis from trace-v2 takes.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use vtuber_gnm::{
    DenseCoveragePolicy, DenseProjection, FixedGnmIdentity, GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM,
    GnmDenseObservation, GnmJointState, accumulate_observability_gram, load_gnm_head_v3,
    non_tongue_projection_jacobian, repository_dense_mapping,
};
use vtuber_tracking::{
    ObservableBasisProvenance, fit_observable_gnm_basis, project_non_tongue_expression,
};

use crate::teacher_fit_prior::load_trace;
use crate::teacher_replay::sha256_hex;

pub struct Options {
    traces: Vec<PathBuf>,
    train_takes: BTreeSet<String>,
    rank: usize,
    output: PathBuf,
    gnm_model: PathBuf,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut traces = Vec::new();
        let mut train_takes = BTreeSet::new();
        let mut rank = None;
        let mut output = None;
        let mut gnm_model = None;
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
                "--trace" => traces.push(PathBuf::from(next(&mut index, "--trace")?)),
                "--train-take" => {
                    train_takes.insert(next(&mut index, "--train-take")?);
                }
                "--rank" => {
                    rank = Some(
                        next(&mut index, "--rank")?
                            .parse()
                            .map_err(|_| "--rank must be an integer")?,
                    );
                }
                "--output" => output = Some(PathBuf::from(next(&mut index, "--output")?)),
                "--gnm-model" => {
                    gnm_model = Some(PathBuf::from(next(&mut index, "--gnm-model")?));
                }
                other => return Err(format!("unknown option {other}")),
            }
            index += 1;
        }
        if traces.is_empty() || train_takes.is_empty() {
            return Err("--trace and --train-take are both required".to_owned());
        }
        Ok(Self {
            traces,
            train_takes,
            rank: rank.ok_or("--rank <n> is required")?,
            output: output.ok_or("--output <artifact.json> is required")?,
            gnm_model: gnm_model.unwrap_or_else(|| PathBuf::from("assets/models/gnm_head.npz")),
        })
    }
}

pub fn print_help() {
    println!(
        "  teacher-fit-observable-basis --trace <trace-v2-dir> [...]\n\
         *       --train-take <take-id> [...] --rank <n> --output <artifact.json>\n\
         *       [--gnm-model <gnm_head.npz>]\n\
         *   Sequentially accumulates geometric J^T W J and exports its top basis."
    );
}

/// Runs the trace-v2 observable-basis fit.
///
/// # Errors
///
/// Returns a descriptive failure for invalid options, trace/model mismatch,
/// numeric fitting failure, or output I/O.
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let model = load_gnm_head_v3(&options.gnm_model)
        .map_err(|error| format!("load {}: {error}", options.gnm_model.display()))?;
    let mapping = repository_dense_mapping()
        .bind(&model)
        .map_err(|error| format!("bind dense mapping: {error}"))?;
    let identity = FixedGnmIdentity::new(model.neutral_identity(), &model)
        .map_err(|error| error.to_string())?;
    let model_sha256 = sha256_hex(&options.gnm_model)?;
    let coverage = DenseCoveragePolicy::new(2, 0.75).map_err(|error| error.to_string())?;
    let triangle_len =
        GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM * (GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM + 1) / 2;
    let mut gram = vec![0.0_f64; triangle_len];
    let mut traces = options
        .traces
        .iter()
        .map(|directory| load_trace(directory))
        .collect::<Result<Vec<_>, _>>()?;
    traces.sort_by(|left, right| left.take_id.cmp(&right.take_id));
    let available: BTreeSet<&str> = traces.iter().map(|trace| trace.take_id.as_str()).collect();
    if !options
        .train_takes
        .iter()
        .all(|take| available.contains(take.as_str()))
    {
        return Err("--train-take names a take not supplied by --trace".to_owned());
    }
    for trace in traces
        .iter()
        .filter(|trace| options.train_takes.contains(&trace.take_id))
    {
        if trace.model_sha256 != model_sha256
            || trace.mapping_schema_revision != mapping.version().schema_revision
        {
            return Err(format!(
                "take {} was generated with a different GNM model or mapping",
                trace.take_id
            ));
        }
    }
    let mut frame_count = 0_usize;
    for trace in traces
        .iter()
        .filter(|trace| options.train_takes.contains(&trace.take_id))
    {
        for sample in &trace.samples {
            let (Some(observation), Some(state)) =
                (&sample.mediapipe_observation, &sample.gnm_state)
            else {
                continue;
            };
            let dense = GnmDenseObservation::from_mediapipe_xy(
                sample.frame_seq,
                sample.timestamp_micros,
                &observation.landmarks_xy,
                &mapping,
                coverage,
            )
            .map_err(|error| {
                format!("take {} frame {}: {error}", trace.take_id, sample.frame_seq)
            })?;
            let joints =
                GnmJointState::new(state.joint_rotations.clone(), [0.0; 3], model.joint_count())
                    .map_err(|error| {
                        format!("take {} frame {}: {error}", trace.take_id, sample.frame_seq)
                    })?;
            let projection = DenseProjection::new(
                state.rigid_yaw_pitch_roll,
                state.camera_translation,
                state.camera_focal,
                state.camera_principal_point,
            )
            .map_err(|error| {
                format!("take {} frame {}: {error}", trace.take_id, sample.frame_seq)
            })?;
            let jacobian = non_tongue_projection_jacobian(
                &model,
                identity.state(),
                &state.expression,
                &joints,
                &mapping,
                &dense,
                &projection,
            )
            .map_err(|error| {
                format!("take {} frame {}: {error}", trace.take_id, sample.frame_seq)
            })?;
            accumulate_observability_gram(&jacobian, &mut gram).map_err(|error| {
                format!("take {} frame {}: {error}", trace.take_id, sample.frame_seq)
            })?;
            frame_count += 1;
        }
    }
    let artifact = fit_observable_gnm_basis(
        &gram,
        frame_count,
        options.rank,
        ObservableBasisProvenance {
            model_sha256,
            mapping_schema_revision: mapping.version().schema_revision,
            training_takes: options.train_takes.into_iter().collect(),
        },
    )
    .map_err(|error| error.to_string())?;
    let neutral = vtuber_gnm::GnmNonTongueExpression::try_from_values(vec![
        0.0;
        GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM
    ])
    .map_err(|error| error.to_string())?;
    let _ = project_non_tongue_expression(&neutral, &artifact)
        .map_err(|error| format!("verify artifact: {error}"))?;
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&artifact)
        .map_err(|error| format!("encode artifact: {error}"))?;
    fs::write(&options.output, json.as_bytes())
        .map_err(|error| format!("write {}: {error}", options.output.display()))?;
    println!("teacher-fit-observable-basis:");
    println!("  training takes: {:?}", artifact.training_takes);
    println!("  frames: {frame_count}");
    println!("  rank: {}", artifact.rank);
    println!(
        "  retained energy: {:.9}",
        artifact.retained_energy_fraction
    );
    let top_eigenvalues: Vec<f64> = artifact
        .eigenvalues_descending
        .iter()
        .take(8)
        .copied()
        .collect();
    println!("  top eigenvalues: {top_eigenvalues:?}");
    println!("  content_hash: {}", artifact.content_hash);
    println!("  artifact: {}", options.output.display());
    Ok(())
}
