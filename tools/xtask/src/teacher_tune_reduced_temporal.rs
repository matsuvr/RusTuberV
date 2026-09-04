//! Training-only fixed-grid reduced temporal tuning and causal 60 Hz replay.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use vtuber_core::Arkit52Coefficients;
use vtuber_gnm::{
    DenseCoveragePolicy, DenseReprojectionReport, FaceRegion, GNM_HEAD_V3_EXPRESSION_DIM,
    GnmDenseObservation, GnmJointState, GnmReducedExpressionBasis, GnmReducedExpressionState,
    GnmRegionFitRecord, GnmSparseVertices, GnmTemporalNormalization, ReducedTemporalHistory,
    SingleFrameFitConfig, TemporalGroupPenaltyWeights, TemporalHistoryTiming,
    TemporalRegularizationConfig, fit_single_frame_reduced_with_temporal, fitting_projection,
    load_gnm_head_v3, repository_dense_mapping,
};
use vtuber_tracking::{
    AlphaBetaGain, GnmSemanticDecoderArtifact, GnmSemanticDecoderKind, GnmSemanticFrame,
    ReducedTemporalProvenance, SourceGroupGains, TeacherAlignedGnmBasisArtifact,
    TemporalCandidateMetrics, TimestampedDirectCoefficients, VariantFrame,
    apply_non_tongue_residual, build_gnm_semantic_features, correct_reduced_temporal_state,
    evaluate_blink_events, evaluate_non_tongue_variant, initialize_reduced_temporal_state,
    load_reduced_gnm_basis, predict_gnm_semantic_raw, project_source_group_gains,
    reduced_temporal_gain_grid, sample_direct_coefficients_at, sample_reduced_state_at,
    seed_gnm_projection_rotation, select_reduced_temporal_artifact,
};

use crate::teacher_fit_prior::{LoadedTrace, load_trace};
use crate::teacher_replay::sha256_hex;

const RENDER_INTERVAL_NUMERATOR: u64 = 1_000_000;
const RENDER_RATE: u64 = 60;

struct Options {
    basis: PathBuf,
    decoder: PathBuf,
    traces: Vec<PathBuf>,
    train_takes: BTreeSet<String>,
    output: PathBuf,
    max_prediction_horizon_micros: u64,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut basis = None;
        let mut decoder = None;
        let mut traces = Vec::new();
        let mut train_takes = BTreeSet::new();
        let mut output = None;
        let mut horizon = None;
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
                "--hybrid-decoder" => {
                    decoder = Some(PathBuf::from(next(&mut index, "--hybrid-decoder")?));
                }
                "--trace" => traces.push(PathBuf::from(next(&mut index, "--trace")?)),
                "--train-take" => {
                    train_takes.insert(next(&mut index, "--train-take")?);
                }
                "--output" => output = Some(PathBuf::from(next(&mut index, "--output")?)),
                "--max-prediction-horizon-micros" => {
                    horizon = Some(
                        next(&mut index, "--max-prediction-horizon-micros")?
                            .parse()
                            .map_err(
                                |_| "--max-prediction-horizon-micros must be a positive integer",
                            )?,
                    );
                }
                other => return Err(format!("unknown option {other}")),
            }
            index += 1;
        }
        let horizon = horizon.ok_or("--max-prediction-horizon-micros <n> is required")?;
        if traces.is_empty() || train_takes.len() < 2 || horizon == 0 {
            return Err(
                "at least two training takes, their traces, and a positive horizon are required"
                    .to_owned(),
            );
        }
        Ok(Self {
            basis: basis.ok_or("--basis <artifact.json> is required")?,
            decoder: decoder.ok_or("--hybrid-decoder <artifact.json> is required")?,
            traces,
            train_takes,
            output: output.ok_or("--output <directory> is required")?,
            max_prediction_horizon_micros: horizon,
        })
    }
}

pub fn print_help() {
    println!(
        "  teacher-tune-reduced-temporal --basis <basis.json> --hybrid-decoder <h.json>\n\
         *       --trace <trace-v2-dir> [...] --train-take <id> [...] --output <dir>\n\
         *       --max-prediction-horizon-micros <n>"
    );
}

#[derive(Clone)]
struct SourceFrame {
    take_id: String,
    frame_seq: u64,
    timestamp_micros: u64,
    reduced: GnmReducedExpressionState,
    joint_rotations: Vec<[f32; 3]>,
    rigid_yaw_pitch_roll: [f32; 3],
    objective: f32,
    region_fits: Vec<GnmRegionFitRecord>,
    direct: Arkit52Coefficients,
    teacher: Arkit52Coefficients,
}

#[derive(Serialize)]
struct MetricSummary {
    frame_count: usize,
    macro_mae: f64,
    macro_rmse: f64,
    velocity_mae: f64,
    acceleration_mae: f64,
    jitter: f64,
    peak_jerk_mae: f64,
    missed_blinks: usize,
    onset_error_ms: Option<f64>,
    peak_error_ms: Option<f64>,
    peak_attenuation: Option<f64>,
}

#[derive(Serialize)]
struct Distribution {
    p50: f64,
    p95: f64,
    max: f64,
}

#[derive(Serialize)]
struct QKinematics {
    velocity: Distribution,
    acceleration: Distribution,
    jerk: Distribution,
}

#[derive(Serialize)]
struct TuningReport {
    schema_version: u32,
    basis_content_hash: u64,
    decoder_content_hash: u64,
    temporal_artifact_content_hash: u64,
    training_takes: Vec<String>,
    leave_one_take_out_folds: usize,
    source_observations: usize,
    source_observation_rate_hz: f64,
    render_samples: usize,
    render_sample_rate_hz: f64,
    temporal_history_resets: usize,
    prediction_horizon_micros: Distribution,
    candidates: Vec<TemporalCandidateMetrics>,
    selected_eye_preset: AlphaBetaGain,
    selected_lower_face_preset: AlphaBetaGain,
    source_direct: MetricSummary,
    source_hybrid: MetricSummary,
    render_hybrid: MetricSummary,
    q_kinematics: QKinematics,
}

struct CandidateReplay {
    direct: Vec<VariantFrame>,
    hybrid: Vec<VariantFrame>,
    q_samples: Vec<(u64, Vec<f32>)>,
    horizons: Vec<f64>,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn temporal_config() -> Result<TemporalRegularizationConfig, String> {
    let expression =
        TemporalGroupPenaltyWeights::new(5.05e-5, 5.05e-7).map_err(|error| error.to_string())?;
    let zero = TemporalGroupPenaltyWeights::new(0.0, 0.0).map_err(|error| error.to_string())?;
    TemporalRegularizationConfig::new(expression, zero, zero, zero, 0.1)
        .map_err(|error| error.to_string())
}

fn region_records(report: &DenseReprojectionReport) -> Vec<GnmRegionFitRecord> {
    [
        FaceRegion::Contour,
        FaceRegion::Brow,
        FaceRegion::Eye,
        FaceRegion::Nose,
        FaceRegion::Mouth,
        FaceRegion::Iris,
        FaceRegion::Other,
    ]
    .into_iter()
    .map(|region| {
        let (valid_points, weighted_sum, weight_sum) = report
            .residuals()
            .iter()
            .filter(|residual| residual.region == region)
            .fold(
                (0_usize, 0.0_f64, 0.0_f64),
                |(count, sum, weights), residual| {
                    let squared = residual.residual_xy[0] * residual.residual_xy[0]
                        + residual.residual_xy[1] * residual.residual_xy[1];
                    (
                        count + 1,
                        sum + f64::from(residual.base_weight * squared),
                        weights + f64::from(residual.base_weight),
                    )
                },
            );
        GnmRegionFitRecord {
            region,
            valid_points,
            weighted_rms: if weight_sum > 0.0 {
                (weighted_sum / weight_sum).sqrt() as f32
            } else {
                0.0
            },
        }
    })
    .collect()
}

fn solve_sources(
    trace: &LoadedTrace,
    basis: &GnmReducedExpressionBasis,
) -> Result<(Vec<SourceFrame>, usize), String> {
    let model_path = PathBuf::from("assets/models/gnm_head.npz");
    let model = load_gnm_head_v3(&model_path).map_err(|error| error.to_string())?;
    let mapping = repository_dense_mapping()
        .bind(&model)
        .map_err(|error| error.to_string())?;
    let identity = model.neutral_identity();
    let neutral_joints = GnmJointState::neutral(model.joint_count());
    let mut surface = GnmSparseVertices::with_len(mapping.len());
    mapping
        .evaluate_surface(
            &model,
            &identity,
            &model.neutral_expression(),
            &neutral_joints,
            &mut surface,
        )
        .map_err(|error| error.to_string())?;
    let base_projection =
        fitting_projection(surface.values(), [0.0; 3]).map_err(|e| e.to_string())?;
    let coverage = DenseCoveragePolicy::new(2, 0.75).map_err(|error| error.to_string())?;
    let defaults = SingleFrameFitConfig::default();
    let solver_config =
        SingleFrameFitConfig::new(defaults.rigid, defaults.expression_joint, 64, 1.0e-4)
            .map_err(|error| error.to_string())?;
    let temporal_config = temporal_config()?;
    let expression_scales = vec![1.0; GNM_HEAD_V3_EXPRESSION_DIM];
    let normalization = GnmTemporalNormalization {
        expression: &expression_scales,
        joints: &[],
        head_pose: &[],
        translation: &[],
    };
    let mut previous: Option<(u64, GnmReducedExpressionState)> = None;
    let mut previous_previous: Option<(u64, GnmReducedExpressionState)> = None;
    let mut joints = neutral_joints;
    let mut output = Vec::new();
    let mut resets = 0;
    for sample in &trace.samples {
        let Some(observation) = sample.mediapipe_observation.as_ref() else {
            continue;
        };
        let teacher = sample.teacher.as_ref().ok_or_else(|| {
            format!(
                "take {} frame {} lacks teacher",
                trace.take_id, sample.frame_seq
            )
        })?;
        let dense = GnmDenseObservation::from_mediapipe_xy(
            sample.frame_seq,
            sample.timestamp_micros,
            &observation.landmarks_xy,
            &mapping,
            coverage,
        )
        .map_err(|error| error.to_string())?;
        let projection =
            seed_gnm_projection_rotation(&observation.camera_to_face, &base_projection)
                .map_err(|error| error.to_string())?;
        let reset = previous.as_ref().is_some_and(|(timestamp, _)| {
            sample.timestamp_micros.saturating_sub(*timestamp) > 100_000
        });
        if reset {
            previous_previous = None;
            previous = None;
            joints = GnmJointState::neutral(model.joint_count());
            resets += 1;
        }
        let history = previous.as_ref().map(|(timestamp, state)| {
            let dt_micros = sample.timestamp_micros.saturating_sub(*timestamp);
            ReducedTemporalHistory {
                previous: state,
                previous_previous: previous_previous.as_ref().map(|(_, state)| state),
                timing: TemporalHistoryTiming {
                    dt_seconds: dt_micros as f64 / 1_000_000.0,
                    previous_dt_seconds: previous_previous
                        .as_ref()
                        .map(|(older, _)| timestamp.saturating_sub(*older) as f64 / 1_000_000.0),
                },
                normalization,
            }
        });
        let initial = previous.as_ref().map_or_else(
            || GnmReducedExpressionState::neutral(basis.rank()),
            |(_, state)| state.clone(),
        );
        let fit = fit_single_frame_reduced_with_temporal(
            &model,
            &identity,
            basis,
            &initial,
            &joints,
            &mapping,
            &dense,
            &projection,
            history,
            solver_config,
            temporal_config,
            None,
        )
        .map_err(|error| format!("take {} frame {}: {error}", trace.take_id, sample.frame_seq))?;
        if !fit.valid() {
            return Err(format!(
                "take {} frame {} reduced temporal fit rejected: {:?}",
                trace.take_id,
                sample.frame_seq,
                fit.status()
            ));
        }
        joints = fit.joints().clone();
        previous_previous = previous.take();
        previous = Some((sample.timestamp_micros, fit.reduced_expression().clone()));
        output.push(SourceFrame {
            take_id: trace.take_id.clone(),
            frame_seq: sample.frame_seq,
            timestamp_micros: sample.timestamp_micros,
            reduced: fit.reduced_expression().clone(),
            joint_rotations: fit.joints().rotations().to_vec(),
            rigid_yaw_pitch_roll: fit.projection().yaw_pitch_roll(),
            objective: fit.objective(),
            region_fits: region_records(fit.final_report()),
            direct: observation.direct_coefficients,
            teacher: teacher.coefficients,
        });
    }
    Ok((output, resets))
}

fn available_source_end(sources: &[SourceFrame], target_micros: u64) -> usize {
    sources.partition_point(|source| source.timestamp_micros <= target_micros)
}

fn interpolate_teacher(
    current: &SourceFrame,
    next: &SourceFrame,
    target_micros: u64,
) -> Result<Arkit52Coefficients, String> {
    if current.timestamp_micros == next.timestamp_micros {
        return Ok(current.teacher);
    }
    let ratio = (target_micros - current.timestamp_micros) as f32
        / (next.timestamp_micros - current.timestamp_micros) as f32;
    let mut values = [0.0; 52];
    for ((output, current), next) in values
        .iter_mut()
        .zip(current.teacher.as_array())
        .zip(next.teacher.as_array())
    {
        *output = current + ratio * (next - current);
    }
    #[allow(clippy::indexing_slicing)] // fixed ARKit52 array; TongueOut is index 51
    {
        values[51] = 0.0;
    }
    Arkit52Coefficients::try_from_array(values).map_err(|error| error.to_string())
}

fn render_timestamps(sources: &[SourceFrame]) -> Vec<u64> {
    let Some(first) = sources.get(1).map(|source| source.timestamp_micros) else {
        return Vec::new();
    };
    let Some(last) = sources.last().map(|source| source.timestamp_micros) else {
        return Vec::new();
    };
    let mut timestamps = Vec::new();
    let mut tick = 0_u64;
    loop {
        let timestamp = first + (tick * RENDER_INTERVAL_NUMERATOR + RENDER_RATE / 2) / RENDER_RATE;
        if timestamp > last {
            break;
        }
        timestamps.push(timestamp);
        tick += 1;
    }
    timestamps
}

fn replay_candidate(
    sources: &[SourceFrame],
    basis: &GnmReducedExpressionBasis,
    decoder: &GnmSemanticDecoderArtifact,
    eye: AlphaBetaGain,
    lower: AlphaBetaGain,
    horizon: u64,
) -> Result<CandidateReplay, String> {
    let gains = project_source_group_gains(
        basis,
        SourceGroupGains {
            eye,
            lower_face: lower,
            iris: eye,
        },
    )
    .map_err(|error| error.to_string())?;
    let timestamps = render_timestamps(sources);
    let mut consumed = 0;
    let mut temporal = None;
    let mut semantic_history = Vec::new();
    let mut direct_frames = Vec::new();
    let mut hybrid_frames = Vec::new();
    let mut q_samples = Vec::new();
    let mut horizons = Vec::new();
    for (render_seq, timestamp) in timestamps.into_iter().enumerate() {
        let available = available_source_end(sources, timestamp);
        while consumed < available {
            let source = sources
                .get(consumed)
                .ok_or("source cursor exceeded trace")?;
            temporal = Some(match temporal {
                None => initialize_reduced_temporal_state(
                    source.timestamp_micros,
                    source.reduced.clone(),
                ),
                Some(ref state) => correct_reduced_temporal_state(
                    state,
                    source.timestamp_micros,
                    &source.reduced,
                    &gains,
                )
                .map_err(|error| error.to_string())?,
            });
            consumed += 1;
        }
        if consumed < 2 {
            continue;
        }
        let current = sources.get(consumed - 1).ok_or("missing current source")?;
        let previous = sources.get(consumed - 2).ok_or("missing previous source")?;
        let state = temporal.as_ref().ok_or("missing temporal state")?;
        let sampled = sample_reduced_state_at(state, timestamp, horizon)
            .map_err(|error| error.to_string())?;
        let direct = sample_direct_coefficients_at(
            &TimestampedDirectCoefficients {
                timestamp_micros: previous.timestamp_micros,
                coefficients: previous.direct,
            },
            &TimestampedDirectCoefficients {
                timestamp_micros: current.timestamp_micros,
                coefficients: current.direct,
            },
            timestamp,
            horizon,
        )
        .map_err(|error| error.to_string())?;
        let next = sources.get(consumed).unwrap_or(current);
        let teacher = interpolate_teacher(current, next, timestamp)?;
        let semantic = GnmSemanticFrame {
            frame_seq: render_seq as u64,
            timestamp_micros: timestamp,
            reduced_expression: sampled.values().to_vec(),
            joint_rotations: current.joint_rotations.clone(),
            rigid_yaw_pitch_roll: current.rigid_yaw_pitch_roll,
            objective: current.objective,
            region_fits: current.region_fits.clone(),
            direct,
        };
        semantic_history.insert(0, semantic);
        semantic_history.truncate(decoder.feature_config.history_len.max(2));
        let Some(features) = build_gnm_semantic_features(
            &semantic_history,
            GnmSemanticDecoderKind::HybridResidual,
            decoder.feature_config,
        )
        .map_err(|error| error.to_string())?
        else {
            continue;
        };
        let residual =
            predict_gnm_semantic_raw(decoder, &features).map_err(|error| format!("{error:?}"))?;
        let hybrid =
            apply_non_tongue_residual(&direct, residual).map_err(|error| error.to_string())?;
        let frame_seq = render_seq as u64;
        direct_frames.push(VariantFrame {
            take_id: current.take_id.clone(),
            frame_seq,
            timestamp_micros: timestamp,
            teacher,
            output: direct,
        });
        hybrid_frames.push(VariantFrame {
            take_id: current.take_id.clone(),
            frame_seq,
            timestamp_micros: timestamp,
            teacher,
            output: hybrid,
        });
        q_samples.push((timestamp, sampled.values().to_vec()));
        horizons.push(timestamp.saturating_sub(current.timestamp_micros) as f64);
    }
    Ok(CandidateReplay {
        direct: direct_frames,
        hybrid: hybrid_frames,
        q_samples,
        horizons,
    })
}

fn source_variants(
    sources: &[SourceFrame],
    decoder: &GnmSemanticDecoderArtifact,
) -> Result<(Vec<VariantFrame>, Vec<VariantFrame>), String> {
    let mut history = Vec::new();
    let mut direct = Vec::new();
    let mut hybrid = Vec::new();
    for source in sources {
        history.insert(
            0,
            GnmSemanticFrame {
                frame_seq: source.frame_seq,
                timestamp_micros: source.timestamp_micros,
                reduced_expression: source.reduced.values().to_vec(),
                joint_rotations: source.joint_rotations.clone(),
                rigid_yaw_pitch_roll: source.rigid_yaw_pitch_roll,
                objective: source.objective,
                region_fits: source.region_fits.clone(),
                direct: source.direct,
            },
        );
        history.truncate(decoder.feature_config.history_len.max(2));
        let Some(features) = build_gnm_semantic_features(
            &history,
            GnmSemanticDecoderKind::HybridResidual,
            decoder.feature_config,
        )
        .map_err(|error| error.to_string())?
        else {
            continue;
        };
        let residual =
            predict_gnm_semantic_raw(decoder, &features).map_err(|error| format!("{error:?}"))?;
        let h = apply_non_tongue_residual(&source.direct, residual)
            .map_err(|error| error.to_string())?;
        direct.push(VariantFrame {
            take_id: source.take_id.clone(),
            frame_seq: source.frame_seq,
            timestamp_micros: source.timestamp_micros,
            teacher: source.teacher,
            output: source.direct,
        });
        hybrid.push(VariantFrame {
            take_id: source.take_id.clone(),
            frame_seq: source.frame_seq,
            timestamp_micros: source.timestamp_micros,
            teacher: source.teacher,
            output: h,
        });
    }
    Ok((direct, hybrid))
}

fn candidate_metrics(replay: &CandidateReplay) -> Result<TemporalCandidateMetrics, String> {
    let h = evaluate_non_tongue_variant(&replay.hybrid).map_err(|error| error.to_string())?;
    let hb = evaluate_blink_events(&replay.hybrid).map_err(|error| error.to_string())?;
    let db = evaluate_blink_events(&replay.direct).map_err(|error| error.to_string())?;
    let required =
        |value: Option<f64>, field: &str| value.ok_or_else(|| format!("missing {field}"));
    Ok(TemporalCandidateMetrics {
        eye_preset: AlphaBetaGain {
            alpha: 0.0,
            beta: 0.0,
        },
        lower_face_preset: AlphaBetaGain {
            alpha: 0.0,
            beta: 0.0,
        },
        h_macro_mae: h.macro_mae,
        h_missed_blinks: hb.missed_events,
        d_missed_blinks: db.missed_events,
        h_onset_error_ms: required(hb.median_absolute_onset_error_ms, "H onset metric")?,
        d_onset_error_ms: required(db.median_absolute_onset_error_ms, "D onset metric")?,
        h_peak_error_ms: required(hb.median_absolute_peak_error_ms, "H peak metric")?,
        d_peak_error_ms: required(db.median_absolute_peak_error_ms, "D peak metric")?,
        h_peak_attenuation: required(hb.median_absolute_peak_attenuation, "H attenuation")?,
        d_peak_attenuation: required(db.median_absolute_peak_attenuation, "D attenuation")?,
    })
}

fn metric_summary(frames: &[VariantFrame]) -> Result<MetricSummary, String> {
    let values = evaluate_non_tongue_variant(frames).map_err(|error| error.to_string())?;
    let blink = evaluate_blink_events(frames).map_err(|error| error.to_string())?;
    Ok(MetricSummary {
        frame_count: values.frame_count,
        macro_mae: values.macro_mae,
        macro_rmse: values.macro_rmse,
        velocity_mae: values.temporal.velocity_mae,
        acceleration_mae: values.temporal.acceleration_mae,
        jitter: values.temporal.neutral_jitter,
        peak_jerk_mae: values.temporal.peak_jerk_mae,
        missed_blinks: blink.missed_events,
        onset_error_ms: blink.median_absolute_onset_error_ms,
        peak_error_ms: blink.median_absolute_peak_error_ms,
        peak_attenuation: blink.median_absolute_peak_attenuation,
    })
}

fn distribution(values: &[f64]) -> Result<Distribution, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("empty or non-finite distribution".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |percent: usize| {
        let index = ((sorted.len() - 1) * percent + 50) / 100;
        #[allow(clippy::indexing_slicing)]
        sorted[index]
    };
    Ok(Distribution {
        p50: at(50),
        p95: at(95),
        max: sorted.last().copied().unwrap_or_default(),
    })
}

fn q_kinematics(samples: &[(u64, Vec<f32>)]) -> Result<QKinematics, String> {
    let derivative = |input: &[(u64, Vec<f32>)]| -> Result<Vec<(u64, Vec<f32>)>, String> {
        input
            .windows(2)
            .filter_map(|window| {
                let [previous, current] = window else {
                    return Some(Err("invalid derivative window".to_owned()));
                };
                if current.0 <= previous.0 {
                    return None;
                }
                let dt = (current.0 - previous.0) as f32 / 1_000_000.0;
                Some(Ok((
                    current.0,
                    current
                        .1
                        .iter()
                        .zip(&previous.1)
                        .map(|(current, previous)| (current - previous) / dt)
                        .collect(),
                )))
            })
            .collect()
    };
    let velocity = derivative(samples)?;
    let acceleration = derivative(&velocity)?;
    let jerk = derivative(&acceleration)?;
    let norms = |values: &[(u64, Vec<f32>)]| {
        values
            .iter()
            .map(|(_, values)| {
                values
                    .iter()
                    .map(|value| f64::from(value * value))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect::<Vec<_>>()
    };
    Ok(QKinematics {
        velocity: distribution(&norms(&velocity))?,
        acceleration: distribution(&norms(&acceleration))?,
        jerk: distribution(&norms(&jerk))?,
    })
}

fn rate(frames: &[SourceFrame]) -> f64 {
    let duration = frames
        .last()
        .map(|frame| frame.timestamp_micros)
        .unwrap_or_default()
        - frames
            .first()
            .map(|frame| frame.timestamp_micros)
            .unwrap_or_default();
    (frames.len().saturating_sub(1)) as f64 * 1_000_000.0 / duration as f64
}

/// Runs training-only fixed-grid tuning and writes artifact/report files.
pub fn run(args: &[String]) -> Result<(), String> {
    if args.iter().any(|argument| argument == "--help") {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let basis_artifact: TeacherAlignedGnmBasisArtifact = read_json(&options.basis)?;
    let decoder: GnmSemanticDecoderArtifact = read_json(&options.decoder)?;
    if decoder.kind != GnmSemanticDecoderKind::HybridResidual
        || decoder.aligned_basis_content_hash != basis_artifact.content_hash
    {
        return Err("hybrid decoder does not match the selected basis".to_owned());
    }
    let model_sha = sha256_hex(Path::new("assets/models/gnm_head.npz"))?;
    let mut traces = options
        .traces
        .iter()
        .map(|path| load_trace(path))
        .collect::<Result<Vec<_>, _>>()?;
    traces.sort_by(|left, right| left.take_id.cmp(&right.take_id));
    if traces
        .iter()
        .map(|trace| trace.take_id.as_str())
        .collect::<BTreeSet<_>>()
        != options.train_takes.iter().map(String::as_str).collect()
    {
        return Err("--trace takes must exactly equal --train-take selections".to_owned());
    }
    if traces.iter().any(|trace| trace.model_sha256 != model_sha) {
        return Err("trace model SHA-256 mismatch".to_owned());
    }
    let mapping_revision = traces.first().ok_or("no traces")?.mapping_schema_revision;
    let basis = load_reduced_gnm_basis(&basis_artifact, &model_sha, mapping_revision)
        .map_err(|error| error.to_string())?;
    let mut solved_takes = Vec::new();
    let mut resets = 0;
    for trace in &traces {
        let (solved, take_resets) = solve_sources(trace, &basis)?;
        resets += take_resets;
        solved_takes.push(solved);
    }

    let mut candidates = Vec::new();
    let mut replays = Vec::new();
    for (eye, lower) in reduced_temporal_gain_grid() {
        let mut combined = CandidateReplay {
            direct: Vec::new(),
            hybrid: Vec::new(),
            q_samples: Vec::new(),
            horizons: Vec::new(),
        };
        for sources in &solved_takes {
            let replay = replay_candidate(
                sources,
                &basis,
                &decoder,
                eye,
                lower,
                options.max_prediction_horizon_micros,
            )?;
            combined.direct.extend(replay.direct);
            combined.hybrid.extend(replay.hybrid);
            combined.q_samples.extend(replay.q_samples);
            combined.horizons.extend(replay.horizons);
        }
        let mut metrics = candidate_metrics(&combined)?;
        metrics.eye_preset = eye;
        metrics.lower_face_preset = lower;
        candidates.push(metrics);
        replays.push(combined);
    }
    fs::create_dir_all(&options.output)
        .map_err(|error| format!("create {}: {error}", options.output.display()))?;
    let candidate_json =
        serde_json::to_string_pretty(&candidates).map_err(|error| error.to_string())?;
    fs::write(
        options.output.join("candidate-grid.json"),
        format!("{candidate_json}\n"),
    )
    .map_err(|error| error.to_string())?;
    let artifact = select_reduced_temporal_artifact(
        &candidates,
        ReducedTemporalProvenance {
            basis_content_hash: basis_artifact.content_hash,
            decoder_content_hash: decoder.content_hash,
            max_prediction_horizon_micros: options.max_prediction_horizon_micros,
            training_takes: options.train_takes.iter().cloned().collect(),
        },
    )
    .map_err(|error| error.to_string())?;
    let selected_index = candidates
        .iter()
        .position(|candidate| {
            candidate.eye_preset == artifact.eye_preset
                && candidate.lower_face_preset == artifact.lower_face_preset
        })
        .ok_or("selected candidate missing from grid")?;
    let selected = replays
        .get(selected_index)
        .ok_or("selected replay missing")?;

    let mut source_direct = Vec::new();
    let mut source_hybrid = Vec::new();
    for sources in &solved_takes {
        let (direct, hybrid) = source_variants(sources, &decoder)?;
        source_direct.extend(direct);
        source_hybrid.extend(hybrid);
    }
    let source_count = solved_takes.iter().map(Vec::len).sum();
    let source_rate =
        solved_takes.iter().map(|take| rate(take)).sum::<f64>() / solved_takes.len() as f64;
    let report = TuningReport {
        schema_version: 1,
        basis_content_hash: basis_artifact.content_hash,
        decoder_content_hash: decoder.content_hash,
        temporal_artifact_content_hash: artifact.content_hash,
        training_takes: artifact.training_takes.clone(),
        leave_one_take_out_folds: solved_takes.len(),
        source_observations: source_count,
        source_observation_rate_hz: source_rate,
        render_samples: selected.hybrid.len(),
        render_sample_rate_hz: RENDER_RATE as f64,
        temporal_history_resets: resets,
        prediction_horizon_micros: distribution(&selected.horizons)?,
        candidates: candidates.clone(),
        selected_eye_preset: artifact.eye_preset,
        selected_lower_face_preset: artifact.lower_face_preset,
        source_direct: metric_summary(&source_direct)?,
        source_hybrid: metric_summary(&source_hybrid)?,
        render_hybrid: metric_summary(&selected.hybrid)?,
        q_kinematics: q_kinematics(&selected.q_samples)?,
    };
    let artifact_json =
        serde_json::to_string_pretty(&artifact).map_err(|error| error.to_string())?;
    fs::write(
        options.output.join("reduced-temporal-artifact.json"),
        format!("{artifact_json}\n"),
    )
    .map_err(|error| error.to_string())?;
    let report_json = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    fs::write(
        options.output.join("report.json"),
        format!("{report_json}\n"),
    )
    .map_err(|error| error.to_string())?;
    println!(
        "selected eye ({:.2}, {:.2}) lower ({:.2}, {:.2}) over {} render frames",
        artifact.eye_preset.alpha,
        artifact.eye_preset.beta,
        artifact.lower_face_preset.alpha,
        artifact.lower_face_preset.beta,
        selected.hybrid.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_source_prefix_excludes_future_frames() {
        let frame = |timestamp_micros| SourceFrame {
            take_id: "take".to_owned(),
            frame_seq: timestamp_micros,
            timestamp_micros,
            reduced: GnmReducedExpressionState::neutral(1),
            joint_rotations: Vec::new(),
            rigid_yaw_pitch_roll: [0.0; 3],
            objective: 0.0,
            region_fits: Vec::new(),
            direct: Arkit52Coefficients::default(),
            teacher: Arkit52Coefficients::default(),
        };
        let original = vec![frame(10), frame(20), frame(30)];
        let mut future_changed = original.clone();
        future_changed[2].reduced = GnmReducedExpressionState::new(vec![999.0], 1).unwrap();
        let original_end = available_source_end(&original, 20);
        let changed_end = available_source_end(&future_changed, 20);
        assert_eq!(original_end, 2);
        assert_eq!(changed_end, 2);
        assert_eq!(
            original[..original_end]
                .iter()
                .map(|f| f.reduced.values())
                .collect::<Vec<_>>(),
            future_changed[..changed_end]
                .iter()
                .map(|f| f.reduced.values())
                .collect::<Vec<_>>()
        );
    }
}
