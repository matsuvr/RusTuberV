//! Unified D/G0/G1/L/H/HL offline ablation and promotion report.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;
use sha2::{Digest, Sha256};
use vtuber_core::{ARKIT_NON_TONGUE_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};
use vtuber_tracking::{
    BlinkEventMetrics, GnmResearchVariant, GnmSemanticDecoderArtifact, GnmSemanticDecoderKind,
    GnmSemanticFeatureConfig, GnmSemanticRow, LandmarkAlignedBasisArtifact,
    LandmarkControlDecoderArtifact, LandmarkControlDecoderKind, LinearPriorTrainingConfig,
    NonTongueVariantMetrics, TeacherAlignedGnmBasisArtifact, VariantFrame,
    apply_non_tongue_residual, avatar_space_rms_errors, build_gnm_semantic_rows,
    build_landmark_alignment_samples, build_landmark_control_rows, build_teacher_alignment_samples,
    evaluate_blink_events, evaluate_non_tongue_variant, exact_common_variant_frames,
    fit_gnm_semantic_decoder, fit_landmark_aligned_basis, fit_landmark_control_decoder,
    fit_teacher_aligned_gnm_basis, gnm_only_prediction_to_arkit52, gnm_semantic_feature_order,
    paired_absolute_error_delta, predict_gnm_semantic_raw, predict_landmark_control_raw,
};

use crate::perfect_sync_morph::load_perfect_sync_morph;
use crate::teacher_fit_observable_basis::fit_observable_basis;
use crate::teacher_fit_prior::{LoadedTrace, load_trace};

const RANKS: [usize; 4] = [16, 24, 32, 48];
const HISTORIES: [usize; 3] = [1, 2, 3];
const RIDGES: [f32; 4] = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1];
const MAX_GAP_MICROS: u64 = 100_000;

type ResidualRows = (
    Vec<GnmSemanticRow>,
    Vec<GnmSemanticRow>,
    Vec<GnmSemanticRow>,
);
type AllDecoderRows = (
    Vec<GnmSemanticRow>,
    Vec<GnmSemanticRow>,
    Vec<GnmSemanticRow>,
    Vec<GnmSemanticRow>,
);

struct Options {
    traces: Vec<PathBuf>,
    outer_train_takes: BTreeSet<String>,
    outer_eval_takes: BTreeSet<String>,
    observable_rank: usize,
    vrms: Vec<PathBuf>,
    output: PathBuf,
    gnm_model: PathBuf,
    reuse_fit: Option<PathBuf>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut traces = Vec::new();
        let mut outer_train_takes = BTreeSet::new();
        let mut outer_eval_takes = BTreeSet::new();
        let mut observable_rank = None;
        let mut vrms = Vec::new();
        let mut output = None;
        let mut gnm_model = None;
        let mut reuse_fit = None;
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
                "--outer-train-take" => {
                    outer_train_takes.insert(next(&mut index, "--outer-train-take")?);
                }
                "--outer-eval-take" => {
                    outer_eval_takes.insert(next(&mut index, "--outer-eval-take")?);
                }
                "--observable-rank" => {
                    observable_rank = Some(
                        next(&mut index, "--observable-rank")?
                            .parse()
                            .map_err(|_| "--observable-rank must be an integer")?,
                    );
                }
                "--vrm" => vrms.push(PathBuf::from(next(&mut index, "--vrm")?)),
                "--output" => output = Some(PathBuf::from(next(&mut index, "--output")?)),
                "--gnm-model" => {
                    gnm_model = Some(PathBuf::from(next(&mut index, "--gnm-model")?));
                }
                "--reuse-fit" => {
                    reuse_fit = Some(PathBuf::from(next(&mut index, "--reuse-fit")?));
                }
                other => return Err(format!("unknown option {other}")),
            }
            index += 1;
        }
        if traces.is_empty()
            || outer_train_takes.len() < 2
            || outer_eval_takes.is_empty()
            || vrms.is_empty()
        {
            return Err(
                "--trace, at least two --outer-train-take, --outer-eval-take, and --vrm are required"
                    .to_owned(),
            );
        }
        if !outer_train_takes.is_disjoint(&outer_eval_takes) {
            return Err("outer training and evaluation takes must be disjoint".to_owned());
        }
        let observable_rank = observable_rank.ok_or("--observable-rank <n> is required")?;
        if observable_rank < *RANKS.last().ok_or("empty rank grid")? {
            return Err("--observable-rank must be at least 48".to_owned());
        }
        Ok(Self {
            traces,
            outer_train_takes,
            outer_eval_takes,
            observable_rank,
            vrms,
            output: output.ok_or("--output <directory> is required")?,
            gnm_model: gnm_model.unwrap_or_else(|| PathBuf::from("assets/models/gnm_head.npz")),
            reuse_fit,
        })
    }
}

pub fn print_help() {
    println!(
        "  teacher-unified-gnm-ablation --trace <trace-v2-dir> [...]\n\
         *       --outer-train-take <id> [...] --outer-eval-take <id> [...]\n\
         *       --observable-rank <n>=48+ --vrm <perfect-sync.vrm> [...]\n\
         *       --output <directory> [--gnm-model <gnm_head.npz>] [--reuse-fit <directory>]"
    );
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct Candidate {
    rank: usize,
    history_len: usize,
    ridge_lambda: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct CandidateResult {
    candidate: Candidate,
    h_validation_macro_mae: f64,
    l_validation_macro_mae: f64,
    score: f64,
    validation_takes: Vec<String>,
}

struct FittedModels {
    observable: vtuber_tracking::ObservableGnmBasisArtifact,
    gnm_basis: TeacherAlignedGnmBasisArtifact,
    landmark_basis: LandmarkAlignedBasisArtifact,
    g1: GnmSemanticDecoderArtifact,
    h: GnmSemanticDecoderArtifact,
    l: LandmarkControlDecoderArtifact,
    hl: LandmarkControlDecoderArtifact,
}

fn load_traces(options: &Options) -> Result<Vec<LoadedTrace>, String> {
    let mut traces = options
        .traces
        .iter()
        .map(|path| load_trace(path))
        .collect::<Result<Vec<_>, _>>()?;
    traces.sort_by(|left, right| left.take_id.cmp(&right.take_id));
    let available: BTreeSet<&str> = traces.iter().map(|trace| trace.take_id.as_str()).collect();
    if !options
        .outer_train_takes
        .iter()
        .chain(&options.outer_eval_takes)
        .all(|take| available.contains(take.as_str()))
    {
        return Err("a split take was not supplied by --trace".to_owned());
    }
    Ok(traces)
}

fn clone_trace(trace: &LoadedTrace) -> LoadedTrace {
    LoadedTrace {
        take_id: trace.take_id.clone(),
        samples: trace.samples.clone(),
        expected_solved: trace.expected_solved,
        paired: trace.paired,
        no_face: trace.no_face,
        observation_insufficient: trace.observation_insufficient,
        fit_rejected: trace.fit_rejected,
        excluded_unpaired_teacher: trace.excluded_unpaired_teacher,
        excluded_unpaired_rgb: trace.excluded_unpaired_rgb,
        trace_sha256: trace.trace_sha256.clone(),
        model_sha256: trace.model_sha256.clone(),
        mapping_schema_revision: trace.mapping_schema_revision,
    }
}

fn alignment_samples(
    traces: &[LoadedTrace],
) -> Result<
    (
        Vec<vtuber_tracking::TeacherAlignmentSample>,
        Vec<vtuber_tracking::LandmarkAlignmentSample>,
    ),
    String,
> {
    let mut gnm = Vec::new();
    let mut landmark = Vec::new();
    for trace in traces {
        gnm.extend(
            build_teacher_alignment_samples(&trace.take_id, &trace.samples)
                .map_err(|error| format!("take {}: {error:?}", trace.take_id))?,
        );
        landmark.extend(
            build_landmark_alignment_samples(&trace.take_id, &trace.samples)
                .map_err(|error| format!("take {}: {error}", trace.take_id))?,
        );
    }
    Ok((gnm, landmark))
}

fn decoder_rows(
    traces: &[LoadedTrace],
    gnm_basis: &TeacherAlignedGnmBasisArtifact,
    landmark_basis: &LandmarkAlignedBasisArtifact,
    config: GnmSemanticFeatureConfig,
) -> Result<ResidualRows, String> {
    let mut gnm = Vec::new();
    let mut landmark = Vec::new();
    let mut upper = Vec::new();
    for trace in traces {
        gnm.extend(
            build_gnm_semantic_rows(
                &trace.take_id,
                &trace.samples,
                gnm_basis,
                GnmSemanticDecoderKind::HybridResidual,
                config,
            )
            .map_err(|error| format!("take {} H rows: {error}", trace.take_id))?,
        );
        landmark.extend(
            build_landmark_control_rows(
                &trace.take_id,
                &trace.samples,
                landmark_basis,
                None,
                LandmarkControlDecoderKind::LandmarkResidual,
                config,
            )
            .map_err(|error| format!("take {} L rows: {error}", trace.take_id))?,
        );
        upper.extend(
            build_landmark_control_rows(
                &trace.take_id,
                &trace.samples,
                landmark_basis,
                Some(gnm_basis),
                LandmarkControlDecoderKind::GnmLandmarkUpperBound,
                config,
            )
            .map_err(|error| format!("take {} HL rows: {error}", trace.take_id))?,
        );
    }
    Ok((gnm, landmark, upper))
}

fn fit_h_l(
    h_rows: &[GnmSemanticRow],
    l_rows: &[GnmSemanticRow],
    training: &BTreeSet<String>,
    gnm_basis: &TeacherAlignedGnmBasisArtifact,
    landmark_basis: &LandmarkAlignedBasisArtifact,
    ridge_lambda: f32,
) -> Result<(GnmSemanticDecoderArtifact, LandmarkControlDecoderArtifact), String> {
    let training_config = LinearPriorTrainingConfig {
        ridge_lambda,
        ..LinearPriorTrainingConfig::default()
    };
    let h = fit_gnm_semantic_decoder(
        h_rows,
        training,
        GnmSemanticDecoderKind::HybridResidual,
        gnm_basis,
        training_config,
        &gnm_semantic_feature_order(GnmSemanticDecoderKind::HybridResidual),
    )
    .map_err(|error| error.to_string())?;
    let l = fit_landmark_control_decoder(
        l_rows,
        training,
        LandmarkControlDecoderKind::LandmarkResidual,
        landmark_basis,
        None,
        training_config,
    )
    .map_err(|error| error.to_string())?;
    Ok((h, l))
}

fn residual_frames(
    trace: &LoadedTrace,
    rows: &[GnmSemanticRow],
    predict: impl Fn(&[f32]) -> Result<[f32; ARKIT_NON_TONGUE_CHANNEL_COUNT], String>,
) -> Result<Vec<VariantFrame>, String> {
    let samples: BTreeMap<u64, _> = trace
        .samples
        .iter()
        .map(|sample| (sample.frame_seq, sample))
        .collect();
    rows.iter()
        .filter(|row| row.take_id == trace.take_id)
        .map(|row| {
            let sample = samples.get(&row.frame_seq).ok_or_else(|| {
                format!("take {} missing row frame {}", trace.take_id, row.frame_seq)
            })?;
            let teacher = sample.teacher.as_ref().ok_or_else(|| {
                format!(
                    "take {} frame {} missing teacher",
                    trace.take_id, row.frame_seq
                )
            })?;
            let direct = sample.mediapipe_observation.as_ref().ok_or_else(|| {
                format!(
                    "take {} frame {} missing Direct",
                    trace.take_id, row.frame_seq
                )
            })?;
            let output =
                apply_non_tongue_residual(&direct.direct_coefficients, predict(&row.features)?)
                    .map_err(|error| error.to_string())?;
            Ok(VariantFrame {
                take_id: trace.take_id.clone(),
                frame_seq: row.frame_seq,
                timestamp_micros: sample.timestamp_micros,
                teacher: teacher.coefficients,
                output,
            })
        })
        .collect()
}

fn common_h_l_metrics(
    trace: &LoadedTrace,
    h_rows: &[GnmSemanticRow],
    l_rows: &[GnmSemanticRow],
    h: &GnmSemanticDecoderArtifact,
    l: &LandmarkControlDecoderArtifact,
) -> Result<(NonTongueVariantMetrics, NonTongueVariantMetrics), String> {
    let h_frames = residual_frames(trace, h_rows, |features| {
        predict_gnm_semantic_raw(h, features).map_err(|error| format!("H predict: {error:?}"))
    })?;
    let l_frames = residual_frames(trace, l_rows, |features| {
        predict_landmark_control_raw(l, features).map_err(|error| format!("L predict: {error:?}"))
    })?;
    let l_identity: BTreeSet<_> = l_frames
        .iter()
        .map(|frame| (frame.frame_seq, frame.timestamp_micros))
        .collect();
    let h_common: Vec<_> = h_frames
        .into_iter()
        .filter(|frame| l_identity.contains(&(frame.frame_seq, frame.timestamp_micros)))
        .collect();
    let h_identity: BTreeSet<_> = h_common
        .iter()
        .map(|frame| (frame.frame_seq, frame.timestamp_micros))
        .collect();
    let l_common: Vec<_> = l_frames
        .into_iter()
        .filter(|frame| h_identity.contains(&(frame.frame_seq, frame.timestamp_micros)))
        .collect();
    Ok((
        evaluate_non_tongue_variant(&h_common).map_err(|error| error.to_string())?,
        evaluate_non_tongue_variant(&l_common).map_err(|error| error.to_string())?,
    ))
}

fn select_candidate(
    traces: &[LoadedTrace],
    outer_training: &BTreeSet<String>,
    observable_rank: usize,
    gnm_model: &Path,
) -> Result<Vec<CandidateResult>, String> {
    let training_traces: Vec<LoadedTrace> = traces
        .iter()
        .filter(|trace| outer_training.contains(&trace.take_id))
        .map(clone_trace)
        .collect();
    let (gnm_samples, landmark_samples) = alignment_samples(&training_traces)?;
    let mut scores: BTreeMap<(usize, usize, u32), (f64, f64, usize)> = BTreeMap::new();
    for held_out in outer_training {
        let fold_training: BTreeSet<String> = outer_training
            .iter()
            .filter(|take| *take != held_out)
            .cloned()
            .collect();
        if fold_training.is_empty() {
            return Err("leave-one-take-out needs at least two outer training takes".to_owned());
        }
        let owned_training: Vec<LoadedTrace> = training_traces
            .iter()
            .filter(|trace| fold_training.contains(&trace.take_id))
            .map(clone_trace)
            .collect();
        let (observable, _) =
            fit_observable_basis(&owned_training, &fold_training, observable_rank, gnm_model)?;
        let validation = training_traces
            .iter()
            .find(|trace| trace.take_id == *held_out)
            .ok_or_else(|| format!("missing validation take {held_out}"))?;
        for rank in RANKS {
            let gnm_basis =
                fit_teacher_aligned_gnm_basis(&observable, &gnm_samples, &fold_training, rank)
                    .map_err(|error| error.to_string())?;
            let landmark_basis =
                fit_landmark_aligned_basis(&landmark_samples, &fold_training, rank)
                    .map_err(|error| error.to_string())?;
            for history_len in HISTORIES {
                let config = GnmSemanticFeatureConfig {
                    history_len,
                    max_gap_micros: MAX_GAP_MICROS,
                };
                let (h_rows, l_rows, _) =
                    decoder_rows(&training_traces, &gnm_basis, &landmark_basis, config)?;
                let ridge_results = std::thread::scope(|scope| {
                    let mut handles = Vec::new();
                    for ridge_lambda in RIDGES {
                        let h_rows = &h_rows;
                        let l_rows = &l_rows;
                        let fold_training = &fold_training;
                        let gnm_basis = &gnm_basis;
                        let landmark_basis = &landmark_basis;
                        handles.push(scope.spawn(move || {
                            let (h, l) = fit_h_l(
                                h_rows,
                                l_rows,
                                fold_training,
                                gnm_basis,
                                landmark_basis,
                                ridge_lambda,
                            )?;
                            let metrics = common_h_l_metrics(validation, h_rows, l_rows, &h, &l)?;
                            Ok::<_, String>((ridge_lambda, metrics))
                        }));
                    }
                    handles
                        .into_iter()
                        .map(|handle| {
                            handle
                                .join()
                                .map_err(|_| "ridge worker panicked".to_owned())?
                        })
                        .collect::<Result<Vec<_>, String>>()
                })?;
                for (ridge_lambda, (h_metrics, l_metrics)) in ridge_results {
                    let entry = scores
                        .entry((rank, history_len, ridge_lambda.to_bits()))
                        .or_insert((0.0, 0.0, 0));
                    entry.0 += h_metrics.macro_mae;
                    entry.1 += l_metrics.macro_mae;
                    entry.2 += 1;
                }
            }
        }
    }
    let validation_takes: Vec<String> = outer_training.iter().cloned().collect();
    let mut results: Vec<CandidateResult> = scores
        .into_iter()
        .map(|((rank, history_len, ridge_bits), (h_sum, l_sum, count))| {
            let h = h_sum / count as f64;
            let l = l_sum / count as f64;
            CandidateResult {
                candidate: Candidate {
                    rank,
                    history_len,
                    ridge_lambda: f32::from_bits(ridge_bits),
                },
                h_validation_macro_mae: h,
                l_validation_macro_mae: l,
                score: (h + l) / 2.0,
                validation_takes: validation_takes.clone(),
            }
        })
        .collect();
    results.sort_by(|left, right| {
        left.score
            .total_cmp(&right.score)
            .then(left.candidate.rank.cmp(&right.candidate.rank))
            .then(left.candidate.history_len.cmp(&right.candidate.history_len))
            .then(
                right
                    .candidate
                    .ridge_lambda
                    .total_cmp(&left.candidate.ridge_lambda),
            )
    });
    Ok(results)
}

fn fit_final_models(
    traces: &[LoadedTrace],
    training: &BTreeSet<String>,
    candidate: Candidate,
    observable_rank: usize,
    gnm_model: &Path,
) -> Result<FittedModels, String> {
    let training_traces: Vec<LoadedTrace> = traces
        .iter()
        .filter(|trace| training.contains(&trace.take_id))
        .map(clone_trace)
        .collect();
    let (observable, _) =
        fit_observable_basis(&training_traces, training, observable_rank, gnm_model)?;
    let (gnm_samples, landmark_samples) = alignment_samples(&training_traces)?;
    let gnm_basis =
        fit_teacher_aligned_gnm_basis(&observable, &gnm_samples, training, candidate.rank)
            .map_err(|error| error.to_string())?;
    let landmark_basis = fit_landmark_aligned_basis(&landmark_samples, training, candidate.rank)
        .map_err(|error| error.to_string())?;
    let config = GnmSemanticFeatureConfig {
        history_len: candidate.history_len,
        max_gap_micros: MAX_GAP_MICROS,
    };
    let (h_rows, l_rows, hl_rows) =
        decoder_rows(&training_traces, &gnm_basis, &landmark_basis, config)?;
    let training_config = LinearPriorTrainingConfig {
        ridge_lambda: candidate.ridge_lambda,
        ..LinearPriorTrainingConfig::default()
    };
    let g1_rows = training_traces
        .iter()
        .map(|trace| {
            build_gnm_semantic_rows(
                &trace.take_id,
                &trace.samples,
                &gnm_basis,
                GnmSemanticDecoderKind::GnmOnly,
                config,
            )
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let g1 = fit_gnm_semantic_decoder(
        &g1_rows,
        training,
        GnmSemanticDecoderKind::GnmOnly,
        &gnm_basis,
        training_config,
        &gnm_semantic_feature_order(GnmSemanticDecoderKind::GnmOnly),
    )
    .map_err(|error| error.to_string())?;
    let (h, l) = fit_h_l(
        &h_rows,
        &l_rows,
        training,
        &gnm_basis,
        &landmark_basis,
        candidate.ridge_lambda,
    )?;
    let hl = fit_landmark_control_decoder(
        &hl_rows,
        training,
        LandmarkControlDecoderKind::GnmLandmarkUpperBound,
        &landmark_basis,
        Some(&gnm_basis),
        training_config,
    )
    .map_err(|error| error.to_string())?;
    Ok(FittedModels {
        observable,
        gnm_basis,
        landmark_basis,
        g1,
        h,
        l,
        hl,
    })
}

fn rows_by_kind(trace: &LoadedTrace, models: &FittedModels) -> Result<AllDecoderRows, String> {
    let config = models.h.feature_config;
    let g1 = build_gnm_semantic_rows(
        &trace.take_id,
        &trace.samples,
        &models.gnm_basis,
        GnmSemanticDecoderKind::GnmOnly,
        config,
    )
    .map_err(|error| error.to_string())?;
    let h = build_gnm_semantic_rows(
        &trace.take_id,
        &trace.samples,
        &models.gnm_basis,
        GnmSemanticDecoderKind::HybridResidual,
        config,
    )
    .map_err(|error| error.to_string())?;
    let l = build_landmark_control_rows(
        &trace.take_id,
        &trace.samples,
        &models.landmark_basis,
        None,
        LandmarkControlDecoderKind::LandmarkResidual,
        config,
    )
    .map_err(|error| error.to_string())?;
    let hl = build_landmark_control_rows(
        &trace.take_id,
        &trace.samples,
        &models.landmark_basis,
        Some(&models.gnm_basis),
        LandmarkControlDecoderKind::GnmLandmarkUpperBound,
        config,
    )
    .map_err(|error| error.to_string())?;
    Ok((g1, h, l, hl))
}

fn final_variant_frames(
    trace: &LoadedTrace,
    models: &FittedModels,
) -> Result<BTreeMap<GnmResearchVariant, Vec<VariantFrame>>, String> {
    let (g1_rows, h_rows, l_rows, hl_rows) = rows_by_kind(trace, models)?;
    let baseline =
        |variant: GnmResearchVariant,
         pick: fn(&vtuber_tracking::PairedTemporalSample) -> Option<Arkit52Coefficients>| {
            trace
                .samples
                .iter()
                .filter_map(|sample| {
                    Some(VariantFrame {
                        take_id: trace.take_id.clone(),
                        frame_seq: sample.frame_seq,
                        timestamp_micros: sample.timestamp_micros,
                        teacher: sample.teacher.as_ref()?.coefficients,
                        output: pick(sample)?,
                    })
                })
                .map(|frame| (variant, frame))
                .collect::<Vec<_>>()
        };
    let mut tagged = baseline(GnmResearchVariant::Direct, |sample| {
        sample
            .mediapipe_observation
            .as_ref()
            .map(|observation| observation.direct_coefficients)
    });
    tagged.extend(baseline(GnmResearchVariant::GnmProjected, |sample| {
        sample
            .gnm_state
            .as_ref()
            .map(|state| state.projected_coefficients)
    }));
    let samples: BTreeMap<u64, _> = trace
        .samples
        .iter()
        .map(|sample| (sample.frame_seq, sample))
        .collect();
    let g1 = g1_rows
        .iter()
        .map(|row| {
            let sample = samples.get(&row.frame_seq).ok_or("G1 sample identity")?;
            let teacher = sample.teacher.as_ref().ok_or("G1 teacher")?;
            let raw = predict_gnm_semantic_raw(&models.g1, &row.features)
                .map_err(|error| format!("G1 predict: {error:?}"))?;
            Ok(VariantFrame {
                take_id: trace.take_id.clone(),
                frame_seq: row.frame_seq,
                timestamp_micros: sample.timestamp_micros,
                teacher: teacher.coefficients,
                output: gnm_only_prediction_to_arkit52(raw).map_err(|error| error.to_string())?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let h = residual_frames(trace, &h_rows, |features| {
        predict_gnm_semantic_raw(&models.h, features)
            .map_err(|error| format!("H predict: {error:?}"))
    })?;
    let l = residual_frames(trace, &l_rows, |features| {
        predict_landmark_control_raw(&models.l, features)
            .map_err(|error| format!("L predict: {error:?}"))
    })?;
    let hl = residual_frames(trace, &hl_rows, |features| {
        predict_landmark_control_raw(&models.hl, features)
            .map_err(|error| format!("HL predict: {error:?}"))
    })?;
    let mut result = BTreeMap::new();
    for variant in GnmResearchVariant::ALL {
        let frames = tagged
            .iter()
            .filter_map(|(tag, frame)| (*tag == variant).then_some(frame.clone()))
            .collect::<Vec<_>>();
        result.insert(variant, frames);
    }
    result.insert(GnmResearchVariant::GnmLearned, g1);
    result.insert(GnmResearchVariant::HybridGnmResidual, h);
    result.insert(GnmResearchVariant::LandmarkResidual, l);
    result.insert(GnmResearchVariant::HybridGnmLandmarkResidual, hl);
    Ok(result)
}

fn metrics_json(metrics: &NonTongueVariantMetrics) -> serde_json::Value {
    json!({
        "frames": metrics.frame_count,
        "macro_mae": metrics.macro_mae,
        "macro_rmse": metrics.macro_rmse,
        "micro_mae": metrics.micro_mae,
        "micro_rmse": metrics.micro_rmse,
        "channels": metrics.channels.iter().map(|channel| json!({
            "channel": channel.channel.canonical_name(),
            "mae": channel.mae,
            "rmse": channel.rmse,
            "pearson": channel.pearson,
            "neutral_bias": channel.neutral_bias,
        })).collect::<Vec<_>>(),
        "left_right": metrics.left_right.iter().map(|pair| json!({
            "left": pair.left.canonical_name(),
            "right": pair.right.canonical_name(),
            "difference_mae": pair.difference_mae,
        })).collect::<Vec<_>>(),
        "temporal": {
            "velocity_mae": metrics.temporal.velocity_mae,
            "acceleration_mae": metrics.temporal.acceleration_mae,
            "neutral_jitter": metrics.temporal.neutral_jitter,
            "peak_jerk_mae": metrics.temporal.peak_jerk_mae,
        }
    })
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn load_fitted_models(
    directory: &Path,
    training: &BTreeSet<String>,
) -> Result<(Vec<CandidateResult>, FittedModels), String> {
    let candidates: Vec<CandidateResult> = read_json(&directory.join("candidate-grid.json"))?;
    let models = FittedModels {
        observable: read_json(&directory.join("observable-basis.json"))?,
        gnm_basis: read_json(&directory.join("gnm-basis.json"))?,
        landmark_basis: read_json(&directory.join("landmark-basis.json"))?,
        g1: read_json(&directory.join("g1-decoder.json"))?,
        h: read_json(&directory.join("h-decoder.json"))?,
        l: read_json(&directory.join("l-decoder.json"))?,
        hl: read_json(&directory.join("hl-decoder.json"))?,
    };
    let expected = training.iter().cloned().collect::<Vec<_>>();
    if models.g1.training_takes != expected
        || models.h.training_takes != expected
        || models.l.training_takes != expected
        || models.hl.training_takes != expected
    {
        return Err("--reuse-fit artifacts use a different outer training split".to_owned());
    }
    Ok((candidates, models))
}

fn variant_name(variant: GnmResearchVariant) -> &'static str {
    match variant {
        GnmResearchVariant::Direct => "D",
        GnmResearchVariant::GnmProjected => "G0",
        GnmResearchVariant::GnmLearned => "G1",
        GnmResearchVariant::LandmarkResidual => "L",
        GnmResearchVariant::HybridGnmResidual => "H",
        GnmResearchVariant::HybridGnmLandmarkResidual => "HL",
    }
}

fn blink_json(metrics: &BlinkEventMetrics) -> serde_json::Value {
    json!({
        "teacher_events": metrics.teacher_events,
        "matched_events": metrics.matched_events,
        "missed_events": metrics.missed_events,
        "extra_events": metrics.extra_events,
        "median_absolute_onset_error_ms": metrics.median_absolute_onset_error_ms,
        "median_absolute_peak_error_ms": metrics.median_absolute_peak_error_ms,
        "median_absolute_release_error_ms": metrics.median_absolute_release_error_ms,
        "median_absolute_peak_attenuation": metrics.median_absolute_peak_attenuation,
    })
}

fn dataset_hash(traces: &[&LoadedTrace]) -> String {
    let mut hasher = Sha256::new();
    for trace in traces {
        hasher.update(trace.take_id.as_bytes());
        hasher.update([0]);
        hasher.update(trace.trace_sha256.as_bytes());
        hasher.update([0]);
    }
    format!("{:X}", hasher.finalize())
}

fn avatar_rms_json(
    frames: &BTreeMap<GnmResearchVariant, Vec<VariantFrame>>,
    responses: &[vtuber_tracking::PerfectSyncMorphResponse],
) -> Result<serde_json::Value, String> {
    let mut models = Vec::new();
    for response in responses {
        let mut variants = serde_json::Map::new();
        for variant in GnmResearchVariant::ALL {
            let series = frames.get(&variant).ok_or("missing avatar RMS variant")?;
            let squares = avatar_space_rms_errors(series, response)
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|value| value * value)
                .collect::<Vec<_>>();
            let rms = (squares.iter().sum::<f64>() / squares.len().max(1) as f64).sqrt();
            variants.insert(variant_name(variant).to_owned(), json!(rms));
        }
        models.push(json!({
            "model_sha256": response.model_sha256,
            "bound_non_tongue_channels": response.channels.len(),
            "variant_rms": variants,
        }));
    }
    Ok(json!(models))
}

fn write_frame_csv(
    path: &Path,
    frames: &BTreeMap<GnmResearchVariant, Vec<VariantFrame>>,
) -> Result<(), String> {
    let mut csv = String::from(
        "take_id,frame_seq,timestamp_micros,variant,channel,teacher,output,absolute_error\n",
    );
    for variant in GnmResearchVariant::ALL {
        for frame in frames.get(&variant).ok_or("missing CSV variant")? {
            for channel in ArkitBlendshape::ALL
                .into_iter()
                .take(ARKIT_NON_TONGUE_CHANNEL_COUNT)
            {
                let teacher = frame.teacher.get(channel);
                let output = frame.output.get(channel);
                csv.push_str(&format!(
                    "{},{},{},{},{},{teacher},{output},{}\n",
                    frame.take_id,
                    frame.frame_seq,
                    frame.timestamp_micros,
                    variant_name(variant),
                    channel.canonical_name(),
                    (output - teacher).abs(),
                ));
            }
        }
    }
    fs::write(path, csv).map_err(|error| format!("write {}: {error}", path.display()))
}

fn optional_le(left: Option<f64>, right: Option<f64>) -> bool {
    left.zip(right).is_some_and(|(left, right)| left <= right)
}

fn markdown_report(
    selected: Candidate,
    aggregate: &serde_json::Map<String, serde_json::Value>,
    criteria: &[(&str, bool)],
    common_frames: usize,
) -> Result<String, String> {
    let mut text = format!(
        "# Unified GNM ablation\n\nSelected: rank {}, history {}, ridge {}. Exact common frames: {}.\n\n| Variant | Macro MAE | Macro RMSE | Missed blinks | Peak error ms |\n|---|---:|---:|---:|---:|\n",
        selected.rank, selected.history_len, selected.ridge_lambda, common_frames
    );
    for variant in GnmResearchVariant::ALL {
        let name = variant_name(variant);
        let value = aggregate
            .get(name)
            .ok_or_else(|| format!("missing {name} aggregate"))?;
        text.push_str(&format!(
            "| {name} | {:.6} | {:.6} | {} | {} |\n",
            value
                .pointer("/metrics/macro_mae")
                .and_then(serde_json::Value::as_f64)
                .ok_or("missing macro_mae")?,
            value
                .pointer("/metrics/macro_rmse")
                .and_then(serde_json::Value::as_f64)
                .ok_or("missing macro_rmse")?,
            value
                .pointer("/blink/missed_events")
                .and_then(serde_json::Value::as_u64)
                .ok_or("missing missed blinks")?,
            value
                .pointer("/blink/median_absolute_peak_error_ms")
                .and_then(serde_json::Value::as_f64)
                .map_or_else(|| "n/a".to_owned(), |number| format!("{number:.3}")),
        ));
    }
    text.push_str("\n## Split criteria\n\n");
    for (name, pass) in criteria {
        text.push_str(&format!(
            "- {}: {}\n",
            if *pass { "PASS" } else { "FAIL" },
            name
        ));
    }
    Ok(text)
}

/// Runs one outer split of the unified offline ablation.
///
/// # Errors
///
/// Returns a descriptive error for invalid splits, artifacts, traces, VRMs,
/// metrics, or output I/O.
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let traces = load_traces(&options)?;
    fs::create_dir_all(&options.output)
        .map_err(|error| format!("create {}: {error}", options.output.display()))?;
    let (candidates, models) = if let Some(directory) = &options.reuse_fit {
        load_fitted_models(directory, &options.outer_train_takes)?
    } else {
        let candidates = select_candidate(
            &traces,
            &options.outer_train_takes,
            options.observable_rank,
            &options.gnm_model,
        )?;
        let selected = candidates
            .first()
            .ok_or("candidate grid produced no result")?
            .candidate;
        let models = fit_final_models(
            &traces,
            &options.outer_train_takes,
            selected,
            options.observable_rank,
            &options.gnm_model,
        )?;
        (candidates, models)
    };
    let selected = candidates
        .first()
        .ok_or("candidate grid produced no result")?
        .candidate;
    write_json(
        &options.output.join("observable-basis.json"),
        &models.observable,
    )?;
    write_json(&options.output.join("gnm-basis.json"), &models.gnm_basis)?;
    write_json(
        &options.output.join("landmark-basis.json"),
        &models.landmark_basis,
    )?;
    write_json(&options.output.join("g1-decoder.json"), &models.g1)?;
    write_json(&options.output.join("h-decoder.json"), &models.h)?;
    write_json(&options.output.join("l-decoder.json"), &models.l)?;
    write_json(&options.output.join("hl-decoder.json"), &models.hl)?;
    write_json(&options.output.join("candidate-grid.json"), &candidates)?;

    let mut all_frames: BTreeMap<GnmResearchVariant, Vec<VariantFrame>> = BTreeMap::new();
    let mut availability: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut per_take = Vec::new();
    for trace in traces
        .iter()
        .filter(|trace| options.outer_eval_takes.contains(&trace.take_id))
    {
        let variants = final_variant_frames(trace, &models)?;
        let common = exact_common_variant_frames(&variants).map_err(|error| error.to_string())?;
        let direct = common
            .get(&GnmResearchVariant::Direct)
            .ok_or("missing common D")?;
        let mut take_metrics = serde_json::Map::new();
        for variant in GnmResearchVariant::ALL {
            let common_frames = common.get(&variant).ok_or("missing common variant")?;
            let metrics =
                evaluate_non_tongue_variant(common_frames).map_err(|error| error.to_string())?;
            let blink = evaluate_blink_events(common_frames).map_err(|error| error.to_string())?;
            let delta = if variant == GnmResearchVariant::Direct {
                0.0
            } else {
                paired_absolute_error_delta(direct, common_frames)
                    .map_err(|error| error.to_string())?
            };
            take_metrics.insert(
                variant_name(variant).to_owned(),
                json!({
                    "metrics": metrics_json(&metrics),
                    "blink": blink_json(&blink),
                    "paired_absolute_error_delta_from_d": delta,
                }),
            );
            all_frames
                .entry(variant)
                .or_default()
                .extend(common_frames.iter().cloned());
            availability.insert(
                variant_name(variant),
                availability
                    .get(variant_name(variant))
                    .copied()
                    .unwrap_or(0)
                    + variants.get(&variant).map_or(0, Vec::len),
            );
        }
        per_take.push(json!({
            "take_id": trace.take_id,
            "exact_paired_teacher_frames": trace.samples.iter().filter(|sample| sample.teacher.is_some()).count(),
            "common_frames": direct.len(),
            "availability": {
                "paired": trace.paired,
                "solved": trace.expected_solved,
                "no_face": trace.no_face,
                "observation_insufficient": trace.observation_insufficient,
                "fit_rejected": trace.fit_rejected,
                "excluded_unpaired_teacher": trace.excluded_unpaired_teacher,
                "excluded_unpaired_rgb": trace.excluded_unpaired_rgb,
                "history_or_gap_excluded": trace.expected_solved.saturating_sub(direct.len()),
            },
            "variants": take_metrics,
        }));
    }
    let common = exact_common_variant_frames(&all_frames).map_err(|error| error.to_string())?;
    let direct = common
        .get(&GnmResearchVariant::Direct)
        .ok_or("missing aggregate D")?;
    let mut aggregate = serde_json::Map::new();
    for variant in GnmResearchVariant::ALL {
        let frames = common.get(&variant).ok_or("missing aggregate variant")?;
        let metrics = evaluate_non_tongue_variant(frames).map_err(|error| error.to_string())?;
        let blink = evaluate_blink_events(frames).map_err(|error| error.to_string())?;
        let delta = if variant == GnmResearchVariant::Direct {
            0.0
        } else {
            paired_absolute_error_delta(direct, frames).map_err(|error| error.to_string())?
        };
        aggregate.insert(
            variant_name(variant).to_owned(),
            json!({
                "metrics": metrics_json(&metrics),
                "blink": blink_json(&blink),
                "paired_absolute_error_delta_from_d": delta,
            }),
        );
    }
    write_frame_csv(&options.output.join("framewise-errors.csv"), &common)?;
    let morph_responses = options
        .vrms
        .iter()
        .map(|path| load_perfect_sync_morph(path))
        .collect::<Result<Vec<_>, _>>()?;
    let avatar_rms = avatar_rms_json(&common, &morph_responses)?;
    let h = aggregate.get("H").ok_or("missing H metrics")?;
    let d = aggregate.get("D").ok_or("missing D metrics")?;
    let l = aggregate.get("L").ok_or("missing L metrics")?;
    let value = |entry: &serde_json::Value, field: &str| {
        entry
            .pointer(&format!("/metrics/{field}"))
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("missing metric {field}"))
    };
    let blink_value = |entry: &serde_json::Value, field: &str| {
        entry
            .pointer(&format!("/blink/{field}"))
            .and_then(serde_json::Value::as_f64)
    };
    let missed = |entry: &serde_json::Value| {
        entry
            .pointer("/blink/missed_events")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| "missing blink.missed_events".to_owned())
    };
    let criteria = [
        (
            "H macro MAE and RMSE are below D",
            value(h, "macro_mae")? < value(d, "macro_mae")?
                && value(h, "macro_rmse")? < value(d, "macro_rmse")?,
        ),
        (
            "H macro MAE and RMSE are below L",
            value(h, "macro_mae")? < value(l, "macro_mae")?
                && value(h, "macro_rmse")? < value(l, "macro_rmse")?,
        ),
        (
            "H missed blinks are no greater than D",
            missed(h)? <= missed(d)?,
        ),
        (
            "H median absolute onset and peak timing errors are no greater than D",
            optional_le(
                blink_value(h, "median_absolute_onset_error_ms"),
                blink_value(d, "median_absolute_onset_error_ms"),
            ) && optional_le(
                blink_value(h, "median_absolute_peak_error_ms"),
                blink_value(d, "median_absolute_peak_error_ms"),
            ),
        ),
        (
            "H median absolute peak attenuation is no greater than D",
            optional_le(
                blink_value(h, "median_absolute_peak_attenuation"),
                blink_value(d, "median_absolute_peak_attenuation"),
            ),
        ),
        (
            "availability is numeric and exact-common evaluation uses no fill",
            availability.values().all(|count| *count > 0) && !direct.is_empty(),
        ),
    ];
    let split_success = criteria.iter().all(|(_, pass)| *pass);
    let evaluation_traces = traces
        .iter()
        .filter(|trace| options.outer_eval_takes.contains(&trace.take_id))
        .collect::<Vec<_>>();
    let training_traces = traces
        .iter()
        .filter(|trace| options.outer_train_takes.contains(&trace.take_id))
        .collect::<Vec<_>>();
    let report_markdown = markdown_report(selected, &aggregate, &criteria, direct.len())?;
    fs::write(options.output.join("report.md"), report_markdown)
        .map_err(|error| format!("write report.md: {error}"))?;
    let report = json!({
        "schema_version": 1,
        "tool": "teacher-unified-gnm-ablation",
        "outer_train_takes": options.outer_train_takes,
        "outer_eval_takes": options.outer_eval_takes,
        "dataset_hashes": {
            "outer_train": dataset_hash(&training_traces),
            "outer_eval": dataset_hash(&evaluation_traces),
            "trace_sha256_by_take": traces.iter().map(|trace| (&trace.take_id, &trace.trace_sha256)).collect::<BTreeMap<_, _>>(),
        },
        "candidate_grid": candidates,
        "selected": selected,
        "artifact_hashes": {
            "observable": models.observable.content_hash,
            "gnm_basis": models.gnm_basis.content_hash,
            "landmark_basis": models.landmark_basis.content_hash,
            "g1": models.g1.content_hash,
            "h": models.h.content_hash,
            "l": models.l.content_hash,
            "hl": models.hl.content_hash,
        },
        "availability_prediction_frames": availability,
        "common_intersection_frames": direct.len(),
        "aggregate": aggregate,
        "per_take": per_take,
        "avatar_morph_space": avatar_rms,
        "split_success_criteria": criteria.iter().map(|(name, pass)| json!({"criterion": name, "pass": pass})).collect::<Vec<_>>(),
        "split_success": split_success,
        "cross_person_delta_requirement": "Compare H-D macro deltas from the A-heldout and B-heldout split reports.",
        "limitations": [
            "Two people recorded on one iPhone model in the documented room; no broader generalization claim.",
            "The local VRM morph-space metric covers recognized non-tongue ARKit-named morph binds only."
        ]
    });
    write_json(&options.output.join("summary.json"), &report)?;
    println!("teacher-unified-gnm-ablation:");
    println!("  selected: {selected:?}");
    println!("  common frames: {}", direct.len());
    println!("  split conditions pass: {split_success}");
    println!("  output: {}", options.output.display());
    Ok(())
}
