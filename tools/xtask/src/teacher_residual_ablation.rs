//! Take-disjoint D/G0/H0 evaluation for the teacher residual decoder.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};
use vtuber_core::{ARKIT_NON_TONGUE_CHANNEL_COUNT, arkit_non_tongue_values};
use vtuber_tracking::{
    LoadedTeacherResidualDecoder, TEACHER_RESIDUAL_FEATURE_ORDER, TeacherResidualDecoderArtifact,
    TeacherResidualFeatureConfig, TeacherResidualHistory, existing_trace_residual_variants,
};

use crate::teacher_ablation::{
    AlignedResidualFrame, ChannelErrorMetrics, EventTimingMetrics, ResidualMetricsSet,
    ResidualVariantMetrics, TemporalMetrics, score_residual_frames,
};
use crate::teacher_fit_prior::load_trace;

struct Options {
    artifact: PathBuf,
    eval_traces: Vec<PathBuf>,
    train_takes: BTreeSet<String>,
    person_count: u32,
    output: PathBuf,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut artifact = None;
        let mut eval_traces = Vec::new();
        let mut train_takes = BTreeSet::new();
        let mut person_count = None;
        let mut output = None;
        let mut index = 0;
        while index < args.len() {
            let next = |index: &mut usize, flag: &str| -> Result<String, String> {
                *index += 1;
                args.get(*index)
                    .cloned()
                    .ok_or_else(|| format!("{flag} requires a value"))
            };
            // Bounds are proven by the loop condition.
            #[allow(clippy::indexing_slicing)]
            match args[index].as_str() {
                "--artifact" => artifact = Some(PathBuf::from(next(&mut index, "--artifact")?)),
                "--eval-trace" => {
                    eval_traces.push(PathBuf::from(next(&mut index, "--eval-trace")?));
                }
                "--train-take" => {
                    train_takes.insert(next(&mut index, "--train-take")?);
                }
                "--person-count" => {
                    person_count = Some(
                        next(&mut index, "--person-count")?
                            .parse::<u32>()
                            .map_err(|_| "--person-count must be a positive integer")?,
                    );
                }
                "--output" => output = Some(PathBuf::from(next(&mut index, "--output")?)),
                other => return Err(format!("unknown option {other}")),
            }
            index += 1;
        }
        if eval_traces.is_empty() || train_takes.is_empty() {
            return Err("--eval-trace and --train-take are both required".to_owned());
        }
        let person_count = person_count.ok_or("--person-count <n> is required")?;
        if person_count == 0 {
            return Err("--person-count must be a positive integer".to_owned());
        }
        Ok(Self {
            artifact: artifact.ok_or("--artifact <artifact.json> is required")?,
            eval_traces,
            train_takes,
            person_count,
            output: output.ok_or("--output <directory> is required")?,
        })
    }
}

#[derive(Serialize)]
struct TakeReport {
    take_id: String,
    source_frames: usize,
    evaluated_frames: usize,
    metrics: ResidualMetricsSet,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    artifact_sha256: String,
    artifact_content_hash: u64,
    train_takes: Vec<String>,
    eval_takes: Vec<String>,
    training_person_count: u32,
    target: &'static str,
    feature_order: &'static str,
    takes: Vec<TakeReport>,
    overall: ResidualMetricsSet,
    hybrid_improved_channels: Vec<&'static str>,
    hybrid_worsened_channels: Vec<&'static str>,
    constraints: Vec<&'static str>,
}

/// Prints command help.
pub fn print_help() {
    println!(
        "  teacher-residual-ablation --artifact <artifact.json> --eval-trace <dir> [...]\n\
         *                            --train-take <id> [...] --person-count <n>\n\
         *                            --output <directory>\n\
         *   Scores D / G0 / H0 on identical held-out non-tongue frames."
    );
}

/// Runs the take-disjoint residual ablation and writes JSON.
///
/// # Errors
///
/// Returns an error for invalid artifacts, overlapping splits, traces, or I/O.
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let artifact_text = fs::read_to_string(&options.artifact)
        .map_err(|error| format!("read {}: {error}", options.artifact.display()))?;
    let artifact: TeacherResidualDecoderArtifact = serde_json::from_str(&artifact_text)
        .map_err(|error| format!("parse {}: {error}", options.artifact.display()))?;
    let artifact_takes: BTreeSet<String> = artifact.training_takes.iter().cloned().collect();
    if artifact_takes != options.train_takes {
        return Err("--train-take set does not match artifact training_takes".to_owned());
    }
    let decoder =
        LoadedTeacherResidualDecoder::load(artifact.clone(), TEACHER_RESIDUAL_FEATURE_ORDER)
            .map_err(|error| format!("load artifact: {error:?}"))?;
    let feature_config = TeacherResidualFeatureConfig {
        history_len: artifact.history_len,
        max_gap_micros: artifact.max_gap_micros,
    };

    let mut take_reports = Vec::new();
    for directory in &options.eval_traces {
        let trace = load_trace(directory)?;
        if options.train_takes.contains(&trace.take_id) {
            return Err(format!(
                "take {} appears in both train and eval",
                trace.take_id
            ));
        }
        let history = TeacherResidualHistory::build(&trace.take_id, &trace.samples, feature_config)
            .map_err(|error| format!("take {}: history: {error:?}", trace.take_id))?;
        let mut frames = Vec::new();
        for sample in &trace.samples {
            let Some(teacher) = sample.teacher.as_ref() else {
                continue;
            };
            let Ok(variants) = existing_trace_residual_variants(sample, &history, &decoder) else {
                continue;
            };
            frames.push(AlignedResidualFrame {
                timestamp_micros: sample.timestamp_micros,
                teacher: to_f64(arkit_non_tongue_values(&teacher.coefficients)),
                direct: to_f64(arkit_non_tongue_values(&variants.direct)),
                gnm: to_f64(arkit_non_tongue_values(&variants.gnm_projected)),
                hybrid: to_f64(arkit_non_tongue_values(&variants.hybrid_projected_residual)),
            });
        }
        let metrics = score_residual_frames(&frames)?;
        println!(
            "  {}: {} frames | D {:.5} | G0 {:.5} | H0 {:.5}",
            trace.take_id,
            frames.len(),
            metrics.direct.micro_mae,
            metrics.gnm_projected.micro_mae,
            metrics.hybrid_projected_residual.micro_mae
        );
        take_reports.push(TakeReport {
            take_id: trace.take_id,
            source_frames: trace.samples.len(),
            evaluated_frames: frames.len(),
            metrics,
        });
    }
    let overall = aggregate_take_metrics(&take_reports);
    let mut improved = Vec::new();
    let mut worsened = Vec::new();
    for (direct, hybrid) in overall
        .direct
        .channels
        .iter()
        .zip(&overall.hybrid_projected_residual.channels)
    {
        if hybrid.mae < direct.mae {
            improved.push(direct.channel);
        } else if hybrid.mae > direct.mae {
            worsened.push(direct.channel);
        }
    }
    let artifact_sha256 = Sha256::digest(artifact_text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect();
    let report = Report {
        schema_version: 1,
        tool: "xtask teacher-residual-ablation",
        artifact_sha256,
        artifact_content_hash: artifact.content_hash,
        train_takes: options.train_takes.into_iter().collect(),
        eval_takes: take_reports
            .iter()
            .map(|take| take.take_id.clone())
            .collect(),
        training_person_count: options.person_count,
        target: "same-frame teacher_51 - MediaPipeDirect_51",
        feature_order: TEACHER_RESIDUAL_FEATURE_ORDER,
        takes: take_reports,
        overall,
        hybrid_improved_channels: improved,
        hybrid_worsened_channels: worsened,
        constraints: vec![
            "offline evaluation only; production runtime unchanged",
            "TongueOut excluded from every metric and fixed to zero in H0",
            "smoothness is reported together with value and blink preservation",
        ],
    };
    fs::create_dir_all(&options.output)
        .map_err(|error| format!("create {}: {error}", options.output.display()))?;
    let path = options.output.join("residual-ablation-report.json");
    let json =
        serde_json::to_string_pretty(&report).map_err(|error| format!("encode report: {error}"))?;
    fs::write(&path, json.as_bytes())
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    println!(
        "  overall: D {:.5} | G0 {:.5} | H0 {:.5}",
        report.overall.direct.micro_mae,
        report.overall.gnm_projected.micro_mae,
        report.overall.hybrid_projected_residual.micro_mae
    );
    println!("report: {}", path.display());
    Ok(())
}

fn to_f64(values: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT]) -> [f64; ARKIT_NON_TONGUE_CHANNEL_COUNT] {
    values.map(f64::from)
}

fn aggregate_take_metrics(takes: &[TakeReport]) -> ResidualMetricsSet {
    ResidualMetricsSet {
        direct: aggregate_variant(takes, |take| &take.metrics.direct),
        gnm_projected: aggregate_variant(takes, |take| &take.metrics.gnm_projected),
        hybrid_projected_residual: aggregate_variant(takes, |take| {
            &take.metrics.hybrid_projected_residual
        }),
    }
}

#[allow(clippy::indexing_slicing)]
fn aggregate_variant(
    takes: &[TakeReport],
    pick: fn(&TakeReport) -> &ResidualVariantMetrics,
) -> ResidualVariantMetrics {
    let frames: u64 = takes.iter().map(|take| pick(take).frames).sum();
    let weight = frames.max(1) as f64;
    let weighted = |value: fn(&ResidualVariantMetrics) -> f64| {
        takes
            .iter()
            .map(|take| value(pick(take)) * pick(take).frames as f64)
            .sum::<f64>()
            / weight
    };
    let mut channels = Vec::with_capacity(ARKIT_NON_TONGUE_CHANNEL_COUNT);
    for index in 0..ARKIT_NON_TONGUE_CHANNEL_COUNT {
        let name = takes
            .first()
            .map_or("", |take| pick(take).channels[index].channel);
        let channel_weight = |value: fn(&ChannelErrorMetrics) -> f64| {
            takes
                .iter()
                .map(|take| value(&pick(take).channels[index]) * pick(take).frames as f64)
                .sum::<f64>()
                / weight
        };
        channels.push(ChannelErrorMetrics {
            channel: name,
            mae: channel_weight(|channel| channel.mae),
            rmse: channel_weight(|channel| channel.rmse * channel.rmse).sqrt(),
            ccc: channel_weight(|channel| channel.ccc),
        });
    }
    let macro_mae = channels.iter().map(|channel| channel.mae).sum::<f64>()
        / ARKIT_NON_TONGUE_CHANNEL_COUNT as f64;
    let macro_rmse = channels.iter().map(|channel| channel.rmse).sum::<f64>()
        / ARKIT_NON_TONGUE_CHANNEL_COUNT as f64;
    let pulse_events: u64 = takes
        .iter()
        .map(|take| pick(take).blink_events.pulse_events)
        .sum();
    let pulse_weight = pulse_events.max(1) as f64;
    ResidualVariantMetrics {
        frames,
        micro_mae: weighted(|metrics| metrics.micro_mae),
        micro_rmse: weighted(|metrics| metrics.micro_rmse * metrics.micro_rmse).sqrt(),
        macro_mae,
        macro_rmse,
        neutral_bias: weighted(|metrics| metrics.neutral_bias),
        left_right_difference_mae: weighted(|metrics| metrics.left_right_difference_mae),
        channels,
        temporal: TemporalMetrics {
            velocity_mae: weighted(|metrics| metrics.temporal.velocity_mae),
            acceleration_mae: weighted(|metrics| metrics.temporal.acceleration_mae),
            jerk_mae: weighted(|metrics| metrics.temporal.jerk_mae),
            variant_jitter_velocity_rms: weighted(|metrics| {
                metrics.temporal.variant_jitter_velocity_rms
                    * metrics.temporal.variant_jitter_velocity_rms
            })
            .sqrt(),
            teacher_jitter_velocity_rms: weighted(|metrics| {
                metrics.temporal.teacher_jitter_velocity_rms
                    * metrics.temporal.teacher_jitter_velocity_rms
            })
            .sqrt(),
        },
        blink_events: EventTimingMetrics {
            step_events: 0,
            mean_onset_delay_ms: 0.0,
            mean_rise_time_ms: 0.0,
            pulse_events,
            pulse_events_detected: takes
                .iter()
                .map(|take| pick(take).blink_events.pulse_events_detected)
                .sum(),
            mean_peak_attenuation: takes
                .iter()
                .map(|take| {
                    let events = pick(take).blink_events.pulse_events;
                    pick(take).blink_events.mean_peak_attenuation * events as f64
                })
                .sum::<f64>()
                / pulse_weight,
            mean_peak_timing_error_ms: takes
                .iter()
                .map(|take| {
                    let events = pick(take).blink_events.pulse_events;
                    pick(take).blink_events.mean_peak_timing_error_ms * events as f64
                })
                .sum::<f64>()
                / pulse_weight,
            events_unmeasurable: takes
                .iter()
                .map(|take| pick(take).blink_events.events_unmeasurable)
                .sum(),
        },
    }
}
