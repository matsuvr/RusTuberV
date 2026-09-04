//! Fits the same-frame teacher-minus-Direct residual decoder from derived traces.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use sha2::{Digest, Sha256};
use vtuber_tracking::{
    LinearPriorTrainingConfig, TEACHER_RESIDUAL_FEATURE_ORDER, TeacherResidualFeatureConfig,
};

use crate::teacher_fit_prior::load_trace;

struct Options {
    traces: Vec<PathBuf>,
    output: PathBuf,
    train_takes: BTreeSet<String>,
    history_len: usize,
    max_gap_micros: u64,
    ridge_lambda: f32,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut traces = Vec::new();
        let mut output = PathBuf::from("data/datasets/teacher-residual.json");
        let mut train_takes = BTreeSet::new();
        let mut history_len = 4;
        let mut max_gap_micros = 100_000;
        let mut ridge_lambda = 1.0e-3;
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
                "--trace" => traces.push(PathBuf::from(next(&mut index, "--trace")?)),
                "--output" => output = PathBuf::from(next(&mut index, "--output")?),
                "--train-take" => {
                    train_takes.insert(next(&mut index, "--train-take")?);
                }
                "--history-len" => {
                    history_len = next(&mut index, "--history-len")?
                        .parse()
                        .map_err(|_| "--history-len must be a positive integer")?;
                }
                "--max-gap-micros" => {
                    max_gap_micros = next(&mut index, "--max-gap-micros")?
                        .parse()
                        .map_err(|_| "--max-gap-micros must be an integer")?;
                }
                "--ridge-lambda" => {
                    ridge_lambda = next(&mut index, "--ridge-lambda")?
                        .parse()
                        .map_err(|_| "--ridge-lambda must be a float")?;
                }
                other => return Err(format!("unknown option {other}")),
            }
            index += 1;
        }
        if traces.is_empty() {
            return Err("at least one --trace <replay-output-dir> is required".to_owned());
        }
        if train_takes.is_empty() {
            return Err("at least one --train-take <take-id> is required".to_owned());
        }
        Ok(Self {
            traces,
            output,
            train_takes,
            history_len,
            max_gap_micros,
            ridge_lambda,
        })
    }
}

/// Prints command help.
pub fn print_help() {
    println!(
        "  teacher-fit-residual --trace <replay-output-dir> [--trace <dir> ...]\n\
         *                       --train-take <id> [--train-take <id> ...]\n\
         *                       [--output <artifact.json>] [--history-len 4]\n\
         *                       [--max-gap-micros 100000] [--ridge-lambda 0.001]\n\
         *   Fits the same-frame non-tongue 51-channel teacher-minus-Direct decoder."
    );
}

/// Runs residual dataset construction, fit, verification, and export.
///
/// # Errors
///
/// Returns an error for invalid traces, split selection, fitting, or file I/O.
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let feature_config = TeacherResidualFeatureConfig {
        history_len: options.history_len,
        max_gap_micros: options.max_gap_micros,
    };
    let mut rows = Vec::new();
    let mut exclusions = BTreeMap::<String, usize>::new();
    let mut take_summary = Vec::new();
    for directory in &options.traces {
        let trace = load_trace(directory)?;
        let dataset = vtuber_tracking::build_teacher_residual_rows(
            &trace.take_id,
            &trace.samples,
            feature_config,
        )
        .map_err(|error| format!("take {}: {error:?}", trace.take_id))?;
        for (reason, count) in dataset.exclusions {
            *exclusions.entry(format!("{reason:?}")).or_default() += count;
        }
        take_summary.push(format!(
            "{}: {} frames, {} rows ({} solved states)",
            trace.take_id,
            trace.samples.len(),
            dataset.rows.len(),
            trace.expected_solved
        ));
        rows.extend(dataset.rows);
    }
    if !options
        .train_takes
        .iter()
        .all(|take| rows.iter().any(|row| &row.take_id == take))
    {
        return Err(format!(
            "--train-take lists a take with no rows: {:?}",
            options.train_takes
        ));
    }

    let training_config = LinearPriorTrainingConfig {
        ridge_lambda: options.ridge_lambda,
        seed: 0,
        pivot_epsilon: 1.0e-8,
    };
    let artifact = vtuber_tracking::fit_teacher_residual_decoder(
        &rows,
        &options.train_takes,
        feature_config,
        training_config,
        TEACHER_RESIDUAL_FEATURE_ORDER,
    )
    .map_err(|error| format!("fit: {error:?}"))?;
    let verification_row = rows
        .iter()
        .find(|row| options.train_takes.contains(&row.take_id))
        .ok_or_else(|| "training selection produced no verification row".to_owned())?;
    let _ = vtuber_tracking::predict_teacher_residual(&artifact, &verification_row.features)
        .map_err(|error| format!("exported artifact failed verification: {error:?}"))?;

    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&artifact)
        .map_err(|error| format!("encode artifact: {error}"))?;
    fs::write(&options.output, json.as_bytes())
        .map_err(|error| format!("write {}: {error}", options.output.display()))?;
    let sha256 = Sha256::digest(json.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect::<String>();

    println!("teacher-fit-residual:");
    for summary in take_summary {
        println!("  take: {summary}");
    }
    for (reason, count) in exclusions {
        println!("  excluded {reason}: {count}");
    }
    println!("  training takes: {:?}", options.train_takes);
    println!("  rows: {}", rows.len());
    println!("  feature_width: {}", feature_config.feature_width());
    println!("  feature_order: {TEACHER_RESIDUAL_FEATURE_ORDER}");
    println!("  ridge_lambda: {}", options.ridge_lambda);
    println!("  artifact: {}", options.output.display());
    println!("  content_hash: {}", artifact.content_hash);
    println!("  artifact_sha256: {sha256}");
    Ok(())
}
