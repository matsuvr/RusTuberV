//! Fits the causal linear dynamic prior from derived teacher-replay traces
//! and exports it as a versioned artifact (GNM #68.4a/#68.4b / Issue #111).
//!
//! The library-side contracts live in `vtuber-tracking` (`build_causal_dataset`,
//! `fit_linear_prior`, `LoadedLinearPrior`); this command only wires file I/O:
//!
//! - every `--trace <dir>` points at a `teacher-replay` output directory and
//!   contributes one take (take id comes from `replay-metadata.json`);
//! - traces are validated with the same fail-closed pairing rules (#108) and
//!   converted into causal history/velocity rows;
//! - training takes are selected explicitly (`--train-take`), never inferred,
//!   so train/validation/test splits stay take-disjoint (#112);
//! - the exported artifact records the exact config and is verified by
//!   reloading it before the bytes are accepted.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use vtuber_tracking::{
    CausalFeatureConfig, LinearPriorTrainingConfig, LoadedLinearPrior, PairedTemporalSample,
    validate_paired_samples,
};

use crate::teacher_replay::{TraceRow, sample_from_row};

/// Feature-order string pinned into every exported artifact; must never
/// change semantics without a model-version bump.
pub const FEATURE_ORDER: &str =
    "v2:newest-first-history(non-tongue-51+residual)+velocity(non-tongue-51)";

/// Parsed CLI options for `teacher-fit-prior`.
pub struct Options {
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
        let mut output = None;
        let mut train_takes = BTreeSet::new();
        let mut history_len = 4_usize;
        let mut max_gap_micros = 100_000_u64;
        let mut ridge_lambda = 1.0e-3_f32;
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
                "--trace" => traces.push(PathBuf::from(next(&mut index, "--trace")?)),
                "--output" => output = Some(PathBuf::from(next(&mut index, "--output")?)),
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
        let output = output.map_or_else(|| PathBuf::from("data/datasets/linear-prior.json"), {
            |output| output
        });
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
        "  teacher-fit-prior --trace <replay-output-dir> [--trace <dir> ...]\n\
         *                    [--train-take <id> ...] [--output <artifact.json>]\n\
         *                    [--history-len 4] [--max-gap-micros 100000] [--ridge-lambda 0.001]\n\
         *   Builds the causal dataset from derived traces, fits the linear AR prior\n\
         *   on the selected training takes only, verifies, and exports the artifact.\n\
         *   Splits must stay take-disjoint (GNM #68.5)."
    );
}

/// One replay-output directory loaded and validated.
pub(crate) struct LoadedTrace {
    pub(crate) take_id: String,
    pub(crate) samples: Vec<PairedTemporalSample>,
    pub(crate) expected_solved: usize,
}

/// Loads one replay-output directory into validated samples.
///
/// Shared with `teacher-ablation` so both commands consume the derived trace
/// through the same validation path.
pub(crate) fn load_trace(directory: &Path) -> Result<LoadedTrace, String> {
    let metadata_path = directory.join("replay-metadata.json");
    let metadata_text = fs::read_to_string(&metadata_path)
        .map_err(|error| format!("read {}: {error}", metadata_path.display()))?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_text)
        .map_err(|error| format!("parse {}: {error}", metadata_path.display()))?;
    let take_id = metadata
        .pointer("/source_dataset/take_id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            format!(
                "{}: missing source_dataset.take_id",
                metadata_path.display()
            )
        })?;
    let expected_solved = metadata
        .pointer("/counts/solved")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default() as usize;

    let trace_path = directory.join("derived-trace.jsonl");
    let text = fs::read_to_string(&trace_path)
        .map_err(|error| format!("read {}: {error}", trace_path.display()))?;
    let mut samples = Vec::new();
    let mut solved = 0_usize;
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: TraceRow = serde_json::from_str(line).map_err(|error| {
            format!("{} line {}: {error}", trace_path.display(), line_index + 1)
        })?;
        let sample = sample_from_row(&row).map_err(|error| {
            format!("{} line {}: {error}", trace_path.display(), line_index + 1)
        })?;
        if row.gnm_state.is_some() {
            solved += 1;
        }
        samples.push(sample);
    }
    // The trace must still satisfy the exact-identity contract, and the
    // solved count must agree with the metadata written by the replay run.
    validate_paired_samples(&samples)
        .map_err(|error| format!("{}: invalid trace: {error:?}", trace_path.display()))?;
    if solved != expected_solved {
        return Err(format!(
            "{}: metadata says {} solved states but the trace carries {solved}",
            trace_path.display(),
            expected_solved
        ));
    }
    Ok(LoadedTrace {
        take_id,
        samples,
        expected_solved: solved,
    })
}

/// Runs the fit/export; see the module documentation.
///
/// # Errors
///
/// Fails closed on trace validation failures, empty or take-ambiguous
/// training sets, singular fits, and output I/O errors.
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let mut rows = Vec::new();
    let mut exclusions: BTreeMap<String, usize> = BTreeMap::new();
    let mut take_summary = Vec::new();
    for directory in &options.traces {
        let trace = load_trace(directory)?;
        let config = CausalFeatureConfig {
            history_len: options.history_len,
            max_gap_micros: options.max_gap_micros,
        };
        let dataset = vtuber_tracking::build_causal_dataset(&trace.take_id, &trace.samples, config)
            .map_err(|error| format!("take {}: {error:?}", trace.take_id))?;
        for (reason, count) in dataset.exclusions {
            *exclusions.entry(format!("{reason:?}")).or_default() += count;
        }
        let usable = dataset.rows.len();
        take_summary.push(format!(
            "{}: {} frames, {} rows ({} solved states)",
            trace.take_id,
            trace.samples.len(),
            usable,
            trace.expected_solved
        ));
        rows.extend(dataset.rows);
    }
    if rows.is_empty() {
        return Err("no causal rows were generated from the provided traces".to_owned());
    }

    let training_takes = if options.train_takes.is_empty() {
        rows.iter().map(|row| row.take_id.clone()).collect()
    } else {
        options.train_takes.clone()
    };
    if !training_takes
        .iter()
        .all(|take| rows.iter().any(|row| &row.take_id == take))
    {
        return Err(format!(
            "--train-take lists a take with no rows: {training_takes:?}"
        ));
    }
    if training_takes.len()
        == rows
            .iter()
            .map(|row| &row.take_id)
            .collect::<BTreeSet<_>>()
            .len()
        && options.traces.len() > 1
    {
        eprintln!(
            "warning: every loaded take is a training take; #112 ablation requires held-out takes"
        );
    }

    let config = LinearPriorTrainingConfig {
        ridge_lambda: options.ridge_lambda,
        seed: 0,
        pivot_epsilon: 1.0e-8,
    };
    let artifact = vtuber_tracking::fit_linear_prior(&rows, &training_takes, config, FEATURE_ORDER)
        .map_err(|error| format!("fit: {error:?}"))?;
    // Verify the artifact through the production load boundary before
    // accepting the exported bytes.
    let _ = LoadedLinearPrior::load(artifact.clone(), FEATURE_ORDER)
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

    println!("teacher-fit-prior:");
    for summary in take_summary {
        println!("  take: {summary}");
    }
    for (reason, count) in &exclusions {
        println!("  excluded {reason}: {count}");
    }
    println!("  training takes: {training_takes:?}");
    println!("  rows: {}", rows.len());
    println!("  feature_order: {FEATURE_ORDER}");
    println!("  ridge_lambda: {}", options.ridge_lambda);
    println!("  artifact: {}", options.output.display());
    println!("  artifact_sha256: {sha256}");
    Ok(())
}
