//! Fits a teacher-residual-aligned subspace of an observable GNM basis.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use vtuber_tracking::{
    ObservableGnmBasisArtifact, build_teacher_alignment_samples, fit_teacher_aligned_gnm_basis,
    project_teacher_aligned_expression,
};

use crate::teacher_fit_prior::load_trace;

pub struct Options {
    observable_basis: PathBuf,
    traces: Vec<PathBuf>,
    train_takes: BTreeSet<String>,
    rank: usize,
    output: PathBuf,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut observable_basis = None;
        let mut traces = Vec::new();
        let mut train_takes = BTreeSet::new();
        let mut rank = None;
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
                "--observable-basis" => {
                    observable_basis = Some(PathBuf::from(next(&mut index, "--observable-basis")?));
                }
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
                other => return Err(format!("unknown option {other}")),
            }
            index += 1;
        }
        if traces.is_empty() || train_takes.is_empty() {
            return Err("--trace and --train-take are both required".to_owned());
        }
        Ok(Self {
            observable_basis: observable_basis.ok_or("--observable-basis <json> is required")?,
            traces,
            train_takes,
            rank: rank.ok_or("--rank <n> is required")?,
            output: output.ok_or("--output <artifact.json> is required")?,
        })
    }
}

pub fn print_help() {
    println!(
        "  teacher-fit-aligned-basis --observable-basis <artifact.json>\n\
         *       --trace <trace-v2-dir> [...] --train-take <take-id> [...]\n\
         *       --rank <1..=51> --output <artifact.json>\n\
         *   Fits B = O U_k from training-only teacher-minus-Direct residuals."
    );
}

/// Runs the trace-v2 teacher-aligned-basis fit.
///
/// # Errors
///
/// Returns a descriptive failure for invalid options, incompatible artifacts,
/// trace validation, numeric fitting, or output I/O.
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let observable_text = fs::read_to_string(&options.observable_basis)
        .map_err(|error| format!("read {}: {error}", options.observable_basis.display()))?;
    let observable: ObservableGnmBasisArtifact = serde_json::from_str(&observable_text)
        .map_err(|error| format!("parse {}: {error}", options.observable_basis.display()))?;
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
    let mut samples = Vec::new();
    for trace in &traces {
        if trace.model_sha256 != observable.model_sha256
            || trace.mapping_schema_revision != observable.mapping_schema_revision
        {
            return Err(format!(
                "take {} does not match the observable basis model or mapping",
                trace.take_id
            ));
        }
        samples.extend(
            build_teacher_alignment_samples(&trace.take_id, &trace.samples)
                .map_err(|error| format!("take {}: {error:?}", trace.take_id))?,
        );
    }
    let artifact =
        fit_teacher_aligned_gnm_basis(&observable, &samples, &options.train_takes, options.rank)
            .map_err(|error| error.to_string())?;
    let neutral = vtuber_gnm::GnmNonTongueExpression::try_from_values(vec![0.0; 351])
        .map_err(|error| error.to_string())?;
    let _ = project_teacher_aligned_expression(&neutral, &artifact)
        .map_err(|error| format!("verify artifact: {error}"))?;
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&artifact)
        .map_err(|error| format!("encode artifact: {error}"))?;
    fs::write(&options.output, json.as_bytes())
        .map_err(|error| format!("write {}: {error}", options.output.display()))?;

    println!("teacher-fit-aligned-basis:");
    println!("  training takes: {:?}", artifact.training_takes);
    println!(
        "  samples: {}",
        samples
            .iter()
            .filter(|sample| options.train_takes.contains(&sample.take_id))
            .count()
    );
    println!("  source rank: {}", artifact.source_rank);
    println!("  rank: {}", artifact.rank);
    println!(
        "  inactive residual channels: {:?}",
        artifact.inactive_residual_channels
    );
    let top_singular_values: Vec<f64> = artifact
        .singular_values_descending
        .iter()
        .take(8)
        .copied()
        .collect();
    println!("  top singular values: {top_singular_values:?}");
    println!("  content_hash: {}", artifact.content_hash);
    println!("  artifact: {}", options.output.display());
    Ok(())
}
