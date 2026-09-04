//! Fits a teacher-residual-aligned MediaPipe landmark control basis.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use vtuber_tracking::{
    build_landmark_alignment_samples, fit_landmark_aligned_basis, project_landmark_latent,
};

use crate::teacher_fit_prior::load_trace;

struct Options {
    traces: Vec<PathBuf>,
    train_takes: BTreeSet<String>,
    rank: usize,
    output: PathBuf,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
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
            traces,
            train_takes,
            rank: rank.ok_or("--rank <same-as-gnm> is required")?,
            output: output.ok_or("--output <artifact.json> is required")?,
        })
    }
}

pub fn print_help() {
    println!(
        "  teacher-fit-landmark-basis --trace <trace-v2-dir> [...]\n\
         *       --train-take <take-id> [...] --rank <same-as-gnm>\n\
         *       --output <artifact.json>"
    );
}

/// Runs landmark-basis fitting and artifact I/O.
///
/// # Errors
///
/// Returns a descriptive option, trace, fit, verification, or I/O failure.
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
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
        samples.extend(
            build_landmark_alignment_samples(&trace.take_id, &trace.samples)
                .map_err(|error| format!("take {}: {error}", trace.take_id))?,
        );
    }
    let artifact = fit_landmark_aligned_basis(&samples, &options.train_takes, options.rank)
        .map_err(|error| error.to_string())?;
    let neutral = [0.0_f32; 956];
    let _ = project_landmark_latent(&neutral, &artifact)
        .map_err(|error| format!("verify artifact: {error}"))?;
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&artifact)
        .map_err(|error| format!("encode artifact: {error}"))?;
    fs::write(&options.output, json.as_bytes())
        .map_err(|error| format!("write {}: {error}", options.output.display()))?;
    println!("teacher-fit-landmark-basis:");
    println!("  training takes: {:?}", artifact.training_takes);
    println!(
        "  samples: {}",
        samples
            .iter()
            .filter(|sample| options.train_takes.contains(&sample.take_id))
            .count()
    );
    println!("  rank: {}", artifact.rank);
    println!(
        "  inactive residual channels: {:?}",
        artifact.inactive_residual_channels
    );
    println!(
        "  top singular values: {:?}",
        artifact
            .singular_values_descending
            .iter()
            .take(8)
            .collect::<Vec<_>>()
    );
    println!("  content_hash: {}", artifact.content_hash);
    println!("  artifact: {}", options.output.display());
    Ok(())
}
