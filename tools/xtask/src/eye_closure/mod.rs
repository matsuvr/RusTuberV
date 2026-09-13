//! Eye-closure data preparation and threshold fitting (Issues #51-#54).
//!
//! Subcommand family under `cargo xtask -- eye-closure ...`. Raw pixels,
//! derived traces, review images, and absolute paths stay in gitignored local
//! working directories; only synthetic fixtures are committed.

mod decode;
mod evaluate;
mod fit;
mod input;
mod labels;
mod prepare;
mod replay;

use std::path::PathBuf;

const HELP_TEXT: &str = "\
eye-closure - prepare eye-closure analysis data (Issues #51-#54)

USAGE:
  cargo run -p xtask --release -- eye-closure inspect --inputs <inputs.json> --output <inventory.json>
  cargo run -p xtask --release -- eye-closure extract --inputs <inputs.json> --output <dir>
  cargo run -p xtask --release -- eye-closure prepare-labels --data <dir> --output <review-dir>
  cargo run -p xtask --release -- eye-closure fit --data <dir> --labels <csv> --split <json> --output <dir>
  cargo run -p xtask --release -- eye-closure evaluate --data <dir> --labels <csv> --split <json> --profile <json> --output <dir>

OPTIONS:
  --inputs <path>          Versioned explicit input list
  --data <path>            Extracted data directory
  --labels <path>          Label CSV
  --split <path>           Train/validation/test split JSON
  --profile <path>         Candidate profile JSON (evaluate)
  --output <path>          Output file (inspect) or directory
  --project-root <path>    Workspace root holding assets/models (default: .)
  -h, --help               Show this help";

/// Prints the `eye-closure` usage block.
pub fn print_help() {
    println!("{HELP_TEXT}");
}

/// Runs an `eye-closure` subcommand.
///
/// # Errors
///
/// Returns a message for unknown commands, malformed options, or any
/// input that fails the issue's fail-closed read/validation rules.
pub fn run(args: &[String]) -> Result<(), String> {
    let Some(command) = args.first() else {
        print_help();
        return Ok(());
    };
    if matches!(command.as_str(), "help" | "--help" | "-h") {
        print_help();
        return Ok(());
    }
    let rest = args.get(1..).unwrap_or_default();
    if rest
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        print_help();
        return Ok(());
    }
    match command.as_str() {
        "inspect" => replay::run_inspect(&Options::parse(rest, Args::Inspect)?),
        "extract" => replay::run_extract(&Options::parse(rest, Args::Extract)?),
        "prepare-labels" => prepare::run(&Options::parse(rest, Args::PrepareLabels)?),
        "fit" => fit::run_fit(&Options::parse(rest, Args::Fit)?),
        "evaluate" => evaluate::run(&Options::parse(rest, Args::Evaluate)?),
        other => Err(format!(
            "unknown eye-closure command: {other}\n\n{HELP_TEXT}"
        )),
    }
}

/// The set of options expected by one subcommand.
pub(crate) enum Args {
    Inspect,
    Extract,
    PrepareLabels,
    Fit,
    Evaluate,
}

/// Common options shared across the `eye-closure` commands.
pub(crate) struct Options {
    pub(crate) inputs: Option<PathBuf>,
    pub(crate) data: Option<PathBuf>,
    pub(crate) labels: Option<PathBuf>,
    pub(crate) split: Option<PathBuf>,
    pub(crate) profile: Option<PathBuf>,
    pub(crate) output: PathBuf,
    pub(crate) project_root: PathBuf,
}

impl Options {
    fn parse(args: &[String], kind: Args) -> Result<Self, String> {
        let mut inputs = None;
        let mut data = None;
        let mut labels = None;
        let mut split = None;
        let mut profile = None;
        let mut output = None;
        let mut project_root = None;
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--inputs" => inputs = Some(next_value(&mut iter, arg)?),
                "--data" => data = Some(next_value(&mut iter, arg)?),
                "--labels" => labels = Some(next_value(&mut iter, arg)?),
                "--split" => split = Some(next_value(&mut iter, arg)?),
                "--profile" => profile = Some(next_value(&mut iter, arg)?),
                "--output" => output = Some(next_value(&mut iter, arg)?),
                "--project-root" => project_root = Some(next_value(&mut iter, arg)?),
                other => return Err(format!("unknown option: {other}\n\n{HELP_TEXT}")),
            }
        }
        let output = output.ok_or("missing required option --output")?.into();
        let options = Self {
            inputs: inputs.map(Into::into),
            data: data.map(Into::into),
            labels: labels.map(Into::into),
            split: split.map(Into::into),
            profile: profile.map(Into::into),
            output,
            project_root: project_root.unwrap_or_else(|| ".".into()).into(),
        };
        match kind {
            Args::Inspect | Args::Extract => {
                options
                    .inputs
                    .as_ref()
                    .ok_or("missing required option --inputs")?;
            }
            Args::PrepareLabels | Args::Fit => {
                options
                    .data
                    .as_ref()
                    .ok_or("missing required option --data")?;
            }
            Args::Evaluate => {
                options
                    .data
                    .as_ref()
                    .ok_or("missing required option --data")?;
                options
                    .labels
                    .as_ref()
                    .ok_or("missing required option --labels")?;
                options
                    .split
                    .as_ref()
                    .ok_or("missing required option --split")?;
                options
                    .profile
                    .as_ref()
                    .ok_or("missing required option --profile")?;
            }
        }
        if matches!(kind, Args::Fit) {
            options
                .labels
                .as_ref()
                .ok_or("missing required option --labels")?;
            options
                .split
                .as_ref()
                .ok_or("missing required option --split")?;
        }
        Ok(options)
    }
}

/// SHA-256 of a byte slice, uppercase hex.
pub(crate) fn hash_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:X}", hasher.finalize())
}

fn next_value(iter: &mut std::slice::Iter<'_, String>, key: &str) -> Result<String, String> {
    iter.next()
        .cloned()
        .ok_or_else(|| format!("{key} requires a value"))
}
