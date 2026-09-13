//! Eye-closure data preparation and threshold fitting (Issues #51-#54).
//!
//! Subcommand family under `cargo xtask -- eye-closure ...`. Raw pixels,
//! derived traces, review images, and absolute paths stay in gitignored local
//! working directories; only synthetic fixtures are committed.

mod decode;
mod input;
mod replay;

use std::path::PathBuf;

const HELP_TEXT: &str = "\
eye-closure - prepare eye-closure analysis data (Issues #51-#54)

USAGE:
  cargo run -p xtask --release -- eye-closure inspect --inputs <inputs.json> --output <inventory.json>
  cargo run -p xtask --release -- eye-closure extract --inputs <inputs.json> --output <dir>

OPTIONS:
  --inputs <path>          Versioned explicit input list
  --output <path>          Output file (inspect) or directory (extract)
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
        "inspect" => replay::run_inspect(&Options::parse(rest)?),
        "extract" => replay::run_extract(&Options::parse(rest)?),
        other => Err(format!(
            "unknown eye-closure command: {other}\n\n{HELP_TEXT}"
        )),
    }
}

/// Common options shared by the `inspect` and `extract` commands.
pub(crate) struct Options {
    pub(crate) inputs: PathBuf,
    pub(crate) output: PathBuf,
    pub(crate) project_root: PathBuf,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut inputs = None;
        let mut output = None;
        let mut project_root = None;
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            let value = |iter: &mut std::slice::Iter<'_, String>| {
                iter.next()
                    .cloned()
                    .ok_or_else(|| format!("{arg} requires a value"))
            };
            match arg.as_str() {
                "--inputs" => inputs = Some(value(&mut iter)?),
                "--output" => output = Some(value(&mut iter)?),
                "--project-root" => project_root = Some(value(&mut iter)?),
                other => return Err(format!("unknown option: {other}\n\n{HELP_TEXT}")),
            }
        }
        Ok(Self {
            inputs: inputs.ok_or("missing required option --inputs")?.into(),
            output: output.ok_or("missing required option --output")?.into(),
            project_root: project_root.unwrap_or_else(|| ".".into()).into(),
        })
    }
}
