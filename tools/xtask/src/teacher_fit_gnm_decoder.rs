//! Fits offline G1/H semantic decoders from teacher-aligned reduced GNM traces.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use vtuber_tracking::{
    GnmSemanticDecoderKind, GnmSemanticFeatureConfig, LinearPriorTrainingConfig,
    TeacherAlignedGnmBasisArtifact, build_gnm_semantic_rows, fit_gnm_semantic_decoder,
    gnm_semantic_feature_order, predict_gnm_semantic_raw,
};

use crate::teacher_fit_prior::load_trace;

pub struct Options {
    kind: GnmSemanticDecoderKind,
    basis: PathBuf,
    traces: Vec<PathBuf>,
    train_takes: BTreeSet<String>,
    history_len: usize,
    max_gap_micros: u64,
    ridge_lambda: f32,
    output: PathBuf,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut kind = None;
        let mut basis = None;
        let mut traces = Vec::new();
        let mut train_takes = BTreeSet::new();
        let mut history_len = None;
        let mut max_gap_micros = None;
        let mut ridge_lambda = None;
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
                "--kind" => {
                    kind = Some(match next(&mut index, "--kind")?.as_str() {
                        "gnm-only" => GnmSemanticDecoderKind::GnmOnly,
                        "hybrid-residual" => GnmSemanticDecoderKind::HybridResidual,
                        _ => return Err("--kind must be gnm-only or hybrid-residual".to_owned()),
                    });
                }
                "--basis" => basis = Some(PathBuf::from(next(&mut index, "--basis")?)),
                "--trace" => traces.push(PathBuf::from(next(&mut index, "--trace")?)),
                "--train-take" => {
                    train_takes.insert(next(&mut index, "--train-take")?);
                }
                "--history-len" => {
                    history_len = Some(
                        next(&mut index, "--history-len")?
                            .parse()
                            .map_err(|_| "--history-len must be an integer")?,
                    );
                }
                "--max-gap-micros" => {
                    max_gap_micros = Some(
                        next(&mut index, "--max-gap-micros")?
                            .parse()
                            .map_err(|_| "--max-gap-micros must be an integer")?,
                    );
                }
                "--ridge-lambda" => {
                    ridge_lambda = Some(
                        next(&mut index, "--ridge-lambda")?
                            .parse()
                            .map_err(|_| "--ridge-lambda must be a float")?,
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
            kind: kind.ok_or("--kind <gnm-only|hybrid-residual> is required")?,
            basis: basis.ok_or("--basis <teacher-aligned-basis.json> is required")?,
            traces,
            train_takes,
            history_len: history_len.ok_or("--history-len <n> is required")?,
            max_gap_micros: max_gap_micros.ok_or("--max-gap-micros <n> is required")?,
            ridge_lambda: ridge_lambda.ok_or("--ridge-lambda <f> is required")?,
            output: output.ok_or("--output <artifact.json> is required")?,
        })
    }
}

pub fn print_help() {
    println!(
        "  teacher-fit-gnm-decoder --kind <gnm-only|hybrid-residual>\n\
         *       --basis <teacher-aligned-basis.json> --trace <trace-v2-dir> [...]\n\
         *       --train-take <take-id> [...] --history-len <n>\n\
         *       --max-gap-micros <n> --ridge-lambda <f> --output <artifact.json>"
    );
}

/// Runs one offline G1 or H decoder fit.
///
/// # Errors
///
/// Returns a descriptive failure for invalid options, incompatible traces or
/// basis, dataset construction, ridge fitting, verification, or output I/O.
pub fn run(args: &[String]) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|arg| arg == "-h" || arg == "--help")
    {
        print_help();
        return Ok(());
    }
    let options = Options::parse(args)?;
    let basis_text = fs::read_to_string(&options.basis)
        .map_err(|error| format!("read {}: {error}", options.basis.display()))?;
    let basis: TeacherAlignedGnmBasisArtifact = serde_json::from_str(&basis_text)
        .map_err(|error| format!("parse {}: {error}", options.basis.display()))?;
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
    let feature_config = GnmSemanticFeatureConfig {
        history_len: options.history_len,
        max_gap_micros: options.max_gap_micros,
    };
    let mut rows = Vec::new();
    for trace in &traces {
        if trace.model_sha256 != basis.model_sha256
            || trace.mapping_schema_revision != basis.mapping_schema_revision
        {
            return Err(format!(
                "take {} does not match the aligned basis model or mapping",
                trace.take_id
            ));
        }
        rows.extend(
            build_gnm_semantic_rows(
                &trace.take_id,
                &trace.samples,
                &basis,
                options.kind,
                feature_config,
            )
            .map_err(|error| format!("take {}: {error}", trace.take_id))?,
        );
    }
    let feature_order = gnm_semantic_feature_order(options.kind);
    let training_config = LinearPriorTrainingConfig {
        ridge_lambda: options.ridge_lambda,
        ..LinearPriorTrainingConfig::default()
    };
    let artifact = fit_gnm_semantic_decoder(
        &rows,
        &options.train_takes,
        options.kind,
        &basis,
        training_config,
        &feature_order,
    )
    .map_err(|error| error.to_string())?;
    let zero_features = vec![0.0; artifact.linear_map.feature_mean.len()];
    let _ = predict_gnm_semantic_raw(&artifact, &zero_features)
        .map_err(|error| format!("verify artifact: {error:?}"))?;
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&artifact)
        .map_err(|error| format!("encode artifact: {error}"))?;
    fs::write(&options.output, json.as_bytes())
        .map_err(|error| format!("write {}: {error}", options.output.display()))?;

    let training_rows = rows
        .iter()
        .filter(|row| options.train_takes.contains(&row.take_id))
        .count();
    println!("teacher-fit-gnm-decoder:");
    println!("  kind: {:?}", artifact.kind);
    println!("  training takes: {:?}", artifact.training_takes);
    println!("  rows: {training_rows}");
    println!("  feature dimension: {}", zero_features.len());
    println!("  rank: {}", artifact.rank);
    println!("  joint count: {}", artifact.joint_count);
    println!("  history length: {}", artifact.feature_config.history_len);
    println!("  ridge lambda: {}", options.ridge_lambda);
    println!("  content_hash: {}", artifact.content_hash);
    println!("  artifact: {}", options.output.display());
    Ok(())
}
