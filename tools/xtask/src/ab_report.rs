// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
//! Robustness/cross-talk/performance A/B report command (GNM #57.5).
//!
//! Reads a JSON description of source-aligned Direct/GNM numeric traces and
//! measured runtime counters, prints per-backend robustness (one-frame jump,
//! dropout gaps, stale duration, reacquire delay), expression/head-pose
//! cross-talk rows, and the latency/iteration/drop table, and finishes with
//! the promotion verdict (`Default` or `Experimental`) listing numeric
//! blockers.
//!
//! Input schema (`cargo xtask ab-report <path>`):
//!
//! ```json
//! {
//!   "expected_dt_micros": 16667,
//!   "gap_tolerance": 1.5,
//!   "stale_epsilon": 0.000001,
//!   "criteria": {
//!     "max_gnm_latency_overhead_ms": 12.0,
//!     "max_head_to_expression_crosstalk": 0.05,
//!     "max_expression_to_head_crosstalk": 0.01,
//!     "max_fit_latency_ms": 10.0,
//!     "max_longest_dropout_ms": 120.0
//!   },
//!   "backends": [
//!     {
//!       "backend": "DirectMediaPipe",
//!       "end_to_end_latency_ms": 45.0,
//!       "fit_iterations": null,
//!       "dropped_or_replaced_frames": null,
//!       "channels": [
//!         {"name": "jawOpen", "samples": [[0, 0.0], [16667, 0.2]]}
//!       ]
//!     }
//!   ],
//!   "crosstalk": [
//!     {
//!       "backend": "GnmTemporal",
//!       "kind": "head_to_expression",
//!       "driver": "headYawMagnitude",
//!       "observed": "mouthSmileLeft",
//!       "driver_threshold_per_second": 5.0
//!     }
//!   ],
//!   "verdict_inputs": {
//!     "direct_end_to_end_ms": 45.0,
//!     "gnm_end_to_end_ms": 55.0,
//!     "gnm_max_fit_ms": 8.0,
//!     "gnm_head_to_expression_crosstalk": 0.01,
//!     "gnm_expression_to_head_crosstalk": 0.002,
//!     "gnm_longest_dropout_ms": 90.0
//!   }
//! }
//! ```

use std::path::Path;
use std::process;

use serde::Deserialize;
use vtuber_tracking::{
    AbMeasuredInputs, FaceTrackingBackend, PromotionCriteria, TemporalSample, TemporalTrace,
    crosstalk_metrics, promotion_verdict, robustness_metrics,
};

#[derive(Deserialize)]
struct ReportInput {
    /// Nominal sample period in microseconds.
    expected_dt_micros: u64,
    /// Gap tolerance multiplier over the nominal period.
    gap_tolerance: f64,
    /// Minimum value change treated as real motion.
    stale_epsilon: f64,
    /// Numeric promotion criteria evaluated at report end.
    criteria: CriteriaInput,
    /// One entry per backend.
    backends: Vec<BackendInput>,
    /// Cross-talk measurement rows.
    crosstalk: Vec<CrosstalkInput>,
    /// Measured inputs to the promotion evaluation.
    verdict_inputs: VerdictInputs,
}

#[derive(Deserialize)]
struct CriteriaInput {
    /// Maximum tolerated GNM latency overhead in ms.
    max_gnm_latency_overhead_ms: f64,
    /// Maximum tolerated expression excess during rigid head motion.
    max_head_to_expression_crosstalk: f64,
    /// Maximum tolerated head-pose excess during expression motion.
    max_expression_to_head_crosstalk: f64,
    /// Maximum tolerated GNM fit latency in ms.
    max_fit_latency_ms: f64,
    /// Maximum tolerated longest GNM dropout in ms.
    max_longest_dropout_ms: f64,
}

#[derive(Deserialize)]
struct BackendInput {
    /// `"DirectMediaPipe"` or `"GnmTemporal"`.
    backend: String,
    /// End-to-end latency in ms.
    end_to_end_latency_ms: f64,
    /// Solver iterations of the most recent fit when instrumented.
    fit_iterations: Option<usize>,
    /// Dropped/replaced frame counter when instrumented.
    dropped_or_replaced_frames: Option<u64>,
    /// Named scalar traces for this backend.
    channels: Vec<ChannelInput>,
}

#[derive(Deserialize)]
struct ChannelInput {
    /// Channel name used in rows and cross-talk references.
    name: String,
    /// `[timestamp_micros, value]` pairs, strictly increasing.
    samples: Vec<(u64, f64)>,
}

#[derive(Deserialize)]
struct CrosstalkInput {
    /// Backend whose pair is measured.
    backend: String,
    /// Free-form label printed on the row.
    kind: String,
    /// Driver channel name (must exist in the backend channels).
    driver: String,
    /// Observed channel name.
    observed: String,
    /// Driver velocity above this counts as driven motion.
    driver_threshold_per_second: f64,
}

#[derive(Deserialize)]
struct VerdictInputs {
    /// Direct end-to-end latency, ms.
    direct_end_to_end_ms: f64,
    /// GNM end-to-end latency, ms.
    gnm_end_to_end_ms: f64,
    /// Worst observed GNM fit latency, ms.
    gnm_max_fit_ms: f64,
    /// Expression excess while only the head moved.
    gnm_head_to_expression_crosstalk: f64,
    /// Head-pose excess while only expressions moved.
    gnm_expression_to_head_crosstalk: f64,
    /// Longest GNM output dropout, ms.
    gnm_longest_dropout_ms: f64,
}

fn parse_backend(name: &str) -> Result<FaceTrackingBackend, String> {
    match name {
        "DirectMediaPipe" => Ok(FaceTrackingBackend::DirectMediaPipe),
        "GnmTemporal" => Ok(FaceTrackingBackend::GnmTemporal),
        other => Err(format!("unknown backend `{other}`")),
    }
}

fn parse_trace(samples: &[(u64, f64)]) -> Result<TemporalTrace, String> {
    let converted = samples
        .iter()
        .map(|(timestamp_micros, value)| TemporalSample {
            timestamp_micros: *timestamp_micros,
            value: *value,
        })
        .collect();
    TemporalTrace::new(converted).map_err(|error| format!("{error:?}"))
}

fn find_channel<'a>(channels: &'a [ChannelInput], name: &str) -> Result<&'a ChannelInput, String> {
    channels
        .iter()
        .find(|channel| channel.name == name)
        .ok_or_else(|| format!("channel `{name}` not found"))
}

/// Runs the A/B report against a JSON description file.
///
/// # Errors
///
/// Returns errors for unreadable/malformed input or typed validation
/// failures; never panics on external data.
pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = args.first() else {
        eprintln!("usage: cargo xtask ab-report <ab-input.json>");
        process::exit(2);
    };
    let raw = std::fs::read_to_string(Path::new(path))?;
    let input: ReportInput = serde_json::from_str(&raw)?;

    println!("== robustness ==");
    println!(
        "| {:<16} | {:<14} | {:>10} | {:>8} | {:>10} | {:>10} |",
        "channel", "backend", "jump", "dropouts", "stale ms", "reacq ms"
    );
    for backend_input in &input.backends {
        let backend = parse_backend(&backend_input.backend)
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
        for channel in &backend_input.channels {
            let trace = parse_trace(&channel.samples)?;
            let metrics = robustness_metrics(
                &trace,
                input.expected_dt_micros,
                input.gap_tolerance,
                input.stale_epsilon,
            )
            .map_err(|error| -> Box<dyn std::error::Error> { format!("{error:?}").into() })?;
            println!(
                "| {:<16} | {:<14} | {:>10.3} | {:>8} | {:>10.2} | {:>10} |",
                channel.name,
                format!("{backend:?}"),
                metrics.max_one_frame_jump,
                metrics.dropout_gap_count,
                metrics.stale_duration_ms,
                metrics
                    .reacquire_delay_ms
                    .map_or_else(|| "-".to_owned(), |ms| format!("{ms:.2}")),
            );
        }
    }

    println!("\n== cross-talk ==");
    println!(
        "| {:<22} | {:<14} | {:>12} | {:>12} | {:>12} |",
        "pair", "backend", "driven rms/s", "rest rms/s", "excess"
    );
    for row in &input.crosstalk {
        let backend_input = input
            .backends
            .iter()
            .find(|candidate| candidate.backend == row.backend)
            .ok_or_else(|| format!("unknown backend `{}` in crosstalk row", row.backend))?;
        let driver = find_channel(&backend_input.channels, &row.driver)?;
        let observed = find_channel(&backend_input.channels, &row.observed)?;
        let metrics = crosstalk_metrics(
            &parse_trace(&driver.samples)?,
            &parse_trace(&observed.samples)?,
            row.driver_threshold_per_second,
        )
        .map_err(|error| -> Box<dyn std::error::Error> { format!("{error:?}").into() })?;
        println!(
            "| {:<22} | {:<14} | {:>12.4} | {:>12.4} | {:>12.4} |",
            format!("{} {}->{}", backend_input.backend, row.kind, row.observed),
            backend_input.backend,
            metrics.driven_rms_per_second,
            metrics.rest_rms_per_second,
            metrics.crosstalk_excess,
        );
    }

    println!("\n== performance ==");
    println!(
        "| {:<14} | {:>12} | {:>12} | {:>12} |",
        "backend", "e2e ms", "iterations", "dropped"
    );
    for backend_input in &input.backends {
        println!(
            "| {:<14} | {:>12.2} | {:>12} | {:>12} |",
            backend_input.backend,
            backend_input.end_to_end_latency_ms,
            backend_input
                .fit_iterations
                .map_or_else(|| "-".to_owned(), |value| value.to_string()),
            backend_input
                .dropped_or_replaced_frames
                .map_or_else(|| "-".to_owned(), |value| value.to_string()),
        );
    }

    let criteria = PromotionCriteria {
        max_gnm_latency_overhead_ms: input.criteria.max_gnm_latency_overhead_ms,
        max_head_to_expression_crosstalk: input.criteria.max_head_to_expression_crosstalk,
        max_expression_to_head_crosstalk: input.criteria.max_expression_to_head_crosstalk,
        max_fit_latency_ms: input.criteria.max_fit_latency_ms,
        max_longest_dropout_ms: input.criteria.max_longest_dropout_ms,
    };
    let inputs = AbMeasuredInputs {
        direct_end_to_end_ms: input.verdict_inputs.direct_end_to_end_ms,
        gnm_end_to_end_ms: input.verdict_inputs.gnm_end_to_end_ms,
        gnm_max_fit_ms: input.verdict_inputs.gnm_max_fit_ms,
        gnm_head_to_expression_crosstalk: input.verdict_inputs.gnm_head_to_expression_crosstalk,
        gnm_expression_to_head_crosstalk: input.verdict_inputs.gnm_expression_to_head_crosstalk,
        gnm_longest_dropout_ms: input.verdict_inputs.gnm_longest_dropout_ms,
    };
    let verdict = promotion_verdict(inputs, criteria)
        .map_err(|error| -> Box<dyn std::error::Error> { format!("{error:?}").into() })?;

    println!("\n== verdict ==");
    match verdict.decision {
        vtuber_tracking::PromotionDecision::Default => println!("Default"),
        vtuber_tracking::PromotionDecision::Experimental => {
            println!("Experimental");
            for blocker in &verdict.blockers {
                println!(
                    "  blocker: {} measured {:.4} bound {:.4}",
                    blocker.criterion, blocker.measured, blocker.bound
                );
            }
        }
    }
    Ok(())
}
