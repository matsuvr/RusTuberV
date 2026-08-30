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
//! Machine-runnable Direct/GNM temporal quality report (GNM #57.4).
//!
//! Reads a JSON description of source-aligned per-backend scalar traces plus
//! their response commands and prints one comparable table per channel:
//! stationary RMS, first/second difference RMS, onset and 10/50/90%
//! crossings, rise time, peak attenuation, overshoot, settling time,
//! blink-pulse peak preservation/timing, and measured end-to-end latency.
//!
//! Input schema (`cargo xtask temporal-report <path>`):
//!
//! ```json
//! {
//!   "channels": [
//!     {
//!       "name": "jawOpen",
//!       "backends": [
//!         {
//!           "backend": "DirectMediaPipe",
//!           "end_to_end_latency_ms": 45.0,
//!           "samples": [[0, 0.0], [16667, 0.5], [33334, 1.0]],
//!           "rise":    {"command_micros": 0, "baseline": 0.0, "target": 1.0,
//!                       "settling_tolerance_fraction": 0.1},
//!           "release": null,
//!           "pulse":   {"onset_micros": 0, "baseline": 0.0, "target_peak": 1.0,
//!                       "expected_peak_micros": 16667}
//!         },
//!         {
//!           "backend": "GnmTemporal",
//!           "...": "same shape"
//!         }
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! The command is deterministic over numeric traces only; no camera, model,
//! or runtime access happens on this path, so reports are reproducible from
//! synthetic or recorded numeric data alone.

use std::path::Path;
use std::process;

use serde::Deserialize;
use vtuber_tracking::{
    BackendTemporalQuality, FaceTrackingBackend, PulseResponseSpec, StepResponseSpec,
    TemporalChannelSpecs, TemporalSample, TemporalTrace, backend_temporal_quality,
};

#[derive(Deserialize)]
struct ReportInput {
    /// One entry per scalar channel to compare.
    channels: Vec<ChannelInput>,
}

#[derive(Deserialize)]
struct ChannelInput {
    /// Channel name used in report rows.
    name: String,
    /// One trace per backend for this exact channel.
    backends: Vec<BackendInput>,
}

#[derive(Deserialize)]
struct BackendInput {
    /// `"DirectMediaPipe"` or `"GnmTemporal"`.
    backend: String,
    /// Measured capture-to-publish latency in milliseconds.
    end_to_end_latency_ms: f64,
    /// `[timestamp_micros, value]` pairs in strictly increasing order.
    samples: Vec<(u64, f64)>,
    /// Optional commanded rise.
    rise: Option<StepSpec>,
    /// Optional commanded release.
    release: Option<StepSpec>,
    /// Optional short pulse such as a blink.
    pulse: Option<PulseSpec>,
}

#[derive(Deserialize)]
struct StepSpec {
    /// Timestamp at which the target changes.
    command_micros: u64,
    /// Pre-command baseline value.
    baseline: f64,
    /// Post-command target value.
    target: f64,
    /// Settling band fraction of the commanded amplitude.
    settling_tolerance_fraction: f64,
}

#[derive(Deserialize)]
struct PulseSpec {
    /// Timestamp at which the pulse begins.
    onset_micros: u64,
    /// Baseline value before the pulse.
    baseline: f64,
    /// Expected pulse peak value.
    target_peak: f64,
    /// Optional reference peak timestamp.
    expected_peak_micros: Option<u64>,
}

fn parse_backend(name: &str) -> Result<FaceTrackingBackend, String> {
    match name {
        "DirectMediaPipe" => Ok(FaceTrackingBackend::DirectMediaPipe),
        "GnmTemporal" => Ok(FaceTrackingBackend::GnmTemporal),
        other => Err(format!("unknown backend `{other}`")),
    }
}

fn build_row(channel_name: &str, input: &BackendInput) -> Result<BackendTemporalQuality, String> {
    let backend = parse_backend(&input.backend)?;
    let samples = input
        .samples
        .iter()
        .map(|(timestamp_micros, value)| TemporalSample {
            timestamp_micros: *timestamp_micros,
            value: *value,
        })
        .collect();
    let trace = TemporalTrace::new(samples).map_err(|error| format!("{error:?}"))?;
    let specs = TemporalChannelSpecs {
        rise: input.rise.as_ref().map(|spec| StepResponseSpec {
            command_micros: spec.command_micros,
            baseline: spec.baseline,
            target: spec.target,
            settling_tolerance_fraction: spec.settling_tolerance_fraction,
        }),
        release: input.release.as_ref().map(|spec| StepResponseSpec {
            command_micros: spec.command_micros,
            baseline: spec.baseline,
            target: spec.target,
            settling_tolerance_fraction: spec.settling_tolerance_fraction,
        }),
        pulse: input.pulse.as_ref().map(|spec| PulseResponseSpec {
            onset_micros: spec.onset_micros,
            baseline: spec.baseline,
            target_peak: spec.target_peak,
            expected_peak_micros: spec.expected_peak_micros,
        }),
    };
    backend_temporal_quality(
        backend,
        input.end_to_end_latency_ms,
        &[(channel_name, trace, specs)],
    )
    .map_err(|error| format!("{error:?}"))
}

fn optional_ms(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |ms| format!("{ms:.2}"))
}

fn print_row(row: &BackendTemporalQuality) {
    let Some(quality) = row.channels.first() else {
        eprintln!("  error: backend produced no channel metrics");
        std::process::exit(1);
    };
    println!(
        "| {:<14} | {:>10.3} | {:>12.3} | {:>12.3}",
        format!("{:?}", row.backend),
        row.end_to_end_latency_ms,
        quality.noise.stationary_rms,
        quality
            .noise
            .first_difference_rms_per_second
            .unwrap_or(f64::NAN)
    );
    if let Some(rise) = &quality.rise {
        println!(
            "| {:<14} | {:>10} | {:>12} | {:>12} | onset {} t50 {} t90 {} rise {} settle {}",
            "  rise",
            "",
            "",
            "",
            optional_ms(rise.onset_delay_ms),
            optional_ms(rise.t50_ms),
            optional_ms(rise.t90_ms),
            optional_ms(rise.rise_time_10_90_ms),
            optional_ms(rise.settling_time_ms),
        );
    }
    if let Some(release) = &quality.release {
        println!(
            "| {:<14} | {:>10} | {:>12} | {:>12} | onset {} t50 {} t90 {} settle {}",
            "  release",
            "",
            "",
            "",
            optional_ms(release.onset_delay_ms),
            optional_ms(release.t50_ms),
            optional_ms(release.t90_ms),
            optional_ms(release.settling_time_ms),
        );
    }
    if let Some(pulse) = &quality.pulse {
        println!(
            "| {:<14} | {:>10} | {:>12} | {:>12} | peak ratio {:.3} attenuation {:.3} timing err {} ms",
            "  pulse",
            "",
            "",
            "",
            pulse.peak_response_ratio,
            pulse.peak_attenuation,
            optional_ms(pulse.peak_timing_error_ms),
        );
    }
}

/// Runs the report command against a JSON trace description file.
///
/// # Errors
///
/// Returns an error message when the file cannot be read/parsed or any trace
/// fails typed validation; never panics on malformed input.
pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = args.first() else {
        eprintln!("usage: cargo xtask temporal-report <trace-input.json>");
        process::exit(2);
    };
    let raw = std::fs::read_to_string(Path::new(path))?;
    let input: ReportInput = serde_json::from_str(&raw)?;

    for channel in &input.channels {
        println!("channel: {}", channel.name);
        println!(
            "| {:<14} | {:>10} | {:>12} | {:>12} |",
            "backend", "latency ms", "stat RMS", "d1 RMS/s"
        );
        for backend_input in &channel.backends {
            match build_row(&channel.name, backend_input) {
                Ok(row) => print_row(&row),
                Err(error) => {
                    eprintln!("  error for backend {}: {error}", backend_input.backend);
                    process::exit(1);
                }
            }
        }
        println!();
    }
    Ok(())
}
