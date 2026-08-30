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
//! CLI entry point for the development-only teacher capture tool.
//!
//! Currently supports a self-check command that writes a synthetic session to
//! an output directory and finalizes it. Real ARKit/RGB capture wiring lands
//! in later issues (#127/#128).

use std::path::PathBuf;
use std::process;

use teacher_capture::{
    synthetic_frame_record, CaptureDatasetWriter, DeviceMetadata, SessionHeader,
    TEACHER_CAPTURE_SCHEMA_VERSION, TIMESTAMP_DOMAIN,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("self-check") => {
            let Some(output) = args.get(1) else {
                eprintln!("usage: teacher-capture self-check <output-dir>");
                process::exit(2);
            };
            if let Err(error) = run_self_check(PathBuf::from(output)) {
                eprintln!("self-check failed: {error}");
                process::exit(1);
            }
        }
        _ => {
            println!("usage: teacher-capture <command>");
            println!("commands:");
            println!("  self-check <output-dir>  write a synthetic session and finalize it");
        }
    }
}

fn run_self_check(output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let header = SessionHeader {
        schema_version: TEACHER_CAPTURE_SCHEMA_VERSION,
        session_id: format!(
            "self-check-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_millis())
                .unwrap_or_default()
        ),
        timestamp_domain: TIMESTAMP_DOMAIN.to_owned(),
        device_metadata: DeviceMetadata {
            model: "host".to_owned(),
            os_version: "unknown".to_owned(),
            app_version: env!("CARGO_PKG_VERSION").to_owned(),
        },
    };
    let mut writer = CaptureDatasetWriter::create(&output, header)?;
    for seq in 0..10_u64 {
        writer.write_record(&synthetic_frame_record(seq, (seq + 1) * 16_667))?;
    }
    let marker = writer.finalize()?;
    println!("self-check dataset finalized: {}", marker.display());
    Ok(())
}
