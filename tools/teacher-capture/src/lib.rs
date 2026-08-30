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
//! Development-only ARKit teacher capture tool scaffold (GNM #68.2a).
//!
//! This crate is intentionally outside the production workspace so an iOS
//! toolchain can build it independently (`cargo build` from this directory;
//! the production workspace excludes it). It currently provides only:
//!
//! - dataset session metadata (schema version, session id, timestamp domain,
//!   device/app metadata),
//! - a JSONL frame-record writer with an atomic finalize marker, so a partial
//!   write is never mistaken for a completed dataset,
//! - synthetic record generation used by tests and by later issues that wire
//!   real ARKit/RGB capture callbacks.
//!
//! ARKit values and camera frames are NOT captured yet; that wiring arrives
//! in #127/#128.
pub mod arkit;
pub mod manifest;
pub mod rgb;

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Schema version shared with `vtuber_tracking::arkit_teacher`.
pub const TEACHER_CAPTURE_SCHEMA_VERSION: u32 = 1;

/// Fixed timestamp domain recorded in every session header.
pub const TIMESTAMP_DOMAIN: &str = "monotonic-micros-since-session-start";

/// Errors surfaced by the dataset writer.
#[derive(Debug)]
pub enum CaptureWriterError {
    /// The output directory could not be created/read.
    Io(std::io::Error),
    /// A JSON record failed to encode.
    Encode(serde_json::Error),
}

impl std::error::Error for CaptureWriterError {}

impl std::fmt::Display for CaptureWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "capture writer I/O failed: {error}"),
            Self::Encode(error) => write!(f, "capture record encode failed: {error}"),
        }
    }
}

impl From<std::io::Error> for CaptureWriterError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for CaptureWriterError {
    fn from(value: serde_json::Error) -> Self {
        Self::Encode(value)
    }
}

/// Session header describing the dataset before any frames are written.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SessionHeader {
    /// Dataset schema version.
    pub schema_version: u32,
    /// Caller-provided unique session id (for example a UUID).
    pub session_id: String,
    /// Fixed timestamp domain string ([`TIMESTAMP_DOMAIN`]).
    pub timestamp_domain: String,
    /// Device metadata reported by the capture host.
    pub device_metadata: DeviceMetadata,
}

/// Free-form but bounded device/app metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeviceMetadata {
    /// Device model string (for example "iPhone14,2").
    pub model: String,
    /// OS version string.
    pub os_version: String,
    /// Capture app/tool version.
    pub app_version: String,
}

/// One generic frame record. Later issues extend this with typed payloads;
/// the writer treats records as opaque validated JSON lines.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameRecord {
    /// Strictly increasing capture sequence within the session.
    pub frame_seq: u64,
    /// Monotonic microseconds since session start.
    pub timestamp_micros: u64,
    /// Record kind token (for example `arkit_teacher`, `rgb_reference`).
    pub kind: String,
    /// Kind-specific JSON payload.
    pub payload: serde_json::Value,
}

/// JSONL dataset writer with atomic completion semantics.
///
/// Layout inside `output_dir`:
///
/// ```text
/// session.json          <- SessionHeader
/// frames.jsonl          <- one FrameRecord per line
/// COMPLETED             <- written last via finalize(); its presence marks
///                          the dataset complete
/// ```
///
/// Readers must ignore any directory without `COMPLETED`; a crashed or
/// interrupted run therefore never looks like a valid dataset.
pub struct CaptureDatasetWriter {
    frames_path: PathBuf,
    marker_path: PathBuf,
    frames_file: fs::File,
    finalized: bool,
}

impl CaptureDatasetWriter {
    /// Creates the output directory and writes the session header.
    ///
    /// # Errors
    ///
    /// Propagates I/O failures from directory creation and header writing.
    pub fn create(output_dir: &Path, header: SessionHeader) -> Result<Self, CaptureWriterError> {
        if header.schema_version != TEACHER_CAPTURE_SCHEMA_VERSION {
            return Err(CaptureWriterError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "unsupported schema version {} (expected {})",
                    header.schema_version, TEACHER_CAPTURE_SCHEMA_VERSION
                ),
            )));
        }
        fs::create_dir_all(output_dir)?;
        let header_path = output_dir.join("session.json");
        let header_json = serde_json::to_string_pretty(&header)?;
        // Write through a temp file + rename so no reader sees a torn header.
        let temp_path = output_dir.join("session.json.tmp");
        fs::write(&temp_path, header_json.as_bytes())?;
        fs::rename(&temp_path, &header_path)?;

        let frames_path = output_dir.join("frames.jsonl");
        let frames_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&frames_path)?;
        Ok(Self {
            marker_path: output_dir.join("COMPLETED"),
            frames_path,
            frames_file,
            finalized: false,
        })
    }

    /// Appends one frame record as a single JSON line.
    ///
    /// # Errors
    ///
    /// Propagates I/O or encoding failures; the caller may retry the record.
    pub fn write_record(&mut self, record: &FrameRecord) -> Result<(), CaptureWriterError> {
        let mut line = serde_json::to_string(record)?;
        line.push('\n');
        self.frames_file.write_all(line.as_bytes())?;
        self.frames_file.flush()?;
        Ok(())
    }

    /// Writes the `COMPLETED` marker, promoting the dataset to completed.
    ///
    /// # Errors
    ///
    /// Propagates the marker-write I/O failure.
    pub fn finalize(mut self) -> Result<PathBuf, CaptureWriterError> {
        fs::write(&self.marker_path, b"completed\n")?;
        self.finalized = true;
        Ok(self.marker_path.clone())
    }

    /// Path of the frame log (exposed mainly for diagnostics/tests).
    #[must_use]
    pub fn frames_path(&self) -> &Path {
        &self.frames_path
    }
}

impl Drop for CaptureDatasetWriter {
    fn drop(&mut self) {
        if !self.finalized {
            // An unfinalized writer never leaves a COMPLETED marker behind.
            let _ = fs::remove_file(&self.marker_path);
        }
    }
}

/// Builds a synthetic frame record for tests and offline fixtures.
#[must_use]
pub fn synthetic_frame_record(frame_seq: u64, timestamp_micros: u64) -> FrameRecord {
    FrameRecord {
        frame_seq,
        timestamp_micros,
        kind: "synthetic".to_owned(),
        payload: serde_json::json!({ "value": frame_seq }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn header() -> SessionHeader {
        SessionHeader {
            schema_version: TEACHER_CAPTURE_SCHEMA_VERSION,
            session_id: "session-0001".to_owned(),
            timestamp_domain: TIMESTAMP_DOMAIN.to_owned(),
            device_metadata: DeviceMetadata {
                model: "iPhone14,2".to_owned(),
                os_version: "17.5".to_owned(),
                app_version: env!("CARGO_PKG_VERSION").to_owned(),
            },
        }
    }

    #[test]
    fn empty_session_can_be_flushed_and_finalized() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("session-a");
        let writer = CaptureDatasetWriter::create(&output, header()).unwrap();
        assert!(output.join("session.json").is_file());
        assert!(!output.join("COMPLETED").is_file());
        writer.finalize().unwrap();
        assert!(output.join("COMPLETED").is_file());
        // Empty frames log is valid: zero-frame sessions are flushable.
        assert_eq!(fs::read(output.join("frames.jsonl")).unwrap().len(), 0);
    }

    #[test]
    fn synthetic_records_round_trip_and_finalize_marks_completion() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("session-b");
        let mut writer = CaptureDatasetWriter::create(&output, header()).unwrap();
        for seq in 0..3_u64 {
            writer
                .write_record(&synthetic_frame_record(seq, (seq + 1) * 16_667))
                .unwrap();
        }
        writer.finalize().unwrap();

        let text = fs::read_to_string(output.join("frames.jsonl")).unwrap();
        let parsed: Vec<FrameRecord> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[2].frame_seq, 2);
        assert!(parsed
            .windows(2)
            .all(|pair| pair[0].frame_seq < pair[1].frame_seq));
    }

    #[test]
    fn dropped_writer_never_leaves_a_completed_marker() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("session-c");
        {
            let mut writer = CaptureDatasetWriter::create(&output, header()).unwrap();
            writer
                .write_record(&synthetic_frame_record(0, 16_667))
                .unwrap();
            // Simulate a crash: drop without finalize().
        }
        assert!(!output.join("COMPLETED").is_file());
        // The partial frames log exists but must be ignored by readers.
        assert!(output.join("frames.jsonl").is_file());
    }

    #[test]
    fn wrong_schema_version_is_rejected() {
        let directory = tempdir().unwrap();
        let mut bad = header();
        bad.schema_version = 99;
        assert!(CaptureDatasetWriter::create(&directory.path().join("bad"), bad).is_err());
    }
}
