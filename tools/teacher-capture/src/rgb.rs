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
//! Front-camera RGB frame reference recording on the session timeline
//! (GNM #68.2c).
//!
//! Each captured frame is stored as an opaque payload file plus a record with
//! its exact frame identity, monotonic timestamp, dimensions, pixel format,
//! and explicit orientation/mirroring metadata. No implicit pixel conversion
//! happens anywhere: the payload bytes are stored exactly as delivered.
//!
//! Dropped camera frames are detected as sequence gaps and reported per
//! write, so downstream pairing can distinguish a real gap from a sparse
//! callback cadence.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Typed failures from the RGB recorder.
#[derive(Debug)]
pub enum RgbRecorderError {
    /// File-system failure while storing the payload or the record.
    Io(std::io::Error),
    /// The frame record failed to encode as JSON.
    Encode(serde_json::Error),
}

impl std::fmt::Display for RgbRecorderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "rgb recorder I/O failed: {error}"),
            Self::Encode(error) => write!(f, "rgb record encode failed: {error}"),
        }
    }
}

impl From<std::io::Error> for RgbRecorderError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for RgbRecorderError {
    fn from(value: serde_json::Error) -> Self {
        Self::Encode(value)
    }
}

/// One recorded RGB frame: exact identity plus explicit format metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RgbFrameRecord {
    /// Capture-time sequence shared with teacher records for pairing.
    pub frame_seq: u64,
    /// Monotonic microseconds since session start.
    pub timestamp_micros: u64,
    /// Stable relative path of the opaque payload inside the dataset.
    pub reference_path: String,
    /// Payload width in pixels.
    pub width_px: u32,
    /// Payload height in pixels.
    pub height_px: u32,
    /// Explicit pixel format token (for example `bgra8888`). Never converted.
    pub pixel_format: String,
    /// Sensor orientation in degrees; one of 0/90/180/270.
    pub orientation_degrees: u16,
    /// Whether the payload mirrors the raw sensor image.
    pub mirrored: bool,
}

/// Explicit per-frame format metadata supplied by the capture host.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbFrameMeta {
    /// Payload width in pixels.
    pub width_px: u32,
    /// Payload height in pixels.
    pub height_px: u32,
    /// Explicit pixel format token (for example `bgra8888`). Never converted.
    pub pixel_format: String,
    /// Sensor orientation in degrees; one of 0/90/180/270.
    pub orientation_degrees: u16,
    /// Whether the payload mirrors the raw sensor image.
    pub mirrored: bool,
}

/// Outcome of one recorded frame, including detected sequence drops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordOutcome {
    /// Number of missing sequences between the previous frame and this one.
    ///
    /// Zero when this frame directly follows the previous one or is the
    /// first frame of the session.
    pub missing_sequences: u64,
}

/// Writes frame payloads and their records under a dataset directory.
///
/// Payloads live in `<dir>/frames/`; records append to `<dir>/rgb.jsonl`.
/// Both stay gitignored by policy (#108); only derived numeric fixtures may
/// ever be committed.
pub struct RgbFrameRecorder {
    records_path: PathBuf,
    frames_dir: PathBuf,
    records_file: fs::File,
    last_frame_seq: Option<u64>,
}

impl RgbFrameRecorder {
    /// Creates a recorder under `dataset_dir`.
    ///
    /// # Errors
    ///
    /// Propagates directory-creation and file-open failures.
    pub fn create(dataset_dir: &Path) -> Result<Self, RgbRecorderError> {
        let frames_dir = dataset_dir.join("frames");
        fs::create_dir_all(&frames_dir)?;
        let records_path = dataset_dir.join("rgb.jsonl");
        let records_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&records_path)?;
        Ok(Self {
            records_path,
            frames_dir,
            records_file,
            last_frame_seq: None,
        })
    }

    /// Stores one frame payload and appends its record.
    ///
    /// Returns [`RecordOutcome`] including how many sequences were skipped
    /// since the previous frame (dropped camera frames), so gaps are visible
    /// instead of being silently absorbed.
    ///
    /// # Errors
    ///
    /// Propagates payload-write or record-encode failures.
    pub fn record(
        &mut self,
        frame_seq: u64,
        timestamp_micros: u64,
        payload: &[u8],
        meta: &RgbFrameMeta,
    ) -> Result<RecordOutcome, RgbRecorderError> {
        if !matches!(meta.orientation_degrees, 0 | 90 | 180 | 270) {
            return Err(RgbRecorderError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid orientation {}", meta.orientation_degrees),
            )));
        }

        let missing_sequences = match self.last_frame_seq {
            Some(previous) if frame_seq > previous => frame_seq - previous - 1,
            // First frame of the session, or a duplicate/regressed identity
            // that pairing validation owns.
            _ => 0,
        };

        let reference_path = format!("frames/frame_{frame_seq:010}.bin");
        let payload_path = self.frames_dir.join(format!("frame_{frame_seq:010}.bin"));
        fs::write(&payload_path, payload)?;

        let record = RgbFrameRecord {
            frame_seq,
            timestamp_micros,
            reference_path: reference_path.clone(),
            width_px: meta.width_px,
            height_px: meta.height_px,
            pixel_format: meta.pixel_format.clone(),
            orientation_degrees: meta.orientation_degrees,
            mirrored: meta.mirrored,
        };
        let mut line = serde_json::to_string(&record)?;
        line.push('\n');
        self.records_file.write_all(line.as_bytes())?;
        self.records_file.flush()?;

        self.last_frame_seq = Some(frame_seq);
        Ok(RecordOutcome { missing_sequences })
    }

    /// Path of the record log (exposed mainly for diagnostics/tests).
    #[must_use]
    pub fn records_path(&self) -> &Path {
        &self.records_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn small_meta() -> RgbFrameMeta {
        RgbFrameMeta {
            width_px: 8,
            height_px: 8,
            pixel_format: "bgra8888".to_owned(),
            orientation_degrees: 0,
            mirrored: false,
        }
    }

    fn bad_orientation_meta() -> RgbFrameMeta {
        RgbFrameMeta {
            orientation_degrees: 45,
            ..small_meta()
        }
    }

    #[test]
    fn synthetic_frames_round_trip_with_explicit_metadata() {
        let directory = tempdir().unwrap();
        let mut recorder = RgbFrameRecorder::create(directory.path()).unwrap();

        let payload_a: Vec<u8> = (0..64_u16).map(|byte| byte as u8).collect();
        let meta = RgbFrameMeta {
            width_px: 1920,
            height_px: 1080,
            pixel_format: "bgra8888".to_owned(),
            orientation_degrees: 90,
            mirrored: false,
        };
        let outcome_a = recorder.record(0, 16_667, &payload_a, &meta).unwrap();
        assert_eq!(outcome_a.missing_sequences, 0);

        let payload_b = vec![7_u8; 32];
        let outcome_b = recorder.record(1, 33_334, &payload_b, &meta).unwrap();
        assert_eq!(outcome_b.missing_sequences, 0);

        // Read back through the record log and verify byte-exact payloads.
        let text = fs::read_to_string(recorder.records_path()).unwrap();
        let records: Vec<RgbFrameRecord> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].reference_path, "frames/frame_0000000001.bin");
        assert_eq!(records[1].pixel_format, "bgra8888");
        assert_eq!(records[1].orientation_degrees, 90);
        assert!(!records[1].mirrored);

        for record in &records {
            let bytes = fs::read(directory.path().join(&record.reference_path)).unwrap();
            let expected = if record.frame_seq == 0 {
                &payload_a
            } else {
                &payload_b
            };
            assert_eq!(&bytes, expected);
        }
    }

    #[test]
    fn dropped_camera_frames_are_reported_as_sequence_gaps() {
        let directory = tempdir().unwrap();
        let mut recorder = RgbFrameRecorder::create(directory.path()).unwrap();
        recorder.record(5, 10, &[0], &small_meta()).unwrap();
        // Sequences 6..9 never arrived; frame 10 completes the sample.
        let outcome = recorder.record(10, 20, &[1], &small_meta()).unwrap();
        assert_eq!(outcome.missing_sequences, 4);
    }

    #[test]
    fn invalid_orientation_is_rejected_before_any_write() {
        let directory = tempdir().unwrap();
        let mut recorder = RgbFrameRecorder::create(directory.path()).unwrap();
        assert!(recorder
            .record(0, 0, &[0], &bad_orientation_meta())
            .is_err());
        // No record was appended; the log exists but stays empty.
        let text = fs::read_to_string(recorder.records_path()).unwrap();
        assert!(text.is_empty());
    }
}
