//! Approved task identity, source resolution and hash validation.
use crate::error::{InferenceError, Result};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
include!(concat!(env!("OUT_DIR"), "/mediapipe_tasks.rs"));

/// Explicit source for one approved MediaPipe task.
#[derive(Clone, Debug, PartialEq)]
pub enum MediaPipeTaskSource {
    /// Read and verify this path; errors do not select another source.
    Path(PathBuf),
    /// Read the task embedded at build time.
    Embedded,
}

impl MediaPipeTaskSource {
    /// Uses the packaged file when present, otherwise the same embedded task.
    #[must_use]
    pub fn from_packaged_path(path: PathBuf) -> Self {
        if path.is_file() {
            Self::Path(path)
        } else {
            Self::Embedded
        }
    }
}

impl MediaPipeTask {
    /// Reads the explicitly selected source and verifies the approved hash.
    pub fn read(self, source: &MediaPipeTaskSource) -> Result<Vec<u8>> {
        let bytes = match source {
            MediaPipeTaskSource::Path(path) => std::fs::read(path).map_err(|error| {
                InferenceError::MediaPipeLoadFailed(format!("{} read failed: {error}", self.file()))
            })?,
            MediaPipeTaskSource::Embedded => self.embedded().to_vec(),
        };
        self.verify(&bytes)?;
        Ok(bytes)
    }

    /// Rejects bytes that are not this approved task.
    pub fn verify(self, bytes: &[u8]) -> Result<()> {
        let actual = hex::encode_upper(Sha256::digest(bytes));
        if actual.eq_ignore_ascii_case(self.sha256()) {
            Ok(())
        } else {
            Err(InferenceError::HashMismatch {
                expected: self.sha256().into(),
                actual,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    #[test]
    fn wrong_task_and_explicit_invalid_file_are_rejected() {
        assert!(matches!(
            MediaPipeTask::Face.verify(MediaPipeTask::Pose.embedded()),
            Err(InferenceError::HashMismatch { .. })
        ));
        let source = MediaPipeTaskSource::from_packaged_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
        );
        assert!(matches!(&source, MediaPipeTaskSource::Path(_)));
        assert!(matches!(
            MediaPipeTask::Face.read(&source),
            Err(InferenceError::HashMismatch { .. })
        ));
    }
}
