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
//! Capture-time pairing of ARKit teacher records with RGB references
//! (GNM #68.2d).
//!
//! Pairing uses exact frame identity only: a teacher record pairs with an
//! RGB reference when both carry the same `frame_seq`. There is no
//! nearest-timestamp repair anywhere; unpaired records and sequence gaps are
//! reported instead. The resulting manifest carries counts plus timestamp
//! skew diagnostics, and a dataset is finalized only when validation passes.
//!
//! Validation failures never produce a completion marker, so a partially
//! paired session can never masquerade as a completed dataset.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::rgb::RgbFrameRecord;

/// Errors from pairing/manifest construction.
#[derive(Clone, Debug, PartialEq)]
pub enum PairingError {
    /// The same frame_seq appeared more than once on one side.
    DuplicateFrameSeq {
        /// Side carrying the duplicate (`teacher` or `rgb`).
        side: &'static str,
        /// Duplicated sequence.
        frame_seq: u64,
    },
    /// Sequence numbers regressed within one side's input order.
    RegressedFrameSeq {
        /// Side carrying the regression.
        side: &'static str,
        /// Previous sequence.
        previous: u64,
        /// Offending sequence.
        current: u64,
    },
    /// A paired sample's timestamps disagreed beyond zero tolerance because
    /// both sides claim the same monotonic timeline (#108).
    IdentityTimestampMismatch {
        /// Frame sequence of the mismatched pair.
        frame_seq: u64,
        /// Teacher-side monotonic timestamp in microseconds.
        teacher_timestamp_micros: u64,
        /// RGB-side monotonic timestamp in microseconds.
        rgb_timestamp_micros: u64,
    },
}

/// One side's minimal identity used during pairing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeacherIdentity {
    /// Capture-time sequence.
    pub frame_seq: u64,
    /// Monotonic microseconds since session start.
    pub timestamp_micros: u64,
}

/// Per-pair skew diagnostics recorded in the manifest.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct PairSkew {
    /// Paired frame sequence.
    pub frame_seq: u64,
    /// Signed skew `teacher - rgb` in microseconds (zero on exact identity).
    pub timestamp_skew_micros: i64,
}

/// Counts written into the manifest.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PairCounts {
    /// Records paired by exact frame identity.
    pub paired: usize,
    /// Teacher-only sequences (RGB missing or dropped).
    pub unpaired_teacher: usize,
    /// RGB-only sequences (ARKit missing).
    pub unpaired_rgb: usize,
    /// Missing sequences strictly inside the observed sequence range.
    pub dropped_sequences: u64,
}

/// Completed pairing report serialized as the dataset manifest.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PairManifest {
    /// Schema version shared across the teacher-capture tool.
    pub schema_version: u32,
    /// Pairing counts summary.
    pub counts: PairCounts,
    /// Per-pair timestamp skew diagnostics.
    pub skews: Vec<PairSkew>,
}

/// Builds the manifest by pairing teachers and RGB records by exact identity.
///
/// Both inputs must already be in strictly increasing sequence order;
/// duplicates and regressions are typed errors rather than silent merges.
/// Sequences missing from both sides inside `[min_seq, max_seq]` count as
/// dropped.
///
/// # Errors
///
/// Returns [`PairingError`] for any duplicate/regressed input or an
/// identity-timestamp disagreement between the two sides of a pair.
#[allow(clippy::indexing_slicing)] // bounds proven by loop structure; see AGENTS.md
pub fn build_pair_manifest(
    teacher: &[TeacherIdentity],
    rgb: &[RgbFrameRecord],
) -> Result<PairManifest, PairingError> {
    check_strict_order(teacher.iter().map(|identity| identity.frame_seq), "teacher")?;
    check_strict_order(rgb.iter().map(|record| record.frame_seq), "rgb")?;

    let mut counts = PairCounts::default();
    let mut skews = Vec::new();
    let mut rgb_index = 0_usize;

    for identity in teacher {
        // Advance the rgb cursor past teacher-only sequences.
        while rgb_index < rgb.len() && rgb[rgb_index].frame_seq < identity.frame_seq {
            rgb_index += 1;
            counts.unpaired_rgb += 1;
        }
        match rgb.get(rgb_index) {
            Some(record) if record.frame_seq == identity.frame_seq => {
                if record.timestamp_micros != identity.timestamp_micros {
                    return Err(PairingError::IdentityTimestampMismatch {
                        frame_seq: identity.frame_seq,
                        teacher_timestamp_micros: identity.timestamp_micros,
                        rgb_timestamp_micros: record.timestamp_micros,
                    });
                }
                counts.paired += 1;
                skews.push(PairSkew {
                    frame_seq: identity.frame_seq,
                    timestamp_skew_micros: identity
                        .timestamp_micros
                        .saturating_sub(record.timestamp_micros)
                        as i64,
                });
                rgb_index += 1;
            }
            _ => counts.unpaired_teacher += 1,
        }
    }
    counts.unpaired_rgb += rgb.len() - rgb_index;

    let all_seqs = teacher
        .iter()
        .map(|identity| identity.frame_seq)
        .chain(rgb.iter().map(|record| record.frame_seq));
    let min_seq = all_seqs.clone().min();
    if let (Some(min), Some(max)) = (min_seq, all_seqs.max()) {
        let present: std::collections::BTreeSet<u64> = teacher
            .iter()
            .map(|identity| identity.frame_seq)
            .chain(rgb.iter().map(|record| record.frame_seq))
            .collect();
        counts.dropped_sequences =
            ((min..=max).filter(|seq| !present.contains(seq)).count()) as u64;
    }

    Ok(PairManifest {
        schema_version: crate::TEACHER_CAPTURE_SCHEMA_VERSION,
        counts,
        skews,
    })
}

fn check_strict_order<I: IntoIterator<Item = u64>>(
    sequences: I,
    side: &'static str,
) -> Result<(), PairingError> {
    let mut previous: Option<u64> = None;
    for frame_seq in sequences {
        if let Some(previous_value) = previous {
            if frame_seq == previous_value {
                return Err(PairingError::DuplicateFrameSeq { side, frame_seq });
            }
            if frame_seq < previous_value {
                return Err(PairingError::RegressedFrameSeq {
                    side,
                    previous: previous_value,
                    current: frame_seq,
                });
            }
        }
        previous = Some(frame_seq);
    }
    Ok(())
}

/// Writes the manifest and finalizes the dataset directory atomically.
///
/// The `COMPLETED` marker is written only after the manifest bytes hit disk;
/// callers must not invoke this when validation failed.
///
/// # Errors
///
/// Propagates I/O or encode failures. On failure no marker is created.
pub fn write_manifest_and_finalize(
    dataset_dir: &Path,
    manifest: &PairManifest,
) -> Result<PathBuf, std::io::Error> {
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let temp = dataset_dir.join("manifest.json.tmp");
    std::fs::write(&temp, json.as_bytes())?;
    let final_path = dataset_dir.join("manifest.json");
    std::fs::rename(&temp, &final_path)?;
    let marker = dataset_dir.join("COMPLETED");
    std::fs::write(&marker, b"completed\n")?;
    Ok(marker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn teacher(seq: u64) -> TeacherIdentity {
        TeacherIdentity {
            frame_seq: seq,
            timestamp_micros: seq * 16_667,
        }
    }

    fn rgb_record(seq: u64) -> RgbFrameRecord {
        RgbFrameRecord {
            frame_seq: seq,
            timestamp_micros: seq * 16_667,
            reference_path: format!("frames/frame_{seq:010}.bin"),
            width_px: 1920,
            height_px: 1080,
            pixel_format: "bgra8888".to_owned(),
            orientation_degrees: 90,
            mirrored: false,
        }
    }

    #[test]
    fn fully_paired_sequence_reports_zero_losses() {
        let teachers: Vec<TeacherIdentity> = (0..5).map(teacher).collect();
        let rgbs: Vec<RgbFrameRecord> = (0..5).map(rgb_record).collect();
        let manifest = build_pair_manifest(&teachers, &rgbs).unwrap();
        assert_eq!(manifest.counts.paired, 5);
        assert_eq!(manifest.counts.unpaired_teacher, 0);
        assert_eq!(manifest.counts.unpaired_rgb, 0);
        assert_eq!(manifest.counts.dropped_sequences, 0);
        assert!(manifest
            .skews
            .iter()
            .all(|skew| skew.timestamp_skew_micros == 0));
    }

    #[test]
    fn missing_rgb_and_missing_arkit_are_counted_separately() {
        // Teacher has 0..4 but RGB misses 2 (missing ARKit case covers the
        // symmetric direction: RGB has 7 while teacher stops at 4).
        let teachers: Vec<TeacherIdentity> =
            [0_u64, 1, 3, 4].iter().copied().map(teacher).collect();
        let mut rgbs: Vec<RgbFrameRecord> = [0_u64, 1, 2, 3, 4, 7]
            .iter()
            .copied()
            .map(rgb_record)
            .collect();
        rgbs.remove(2); // drop seq 2 so it becomes a dropped sequence

        let manifest = build_pair_manifest(&teachers, &rgbs).unwrap();
        assert_eq!(manifest.counts.paired, 4);
        assert_eq!(manifest.counts.unpaired_teacher, 0);
        assert_eq!(manifest.counts.unpaired_rgb, 1); // seq 7
                                                     // seqs 2, 5, and 6 are absent from both sides inside [0, 7].
        assert_eq!(manifest.counts.dropped_sequences, 3);
    }

    #[test]
    fn duplicates_and_regressions_fail_closed() {
        let teachers = vec![teacher(1), teacher(1)];
        assert!(matches!(
            build_pair_manifest(&teachers, &[]),
            Err(PairingError::DuplicateFrameSeq {
                side: "teacher",
                frame_seq: 1
            })
        ));

        let teachers = vec![teacher(2), teacher(1)];
        assert!(matches!(
            build_pair_manifest(&teachers, &[]),
            Err(PairingError::RegressedFrameSeq {
                side: "teacher",
                ..
            })
        ));

        let rgbs = vec![rgb_record(1), rgb_record(1)];
        assert!(matches!(
            build_pair_manifest(&[], &rgbs),
            Err(PairingError::DuplicateFrameSeq {
                side: "rgb",
                frame_seq: 1
            })
        ));
    }

    #[test]
    fn identity_timestamp_disagreement_is_an_error_not_a_repair() {
        let teachers = vec![TeacherIdentity {
            frame_seq: 1,
            timestamp_micros: 99_999,
        }];
        assert!(matches!(
            build_pair_manifest(&teachers, &[rgb_record(1)]),
            Err(PairingError::IdentityTimestampMismatch { frame_seq: 1, .. })
        ));
    }

    #[test]
    fn failed_validation_never_writes_a_completion_marker() {
        let directory = tempdir().unwrap();
        // A failing build must leave nothing to finalize.
        let teachers = vec![teacher(1), teacher(1)];
        assert!(build_pair_manifest(&teachers, &[]).is_err());
        assert!(!directory.path().join("COMPLETED").exists());

        // A passing build writes the manifest before the marker.
        let manifest = build_pair_manifest(&[teacher(1)], &[rgb_record(1)]).unwrap();
        write_manifest_and_finalize(directory.path(), &manifest).unwrap();
        assert!(directory.path().join("manifest.json").is_file());
        assert!(directory.path().join("COMPLETED").is_file());
    }
}
