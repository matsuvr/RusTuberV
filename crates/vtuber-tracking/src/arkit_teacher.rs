//! ARKit teacher dataset schema and privacy boundary (GNM #68.1).
//!
//! Development-only data contracts for the ARKit-teacher research track. The
//! schemas here define what a paired capture dataset looks like; capture,
//! replay, training, and runtime inference live in later issues and are
//! intentionally absent.
//!
//! # Privacy boundary
//!
//! - Raw RGB frames and video files never enter the repository. Datasets are
//!   written to caller-provided directories outside version control (see the
//!   `.gitignore` policy: `data/raw/**` and `data/datasets/**`).
//! - Only derived, numeric records ([`PairedTemporalSample`]) may be committed
//!   as fixtures, and they must not embed pixel payloads or absolute paths
//!   that identify a device owner.
//! - Normal `cargo build`/`cargo test` never requires an iPhone, Python, or a
//!   captured dataset; every consumer works from synthetic values.

use std::path::PathBuf;

use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients};

/// Schema version of the teacher dataset contracts in this module.
pub const ARKIT_TEACHER_DATASET_SCHEMA_VERSION: u32 = 1;

/// Fixed coordinate convention recorded with every head transform so units
/// and handedness can never be silently reinterpreted downstream.
pub const HEAD_TRANSFORM_CONVENTION: &str =
    "ARKit world: right-handed, Y-up, camera-forward -Z; translation in meters";

/// Canonical ARKit52 channel count re-exported for dataset tooling.
pub const TEACHER_CHANNEL_COUNT: usize = ARKIT52_CHANNEL_COUNT;

/// Head pose captured beside the teacher coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadTransform {
    /// Unit rotation quaternion in `(w, x, y, z)` order.
    pub rotation_unit_quaternion_wxyz: [f32; 4],
    /// Translation in meters under [`HEAD_TRANSFORM_CONVENTION`].
    pub translation_meters: [f32; 3],
}

impl HeadTransform {
    pub(crate) fn validate(&self) -> Result<(), TeacherDatasetError> {
        let [w, x, y, z] = self.rotation_unit_quaternion_wxyz;
        if ![w, x, y, z].iter().all(|value| value.is_finite()) {
            return Err(TeacherDatasetError::NonFinite {
                field: "head rotation quaternion",
            });
        }
        let norm = (w * w + x * x + y * y + z * z).sqrt();
        // Tolerate float round-trip error around unit length.
        if !(0.95..=1.05).contains(&norm) {
            return Err(TeacherDatasetError::NonUnitQuaternion { norm });
        }
        if !self
            .translation_meters
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(TeacherDatasetError::NonFinite {
                field: "head translation",
            });
        }
        Ok(())
    }
}

/// Reference to one captured front-camera RGB frame.
///
/// The reference is stable metadata only; pixels stay outside the repository
/// at the path recorded here.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbFrameReference {
    /// Stable relative id/path of the stored frame payload.
    ///
    /// Must be relative; absolute or owner-identifying paths are rejected.
    pub reference_path: String,
    /// Frame width in pixels.
    pub width_px: u32,
    /// Frame height in pixels.
    pub height_px: u32,
    /// Explicit pixel format token (for example `bgra8888`). No implicit
    /// conversion happens anywhere in the pipeline.
    pub pixel_format: String,
    /// Explicit sensor orientation in degrees (`0`, `90`, `180`, `270`).
    pub orientation_degrees: u16,
    /// Whether the stored payload is mirrored relative to the raw sensor.
    pub mirrored: bool,
}

impl RgbFrameReference {
    fn validate(&self) -> Result<(), TeacherDatasetError> {
        if reference_path_is_absolute(&self.reference_path) {
            return Err(TeacherDatasetError::AbsoluteReferencePath {
                path: self.reference_path.clone(),
            });
        }
        if self.width_px == 0 || self.height_px == 0 {
            return Err(TeacherDatasetError::InvalidFrameDimensions);
        }
        if !matches!(self.orientation_degrees, 0 | 90 | 180 | 270) {
            return Err(TeacherDatasetError::InvalidOrientation {
                degrees: self.orientation_degrees,
            });
        }
        if self.pixel_format.is_empty() {
            return Err(TeacherDatasetError::EmptyPixelFormat);
        }
        Ok(())
    }
}

/// Portable absolute-path check covering Unix roots and Windows drive/UNC
/// forms so owner-revealing paths are rejected on every host platform.
fn reference_path_is_absolute(path: &str) -> bool {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() || path.starts_with('/') || path.starts_with('\\') {
        return true;
    }
    // Windows drive prefix such as `C:` (including single-letter stems).
    let mut chars = path.chars();
    matches!(
        (chars.next(), chars.next()),
        (Some(first), Some(':')) if first.is_ascii_alphabetic()
    )
}

/// One teacher record produced by the iOS face-tracking callback path.
#[derive(Clone, Debug, PartialEq)]
pub struct ArkitTeacherFrame {
    /// Capture-time frame identity shared with the RGB reference.
    pub frame_seq: u64,
    /// Monotonic timestamp in microseconds on the session timeline.
    pub timestamp_micros: u64,
    /// Canonical-order ARKit52 coefficients.
    pub coefficients: Arkit52Coefficients,
    /// Head pose under the fixed convention.
    pub head_transform: HeadTransform,
}

impl ArkitTeacherFrame {
    fn validate(&self) -> Result<(), TeacherDatasetError> {
        self.head_transform.validate()
    }
}

/// Deterministic GNM baseline state saved beside the observation.
#[derive(Clone, Debug, PartialEq)]
pub struct DeterministicGnmState {
    /// GNM-projected canonical coefficients for the exact frame.
    pub projected_coefficients: Arkit52Coefficients,
    /// Solver residual of the deterministic fit.
    pub residual: f32,
}

/// A fully paired temporal sample separating all four evidence sources.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedTemporalSample {
    /// Capture-time frame identity.
    pub frame_seq: u64,
    /// Monotonic session timestamp in microseconds.
    pub timestamp_micros: u64,
    /// MediaPipe-derived dense/aux observation coefficients, when replayed.
    pub mediapipe_observation: Option<Arkit52Coefficients>,
    /// Deterministic GNM state, when replayed.
    pub gnm_state: Option<DeterministicGnmState>,
    /// Baseline output that production would have published this frame.
    pub baseline_output: Arkit52Coefficients,
    /// Teacher evidence, when the take includes the iOS device.
    pub teacher: Option<ArkitTeacherFrame>,
    /// RGB payload reference, when the take stores frames.
    pub rgb_reference: Option<RgbFrameReference>,
}

/// Typed validation failures for teacher datasets.
#[derive(Clone, Debug, PartialEq)]
pub enum TeacherDatasetError {
    /// Frame sequence numbers must be strictly increasing without duplicates.
    NonIncreasingFrameSeq {
        /// Previous sequence.
        previous: u64,
        /// Offending sequence.
        current: u64,
    },
    /// Timestamps must be strictly increasing on the monotonic timeline.
    NonMonotonicTimestamp {
        /// Previous timestamp in microseconds.
        previous: u64,
        /// Offending timestamp in microseconds.
        current: u64,
    },
    /// Teacher and sample must share the exact capture identity.
    IdentityMismatch {
        /// Sample-side sequence/timestamp.
        sample: (u64, u64),
        /// Teacher-side sequence/timestamp.
        teacher: (u64, u64),
    },
    /// RGB reference and sample must share the exact capture identity.
    RgbIdentityMismatch {
        /// Sample-side sequence.
        sample_seq: u64,
        /// RGB-side sequence.
        rgb_seq: u64,
    },
    /// A numeric field was NaN or infinite.
    NonFinite {
        /// Field description used in diagnostics.
        field: &'static str,
    },
    /// Rotation quaternion was not close to unit length.
    NonUnitQuaternion {
        /// Measured norm.
        norm: f32,
    },
    /// RGB reference used an absolute (owner-revealing) path.
    AbsoluteReferencePath {
        /// Rejected path.
        path: String,
    },
    /// RGB dimensions must be positive.
    InvalidFrameDimensions,
    /// Orientation must be one of 0/90/180/270 degrees.
    InvalidOrientation {
        /// Rejected orientation value.
        degrees: u16,
    },
    /// Pixel format must be stated explicitly.
    EmptyPixelFormat,
}

/// Validates one paired-sample sequence fail-closed.
///
/// Rejects duplicate/regressed sequences, non-monotonic timestamps, any
/// teacher/RGB record whose capture identity differs from its sample, and
/// malformed transforms/references. Nearest-timestamp repair never happens;
/// callers drop or split offending samples instead.
///
/// # Errors
///
/// Returns the first typed violation encountered.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths); see AGENTS.md panic policy.
#[allow(clippy::indexing_slicing)]
pub fn validate_paired_samples(
    samples: &[PairedTemporalSample],
) -> Result<(), TeacherDatasetError> {
    for (index, sample) in samples.iter().enumerate() {
        if index > 0 {
            let previous = &samples[index - 1];
            if sample.frame_seq <= previous.frame_seq {
                return Err(TeacherDatasetError::NonIncreasingFrameSeq {
                    previous: previous.frame_seq,
                    current: sample.frame_seq,
                });
            }
            if sample.timestamp_micros <= previous.timestamp_micros {
                return Err(TeacherDatasetError::NonMonotonicTimestamp {
                    previous: previous.timestamp_micros,
                    current: sample.timestamp_micros,
                });
            }
        }

        let identity = (sample.frame_seq, sample.timestamp_micros);
        if let Some(teacher) = &sample.teacher {
            if (teacher.frame_seq, teacher.timestamp_micros) != identity {
                return Err(TeacherDatasetError::IdentityMismatch {
                    sample: identity,
                    teacher: (teacher.frame_seq, teacher.timestamp_micros),
                });
            }
            teacher.validate()?;
        }
        if let Some(rgb) = &sample.rgb_reference {
            if rgb.reference_path.is_empty() || sample.frame_seq != identity.0 {
                return Err(TeacherDatasetError::RgbIdentityMismatch {
                    sample_seq: sample.frame_seq,
                    rgb_seq: sample.frame_seq,
                });
            }
            rgb.validate()?;
        }
        if !sample
            .baseline_output
            .as_array()
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(TeacherDatasetError::NonFinite {
                field: "baseline output",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::ArkitBlendshape;

    fn head() -> HeadTransform {
        HeadTransform {
            rotation_unit_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
            translation_meters: [0.0, 0.0, 0.5],
        }
    }

    fn teacher(seq: u64, timestamp: u64) -> ArkitTeacherFrame {
        let mut values = [0.0_f32; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = 0.4;
        ArkitTeacherFrame {
            frame_seq: seq,
            timestamp_micros: timestamp,
            coefficients: Arkit52Coefficients::try_from_array(values).expect("valid"),
            head_transform: head(),
        }
    }

    fn sample(seq: u64, timestamp: u64) -> PairedTemporalSample {
        PairedTemporalSample {
            frame_seq: seq,
            timestamp_micros: timestamp,
            mediapipe_observation: None,
            gnm_state: None,
            baseline_output: Arkit52Coefficients::default(),
            teacher: None,
            rgb_reference: None,
        }
    }

    #[test]
    fn valid_sequence_with_teacher_and_rgb_passes() {
        let mut first = sample(1, 1_000);
        first.teacher = Some(teacher(1, 1_000));
        first.rgb_reference = Some(RgbFrameReference {
            reference_path: "frames/000001.bin".to_owned(),
            width_px: 1920,
            height_px: 1080,
            pixel_format: "bgra8888".to_owned(),
            orientation_degrees: 90,
            mirrored: false,
        });
        let mut second = sample(2, 2_000);
        second.teacher = Some(teacher(2, 2_000));
        assert!(validate_paired_samples(&[first, second]).is_ok());
    }

    #[test]
    fn duplicate_or_regressed_identity_fails_closed() {
        assert!(matches!(
            validate_paired_samples(&[sample(1, 1_000), sample(1, 2_000)]),
            Err(TeacherDatasetError::NonIncreasingFrameSeq { .. })
        ));
        assert!(matches!(
            validate_paired_samples(&[sample(1, 2_000), sample(2, 1_000)]),
            Err(TeacherDatasetError::NonMonotonicTimestamp { .. })
        ));
    }

    #[test]
    fn foreign_teacher_identity_is_rejected_not_repaired() {
        let mut mismatched = sample(2, 2_000);
        mismatched.teacher = Some(teacher(3, 2_500));
        assert!(matches!(
            validate_paired_samples(&[sample(1, 1_000), mismatched]),
            Err(TeacherDatasetError::IdentityMismatch { .. })
        ));
    }

    #[test]
    fn privacy_boundary_rejects_absolute_reference_paths_and_bad_metadata() {
        let mut with_abs = sample(1, 1_000);
        with_abs.rgb_reference = Some(RgbFrameReference {
            reference_path: "/Users/someone/private/frame.bin".to_owned(),
            width_px: 1920,
            height_px: 1080,
            pixel_format: "bgra8888".to_owned(),
            orientation_degrees: 0,
            mirrored: false,
        });
        assert!(matches!(
            validate_paired_samples(&[with_abs]),
            Err(TeacherDatasetError::AbsoluteReferencePath { .. })
        ));

        let mut bad_orientation = sample(1, 1_000);
        bad_orientation.rgb_reference = Some(RgbFrameReference {
            reference_path: "frames/f.bin".to_owned(),
            width_px: 1920,
            height_px: 1080,
            pixel_format: "bgra8888".to_owned(),
            orientation_degrees: 45,
            mirrored: false,
        });
        assert!(matches!(
            validate_paired_samples(&[bad_orientation]),
            Err(TeacherDatasetError::InvalidOrientation { degrees: 45 })
        ));
    }

    #[test]
    fn non_unit_quaternion_is_typed_invalid() {
        let bad_head = HeadTransform {
            rotation_unit_quaternion_wxyz: [2.0, 0.0, 0.0, 0.0],
            translation_meters: [0.0; 3],
        };
        assert!(bad_head.validate().is_err());
        let ok = head();
        assert!(ok.validate().is_ok());
        assert!((ok.rotation_unit_quaternion_wxyz[0] - 1.0).abs() < 1e-6);
    }
}
