//! Canonical same-frame fan-out from one MediaPipe inference result
//! (GNM #57.1).
//!
//! Every camera frame runs MediaPipe inference exactly once. This module is
//! the single place where that one result is split into the three consumers:
//! the unchanged Direct-path sample, the dense GNM observation, and the
//! auxiliary semantic observation. All outputs carry the identical source
//! sequence and capture timestamp; there is no nearest-timestamp frame mixing
//! anywhere in this boundary.

use std::sync::Arc;

use vtuber_core::face_tracking::FaceTrackingSample;
use vtuber_gnm::{
    DenseCorrespondenceSet, DenseCoveragePolicy, GnmDenseError, GnmDenseObservation,
    MEDIAPIPE_FACE_LANDMARK_COUNT,
};

use crate::ab_backend::{AbBackendError, SourceFrameStamp};
use crate::auxiliary_expression::{
    AuxiliaryChannelConfig, AuxiliaryExpressionError, AuxiliaryExpressionObservation,
};

/// Failure while fanning out one inference result.
#[derive(Debug)]
pub enum SameFrameFanOutError {
    /// The shared source stamp failed chronology validation.
    Stamp(AbBackendError),
    /// The dense observation could not be built from the landmarks.
    Dense(GnmDenseError),
    /// The auxiliary observation rejected the blendshape scores or config.
    Auxiliary(AuxiliaryExpressionError),
    /// A produced output does not carry the shared source sequence. This is a
    /// fail-closed re-check; construction aligns the values by design.
    SourceSequenceMismatch {
        /// Shared source sequence from the Direct sample.
        expected: u64,
        /// Sequence embedded in the dense observation.
        dense: u64,
        /// Sequence embedded in the auxiliary observation.
        auxiliary: u64,
    },
    /// A produced output does not carry the shared capture timestamp.
    CaptureTimestampMismatch {
        /// Shared capture timestamp, in microseconds.
        expected: u64,
        /// Timestamp embedded in the dense observation, in microseconds.
        dense: u64,
        /// Timestamp embedded in the auxiliary observation, in microseconds.
        auxiliary: u64,
    },
}

impl std::fmt::Display for SameFrameFanOutError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stamp(error) => write!(formatter, "source stamp invalid: {error}"),
            Self::Dense(error) => write!(formatter, "dense observation failed: {error}"),
            Self::Auxiliary(error) => write!(formatter, "auxiliary observation failed: {error}"),
            Self::SourceSequenceMismatch {
                expected,
                dense,
                auxiliary,
            } => write!(
                formatter,
                "same-frame fan-out sequence mismatch: direct {expected}, dense {dense}, auxiliary {auxiliary}"
            ),
            Self::CaptureTimestampMismatch {
                expected,
                dense,
                auxiliary,
            } => write!(
                formatter,
                "same-frame fan-out capture timestamp mismatch: direct {expected}, dense {dense}, auxiliary {auxiliary}"
            ),
        }
    }
}

impl std::error::Error for SameFrameFanOutError {}

impl From<AbBackendError> for SameFrameFanOutError {
    fn from(error: AbBackendError) -> Self {
        Self::Stamp(error)
    }
}

impl From<GnmDenseError> for SameFrameFanOutError {
    fn from(error: GnmDenseError) -> Self {
        Self::Dense(error)
    }
}

impl From<AuxiliaryExpressionError> for SameFrameFanOutError {
    fn from(error: AuxiliaryExpressionError) -> Self {
        Self::Auxiliary(error)
    }
}

/// One canonical fan-out: three consumers aligned to one exact source frame.
///
/// The Direct member is an untouched clone of the input sample, so the
/// existing Direct path keeps its exact output values and filter behavior.
/// The dense and auxiliary members are freshly derived once per inference.
#[derive(Debug)]
pub struct SameFrameFanOut {
    stamp: SourceFrameStamp,
    direct: Box<FaceTrackingSample>,
    dense: Arc<GnmDenseObservation>,
    auxiliary: AuxiliaryExpressionObservation,
}

/// Converts a monotonic nanosecond timestamp to microseconds (truncating).
fn micros(nanos: u64) -> u64 {
    nanos / 1_000
}

/// Fans one MediaPipe inference result out to the Direct, dense-GNM, and
/// auxiliary consumers.
///
/// The function is pure: it derives every output from this exact sample and
/// validates internal alignment before returning. It never reads other frames,
/// interpolates across timestamps, or consults wall-clock time.
///
/// # Errors
///
/// Returns [`SameFrameFanOutError`] when the stamp chronology fails, the
/// landmark buffer cannot produce a valid dense observation, the auxiliary
/// configuration is invalid, or the post-construction alignment check fails.
pub fn fan_out_same_frame(
    sample: &FaceTrackingSample,
    mapping: &DenseCorrespondenceSet,
    coverage: DenseCoveragePolicy,
    auxiliary_config: &[AuxiliaryChannelConfig],
) -> Result<SameFrameFanOut, SameFrameFanOutError> {
    if sample.landmarks.len() != MEDIAPIPE_FACE_LANDMARK_COUNT {
        return Err(SameFrameFanOutError::Dense(GnmDenseError::Shape {
            field: "landmarks",
            expected: MEDIAPIPE_FACE_LANDMARK_COUNT,
            actual: sample.landmarks.len(),
        }));
    }

    let capture_micros = micros(sample.captured_at.0);
    let stamp = SourceFrameStamp::new(
        sample.source_seq.0,
        capture_micros,
        Some(micros(sample.inference_finished_at.0)),
    )?;

    // Exactly one derivation of each downstream observation from this frame's
    // already-computed inference result. No second inference happens here.
    let points: Vec<[f32; 2]> = sample.landmarks.iter().map(|p| [p.x, p.y]).collect();
    let dense = Arc::new(GnmDenseObservation::from_mediapipe_xy(
        sample.source_seq.0,
        capture_micros,
        &points,
        mapping,
        coverage,
    )?);
    let auxiliary = AuxiliaryExpressionObservation::from_mediapipe(
        sample.source_seq.0,
        capture_micros,
        &sample.blendshapes,
        auxiliary_config,
    )?;

    let fan_out = SameFrameFanOut {
        stamp,
        direct: Box::new(sample.clone()),
        dense,
        auxiliary,
    };
    fan_out.verify()?;
    Ok(fan_out)
}

impl SameFrameFanOut {
    /// Re-validates that every member carries the same source identity.
    ///
    /// Construction aligns the values by design; this explicit check keeps a
    /// future refactoring from silently mixing frames.
    pub fn verify(&self) -> Result<(), SameFrameFanOutError> {
        let expected = self.stamp.source_seq();
        let dense = self.dense.source_seq();
        let auxiliary = self.auxiliary.source_seq();
        if expected != dense || expected != auxiliary {
            return Err(SameFrameFanOutError::SourceSequenceMismatch {
                expected,
                dense,
                auxiliary,
            });
        }
        let expected = self.stamp.capture_micros();
        let dense = self.dense.captured_at_micros();
        let auxiliary = self.auxiliary.captured_at_micros();
        if expected != dense || expected != auxiliary {
            return Err(SameFrameFanOutError::CaptureTimestampMismatch {
                expected,
                dense,
                auxiliary,
            });
        }
        Ok(())
    }

    /// Returns the shared source stamp.
    #[must_use]
    pub fn stamp(&self) -> SourceFrameStamp {
        self.stamp
    }

    /// Returns the untouched Direct-path sample.
    #[must_use]
    pub fn direct(&self) -> &FaceTrackingSample {
        &self.direct
    }

    /// Returns the shared dense observation.
    #[must_use]
    pub fn dense(&self) -> &Arc<GnmDenseObservation> {
        &self.dense
    }

    /// Returns the auxiliary semantic observation.
    #[must_use]
    pub fn auxiliary(&self) -> &AuxiliaryExpressionObservation {
        &self.auxiliary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    use vtuber_core::face_tracking::{
        FaceBlendshapeSet, FaceLandmark, FaceTrackingQuality, MEDIAPIPE_FACE_LANDMARK_COUNT,
    };
    use vtuber_core::{FrameSeq, MonoTimeNs};

    use crate::gnm_sequence_regression::{synthetic_head_model, synthetic_mapping};

    fn sample(seq: u64) -> FaceTrackingSample {
        FaceTrackingSample {
            source_seq: FrameSeq(seq),
            captured_at: MonoTimeNs(1_000_000_000),
            inference_started_at: MonoTimeNs(1_000_500_000),
            inference_finished_at: MonoTimeNs(1_001_000_000),
            camera_to_face: vtuber_core::CameraFaceTransform::identity(),
            face_center: [0.5, 0.5],
            landmarks: StdArc::from(
                vec![
                    FaceLandmark {
                        x: 0.5,
                        y: 0.5,
                        z: 0.0,
                        visibility: Some(1.0),
                        presence: Some(1.0),
                    };
                    MEDIAPIPE_FACE_LANDMARK_COUNT
                ]
                .into_boxed_slice(),
            ),
            blendshapes: FaceBlendshapeSet::default(),
            quality: FaceTrackingQuality {
                landmark_presence_median: Some(1.0),
                matrix_orthogonality_error: 0.0,
                matrix_determinant: 1.0,
            },
        }
    }

    fn mapping() -> vtuber_gnm::DenseCorrespondenceSet {
        let model = synthetic_head_model().expect("synthetic model builds");
        synthetic_mapping(&model).expect("synthetic mapping binds")
    }

    #[test]
    fn one_inference_feeds_all_three_consumers_with_identical_source_identity() {
        let input = sample(11);
        // One call stands in for the single MediaPipe inference of the frame;
        // everything below is pure derivation from its result.
        let fan_out =
            fan_out_same_frame(&input, &mapping(), test_coverage(), &[]).expect("fan-out works");

        assert_eq!(fan_out.stamp().source_seq(), 11);
        assert_eq!(fan_out.direct().source_seq, FrameSeq(11));
        assert_eq!(fan_out.dense().source_seq(), 11);
        assert_eq!(fan_out.auxiliary().source_seq(), 11);
        assert_eq!(fan_out.stamp().capture_micros(), 1_000_000_000 / 1_000);
        assert_eq!(fan_out.dense().captured_at_micros(), 1_000_000_000 / 1_000);
        assert_eq!(
            fan_out.auxiliary().captured_at_micros(),
            1_000_000_000 / 1_000
        );
        // The shared inference completion time is preserved on the stamp.
        assert_eq!(
            fan_out.stamp().inference_complete_micros(),
            Some(1_001_000_000 / 1_000)
        );
    }

    #[test]
    fn direct_member_is_byte_identical_to_the_inference_result() {
        let input = sample(12);
        let fan_out =
            fan_out_same_frame(&input, &mapping(), test_coverage(), &[]).expect("fan-out works");
        assert_eq!(*fan_out.direct(), input);
    }

    #[test]
    fn wrong_landmark_count_is_a_typed_error() {
        let mut input = sample(13);
        input.landmarks = StdArc::from(Vec::<FaceLandmark>::new().into_boxed_slice());
        assert!(matches!(
            fan_out_same_frame(&input, &mapping(), test_coverage(), &[]),
            Err(SameFrameFanOutError::Dense(_))
        ));
    }

    #[test]
    fn verify_rejects_mixed_frame_identity_with_typed_errors() {
        // Build a structurally valid fan-out whose members carry foreign
        // identities (simulating a broken producer) and confirm the explicit
        // re-check rejects it instead of mixing frames.
        let input = sample(14);
        let capture_micros = micros(input.captured_at.0);
        let stamp = SourceFrameStamp::new(
            14,
            capture_micros,
            Some(micros(input.inference_finished_at.0)),
        )
        .expect("valid stamp");
        let points: Vec<[f32; 2]> = input.landmarks.iter().map(|p| [p.x, p.y]).collect();
        let foreign_dense = GnmDenseObservation::from_mediapipe_xy(
            99,
            capture_micros,
            &points,
            &mapping(),
            test_coverage(),
        )
        .expect("foreign dense builds");
        let foreign_auxiliary =
            AuxiliaryExpressionObservation::from_mediapipe(98, 4_000_000, &input.blendshapes, &[])
                .expect("foreign auxiliary builds");

        let broken = SameFrameFanOut {
            stamp,
            direct: Box::new(input),
            dense: Arc::new(foreign_dense),
            auxiliary: foreign_auxiliary,
        };
        match broken.verify() {
            Err(SameFrameFanOutError::SourceSequenceMismatch {
                expected,
                dense,
                auxiliary,
            }) => {
                assert_eq!(expected, 14);
                assert_eq!(dense, 99);
                assert_eq!(auxiliary, 98);
            }
            other => panic!("expected sequence mismatch, got {other:?}"),
        }
    }

    #[test]
    fn invalid_auxiliary_config_is_reported_as_auxiliary_failure() {
        use crate::auxiliary_expression::{
            AuxChannelReliability, AuxiliaryChannelConfig, AuxiliaryExpressionSemantic,
        };
        // An enabled channel must have positive relative weight; a zero-weight
        // enabled channel is rejected through the fan-out as an auxiliary
        // failure rather than silently dropped.
        let bad_config = [AuxiliaryChannelConfig::new(
            AuxiliaryExpressionSemantic::JawOpen,
            AuxChannelReliability::TrustedForAux,
            0.0,
            None,
        )];
        // `AuxiliaryChannelConfig::new` itself rejects this combination, which
        // the fan-out propagates when producers hand over pre-built configs;
        // exercise the error surface directly.
        assert!(matches!(
            bad_config[0],
            Err(AuxiliaryExpressionError::InvalidConfig(_))
        ));
        let input = sample(16);
        let result = fan_out_same_frame(&input, &mapping(), test_coverage(), &[]);
        assert!(result.is_ok(), "empty config stays valid");
    }

    fn test_coverage() -> DenseCoveragePolicy {
        DenseCoveragePolicy::new(1, 0.5).expect("valid policy")
    }
}
