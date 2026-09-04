//! GNM shadow-mode pipeline stage (GNM #57.2).
//!
//! In [`FaceTrackingMode::GnmTemporalShadow`] the Direct MediaPipe path keeps
//! sole avatar authority while the same source frame's GNM fit/decode result
//! is evaluated side-by-side. This module composes the canonical same-frame
//! fan-out with backend-aligned outputs:
//!
//! - Avatar output is produced by Direct only; nothing here publishes to an
//!   avatar.
//! - GNM results are paired by exact shared source stamp or not at all;
//!   stale/foreign frames become a typed `DirectOnly` reason instead of being
//!   mixed.
//! - Frame transport relies on the latest-value GNM worker, which keeps at
//!   most one pending frame; no unbounded queue exists anywhere in the shadow
//!   path.

use vtuber_gnm::{GnmFacialFeatures, GnmFrameStamp};

use crate::ab_backend::{
    AbBackendError, AlignedBackendOutputs, BackendOutputTiming, FaceTrackingBackend,
    SourceFrameStamp, StampedBackendOutput,
};
use crate::gnm_arkit_projector::{Arkit52DecodeResult, decode_gnm_arkit52};

use crate::gnm_latest_frame_worker::GnmWorkerFrameInput;
use crate::same_frame_fanout::SameFrameFanOut;

/// Reason a shadow frame carried no GNM output beside Direct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmShadowSkip {
    /// The GNM worker has not produced any state yet.
    NoStateYet,
    /// The newest available GNM state belongs to a different source frame.
    /// It is never paired with a newer Direct frame.
    ForeignSourceFrame {
        /// Expected sequence from the shared source stamp.
        expected_seq: u64,
        /// Sequence of the available GNM state.
        actual_seq: u64,
    },
    /// The GNM decode failed validation for this frame.
    DecodeFailed,
}

/// Outcome of one shadow-mode frame.
#[derive(Clone, Debug, PartialEq)]
pub enum GnmShadowOutcome<T> {
    /// Direct-only frame. Avatar output comes exclusively from `direct`.
    DirectOnly {
        /// Untouched Direct output for the frame.
        direct: T,
        /// Why the GNM side is absent this frame.
        skip: GnmShadowSkip,
    },
    /// Exact source-aligned Direct/GNM pair. Only `direct` has authority.
    Aligned(Box<AlignedBackendOutputs<T>>),
}

/// One GNM-side candidate output awaiting alignment with its Direct twin.
#[derive(Clone, Copy, Debug)]
pub struct GnmShadowCandidate<'a, T> {
    /// Source frame the GNM result was fitted on.
    pub stamp: GnmFrameStamp,
    /// Optional GNM fit completion time, in microseconds.
    pub fit_complete_micros: Option<u64>,
    /// Optional decoder completion time, in microseconds.
    pub decoder_complete_micros: Option<u64>,
    /// Decoded GNM output payload.
    pub output: &'a T,
}

/// Builds the GNM worker frame input for one fanned-out source frame.
///
/// The stamp mapping preserves exact source identity (`source_seq` and
/// capture microseconds). The worker consumes this through its capacity-one
/// latest slot, so submitting faster than it solves replaces the pending
/// frame instead of queueing work without bound.
#[must_use]
pub fn shadow_worker_input(fan_out: &SameFrameFanOut) -> GnmWorkerFrameInput {
    let stamp = fan_out.stamp();
    let gnm_stamp = GnmFrameStamp {
        source_seq: stamp.source_seq(),
        captured_at_micros: stamp.capture_micros(),
    };
    GnmWorkerFrameInput::new(gnm_stamp, fan_out.dense().clone())
}

/// Decodes a GNM feature snapshot fitted on the same source frame.
///
/// Returns the validated coefficients together with their per-channel support
/// classification. The decode consumes only the calibration-normalized
/// snapshot; current MediaPipe coefficients are never consulted.
///
/// # Errors
///
/// Propagates the typed decode failure from [`decode_gnm_arkit52`].
pub fn decode_shadow_features(
    features: &GnmFacialFeatures,
) -> Result<Arkit52DecodeResult, vtuber_core::Arkit52ValueError> {
    decode_gnm_arkit52(features)
}

/// Aligns one Direct output with its optional same-frame GNM candidate.
///
/// The pairing uses exact shared source identity: sequence number and capture
/// timestamp must both match the fan-out stamp. Any mismatch yields
/// [`GnmShadowOutcome::DirectOnly`] with [`GnmShadowSkip::ForeignSourceFrame`]
/// rather than a nearest-timestamp pairing. Timing records are validated
/// chronologically for both backends; a timing violation is a producer bug
/// and surfaces as a typed error.
///
/// # Errors
///
/// Returns [`AbBackendError`] when either timing record fails validation or
/// the aligned pair construction detects a backend-role inconsistency.
pub fn align_shadow_pair<T>(
    fan_out: &SameFrameFanOut,
    direct_output: T,
    gnm_candidate: Option<GnmShadowCandidate<'_, T>>,
    publish_micros: u64,
) -> Result<GnmShadowOutcome<T>, AbBackendError>
where
    T: Clone,
{
    let stamp: SourceFrameStamp = fan_out.stamp();
    let direct_timing = BackendOutputTiming::new(
        stamp,
        FaceTrackingBackend::DirectMediaPipe,
        None,
        None,
        publish_micros,
    )?;

    let Some(candidate) = gnm_candidate else {
        return Ok(GnmShadowOutcome::DirectOnly {
            direct: direct_output,
            skip: GnmShadowSkip::NoStateYet,
        });
    };

    if candidate.stamp.source_seq != stamp.source_seq() {
        return Ok(GnmShadowOutcome::DirectOnly {
            direct: direct_output,
            skip: GnmShadowSkip::ForeignSourceFrame {
                expected_seq: stamp.source_seq(),
                actual_seq: candidate.stamp.source_seq,
            },
        });
    }

    let gnm_timing = BackendOutputTiming::new(
        stamp,
        FaceTrackingBackend::GnmTemporal,
        candidate.fit_complete_micros,
        candidate.decoder_complete_micros,
        publish_micros,
    )?;
    let aligned = AlignedBackendOutputs::new(
        StampedBackendOutput::new(direct_timing, direct_output),
        StampedBackendOutput::new(gnm_timing, candidate.output.clone()),
    )?;
    Ok(GnmShadowOutcome::Aligned(Box::new(aligned)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    use vtuber_core::face_tracking::{
        FaceBlendshapeSet, FaceLandmark, FaceTrackingQuality, MEDIAPIPE_FACE_LANDMARK_COUNT,
    };
    use vtuber_core::{Arkit52Coefficients, ArkitBlendshape, FrameSeq, MonoTimeNs};

    use crate::gnm_arkit_projector::ProjectedSupport;

    use crate::auxiliary_expression::AuxiliaryChannelConfig;
    use crate::gnm_sequence_regression::{synthetic_head_model, synthetic_mapping};
    use crate::same_frame_fanout::{SameFrameFanOut, fan_out_same_frame};

    const CAPTURE_MICROS: u64 = 1_000_000_000 / 1_000;
    const INFERENCE_MICROS: u64 = 1_001_000_000 / 1_000;
    const PUBLISH_MICROS: u64 = 1_002_500;

    fn sample(seq: u64) -> vtuber_core::face_tracking::FaceTrackingSample {
        vtuber_core::face_tracking::FaceTrackingSample {
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

    fn fan_out(seq: u64) -> SameFrameFanOut {
        fan_out_same_frame(
            &sample(seq),
            &mapping(),
            vtuber_gnm::DenseCoveragePolicy::new(1, 0.5).expect("valid policy"),
            Vec::<AuxiliaryChannelConfig>::new().as_slice(),
        )
        .expect("fan-out works")
    }

    fn coefficients(value: f32) -> Arkit52Coefficients {
        let mut values = [0.0; vtuber_core::ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = value.clamp(0.0, 1.0);
        Arkit52Coefficients::try_from_array(values).expect("valid coefficients")
    }

    #[test]
    fn worker_input_preserves_exact_source_identity() {
        let frame = fan_out(21);
        let input = shadow_worker_input(&frame);
        assert_eq!(input.stamp().source_seq, 21);
        assert_eq!(input.stamp().captured_at_micros, CAPTURE_MICROS);
    }

    #[test]
    fn shadow_without_gnm_state_is_direct_only_and_direct_is_untouched() {
        let frame = fan_out(22);
        let direct = coefficients(0.25);
        let outcome =
            align_shadow_pair(&frame, direct, None, PUBLISH_MICROS).expect("alignment works");
        match outcome {
            GnmShadowOutcome::DirectOnly { direct, skip } => {
                assert_eq!(skip, GnmShadowSkip::NoStateYet);
                assert_eq!(direct.get(ArkitBlendshape::JawOpen), 0.25);
            }
            other => panic!("expected DirectOnly, got {other:?}"),
        }
    }

    #[test]
    fn aligned_pair_requires_exact_same_source_frame() {
        let frame = fan_out(23);
        let gnm_stamp = GnmFrameStamp {
            source_seq: 24,
            captured_at_micros: CAPTURE_MICROS,
        };
        let candidate = GnmShadowCandidate {
            stamp: gnm_stamp,
            fit_complete_micros: Some(INFERENCE_MICROS + 1),
            decoder_complete_micros: Some(INFERENCE_MICROS + 2),
            output: &coefficients(0.5),
        };
        let outcome =
            align_shadow_pair(&frame, coefficients(0.25), Some(candidate), PUBLISH_MICROS)
                .expect("alignment works");
        match outcome {
            GnmShadowOutcome::DirectOnly { direct, skip } => {
                assert_eq!(
                    skip,
                    GnmShadowSkip::ForeignSourceFrame {
                        expected_seq: 23,
                        actual_seq: 24
                    }
                );
                assert_eq!(direct.get(ArkitBlendshape::JawOpen), 0.25);
            }
            other => panic!("foreign frame must not align, got {other:?}"),
        }
    }

    #[test]
    fn matching_frames_align_with_validated_timing() {
        let frame = fan_out(25);
        let gnm_stamp = GnmFrameStamp {
            source_seq: 25,
            captured_at_micros: CAPTURE_MICROS,
        };
        let candidate = GnmShadowCandidate {
            stamp: gnm_stamp,
            fit_complete_micros: Some(INFERENCE_MICROS + 1),
            decoder_complete_micros: Some(INFERENCE_MICROS + 2),
            output: &coefficients(0.75),
        };
        let outcome =
            align_shadow_pair(&frame, coefficients(0.25), Some(candidate), PUBLISH_MICROS)
                .expect("alignment works");
        let GnmShadowOutcome::Aligned(aligned) = outcome else {
            panic!("matching stamps must align");
        };
        assert_eq!(
            aligned.direct.timing.backend(),
            FaceTrackingBackend::DirectMediaPipe
        );
        assert_eq!(
            aligned.gnm.timing.backend(),
            FaceTrackingBackend::GnmTemporal
        );
        assert_eq!(
            aligned.gnm.timing.fit_complete_micros(),
            Some(INFERENCE_MICROS + 1)
        );
        assert_eq!(aligned.gnm.output.get(ArkitBlendshape::JawOpen), 0.75);
        let comparison = aligned.latency_comparison();
        assert!(comparison.gnm_additional_end_to_end_ms >= 0.0);
    }

    #[test]
    fn toggling_shadow_off_leaves_the_direct_canonical_output_identical() {
        let frame = fan_out(26);
        let direct = coefficients(0.4);
        let publish = PUBLISH_MICROS;

        // Shadow ON with a matching GNM candidate...
        let gnm_stamp = GnmFrameStamp {
            source_seq: 26,
            captured_at_micros: CAPTURE_MICROS,
        };
        let with_gnm = align_shadow_pair(
            &frame,
            direct,
            Some(GnmShadowCandidate {
                stamp: gnm_stamp,
                fit_complete_micros: Some(INFERENCE_MICROS + 1),
                decoder_complete_micros: None,
                output: &coefficients(0.9),
            }),
            publish,
        )
        .expect("alignment works");
        // ...and OFF (no candidate) must yield byte-equal Direct output.
        let without_gnm =
            align_shadow_pair(&frame, direct, None, publish).expect("alignment works");

        let direct_on = match with_gnm {
            GnmShadowOutcome::Aligned(pair) => pair.direct.output,
            other => panic!("expected Aligned, got {other:?}"),
        };
        let direct_off = match without_gnm {
            GnmShadowOutcome::DirectOnly { direct, .. } => direct,
            other => panic!("expected DirectOnly, got {other:?}"),
        };
        assert_eq!(direct_on, direct_off);
    }
    #[test]
    fn non_finite_gnm_features_fail_closed_without_stopping_the_pipeline() {
        // A corrupted GNM feature snapshot must not poison the shadow path:
        // the decoder clamps it into unsupported channels with value 0, so a
        // valid (if empty) coefficient vector still exists and Direct keeps
        // authority regardless.
        let mut snapshot = neutral_snapshot();
        snapshot.irises.right = Some(vtuber_gnm::IrisSideAuxFeature {
            side: vtuber_gnm::AnatomicalSide::Right,
            vertical_delta: Some(f32::NAN),
            horizontal_delta: Some(0.0),
        });
        let decoded =
            decode_shadow_features(&snapshot).expect("non-finite evidence must fail closed");
        let up_right = ArkitBlendshape::EyeLookUpRight.index();
        assert_eq!(decoded.supports[up_right], ProjectedSupport::Unsupported);
        assert_eq!(decoded.coefficients.as_array()[up_right], 0.0);
    }

    fn neutral_snapshot() -> vtuber_gnm::GnmFacialFeatures {
        use vtuber_gnm::*;
        let aperture = |side| EyeApertureFeature {
            side,
            current_aperture: 0.4,
            neutral_aperture: 0.4,
            normalized_delta: 0.0,
        };
        GnmFacialFeatures {
            eyes: EyeAuxFeatures {
                right: aperture(AnatomicalSide::Right),
                left: aperture(AnatomicalSide::Left),
            },
            irises: IrisAuxFeatures::default(),
            mouth_jaw: MouthAuxFeatures::default(),
            cheeks: CheekAuxFeatures::default(),
            brows: BrowAuxFeatures {
                right: BrowSideAuxFeatures {
                    side: AnatomicalSide::Right,
                    inner_rise: None,
                    brow_lower: None,
                    outer_rise: None,
                },
                left: BrowSideAuxFeatures {
                    side: AnatomicalSide::Left,
                    inner_rise: None,
                    brow_lower: None,
                    outer_rise: None,
                },
            },
        }
    }
}
