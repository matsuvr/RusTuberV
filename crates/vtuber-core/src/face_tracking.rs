//! Canonical engine-independent face-tracking contracts.
//!
//! These types describe the approved MediaPipe Face Landmarker result without
//! exposing MediaPipe, Bevy, camera-backend, or inference-runtime types. The
//! legacy [`crate::RawFaceObservation`] remains available during migration but
//! cannot represent a canonical face sample without manufacturing a transform
//! or detector confidence.

use std::fmt;
use std::sync::Arc;

use crate::{FrameSeq, MonoTimeNs};

/// Number of landmarks produced by the approved Face Landmarker task.
pub const MEDIAPIPE_FACE_LANDMARK_COUNT: usize = 478;

/// Number of blendshape categories produced by the approved task.
pub const MEDIAPIPE_FACE_BLENDSHAPE_COUNT: usize = 52;

/// The stable semantic names emitted by the approved MediaPipe task.
///
/// The order matches MediaPipe's published `Blendshapes` enum. `Neutral` is
/// the task's `_neutral` category; the remaining 51 categories are the
/// expression coefficients used by the application.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum MediaPipeBlendshape {
    /// Neutral baseline category, emitted as `_neutral`.
    Neutral = 0,
    /// Left brow down.
    BrowDownLeft = 1,
    /// Right brow down.
    BrowDownRight = 2,
    /// Inner brows up.
    BrowInnerUp = 3,
    /// Left outer brow up.
    BrowOuterUpLeft = 4,
    /// Right outer brow up.
    BrowOuterUpRight = 5,
    /// Cheeks puffed.
    CheekPuff = 6,
    /// Left cheek squint.
    CheekSquintLeft = 7,
    /// Right cheek squint.
    CheekSquintRight = 8,
    /// Left eye blink.
    EyeBlinkLeft = 9,
    /// Right eye blink.
    EyeBlinkRight = 10,
    /// Left eye looks down.
    EyeLookDownLeft = 11,
    /// Right eye looks down.
    EyeLookDownRight = 12,
    /// Left eye looks inward.
    EyeLookInLeft = 13,
    /// Right eye looks inward.
    EyeLookInRight = 14,
    /// Left eye looks outward.
    EyeLookOutLeft = 15,
    /// Right eye looks outward.
    EyeLookOutRight = 16,
    /// Left eye looks up.
    EyeLookUpLeft = 17,
    /// Right eye looks up.
    EyeLookUpRight = 18,
    /// Left eye squint.
    EyeSquintLeft = 19,
    /// Right eye squint.
    EyeSquintRight = 20,
    /// Left eye wide.
    EyeWideLeft = 21,
    /// Right eye wide.
    EyeWideRight = 22,
    /// Jaw forward.
    JawForward = 23,
    /// Jaw left.
    JawLeft = 24,
    /// Jaw open.
    JawOpen = 25,
    /// Jaw right.
    JawRight = 26,
    /// Mouth closed.
    MouthClose = 27,
    /// Left mouth dimple.
    MouthDimpleLeft = 28,
    /// Right mouth dimple.
    MouthDimpleRight = 29,
    /// Left mouth frown.
    MouthFrownLeft = 30,
    /// Right mouth frown.
    MouthFrownRight = 31,
    /// Mouth funnel.
    MouthFunnel = 32,
    /// Mouth left.
    MouthLeft = 33,
    /// Left lower lip down.
    MouthLowerDownLeft = 34,
    /// Right lower lip down.
    MouthLowerDownRight = 35,
    /// Left mouth press.
    MouthPressLeft = 36,
    /// Right mouth press.
    MouthPressRight = 37,
    /// Mouth pucker.
    MouthPucker = 38,
    /// Mouth right.
    MouthRight = 39,
    /// Lower lip roll.
    MouthRollLower = 40,
    /// Upper lip roll.
    MouthRollUpper = 41,
    /// Lower mouth shrug.
    MouthShrugLower = 42,
    /// Upper mouth shrug.
    MouthShrugUpper = 43,
    /// Left mouth smile.
    MouthSmileLeft = 44,
    /// Right mouth smile.
    MouthSmileRight = 45,
    /// Left mouth stretch.
    MouthStretchLeft = 46,
    /// Right mouth stretch.
    MouthStretchRight = 47,
    /// Left upper lip up.
    MouthUpperUpLeft = 48,
    /// Right upper lip up.
    MouthUpperUpRight = 49,
    /// Left nose sneer.
    NoseSneerLeft = 50,
    /// Right nose sneer.
    NoseSneerRight = 51,
}

impl MediaPipeBlendshape {
    /// All 52 categories in the published MediaPipe order.
    pub const ALL: [Self; MEDIAPIPE_FACE_BLENDSHAPE_COUNT] = [
        Self::Neutral,
        Self::BrowDownLeft,
        Self::BrowDownRight,
        Self::BrowInnerUp,
        Self::BrowOuterUpLeft,
        Self::BrowOuterUpRight,
        Self::CheekPuff,
        Self::CheekSquintLeft,
        Self::CheekSquintRight,
        Self::EyeBlinkLeft,
        Self::EyeBlinkRight,
        Self::EyeLookDownLeft,
        Self::EyeLookDownRight,
        Self::EyeLookInLeft,
        Self::EyeLookInRight,
        Self::EyeLookOutLeft,
        Self::EyeLookOutRight,
        Self::EyeLookUpLeft,
        Self::EyeLookUpRight,
        Self::EyeSquintLeft,
        Self::EyeSquintRight,
        Self::EyeWideLeft,
        Self::EyeWideRight,
        Self::JawForward,
        Self::JawLeft,
        Self::JawOpen,
        Self::JawRight,
        Self::MouthClose,
        Self::MouthDimpleLeft,
        Self::MouthDimpleRight,
        Self::MouthFrownLeft,
        Self::MouthFrownRight,
        Self::MouthFunnel,
        Self::MouthLeft,
        Self::MouthLowerDownLeft,
        Self::MouthLowerDownRight,
        Self::MouthPressLeft,
        Self::MouthPressRight,
        Self::MouthPucker,
        Self::MouthRight,
        Self::MouthRollLower,
        Self::MouthRollUpper,
        Self::MouthShrugLower,
        Self::MouthShrugUpper,
        Self::MouthSmileLeft,
        Self::MouthSmileRight,
        Self::MouthStretchLeft,
        Self::MouthStretchRight,
        Self::MouthUpperUpLeft,
        Self::MouthUpperUpRight,
        Self::NoseSneerLeft,
        Self::NoseSneerRight,
    ];

    /// Returns the stable MediaPipe category name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Neutral => "_neutral",
            Self::BrowDownLeft => "browDownLeft",
            Self::BrowDownRight => "browDownRight",
            Self::BrowInnerUp => "browInnerUp",
            Self::BrowOuterUpLeft => "browOuterUpLeft",
            Self::BrowOuterUpRight => "browOuterUpRight",
            Self::CheekPuff => "cheekPuff",
            Self::CheekSquintLeft => "cheekSquintLeft",
            Self::CheekSquintRight => "cheekSquintRight",
            Self::EyeBlinkLeft => "eyeBlinkLeft",
            Self::EyeBlinkRight => "eyeBlinkRight",
            Self::EyeLookDownLeft => "eyeLookDownLeft",
            Self::EyeLookDownRight => "eyeLookDownRight",
            Self::EyeLookInLeft => "eyeLookInLeft",
            Self::EyeLookInRight => "eyeLookInRight",
            Self::EyeLookOutLeft => "eyeLookOutLeft",
            Self::EyeLookOutRight => "eyeLookOutRight",
            Self::EyeLookUpLeft => "eyeLookUpLeft",
            Self::EyeLookUpRight => "eyeLookUpRight",
            Self::EyeSquintLeft => "eyeSquintLeft",
            Self::EyeSquintRight => "eyeSquintRight",
            Self::EyeWideLeft => "eyeWideLeft",
            Self::EyeWideRight => "eyeWideRight",
            Self::JawForward => "jawForward",
            Self::JawLeft => "jawLeft",
            Self::JawOpen => "jawOpen",
            Self::JawRight => "jawRight",
            Self::MouthClose => "mouthClose",
            Self::MouthDimpleLeft => "mouthDimpleLeft",
            Self::MouthDimpleRight => "mouthDimpleRight",
            Self::MouthFrownLeft => "mouthFrownLeft",
            Self::MouthFrownRight => "mouthFrownRight",
            Self::MouthFunnel => "mouthFunnel",
            Self::MouthLeft => "mouthLeft",
            Self::MouthLowerDownLeft => "mouthLowerDownLeft",
            Self::MouthLowerDownRight => "mouthLowerDownRight",
            Self::MouthPressLeft => "mouthPressLeft",
            Self::MouthPressRight => "mouthPressRight",
            Self::MouthPucker => "mouthPucker",
            Self::MouthRight => "mouthRight",
            Self::MouthRollLower => "mouthRollLower",
            Self::MouthRollUpper => "mouthRollUpper",
            Self::MouthShrugLower => "mouthShrugLower",
            Self::MouthShrugUpper => "mouthShrugUpper",
            Self::MouthSmileLeft => "mouthSmileLeft",
            Self::MouthSmileRight => "mouthSmileRight",
            Self::MouthStretchLeft => "mouthStretchLeft",
            Self::MouthStretchRight => "mouthStretchRight",
            Self::MouthUpperUpLeft => "mouthUpperUpLeft",
            Self::MouthUpperUpRight => "mouthUpperUpRight",
            Self::NoseSneerLeft => "noseSneerLeft",
            Self::NoseSneerRight => "noseSneerRight",
        }
    }

    /// Parses one official category name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|category| category.as_str() == name)
    }

    /// Returns the fixed array index for this category.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Fixed-size typed set of the approved 52 blendshape coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceBlendshapeSet {
    values: [f32; MEDIAPIPE_FACE_BLENDSHAPE_COUNT],
}

impl Default for FaceBlendshapeSet {
    fn default() -> Self {
        Self {
            values: [0.0; MEDIAPIPE_FACE_BLENDSHAPE_COUNT],
        }
    }
}

impl FaceBlendshapeSet {
    /// Builds a set from exactly one value for each official category.
    ///
    /// Unknown names, duplicate names, missing categories, non-finite values,
    /// and values outside `[0, 1]` are rejected before the set is published.
    // Invariant: `MediaPipeBlendshape::index()` is always
    // `< MEDIAPIPE_FACE_BLENDSHAPE_COUNT`, so every index below is in bounds
    // by construction.
    #[allow(clippy::indexing_slicing)]
    pub fn from_pairs(pairs: &[(&str, f32)]) -> Result<Self, FaceTrackingContractError> {
        if pairs.len() != MEDIAPIPE_FACE_BLENDSHAPE_COUNT {
            return Err(FaceTrackingContractError::BlendshapeCount {
                expected: MEDIAPIPE_FACE_BLENDSHAPE_COUNT,
                actual: pairs.len(),
            });
        }
        let mut values = [0.0; MEDIAPIPE_FACE_BLENDSHAPE_COUNT];
        let mut seen = [false; MEDIAPIPE_FACE_BLENDSHAPE_COUNT];
        for (name, value) in pairs {
            let category = MediaPipeBlendshape::from_name(name)
                .ok_or_else(|| FaceTrackingContractError::UnknownBlendshape((*name).into()))?;
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(FaceTrackingContractError::InvalidBlendshapeValue {
                    name: (*name).into(),
                    value: *value,
                });
            }
            let index = category.index();
            if seen[index] {
                return Err(FaceTrackingContractError::DuplicateBlendshape(
                    (*name).into(),
                ));
            }
            seen[index] = true;
            values[index] = *value;
        }
        if let Some(missing) = MediaPipeBlendshape::ALL
            .into_iter()
            .find(|category| !seen[category.index()])
        {
            return Err(FaceTrackingContractError::MissingBlendshape(
                missing.as_str().into(),
            ));
        }
        Ok(Self { values })
    }

    /// Returns the coefficient for one category.
    // Invariant: `category.index() < MEDIAPIPE_FACE_BLENDSHAPE_COUNT`.
    #[allow(clippy::indexing_slicing)]
    #[must_use]
    pub const fn get(&self, category: MediaPipeBlendshape) -> f32 {
        self.values[category.index()]
    }

    /// Returns all coefficients in the official category order.
    #[must_use]
    pub const fn as_array(&self) -> &[f32; MEDIAPIPE_FACE_BLENDSHAPE_COUNT] {
        &self.values
    }

    /// Returns true when every stored coefficient is finite and in range.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.values
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    }
}

/// Camera-space rigid transform of the canonical MediaPipe face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraFaceTransform {
    /// Unit quaternion in `(x, y, z, w)` order.
    pub rotation_xyzw: [f32; 4],
    /// MediaPipe camera-space translation in model-defined units.
    pub translation_xyz: [f32; 3],
}

impl CameraFaceTransform {
    /// Identity camera-to-face transform.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            translation_xyz: [0.0, 0.0, 0.0],
        }
    }

    /// Returns whether the transform is finite and has a unit quaternion.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let rotation_norm = self
            .rotation_xyzw
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        self.rotation_xyzw.iter().all(|value| value.is_finite())
            && self.translation_xyz.iter().all(|value| value.is_finite())
            && rotation_norm.is_finite()
            && (rotation_norm - 1.0).abs() <= 1.0e-3
    }
}

/// One of the 478 normalized face landmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FaceLandmark {
    /// Normalized image-space x coordinate.
    pub x: f32,
    /// Normalized image-space y coordinate.
    pub y: f32,
    /// MediaPipe model-defined relative depth.
    pub z: f32,
    /// Optional visibility score in `[0, 1]`.
    pub visibility: Option<f32>,
    /// Optional presence score in `[0, 1]`.
    pub presence: Option<f32>,
}

/// Quality values retained from face-result validation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FaceTrackingQuality {
    /// Median of available landmark presence scores.
    pub landmark_presence_median: Option<f32>,
    /// Frobenius error of the extracted matrix rotation block against SO(3).
    pub matrix_orthogonality_error: f32,
    /// Determinant of the source matrix rotation block.
    pub matrix_determinant: f32,
}

/// A validated one-face MediaPipe sample.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceTrackingSample {
    /// Source camera frame sequence.
    pub source_seq: FrameSeq,
    /// Original monotonic capture timestamp.
    pub captured_at: MonoTimeNs,
    /// Inference start timestamp.
    pub inference_started_at: MonoTimeNs,
    /// Inference completion timestamp.
    pub inference_finished_at: MonoTimeNs,
    /// Camera-space canonical face transform.
    pub camera_to_face: CameraFaceTransform,
    /// Normalized image-space face centre.
    pub face_center: [f32; 2],
    /// Exactly 478 face landmarks for a valid sample.
    pub landmarks: Arc<[FaceLandmark]>,
    /// Exactly the approved 52 typed blendshape categories.
    pub blendshapes: FaceBlendshapeSet,
    /// Matrix and landmark quality diagnostics.
    pub quality: FaceTrackingQuality,
}

impl FaceTrackingSample {
    /// Validates the canonical output contract.
    pub fn validate(&self) -> Result<(), FaceTrackingContractError> {
        if self.inference_started_at < self.captured_at
            || self.inference_finished_at < self.inference_started_at
        {
            return Err(FaceTrackingContractError::TimestampOrder);
        }
        if !self.camera_to_face.is_valid() {
            return Err(FaceTrackingContractError::InvalidTransform);
        }
        if !self.face_center.iter().all(|value| value.is_finite()) {
            return Err(FaceTrackingContractError::InvalidFaceCenter);
        }
        if self.landmarks.len() != MEDIAPIPE_FACE_LANDMARK_COUNT {
            return Err(FaceTrackingContractError::LandmarkCount {
                expected: MEDIAPIPE_FACE_LANDMARK_COUNT,
                actual: self.landmarks.len(),
            });
        }
        for (index, landmark) in self.landmarks.iter().enumerate() {
            if !landmark.x.is_finite() || !landmark.y.is_finite() || !landmark.z.is_finite() {
                return Err(FaceTrackingContractError::NonFiniteLandmark { index });
            }
            for confidence in [landmark.visibility, landmark.presence]
                .into_iter()
                .flatten()
            {
                if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                    return Err(FaceTrackingContractError::InvalidLandmarkConfidence { index });
                }
            }
        }
        if !self.blendshapes.is_valid() {
            return Err(FaceTrackingContractError::InvalidBlendshapeSet);
        }
        if self
            .quality
            .landmark_presence_median
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
            || !self.quality.matrix_orthogonality_error.is_finite()
            || self.quality.matrix_orthogonality_error < 0.0
            || !self.quality.matrix_determinant.is_finite()
            || self.quality.matrix_determinant <= 0.0
        {
            return Err(FaceTrackingContractError::InvalidQuality);
        }
        Ok(())
    }

    /// Constructs a sample only when the canonical contract is valid.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        source_seq: FrameSeq,
        captured_at: MonoTimeNs,
        inference_started_at: MonoTimeNs,
        inference_finished_at: MonoTimeNs,
        camera_to_face: CameraFaceTransform,
        face_center: [f32; 2],
        landmarks: Arc<[FaceLandmark]>,
        blendshapes: FaceBlendshapeSet,
        quality: FaceTrackingQuality,
    ) -> Result<Self, FaceTrackingContractError> {
        let sample = Self {
            source_seq,
            captured_at,
            inference_started_at,
            inference_finished_at,
            camera_to_face,
            face_center,
            landmarks,
            blendshapes,
            quality,
        };
        sample.validate()?;
        Ok(sample)
    }
}

/// Result of one inference attempt at the canonical tracking boundary.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum FaceTrackingOutcome {
    /// Exactly one valid face sample.
    Face(FaceTrackingSample),
    /// No face was found; this is a normal tracking state, not an error.
    NoFace {
        /// Source camera frame sequence.
        source_seq: FrameSeq,
        /// Original monotonic capture timestamp.
        captured_at: MonoTimeNs,
        /// Inference start timestamp.
        inference_started_at: MonoTimeNs,
        /// Inference completion timestamp.
        inference_finished_at: MonoTimeNs,
    },
}

impl FaceTrackingOutcome {
    /// Returns the source sequence regardless of whether a face was found.
    #[must_use]
    pub const fn source_seq(&self) -> FrameSeq {
        match self {
            Self::Face(sample) => sample.source_seq,
            Self::NoFace { source_seq, .. } => *source_seq,
        }
    }
}

/// Typed failure while constructing or validating a canonical face result.
#[derive(Clone, Debug, PartialEq)]
pub enum FaceTrackingContractError {
    /// Timestamps did not preserve capture -> start -> finish ordering.
    TimestampOrder,
    /// Landmark count was not exactly 478.
    LandmarkCount {
        /// Required landmark count.
        expected: usize,
        /// Actual landmark count.
        actual: usize,
    },
    /// Blendshape count was not exactly 52.
    BlendshapeCount {
        /// Required blendshape count.
        expected: usize,
        /// Actual blendshape count.
        actual: usize,
    },
    /// A category name was not in the approved set.
    UnknownBlendshape(String),
    /// A category appeared more than once.
    DuplicateBlendshape(String),
    /// A category was absent from an otherwise complete set.
    MissingBlendshape(String),
    /// A blendshape score was non-finite or outside `[0, 1]`.
    InvalidBlendshapeValue {
        /// Category name associated with the invalid value.
        name: String,
        /// Invalid score.
        value: f32,
    },
    /// The camera-to-face transform was not finite or unit-quaternion based.
    InvalidTransform,
    /// The normalized face centre contained a non-finite value.
    InvalidFaceCenter,
    /// A landmark coordinate was non-finite.
    NonFiniteLandmark {
        /// Landmark index containing a non-finite coordinate.
        index: usize,
    },
    /// A landmark visibility/presence score was invalid.
    InvalidLandmarkConfidence {
        /// Landmark index containing the invalid confidence.
        index: usize,
    },
    /// The fixed blendshape set contained an invalid score.
    InvalidBlendshapeSet,
    /// Matrix and presence quality values were invalid.
    InvalidQuality,
}

impl fmt::Display for FaceTrackingContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimestampOrder => {
                formatter.write_str("face tracking timestamps are out of order")
            }
            Self::LandmarkCount { expected, actual } => {
                write!(formatter, "expected {expected} landmarks, got {actual}")
            }
            Self::BlendshapeCount { expected, actual } => {
                write!(formatter, "expected {expected} blendshapes, got {actual}")
            }
            Self::UnknownBlendshape(name) => write!(formatter, "unknown blendshape `{name}`"),
            Self::DuplicateBlendshape(name) => write!(formatter, "duplicate blendshape `{name}`"),
            Self::MissingBlendshape(name) => write!(formatter, "missing blendshape `{name}`"),
            Self::InvalidBlendshapeValue { name, value } => {
                write!(formatter, "invalid blendshape `{name}` value {value}")
            }
            Self::InvalidTransform => formatter.write_str("invalid camera-to-face transform"),
            Self::InvalidFaceCenter => formatter.write_str("invalid normalized face centre"),
            Self::NonFiniteLandmark { index } => {
                write!(
                    formatter,
                    "landmark {index} contains a non-finite coordinate"
                )
            }
            Self::InvalidLandmarkConfidence { index } => {
                write!(formatter, "landmark {index} contains an invalid confidence")
            }
            Self::InvalidBlendshapeSet => formatter.write_str("invalid blendshape set"),
            Self::InvalidQuality => formatter.write_str("invalid face tracking quality"),
        }
    }
}

impl std::error::Error for FaceTrackingContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_pairs() -> Vec<(&'static str, f32)> {
        MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| (category.as_str(), 0.0))
            .collect()
    }

    fn valid_sample() -> FaceTrackingSample {
        FaceTrackingSample {
            source_seq: FrameSeq(4),
            captured_at: MonoTimeNs(10),
            inference_started_at: MonoTimeNs(11),
            inference_finished_at: MonoTimeNs(12),
            camera_to_face: CameraFaceTransform::identity(),
            face_center: [0.5, 0.5],
            landmarks: vec![FaceLandmark::default(); MEDIAPIPE_FACE_LANDMARK_COUNT].into(),
            blendshapes: FaceBlendshapeSet::from_pairs(&all_pairs()).expect("all names are valid"),
            quality: FaceTrackingQuality {
                landmark_presence_median: Some(0.9),
                matrix_orthogonality_error: 0.0,
                matrix_determinant: 1.0,
            },
        }
    }

    #[test]
    fn official_category_set_is_exactly_52_and_round_trips_names() {
        assert_eq!(MediaPipeBlendshape::ALL.len(), 52);
        for category in MediaPipeBlendshape::ALL {
            assert_eq!(
                MediaPipeBlendshape::from_name(category.as_str()),
                Some(category)
            );
        }
    }

    #[test]
    fn typed_set_rejects_unknown_and_duplicate_names() {
        let mut pairs = all_pairs();
        pairs[0] = ("notARealBlendshape", 0.0);
        assert!(matches!(
            FaceBlendshapeSet::from_pairs(&pairs),
            Err(FaceTrackingContractError::UnknownBlendshape(_))
        ));

        let mut pairs = all_pairs();
        pairs[1] = (pairs[0].0, 0.0);
        assert!(matches!(
            FaceBlendshapeSet::from_pairs(&pairs),
            Err(FaceTrackingContractError::DuplicateBlendshape(_))
        ));
    }

    #[test]
    fn canonical_sample_requires_478_landmarks_and_valid_quality() {
        let sample = valid_sample();
        assert!(sample.validate().is_ok());

        let mut invalid = sample.clone();
        invalid.landmarks = Arc::from(vec![FaceLandmark::default()]);
        assert!(matches!(
            invalid.validate(),
            Err(FaceTrackingContractError::LandmarkCount { .. })
        ));

        let mut invalid = sample;
        invalid.quality.matrix_determinant = -1.0;
        assert_eq!(
            invalid.validate(),
            Err(FaceTrackingContractError::InvalidQuality)
        );
    }

    #[test]
    fn no_face_preserves_all_source_timing_metadata() {
        let outcome = FaceTrackingOutcome::NoFace {
            source_seq: FrameSeq(8),
            captured_at: MonoTimeNs(10),
            inference_started_at: MonoTimeNs(11),
            inference_finished_at: MonoTimeNs(12),
        };
        assert_eq!(outcome.source_seq(), FrameSeq(8));
        assert!(matches!(outcome, FaceTrackingOutcome::NoFace { .. }));
    }
}
