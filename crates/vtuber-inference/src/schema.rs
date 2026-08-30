//! Landmark schema definitions and basic expression fallback for the
//! PeppaPig-98 model.

use vtuber_core::observation::RawExpressionObservation;
use vtuber_core::types::{Landmark3, LandmarkSchemaId};

/// Schema ID for the Peppa_Pig_Face_Landmark student 256x256 model.
pub const SCHEMA_PEPPAPIG_98: LandmarkSchemaId = LandmarkSchemaId("peppapig-98");

/// Landmark indices used by the basic expression fallback.
///
/// These are schema-specific placeholders.  They are grouped in a single
/// data structure so the decoder does not hard-code individual indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeIndices {
    /// Outer eye corner.
    pub outer: usize,
    /// Inner eye corner.
    pub inner: usize,
    /// Upper eyelid landmarks.
    pub top: [usize; 2],
    /// Lower eyelid landmarks.
    pub bottom: [usize; 2],
}

/// Landmark indices used by the mouth openness fallback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MouthIndices {
    /// Left mouth corner.
    pub left: usize,
    /// Right mouth corner.
    pub right: usize,
    /// Upper inner lip.
    pub top: usize,
    /// Lower inner lip.
    pub bottom: usize,
}

/// Full landmark index set for the basic expression fallback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandmarkExpressionSchema {
    /// Left eye indices.
    pub left_eye: EyeIndices,
    /// Right eye indices.
    pub right_eye: EyeIndices,
    /// Mouth indices.
    pub mouth: MouthIndices,
}

/// Placeholder index mapping for the 98-point PeppaPig-98 landmark set.
///
/// The concrete indices must be validated against the model output before
/// this fallback is used in production.
pub const PEPPAPIG_98_EXPRESSIONS: LandmarkExpressionSchema = LandmarkExpressionSchema {
    left_eye: EyeIndices {
        outer: 33,
        inner: 133,
        top: [160, 158],
        bottom: [153, 144],
    },
    right_eye: EyeIndices {
        outer: 263,
        inner: 362,
        top: [388, 385],
        bottom: [382, 373],
    },
    mouth: MouthIndices {
        left: 0,
        right: 291,
        top: 37,
        bottom: 17,
    },
};

/// Minimal landmark-based fallback for basic expression coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BasicExpressionFallback;

impl BasicExpressionFallback {
    /// Computes raw blink and mouth-open coefficients from landmark ratios.
    ///
    /// Returns `None` if the schema is not supported or any required landmark
    /// is missing.
    pub fn from_landmarks(
        landmarks: &[Landmark3],
        schema: LandmarkSchemaId,
        face_confidence: f32,
    ) -> Option<RawExpressionObservation> {
        if schema != SCHEMA_PEPPAPIG_98 {
            return None;
        }

        let left_openness = eye_openness(landmarks, &PEPPAPIG_98_EXPRESSIONS.left_eye)?;
        let right_openness = eye_openness(landmarks, &PEPPAPIG_98_EXPRESSIONS.right_eye)?;
        let mouth_openness = mouth_openness(landmarks, &PEPPAPIG_98_EXPRESSIONS.mouth)?;

        let confidence = face_confidence.clamp(0.0, 1.0)
            * average_visibility(
                landmarks,
                &[
                    PEPPAPIG_98_EXPRESSIONS.left_eye.outer,
                    PEPPAPIG_98_EXPRESSIONS.left_eye.inner,
                    PEPPAPIG_98_EXPRESSIONS.left_eye.top[0],
                    PEPPAPIG_98_EXPRESSIONS.left_eye.top[1],
                    PEPPAPIG_98_EXPRESSIONS.left_eye.bottom[0],
                    PEPPAPIG_98_EXPRESSIONS.left_eye.bottom[1],
                    PEPPAPIG_98_EXPRESSIONS.right_eye.outer,
                    PEPPAPIG_98_EXPRESSIONS.right_eye.inner,
                    PEPPAPIG_98_EXPRESSIONS.right_eye.top[0],
                    PEPPAPIG_98_EXPRESSIONS.right_eye.top[1],
                    PEPPAPIG_98_EXPRESSIONS.right_eye.bottom[0],
                    PEPPAPIG_98_EXPRESSIONS.right_eye.bottom[1],
                    PEPPAPIG_98_EXPRESSIONS.mouth.left,
                    PEPPAPIG_98_EXPRESSIONS.mouth.right,
                    PEPPAPIG_98_EXPRESSIONS.mouth.top,
                    PEPPAPIG_98_EXPRESSIONS.mouth.bottom,
                ],
            );

        Some(RawExpressionObservation {
            blink_left: (1.0 - left_openness).clamp(0.0, 1.0),
            blink_right: (1.0 - right_openness).clamp(0.0, 1.0),
            mouth_open: mouth_openness,
            blink_left_confidence: confidence,
            blink_right_confidence: confidence,
            mouth_open_confidence: confidence,
        })
    }
}

fn eye_openness(landmarks: &[Landmark3], indices: &EyeIndices) -> Option<f32> {
    let outer = landmarks.get(indices.outer)?;
    let inner = landmarks.get(indices.inner)?;
    let top0 = landmarks.get(indices.top[0])?;
    let top1 = landmarks.get(indices.top[1])?;
    let bot0 = landmarks.get(indices.bottom[0])?;
    let bot1 = landmarks.get(indices.bottom[1])?;

    let horizontal = (outer.x - inner.x).abs().max(1e-6);
    let vertical = ((top0.y - bot0.y).abs() + (top1.y - bot1.y).abs()) * 0.5;

    Some((vertical / horizontal).clamp(0.0, 1.0))
}

fn mouth_openness(landmarks: &[Landmark3], indices: &MouthIndices) -> Option<f32> {
    let left = landmarks.get(indices.left)?;
    let right = landmarks.get(indices.right)?;
    let top = landmarks.get(indices.top)?;
    let bottom = landmarks.get(indices.bottom)?;

    let horizontal = (left.x - right.x).abs().max(1e-6);
    let vertical = (top.y - bottom.y).abs();

    Some((vertical / horizontal).clamp(0.0, 1.0))
}

fn average_visibility(landmarks: &[Landmark3], indices: &[usize]) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;

    for &i in indices {
        if let Some(lm) = landmarks.get(i)
            && lm.visibility.is_finite()
        {
            sum += lm.visibility.clamp(0.0, 1.0);
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}
