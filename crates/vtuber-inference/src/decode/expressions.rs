//! Decode expression coefficients from backend blendshape output or landmark
//! ratios.
//!
//! The decoder prefers manifest-defined blendshape mappings when the backend
//! provides named coefficients.  Otherwise it falls back to a minimal
//! landmark-ratio heuristic.  Unsupported backends return `None` rather than
//! panicking.

use vtuber_core::observation::RawExpressionObservation;
use vtuber_core::types::{Landmark3, LandmarkSchemaId, NamedCoefficient};

use crate::descriptor::ExpressionMapping;
use crate::error::Result;
use crate::schema::BasicExpressionFallback;

/// Decodes raw expression coefficients from the available inference outputs.
///
/// # Arguments
///
/// * `blendshapes` - Optional named coefficients from the backend.
/// * `mapping` - Optional manifest mapping from backend names to canonical
///   expressions.
/// * `landmarks` - Optional facial landmarks for the fallback path.
/// * `schema` - Landmark schema ID used by `landmarks`.
/// * `face_confidence` - Overall face confidence in `[0, 1]`.
///
/// # Returns
///
/// `Ok(Some(observation))` when coefficients can be decoded, `Ok(None)` when
/// no backend output or fallback is available, and `Err` only for invalid
/// numeric values in the provided blendshape output.
pub fn decode_expressions(
    blendshapes: Option<&[NamedCoefficient]>,
    mapping: Option<&ExpressionMapping>,
    landmarks: Option<&[Landmark3]>,
    schema: LandmarkSchemaId,
    face_confidence: f32,
) -> Result<Option<RawExpressionObservation>> {
    let base_confidence = face_confidence.clamp(0.0, 1.0);

    if let (Some(blendshapes), Some(mapping)) = (blendshapes, mapping) {
        let left = pick(blendshapes, &mapping.blink_left, base_confidence);
        let right = pick(blendshapes, &mapping.blink_right, base_confidence);
        let mouth = pick(blendshapes, &mapping.mouth_open, base_confidence);

        return Ok(Some(RawExpressionObservation {
            blink_left: left.value,
            blink_left_confidence: left.confidence,
            blink_right: right.value,
            blink_right_confidence: right.confidence,
            mouth_open: mouth.value,
            mouth_open_confidence: mouth.confidence,
        }));
    }

    if let Some(landmarks) = landmarks
        && let Some(obs) =
            BasicExpressionFallback::from_landmarks(landmarks, schema, base_confidence)
    {
        return Ok(Some(obs));
    }

    Ok(None)
}

struct Picked {
    value: f32,
    confidence: f32,
}

fn pick(blendshapes: &[NamedCoefficient], names: &[String], base_confidence: f32) -> Picked {
    for name in names {
        if let Some(c) = blendshapes.iter().find(|c| &c.name == name)
            && c.value.is_finite()
        {
            return Picked {
                value: c.value.clamp(0.0, 1.0),
                confidence: base_confidence,
            };
        }
    }

    Picked {
        value: 0.0,
        confidence: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::types::Landmark3;

    fn mapping() -> ExpressionMapping {
        ExpressionMapping {
            blink_left: vec!["eyeBlinkLeft".into(), "blinkLeft".into()],
            blink_right: vec!["eyeBlinkRight".into(), "blinkRight".into()],
            mouth_open: vec!["mouthOpen".into(), "aa".into()],
        }
    }

    fn blendshapes() -> Vec<NamedCoefficient> {
        vec![
            NamedCoefficient {
                name: "eyeBlinkLeft".into(),
                value: 0.75,
            },
            NamedCoefficient {
                name: "eyeBlinkRight".into(),
                value: 0.25,
            },
            NamedCoefficient {
                name: "mouthOpen".into(),
                value: 0.6,
            },
        ]
    }

    fn landmark(x: f32, y: f32, visibility: f32) -> Landmark3 {
        Landmark3 {
            x,
            y,
            z: 0.0,
            visibility,
        }
    }

    #[test]
    fn expression_decode_from_blendshape_mapping() {
        let obs = decode_expressions(
            Some(&blendshapes()),
            Some(&mapping()),
            None,
            LandmarkSchemaId("unused"),
            0.9,
        )
        .unwrap()
        .expect("expected an observation");

        assert!((obs.blink_left - 0.75).abs() < 1e-6);
        assert!((obs.blink_right - 0.25).abs() < 1e-6);
        assert!((obs.mouth_open - 0.6).abs() < 1e-6);

        assert!((obs.blink_left_confidence - 0.9).abs() < 1e-6);
        assert!(obs.is_valid());
    }

    #[test]
    fn expression_decode_clamps_and_ignores_non_finite() {
        let blends = vec![
            NamedCoefficient {
                name: "eyeBlinkLeft".into(),
                value: 1.2,
            },
            NamedCoefficient {
                name: "eyeBlinkRight".into(),
                value: f32::NAN,
            },
        ];

        let obs = decode_expressions(
            Some(&blends),
            Some(&mapping()),
            None,
            LandmarkSchemaId("unused"),
            1.0,
        )
        .unwrap()
        .expect("expected an observation");

        assert!((obs.blink_left - 1.0).abs() < 1e-6);
        assert_eq!(obs.blink_right, 0.0);
        assert_eq!(obs.blink_right_confidence, 0.0);
        assert!(obs.is_valid());
    }

    #[test]
    fn expression_decode_missing_name_returns_zero_confidence() {
        let blends = vec![NamedCoefficient {
            name: "unknown".into(),
            value: 0.5,
        }];

        let obs = decode_expressions(
            Some(&blends),
            Some(&mapping()),
            None,
            LandmarkSchemaId("unused"),
            1.0,
        )
        .unwrap()
        .expect("expected an observation");

        assert_eq!(obs.blink_left, 0.0);
        assert_eq!(obs.blink_left_confidence, 0.0);
        assert!(obs.is_valid());
    }

    #[test]
    fn expression_decode_fallback_from_landmarks() {
        // Build a synthetic landmark set large enough for the placeholder
        // PeppaPig-98 indices.  The eyes are open and the mouth is slightly
        // open.
        let mut landmarks: Vec<Landmark3> = (0..400).map(|_| landmark(0.5, 0.5, 1.0)).collect();

        // Left eye: outer (33), inner (133).  Top/bottom separation gives
        // vertical distance 0.1, horizontal distance 0.4 -> openness 0.25.
        landmarks[33] = landmark(0.3, 0.5, 1.0);
        landmarks[133] = landmark(0.7, 0.5, 1.0);
        landmarks[160] = landmark(0.5, 0.45, 1.0);
        landmarks[158] = landmark(0.5, 0.45, 1.0);
        landmarks[153] = landmark(0.5, 0.55, 1.0);
        landmarks[144] = landmark(0.5, 0.55, 1.0);

        // Right eye: openness 0.5.
        landmarks[263] = landmark(0.2, 0.5, 1.0);
        landmarks[362] = landmark(0.8, 0.5, 1.0);
        landmarks[388] = landmark(0.5, 0.4, 1.0);
        landmarks[385] = landmark(0.5, 0.4, 1.0);
        landmarks[382] = landmark(0.5, 0.6, 1.0);
        landmarks[373] = landmark(0.5, 0.6, 1.0);

        // Mouth: horizontal 0.5, vertical 0.2 -> openness 0.4.
        landmarks[0] = landmark(0.3, 0.8, 1.0);
        landmarks[291] = landmark(0.8, 0.8, 1.0);
        landmarks[37] = landmark(0.55, 0.7, 1.0);
        landmarks[17] = landmark(0.55, 0.9, 1.0);

        let obs = decode_expressions(
            None,
            None,
            Some(&landmarks),
            LandmarkSchemaId("peppapig-98"),
            0.8,
        )
        .unwrap()
        .expect("expected fallback observation");

        // blink = 1 - openness.
        assert!(
            (obs.blink_left - 0.75).abs() < 1e-5,
            "blink_left = {}",
            obs.blink_left
        );
        assert!(
            (obs.blink_right - 0.666_666_7).abs() < 1e-5,
            "blink_right = {}",
            obs.blink_right
        );
        assert!(
            (obs.mouth_open - 0.4).abs() < 1e-5,
            "mouth_open = {}",
            obs.mouth_open
        );
        assert!((obs.mouth_open_confidence - 0.8).abs() < 1e-6);
        assert!(obs.is_valid());
    }

    #[test]
    fn expression_decode_unsupported_schema_returns_none() {
        let landmarks: Vec<Landmark3> = (0..10).map(|_| landmark(0.5, 0.5, 1.0)).collect();

        let result = decode_expressions(
            None,
            None,
            Some(&landmarks),
            LandmarkSchemaId("unknown-schema"),
            1.0,
        )
        .unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn expression_decode_no_backend_returns_none() {
        let result =
            decode_expressions(None, None, None, LandmarkSchemaId("peppapig-98"), 1.0).unwrap();

        assert!(result.is_none());
    }
}
