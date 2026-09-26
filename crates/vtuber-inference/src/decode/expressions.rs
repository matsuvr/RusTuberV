//! Decode expression coefficients from backend blendshape output.
//!
//! This decoder needs named blendshape coefficients plus a manifest mapping from
//! those names to the canonical expression channels. A backend that produces no
//! named coefficients has no expression mapping in this decoder at all, so it
//! reports "unsupported" instead of guessing indices into its landmark output.

use vtuber_core::observation::RawExpressionObservation;
use vtuber_core::types::NamedCoefficient;

use crate::descriptor::ExpressionMapping;

/// Decodes raw expression coefficients from named blendshape output.
///
/// # Arguments
///
/// * `blendshapes` - Named coefficients from the backend, if it produced any.
/// * `mapping` - Manifest mapping from backend names to canonical expressions.
/// * `face_confidence` - Overall face confidence in `[0, 1]`; it becomes the
///   confidence of every coefficient that was found.
///
/// # Returns
///
/// `Some(observation)` when both `blendshapes` and `mapping` are present. Each
/// channel is the first mapped name whose coefficient is finite, clamped to
/// `[0, 1]`; a name that is absent, or whose coefficient is not finite, yields
/// that channel's value and confidence as `0.0` rather than failing.
///
/// `None` means this decoder cannot estimate expressions for that input: the
/// backend produced no named output, or no mapping was supplied. There is no
/// landmark-ratio path here, so a 98-point landmark model has no expression
/// support in this decoder at all.
///
/// The function never fails, so it returns no error type. A `None` result is
/// not evidence that an expression was observed; callers that must distinguish
/// "not observed" represent it as a zero-confidence
/// [`RawExpressionObservation`].
#[must_use]
pub fn decode_expressions(
    blendshapes: Option<&[NamedCoefficient]>,
    mapping: Option<&ExpressionMapping>,
    face_confidence: f32,
) -> Option<RawExpressionObservation> {
    let (blendshapes, mapping) = match (blendshapes, mapping) {
        (Some(blendshapes), Some(mapping)) => (blendshapes, mapping),
        _ => return None,
    };
    let base_confidence = face_confidence.clamp(0.0, 1.0);

    let left = pick(blendshapes, &mapping.blink_left, base_confidence);
    let right = pick(blendshapes, &mapping.blink_right, base_confidence);
    let mouth = pick(blendshapes, &mapping.mouth_open, base_confidence);

    Some(RawExpressionObservation {
        blink_left: left.value,
        blink_left_confidence: left.confidence,
        blink_right: right.value,
        blink_right_confidence: right.confidence,
        mouth_open: mouth.value,
        mouth_open_confidence: mouth.confidence,
    })
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
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;

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

    #[test]
    fn expression_decode_from_blendshape_mapping() {
        let obs = decode_expressions(Some(&blendshapes()), Some(&mapping()), 0.9)
            .expect("named output with a mapping is supported");

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

        let obs = decode_expressions(Some(&blends), Some(&mapping()), 1.0)
            .expect("a non-finite coefficient is ignored, not an error");

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

        let obs = decode_expressions(Some(&blends), Some(&mapping()), 1.0)
            .expect("the mapped channels are all reported");

        assert_eq!(obs.blink_left, 0.0);
        assert_eq!(obs.blink_left_confidence, 0.0);
        assert!(obs.is_valid());
    }

    #[test]
    fn without_named_output_expressions_are_unsupported() {
        // No named coefficients at all: this decoder has no landmark path, so
        // it must say "unsupported" instead of estimating from indices.
        assert!(decode_expressions(None, Some(&mapping()), 1.0).is_none());
        // Named output without a mapping cannot be routed to channels.
        assert!(decode_expressions(Some(&blendshapes()), None, 1.0).is_none());
        assert!(decode_expressions(None, None, 1.0).is_none());

        // An empty named output is still "present", so it reports the channels
        // as not observed rather than as unsupported.
        let obs = decode_expressions(Some(&[]), Some(&mapping()), 1.0)
            .expect("an empty named output is present, not unsupported");
        assert_eq!(obs.blink_left, 0.0);
        assert_eq!(obs.blink_left_confidence, 0.0);
        assert_eq!(obs.mouth_open_confidence, 0.0);
    }
}
