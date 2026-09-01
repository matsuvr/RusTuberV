//! MediaPipe blendshape to engine-neutral VRM control mapping.
//!
//! MediaPipe's 52 categories are validated by `vtuber-core` before reaching
//! this module.  Keeping the mapping here makes the semantic choices
//! deterministic and prevents model-specific landmark indices from leaking
//! into tracking or avatar code.

use vtuber_core::{
    ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape, ExpressionCoefficients,
    FaceBlendshapeSet, FaceTrackingContractError, GazeSignal, MediaPipeBlendshape,
    RawExpressionObservation,
};

/// Parses exactly the official MediaPipe 52-category set.
pub fn parse_mediapipe_blendshapes(
    pairs: &[(&str, f32)],
) -> Result<FaceBlendshapeSet, FaceTrackingContractError> {
    FaceBlendshapeSet::from_pairs(pairs)
}

/// Converts a typed MediaPipe set into VRM preset-space coefficients.
///
/// Every output is finite and in `[0, 1]`.  A model that lacks a particular
/// VRM preset can later omit that command in the avatar capability adapter;
/// no synthetic landmark-based replacement is inserted here.
#[must_use]
pub fn map_mediapipe_expressions(values: &FaceBlendshapeSet) -> ExpressionCoefficients {
    let left_smile = value(values, MediaPipeBlendshape::MouthSmileLeft);
    let right_smile = value(values, MediaPipeBlendshape::MouthSmileRight);
    let smile = average(left_smile, right_smile);
    let left_stretch = value(values, MediaPipeBlendshape::MouthStretchLeft);
    let right_stretch = value(values, MediaPipeBlendshape::MouthStretchRight);
    let stretch = average(left_stretch, right_stretch);
    let brow_down = average(
        value(values, MediaPipeBlendshape::BrowDownLeft),
        value(values, MediaPipeBlendshape::BrowDownRight),
    );
    let brow_inner_up = value(values, MediaPipeBlendshape::BrowInnerUp);
    let frown = average(
        value(values, MediaPipeBlendshape::MouthFrownLeft),
        value(values, MediaPipeBlendshape::MouthFrownRight),
    );
    let eye_wide = average(
        value(values, MediaPipeBlendshape::EyeWideLeft),
        value(values, MediaPipeBlendshape::EyeWideRight),
    );
    let jaw_open = value(values, MediaPipeBlendshape::JawOpen);

    ExpressionCoefficients {
        blink_left: value(values, MediaPipeBlendshape::EyeBlinkLeft),
        blink_right: value(values, MediaPipeBlendshape::EyeBlinkRight),
        aa: jaw_open,
        ih: smile,
        ou: value(values, MediaPipeBlendshape::MouthPucker),
        ee: stretch,
        oh: value(values, MediaPipeBlendshape::MouthFunnel),
        look_left: look_left(values),
        look_right: look_right(values),
        look_up: look_up(values),
        look_down: look_down(values),
        happy: smile,
        angry: brow_down,
        sad: frown.max(brow_inner_up),
        relaxed: value(values, MediaPipeBlendshape::Neutral),
        surprised: jaw_open.max(eye_wide),
    }
}

/// Converts a typed MediaPipe set into the ARKit52 Perfect Sync contract.
///
/// MediaPipe's 51 expression categories share their semantics with the ARKit
/// channels of the same name, so they are copied by semantic rather than by
/// array position.  MediaPipe's `_neutral` category has no ARKit semantic and
/// is dropped.  MediaPipe publishes no tongue category, so `TongueOut`, the
/// one ARKit channel without a MediaPipe source, stays `0.0`.
// Invariant: `FaceBlendshapeSet` rejects non-finite scores and scores outside
// `[0, 1]`, and every slot without a MediaPipe source is a literal `0.0`, so
// the ARKit52 validation cannot reject this vector.
#[allow(clippy::expect_used)]
#[must_use]
pub fn map_mediapipe_perfect_sync(values: &FaceBlendshapeSet) -> Arkit52Coefficients {
    let mut coefficients = [0.0; ARKIT52_CHANNEL_COUNT];
    for (slot, channel) in coefficients.iter_mut().zip(ArkitBlendshape::ALL) {
        *slot = mediapipe_channel(channel).map_or(0.0, |category| value(values, category));
    }
    Arkit52Coefficients::try_from_array(coefficients)
        .expect("MediaPipe scores are validated to be finite and within [0, 1]")
}

/// Returns the MediaPipe category that carries one ARKit semantic.
///
/// `TongueOut` is the only channel without a MediaPipe source; a missing
/// source makes the channel `0.0` instead of inventing a value.
const fn mediapipe_channel(channel: ArkitBlendshape) -> Option<MediaPipeBlendshape> {
    match channel {
        ArkitBlendshape::TongueOut => None,
        ArkitBlendshape::BrowDownLeft => Some(MediaPipeBlendshape::BrowDownLeft),
        ArkitBlendshape::BrowDownRight => Some(MediaPipeBlendshape::BrowDownRight),
        ArkitBlendshape::BrowInnerUp => Some(MediaPipeBlendshape::BrowInnerUp),
        ArkitBlendshape::BrowOuterUpLeft => Some(MediaPipeBlendshape::BrowOuterUpLeft),
        ArkitBlendshape::BrowOuterUpRight => Some(MediaPipeBlendshape::BrowOuterUpRight),
        ArkitBlendshape::CheekPuff => Some(MediaPipeBlendshape::CheekPuff),
        ArkitBlendshape::CheekSquintLeft => Some(MediaPipeBlendshape::CheekSquintLeft),
        ArkitBlendshape::CheekSquintRight => Some(MediaPipeBlendshape::CheekSquintRight),
        ArkitBlendshape::EyeBlinkLeft => Some(MediaPipeBlendshape::EyeBlinkLeft),
        ArkitBlendshape::EyeBlinkRight => Some(MediaPipeBlendshape::EyeBlinkRight),
        ArkitBlendshape::EyeLookDownLeft => Some(MediaPipeBlendshape::EyeLookDownLeft),
        ArkitBlendshape::EyeLookDownRight => Some(MediaPipeBlendshape::EyeLookDownRight),
        ArkitBlendshape::EyeLookInLeft => Some(MediaPipeBlendshape::EyeLookInLeft),
        ArkitBlendshape::EyeLookInRight => Some(MediaPipeBlendshape::EyeLookInRight),
        ArkitBlendshape::EyeLookOutLeft => Some(MediaPipeBlendshape::EyeLookOutLeft),
        ArkitBlendshape::EyeLookOutRight => Some(MediaPipeBlendshape::EyeLookOutRight),
        ArkitBlendshape::EyeLookUpLeft => Some(MediaPipeBlendshape::EyeLookUpLeft),
        ArkitBlendshape::EyeLookUpRight => Some(MediaPipeBlendshape::EyeLookUpRight),
        ArkitBlendshape::EyeSquintLeft => Some(MediaPipeBlendshape::EyeSquintLeft),
        ArkitBlendshape::EyeSquintRight => Some(MediaPipeBlendshape::EyeSquintRight),
        ArkitBlendshape::EyeWideLeft => Some(MediaPipeBlendshape::EyeWideLeft),
        ArkitBlendshape::EyeWideRight => Some(MediaPipeBlendshape::EyeWideRight),
        ArkitBlendshape::JawForward => Some(MediaPipeBlendshape::JawForward),
        ArkitBlendshape::JawLeft => Some(MediaPipeBlendshape::JawLeft),
        ArkitBlendshape::JawOpen => Some(MediaPipeBlendshape::JawOpen),
        ArkitBlendshape::JawRight => Some(MediaPipeBlendshape::JawRight),
        ArkitBlendshape::MouthClose => Some(MediaPipeBlendshape::MouthClose),
        ArkitBlendshape::MouthDimpleLeft => Some(MediaPipeBlendshape::MouthDimpleLeft),
        ArkitBlendshape::MouthDimpleRight => Some(MediaPipeBlendshape::MouthDimpleRight),
        ArkitBlendshape::MouthFrownLeft => Some(MediaPipeBlendshape::MouthFrownLeft),
        ArkitBlendshape::MouthFrownRight => Some(MediaPipeBlendshape::MouthFrownRight),
        ArkitBlendshape::MouthFunnel => Some(MediaPipeBlendshape::MouthFunnel),
        ArkitBlendshape::MouthLeft => Some(MediaPipeBlendshape::MouthLeft),
        ArkitBlendshape::MouthLowerDownLeft => Some(MediaPipeBlendshape::MouthLowerDownLeft),
        ArkitBlendshape::MouthLowerDownRight => Some(MediaPipeBlendshape::MouthLowerDownRight),
        ArkitBlendshape::MouthPressLeft => Some(MediaPipeBlendshape::MouthPressLeft),
        ArkitBlendshape::MouthPressRight => Some(MediaPipeBlendshape::MouthPressRight),
        ArkitBlendshape::MouthPucker => Some(MediaPipeBlendshape::MouthPucker),
        ArkitBlendshape::MouthRight => Some(MediaPipeBlendshape::MouthRight),
        ArkitBlendshape::MouthRollLower => Some(MediaPipeBlendshape::MouthRollLower),
        ArkitBlendshape::MouthRollUpper => Some(MediaPipeBlendshape::MouthRollUpper),
        ArkitBlendshape::MouthShrugLower => Some(MediaPipeBlendshape::MouthShrugLower),
        ArkitBlendshape::MouthShrugUpper => Some(MediaPipeBlendshape::MouthShrugUpper),
        ArkitBlendshape::MouthSmileLeft => Some(MediaPipeBlendshape::MouthSmileLeft),
        ArkitBlendshape::MouthSmileRight => Some(MediaPipeBlendshape::MouthSmileRight),
        ArkitBlendshape::MouthStretchLeft => Some(MediaPipeBlendshape::MouthStretchLeft),
        ArkitBlendshape::MouthStretchRight => Some(MediaPipeBlendshape::MouthStretchRight),
        ArkitBlendshape::MouthUpperUpLeft => Some(MediaPipeBlendshape::MouthUpperUpLeft),
        ArkitBlendshape::MouthUpperUpRight => Some(MediaPipeBlendshape::MouthUpperUpRight),
        ArkitBlendshape::NoseSneerLeft => Some(MediaPipeBlendshape::NoseSneerLeft),
        ArkitBlendshape::NoseSneerRight => Some(MediaPipeBlendshape::NoseSneerRight),
    }
}

/// Extracts raw blink/mouth channels for the existing smoothing filter.
#[must_use]
pub fn map_mediapipe_raw_expressions(
    values: &FaceBlendshapeSet,
    face_confidence: f32,
) -> RawExpressionObservation {
    let confidence = face_confidence.clamp(0.0, 1.0);
    RawExpressionObservation {
        blink_left: value(values, MediaPipeBlendshape::EyeBlinkLeft),
        blink_left_confidence: confidence,
        blink_right: value(values, MediaPipeBlendshape::EyeBlinkRight),
        blink_right_confidence: confidence,
        mouth_open: value(values, MediaPipeBlendshape::JawOpen),
        mouth_open_confidence: confidence,
    }
}

/// One eye's normalized eye-in-head observation before binocular fusion.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PerEyeGazeObservation {
    /// Horizontal in `[-1, 1]`; unmirrored image right is positive.
    pub horizontal: f32,
    /// Vertical in `[-1, 1]`; up is positive.
    pub vertical: f32,
    /// Eye-openness-derived weight in `[0, 1]`.
    pub weight: f32,
}

/// Binocular diagnostics retained alongside the fused common gaze.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BinocularGazeObservation {
    /// Left-eye observation.
    pub left: PerEyeGazeObservation,
    /// Right-eye observation.
    pub right: PerEyeGazeObservation,
    /// Agreement factor in `[0, 1]`.
    pub agreement: f32,
    /// Common gaze used by VRM LookAt.
    pub common: GazeSignal,
}

/// Converts typed MediaPipe eye channels directly to normalized common gaze.
///
/// This deliberately does not create synthetic named coefficients and does
/// not claim that the model scores are calibrated physical angles.
#[must_use]
pub fn map_mediapipe_gaze(values: &FaceBlendshapeSet) -> GazeSignal {
    observe_mediapipe_gaze(values).common
}

/// Produces per-eye diagnostics and the fused common gaze.
#[must_use]
pub fn observe_mediapipe_gaze(values: &FaceBlendshapeSet) -> BinocularGazeObservation {
    let left = PerEyeGazeObservation {
        horizontal: (value(values, MediaPipeBlendshape::EyeLookOutLeft)
            - value(values, MediaPipeBlendshape::EyeLookInLeft))
        .clamp(-1.0, 1.0),
        vertical: (value(values, MediaPipeBlendshape::EyeLookUpLeft)
            - value(values, MediaPipeBlendshape::EyeLookDownLeft))
        .clamp(-1.0, 1.0),
        weight: (1.0 - value(values, MediaPipeBlendshape::EyeBlinkLeft)).clamp(0.0, 1.0),
    };
    let right = PerEyeGazeObservation {
        horizontal: (value(values, MediaPipeBlendshape::EyeLookInRight)
            - value(values, MediaPipeBlendshape::EyeLookOutRight))
        .clamp(-1.0, 1.0),
        vertical: (value(values, MediaPipeBlendshape::EyeLookUpRight)
            - value(values, MediaPipeBlendshape::EyeLookDownRight))
        .clamp(-1.0, 1.0),
        weight: (1.0 - value(values, MediaPipeBlendshape::EyeBlinkRight)).clamp(0.0, 1.0),
    };
    fuse_binocular_gaze(left, right)
}

/// Fuses two baseline-adjusted per-eye observations into one common gaze.
#[must_use]
pub fn fuse_binocular_gaze(
    left: PerEyeGazeObservation,
    right: PerEyeGazeObservation,
) -> BinocularGazeObservation {
    let disagreement = ((left.horizontal - right.horizontal).abs()
        + (left.vertical - right.vertical).abs())
        * 0.25;
    let agreement = (1.0 - disagreement).clamp(0.0, 1.0);
    let weight_sum = left.weight + right.weight;
    let common = if weight_sum <= 0.15 {
        GazeSignal::UNAVAILABLE
    } else {
        let horizontal =
            (left.horizontal * left.weight + right.horizontal * right.weight) / weight_sum;
        let vertical = (left.vertical * left.weight + right.vertical * right.weight) / weight_sum;
        let openness = (weight_sum * 0.5).clamp(0.0, 1.0);
        let confidence = (openness * agreement).clamp(0.0, 1.0);
        if left.weight > 0.2 && right.weight > 0.2 && agreement >= 0.5 {
            GazeSignal::tracked(horizontal, vertical, confidence)
        } else {
            GazeSignal::degraded(horizontal, vertical, confidence.min(0.5))
        }
    };
    BinocularGazeObservation {
        left,
        right,
        agreement,
        common,
    }
}

fn look_left(values: &FaceBlendshapeSet) -> f32 {
    average(
        value(values, MediaPipeBlendshape::EyeLookInLeft),
        value(values, MediaPipeBlendshape::EyeLookOutRight),
    )
}

fn look_right(values: &FaceBlendshapeSet) -> f32 {
    average(
        value(values, MediaPipeBlendshape::EyeLookOutLeft),
        value(values, MediaPipeBlendshape::EyeLookInRight),
    )
}

fn look_up(values: &FaceBlendshapeSet) -> f32 {
    average(
        value(values, MediaPipeBlendshape::EyeLookUpLeft),
        value(values, MediaPipeBlendshape::EyeLookUpRight),
    )
}

fn look_down(values: &FaceBlendshapeSet) -> f32 {
    average(
        value(values, MediaPipeBlendshape::EyeLookDownLeft),
        value(values, MediaPipeBlendshape::EyeLookDownRight),
    )
}

fn value(values: &FaceBlendshapeSet, category: MediaPipeBlendshape) -> f32 {
    values.get(category).clamp(0.0, 1.0)
}

fn average(left: f32, right: f32) -> f32 {
    ((left + right) * 0.5).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::GazeTrackingState;

    fn set(values: &[(MediaPipeBlendshape, f32)]) -> FaceBlendshapeSet {
        let pairs: Vec<(&str, f32)> = MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| {
                let value = values
                    .iter()
                    .find(|(candidate, _)| *candidate == category)
                    .map_or(0.0, |(_, value)| *value);
                (category.as_str(), value)
            })
            .collect();
        parse_mediapipe_blendshapes(&pairs).expect("official set is complete")
    }

    #[test]
    fn perfect_sync_mapping_copies_shared_semantics_by_name() {
        let values = set(&[
            (MediaPipeBlendshape::JawOpen, 0.7),
            (MediaPipeBlendshape::EyeBlinkLeft, 0.8),
            (MediaPipeBlendshape::EyeBlinkRight, 0.6),
            (MediaPipeBlendshape::BrowInnerUp, 0.3),
            (MediaPipeBlendshape::BrowDownLeft, 0.55),
            (MediaPipeBlendshape::CheekPuff, 0.45),
            (MediaPipeBlendshape::MouthSmileLeft, 0.4),
            (MediaPipeBlendshape::MouthSmileRight, 0.2),
            (MediaPipeBlendshape::EyeLookUpLeft, 0.15),
            (MediaPipeBlendshape::NoseSneerRight, 0.9),
        ]);
        let coefficients = map_mediapipe_perfect_sync(&values);
        assert!((coefficients.get(ArkitBlendshape::JawOpen) - 0.7).abs() < 1.0e-6);
        assert!((coefficients.get(ArkitBlendshape::EyeBlinkLeft) - 0.8).abs() < 1.0e-6);
        assert!((coefficients.get(ArkitBlendshape::EyeBlinkRight) - 0.6).abs() < 1.0e-6);
        assert!((coefficients.get(ArkitBlendshape::BrowInnerUp) - 0.3).abs() < 1.0e-6);
        assert!((coefficients.get(ArkitBlendshape::BrowDownLeft) - 0.55).abs() < 1.0e-6);
        assert!((coefficients.get(ArkitBlendshape::CheekPuff) - 0.45).abs() < 1.0e-6);
        assert!((coefficients.get(ArkitBlendshape::MouthSmileLeft) - 0.4).abs() < 1.0e-6);
        assert!((coefficients.get(ArkitBlendshape::MouthSmileRight) - 0.2).abs() < 1.0e-6);
        assert!((coefficients.get(ArkitBlendshape::EyeLookUpLeft) - 0.15).abs() < 1.0e-6);
        assert!((coefficients.get(ArkitBlendshape::NoseSneerRight) - 0.9).abs() < 1.0e-6);
    }

    #[test]
    fn perfect_sync_mapping_covers_51_unique_mediapipe_categories() {
        let mut sources = Vec::new();
        for channel in ArkitBlendshape::ALL {
            let Some(source) = mediapipe_channel(channel) else {
                continue;
            };
            assert_eq!(
                source.as_str(),
                channel.lower_camel_alias(),
                "semantic mismatch for {channel:?}"
            );
            assert!(
                !sources.contains(&source),
                "MediaPipe category {source:?} is mapped more than once"
            );
            sources.push(source);
        }
        assert_eq!(sources.len(), ARKIT52_CHANNEL_COUNT - 1);
        for category in MediaPipeBlendshape::ALL {
            if category != MediaPipeBlendshape::Neutral {
                assert!(
                    sources.contains(&category),
                    "unmapped category {category:?}"
                );
            }
        }
    }

    #[test]
    fn perfect_sync_mapping_ignores_neutral_and_pins_tongue_out() {
        let neutral_only = set(&[(MediaPipeBlendshape::Neutral, 1.0)]);
        let coefficients = map_mediapipe_perfect_sync(&neutral_only);
        assert_eq!(coefficients, Arkit52Coefficients::default());

        let full_scale = set(&MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| (category, 1.0))
            .collect::<Vec<_>>());
        let coefficients = map_mediapipe_perfect_sync(&full_scale);
        assert_eq!(coefficients.get(ArkitBlendshape::TongueOut), 0.0);
        assert_eq!(coefficients.get(ArkitBlendshape::JawOpen), 1.0);
        assert!(
            coefficients
                .as_array()
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );
    }

    #[test]
    fn parser_rejects_missing_or_unknown_categories() {
        let pairs = vec![("_neutral", 0.0)];
        assert!(matches!(
            parse_mediapipe_blendshapes(&pairs),
            Err(FaceTrackingContractError::BlendshapeCount { .. })
        ));
    }

    #[test]
    fn maps_blink_vowels_emotions_and_gaze_from_typed_categories() {
        let values = set(&[
            (MediaPipeBlendshape::EyeBlinkLeft, 0.8),
            (MediaPipeBlendshape::EyeBlinkRight, 0.6),
            (MediaPipeBlendshape::JawOpen, 0.7),
            (MediaPipeBlendshape::MouthSmileLeft, 0.4),
            (MediaPipeBlendshape::MouthSmileRight, 0.8),
            (MediaPipeBlendshape::MouthPucker, 0.3),
            (MediaPipeBlendshape::EyeLookOutLeft, 0.9),
            (MediaPipeBlendshape::EyeLookInRight, 0.7),
            (MediaPipeBlendshape::EyeLookUpLeft, 0.5),
            (MediaPipeBlendshape::EyeLookUpRight, 0.3),
        ]);
        let expressions = map_mediapipe_expressions(&values);
        assert!((expressions.blink_left - 0.8).abs() < 1.0e-6);
        assert!((expressions.blink_right - 0.6).abs() < 1.0e-6);
        assert!((expressions.aa - 0.7).abs() < 1.0e-6);
        assert!((expressions.ih - 0.6).abs() < 1.0e-6);
        assert!((expressions.happy - 0.6).abs() < 1.0e-6);
        assert!((expressions.look_right - 0.8).abs() < 1.0e-6);
        assert!(expressions.look_left.abs() < 1.0e-6);
        assert!((expressions.look_up - 0.4).abs() < 1.0e-6);
        let gaze = map_mediapipe_gaze(&values);
        assert!((gaze.horizontal - 0.766_666_65).abs() < 1.0e-6);
        assert!((gaze.vertical - 0.366_666_67).abs() < 1.0e-6);
        assert_eq!(gaze.state, GazeTrackingState::Degraded);
    }

    #[test]
    fn binocular_gaze_signs_agreement_and_blink_weight_are_explicit() {
        let right = set(&[
            (MediaPipeBlendshape::EyeLookOutLeft, 0.8),
            (MediaPipeBlendshape::EyeLookInRight, 0.6),
            (MediaPipeBlendshape::EyeLookUpLeft, 0.4),
            (MediaPipeBlendshape::EyeLookUpRight, 0.4),
        ]);
        let observation = observe_mediapipe_gaze(&right);
        assert!(observation.left.horizontal > 0.0);
        assert!(observation.right.horizontal > 0.0);
        assert!(observation.common.horizontal > 0.0);
        assert!(observation.common.vertical > 0.0);

        let disagrees = set(&[
            (MediaPipeBlendshape::EyeLookOutLeft, 1.0),
            (MediaPipeBlendshape::EyeLookOutRight, 1.0),
        ]);
        assert!(
            observe_mediapipe_gaze(&disagrees).common.confidence < observation.common.confidence
        );

        let left_blinked = set(&[
            (MediaPipeBlendshape::EyeBlinkLeft, 1.0),
            (MediaPipeBlendshape::EyeLookOutLeft, 1.0),
            (MediaPipeBlendshape::EyeLookInRight, 0.5),
        ]);
        let blinked = observe_mediapipe_gaze(&left_blinked);
        assert!((blinked.common.horizontal - 0.5).abs() < 1.0e-6);
        assert_eq!(blinked.common.state, GazeTrackingState::Degraded);

        let both_blinked = set(&[
            (MediaPipeBlendshape::EyeBlinkLeft, 1.0),
            (MediaPipeBlendshape::EyeBlinkRight, 1.0),
            (MediaPipeBlendshape::EyeLookOutLeft, 1.0),
            (MediaPipeBlendshape::EyeLookInRight, 1.0),
        ]);
        assert_eq!(
            observe_mediapipe_gaze(&both_blinked).common,
            GazeSignal::UNAVAILABLE
        );
    }

    #[test]
    fn raw_expression_mapping_keeps_confidence_separate_and_bounded() {
        let values = set(&[(MediaPipeBlendshape::JawOpen, 1.0)]);
        let raw = map_mediapipe_raw_expressions(&values, 1.5);
        assert_eq!(raw.mouth_open, 1.0);
        assert_eq!(raw.mouth_open_confidence, 1.0);
        assert!(raw.is_valid());
    }
}
