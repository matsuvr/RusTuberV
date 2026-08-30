//! Raw face observations produced by the inference worker.
//!
//! These types are engine-independent and represent uncalibrated expression
//! coefficients before tracking filters or avatar mapping are applied.

/// Raw expression coefficients extracted from a single face observation.
///
/// Values are normalized to `[0, 1]`.  The corresponding `*_confidence`
/// fields are kept separate so downstream filters can treat low-confidence
/// observations differently from confident zeros.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawExpressionObservation {
    /// Left eye blink coefficient in `[0, 1]`.
    pub blink_left: f32,
    /// Confidence for [`blink_left`](Self::blink_left) in `[0, 1]`.
    pub blink_left_confidence: f32,
    /// Right eye blink coefficient in `[0, 1]`.
    pub blink_right: f32,
    /// Confidence for [`blink_right`](Self::blink_right) in `[0, 1]`.
    pub blink_right_confidence: f32,
    /// Mouth openness coefficient in `[0, 1]`.
    pub mouth_open: f32,
    /// Confidence for [`mouth_open`](Self::mouth_open) in `[0, 1]`.
    pub mouth_open_confidence: f32,
}

impl RawExpressionObservation {
    /// Returns true if all coefficients and confidences are finite and in `[0, 1]`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        is_valid_coefficient(self.blink_left)
            && is_valid_coefficient(self.blink_left_confidence)
            && is_valid_coefficient(self.blink_right)
            && is_valid_coefficient(self.blink_right_confidence)
            && is_valid_coefficient(self.mouth_open)
            && is_valid_coefficient(self.mouth_open_confidence)
    }
}

fn is_valid_coefficient(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_observation_is_valid() {
        let obs = RawExpressionObservation::default();
        assert!(obs.is_valid());
        assert_eq!(obs.blink_left, 0.0);
        assert_eq!(obs.blink_right, 0.0);
        assert_eq!(obs.mouth_open, 0.0);
        assert_eq!(obs.blink_left_confidence, 0.0);
    }

    #[test]
    fn valid_observation_with_confidence() {
        let obs = RawExpressionObservation {
            blink_left: 0.8,
            blink_left_confidence: 0.9,
            blink_right: 0.2,
            blink_right_confidence: 0.95,
            mouth_open: 0.5,
            mouth_open_confidence: 0.85,
        };
        assert!(obs.is_valid());
    }

    #[test]
    fn invalid_when_coefficient_out_of_bounds() {
        let obs = RawExpressionObservation {
            blink_left: 1.5,
            ..RawExpressionObservation::default()
        };
        assert!(!obs.is_valid());
    }

    #[test]
    fn invalid_when_confidence_out_of_bounds() {
        let obs = RawExpressionObservation {
            blink_left_confidence: -0.1,
            ..RawExpressionObservation::default()
        };
        assert!(!obs.is_valid());
    }

    #[test]
    fn invalid_when_non_finite() {
        let obs = RawExpressionObservation {
            mouth_open: f32::NAN,
            ..RawExpressionObservation::default()
        };
        assert!(!obs.is_valid());
    }
}
