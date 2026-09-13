//! Shared per-eye closure judgement and profile validation (Issues #52/#53).
//!
//! The runtime and the offline threshold fitter use this module so the
//! judgement column in the evaluation and the value sent to the avatar come
//! from exactly one implementation.
//!
//! The feature is the raw MediaPipe `EyeBlinkLeft/Right` score expressed as an
//! openness proxy `o = 1 - raw_blink`. `o` is not a physical lid distance and
//! is never called a closure probability. Each eye gets an independent
//! hysteresis pair `0 <= close_at < reopen_at <= 1`; the opposite eye never
//! contributes and closed values are never mirrored.

use serde::{Deserialize, Serialize};

/// Feature identity recorded in every profile and extraction.
pub const EYE_CLOSURE_FEATURE: &str = "mediapipe_raw_eye_blink_openness_v1";
/// Schema version of the serialized profile document.
pub const EYE_CLOSURE_PROFILE_SCHEMA_VERSION: u32 = 1;
/// Algorithm version of the one-dimensional hysteresis judgement.
pub const EYE_CLOSURE_ALGORITHM_VERSION: u32 = 1;

/// Which anatomical eye a threshold belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EyeSide {
    /// The performer's anatomical left eye.
    Left,
    /// The performer's anatomical right eye.
    Right,
}

impl EyeSide {
    /// Stable lowercase name used in reports and CSV rows.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

/// Latched per-eye openness state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EyeOpenness {
    /// The eye is treated as open.
    #[default]
    Open,
    /// The eye is treated as fully closed.
    Closed,
}

impl EyeOpenness {
    /// Returns `true` when the eye is latched closed.
    #[must_use]
    pub const fn is_closed(self) -> bool {
        matches!(self, Self::Closed)
    }
}

/// Threshold validation failures.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum EyeThresholdError {
    /// `close_at` or `reopen_at` was NaN or infinite.
    #[error("eye-closure threshold is non-finite: close_at={close_at}, reopen_at={reopen_at}")]
    NonFinite {
        /// Supplied close threshold.
        close_at: f32,
        /// Supplied reopen threshold.
        reopen_at: f32,
    },
    /// The pair does not satisfy `0 <= close_at < reopen_at <= 1`.
    #[error(
        "eye-closure threshold must satisfy 0 <= close_at < reopen_at <= 1 (close_at={close_at}, reopen_at={reopen_at})"
    )]
    OutOfOrder {
        /// Supplied close threshold.
        close_at: f32,
        /// Supplied reopen threshold.
        reopen_at: f32,
    },
}

/// Profile document validation failures.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum EyeClosureProfileError {
    /// The document uses an unsupported schema version.
    #[error("unsupported eye-closure profile schema version {found}")]
    UnsupportedSchemaVersion {
        /// Version found in the document.
        found: u32,
    },
    /// The document uses an unsupported algorithm version.
    #[error("unsupported eye-closure algorithm version {found}")]
    UnsupportedAlgorithmVersion {
        /// Version found in the document.
        found: u32,
    },
    /// The feature string is not the one this build can judge.
    #[error("unknown eye-closure feature {found:?}")]
    UnknownFeature {
        /// Feature found in the document.
        found: String,
    },
    /// One eye's thresholds are invalid.
    #[error("{side} thresholds are invalid: {source}")]
    InvalidThreshold {
        /// Which eye failed.
        side: &'static str,
        /// Underlying validation failure.
        #[source]
        source: EyeThresholdError,
    },
}

/// One eye's validated hysteresis thresholds in openness units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeThreshold {
    close_at: f32,
    reopen_at: f32,
}

impl EyeThreshold {
    /// Validates and creates a threshold pair.
    ///
    /// # Errors
    ///
    /// Returns [`EyeThresholdError`] for non-finite values or a pair that
    /// does not satisfy `0 <= close_at < reopen_at <= 1`.
    pub fn new(close_at: f32, reopen_at: f32) -> Result<Self, EyeThresholdError> {
        if !close_at.is_finite() || !reopen_at.is_finite() {
            return Err(EyeThresholdError::NonFinite {
                close_at,
                reopen_at,
            });
        }
        if close_at < 0.0 || close_at >= reopen_at || reopen_at > 1.0 {
            return Err(EyeThresholdError::OutOfOrder {
                close_at,
                reopen_at,
            });
        }
        Ok(Self {
            close_at,
            reopen_at,
        })
    }

    /// Openness at or below which an open eye latches closed.
    #[must_use]
    pub const fn close_at(self) -> f32 {
        self.close_at
    }

    /// Openness at or above which a closed eye returns open.
    #[must_use]
    pub const fn reopen_at(self) -> f32 {
        self.reopen_at
    }

    /// Advances one eye's latch for a fresh openness observation.
    ///
    /// An open eye closes exactly at `o <= close_at`; a closed eye reopens at
    /// `o >= reopen_at`; everything else keeps the previous state. A NaN
    /// observation keeps the previous state and never toggles.
    #[must_use]
    pub fn decide(self, previous: EyeOpenness, openness: f32) -> EyeOpenness {
        match previous {
            EyeOpenness::Open if openness <= self.close_at => EyeOpenness::Closed,
            EyeOpenness::Closed if openness >= self.reopen_at => EyeOpenness::Open,
            _ => previous,
        }
    }

    /// Stateless classification of a single sample for comparison reports.
    ///
    /// This is not a state machine, so it has no equality-point oscillation;
    /// the midpoint keeps the comparison neutral between the two edges.
    #[must_use]
    pub fn is_closed_stateless(self, openness: f32) -> bool {
        openness <= (self.close_at + self.reopen_at) * 0.5
    }
}

/// Both eyes' validated thresholds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeClosureThresholds {
    left: EyeThreshold,
    right: EyeThreshold,
}

impl EyeClosureThresholds {
    /// Bundles validated per-eye thresholds.
    #[must_use]
    pub const fn new(left: EyeThreshold, right: EyeThreshold) -> Self {
        Self { left, right }
    }

    /// Left-eye thresholds.
    #[must_use]
    pub const fn left(&self) -> EyeThreshold {
        self.left
    }

    /// Right-eye thresholds.
    #[must_use]
    pub const fn right(&self) -> EyeThreshold {
        self.right
    }

    /// Returns the threshold for one side.
    #[must_use]
    pub const fn for_side(&self, side: EyeSide) -> EyeThreshold {
        match side {
            EyeSide::Left => self.left,
            EyeSide::Right => self.right,
        }
    }
}

/// Latched closure state for both eyes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EyeClosureState {
    /// Left-eye latch.
    pub left: EyeOpenness,
    /// Right-eye latch.
    pub right: EyeOpenness,
}

impl EyeClosureState {
    /// Returns the latch for one side.
    #[must_use]
    pub const fn for_side(&self, side: EyeSide) -> EyeOpenness {
        match side {
            EyeSide::Left => self.left,
            EyeSide::Right => self.right,
        }
    }
}

/// A fresh per-eye observation in openness units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EyeClosureObservation {
    /// Left-eye openness, or `None` when the eye was not observed.
    pub left_openness: Option<f32>,
    /// Right-eye openness, or `None` when the eye was not observed.
    pub right_openness: Option<f32>,
}

/// Stateful hysteresis tracker shared by the runtime and the evaluator.
///
/// Re-feeding the same `source_seq` never advances the latch or counts as a
/// new closure event, so the per-render-tick reuse of one inference sample
/// cannot inflate closure time. A sequence gap or an explicit [`Self::reset`]
/// drops any carried closure before the next fresh observation.
#[derive(Clone, Debug, PartialEq)]
pub struct EyeClosureTracker {
    thresholds: EyeClosureThresholds,
    state: EyeClosureState,
    last_seq: Option<u64>,
}

impl EyeClosureTracker {
    /// Creates a tracker at the open state.
    #[must_use]
    pub const fn new(thresholds: EyeClosureThresholds) -> Self {
        Self {
            thresholds,
            state: EyeClosureState {
                left: EyeOpenness::Open,
                right: EyeOpenness::Open,
            },
            last_seq: None,
        }
    }

    /// Returns the configured thresholds.
    #[must_use]
    pub const fn thresholds(&self) -> &EyeClosureThresholds {
        &self.thresholds
    }

    /// Returns the current latch.
    #[must_use]
    pub const fn state(&self) -> EyeClosureState {
        self.state
    }

    /// Drops the latch and the sequence boundary (session/stop/reset).
    pub fn reset(&mut self) {
        self.state = EyeClosureState::default();
        self.last_seq = None;
    }

    /// Advances the latch for one fresh sample.
    ///
    /// The same `seq` as the previous call is a no-op. A non-contiguous
    /// sequence resets the latch before judging the new sample. An eye whose
    /// openness is `None` keeps its previous latch and is never inferred from
    /// the opposite eye.
    pub fn observe(&mut self, seq: u64, observation: EyeClosureObservation) -> EyeClosureState {
        if self.last_seq == Some(seq) {
            return self.state;
        }
        let contiguous = self.last_seq.is_some_and(|last| seq == last + 1);
        if !contiguous {
            self.state = EyeClosureState::default();
        }
        self.last_seq = Some(seq);
        self.state.left = decide_side(
            self.thresholds.left,
            self.state.left,
            observation.left_openness,
        );
        self.state.right = decide_side(
            self.thresholds.right,
            self.state.right,
            observation.right_openness,
        );
        self.state
    }
}

fn decide_side(
    threshold: EyeThreshold,
    previous: EyeOpenness,
    openness: Option<f32>,
) -> EyeOpenness {
    match openness {
        Some(value) if value.is_finite() => threshold.decide(previous, value),
        _ => previous,
    }
}

/// Threshold values as stored in a profile document, before validation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EyeThresholdValues {
    /// Openness at or below which the eye closes.
    pub close_at: f32,
    /// Openness at or above which the eye reopens.
    pub reopen_at: f32,
}

/// Verification status carried by a profile document.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EyeClosureVerificationStatus {
    /// Fit on train/validation only; not installable at runtime.
    Candidate,
    /// Held-out evaluated; installable.
    Verified,
}

/// Inference fingerprints recorded beside the thresholds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EyeClosureFingerprints {
    /// MediaPipe task bundle hash observed during fitting, when known.
    pub task_bundle_sha256: Option<String>,
    /// Feature pipeline identity.
    pub feature: String,
    /// Preprocessing identity, when known.
    pub preprocess: Option<String>,
}

/// Serialized eye-closure profile document.
///
/// The type is only a transport shape; call [`Self::validate`] to obtain
/// validated thresholds. A `Verified` status is not by itself a guarantee:
/// the caller must match the fingerprints to its own inference contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EyeClosureProfileDocument {
    /// Document schema version.
    pub schema_version: u32,
    /// Judgement algorithm version.
    pub algorithm_version: u32,
    /// Feature identity.
    pub feature: String,
    /// Verification status.
    pub status: EyeClosureVerificationStatus,
    /// Left-eye thresholds.
    pub left: EyeThresholdValues,
    /// Right-eye thresholds.
    pub right: EyeThresholdValues,
    /// Inference fingerprints.
    pub fingerprints: EyeClosureFingerprints,
    /// Free-form applicability note, e.g. webcam conditions.
    #[serde(default)]
    pub applies_to: Option<String>,
}

impl EyeClosureProfileDocument {
    /// Validates schema, feature, and threshold values.
    ///
    /// # Errors
    ///
    /// Returns [`EyeClosureProfileError`] for an unsupported schema or
    /// algorithm version, a foreign feature string, or invalid thresholds.
    pub fn validate(&self) -> Result<EyeClosureThresholds, EyeClosureProfileError> {
        if self.schema_version != EYE_CLOSURE_PROFILE_SCHEMA_VERSION {
            return Err(EyeClosureProfileError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        if self.algorithm_version != EYE_CLOSURE_ALGORITHM_VERSION {
            return Err(EyeClosureProfileError::UnsupportedAlgorithmVersion {
                found: self.algorithm_version,
            });
        }
        if self.feature != EYE_CLOSURE_FEATURE {
            return Err(EyeClosureProfileError::UnknownFeature {
                found: self.feature.clone(),
            });
        }
        let left = validate_side(EyeSide::Left, self.left)?;
        let right = validate_side(EyeSide::Right, self.right)?;
        Ok(EyeClosureThresholds::new(left, right))
    }
}

fn validate_side(
    side: EyeSide,
    values: EyeThresholdValues,
) -> Result<EyeThreshold, EyeClosureProfileError> {
    EyeThreshold::new(values.close_at, values.reopen_at).map_err(|source| {
        EyeClosureProfileError::InvalidThreshold {
            side: side.as_str(),
            source,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds(left: (f32, f32), right: (f32, f32)) -> EyeClosureThresholds {
        EyeClosureThresholds::new(
            EyeThreshold::new(left.0, left.1).unwrap(),
            EyeThreshold::new(right.0, right.1).unwrap(),
        )
    }

    #[test]
    fn threshold_validation_rejects_non_finite_and_out_of_order() {
        assert!(EyeThreshold::new(f32::NAN, 0.5).is_err());
        assert!(EyeThreshold::new(-0.1, 0.5).is_err());
        assert!(EyeThreshold::new(0.5, 0.5).is_err());
        assert!(EyeThreshold::new(0.6, 0.5).is_err());
        assert!(EyeThreshold::new(0.0, 1.0).is_ok());
    }

    #[test]
    fn equality_at_each_edge_matches_the_documented_direction() {
        let threshold = EyeThreshold::new(0.4, 0.6).unwrap();
        assert_eq!(
            threshold.decide(EyeOpenness::Open, 0.4),
            EyeOpenness::Closed
        );
        assert_eq!(
            threshold.decide(EyeOpenness::Closed, 0.6),
            EyeOpenness::Open
        );
    }

    #[test]
    fn hysteresis_holds_half_open_values_closed() {
        let threshold = EyeThreshold::new(0.4, 0.6).unwrap();
        let closed = threshold.decide(EyeOpenness::Open, 0.35);
        assert_eq!(closed, EyeOpenness::Closed);
        // 0.5 is between the edges: a closed eye stays closed, an open eye
        // stays open, so a single sample can never oscillate.
        assert_eq!(threshold.decide(closed, 0.5), EyeOpenness::Closed);
        assert_eq!(threshold.decide(EyeOpenness::Open, 0.5), EyeOpenness::Open);
    }

    #[test]
    fn nan_observation_keeps_the_previous_state() {
        let threshold = EyeThreshold::new(0.4, 0.6).unwrap();
        assert_eq!(
            threshold.decide(EyeOpenness::Closed, f32::NAN),
            EyeOpenness::Closed
        );
        assert_eq!(
            threshold.decide(EyeOpenness::Open, f32::NAN),
            EyeOpenness::Open
        );
    }

    #[test]
    fn re_feeding_the_same_sequence_never_advances_the_latch() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.4, 0.6), (0.4, 0.6)));
        let closed = EyeClosureObservation {
            left_openness: Some(0.1),
            right_openness: Some(0.1),
        };
        assert_eq!(
            tracker.observe(10, closed),
            EyeClosureState {
                left: EyeOpenness::Closed,
                right: EyeOpenness::Closed,
            }
        );
        // A re-used sample with an open value must not reopen or re-count.
        let stale = EyeClosureObservation {
            left_openness: Some(0.9),
            right_openness: Some(0.9),
        };
        assert_eq!(tracker.observe(10, stale), tracker.state());
    }

    #[test]
    fn a_sequence_gap_drops_a_carried_closure() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.4, 0.6), (0.4, 0.6)));
        tracker.observe(
            1,
            EyeClosureObservation {
                left_openness: Some(0.1),
                right_openness: Some(0.1),
            },
        );
        assert!(tracker.state().left.is_closed());
        // seq 3 is non-contiguous: the closure must not carry across the gap.
        let state = tracker.observe(3, EyeClosureObservation::default());
        assert_eq!(state, EyeClosureState::default());
    }

    #[test]
    fn the_first_closed_observation_closes_immediately() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.4, 0.6), (0.4, 0.6)));
        let state = tracker.observe(
            1,
            EyeClosureObservation {
                left_openness: Some(0.0),
                right_openness: Some(0.5),
            },
        );
        assert!(state.left.is_closed());
        assert!(!state.right.is_closed());
    }

    #[test]
    fn a_missing_eye_keeps_its_latch_and_is_not_mirrored() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.4, 0.6), (0.4, 0.6)));
        tracker.observe(
            1,
            EyeClosureObservation {
                left_openness: Some(0.0),
                right_openness: Some(0.9),
            },
        );
        assert!(tracker.state().left.is_closed());
        let state = tracker.observe(
            2,
            EyeClosureObservation {
                left_openness: None,
                right_openness: Some(0.9),
            },
        );
        assert!(state.left.is_closed(), "missing left keeps its latch");
        assert!(!state.right.is_closed());
    }

    #[test]
    fn sides_use_independent_thresholds() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.2, 0.3), (0.7, 0.8)));
        let state = tracker.observe(
            1,
            EyeClosureObservation {
                left_openness: Some(0.25),
                right_openness: Some(0.25),
            },
        );
        assert!(
            !state.left.is_closed(),
            "0.25 is above the left close point"
        );
        assert!(
            state.right.is_closed(),
            "0.25 is below the right close point"
        );
    }

    #[test]
    fn document_validation_rejects_wrong_versions_and_features() {
        let document = EyeClosureProfileDocument {
            schema_version: 99,
            algorithm_version: EYE_CLOSURE_ALGORITHM_VERSION,
            feature: EYE_CLOSURE_FEATURE.into(),
            status: EyeClosureVerificationStatus::Verified,
            left: EyeThresholdValues {
                close_at: 0.4,
                reopen_at: 0.6,
            },
            right: EyeThresholdValues {
                close_at: 0.4,
                reopen_at: 0.6,
            },
            fingerprints: EyeClosureFingerprints {
                task_bundle_sha256: None,
                feature: EYE_CLOSURE_FEATURE.into(),
                preprocess: None,
            },
            applies_to: None,
        };
        assert!(matches!(
            document.validate(),
            Err(EyeClosureProfileError::UnsupportedSchemaVersion { found: 99 })
        ));
        let mut foreign = document.clone();
        foreign.schema_version = EYE_CLOSURE_PROFILE_SCHEMA_VERSION;
        foreign.feature = "something_else".into();
        assert!(matches!(
            foreign.validate(),
            Err(EyeClosureProfileError::UnknownFeature { .. })
        ));
    }

    #[test]
    fn document_validation_reports_the_invalid_side() {
        let document = EyeClosureProfileDocument {
            schema_version: EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
            algorithm_version: EYE_CLOSURE_ALGORITHM_VERSION,
            feature: EYE_CLOSURE_FEATURE.into(),
            status: EyeClosureVerificationStatus::Candidate,
            left: EyeThresholdValues {
                close_at: 0.4,
                reopen_at: 0.6,
            },
            right: EyeThresholdValues {
                close_at: 0.7,
                reopen_at: 0.2,
            },
            fingerprints: EyeClosureFingerprints {
                task_bundle_sha256: None,
                feature: EYE_CLOSURE_FEATURE.into(),
                preprocess: None,
            },
            applies_to: None,
        };
        match document.validate() {
            Err(EyeClosureProfileError::InvalidThreshold { side, .. }) => assert_eq!(side, "right"),
            other => panic!("expected a right-side failure, got {other:?}"),
        }
    }
}
