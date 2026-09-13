//! Shared per-eye closure judgement and profile validation (Issues #52-#66).
//!
//! The runtime and the offline threshold fitter use this module so the
//! judgement column in the evaluation and the value sent to the avatar come
//! from exactly one implementation.
//!
//! Two features exist:
//!
//! - `mediapipe_raw_eye_blink_openness_v1`: the raw MediaPipe
//!   `EyeBlinkLeft/Right` score as an openness proxy `o = 1 - raw_blink`. It is
//!   retained for the offline R baseline comparison only; no runtime profile
//!   uses it after algorithm v2.
//! - `mediapipe_max_lid_gap_blink_v2`: the per-eye `max_lid_gap_ratio` from
//!   [`crate::eye_geometry`] as the required closure condition, with the same
//!   eye's raw blink as an optional auxiliary condition.
//!
//! Missing or non-finite per-eye observations become [`EyeOpenness::Unknown`],
//! never a carried-over [`EyeOpenness::Closed`]. A closure pin is only emitted
//! for [`EyeOpenness::Closed`].

use serde::{Deserialize, Serialize};

use crate::eye_geometry::EyeClosureFeatures;
use vtuber_core::{FrameSeq, MonoTimeNs};

/// Feature identity recorded in every profile and extraction.
pub const EYE_CLOSURE_FEATURE: &str = "mediapipe_max_lid_gap_blink_v2";
/// Schema version of the serialized profile document.
pub const EYE_CLOSURE_PROFILE_SCHEMA_VERSION: u32 = 2;
/// Algorithm version of the closure judgement.
pub const EYE_CLOSURE_ALGORITHM_VERSION: u32 = 2;
/// Capture-time gap at or above which a fresh sample starts a new segment.
///
/// This is the initial value moved from the earlier offline evaluator's 250 ms
/// duration clamp; it is not claimed to be a measured optimum.
pub const EYE_CLOSURE_SAMPLE_GAP_NS: u64 = 250_000_000;

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
    /// No usable observation; the eye must not emit a closure pin.
    Unknown,
}

impl EyeOpenness {
    /// Returns `true` only when the eye is latched fully closed.
    #[must_use]
    pub const fn is_closed(self) -> bool {
        matches!(self, Self::Closed)
    }
}

/// How a fresh observation relates to the previous one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeSampleStep {
    /// The same inference sample was consumed again (same sequence number).
    Reused,
    /// A new observation; `reset` drops any latched closure first.
    Fresh {
        /// Whether the latch must be dropped before judging this sample.
        reset: bool,
    },
}

/// Classifies one observation against the previous accepted observation.
///
/// A consumer that re-feeds the same sample (same `frame_seq`) is [`EyeSampleStep::Reused`]
/// and must not advance the latch. A sequence-number jump with an ordinary
/// capture-time step is a normal fresh sample, not a discontinuity. The first
/// sample, an explicit disconnect, a sequence or capture-time regression, and a
/// capture-time gap of at least [`EYE_CLOSURE_SAMPLE_GAP_NS`] drop the latch.
#[must_use]
pub fn classify_eye_sample(
    previous: Option<(FrameSeq, MonoTimeNs)>,
    current: (FrameSeq, MonoTimeNs),
    disconnected: bool,
) -> EyeSampleStep {
    let Some((previous_seq, previous_time)) = previous else {
        return EyeSampleStep::Fresh { reset: true };
    };
    if disconnected {
        return EyeSampleStep::Fresh { reset: true };
    }
    if current.0 == previous_seq {
        return EyeSampleStep::Reused;
    }
    let reset = current.0 < previous_seq
        || current.1 <= previous_time
        || current.1.0 - previous_time.0 >= EYE_CLOSURE_SAMPLE_GAP_NS;
    EyeSampleStep::Fresh { reset }
}

/// Threshold validation failures shared by both judgement versions.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum EyeThresholdError {
    /// A threshold value was NaN or infinite.
    #[error("eye-closure threshold is non-finite: close={close}, reopen={reopen}")]
    NonFinite {
        /// Supplied close value.
        close: f32,
        /// Supplied reopen value.
        reopen: f32,
    },
    /// The pair does not satisfy `0 <= close < reopen`.
    #[error(
        "eye-closure threshold must satisfy 0 <= close < reopen (close={close}, reopen={reopen})"
    )]
    OutOfOrder {
        /// Supplied close value.
        close: f32,
        /// Supplied reopen value.
        reopen: f32,
    },
    /// The auxiliary blink condition is outside `[0, 1]`.
    #[error("eye-closure min_blink must satisfy 0 <= min_blink <= 1, found {min_blink}")]
    MinBlinkOutOfRange {
        /// Supplied auxiliary blink condition.
        min_blink: f32,
    },
}

/// One eye's validated hysteresis thresholds in openness units (v1, offline).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeThreshold {
    close_at: f32,
    reopen_at: f32,
}

impl EyeThreshold {
    /// Validates and creates an openness hysteresis pair.
    ///
    /// # Errors
    ///
    /// Returns [`EyeThresholdError`] for non-finite values or a pair that does
    /// not satisfy `0 <= close_at < reopen_at <= 1`.
    pub fn new(close_at: f32, reopen_at: f32) -> Result<Self, EyeThresholdError> {
        if !close_at.is_finite() || !reopen_at.is_finite() {
            return Err(EyeThresholdError::NonFinite {
                close: close_at,
                reopen: reopen_at,
            });
        }
        if close_at < 0.0 || close_at >= reopen_at || reopen_at > 1.0 {
            return Err(EyeThresholdError::OutOfOrder {
                close: close_at,
                reopen: reopen_at,
            });
        }
        Ok(Self {
            close_at,
            reopen_at,
        })
    }

    /// Openness at or below which an eye in entry position latches closed.
    #[must_use]
    pub const fn close_at(self) -> f32 {
        self.close_at
    }

    /// Openness at or above which a closed eye returns open.
    #[must_use]
    pub const fn reopen_at(self) -> f32 {
        self.reopen_at
    }

    /// Advances one eye's latch for a finite fresh openness observation.
    ///
    /// [`EyeOpenness::Closed`] keeps its latch until `o >= reopen_at`. Every
    /// other preceding state uses the entry condition `o <= close_at`; that is
    /// how a valid observation after [`EyeOpenness::Unknown`] recovers.
    #[must_use]
    pub fn decide(self, previous: EyeOpenness, openness: f32) -> EyeOpenness {
        match previous {
            EyeOpenness::Closed if openness >= self.reopen_at => EyeOpenness::Open,
            _ if openness <= self.close_at => EyeOpenness::Closed,
            EyeOpenness::Closed => EyeOpenness::Closed,
            _ => EyeOpenness::Open,
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

/// Both eyes' validated v1 openness thresholds.
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

/// A fresh per-eye v1 observation in openness units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EyeClosureObservation {
    /// Left-eye openness, or `None` when the eye was not observed.
    pub left_openness: Option<f32>,
    /// Right-eye openness, or `None` when the eye was not observed.
    pub right_openness: Option<f32>,
}

/// One eye's validated geometry thresholds (v2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeGeometryThreshold {
    close_gap: f32,
    reopen_gap: f32,
    min_blink: f32,
}

impl EyeGeometryThreshold {
    /// Validates and creates a geometry threshold triple.
    ///
    /// # Errors
    ///
    /// Returns [`EyeThresholdError`] for non-finite values, a pair that does
    /// not satisfy `0 <= close_gap < reopen_gap`, or a `min_blink` outside
    /// `[0, 1]`.
    pub fn new(close_gap: f32, reopen_gap: f32, min_blink: f32) -> Result<Self, EyeThresholdError> {
        if !close_gap.is_finite() || !reopen_gap.is_finite() || !min_blink.is_finite() {
            return Err(EyeThresholdError::NonFinite {
                close: close_gap,
                reopen: reopen_gap,
            });
        }
        if close_gap < 0.0 || close_gap >= reopen_gap {
            return Err(EyeThresholdError::OutOfOrder {
                close: close_gap,
                reopen: reopen_gap,
            });
        }
        if !(0.0..=1.0).contains(&min_blink) {
            return Err(EyeThresholdError::MinBlinkOutOfRange { min_blink });
        }
        Ok(Self {
            close_gap,
            reopen_gap,
            min_blink,
        })
    }

    /// Lid-gap ratio at or below which an eye in entry position closes.
    #[must_use]
    pub const fn close_gap(self) -> f32 {
        self.close_gap
    }

    /// Lid-gap ratio at or above which a closed eye reopens.
    #[must_use]
    pub const fn reopen_gap(self) -> f32 {
        self.reopen_gap
    }

    /// Auxiliary raw blink score required at the entry edge.
    #[must_use]
    pub const fn min_blink(self) -> f32 {
        self.min_blink
    }

    /// Width of the hysteresis band in lid-gap units.
    #[must_use]
    pub fn width(self) -> f32 {
        self.reopen_gap - self.close_gap
    }
}

/// Both eyes' validated geometry thresholds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeGeometryThresholds {
    left: EyeGeometryThreshold,
    right: EyeGeometryThreshold,
}

impl EyeGeometryThresholds {
    /// Bundles validated per-eye geometry thresholds.
    #[must_use]
    pub const fn new(left: EyeGeometryThreshold, right: EyeGeometryThreshold) -> Self {
        Self { left, right }
    }

    /// Left-eye thresholds.
    #[must_use]
    pub const fn left(&self) -> EyeGeometryThreshold {
        self.left
    }

    /// Right-eye thresholds.
    #[must_use]
    pub const fn right(&self) -> EyeGeometryThreshold {
        self.right
    }

    /// Returns the threshold for one side.
    #[must_use]
    pub const fn for_side(&self, side: EyeSide) -> EyeGeometryThreshold {
        match side {
            EyeSide::Left => self.left,
            EyeSide::Right => self.right,
        }
    }
}

/// Advances one eye with the geometry-primary decision (Issue #66).
///
/// A missing or non-finite observation is [`EyeOpenness::Unknown`]: it is not
/// an open observation, it only means there is no closure pin. From
/// [`EyeOpenness::Open`] or [`EyeOpenness::Unknown`], closing requires the
/// lid-gap entry condition and the auxiliary blink condition; from
/// [`EyeOpenness::Closed`], only the lid-gap release condition reopens.
#[must_use]
pub fn decide_geometry_eye(
    previous: EyeOpenness,
    observed: Option<EyeClosureFeatures>,
    threshold: EyeGeometryThreshold,
) -> EyeOpenness {
    let Some(features) = observed else {
        return EyeOpenness::Unknown;
    };
    if !features.lid_gap_ratio.is_finite() || !features.raw_blink.is_finite() {
        return EyeOpenness::Unknown;
    }
    match previous {
        EyeOpenness::Closed if features.lid_gap_ratio >= threshold.reopen_gap => EyeOpenness::Open,
        EyeOpenness::Closed => EyeOpenness::Closed,
        _ if features.lid_gap_ratio <= threshold.close_gap
            && features.raw_blink >= threshold.min_blink =>
        {
            EyeOpenness::Closed
        }
        _ => EyeOpenness::Open,
    }
}

/// Advances both eyes with independent geometry decisions.
#[must_use]
pub fn decide_geometry_pair(
    previous: EyeClosureState,
    observed: [Option<EyeClosureFeatures>; 2],
    thresholds: [EyeGeometryThreshold; 2],
) -> EyeClosureState {
    let [left_observed, right_observed] = observed;
    let [left_threshold, right_threshold] = thresholds;
    EyeClosureState {
        left: decide_geometry_eye(previous.left, left_observed, left_threshold),
        right: decide_geometry_eye(previous.right, right_observed, right_threshold),
    }
}

/// Stateful v1 hysteresis tracker used by the offline R baseline.
///
/// Re-feeding the same sample never advances the latch. The sequence and
/// capture-time rules come from [`classify_eye_sample`].
#[derive(Clone, Debug, PartialEq)]
pub struct EyeClosureTracker {
    thresholds: EyeClosureThresholds,
    state: EyeClosureState,
    last: Option<(FrameSeq, MonoTimeNs)>,
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
            last: None,
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

    /// Drops the latch and the sample boundary (session/stop/reset).
    pub fn reset(&mut self) {
        self.state = EyeClosureState::default();
        self.last = None;
    }

    /// Advances the latch for one observed sample.
    ///
    /// A missing or non-finite eye becomes [`EyeOpenness::Unknown`] for that
    /// eye only and is never inferred from the opposite eye.
    pub fn observe(
        &mut self,
        seq: FrameSeq,
        captured_at: MonoTimeNs,
        observation: EyeClosureObservation,
    ) -> EyeClosureState {
        match classify_eye_sample(self.last, (seq, captured_at), false) {
            EyeSampleStep::Reused => return self.state,
            EyeSampleStep::Fresh { reset } => {
                if reset {
                    self.state = EyeClosureState::default();
                }
            }
        }
        self.last = Some((seq, captured_at));
        self.state.left = decide_openness_side(
            self.thresholds.left(),
            self.state.left,
            observation.left_openness,
        );
        self.state.right = decide_openness_side(
            self.thresholds.right(),
            self.state.right,
            observation.right_openness,
        );
        self.state
    }
}

/// Stateful geometry tracker shared by the runtime and the offline fitter.
///
/// The sequence and capture-time rules come from [`classify_eye_sample`]; the
/// judgement comes from [`decide_geometry_pair`].
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryEyeClosureTracker {
    thresholds: EyeGeometryThresholds,
    state: EyeClosureState,
    last: Option<(FrameSeq, MonoTimeNs)>,
}

impl GeometryEyeClosureTracker {
    /// Creates a tracker at the open state.
    #[must_use]
    pub const fn new(thresholds: EyeGeometryThresholds) -> Self {
        Self {
            thresholds,
            state: EyeClosureState {
                left: EyeOpenness::Open,
                right: EyeOpenness::Open,
            },
            last: None,
        }
    }

    /// Returns the configured thresholds.
    #[must_use]
    pub const fn thresholds(&self) -> &EyeGeometryThresholds {
        &self.thresholds
    }

    /// Returns the current latch.
    #[must_use]
    pub const fn state(&self) -> EyeClosureState {
        self.state
    }

    /// Drops the latch and the sample boundary (session/stop/reset).
    pub fn reset(&mut self) {
        self.state = EyeClosureState::default();
        self.last = None;
    }

    /// Advances the latch for one observed sample.
    pub fn observe(
        &mut self,
        seq: FrameSeq,
        captured_at: MonoTimeNs,
        observed: [Option<EyeClosureFeatures>; 2],
    ) -> EyeClosureState {
        match classify_eye_sample(self.last, (seq, captured_at), false) {
            EyeSampleStep::Reused => return self.state,
            EyeSampleStep::Fresh { reset } => {
                if reset {
                    self.state = EyeClosureState::default();
                }
            }
        }
        self.last = Some((seq, captured_at));
        self.state = decide_geometry_pair(
            self.state,
            observed,
            [self.thresholds.left(), self.thresholds.right()],
        );
        self.state
    }
}

fn decide_openness_side(
    threshold: EyeThreshold,
    previous: EyeOpenness,
    openness: Option<f32>,
) -> EyeOpenness {
    match openness {
        Some(value) if value.is_finite() => threshold.decide(previous, value),
        _ => EyeOpenness::Unknown,
    }
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

/// Geometry threshold values as stored in a profile document, before validation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EyeGeometryThresholdValues {
    /// Lid-gap ratio at or below which the eye closes.
    pub close_gap: f32,
    /// Lid-gap ratio at or above which the eye reopens.
    pub reopen_gap: f32,
    /// Auxiliary raw blink score required at the entry edge.
    pub min_blink: f32,
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

/// Serialized eye-closure profile document (v2).
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
    pub left: EyeGeometryThresholdValues,
    /// Right-eye thresholds.
    pub right: EyeGeometryThresholdValues,
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
    pub fn validate(&self) -> Result<EyeGeometryThresholds, EyeClosureProfileError> {
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
        Ok(EyeGeometryThresholds::new(left, right))
    }
}

fn validate_side(
    side: EyeSide,
    values: EyeGeometryThresholdValues,
) -> Result<EyeGeometryThreshold, EyeClosureProfileError> {
    EyeGeometryThreshold::new(values.close_gap, values.reopen_gap, values.min_blink).map_err(
        |source| EyeClosureProfileError::InvalidThreshold {
            side: side.as_str(),
            source,
        },
    )
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

    fn features(lid_gap_ratio: f32, raw_blink: f32) -> EyeClosureFeatures {
        EyeClosureFeatures {
            lid_gap_ratio,
            raw_blink,
        }
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
    fn geometry_threshold_validation_covers_gap_and_blink() {
        assert!(EyeGeometryThreshold::new(f32::NAN, 0.5, 0.0).is_err());
        assert!(EyeGeometryThreshold::new(0.5, 0.5, 0.0).is_err());
        assert!(EyeGeometryThreshold::new(-0.1, 0.5, 0.0).is_err());
        assert!(EyeGeometryThreshold::new(0.2, 0.5, 1.1).is_err());
        assert!(EyeGeometryThreshold::new(0.2, 0.5, -0.1).is_err());
        let threshold = EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap();
        assert_eq!(threshold.close_gap(), 0.2);
        assert_eq!(threshold.reopen_gap(), 0.5);
        assert_eq!(threshold.min_blink(), 0.0);
        assert!((threshold.width() - 0.3).abs() < f32::EPSILON);
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
        assert_eq!(threshold.decide(closed, 0.5), EyeOpenness::Closed);
        assert_eq!(threshold.decide(EyeOpenness::Open, 0.5), EyeOpenness::Open);
    }

    #[test]
    fn unknown_recovers_through_the_entry_condition() {
        let threshold = EyeThreshold::new(0.4, 0.6).unwrap();
        assert_eq!(
            threshold.decide(EyeOpenness::Unknown, 0.5),
            EyeOpenness::Open
        );
        assert_eq!(
            threshold.decide(EyeOpenness::Unknown, 0.3),
            EyeOpenness::Closed
        );
        assert!(!EyeOpenness::Unknown.is_closed());
    }

    #[test]
    fn re_feeding_the_same_sequence_never_advances_the_latch() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.4, 0.6), (0.4, 0.6)));
        let closed = EyeClosureObservation {
            left_openness: Some(0.1),
            right_openness: Some(0.1),
        };
        assert_eq!(
            tracker.observe(FrameSeq(10), MonoTimeNs(330_000_000), closed),
            EyeClosureState {
                left: EyeOpenness::Closed,
                right: EyeOpenness::Closed,
            }
        );
        let stale = EyeClosureObservation {
            left_openness: Some(0.9),
            right_openness: Some(0.9),
        };
        assert_eq!(
            tracker.observe(FrameSeq(10), MonoTimeNs(660_000_000), stale),
            tracker.state()
        );
    }

    #[test]
    fn a_normal_sequence_jump_with_contiguous_time_keeps_the_latch() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.4, 0.6), (0.4, 0.6)));
        tracker.observe(
            FrameSeq(100),
            MonoTimeNs(0),
            EyeClosureObservation {
                left_openness: Some(0.1),
                right_openness: Some(0.1),
            },
        );
        // seq 100 -> 102 with a 33 ms capture step is a dropped render tick,
        // not a tracking discontinuity: a 0.5 observation stays closed.
        let state = tracker.observe(
            FrameSeq(102),
            MonoTimeNs(33_000_000),
            EyeClosureObservation {
                left_openness: Some(0.5),
                right_openness: Some(0.5),
            },
        );
        assert!(state.left.is_closed(), "{state:?}");
    }

    #[test]
    fn a_long_capture_gap_drops_a_carried_closure() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.4, 0.6), (0.4, 0.6)));
        tracker.observe(
            FrameSeq(0),
            MonoTimeNs(0),
            EyeClosureObservation {
                left_openness: Some(0.1),
                right_openness: Some(0.1),
            },
        );
        assert!(tracker.state().left.is_closed());
        // seq 1 is contiguous, but 500 ms of capture time is a break.
        let state = tracker.observe(
            FrameSeq(1),
            MonoTimeNs(500_000_000),
            EyeClosureObservation {
                left_openness: Some(0.9),
                right_openness: Some(0.9),
            },
        );
        assert!(!state.left.is_closed());
    }

    #[test]
    fn classify_distinguishes_reuse_jumps_resets_and_disconnects() {
        let previous = Some((FrameSeq(100), MonoTimeNs(1_000)));
        assert_eq!(
            classify_eye_sample(previous, (FrameSeq(100), MonoTimeNs(2_000)), false),
            EyeSampleStep::Reused
        );
        assert_eq!(
            classify_eye_sample(previous, (FrameSeq(102), MonoTimeNs(33_033_000)), false),
            EyeSampleStep::Fresh { reset: false }
        );
        assert_eq!(
            classify_eye_sample(previous, (FrameSeq(101), MonoTimeNs(500_000_000)), false),
            EyeSampleStep::Fresh { reset: true }
        );
        assert_eq!(
            classify_eye_sample(previous, (FrameSeq(99), MonoTimeNs(33_000_000)), false),
            EyeSampleStep::Fresh { reset: true }
        );
        assert_eq!(
            classify_eye_sample(previous, (FrameSeq(101), MonoTimeNs(500)), false),
            EyeSampleStep::Fresh { reset: true }
        );
        assert_eq!(
            classify_eye_sample(previous, (FrameSeq(101), MonoTimeNs(33_000)), true),
            EyeSampleStep::Fresh { reset: true }
        );
        assert_eq!(
            classify_eye_sample(None, (FrameSeq(0), MonoTimeNs(0)), false),
            EyeSampleStep::Fresh { reset: true }
        );
    }

    #[test]
    fn the_first_closed_observation_closes_immediately() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.4, 0.6), (0.4, 0.6)));
        let state = tracker.observe(
            FrameSeq(1),
            MonoTimeNs(0),
            EyeClosureObservation {
                left_openness: Some(0.0),
                right_openness: Some(0.5),
            },
        );
        assert!(state.left.is_closed());
        assert!(!state.right.is_closed());
    }

    #[test]
    fn a_missing_eye_becomes_unknown_and_is_not_mirrored() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.4, 0.6), (0.4, 0.6)));
        tracker.observe(
            FrameSeq(1),
            MonoTimeNs(0),
            EyeClosureObservation {
                left_openness: Some(0.0),
                right_openness: Some(0.9),
            },
        );
        assert!(tracker.state().left.is_closed());
        let state = tracker.observe(
            FrameSeq(2),
            MonoTimeNs(33_000_000),
            EyeClosureObservation {
                left_openness: None,
                right_openness: Some(0.9),
            },
        );
        assert_eq!(state.left, EyeOpenness::Unknown, "missing left is unknown");
        assert!(!state.left.is_closed());
        // The next valid observation uses the entry condition, not hysteresis.
        let state = tracker.observe(
            FrameSeq(3),
            MonoTimeNs(66_000_000),
            EyeClosureObservation {
                left_openness: Some(0.5),
                right_openness: Some(0.9),
            },
        );
        assert_eq!(state.left, EyeOpenness::Open);
    }

    #[test]
    fn sides_use_independent_thresholds() {
        let mut tracker = EyeClosureTracker::new(thresholds((0.2, 0.3), (0.7, 0.8)));
        let state = tracker.observe(
            FrameSeq(1),
            MonoTimeNs(0),
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
    fn geometry_entry_requires_the_gap_and_blink_conditions() {
        let threshold = EyeGeometryThreshold::new(0.2, 0.5, 0.6).unwrap();
        assert_eq!(
            decide_geometry_eye(EyeOpenness::Open, Some(features(0.1, 0.9)), threshold),
            EyeOpenness::Closed
        );
        // High blink but an open gap must not close: the geometry is required.
        assert_eq!(
            decide_geometry_eye(EyeOpenness::Open, Some(features(0.3, 0.9)), threshold),
            EyeOpenness::Open
        );
        // A closed gap but insufficient auxiliary blink stays open.
        assert_eq!(
            decide_geometry_eye(EyeOpenness::Open, Some(features(0.1, 0.5)), threshold),
            EyeOpenness::Open
        );
        // The equality points are inside the close region.
        assert_eq!(
            decide_geometry_eye(EyeOpenness::Open, Some(features(0.2, 0.6)), threshold),
            EyeOpenness::Closed
        );
        assert_eq!(
            decide_geometry_eye(EyeOpenness::Unknown, Some(features(0.1, 0.6)), threshold),
            EyeOpenness::Closed
        );
    }

    #[test]
    fn geometry_release_uses_only_the_gap_and_unknown_has_no_pin() {
        let threshold = EyeGeometryThreshold::new(0.2, 0.5, 0.6).unwrap();
        assert_eq!(
            decide_geometry_eye(EyeOpenness::Closed, Some(features(0.49, 0.0)), threshold),
            EyeOpenness::Closed
        );
        assert_eq!(
            decide_geometry_eye(EyeOpenness::Closed, Some(features(0.5, 0.0)), threshold),
            EyeOpenness::Open
        );
        assert_eq!(
            decide_geometry_eye(EyeOpenness::Closed, None, threshold),
            EyeOpenness::Unknown
        );
    }

    #[test]
    fn min_blink_zero_ignores_the_raw_blink() {
        let threshold = EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap();
        assert_eq!(
            decide_geometry_eye(EyeOpenness::Open, Some(features(0.2, 0.0)), threshold),
            EyeOpenness::Closed
        );
    }

    #[test]
    fn a_geometry_pair_judges_each_eye_and_never_copies() {
        let left = EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap();
        let right = EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap();
        let state = decide_geometry_pair(
            EyeClosureState::default(),
            [Some(features(0.1, 0.0)), None],
            [left, right],
        );
        assert!(state.left.is_closed());
        assert_eq!(state.right, EyeOpenness::Unknown);
    }

    #[test]
    fn a_geometry_tracker_uses_time_rules_and_reuse() {
        let threshold = EyeGeometryThreshold::new(0.2, 0.5, 0.0).unwrap();
        let thresholds = EyeGeometryThresholds::new(threshold, threshold);
        let mut tracker = GeometryEyeClosureTracker::new(thresholds);
        let closed = [Some(features(0.1, 0.0)), Some(features(0.1, 0.0))];
        let state = tracker.observe(FrameSeq(100), MonoTimeNs(0), closed);
        assert!(state.left.is_closed() && state.right.is_closed());
        // Re-used sample: a stale open observation must not advance the latch.
        let open = [Some(features(0.9, 0.0)), Some(features(0.9, 0.0))];
        assert_eq!(
            tracker.observe(FrameSeq(100), MonoTimeNs(0), open),
            tracker.state()
        );
        // Normal seq jump: the closure carries and 0.9 is above reopen.
        assert_eq!(
            tracker.observe(FrameSeq(102), MonoTimeNs(33_000_000), open),
            EyeClosureState {
                left: EyeOpenness::Open,
                right: EyeOpenness::Open,
            }
        );
        // Long capture gap with missing observation: unknown, not closed.
        tracker.observe(
            FrameSeq(103),
            MonoTimeNs(66_000_000),
            [Some(features(0.1, 0.0)), Some(features(0.1, 0.0))],
        );
        let state = tracker.observe(FrameSeq(104), MonoTimeNs(600_000_000), [None, None]);
        assert_eq!(state.left, EyeOpenness::Unknown);
        assert!(!state.left.is_closed());
    }

    #[test]
    fn document_validation_rejects_wrong_versions_and_features() {
        let document = EyeClosureProfileDocument {
            schema_version: 99,
            algorithm_version: EYE_CLOSURE_ALGORITHM_VERSION,
            feature: EYE_CLOSURE_FEATURE.into(),
            status: EyeClosureVerificationStatus::Verified,
            left: EyeGeometryThresholdValues {
                close_gap: 0.2,
                reopen_gap: 0.5,
                min_blink: 0.0,
            },
            right: EyeGeometryThresholdValues {
                close_gap: 0.2,
                reopen_gap: 0.5,
                min_blink: 0.0,
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
        let mut v1 = document.clone();
        v1.schema_version = EYE_CLOSURE_PROFILE_SCHEMA_VERSION;
        v1.algorithm_version = 1;
        assert!(matches!(
            v1.validate(),
            Err(EyeClosureProfileError::UnsupportedAlgorithmVersion { found: 1 })
        ));
    }

    #[test]
    fn document_validation_reports_the_invalid_side() {
        let document = EyeClosureProfileDocument {
            schema_version: EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
            algorithm_version: EYE_CLOSURE_ALGORITHM_VERSION,
            feature: EYE_CLOSURE_FEATURE.into(),
            status: EyeClosureVerificationStatus::Candidate,
            left: EyeGeometryThresholdValues {
                close_gap: 0.2,
                reopen_gap: 0.5,
                min_blink: 0.0,
            },
            right: EyeGeometryThresholdValues {
                close_gap: 0.7,
                reopen_gap: 0.2,
                min_blink: 0.0,
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
