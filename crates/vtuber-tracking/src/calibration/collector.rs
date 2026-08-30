//! Calibration sample collector.
//!
//! Accepts [`CalibrationInput`] frames in sequence order and retains only
//! neutral candidates that pass confidence, finite-value, motion, and
//! monotonicity checks. The collector caps the number of retained samples at
//! the configured requirement and reports per-frame rejection reasons.
//!
//! The collector deliberately does **not** collect filtered values or treat
//! faceless frames as neutral. Every retained sample has a valid face, valid
//! expression coefficients, and sufficiently low head/expression motion
//! relative to the previously accepted sample.

use vtuber_core::control::CalibrationSettings;
use vtuber_core::types::{FrameSeq, HeadPose, Landmark3, LandmarkSchemaId, MonoTimeNs};

use crate::calibration::CalibrationInput;
use crate::pose::planar::{
    CANONICAL_FACE_TEMPLATE, PlanarCorrespondence, PlanarLandmark, solve_planar_pose,
};
use crate::pose::{LandmarkSet, PoseError, solve_relative_pose};

/// Why a single frame was rejected by the collector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RejectionReason {
    /// The collector already reached its configured sample capacity.
    SessionFull {
        /// Configured capacity.
        capacity: usize,
    },
    /// Face confidence was below the configured minimum.
    LowConfidence {
        /// Observed face confidence.
        face_confidence: f32,
        /// Minimum required confidence.
        min_confidence: f32,
    },
    /// Expression coefficients or confidences were non-finite or outside `[0, 1]`.
    InvalidValues,
    /// Not enough landmarks, or landmark coordinates/visibility were invalid.
    InvalidLandmarks,
    /// Source sequence was not strictly greater than every previously seen
    /// sequence.
    DuplicateOrOldSeq {
        /// Sequence of the offered frame.
        seq: FrameSeq,
        /// Highest sequence seen so far.
        max_seq_seen: FrameSeq,
    },
    /// Source timestamp regressed relative to the previously accepted sample.
    TimestampRegression {
        /// Timestamp of the offered frame.
        captured_at: MonoTimeNs,
        /// Timestamp of the last accepted frame.
        last_captured_at: MonoTimeNs,
    },
    /// Landmark schema changed during the session.
    SchemaMismatch,
    /// Head motion relative to the previous accepted sample was too large.
    TooMuchHeadMotion {
        /// Relative head pose between the previous accepted sample and this one.
        head_pose: HeadPose,
        /// Configured maximum head motion, in radians.
        threshold: f32,
    },
    /// Expression motion relative to the previous accepted sample was too large.
    TooMuchExpressionMotion {
        /// Largest expression coefficient change.
        motion: f32,
        /// Configured maximum expression motion.
        threshold: f32,
    },
    /// Relative pose could not be solved between consecutive accepted samples.
    DegenerateLandmarks,
}

/// Result of offering one frame to [`CalibrationCollector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleDecision {
    /// Frame was accepted and stored.
    Accepted,
    /// Frame was rejected with a stable reason.
    Rejected(RejectionReason),
}

/// Running counters for collector diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectorMetrics {
    /// Number of accepted samples.
    pub accepted: u64,
    /// Rejected because the collector was already full.
    pub rejected_session_full: u64,
    /// Rejected because face confidence was too low.
    pub rejected_low_confidence: u64,
    /// Rejected because expression values were non-finite or out of range.
    pub rejected_invalid_values: u64,
    /// Rejected because landmark data was invalid or insufficient.
    pub rejected_invalid_landmarks: u64,
    /// Rejected because the source sequence was a duplicate or regressed.
    pub rejected_duplicate_or_old_seq: u64,
    /// Rejected because the source timestamp regressed.
    pub rejected_timestamp_regression: u64,
    /// Rejected because the landmark schema changed.
    pub rejected_schema_mismatch: u64,
    /// Rejected because head motion exceeded the threshold.
    pub rejected_head_motion: u64,
    /// Rejected because expression motion exceeded the threshold.
    pub rejected_expression_motion: u64,
    /// Rejected because relative pose solving failed.
    pub rejected_degenerate_landmarks: u64,
}

impl CollectorMetrics {
    /// Total number of frames offered to the collector.
    #[must_use]
    pub fn total_offered(&self) -> u64 {
        self.accepted
            + self.rejected_session_full
            + self.rejected_low_confidence
            + self.rejected_invalid_values
            + self.rejected_invalid_landmarks
            + self.rejected_duplicate_or_old_seq
            + self.rejected_timestamp_regression
            + self.rejected_schema_mismatch
            + self.rejected_head_motion
            + self.rejected_expression_motion
            + self.rejected_degenerate_landmarks
    }

    /// Total number of frames rejected for any reason.
    #[must_use]
    pub fn total_rejected(&self) -> u64 {
        self.total_offered() - self.accepted
    }
}

/// Collects neutral calibration samples.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationCollector {
    settings: CalibrationSettings,
    samples: Vec<CalibrationInput>,
    last_accepted: Option<CalibrationInput>,
    metrics: CollectorMetrics,
    max_seq_seen: FrameSeq,
    schema: Option<LandmarkSchemaId>,
}

impl CalibrationCollector {
    /// Creates a new collector with the given settings.
    #[must_use]
    pub fn new(settings: CalibrationSettings) -> Self {
        let capacity = settings.required_sample_count();
        Self {
            settings,
            samples: Vec::with_capacity(capacity),
            last_accepted: None,
            metrics: CollectorMetrics::default(),
            max_seq_seen: FrameSeq(0),
            schema: None,
        }
    }

    /// Returns the collector settings.
    #[must_use]
    pub fn settings(&self) -> &CalibrationSettings {
        &self.settings
    }

    /// Returns the retained valid samples in acceptance order.
    #[must_use]
    pub fn samples(&self) -> &[CalibrationInput] {
        &self.samples
    }

    /// Returns a mutable reference to the retained samples.
    ///
    /// # Safety
    ///
    /// This is intended for tests that need to bypass the collector's
    /// validation to exercise downstream aggregation.  Mutating the samples
    /// can violate the collector's invariants.
    #[cfg(test)]
    #[must_use]
    pub fn samples_mut(&mut self) -> &mut Vec<CalibrationInput> {
        &mut self.samples
    }

    /// Returns the current diagnostic counters.
    #[must_use]
    pub fn metrics(&self) -> &CollectorMetrics {
        &self.metrics
    }

    /// Returns `true` if the collector has reached its configured capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.samples.len() >= self.settings.required_sample_count()
    }

    /// Returns `true` if enough valid samples have been collected to form a
    /// neutral reference.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.samples.len() >= self.settings.required_sample_count()
    }

    /// Returns the landmark schema observed so far, if any.
    #[must_use]
    pub fn schema_id(&self) -> Option<LandmarkSchemaId> {
        self.schema
    }

    /// Offers a single frame to the collector.
    ///
    /// The frame is validated in a fixed order: monotonicity, capacity,
    /// confidence/finite values, schema consistency, and finally motion
    /// relative to the previously accepted sample. The same input stream
    /// always yields the same sequence of decisions.
    pub fn offer(&mut self, input: CalibrationInput) -> SampleDecision {
        // Sequence must be strictly increasing across the whole stream.
        if input.source_seq <= self.max_seq_seen {
            self.metrics.rejected_duplicate_or_old_seq += 1;
            return SampleDecision::Rejected(RejectionReason::DuplicateOrOldSeq {
                seq: input.source_seq,
                max_seq_seen: self.max_seq_seen,
            });
        }
        self.max_seq_seen = input.source_seq;

        // Timestamp must not regress relative to the last accepted sample.
        if let Some(ref last) = self.last_accepted
            && input.captured_at < last.captured_at
        {
            self.metrics.rejected_timestamp_regression += 1;
            return SampleDecision::Rejected(RejectionReason::TimestampRegression {
                captured_at: input.captured_at,
                last_captured_at: last.captured_at,
            });
        }

        // Capacity guard: do not grow the sample buffer without bound.
        if self.is_full() {
            self.metrics.rejected_session_full += 1;
            return SampleDecision::Rejected(RejectionReason::SessionFull {
                capacity: self.settings.required_sample_count(),
            });
        }

        // Confidence must be finite, in range, and above threshold.
        if !input.face_confidence.is_finite()
            || input.face_confidence < 0.0
            || input.face_confidence > 1.0
        {
            self.metrics.rejected_invalid_values += 1;
            return SampleDecision::Rejected(RejectionReason::InvalidValues);
        }
        if input.face_confidence < self.settings.min_confidence() {
            self.metrics.rejected_low_confidence += 1;
            return SampleDecision::Rejected(RejectionReason::LowConfidence {
                face_confidence: input.face_confidence,
                min_confidence: self.settings.min_confidence(),
            });
        }

        // Expression coefficients and confidences must be valid.
        if !input.expressions.is_valid() {
            self.metrics.rejected_invalid_values += 1;
            return SampleDecision::Rejected(RejectionReason::InvalidValues);
        }

        // Landmark data must be finite and have enough points for pose solving.
        if !validate_landmarks(&input.landmarks) {
            self.metrics.rejected_invalid_landmarks += 1;
            return SampleDecision::Rejected(RejectionReason::InvalidLandmarks);
        }

        // Schema must remain consistent for the whole session.
        match self.schema {
            Some(schema) if schema != input.schema => {
                self.metrics.rejected_schema_mismatch += 1;
                return SampleDecision::Rejected(RejectionReason::SchemaMismatch);
            }
            None => self.schema = Some(input.schema),
            _ => {}
        }

        // Motion checks require a previous accepted sample.
        if let Some(ref last) = self.last_accepted {
            match relative_head_pose(last, &input) {
                Ok(pose) => {
                    let head_motion = max_euler_component(pose);
                    if head_motion > self.settings.max_head_motion_rad() {
                        self.metrics.rejected_head_motion += 1;
                        return SampleDecision::Rejected(RejectionReason::TooMuchHeadMotion {
                            head_pose: pose,
                            threshold: self.settings.max_head_motion_rad(),
                        });
                    }
                }
                Err(_) => {
                    self.metrics.rejected_degenerate_landmarks += 1;
                    return SampleDecision::Rejected(RejectionReason::DegenerateLandmarks);
                }
            }

            let expression_motion = max_expression_delta(last, &input);
            if expression_motion > self.settings.max_expression_motion() {
                self.metrics.rejected_expression_motion += 1;
                return SampleDecision::Rejected(RejectionReason::TooMuchExpressionMotion {
                    motion: expression_motion,
                    threshold: self.settings.max_expression_motion(),
                });
            }
        }

        // Accept the sample.
        self.last_accepted = Some(input.clone());
        self.samples.push(input);
        self.metrics.accepted += 1;
        SampleDecision::Accepted
    }

    /// Resets the collector to an empty state, keeping the same settings.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.last_accepted = None;
        self.metrics = CollectorMetrics::default();
        self.max_seq_seen = FrameSeq(0);
        self.schema = None;
    }
}

fn validate_landmarks(landmarks: &[Landmark3]) -> bool {
    if landmarks.len() < crate::pose::MIN_LANDMARK_POINTS {
        return false;
    }
    landmarks.iter().all(|lm| {
        lm.x.is_finite()
            && lm.y.is_finite()
            && lm.z.is_finite()
            && lm.visibility.is_finite()
            && (0.0..=1.0).contains(&lm.visibility)
    })
}

fn relative_head_pose(
    previous: &CalibrationInput,
    current: &CalibrationInput,
) -> Result<HeadPose, PoseError> {
    if previous.schema.0 == "peppapig-98" && current.schema == previous.schema {
        let previous_pose = planar_absolute_pose(&previous.landmarks)
            .map_err(|_| PoseError::DegeneratePointCloud)?;
        let current_pose = planar_absolute_pose(&current.landmarks)
            .map_err(|_| PoseError::DegeneratePointCloud)?;
        return Ok(HeadPose {
            yaw_rad: current_pose.yaw_rad - previous_pose.yaw_rad,
            pitch_rad: current_pose.pitch_rad - previous_pose.pitch_rad,
            roll_rad: current_pose.roll_rad - previous_pose.roll_rad,
        });
    }
    let neutral = landmarks_to_set(&previous.landmarks);
    let current = landmarks_to_set(&current.landmarks);
    let alignment = solve_relative_pose(&neutral, &current)?;
    Ok(alignment.pose)
}

fn planar_absolute_pose(landmarks: &[Landmark3]) -> Result<HeadPose, ()> {
    let points = CANONICAL_FACE_TEMPLATE
        .iter()
        .map(|canonical| {
            let landmark = landmarks.get(canonical.index).ok_or(())?;
            Ok(PlanarCorrespondence {
                canonical: *canonical,
                reference: PlanarLandmark {
                    x: landmark.x,
                    y: landmark.y,
                    confidence: landmark.visibility,
                },
                current: PlanarLandmark {
                    x: landmark.x,
                    y: landmark.y,
                    confidence: landmark.visibility,
                },
            })
        })
        .collect::<Result<Vec<_>, ()>>()?;
    Ok(solve_planar_pose(&points).map_err(|_| ())?.pose)
}

fn landmarks_to_set(landmarks: &[Landmark3]) -> LandmarkSet {
    let mut set = LandmarkSet::new();
    for lm in landmarks {
        set.push([lm.x, lm.y, lm.z], lm.visibility);
    }
    set
}

fn max_euler_component(pose: HeadPose) -> f32 {
    pose.yaw_rad
        .abs()
        .max(pose.pitch_rad.abs())
        .max(pose.roll_rad.abs())
}

fn max_expression_delta(previous: &CalibrationInput, current: &CalibrationInput) -> f32 {
    let d_left = (previous.expressions.blink_left - current.expressions.blink_left).abs();
    let d_right = (previous.expressions.blink_right - current.expressions.blink_right).abs();
    let d_mouth = (previous.expressions.mouth_open - current.expressions.mouth_open).abs();
    d_left.max(d_right).max(d_mouth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::types::{Landmark3, RawExpressionObservation};

    fn settings() -> CalibrationSettings {
        CalibrationSettings::try_new(5, 5.0, 0.5, 5.0f32.to_radians(), 0.15).unwrap()
    }

    fn neutral_landmarks() -> Vec<Landmark3> {
        vec![
            Landmark3 {
                x: -1.0,
                y: 0.0,
                z: 0.05,
                visibility: 1.0,
            },
            Landmark3 {
                x: 1.0,
                y: 0.0,
                z: 0.05,
                visibility: 1.0,
            },
            Landmark3 {
                x: 0.0,
                y: 0.8,
                z: 0.0,
                visibility: 1.0,
            },
            Landmark3 {
                x: 0.0,
                y: -0.6,
                z: 0.1,
                visibility: 1.0,
            },
        ]
    }

    fn input(seq: u64, confidence: f32) -> CalibrationInput {
        CalibrationInput {
            source_seq: FrameSeq(seq),
            captured_at: MonoTimeNs(seq * 33_333_333),
            face_confidence: confidence,
            landmarks: neutral_landmarks(),
            expressions: RawExpressionObservation {
                blink_left: 0.1,
                blink_left_confidence: 0.9,
                blink_right: 0.1,
                blink_right_confidence: 0.9,
                mouth_open: 0.05,
                mouth_open_confidence: 0.9,
            },
            schema: LandmarkSchemaId("test"),
        }
    }

    #[test]
    fn accepts_valid_frames_until_full() {
        let mut collector = CalibrationCollector::new(settings());
        for seq in 1..=5 {
            let decision = collector.offer(input(seq, 0.9));
            assert_eq!(decision, SampleDecision::Accepted);
        }
        assert!(collector.is_full());
        assert!(collector.is_ready());
        assert_eq!(collector.samples().len(), 5);
    }

    #[test]
    fn rejects_low_confidence() {
        let mut collector = CalibrationCollector::new(settings());
        let decision = collector.offer(input(1, 0.1));
        assert!(
            matches!(
                decision,
                SampleDecision::Rejected(RejectionReason::LowConfidence { .. })
            ),
            "expected LowConfidence, got {decision:?}"
        );
        assert_eq!(collector.metrics().rejected_low_confidence, 1);
    }

    #[test]
    fn rejects_invalid_expression_values() {
        let mut collector = CalibrationCollector::new(settings());
        let mut bad = input(1, 0.9);
        bad.expressions.blink_left = f32::NAN;
        let decision = collector.offer(bad);
        assert_eq!(
            decision,
            SampleDecision::Rejected(RejectionReason::InvalidValues)
        );
    }

    #[test]
    fn rejects_duplicate_sequence() {
        let mut collector = CalibrationCollector::new(settings());
        assert_eq!(collector.offer(input(1, 0.9)), SampleDecision::Accepted);
        let decision = collector.offer(input(1, 0.9));
        assert!(
            matches!(
                decision,
                SampleDecision::Rejected(RejectionReason::DuplicateOrOldSeq { .. })
            ),
            "expected DuplicateOrOldSeq, got {decision:?}"
        );
    }

    #[test]
    fn rejects_timestamp_regression() {
        let mut collector = CalibrationCollector::new(settings());
        let mut first = input(1, 0.9);
        first.captured_at = MonoTimeNs(200);
        assert_eq!(collector.offer(first), SampleDecision::Accepted);

        let mut second = input(2, 0.9);
        second.captured_at = MonoTimeNs(100);
        let decision = collector.offer(second);
        assert!(
            matches!(
                decision,
                SampleDecision::Rejected(RejectionReason::TimestampRegression { .. })
            ),
            "expected TimestampRegression, got {decision:?}"
        );
    }

    #[test]
    fn rejects_excessive_head_motion() {
        let mut collector = CalibrationCollector::new(settings());
        let mut moving = input(2, 0.9);
        // Rotate the second set of landmarks around Y by 20 degrees.
        moving.landmarks = neutral_landmarks()
            .iter()
            .map(|lm| {
                let angle = 20.0f32.to_radians();
                Landmark3 {
                    x: lm.x * angle.cos() + lm.z * angle.sin(),
                    y: lm.y,
                    z: -lm.x * angle.sin() + lm.z * angle.cos(),
                    visibility: lm.visibility,
                }
            })
            .collect();

        assert_eq!(collector.offer(input(1, 0.9)), SampleDecision::Accepted);
        let decision = collector.offer(moving);
        assert!(
            matches!(
                decision,
                SampleDecision::Rejected(RejectionReason::TooMuchHeadMotion { .. })
            ),
            "expected TooMuchHeadMotion, got {decision:?}"
        );
    }

    #[test]
    fn rejects_excessive_expression_motion() {
        let mut collector = CalibrationCollector::new(settings());
        assert_eq!(collector.offer(input(1, 0.9)), SampleDecision::Accepted);

        let mut talking = input(2, 0.9);
        talking.expressions.mouth_open = 0.9;
        let decision = collector.offer(talking);
        assert!(
            matches!(
                decision,
                SampleDecision::Rejected(RejectionReason::TooMuchExpressionMotion { .. })
            ),
            "expected TooMuchExpressionMotion, got {decision:?}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut collector = CalibrationCollector::new(settings());
        collector.offer(input(1, 0.9));
        collector.reset();
        assert!(collector.samples().is_empty());
        assert_eq!(collector.metrics().accepted, 0);
        assert_eq!(collector.offer(input(1, 0.9)), SampleDecision::Accepted);
    }
}
