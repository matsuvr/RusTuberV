//! Neutral reference aggregation and validation.
//!
//! Turns a full [`CalibrationCollector`] into a validated [`NeutralProfile`].
//! Landmarks are aggregated with a component-wise median, expression raw
//! values are summarized by median baselines, and the resulting point cloud
//! is checked for degeneracy and stability.  The module does not perform any
//! persistence I/O.

use vtuber_core::control::CalibrationError;
use vtuber_core::types::{HeadPose, Landmark3, MonoTimeNs, RawExpressionObservation};

use crate::NeutralProfile;

use crate::calibration::CalibrationInput;
use crate::calibration::collector::CalibrationCollector;
use crate::pose::{LandmarkSet, PoseError, solve_relative_pose};

/// Minimum number of points required by the Kabsch solver.
pub use crate::pose::MIN_LANDMARK_POINTS;

/// Threshold below which the covariance matrix of a point cloud is treated
/// as degenerate.  This mirrors the constant in `crate::pose` so validation
/// stays consistent with pose solving.
const DEGENERACY_THRESHOLD: f32 = 1e-6;

/// Settings that control neutral-reference quality thresholds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeutralValidationSettings {
    /// Maximum allowed head pose spread across accepted samples, in radians.
    pub max_head_pose_spread_rad: f32,
    /// Maximum allowed landmark position spread, expressed as a ratio of
    /// the face scale estimate.
    pub max_landmark_spread_ratio: f32,
}

impl NeutralValidationSettings {
    /// Creates settings from raw thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError::InvalidMotionThreshold`] if either
    /// threshold is not a positive finite number.
    pub fn try_new(
        max_head_pose_spread_rad: f32,
        max_landmark_spread_ratio: f32,
    ) -> Result<Self, CalibrationError> {
        if !max_head_pose_spread_rad.is_finite() || max_head_pose_spread_rad <= 0.0 {
            return Err(CalibrationError::InvalidMotionThreshold(
                max_head_pose_spread_rad,
            ));
        }
        if !max_landmark_spread_ratio.is_finite() || max_landmark_spread_ratio <= 0.0 {
            return Err(CalibrationError::InvalidMotionThreshold(
                max_landmark_spread_ratio,
            ));
        }
        Ok(Self {
            max_head_pose_spread_rad,
            max_landmark_spread_ratio,
        })
    }
}

impl Default for NeutralValidationSettings {
    /// Defaults chosen for a 30 FPS webcam calibration:
    ///
    /// * head pose spread: 5 degrees — the user can hold this still for one
    ///   second without discomfort.
    /// * landmark spread: 5% of face scale — allows minor expression drift
    ///   while rejecting talking or large motion.
    fn default() -> Self {
        Self {
            max_head_pose_spread_rad: 5.0f32.to_radians(),
            max_landmark_spread_ratio: 0.05,
        }
    }
}

/// Context carried alongside the collected samples into the neutral profile.
#[derive(Clone, Debug, PartialEq)]
pub struct NeutralContext {
    /// When the profile was finalized.
    pub now: MonoTimeNs,
    /// Hash of the VRM model the profile was collected for, if known.
    pub model_hash: Option<String>,
    /// Fingerprint of the camera used during collection, if known.
    pub camera_fingerprint: Option<String>,
}

impl NeutralContext {
    /// Creates a new context with the given metadata.
    #[must_use]
    pub fn new(
        now: MonoTimeNs,
        model_hash: Option<String>,
        camera_fingerprint: Option<String>,
    ) -> Self {
        Self {
            now,
            model_hash,
            camera_fingerprint,
        }
    }
}

impl Default for NeutralContext {
    fn default() -> Self {
        Self {
            now: MonoTimeNs(0),
            model_hash: None,
            camera_fingerprint: None,
        }
    }
}

/// Aggregates a validated neutral reference from a full collector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeutralReference;

impl NeutralReference {
    /// Aggregates the samples in `collector` into a [`NeutralProfile`].
    ///
    /// The algorithm:
    ///
    /// 1. Verify the collector reached its required sample count.
    /// 2. Compute component-wise median landmarks.
    /// 3. Verify the median point cloud is non-degenerate.
    /// 4. Compute median expression baselines.
    /// 5. Estimate face scale from the median landmarks.
    /// 6. Compute head-pose spread of every sample relative to the median.
    /// 7. Compute normalized landmark position spread.
    /// 8. Reject if either spread exceeds the configured threshold.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError`] when the collected data is insufficient
    /// or unstable.  Degenerate point clouds are reported as
    /// [`CalibrationError::DegeneratePointCloud`], mirroring the G0-06 pose
    /// solver failure mode.
    pub fn aggregate(
        collector: &CalibrationCollector,
        settings: &NeutralValidationSettings,
        context: &NeutralContext,
    ) -> Result<NeutralProfile, CalibrationError> {
        let samples = collector.samples();
        if samples.len() < collector.settings().required_sample_count() {
            return Err(CalibrationError::InsufficientSamples(samples.len()));
        }

        let schema = collector
            .schema_id()
            .ok_or(CalibrationError::InsufficientSamples(samples.len()))?;

        // All accepted samples must share the same landmark count because
        // they share the same schema.
        let landmark_count = samples.first().map_or(0, |s| s.landmarks.len());
        if landmark_count < MIN_LANDMARK_POINTS {
            return Err(CalibrationError::InsufficientLandmarks(landmark_count));
        }
        if samples.iter().any(|s| s.landmarks.len() != landmark_count) {
            return Err(CalibrationError::InsufficientLandmarks(landmark_count));
        }

        let landmarks = median_landmarks(samples, landmark_count);
        if landmarks.len() < MIN_LANDMARK_POINTS {
            return Err(CalibrationError::InsufficientLandmarks(landmarks.len()));
        }

        let median_set = landmarks_to_set(&landmarks);
        let is_planar = schema.0 == "peppapig-98";
        if !is_planar && is_degenerate(&median_set) {
            return Err(CalibrationError::DegeneratePointCloud);
        }

        let expressions = median_expressions(samples);
        let face_scale = estimate_face_scale(&landmarks);
        if !face_scale.is_finite() || face_scale <= 0.0 {
            return Err(CalibrationError::DegeneratePointCloud);
        }

        let head_spread = if is_planar {
            HeadPose::default()
        } else {
            head_pose_spread(samples, &median_set)?
        };
        let max_head_component = head_spread
            .yaw_rad
            .max(head_spread.pitch_rad)
            .max(head_spread.roll_rad);
        if max_head_component > settings.max_head_pose_spread_rad {
            return Err(CalibrationError::PoseSpreadTooLarge {
                yaw: head_spread.yaw_rad,
                pitch: head_spread.pitch_rad,
                roll: head_spread.roll_rad,
                max: settings.max_head_pose_spread_rad,
            });
        }

        let landmark_spread = normalized_landmark_spread(samples, &landmarks, face_scale);
        if landmark_spread > settings.max_landmark_spread_ratio {
            return Err(CalibrationError::LandmarkSpreadTooLarge {
                spread: landmark_spread,
                max: settings.max_landmark_spread_ratio,
            });
        }

        let head_pose = if is_planar {
            HeadPose::default()
        } else {
            median_head_pose(samples, &median_set)?
        };
        let confidence_baseline = median_confidence(samples);

        Ok(NeutralProfile {
            version: 2,
            schema,
            landmarks,
            head_pose,
            gaze_baseline: crate::calibration::GazeNeutralBaseline::default(),
            blink_left_baseline: expressions.blink_left,
            blink_right_baseline: expressions.blink_right,
            mouth_open_baseline: expressions.mouth_open,
            face_scale,
            confidence_baseline,
            collected_at: context.now,
            model_hash: context.model_hash.clone(),
            camera_fingerprint: context.camera_fingerprint.clone(),
        })
    }
}

/// Component-wise median of all landmarks at each index.
fn median_landmarks(samples: &[CalibrationInput], count: usize) -> Vec<Landmark3> {
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let mut xs: Vec<f32> = Vec::with_capacity(samples.len());
        let mut ys: Vec<f32> = Vec::with_capacity(samples.len());
        let mut zs: Vec<f32> = Vec::with_capacity(samples.len());
        let mut vs: Vec<f32> = Vec::with_capacity(samples.len());
        for s in samples {
            if let Some(lm) = s.landmarks.get(i) {
                xs.push(lm.x);
                ys.push(lm.y);
                zs.push(lm.z);
                vs.push(lm.visibility);
            }
        }
        result.push(Landmark3 {
            x: median_sorted(&mut xs),
            y: median_sorted(&mut ys),
            z: median_sorted(&mut zs),
            visibility: median_sorted(&mut vs),
        });
    }
    result
}

/// Median of each expression coefficient and confidence.
fn median_expressions(samples: &[CalibrationInput]) -> RawExpressionObservation {
    let mut left: Vec<f32> = Vec::with_capacity(samples.len());
    let mut left_conf: Vec<f32> = Vec::with_capacity(samples.len());
    let mut right: Vec<f32> = Vec::with_capacity(samples.len());
    let mut right_conf: Vec<f32> = Vec::with_capacity(samples.len());
    let mut mouth: Vec<f32> = Vec::with_capacity(samples.len());
    let mut mouth_conf: Vec<f32> = Vec::with_capacity(samples.len());

    for s in samples {
        left.push(s.expressions.blink_left);
        left_conf.push(s.expressions.blink_left_confidence);
        right.push(s.expressions.blink_right);
        right_conf.push(s.expressions.blink_right_confidence);
        mouth.push(s.expressions.mouth_open);
        mouth_conf.push(s.expressions.mouth_open_confidence);
    }

    RawExpressionObservation {
        blink_left: median_sorted(&mut left),
        blink_left_confidence: median_sorted(&mut left_conf),
        blink_right: median_sorted(&mut right),
        blink_right_confidence: median_sorted(&mut right_conf),
        mouth_open: median_sorted(&mut mouth),
        mouth_open_confidence: median_sorted(&mut mouth_conf),
    }
}

/// Median face confidence.
fn median_confidence(samples: &[CalibrationInput]) -> f32 {
    let mut values: Vec<f32> = samples.iter().map(|s| s.face_confidence).collect();
    median_sorted(&mut values)
}

/// Head pose spread of every sample relative to the median landmarks.
fn head_pose_spread(
    samples: &[CalibrationInput],
    median_set: &LandmarkSet,
) -> Result<HeadPose, CalibrationError> {
    let mut yaw_max = 0.0f32;
    let mut pitch_max = 0.0f32;
    let mut roll_max = 0.0f32;
    for s in samples {
        let current = landmarks_to_set(&s.landmarks);
        let alignment = solve_relative_pose(median_set, &current)
            .map_err(map_pose_error_to_calibration_error)?;
        yaw_max = yaw_max.max(alignment.pose.yaw_rad.abs());
        pitch_max = pitch_max.max(alignment.pose.pitch_rad.abs());
        roll_max = roll_max.max(alignment.pose.roll_rad.abs());
    }
    Ok(HeadPose {
        yaw_rad: yaw_max,
        pitch_rad: pitch_max,
        roll_rad: roll_max,
    })
}

/// Median head pose of samples relative to the median landmarks.
fn median_head_pose(
    samples: &[CalibrationInput],
    median_set: &LandmarkSet,
) -> Result<HeadPose, CalibrationError> {
    let mut yaws: Vec<f32> = Vec::with_capacity(samples.len());
    let mut pitches: Vec<f32> = Vec::with_capacity(samples.len());
    let mut rolls: Vec<f32> = Vec::with_capacity(samples.len());
    for s in samples {
        let current = landmarks_to_set(&s.landmarks);
        let alignment = solve_relative_pose(median_set, &current)
            .map_err(map_pose_error_to_calibration_error)?;
        yaws.push(alignment.pose.yaw_rad);
        pitches.push(alignment.pose.pitch_rad);
        rolls.push(alignment.pose.roll_rad);
    }
    Ok(HeadPose {
        yaw_rad: median_sorted(&mut yaws),
        pitch_rad: median_sorted(&mut pitches),
        roll_rad: median_sorted(&mut rolls),
    })
}

/// Largest normalized RMS distance from any sample to the median landmarks.
fn normalized_landmark_spread(
    samples: &[CalibrationInput],
    median: &[Landmark3],
    face_scale: f32,
) -> f32 {
    let mut max_spread = 0.0f32;
    for s in samples {
        let mut sum_sq = 0.0f32;
        let mut count = 0usize;
        for (i, lm) in s.landmarks.iter().enumerate() {
            if let Some(m) = median.get(i) {
                let dx = lm.x - m.x;
                let dy = lm.y - m.y;
                let dz = lm.z - m.z;
                sum_sq += dx * dx + dy * dy + dz * dz;
                count += 1;
            }
        }
        if count > 0 {
            let rms = (sum_sq / count as f32).sqrt();
            max_spread = max_spread.max(rms / face_scale);
        }
    }
    max_spread
}

/// Estimate face scale as the average pairwise distance among landmarks.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn estimate_face_scale(landmarks: &[Landmark3]) -> f32 {
    let n = landmarks.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = landmarks[i];
            let b = landmarks[j];
            let dx = f64::from(a.x - b.x);
            let dy = f64::from(a.y - b.y);
            let dz = f64::from(a.z - b.z);
            sum += (dx * dx + dy * dy + dz * dz).sqrt();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Converts a collected landmark vector into the Kabsch input set.
fn landmarks_to_set(landmarks: &[Landmark3]) -> LandmarkSet {
    let mut set = LandmarkSet::new();
    for lm in landmarks {
        set.push([lm.x, lm.y, lm.z], lm.visibility);
    }
    set
}

/// Returns `true` if the point cloud lacks enough volume for pose solving.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn is_degenerate(set: &LandmarkSet) -> bool {
    let n = set.len();
    if n < MIN_LANDMARK_POINTS {
        return true;
    }

    // Build the 3 x N centered matrix and compute covariance exactly as
    // `solve_relative_pose` does, so validation matches the solver.
    let centroid = match set.centroid() {
        Some(c) => c,
        None => return true,
    };
    let mut m = nalgebra::OMatrix::<f32, nalgebra::U3, nalgebra::Dyn>::zeros(n);
    for (i, p) in set.points.iter().enumerate() {
        for j in 0..3 {
            m[(j, i)] = (p.position[j] - centroid[j]) * p.weight.sqrt();
        }
    }
    let cov = &m * m.transpose();
    let svd = nalgebra::SVD::new(cov, true, true);
    // `total_cmp` keeps the comparator panic-free even if a singular value
    // were NaN.
    match svd.singular_values.iter().min_by(|a, b| a.total_cmp(b)) {
        Some(&min) => min < DEGENERACY_THRESHOLD,
        None => true,
    }
}

/// Computes the median of a slice by sorting it in place.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn median_sorted(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    // `total_cmp` is a total order: it never panics on NaN input.
    values.sort_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// Maps G0-06 pose errors to calibration validation errors.
fn map_pose_error_to_calibration_error(err: PoseError) -> CalibrationError {
    match err {
        PoseError::InsufficientPoints(n) => CalibrationError::InsufficientLandmarks(n),
        PoseError::DegeneratePointCloud | PoseError::ZeroWeight(_) => {
            CalibrationError::DegeneratePointCloud
        }
        PoseError::ReflectionDetected => CalibrationError::DegeneratePointCloud,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SampleDecision;
    use vtuber_core::types::{FrameSeq, LandmarkSchemaId};

    fn settings() -> CalibrationSettings {
        CalibrationSettings::try_new(5, 5.0, 0.5, 5.0f32.to_radians(), 0.15).unwrap()
    }

    fn face_landmarks(offset: [f32; 3]) -> Vec<Landmark3> {
        // A non-degenerate face-like shape with enough points for Kabsch.
        // The points are not symmetric so that 180-degree degeneracies are
        // also avoided.
        vec![
            Landmark3 {
                x: -1.0 + offset[0],
                y: 0.1 + offset[1],
                z: 0.05 + offset[2],
                visibility: 1.0,
            },
            Landmark3 {
                x: 1.0 + offset[0],
                y: -0.1 + offset[1],
                z: 0.05 + offset[2],
                visibility: 1.0,
            },
            Landmark3 {
                x: 0.1 + offset[0],
                y: 0.8 + offset[1],
                z: -0.05 + offset[2],
                visibility: 1.0,
            },
            Landmark3 {
                x: -0.1 + offset[0],
                y: -0.6 + offset[1],
                z: 0.15 + offset[2],
                visibility: 1.0,
            },
            Landmark3 {
                x: 0.0 + offset[0],
                y: 0.0 + offset[1],
                z: 0.3 + offset[2],
                visibility: 1.0,
            },
        ]
    }

    fn input(seq: u64, offset: [f32; 3]) -> CalibrationInput {
        CalibrationInput {
            source_seq: FrameSeq(seq),
            captured_at: MonoTimeNs(seq * 33_333_333),
            face_confidence: 0.9,
            landmarks: face_landmarks(offset),
            expressions: RawExpressionObservation {
                blink_left: 0.1,
                blink_left_confidence: 0.9,
                blink_right: 0.1,
                blink_right_confidence: 0.9,
                mouth_open: 0.05,
                mouth_open_confidence: 0.9,
            },
            schema: LandmarkSchemaId("neutral-test"),
        }
    }

    fn filled_collector() -> CalibrationCollector {
        let mut collector = CalibrationCollector::new(settings());
        for seq in 1..=5 {
            assert_eq!(
                collector.offer(input(seq, [0.0; 3])),
                SampleDecision::Accepted
            );
        }
        collector
    }

    use vtuber_core::control::CalibrationSettings;

    #[test]
    fn neutral_reference_aggregates_median_landmarks_and_expressions() {
        let collector = filled_collector();
        let profile = NeutralReference::aggregate(
            &collector,
            &NeutralValidationSettings::default(),
            &NeutralContext {
                now: MonoTimeNs(1_000_000_000),
                model_hash: Some("model-a".into()),
                camera_fingerprint: Some("camera-1".into()),
            },
        )
        .unwrap();

        assert_eq!(profile.schema, LandmarkSchemaId("neutral-test"));
        assert_eq!(profile.landmarks.len(), 5);
        assert!((profile.blink_left_baseline - 0.1).abs() < 1e-6);
        assert!((profile.blink_right_baseline - 0.1).abs() < 1e-6);
        assert!((profile.mouth_open_baseline - 0.05).abs() < 1e-6);
        assert!((profile.confidence_baseline - 0.9).abs() < 1e-6);
        assert!((profile.face_scale - estimate_face_scale(&face_landmarks([0.0; 3]))).abs() < 1e-3);
        assert_eq!(profile.model_hash, Some("model-a".into()));
        assert_eq!(profile.camera_fingerprint, Some("camera-1".into()));
    }

    #[test]
    fn neutral_reference_rejects_insufficient_samples() {
        let mut collector = CalibrationCollector::new(settings());
        collector.offer(input(1, [0.0; 3]));
        let err = NeutralReference::aggregate(
            &collector,
            &NeutralValidationSettings::default(),
            &NeutralContext::default(),
        )
        .unwrap_err();
        assert_eq!(err.code(), "CALIBRATION_INSUFFICIENT_SAMPLES");
    }

    #[test]
    fn neutral_reference_rejects_degenerate_point_cloud() {
        // Build the collector by hand so the degenerate samples bypass the
        // collector's motion check.  We only need to test the aggregator's
        // degeneracy detection.
        let mut collector = CalibrationCollector::new(settings());
        for seq in 1..=5 {
            let mut sample = input(seq, [0.0; 3]);
            // Make all landmarks collinear on the X axis.
            sample.landmarks = (0..5)
                .map(|i| Landmark3 {
                    x: i as f32 * 0.1,
                    y: 0.0,
                    z: 0.0,
                    visibility: 1.0,
                })
                .collect();
            // The first sample has no previous frame, so it is always accepted.
            // Force-accept remaining samples by relaxing the collector after the
            // first sample.  This is a test-only backdoor to exercise validation.
            if seq == 1 {
                assert_eq!(collector.offer(sample), SampleDecision::Accepted);
            } else {
                collector.samples_mut().push(sample);
            }
        }
        assert!(collector.is_ready());
        let err = NeutralReference::aggregate(
            &collector,
            &NeutralValidationSettings::default(),
            &NeutralContext::default(),
        )
        .unwrap_err();
        assert_eq!(err.code(), "CALIBRATION_DEGENERATE_POINT_CLOUD");
    }

    #[test]
    fn neutral_reference_rejects_large_head_pose_spread() {
        // Tight settings so a modest rotation exceeds the threshold.
        let tight = NeutralValidationSettings::try_new(0.01, 1.0).unwrap();
        // Relax the collector's own head-motion threshold so the rotated
        // sample is accepted; we want to test the neutral-reference
        // validation, not the collector.
        let loose_settings =
            CalibrationSettings::try_new(5, 5.0, 0.5, 30.0f32.to_radians(), 0.15).unwrap();
        let mut collector = CalibrationCollector::new(loose_settings);
        for seq in 1..=5 {
            let mut sample = input(seq, [0.0; 3]);
            if seq == 5 {
                // Rotate the last sample 10 degrees around Y.
                sample.landmarks = sample
                    .landmarks
                    .iter()
                    .map(|lm| {
                        let angle = 10.0f32.to_radians();
                        Landmark3 {
                            x: lm.x * angle.cos() + lm.z * angle.sin(),
                            y: lm.y,
                            z: -lm.x * angle.sin() + lm.z * angle.cos(),
                            visibility: lm.visibility,
                        }
                    })
                    .collect();
            }
            assert_eq!(collector.offer(sample), SampleDecision::Accepted);
        }
        let err = NeutralReference::aggregate(&collector, &tight, &NeutralContext::default())
            .unwrap_err();
        assert_eq!(err.code(), "CALIBRATION_POSE_SPREAD_TOO_LARGE");
    }

    #[test]
    fn neutral_reference_rejects_large_landmark_spread() {
        let mut collector = CalibrationCollector::new(settings());
        let tight = NeutralValidationSettings::try_new(1.0, 0.001).unwrap();
        for seq in 1..=5 {
            let offset = if seq == 5 { [0.5, 0.0, 0.0] } else { [0.0; 3] };
            assert_eq!(
                collector.offer(input(seq, offset)),
                SampleDecision::Accepted
            );
        }
        let err = NeutralReference::aggregate(&collector, &tight, &NeutralContext::default())
            .unwrap_err();
        assert_eq!(err.code(), "CALIBRATION_LANDMARK_SPREAD_TOO_LARGE");
    }

    #[test]
    fn neutral_reference_robust_to_single_outlier() {
        // Relax the collector's expression-motion threshold so the outlier
        // sample is retained; we are testing the aggregator's median, not
        // the collector's rejection logic.
        let loose_settings =
            CalibrationSettings::try_new(5, 5.0, 0.5, 5.0f32.to_radians(), 1.0).unwrap();
        let mut collector = CalibrationCollector::new(loose_settings);
        for seq in 1..=5 {
            let mut sample = input(seq, [0.0; 3]);
            if seq == 3 {
                // One outlier sample with a large expression spike.
                sample.expressions.mouth_open = 0.9;
            }
            assert_eq!(collector.offer(sample), SampleDecision::Accepted);
        }

        let profile = NeutralReference::aggregate(
            &collector,
            &NeutralValidationSettings::default(),
            &NeutralContext::default(),
        )
        .unwrap();

        // Median mouth baseline should ignore the single outlier.
        assert!((profile.mouth_open_baseline - 0.05).abs() < 1e-6);
    }

    #[test]
    fn neutral_reference_model_hash_mismatch_cannot_reuse() {
        let profile = NeutralReference::aggregate(
            &filled_collector(),
            &NeutralValidationSettings::default(),
            &NeutralContext {
                now: MonoTimeNs(0),
                model_hash: Some("model-a".into()),
                camera_fingerprint: None,
            },
        )
        .unwrap();

        assert!(profile.is_compatible_with(Some("model-a")));
        assert!(!profile.is_compatible_with(Some("model-b")));
    }

    #[test]
    fn neutral_reference_preserves_schema_version_and_context() {
        let profile = NeutralReference::aggregate(
            &filled_collector(),
            &NeutralValidationSettings::default(),
            &NeutralContext {
                now: MonoTimeNs(42),
                model_hash: Some("hash".into()),
                camera_fingerprint: Some("fingerprint".into()),
            },
        )
        .unwrap();

        assert_eq!(profile.version, 2);
        assert_eq!(profile.schema, LandmarkSchemaId("neutral-test"));
        assert_eq!(profile.collected_at, MonoTimeNs(42));
        assert_eq!(profile.model_hash, Some("hash".into()));
        assert_eq!(profile.camera_fingerprint, Some("fingerprint".into()));
    }
}
