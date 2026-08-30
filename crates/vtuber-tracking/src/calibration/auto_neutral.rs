//! MediaPipe auto-neutral reference collection.
//!
//! The production MediaPipe path does not require a blocking, user-visible
//! calibration session.  The first valid face is usable immediately.  While
//! the stream continues, a short recent window replaces that fallback with a
//! robust reference.  This module only owns neutral-reference selection; the
//! caller resets pose/expression filters when the returned reference changes.

use std::collections::VecDeque;
use std::time::Duration;

use nalgebra::{Quaternion, UnitQuaternion};
use thiserror::Error;
use vtuber_core::{CameraFaceTransform, FaceTrackingSample, MonoTimeNs};

use crate::expressions::{BinocularGazeObservation, observe_mediapipe_gaze};

/// Duration of the robust recent auto-neutral window.
pub const AUTO_NEUTRAL_WINDOW: Duration = Duration::from_millis(1_200);
/// Minimum number of valid samples required for robust window aggregation.
pub const AUTO_NEUTRAL_MIN_SAMPLES: usize = 15;
/// Minimum per-eye openness weight accepted for a neutral gaze candidate.
const GAZE_NEUTRAL_MIN_WEIGHT: f32 = 0.6;
/// One sample-slot tolerance for nanosecond timestamp quantization.
///
/// The configured window is 1.2 seconds and the minimum sample count is 15.
/// At 15 Hz, integer nanosecond timestamps can leave a freshly trimmed window
/// a few nanoseconds short of the exact duration.  This bounded tolerance is
/// one configured sample slot; it does not retain or aggregate samples older
/// than the trimmed window.
const AUTO_NEUTRAL_CADENCE_TOLERANCE_NS: u64 =
    (AUTO_NEUTRAL_WINDOW.as_nanos() / AUTO_NEUTRAL_MIN_SAMPLES as u128) as u64;
/// Maximum retained candidates, derived from the required 30 Hz production
/// rate plus two boundary samples.  The time trim remains authoritative for
/// the temporal window; this cap also bounds bursts with identical timestamps.
const AUTO_NEUTRAL_MAX_RECENT_SAMPLES: usize =
    (AUTO_NEUTRAL_WINDOW.as_millis() as usize * 30 / 1_000) + 2;

/// Per-eye neutral gaze baselines captured while looking near the camera.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GazeNeutralBaseline {
    /// Left eye horizontal baseline.
    pub left_horizontal: f32,
    /// Right eye horizontal baseline.
    pub right_horizontal: f32,
    /// Left eye vertical baseline.
    pub left_vertical: f32,
    /// Right eye vertical baseline.
    pub right_vertical: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct GazeNeutralCandidate {
    left: Option<[f32; 2]>,
    right: Option<[f32; 2]>,
}

impl GazeNeutralCandidate {
    fn from_observation(observation: BinocularGazeObservation) -> Self {
        Self {
            left: (observation.left.weight >= GAZE_NEUTRAL_MIN_WEIGHT)
                .then_some([observation.left.horizontal, observation.left.vertical]),
            right: (observation.right.weight >= GAZE_NEUTRAL_MIN_WEIGHT)
                .then_some([observation.right.horizontal, observation.right.vertical]),
        }
    }

    fn merge_missing(&mut self, candidate: Self) {
        if self.left.is_none() {
            self.left = candidate.left;
        }
        if self.right.is_none() {
            self.right = candidate.right;
        }
    }

    fn baseline(self) -> Option<GazeNeutralBaseline> {
        (self.left.is_some() || self.right.is_some()).then(|| {
            let left = self.left.unwrap_or([0.0; 2]);
            let right = self.right.unwrap_or([0.0; 2]);
            GazeNeutralBaseline {
                left_horizontal: left[0],
                right_horizontal: right[0],
                left_vertical: left[1],
                right_vertical: right[1],
            }
        })
    }
}

/// Lifecycle of the automatic neutral reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutoNeutralState {
    /// No valid face sample has been observed yet.
    #[default]
    WaitingForFace,
    /// A neutral reference is available and can be used for tracking.
    Ready,
}

/// Why an auto-neutral candidate could not be accepted.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AutoNeutralError {
    /// The sample's camera transform was not finite and unit-quaternion based.
    #[error("MediaPipe camera transform is invalid")]
    InvalidTransform,
}

/// Result of offering one canonical MediaPipe face sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoNeutralUpdate {
    /// Current auto-neutral lifecycle.
    pub state: AutoNeutralState,
    /// Neutral transform to use for the current tracking generation.
    pub reference: CameraFaceTransform,
    /// Neutral per-eye gaze baselines.
    pub gaze_baseline: GazeNeutralBaseline,
    /// Number of samples retained in the recent window.
    pub recent_sample_count: usize,
    /// Whether the robust window replaced the first-valid fallback.
    pub used_robust_window: bool,
    /// Whether the caller should reset pose/expression filters.
    pub pose_reference_changed: bool,
    /// Whether the caller should reset the gaze filter.
    pub gaze_baseline_changed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    captured_at: MonoTimeNs,
    transform: CameraFaceTransform,
    gaze: GazeNeutralCandidate,
}

/// Selects an immediately usable and then robustly aggregated neutral pose.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AutoNeutralCollector {
    state: AutoNeutralState,
    first_valid: Option<Candidate>,
    reference: Option<CameraFaceTransform>,
    gaze_baseline: Option<GazeNeutralBaseline>,
    first_gaze: GazeNeutralCandidate,
    recent: VecDeque<Candidate>,
    robust_window_active: bool,
}

impl AutoNeutralCollector {
    /// Creates an empty collector in [`AutoNeutralState::WaitingForFace`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current lifecycle state.
    #[must_use]
    pub fn state(&self) -> AutoNeutralState {
        self.state
    }

    /// Returns the currently selected neutral transform, if a face was seen.
    #[must_use]
    pub fn reference(&self) -> Option<CameraFaceTransform> {
        self.reference
    }

    /// Returns the current per-eye neutral gaze baseline.
    #[must_use]
    pub fn gaze_baseline(&self) -> Option<GazeNeutralBaseline> {
        self.gaze_baseline
    }

    /// Returns the number of retained recent candidates.
    #[must_use]
    pub fn recent_sample_count(&self) -> usize {
        self.recent.len()
    }

    /// Offers a validated canonical MediaPipe sample.
    ///
    /// The first valid transform becomes the fallback immediately.  Samples
    /// are never rejected because of head movement or expression movement:
    /// those values are exactly what the old blocking calibration incorrectly
    /// treated as a reason to keep waiting.  Once the recent window contains
    /// at least [`AUTO_NEUTRAL_MIN_SAMPLES`] samples spanning the configured
    /// interval, a robust aggregate becomes the reference.
    pub fn observe(
        &mut self,
        sample: &FaceTrackingSample,
    ) -> Result<AutoNeutralUpdate, AutoNeutralError> {
        if !sample.camera_to_face.is_valid() {
            return Err(AutoNeutralError::InvalidTransform);
        }

        let candidate = Candidate {
            captured_at: sample.inference_finished_at,
            transform: sample.camera_to_face,
            gaze: GazeNeutralCandidate::from_observation(observe_mediapipe_gaze(
                &sample.blendshapes,
            )),
        };
        let reference_was = self.reference;
        let gaze_baseline_was = self.gaze_baseline;

        if self.first_valid.is_none() {
            self.first_valid = Some(candidate);
            self.reference = Some(candidate.transform);
            self.state = AutoNeutralState::Ready;
        }
        if !self.robust_window_active {
            self.first_gaze.merge_missing(candidate.gaze);
            self.gaze_baseline = self.first_gaze.baseline();
        }

        self.recent.push_back(candidate);
        self.trim_window(candidate.captured_at);

        if !self.robust_window_active
            && self.recent.len() >= AUTO_NEUTRAL_MIN_SAMPLES
            && spans_window(&self.recent, AUTO_NEUTRAL_WINDOW)
        {
            let aggregate =
                aggregate_candidates(&self.recent).ok_or(AutoNeutralError::InvalidTransform)?;
            self.reference = Some(aggregate);
            self.gaze_baseline = aggregate_gaze_baseline(&self.recent, self.gaze_baseline);
            self.robust_window_active = true;
        }

        let reference = self
            .reference
            .or_else(|| self.first_valid.map(|candidate| candidate.transform))
            .ok_or(AutoNeutralError::InvalidTransform)?;
        let gaze_baseline = self.gaze_baseline.unwrap_or_default();
        Ok(AutoNeutralUpdate {
            state: self.state,
            reference,
            gaze_baseline,
            recent_sample_count: self.recent.len(),
            used_robust_window: self.robust_window_active,
            pose_reference_changed: reference_was != Some(reference),
            gaze_baseline_changed: gaze_baseline_was != self.gaze_baseline,
        })
    }

    /// Replaces the neutral reference immediately with the supplied sample.
    ///
    /// Recenter intentionally clears the recent window: post-recenter frames
    /// must establish a fresh local reference instead of blending old and new
    /// camera geometry.  The returned update tells the caller to reset all
    /// smoothing state before applying the next tracked frame.
    pub fn recenter(
        &mut self,
        sample: &FaceTrackingSample,
    ) -> Result<AutoNeutralUpdate, AutoNeutralError> {
        if !sample.camera_to_face.is_valid() {
            return Err(AutoNeutralError::InvalidTransform);
        }
        let candidate = Candidate {
            captured_at: sample.inference_finished_at,
            transform: sample.camera_to_face,
            gaze: GazeNeutralCandidate::from_observation(observe_mediapipe_gaze(
                &sample.blendshapes,
            )),
        };
        let gaze_baseline_was = self.gaze_baseline;
        self.state = AutoNeutralState::Ready;
        self.first_valid = Some(candidate);
        self.reference = Some(candidate.transform);
        self.first_gaze = candidate.gaze;
        self.gaze_baseline = self.first_gaze.baseline();
        self.recent.clear();
        self.recent.push_back(candidate);
        self.robust_window_active = false;
        Ok(AutoNeutralUpdate {
            state: self.state,
            reference: candidate.transform,
            gaze_baseline: self.gaze_baseline.unwrap_or_default(),
            recent_sample_count: 1,
            used_robust_window: false,
            pose_reference_changed: true,
            gaze_baseline_changed: gaze_baseline_was != self.gaze_baseline,
        })
    }

    /// Clears the neutral reference and waits for the next valid face.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn trim_window(&mut self, newest: MonoTimeNs) {
        while self.recent.front().is_some_and(|candidate| {
            newest.0.saturating_sub(candidate.captured_at.0) > AUTO_NEUTRAL_WINDOW.as_nanos() as u64
        }) {
            let _ = self.recent.pop_front();
        }
        while self.recent.len() > AUTO_NEUTRAL_MAX_RECENT_SAMPLES {
            let _ = self.recent.pop_front();
        }
    }
}

fn spans_window(candidates: &VecDeque<Candidate>, window: Duration) -> bool {
    match (candidates.front(), candidates.back()) {
        (Some(first), Some(last)) => {
            last.captured_at
                .0
                .saturating_sub(first.captured_at.0)
                .saturating_add(AUTO_NEUTRAL_CADENCE_TOLERANCE_NS)
                >= window.as_nanos() as u64
        }
        _ => false,
    }
}

fn aggregate_candidates(candidates: &VecDeque<Candidate>) -> Option<CameraFaceTransform> {
    let first = candidates.front()?.transform;
    let mut quaternions = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let quaternion = to_quaternion(candidate.transform)?;
        let quaternion = if dot(first, candidate.transform) < 0.0 {
            negate(quaternion)
        } else {
            quaternion
        };
        quaternions.push(quaternion);
    }

    let mut sum = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    for quaternion in &quaternions {
        let value = quaternion.quaternion();
        sum.w += value.w;
        sum.i += value.i;
        sum.j += value.j;
        sum.k += value.k;
    }
    let rotation = UnitQuaternion::try_new(sum, 1.0e-6)?;
    Some(CameraFaceTransform {
        rotation_xyzw: [
            rotation.quaternion().i,
            rotation.quaternion().j,
            rotation.quaternion().k,
            rotation.quaternion().w,
        ],
        translation_xyz: [
            median(
                candidates
                    .iter()
                    .map(|candidate| candidate.transform.translation_xyz[0]),
            ),
            median(
                candidates
                    .iter()
                    .map(|candidate| candidate.transform.translation_xyz[1]),
            ),
            median(
                candidates
                    .iter()
                    .map(|candidate| candidate.transform.translation_xyz[2]),
            ),
        ],
    })
}

fn aggregate_gaze_baseline(
    candidates: &VecDeque<Candidate>,
    fallback: Option<GazeNeutralBaseline>,
) -> Option<GazeNeutralBaseline> {
    let has_fallback = fallback.is_some();
    let fallback = fallback.unwrap_or_default();
    let left_horizontal = median_nonempty(
        candidates
            .iter()
            .filter_map(|candidate| candidate.gaze.left.map(|value| value[0])),
    );
    let left_vertical = median_nonempty(
        candidates
            .iter()
            .filter_map(|candidate| candidate.gaze.left.map(|value| value[1])),
    );
    let right_horizontal = median_nonempty(
        candidates
            .iter()
            .filter_map(|candidate| candidate.gaze.right.map(|value| value[0])),
    );
    let right_vertical = median_nonempty(
        candidates
            .iter()
            .filter_map(|candidate| candidate.gaze.right.map(|value| value[1])),
    );
    let has_any = left_horizontal.is_some()
        || left_vertical.is_some()
        || right_horizontal.is_some()
        || right_vertical.is_some();
    (has_fallback || has_any).then(|| GazeNeutralBaseline {
        left_horizontal: left_horizontal.unwrap_or(fallback.left_horizontal),
        right_horizontal: right_horizontal.unwrap_or(fallback.right_horizontal),
        left_vertical: left_vertical.unwrap_or(fallback.left_vertical),
        right_vertical: right_vertical.unwrap_or(fallback.right_vertical),
    })
}

fn to_quaternion(transform: CameraFaceTransform) -> Option<UnitQuaternion<f32>> {
    if !transform.is_valid() {
        return None;
    }
    Some(UnitQuaternion::from_quaternion(Quaternion::new(
        transform.rotation_xyzw[3],
        transform.rotation_xyzw[0],
        transform.rotation_xyzw[1],
        transform.rotation_xyzw[2],
    )))
}

fn dot(left: CameraFaceTransform, right: CameraFaceTransform) -> f32 {
    left.rotation_xyzw
        .iter()
        .zip(right.rotation_xyzw)
        .map(|(a, b)| a * b)
        .sum()
}

fn negate(value: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    let q = value.quaternion();
    UnitQuaternion::from_quaternion(Quaternion::new(-q.w, -q.i, -q.j, -q.k))
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn median(values: impl Iterator<Item = f32>) -> f32 {
    let mut values: Vec<f32> = values.collect();
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn median_nonempty(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut values: Vec<f32> = values.collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    Some(values[values.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vtuber_core::{
        FaceBlendshapeSet, FaceLandmark, FaceTrackingQuality, FrameSeq, MediaPipeBlendshape,
    };

    fn sample(seq: u64, time_ns: u64, translation_x: f32) -> FaceTrackingSample {
        FaceTrackingSample {
            source_seq: FrameSeq(seq),
            captured_at: MonoTimeNs(time_ns),
            inference_started_at: MonoTimeNs(time_ns + 1),
            inference_finished_at: MonoTimeNs(time_ns + 2),
            camera_to_face: CameraFaceTransform {
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                translation_xyz: [translation_x, 0.0, 0.0],
            },
            face_center: [0.5, 0.5],
            landmarks: Arc::from(vec![FaceLandmark::default(); 478]),
            blendshapes: FaceBlendshapeSet::default(),
            quality: FaceTrackingQuality {
                matrix_determinant: 1.0,
                ..FaceTrackingQuality::default()
            },
        }
    }

    fn sample_with_gaze(
        seq: u64,
        time_ns: u64,
        values: &[(MediaPipeBlendshape, f32)],
    ) -> FaceTrackingSample {
        let mut sample = sample(seq, time_ns, 0.0);
        let pairs: Vec<_> = MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| {
                let value = values
                    .iter()
                    .find(|(candidate, _)| *candidate == category)
                    .map_or(0.0, |(_, value)| *value);
                (category.as_str(), value)
            })
            .collect();
        sample.blendshapes = FaceBlendshapeSet::from_pairs(&pairs).expect("complete typed set");
        sample
    }

    #[test]
    fn first_valid_sample_is_available_without_blocking() {
        let mut collector = AutoNeutralCollector::new();
        let update = collector.observe(&sample(1, 1, 2.0)).unwrap();
        assert_eq!(update.state, AutoNeutralState::Ready);
        assert_eq!(update.recent_sample_count, 1);
        assert!(!update.used_robust_window);
        assert_eq!(update.reference.translation_xyz[0], 2.0);
    }

    #[test]
    fn neutral_records_all_four_per_eye_gaze_baselines() {
        let sample = sample_with_gaze(
            1,
            1,
            &[
                (MediaPipeBlendshape::EyeLookOutLeft, 0.4),
                (MediaPipeBlendshape::EyeLookInRight, 0.3),
                (MediaPipeBlendshape::EyeLookUpLeft, 0.2),
                (MediaPipeBlendshape::EyeLookDownRight, 0.1),
            ],
        );
        let mut collector = AutoNeutralCollector::new();
        let update = collector.observe(&sample).unwrap();
        assert_eq!(update.gaze_baseline.left_horizontal, 0.4);
        assert_eq!(update.gaze_baseline.right_horizontal, 0.3);
        assert_eq!(update.gaze_baseline.left_vertical, 0.2);
        assert_eq!(update.gaze_baseline.right_vertical, -0.1);
    }

    #[test]
    fn robust_window_activates_at_15_hz_and_rejects_a_pose_outlier() {
        let mut collector = AutoNeutralCollector::new();
        for seq in 0..20 {
            let x = if seq == 19 {
                100.0
            } else {
                1.0 + seq as f32 * 0.01
            };
            let update = collector
                .observe(&sample(seq + 1, seq * (1_000_000_000 / 15), x))
                .unwrap();
            if seq == 19 {
                assert!(update.used_robust_window);
                assert!((update.reference.translation_xyz[0] - 1.10).abs() < 0.03);
            }
        }
    }

    #[test]
    fn robust_window_activates_at_30_hz() {
        let mut collector = AutoNeutralCollector::new();
        let mut update = None;
        for seq in 0..40 {
            update = Some(
                collector
                    .observe(&sample(seq + 1, seq * (1_000_000_000 / 30), 0.0))
                    .unwrap(),
            );
        }
        assert!(update.unwrap().used_robust_window);
    }

    #[test]
    fn tracking_gap_discards_stale_samples_before_robust_activation() {
        let mut collector = AutoNeutralCollector::new();
        let interval = 1_000_000_000 / 15;
        for seq in 0..14 {
            let _ = collector.observe(&sample(seq + 1, seq * interval, 1.0));
        }

        let update = collector.observe(&sample(15, 3_000_000_000, 99.0)).unwrap();
        assert!(!update.used_robust_window);
        assert_eq!(update.recent_sample_count, 1);
        assert_eq!(update.reference.translation_xyz[0], 1.0);
    }

    #[test]
    fn fresh_samples_after_tracking_gap_eventually_form_the_robust_reference() {
        let mut collector = AutoNeutralCollector::new();
        let interval = 1_000_000_000 / 15;
        for seq in 0..14 {
            let _ = collector.observe(&sample(seq + 1, seq * interval, 1.0));
        }

        let mut activated = None;
        for seq in 0..20 {
            let update = collector
                .observe(&sample(
                    100 + seq,
                    3_000_000_000 + seq * interval,
                    10.0 + seq as f32,
                ))
                .unwrap();
            if update.used_robust_window {
                activated = Some(update);
                break;
            }
        }

        let update = activated.expect("fresh post-gap samples should activate robust neutral");
        assert!(update.reference.translation_xyz[0] > 9.0);
        assert!(update.recent_sample_count <= AUTO_NEUTRAL_MAX_RECENT_SAMPLES);
    }

    #[test]
    fn recent_sample_count_stays_bounded_for_long_running_input() {
        let mut collector = AutoNeutralCollector::new();
        let interval = 1_000_000_000 / 30;
        for seq in 0..5_000 {
            let update = collector
                .observe(&sample(seq + 1, seq * interval, 0.0))
                .unwrap();
            assert!(update.recent_sample_count <= AUTO_NEUTRAL_MAX_RECENT_SAMPLES);
        }
    }

    #[test]
    fn robust_gaze_median_rejects_outlier_and_remains_active() {
        let mut collector = AutoNeutralCollector::new();
        let interval = 1_000_000_000 / 15;
        for seq in 0..20 {
            let horizontal = if seq == 10 { 0.95 } else { 0.2 };
            let sample = sample_with_gaze(
                seq + 1,
                seq * interval,
                &[
                    (MediaPipeBlendshape::EyeLookOutLeft, horizontal),
                    (MediaPipeBlendshape::EyeLookInRight, horizontal),
                ],
            );
            let _ = collector.observe(&sample).unwrap();
        }
        let robust = collector.gaze_baseline().unwrap();
        assert_eq!(robust.left_horizontal, 0.2);
        assert_eq!(robust.right_horizontal, 0.2);

        let later = sample_with_gaze(
            21,
            20 * interval,
            &[
                (MediaPipeBlendshape::EyeLookOutLeft, 0.8),
                (MediaPipeBlendshape::EyeLookInRight, 0.8),
            ],
        );
        let update = collector.observe(&later).unwrap();
        assert!(update.used_robust_window);
        assert_eq!(update.gaze_baseline, robust);
        assert!(!update.gaze_baseline_changed);
    }

    #[test]
    fn blink_and_low_weight_eyes_are_excluded_from_gaze_baseline() {
        let blink = sample_with_gaze(
            1,
            1,
            &[
                (MediaPipeBlendshape::EyeLookOutLeft, 0.9),
                (MediaPipeBlendshape::EyeLookInRight, 0.8),
                (MediaPipeBlendshape::EyeBlinkLeft, 1.0),
                (MediaPipeBlendshape::EyeBlinkRight, 0.5),
            ],
        );
        let mut collector = AutoNeutralCollector::new();
        let update = collector.observe(&blink).unwrap();
        assert_eq!(collector.gaze_baseline(), None);
        assert!(!update.gaze_baseline_changed);

        let open = sample_with_gaze(
            2,
            70_000_000,
            &[
                (MediaPipeBlendshape::EyeLookOutLeft, 0.2),
                (MediaPipeBlendshape::EyeLookInRight, 0.3),
            ],
        );
        let update = collector.observe(&open).unwrap();
        assert!(update.gaze_baseline_changed);
        assert_eq!(update.gaze_baseline.left_horizontal, 0.2);
        assert_eq!(update.gaze_baseline.right_horizontal, 0.3);
    }

    #[test]
    fn robust_gaze_keeps_a_valid_fallback_when_all_candidates_are_excluded() {
        let mut collector = AutoNeutralCollector::new();
        let interval = 1_000_000_000 / 15;
        let fallback_sample = sample_with_gaze(
            1,
            0,
            &[
                (MediaPipeBlendshape::EyeLookOutLeft, 0.2),
                (MediaPipeBlendshape::EyeLookUpLeft, 0.1),
                (MediaPipeBlendshape::EyeLookInRight, 0.3),
                (MediaPipeBlendshape::EyeLookUpRight, 0.4),
            ],
        );
        let fallback = collector.observe(&fallback_sample).unwrap().gaze_baseline;

        let mut activated = None;
        for seq in 1..=20 {
            let sample = sample_with_gaze(
                seq + 1,
                seq * interval,
                &[
                    (MediaPipeBlendshape::EyeBlinkLeft, 1.0),
                    (MediaPipeBlendshape::EyeBlinkRight, 1.0),
                ],
            );
            let update = collector.observe(&sample).unwrap();
            if update.used_robust_window {
                activated = Some(update);
                break;
            }
        }

        let update = activated.expect("excluded gaze candidates should not block robust pose");
        assert_eq!(update.gaze_baseline, fallback);
        assert!(!update.gaze_baseline_changed);
    }

    #[test]
    fn robust_gaze_updates_one_eye_and_retains_the_other_fallback() {
        let mut collector = AutoNeutralCollector::new();
        let interval = 1_000_000_000 / 15;
        let fallback_sample = sample_with_gaze(
            1,
            0,
            &[
                (MediaPipeBlendshape::EyeLookOutLeft, 0.2),
                (MediaPipeBlendshape::EyeLookUpLeft, 0.1),
                (MediaPipeBlendshape::EyeLookInRight, 0.3),
                (MediaPipeBlendshape::EyeLookUpRight, 0.4),
            ],
        );
        let fallback = collector.observe(&fallback_sample).unwrap().gaze_baseline;

        let mut activated = None;
        for seq in 1..=20 {
            let sample = sample_with_gaze(
                seq + 1,
                seq * interval,
                &[
                    (MediaPipeBlendshape::EyeLookOutLeft, 0.8),
                    (MediaPipeBlendshape::EyeLookUpLeft, 0.6),
                    (MediaPipeBlendshape::EyeBlinkRight, 1.0),
                ],
            );
            let update = collector.observe(&sample).unwrap();
            if update.used_robust_window {
                activated = Some(update);
                break;
            }
        }

        let update = activated.expect("left-eye candidates should still activate robust pose");
        assert_eq!(update.gaze_baseline.left_horizontal, 0.8);
        assert_eq!(update.gaze_baseline.left_vertical, 0.6);
        assert_eq!(
            update.gaze_baseline.right_horizontal,
            fallback.right_horizontal
        );
        assert_eq!(update.gaze_baseline.right_vertical, fallback.right_vertical);
        assert!(update.gaze_baseline_changed);
    }

    #[test]
    fn robust_gaze_stays_absent_without_fallback_or_valid_candidates() {
        let mut collector = AutoNeutralCollector::new();
        let interval = 1_000_000_000 / 15;
        for seq in 0..20 {
            let sample = sample_with_gaze(
                seq + 1,
                seq * interval,
                &[
                    (MediaPipeBlendshape::EyeBlinkLeft, 1.0),
                    (MediaPipeBlendshape::EyeBlinkRight, 1.0),
                ],
            );
            let update = collector.observe(&sample).unwrap();
            if update.used_robust_window {
                assert_eq!(collector.gaze_baseline(), None);
                assert_eq!(update.gaze_baseline, GazeNeutralBaseline::default());
                assert!(!update.gaze_baseline_changed);
            }
        }
        assert_eq!(collector.gaze_baseline(), None);
    }

    #[test]
    fn recenter_is_instant_and_starts_a_fresh_window() {
        let mut collector = AutoNeutralCollector::new();
        for seq in 0..16 {
            let _ = collector.observe(&sample(seq + 1, seq * 20_000_000, 0.0));
        }
        let update = collector.recenter(&sample(99, 1_000_000_000, 4.0)).unwrap();
        assert!(update.pose_reference_changed);
        assert_eq!(update.reference.translation_xyz[0], 4.0);
        assert_eq!(update.recent_sample_count, 1);
        assert!(!update.used_robust_window);
    }

    #[test]
    fn no_face_keeps_waiting_without_fabricating_a_reference() {
        let collector = AutoNeutralCollector::new();
        assert_eq!(collector.state(), AutoNeutralState::WaitingForFace);
        assert!(collector.reference().is_none());
    }
}
