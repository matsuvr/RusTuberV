//! Pure shoulder-relative retargeting and observation-rate arm smoothing.
//!
//! Visibility, loss/recovery, calibration sample selection, and render-clock
//! interpolation are separate policies. None of them is silently performed here.

use std::num::NonZeroU64;

use nalgebra::Vector3;
use vtuber_core::arm_tracking::{ArmLandmarks, ArmTrackingTarget};

/// A fixed subject arm length measured at calibration, never remeasured per tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmReferenceLength(f32);

impl ArmReferenceLength {
    /// Total upper-arm plus forearm length in meters.
    #[must_use]
    pub const fn meters(self) -> f32 {
        self.0
    }
}

/// Measures one calibration sample. The caller selects a reliably observed sample.
///
/// Zero-length bones have no usable geometry and return None. This does not
/// invent an average-human arm, reuse another side, or select an idle pose.
#[must_use]
pub fn measure_arm_reference(arm: ArmLandmarks) -> Option<ArmReferenceLength> {
    let upper = (vector(arm.elbow.meters) - vector(arm.shoulder.meters)).norm();
    let lower = (vector(arm.wrist.meters) - vector(arm.elbow.meters)).norm();
    let total = upper + lower;
    (upper > 0.0 && lower > 0.0 && total.is_finite()).then_some(ArmReferenceLength(total))
}

/// Removes subject translation, converts basis once, and divides by a fixed scale.
///
/// Inputs are validated world observations, not normalized image coordinates.
/// The +Y/+Z sign change converts Pose's basis to ArmTrackingTarget's canonical
/// front view. No torso rotation, mirror, avatar scale, confidence policy, or
/// bone-length stretching is applied here.
#[must_use]
pub fn retarget_arm_landmarks(
    arm: ArmLandmarks,
    reference: ArmReferenceLength,
) -> ArmTrackingTarget {
    let shoulder = vector(arm.shoulder.meters);
    let offset = |point: [f32; 3]| {
        let v = (vector(point) - shoulder) / reference.meters();
        [v.x, -v.y, -v.z]
    };
    ArmTrackingTarget {
        wrist: offset(arm.wrist.meters),
        elbow_pole: offset(arm.elbow.meters),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointFilterState {
    raw: Vector3<f32>,
    filtered: Vector3<f32>,
    velocity: Vector3<f32>,
}

impl PointFilterState {
    fn new(value: [f32; 3]) -> Self {
        let value = vector(value);
        Self {
            raw: value,
            filtered: value,
            velocity: Vector3::zeros(),
        }
    }
}

/// Explicit state passed between pure filter calls; it does not own a clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmFilterState {
    wrist: PointFilterState,
    elbow: PointFilterState,
}

impl ArmFilterState {
    /// Seeds observation filtering. Visible acquisition blending belongs to the compositor.
    #[must_use]
    pub fn new(target: ArmTrackingTarget) -> Self {
        Self {
            wrist: PointFilterState::new(target.wrist),
            elbow: PointFilterState::new(target.elbow_pole),
        }
    }
}

/// One adaptive low-pass update per NEW camera observation.
///
/// elapsed_ns is the positive difference of capture timestamps, not render dt.
/// Callers do not re-feed a retained sample on every draw. Fixed research
/// starting values use a 1 Hz derivative cutoff, 1.5 Hz wrist / 1 Hz elbow
/// minimum cutoff and less depth bandwidth. They are not validated aesthetic
/// presets. Final bone lengths are enforced by the existing analytic IK.
///
/// This is a value-in/value-out function: no thread, clock, global, or ECS writes.
#[must_use]
pub fn filter_arm_target(
    previous: ArmFilterState,
    target: ArmTrackingTarget,
    elapsed_ns: NonZeroU64,
) -> (ArmFilterState, ArmTrackingTarget) {
    let seconds = elapsed_ns.get() as f32 * 1.0e-9;
    let wrist = filter_point(previous.wrist, target.wrist, seconds, 1.5, 1.0);
    let elbow = filter_point(previous.elbow, target.elbow_pole, seconds, 1.0, 0.5);
    let filtered = ArmTrackingTarget {
        wrist: array(wrist.filtered),
        elbow_pole: array(elbow.filtered),
    };
    (ArmFilterState { wrist, elbow }, filtered)
}

fn filter_point(
    previous: PointFilterState,
    target: [f32; 3],
    seconds: f32,
    minimum_cutoff_hz: f32,
    beta: f32,
) -> PointFilterState {
    let raw = vector(target);
    let derivative = (raw - previous.raw) / seconds;
    let velocity = previous.velocity + (derivative - previous.velocity) * alpha(1.0, seconds);
    let cutoff = minimum_cutoff_hz + beta * velocity.norm();
    let gain = Vector3::new(
        alpha(cutoff, seconds),
        alpha(cutoff, seconds),
        alpha(cutoff * 0.75, seconds),
    );
    let filtered = previous.filtered + (raw - previous.filtered).component_mul(&gain);
    PointFilterState {
        raw,
        filtered,
        velocity,
    }
}

fn alpha(cutoff_hz: f32, seconds: f32) -> f32 {
    let value = std::f32::consts::TAU * cutoff_hz * seconds;
    value / (1.0 + value)
}

fn vector([x, y, z]: [f32; 3]) -> Vector3<f32> {
    Vector3::new(x, y, z)
}

fn array(value: Vector3<f32>) -> [f32; 3] {
    [value.x, value.y, value.z]
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::arm_tracking::PoseWorldLandmark;

    fn point(meters: [f32; 3]) -> PoseWorldLandmark {
        PoseWorldLandmark {
            meters,
            visibility: Some(1.0),
            presence: Some(1.0),
        }
    }

    fn arm() -> ArmLandmarks {
        ArmLandmarks {
            shoulder: point([0.0, 0.0, 0.0]),
            elbow: point([0.3, 0.0, 0.0]),
            wrist: point([0.3, -0.4, 0.0]),
        }
    }

    fn target(value: f32) -> ArmTrackingTarget {
        ArmTrackingTarget {
            wrist: [value; 3],
            elbow_pole: [value; 3],
        }
    }

    fn dt() -> NonZeroU64 {
        NonZeroU64::new(16_666_667).unwrap()
    }

    fn near(a: [f32; 3], b: [f32; 3]) {
        assert!((vector(a) - vector(b)).norm() < 1.0e-5, "{a:?} != {b:?}");
    }

    #[test]
    fn calibration_is_total_bone_length_not_shoulder_wrist_distance() {
        let reference = measure_arm_reference(arm()).unwrap();
        assert!((reference.meters() - 0.7).abs() < 1.0e-6);
    }

    #[test]
    fn retarget_removes_translation_and_converts_pose_basis() {
        let reference = measure_arm_reference(arm()).unwrap();
        let original = retarget_arm_landmarks(arm(), reference);
        near(original.wrist, [3.0 / 7.0, 4.0 / 7.0, 0.0]);
        let shift =
            |p: PoseWorldLandmark| point(array(vector(p.meters) + Vector3::new(1.0, 2.0, 3.0)));
        let a = arm();
        let translated = ArmLandmarks {
            shoulder: shift(a.shoulder),
            elbow: shift(a.elbow),
            wrist: shift(a.wrist),
        };
        near(
            retarget_arm_landmarks(translated, reference).wrist,
            original.wrist,
        );
        let closer = ArmLandmarks {
            wrist: point([0.3, -0.4, -0.2]),
            ..a
        };
        assert!(retarget_arm_landmarks(closer, reference).wrist[2] > 0.0);
    }

    #[test]
    fn calibration_scale_is_frozen_when_observed_reach_changes() {
        let a = arm();
        let reference = measure_arm_reference(a).unwrap();
        let extended = ArmLandmarks {
            wrist: point([0.6, 0.0, 0.0]),
            ..a
        };
        near(
            retarget_arm_landmarks(extended, reference).wrist,
            [6.0 / 7.0, 0.0, 0.0],
        );
    }

    #[test]
    fn subject_scale_does_not_change_normalized_targets() {
        let a = arm();
        let twice = |p: PoseWorldLandmark| point(array(vector(p.meters) * 2.0));
        let b = ArmLandmarks {
            shoulder: twice(a.shoulder),
            elbow: twice(a.elbow),
            wrist: twice(a.wrist),
        };
        let a = retarget_arm_landmarks(a, measure_arm_reference(a).unwrap());
        let b = retarget_arm_landmarks(b, measure_arm_reference(b).unwrap());
        near(a.wrist, b.wrist);
        near(a.elbow_pole, b.elbow_pole);
    }

    #[test]
    fn zero_bone_has_no_calibration_instead_of_a_default_length() {
        let a = arm();
        assert_eq!(
            measure_arm_reference(ArmLandmarks {
                elbow: a.shoulder,
                ..a
            }),
            None
        );
        assert_eq!(
            measure_arm_reference(ArmLandmarks {
                wrist: a.elbow,
                ..a
            }),
            None
        );
    }

    #[test]
    fn constant_input_is_unchanged_at_both_observation_rates() {
        for ns in [16_666_667, 33_333_333] {
            let target = target(0.3);
            let mut state = ArmFilterState::new(target);
            for _ in 0..60 {
                let (next, value) = filter_arm_target(state, target, NonZeroU64::new(ns).unwrap());
                assert_eq!(value, target);
                state = next;
            }
        }
    }

    #[test]
    fn rapid_motion_gets_more_bandwidth_and_depth_less() {
        let initial = ArmFilterState::new(target(0.0));
        let (_, small) = filter_arm_target(initial, target(0.01), dt());
        let (_, large) = filter_arm_target(initial, target(1.0), dt());
        assert!(large.wrist[0] > small.wrist[0] / 0.01);
        assert!(large.wrist[2] < large.wrist[0]);
        assert!(large.elbow_pole[0] < large.wrist[0]);
    }

    #[test]
    fn stationary_jitter_is_reduced_and_calls_are_deterministic() {
        let mut state = ArmFilterState::new(target(0.0));
        let mut squared = 0.0;
        for frame in 0..120 {
            let raw = target(if frame % 2 == 0 { 0.01 } else { -0.01 });
            let result = filter_arm_target(state, raw, dt());
            assert_eq!(result, filter_arm_target(state, raw, dt()));
            state = result.0;
            squared += result.1.wrist[0].powi(2);
        }
        assert!((squared / 120.0).sqrt() < 0.005);
    }
}
