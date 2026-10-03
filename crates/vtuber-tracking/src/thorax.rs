//! Pose-derived torso proxy. Shoulder line and hip-to-shoulder midpoint define
//! its orientation; original centres remain available for girdle estimation.

use crate::filter::damped::{DEFAULT_MAX_DT_SEC, RotationSpring, VectorSpring};
use crate::filter::time::bounded_dt;
use crate::loss_blend::{LossBlend, LossBlendProfile};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use vtuber_core::MonoTimeNs;
use vtuber_core::arm_tracking::{PoseArmObservation, PoseWorldLandmark, ThoraxTarget};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Sample {
    rotation: UnitQuaternion<f32>,
    shoulders: [Vector3<f32>; 2],
    width: f32,
}

fn sample(pose: &PoseArmObservation, threshold: f32) -> Option<Sample> {
    let [left_hip, right_hip] = pose.hips?;
    let point = |p: PoseWorldLandmark| {
        let [x, y, z] = p.meters;
        (p.visibility.or(p.presence)? >= threshold && [x, y, z].iter().all(|v| v.is_finite()))
            .then_some(Vector3::new(x, -y, -z))
    };
    let left = point(pose.left.shoulder)?;
    let right = point(pose.right.shoulder)?;
    let lh = point(left_hip)?;
    let rh = point(right_hip)?;
    let origin = (lh + rh) * 0.5;
    let up = ((left + right) * 0.5 - origin).try_normalize(f32::EPSILON)?;
    let lateral = left - right;
    let lateral = (lateral - up * lateral.dot(&up)).try_normalize(f32::EPSILON)?;
    let forward = lateral.cross(&up);
    let rotation = UnitQuaternion::from_matrix(&Matrix3::from_columns(&[lateral, up, forward]));
    let width = (left - right).norm();
    if width <= f32::EPSILON {
        return None;
    }
    Some(Sample {
        rotation,
        width,
        shoulders: [
            rotation.inverse() * (left - origin),
            rotation.inverse() * (right - origin),
        ],
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ThoraxState {
    neutral: Option<Sample>,
    source: Option<Sample>,
    rotation: Option<RotationSpring>,
    shoulders: Option<[VectorSpring; 2]>,
    present: bool,
    blend: LossBlend,
}

impl ThoraxState {
    pub const fn new() -> Self {
        Self {
            neutral: None,
            source: None,
            rotation: None,
            shoulders: None,
            present: false,
            blend: LossBlend::new(),
        }
    }

    pub fn without_motion(self) -> Self {
        Self {
            neutral: self.neutral,
            ..Self::new()
        }
    }

    pub fn consume(&mut self, pose: Option<&PoseArmObservation>, threshold: f32) {
        let observed = pose.and_then(|p| sample(p, threshold));
        self.present = observed.is_some();
        if let Some(observed) = observed {
            self.neutral.get_or_insert(observed);
            self.source = Some(observed);
        }
    }

    pub fn advance(
        &mut self,
        now: MonoTimeNs,
        dt_ns: Option<u64>,
        profile: &LossBlendProfile,
    ) -> Option<ThoraxTarget> {
        self.blend.advance(now, self.present, profile);
        let neutral = self.neutral?;
        let source = self.source?;
        let dt = bounded_dt(dt_ns.unwrap_or(0) as f32 * 1.0e-9, DEFAULT_MAX_DT_SEC);
        let rotation = self
            .rotation
            .get_or_insert_with(|| RotationSpring::new(UnitQuaternion::identity()));
        rotation.step_world(source.rotation * neutral.rotation.inverse(), dt, 0.15);
        let shoulders = self
            .shoulders
            .get_or_insert_with(|| [VectorSpring::new(Vector3::zeros()); 2]);
        for ((state, point), rest) in shoulders
            .iter_mut()
            .zip(source.shoulders)
            .zip(neutral.shoulders)
        {
            let calibrated = neutral.rotation.inverse() * source.rotation * point;
            state.step((calibrated - rest) / neutral.width, dt, 0.15);
        }
        let q = rotation.value.quaternion();
        Some(ThoraxTarget {
            rotation: [q.i, q.j, q.k, q.w],
            shoulder_offsets: shoulders.map(|s| [s.position.x, s.position.y, s.position.z]),
            weight: self.blend.weight(),
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]
    use super::*;
    use vtuber_core::arm_tracking::ArmLandmarks;

    fn pose(yaw: f32) -> PoseArmObservation {
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);
        let point = |x, y| {
            let p = rotation * Vector3::new(x, y, 0.0);
            PoseWorldLandmark {
                meters: [p.x, -p.y, -p.z],
                visibility: Some(1.0),
                presence: Some(1.0),
            }
        };
        let arm = |x| ArmLandmarks {
            shoulder: point(x, 0.5),
            elbow: point(x, 0.25),
            wrist: point(x, 0.0),
            hand: None,
        };
        PoseArmObservation {
            hips: Some([point(0.15, 0.0), point(-0.15, 0.0)]),
            left: arm(0.2),
            right: arm(-0.2),
        }
    }

    #[test]
    fn torso_retains_shoulder_motion_and_returns_on_missing_hips() {
        let mut state = ThoraxState::new();
        let profile = LossBlendProfile::default();
        state.consume(Some(&pose(0.0)), 0.5);
        state.advance(MonoTimeNs(0), None, &profile).unwrap();
        state.consume(Some(&pose(0.4)), 0.5);
        let mut output = None;
        for tick in 1..=300 {
            output = state.advance(MonoTimeNs(tick * 16_000_000), Some(16_000_000), &profile);
        }
        let output = output.unwrap();
        let [x, y, z, w] = output.rotation;
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, x, y, z));
        assert!(q.angle_to(&UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.4)) < 1.0e-5);
        assert!(output.shoulder_offsets[0][2] < -0.1);
        assert!(output.shoulder_offsets[1][2] > 0.1);
        assert_eq!(output.mirrored().mirrored(), output);
        let mut missing = pose(0.4);
        missing.hips = None;
        state.consume(Some(&missing), 0.5);
        for tick in 301..=800 {
            let output = state
                .advance(MonoTimeNs(tick * 16_000_000), Some(16_000_000), &profile)
                .unwrap();
            if tick == 800 {
                assert_eq!(output.weight, 0.0);
            }
        }
        let mut reset = state.without_motion();
        assert!(reset.advance(MonoTimeNs(0), None, &profile).is_none());
        reset.consume(Some(&pose(0.0)), 0.5);
        assert_eq!(
            reset
                .advance(MonoTimeNs(1), None, &profile)
                .unwrap()
                .shoulder_offsets,
            [[0.0; 3]; 2]
        );
    }
}
