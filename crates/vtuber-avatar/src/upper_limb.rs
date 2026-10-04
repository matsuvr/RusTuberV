//! Joint-coordinate candidate and single FK for the upper-limb solver.

use crate::{
    arm::{ArmChainBinding, ArmIkInput, ArmIkTarget, RestSpaceBonePose},
    arm_pose::{ResolvedArmPose, ResolvedBoneDelta, ResolvedFingerPose},
    collision::BoneMotion,
};
use bevy::prelude::*;
use std::collections::HashMap;
use vtuber_core::arm_tracking::HandFingerPose;

/// Plane, elevation, axial rotation, elbow, radioulnar rotation, wrist
/// flexion/deviation, independent SC protraction/elevation. AC remains
/// coupled to humeral plane/elevation by the published rhythm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ArmJoints {
    pub angles: [f32; 9],
    pub fingers: Option<HandFingerPose>,
    pub finger_weight: f32,
    pub rest_curl: f32,
}

pub(crate) const BOUNDS: [(f32, f32); 9] = [
    // Same MyoArm 93b0ca8 revision as the axes and offsets. Its inverse
    // phantom joint cancels the plane rotation before shoulder_rot, matching
    // our globe axial coordinate. Do not mix in a different model's bounds.
    (-1.658, 2.269),
    (0.0, std::f32::consts::PI),
    (-1.571, 2.094),
    (0.0, crate::arm::ELBOW_FLEXION_LIMIT_RAD),
    (
        -crate::arm::FOREARM_ROLL_LIMIT_RAD,
        crate::arm::FOREARM_ROLL_LIMIT_RAD,
    ),
    (-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4),
    (
        -10.0 * std::f32::consts::PI / 180.0,
        25.0 * std::f32::consts::PI / 180.0,
    ),
    // Envelope of the adopted bivariate rhythm over the admitted humeral
    // plane/elevation domain. MyoArm's one-variable rhythm limits would
    // incorrectly forbid the positive protraction term at anterior planes.
    (-0.120 * 1.658 - 0.242 * std::f32::consts::PI, 0.120 * 2.269),
    (-0.046 * 2.269, 0.046 * 1.658 + 0.123 * std::f32::consts::PI),
];

#[derive(Clone, Debug)]
pub(crate) struct ArmCandidate {
    pub joint_margin: f32,
    pub joint_margins: [f32; 27],
    pub chart_margin: f32,
    pub resolved: ResolvedArmPose,
    pub shoulder: Vec3,
    pub elbow: Vec3,
    pub wrist: Vec3,
    pub palm: Option<(Vec3, Vec3)>,
    pub motion: HashMap<Entity, BoneMotion>,
}

impl ArmJoints {
    /// Interpret the shared resting profile in the imported model's space.
    pub fn resting(
        chain: &ArmChainBinding,
        profile: crate::arm::ArmPoseProfile,
    ) -> Option<(Self, ArmIkTarget)> {
        let target = crate::arm::default_arm_target(chain, profile).ok()?;
        let solution =
            crate::arm::solve_two_bone_arm(ArmIkInput::from_chain(chain, target)).ok()?;
        let mut joints = Self::from_solution(chain, solution)?;
        joints.rest_curl = profile.finger_curl_radians;
        Some((joints, target))
    }

    pub fn valid(&self) -> bool {
        self.angles
            .iter()
            .zip(BOUNDS)
            .all(|(v, (min, max))| v.is_finite() && *v >= min && *v <= max)
    }

    pub fn bound(&mut self) {
        for (angle, (min, max)) in self.angles.iter_mut().zip(BOUNDS) {
            *angle = angle.clamp(min, max);
        }
    }

    pub fn from_solution(
        chain: &ArmChainBinding,
        solution: crate::arm::ArmIkSolution,
    ) -> Option<Self> {
        let input = ArmIkInput::from_chain(
            chain,
            ArmIkTarget {
                wrist: solution.wrist,
                elbow_pole: solution.elbow,
            },
        );
        let shoulder = crate::shoulder::coordinates(chain, solution.upper_arm_global_rotation)?;
        let elbow =
            crate::skeleton::joint_coordinates(input.skeleton_rest(), solution.skeleton_pose())?;
        let p = shoulder.plane.unwrap_or(0.0);
        let sc = crate::girdle::rhythm(p, shoulder.elevation);
        let mut state = Self {
            angles: [
                p,
                shoulder.elevation,
                shoulder.axial,
                elbow.x,
                elbow.y,
                0.0,
                0.0,
                sc[0],
                sc[1],
            ],
            fingers: None,
            finger_weight: 0.0,
            rest_curl: 0.0,
        };
        state.bound();
        Some(state)
    }

    pub fn forward(self, chain: &ArmChainBinding) -> Option<ArmCandidate> {
        if !self.valid() {
            return None;
        }
        let [
            plane,
            elevation,
            axial,
            flexion,
            roll,
            wrist_f,
            wrist_d,
            sc_pro,
            sc_elv,
        ] = self.angles;
        let upper = crate::shoulder::from_coordinates(
            chain,
            crate::shoulder::ShoulderCoordinates {
                plane: Some(plane),
                elevation,
                axial,
            },
        )?;
        let input = ArmIkInput::from_chain(
            chain,
            ArmIkTarget {
                wrist: chain.rest.wrist.position,
                elbow_pole: chain.rest.elbow.position,
            },
        );
        let pose = crate::skeleton::from_joints(input.skeleton_rest(), upper, flexion, roll)?;
        let girdle =
            crate::girdle::forward(chain, plane, elevation, Some(Vec2::new(sc_pro, sc_elv)))?;
        let displacement = girdle.centre - chain.rest.upper_arm.position;
        let elbow = pose.middle + displacement;
        let wrist = pose.end + displacement;
        let lower_model = pose.middle_rotation * chain.rest.elbow.global_rotation.inverse();
        let palm =
            crate::arm::rest_palm_normal(chain).zip(crate::tracked_arm::rest_palm_forward(chain));
        let wrist_delta = if let Some((normal, _)) = palm {
            let frame = crate::tracked_arm::palm_frame(
                normal,
                chain.rest.wrist.position - chain.rest.elbow.position,
            )?;
            crate::skeleton::rest_delta(
                frame
                    * Quat::from_rotation_x(wrist_f)
                    * Quat::from_rotation_z(wrist_d)
                    * frame.inverse(),
                chain.rest.wrist.global_rotation,
            )
        } else if wrist_f == 0.0 && wrist_d == 0.0 {
            Quat::IDENTITY
        } else {
            return None;
        };
        let hand_model = lower_model
            * chain.rest.wrist.global_rotation
            * wrist_delta
            * chain.rest.wrist.global_rotation.inverse();
        let fingers = self
            .fingers
            .and_then(|f| {
                crate::tracked_arm::observed_finger_deltas(
                    chain,
                    f,
                    self.finger_weight,
                    self.rest_curl,
                )
            })
            .unwrap_or_else(|| crate::arm_pose::resolve_finger_pose(chain, self.rest_curl));
        let shoulder = chain
            .shoulder
            .zip(chain.rest.shoulder)
            .map(|(entity, rest)| ResolvedBoneDelta {
                entity,
                delta: crate::skeleton::rest_delta(girdle.rotation, rest.global_rotation),
            });
        let resolved = ResolvedArmPose {
            upper_arm: chain.upper_arm,
            lower_arm: chain.lower_arm,
            upper_arm_delta: (chain.rest.upper_arm.global_rotation.inverse()
                * girdle.rotation.inverse()
                * upper)
                .normalize(),
            lower_arm_delta: pose.middle_delta,
            shoulder,
            hand: Some(ResolvedBoneDelta {
                entity: chain.hand,
                delta: wrist_delta,
            }),
            fingers,
        };
        let mut motion = HashMap::new();
        let mut add = |entity, rest: RestSpaceBonePose, position, rotation: Quat| {
            motion.insert(
                entity,
                BoneMotion {
                    rotation,
                    translation: position - rotation * rest.position,
                },
            );
        };
        if let Some((bone, rest)) = chain.shoulder.zip(chain.rest.shoulder) {
            add(bone, rest, rest.position, girdle.rotation);
        }
        add(
            chain.upper_arm,
            chain.rest.upper_arm,
            girdle.centre,
            upper * chain.rest.upper_arm.global_rotation.inverse(),
        );
        add(chain.lower_arm, chain.rest.elbow, elbow, lower_model);
        add(chain.hand, chain.rest.wrist, wrist, hand_model);
        append_fingers(chain, &fingers, hand_model, wrist, &mut motion);
        let (joint_margins, chart_margin) =
            crate::joint_limits::shoulder_margins(chain, girdle.rotation, upper)?;
        Some(ArmCandidate {
            joint_margin: joint_margins.into_iter().fold(f32::INFINITY, f32::min),
            joint_margins,
            chart_margin,
            resolved,
            shoulder: girdle.centre,
            elbow,
            wrist,
            palm: palm.map(|(n, f)| (hand_model * n, hand_model * f)),
            motion,
        })
    }
}

fn append_fingers(
    chain: &ArmChainBinding,
    pose: &ResolvedFingerPose,
    hand_rotation: Quat,
    wrist: Vec3,
    motion: &mut HashMap<Entity, BoneMotion>,
) {
    for (rest, solved) in [
        (chain.finger_rest.thumb, pose.thumb),
        (chain.finger_rest.index, pose.index),
        (chain.finger_rest.middle, pose.middle),
        (chain.finger_rest.ring, pose.ring),
        (chain.finger_rest.little, pose.little),
    ] {
        let mut parent_rest = chain.rest.wrist.position;
        let mut parent_position = wrist;
        let mut parent_rotation = hand_rotation;
        for (joint, delta) in [
            (rest.metacarpal, solved.metacarpal),
            (rest.proximal, solved.proximal),
            (rest.intermediate, solved.intermediate),
            (rest.distal, solved.distal),
        ] {
            let Some(joint) = joint else { continue };
            let position = parent_position + parent_rotation * (joint.rest.position - parent_rest);
            let rotation = parent_rotation
                * joint.rest.global_rotation
                * delta.map_or(Quat::IDENTITY, |d| d.delta)
                * joint.rest.global_rotation.inverse();
            motion.insert(
                joint.entity,
                BoneMotion {
                    rotation,
                    translation: position - rotation * joint.rest.position,
                },
            );
            parent_rest = joint.rest.position;
            parent_position = position;
            parent_rotation = rotation;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;
    use crate::arm::*;

    #[test]
    fn submilliradian_fk_arcs_do_not_snap_to_identity() {
        let from = Quat::from_euler(EulerRot::XYZ, 0.3, -0.2, 0.4) * Vec3::Y;
        let axis = from.cross(Vec3::Z).normalize();
        for i in -100..=100 {
            let angle = i as f32 * 0.00001;
            let to = Quat::from_axis_angle(axis, angle) * from;
            let arc = crate::skeleton::minimal_arc(from, to).unwrap();
            assert!((arc * from).distance(to) < 4.0 * f32::EPSILON);
        }
    }

    pub fn chain(side: ArmSide) -> ArmChainBinding {
        let s = if side == ArmSide::Left { 1.0 } else { -1.0 };
        let base = if side == ArmSide::Left { 10 } else { 30 };
        let entity = |i: u32| Entity::from_raw_u32(base + i).unwrap();
        let rest = |p| RestSpaceBonePose {
            position: p,
            global_rotation: Quat::from_euler(EulerRot::XYZ, 0.3, -0.2, 0.4),
            local_rotation: Quat::IDENTITY,
        };
        let shoulder = rest(Vec3::new(s * 0.035, 1.35, 0.0));
        let upper = rest(Vec3::new(s * 0.16, 1.35, 0.0));
        let elbow = rest(upper.position + Vec3::new(s * 0.30, 0.0, 0.0));
        let wrist = rest(elbow.position + Vec3::new(s * 0.26, -0.01, 0.0));
        let finger = |i, z| {
            Some(FingerJointRestBinding {
                entity: entity(i),
                rest: rest(wrist.position + Vec3::new(s * 0.06, 0.0, z)),
            })
        };
        ArmChainBinding {
            side,
            shoulder: Some(entity(0)),
            upper_arm: entity(1),
            lower_arm: entity(2),
            hand: entity(3),
            fingers: Default::default(),
            finger_rest: FingerRestReferences {
                index: FingerJointRestReferences {
                    proximal: finger(4, 0.02),
                    ..Default::default()
                },
                little: FingerJointRestReferences {
                    proximal: finger(5, -0.02),
                    ..Default::default()
                },
                ..Default::default()
            },
            rest: ArmRestGeometry {
                shoulder: Some(shoulder),
                upper_arm: upper,
                elbow,
                wrist,
                upper_arm_length: upper.position.distance(elbow.position),
                forearm_length: elbow.position.distance(wrist.position),
                total_arm_length: upper.position.distance(elbow.position)
                    + elbow.position.distance(wrist.position),
            },
            capabilities: ArmChainCapabilities {
                has_shoulder: true,
                has_fingers: true,
            },
        }
    }

    pub fn state(p: f32, e: f32, a: f32, f: f32) -> ArmJoints {
        let [sp, se, ..] = crate::girdle::rhythm(p, e);
        let mut state = ArmJoints {
            angles: [p, e, a, f, 0.0, 0.0, 0.0, sp, se],
            fingers: None,
            finger_weight: 0.0,
            rest_curl: 0.0,
        };
        state.bound();
        state
    }

    #[test]
    fn candidate_matches_local_hierarchy_with_oblique_elbow_and_moving_shoulder() {
        for side in [ArmSide::Left, ArmSide::Right] {
            let chain = chain(side);
            let mut joints = state(1.2, 1.5, -0.4, 1.1);
            joints.angles[4] = 0.7;
            joints.angles[5] = 0.3;
            joints.angles[6] = 0.15;
            let pose = joints.forward(&chain).unwrap();
            let rs = chain.rest.shoulder.unwrap();
            let gs = rs.global_rotation
                * pose.resolved.shoulder.unwrap().delta
                * rs.global_rotation.inverse();
            let ru = chain.rest.upper_arm;
            let gu = gs
                * ru.global_rotation
                * pose.resolved.upper_arm_delta
                * ru.global_rotation.inverse();
            let re = chain.rest.elbow;
            let gl = gu
                * re.global_rotation
                * pose.resolved.lower_arm_delta
                * re.global_rotation.inverse();
            let shoulder = rs.position + gs * (ru.position - rs.position);
            let elbow = shoulder + gu * (re.position - ru.position);
            let wrist = elbow + gl * (chain.rest.wrist.position - re.position);
            assert!(shoulder.distance(pose.shoulder) < 2.0e-6);
            assert!(elbow.distance(pose.elbow) < 2.0e-6);
            assert!(wrist.distance(pose.wrist) < 2.0e-6);
            assert!((elbow.distance(wrist) - chain.rest.forearm_length).abs() < 2.0e-6);
        }
    }

    #[test]
    fn reflection_preserves_centres_and_axial_palm_semantics() {
        let left = chain(ArmSide::Left);
        let right = chain(ArmSide::Right);
        let mut l = state(1.2, 1.5, -0.4, 1.1);
        l.angles[4] = 0.7;
        l.angles[5] = 0.3;
        l.angles[6] = 0.15;
        let mut r = l;
        r.angles[4] *= -1.0;
        r.angles[5] *= -1.0;
        let l = l.forward(&left).unwrap();
        let r = r.forward(&right).unwrap();
        let reflect = |v: Vec3| Vec3::new(-v.x, v.y, v.z);
        for (a, b) in [
            (l.shoulder, r.shoulder),
            (l.elbow, r.elbow),
            (l.wrist, r.wrist),
        ] {
            assert!(reflect(a).distance(b) < 2.0e-6);
        }
        let (ln, lf) = l.palm.unwrap();
        let (rn, rf) = r.palm.unwrap();
        assert!((-reflect(ln)).distance(rn) < 2.0e-6);
        assert!(reflect(lf).distance(rf) < 2.0e-6);
    }

    #[test]
    fn fixed_plane_fk_is_defined_at_both_poles() {
        let chain = chain(ArmSide::Left);
        for e in [
            0.0,
            1.0e-7,
            std::f32::consts::PI - 1.0e-7,
            std::f32::consts::PI,
        ] {
            let pose = state(1.2, e, -0.4, 0.7).forward(&chain).unwrap();
            assert!(pose.wrist.is_finite());
        }
    }
    #[test]
    fn radius_offset_roundtrips_and_wrist_hinges_do_not_move_the_wrist() {
        for side in [ArmSide::Left, ArmSide::Right] {
            let chain = chain(side);
            let input = ArmIkInput::from_chain(
                &chain,
                ArmIkTarget {
                    wrist: chain.rest.wrist.position,
                    elbow_pole: chain.rest.elbow.position,
                },
            );
            let upper = crate::shoulder::from_coordinates(
                &chain,
                crate::shoulder::ShoulderCoordinates {
                    plane: Some(0.8),
                    elevation: 0.9,
                    axial: -0.4,
                },
            )
            .unwrap();
            for flexion in [0.0, 0.8, 2.0] {
                for roll in [-1.5, -0.5, 0.0, 0.5, 1.5] {
                    let pose =
                        crate::skeleton::from_joints(input.skeleton_rest(), upper, flexion, roll)
                            .unwrap();
                    let inverse = crate::skeleton::joint_coordinates(input.skeleton_rest(), pose)
                        .unwrap_or_else(|| panic!("side={side:?} flexion={flexion} roll={roll}"));
                    assert!((inverse.x - flexion).abs() < 64.0 * f32::EPSILON);
                    assert!((inverse.y - roll).abs() < 64.0 * f32::EPSILON);
                }
            }
            let base = state(0.8, 0.9, -0.4, 0.8);
            let initial = base.forward(&chain).unwrap();
            for (flexion, deviation) in [(0.4, 0.0), (0.0, 0.2), (-0.4, -0.1)] {
                let mut moved = base;
                moved.angles[5] = flexion;
                moved.angles[6] = deviation;
                let pose = moved.forward(&chain).unwrap();
                assert_eq!(pose.wrist, initial.wrist);
                assert_eq!(pose.elbow, initial.elbow);
                assert!(
                    pose.palm.unwrap().0.distance(initial.palm.unwrap().0)
                        + pose.palm.unwrap().1.distance(initial.palm.unwrap().1)
                        > 0.01
                );
            }
        }
    }
}
