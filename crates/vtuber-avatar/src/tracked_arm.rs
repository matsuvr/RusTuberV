//! Pure adapter from engine-neutral tracking targets to the existing arm IK.
//!
//! This module never writes Transforms. Live integration must route its output
//! through the existing arm compositor, not register another bone writer.

use bevy::prelude::{Mat3, Quat, Vec3};
use vtuber_core::arm_tracking::{ArmTrackingTarget, HandFingerPose};

use crate::arm::rest_palm_normal;
use crate::arm::{ArmChainBinding, ArmIkTarget, ArmRestGeometry, FingerJointRestReferences};
use crate::arm_pose::{ResolvedBoneDelta, ResolvedFingerJointPose, ResolvedFingerPose};

/// Rotation that carries the canonical tracking basis into the solver's
/// model/rest basis while removing the current shoulder-parent motion.
///
/// `B` is the parent's rest rotation, `A` its current rotation, and `V` the
/// fixed rotation from the canonical tracking basis (its +Z points toward the
/// camera) into the model basis. Removing `A` exactly once prevents a torso
/// turn from being applied to the arm twice.
#[must_use]
pub fn tracking_to_rest_rotation(
    parent_rest: Quat,
    parent_current: Quat,
    view_to_model: Quat,
) -> Quat {
    parent_rest * parent_current.inverse() * view_to_model
}

/// Converts hand-local articulation to the rig's rest-relative finger rotations.
/// The palm frame is built from the same wrist/index/little rays as tracking.
/// No solved arm orientation is read: arm swing, shoulder motion and the
/// forearm rotation have already been removed by the observation's
/// hand-local frame, before its independent smoothing.
#[must_use]
pub fn observed_finger_deltas(
    chain: &ArmChainBinding,
    observed: HandFingerPose,
    weight: f32,
    rest_curl: f32,
) -> Option<ResolvedFingerPose> {
    if !weight.is_finite() || weight <= f32::EPSILON {
        return None;
    }
    let weight = weight.clamp(0.0, 1.0);
    let baseline = crate::arm_pose::signed_finger_curl(chain.side, rest_curl) * (1.0 - weight);
    let normal = rest_palm_normal(chain)?;
    let wrist = chain.rest.wrist.position;
    let index =
        crate::arm::finite_normalized(chain.finger_rest.index.proximal?.rest.position - wrist)?;
    let little =
        crate::arm::finite_normalized(chain.finger_rest.little.proximal?.rest.position - wrist)?;
    let forward = crate::arm::finite_normalized(index + little)?;
    let rest = &chain.finger_rest;
    let [index_curl, middle_curl, ring_curl, little_curl] = observed.fingers;
    let [index_spread, middle_spread, ring_spread, little_spread] = observed.spread;
    let [thumb_curl, thumb_ip] = observed.thumb;
    let four = |finger, curl, spread: f32| {
        three_joint_deltas(finger, curl, spread, forward, normal, weight, baseline)
    };
    Some(ResolvedFingerPose {
        thumb: ResolvedFingerJointPose {
            metacarpal: crate::thumb::cmc_delta(chain, observed.thumb_cmc, weight),
            ..thumb_deltas(
                &rest.thumb,
                [thumb_curl, thumb_ip],
                normal,
                weight,
                baseline,
            )
        },
        index: four(&rest.index, index_curl, index_spread),
        middle: four(&rest.middle, middle_curl, middle_spread),
        ring: four(&rest.ring, ring_curl, ring_spread),
        little: four(&rest.little, little_curl, little_spread),
    })
}

/// MCP/IP retain the authored rest basis. CMC is supplied independently by
/// the public two-axis model, never by redirecting MCP toward a landmark ray.
fn thumb_deltas(
    finger: &FingerJointRestReferences,
    [mcp, ip]: [f32; 2],
    normal: Vec3,
    weight: f32,
    baseline: f32,
) -> ResolvedFingerJointPose {
    ResolvedFingerJointPose {
        metacarpal: None,
        proximal: crate::arm_pose::resolve_finger_joint(
            finger.proximal,
            finger.distal,
            finger.metacarpal,
            mcp * weight + baseline,
            normal,
        ),
        intermediate: None,
        distal: observed_joint_bend(
            finger.distal,
            None,
            finger.proximal,
            ip,
            normal,
            weight,
            baseline,
        ),
    }
}

fn three_joint_deltas(
    finger: &FingerJointRestReferences,
    [mcp, pip, dip]: [f32; 3],
    spread: f32,
    forward: Vec3,
    normal: Vec3,
    weight: f32,
    baseline: f32,
) -> ResolvedFingerJointPose {
    let proximal = (|| {
        let joint = finger.proximal?;
        let segment = crate::arm::finite_normalized(
            finger.intermediate?.rest.position - joint.rest.position,
        )?;
        let rest_elevation = segment.dot(normal).clamp(-1.0, 1.0).asin();
        let curl = crate::arm_pose::resolve_finger_joint(
            finger.proximal,
            finger.intermediate,
            None,
            (mcp - rest_elevation) * weight + baseline,
            normal,
        )?;
        let opening = opening_delta(
            finger.proximal,
            finger.intermediate,
            spread,
            forward,
            normal,
            weight,
        )?;
        Some(ResolvedBoneDelta {
            entity: joint.entity,
            delta: opening.delta * curl.delta,
        })
    })();
    ResolvedFingerJointPose {
        metacarpal: None,
        proximal,
        intermediate: observed_joint_bend(
            finger.intermediate,
            finger.distal,
            finger.proximal,
            pip,
            normal,
            weight,
            baseline,
        ),
        distal: observed_joint_bend(
            finger.distal,
            None,
            finger.intermediate,
            dip,
            normal,
            weight,
            baseline,
        ),
    }
}

/// Subtract the authored bend where both adjacent segments exist. VRM has no
/// fingertip bone, so the terminal joint uses the preceding segment as its axis.
fn observed_joint_bend(
    joint: Option<crate::arm::FingerJointRestBinding>,
    next: Option<crate::arm::FingerJointRestBinding>,
    previous: Option<crate::arm::FingerJointRestBinding>,
    angle: f32,
    normal: Vec3,
    weight: f32,
    baseline: f32,
) -> Option<ResolvedBoneDelta> {
    let rest_angle = if let (Some(joint), Some(next), Some(previous)) = (joint, next, previous) {
        let incoming = crate::arm::finite_normalized(joint.rest.position - previous.rest.position)?;
        let outgoing = crate::arm::finite_normalized(next.rest.position - joint.rest.position)?;
        let axis = crate::arm::finite_normalized(incoming.cross(normal))?;
        axis.dot(incoming.cross(outgoing))
            .atan2(incoming.dot(outgoing))
    } else {
        0.0
    };
    crate::arm_pose::resolve_finger_joint(
        joint,
        next,
        previous,
        (angle - rest_angle) * weight + baseline,
        normal,
    )
}

/// Compare the same segment in the rest palm frame and rotate about its normal.
fn opening_delta(
    joint: Option<crate::arm::FingerJointRestBinding>,
    next: Option<crate::arm::FingerJointRestBinding>,
    spread: f32,
    forward: Vec3,
    normal: Vec3,
    weight: f32,
) -> Option<ResolvedBoneDelta> {
    let joint = joint?;
    let rest_ray = next?.rest.position - joint.rest.position;
    let in_plane = |ray: Vec3| crate::arm::finite_normalized(ray - normal * ray.dot(normal));
    let rest = in_plane(rest_ray)?;
    let reference = in_plane(forward)?;
    let rest_angle = normal.dot(reference.cross(rest)).atan2(reference.dot(rest));
    // Keep one authored coordinate branch while weight changes. Wrapping the
    // observed/rest difference through atan2 each frame makes +/-pi become
    // opposite rotations at partial weight, invalidating a continuous path.
    let angle = (-spread - rest_angle) * weight;
    let rotation = joint.rest.global_rotation;
    Some(ResolvedBoneDelta {
        entity: joint.entity,
        delta: normalized_finite(crate::skeleton::rest_delta(
            Quat::from_axis_angle(normal, angle),
            rotation,
        ))?,
    })
}

/// Normalizes a quaternion, or reports that it is not usable.
fn normalized_finite(value: Quat) -> Option<Quat> {
    (value.is_finite() && value.length_squared() > f32::EPSILON).then(|| value.normalize())
}

/// Converts subject-arm units into avatar rest-space positions.
///
/// tracking_to_rest is a UNIT rotation from the canonical tracking basis into
/// the solver's model/rest basis, including removal of the current shoulder
/// parent's motion. If B is the parent's rest rotation, A its current rotation,
/// and V maps the tracking view into the same model frame, use B * inverse(A) * V.
/// Feeding V alone when the torso turns applies torso motion twice.
///
/// Origin is the upper-arm bone, NOT the optional clavicle/shoulder bone.
/// This conversion adds neither the face's translation nor virtual-hand lag.
#[must_use]
pub fn tracked_arm_ik_target(
    rest: ArmRestGeometry,
    target: ArmTrackingTarget,
    tracking_to_rest: Quat,
) -> ArmIkTarget {
    let position = |[x, y, z]: [f32; 3]| {
        rest.upper_arm.position + (tracking_to_rest * Vec3::new(x, y, z)) * rest.total_arm_length
    };
    ArmIkTarget {
        wrist: position(target.wrist),
        elbow_pole: position(target.elbow_pole),
    }
}

/// Orthonormal palm frame; the normal and forward are independent observations
/// but their filter outputs need orthogonalization before matrix conversion.
pub(crate) fn palm_frame(normal: Vec3, forward: Vec3) -> Option<Quat> {
    let forward = forward.try_normalize()?;
    let across = forward.cross(normal).try_normalize()?;
    Some(Quat::from_mat3(&Mat3::from_cols(
        across,
        forward,
        across.cross(forward),
    )))
}

pub(crate) fn rest_palm_forward(chain: &ArmChainBinding) -> Option<Vec3> {
    let wrist = chain.rest.wrist.position;
    let index = (chain.finger_rest.index.proximal?.rest.position - wrist).try_normalize()?;
    let little = (chain.finger_rest.little.proximal?.rest.position - wrist).try_normalize()?;
    (index + little).try_normalize()
}

/// Apply a camera palm's normalized T-pose rotation to the authored hand.
/// Slightly slanted model T-poses retain their own rest geometry, as in the
/// VRM normalized-pose conversion, rather than biasing a half-turn's branch.
pub(crate) fn retarget_palm_frame(
    chain: &ArmChainBinding,
    normal: Vec3,
    forward: Vec3,
) -> Option<(Vec3, Vec3)> {
    let side_sign = if chain.side == crate::arm::ArmSide::Left {
        1.0
    } else {
        -1.0
    };
    let canonical = palm_frame(Vec3::Y * side_sign, Vec3::X * side_sign)?;
    let delta = palm_frame(normal, forward)? * canonical.inverse();
    Some((
        delta * rest_palm_normal(chain)?,
        delta * rest_palm_forward(chain)?,
    ))
}

/// SO(3) palm orientation error, scaled by authored palm length. Each
/// rotational axis has the same observational confidence; a narrow palm must
/// not make a half-turn around its longitudinal axis nearly unobservable.
/// The length keeps angular and Cartesian tasks in physical distance units.
pub(crate) fn palm_orientation_error(
    chain: &ArmChainBinding,
    current: (Vec3, Vec3),
    wanted: (Vec3, Vec3),
) -> Option<Vec3> {
    Some(palm_rotation_error(current, wanted)? * palm_length(chain)?)
}

/// Return palm-length-scaled rotation error on the logarithm branch nearest
/// the reference palm's error. This supplies continuous residuals for finite
/// differences; it does not compute a derivative. The shortest logarithm alone
/// flips sign at a half-turn, producing a spurious derivative near PI / h.
pub(crate) fn palm_orientation_error_near_reference(
    chain: &ArmChainBinding,
    current: (Vec3, Vec3),
    wanted: (Vec3, Vec3),
    reference_palm: (Vec3, Vec3),
) -> Option<Vec3> {
    let error = palm_rotation_error(current, wanted)?;
    let reference_error = palm_rotation_error(reference_palm, wanted)?;
    let error = error
        .try_normalize()
        .map(|axis| error - std::f32::consts::TAU * axis)
        .filter(|alternate| {
            alternate.distance_squared(reference_error) < error.distance_squared(reference_error)
        })
        .unwrap_or(error);
    Some(error * palm_length(chain)?)
}

fn palm_rotation_error(current: (Vec3, Vec3), wanted: (Vec3, Vec3)) -> Option<Vec3> {
    let mut rotation =
        (palm_frame(wanted.0, wanted.1)? * palm_frame(current.0, current.1)?.inverse()).normalize();
    if rotation.w < 0.0 {
        rotation = -rotation;
    }
    Some(rotation.to_scaled_axis())
}

fn palm_length(chain: &ArmChainBinding) -> Option<f32> {
    let wrist = chain.rest.wrist.position;
    let index = chain.finger_rest.index.proximal?.rest.position - wrist;
    let little = chain.finger_rest.little.proximal?.rest.position - wrist;
    Some(((index.length_squared() + little.length_squared()) * 0.5).sqrt())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use crate::arm::RestSpaceBonePose;
    use crate::arm::rest_palm_normal;
    use crate::arm_pose::ResolvedArmPose;
    use bevy::prelude::EulerRot;

    fn bone(position: Vec3) -> RestSpaceBonePose {
        RestSpaceBonePose {
            position,
            global_rotation: Quat::IDENTITY,
            local_rotation: Quat::IDENTITY,
        }
    }

    fn geometry(scale: f32) -> ArmRestGeometry {
        let origin = Vec3::new(0.2, 1.0, 0.0);
        ArmRestGeometry {
            shoulder: None,
            upper_arm: bone(origin),
            elbow: bone(origin + Vec3::X * (0.4 * scale)),
            wrist: bone(origin + Vec3::X * (0.7 * scale)),
            upper_arm_length: 0.4 * scale,
            forearm_length: 0.3 * scale,
            total_arm_length: 0.7 * scale,
        }
    }

    fn target() -> ArmTrackingTarget {
        ArmTrackingTarget {
            wrist: [0.4, -0.3, 0.5],
            elbow_pole: [0.7, -0.5, -0.1],
            palm_normal: None,
            palm_forward: None,
            fingers: None,
        }
    }

    #[test]
    fn half_turn_palm_has_a_unit_angular_derivative() {
        for side in [crate::arm::ArmSide::Left, crate::arm::ArmSide::Right] {
            let chain = crate::upper_limb::tests::chain(side);
            let normal = rest_palm_normal(&chain).unwrap();
            let forward = rest_palm_forward(&chain).unwrap();
            let wanted = (-normal, forward);
            let h = f32::EPSILON.cbrt();
            let sample = |angle| {
                let turn = Quat::from_axis_angle(forward, angle);
                palm_orientation_error_near_reference(
                    &chain,
                    (turn * normal, forward),
                    wanted,
                    (normal, forward),
                )
                .unwrap()
            };
            let wrist = chain.rest.wrist.position;
            let index = chain.finger_rest.index.proximal.unwrap().rest.position - wrist;
            let little = chain.finger_rest.little.proximal.unwrap().rest.position - wrist;
            let length = ((index.length_squared() + little.length_squared()) * 0.5).sqrt();
            let derivative = (sample(h) - sample(-h)) / (2.0 * h * length);
            assert!((derivative.length() - 1.0).abs() < 0.001, "{derivative:?}");
        }
    }

    #[test]
    fn palm_retargeting_rotates_rest_frame_on_both_sides() {
        for side in [crate::arm::ArmSide::Left, crate::arm::ArmSide::Right] {
            let sign = if side == crate::arm::ArmSide::Left {
                1.0
            } else {
                -1.0
            };
            for tilt in [-0.05, 0.0, 0.05] {
                let mut chain = crate::upper_limb::tests::chain(side);
                let wrist = chain.rest.wrist.position;
                let authored = Quat::from_euler(EulerRot::XYZ, tilt, -tilt * 0.5, tilt * 0.3);
                for finger in [&mut chain.finger_rest.index, &mut chain.finger_rest.little] {
                    let joint = finger.proximal.as_mut().unwrap();
                    joint.rest.position = wrist + authored * (joint.rest.position - wrist);
                }
                let rest = (
                    rest_palm_normal(&chain).unwrap(),
                    rest_palm_forward(&chain).unwrap(),
                );
                for angle in [
                    0.0,
                    0.7,
                    std::f32::consts::PI - 0.01,
                    std::f32::consts::PI + 0.01,
                ] {
                    let delta = Quat::from_axis_angle(Vec3::new(0.2, 0.8, -0.3).normalize(), angle);
                    let mapped =
                        retarget_palm_frame(&chain, delta * Vec3::Y * sign, delta * Vec3::X * sign)
                            .unwrap();
                    near(mapped.0, delta * rest.0);
                    near(mapped.1, delta * rest.1);
                    let normalized = (palm_frame(mapped.0, mapped.1).unwrap()
                        * palm_frame(rest.0, rest.1).unwrap().inverse())
                    .normalize();
                    assert!(normalized.dot(delta).abs() > 1.0 - 8.0 * f32::EPSILON);
                }
            }
        }
    }

    fn near(a: Vec3, b: Vec3) {
        assert!((a - b).length() < 1.0e-5, "{a:?} != {b:?}");
    }

    #[test]
    fn avatar_length_and_upper_arm_origin_determine_target() {
        let rest = geometry(2.0);
        let mapped = tracked_arm_ik_target(rest, target(), Quat::IDENTITY);
        near(
            mapped.wrist,
            rest.upper_arm.position + Vec3::new(0.4, -0.3, 0.5) * 1.4,
        );
        near(
            mapped.elbow_pole,
            rest.upper_arm.position + Vec3::new(0.7, -0.5, -0.1) * 1.4,
        );
    }

    fn chain() -> ArmChainBinding {
        let rest = geometry(1.0);
        ArmChainBinding {
            side: crate::arm::ArmSide::Left,
            shoulder: None,
            upper_arm: bevy::prelude::Entity::from_raw_u32(0).unwrap(),
            lower_arm: bevy::prelude::Entity::from_raw_u32(1).unwrap(),
            hand: bevy::prelude::Entity::from_raw_u32(2).unwrap(),
            fingers: crate::arm::FingerReferences::default(),
            finger_rest: crate::arm::FingerRestReferences::default(),
            rest,
            capabilities: crate::arm::ArmChainCapabilities::default(),
        }
    }

    fn finger_binding(position: Vec3) -> crate::arm::FingerJointRestBinding {
        crate::arm::FingerJointRestBinding {
            entity: bevy::prelude::Entity::from_raw_u32(3).unwrap(),
            rest: bone(position),
        }
    }

    /// The default test chain plus authored index/little-finger rest positions
    /// whose cross product points along +Y, i.e. a palm facing up.

    /// A chain whose fingers form straight +X chains, so a flexion of zero
    /// leaves every bone at its rest orientation and any non-zero flexion is
    /// visible as a real rotation. The four roots are spread across Z so the
    /// authored palm normal is well defined, as a real hand's is.
    fn articulated_chain() -> ArmChainBinding {
        let mut chain = chain();
        let wrist = chain.rest.wrist.position;
        let row = |dx: [f32; 4], z: f32| crate::arm::FingerJointRestReferences {
            metacarpal: None,
            proximal: Some(finger_binding(wrist + Vec3::new(dx[0], 0.0, z))),
            intermediate: Some(finger_binding(wrist + Vec3::new(dx[1], 0.0, z))),
            distal: Some(finger_binding(wrist + Vec3::new(dx[2], 0.0, z))),
        };
        chain.finger_rest.index = row([0.02, 0.04, 0.055, 0.065], 0.004);
        chain.finger_rest.middle = row([0.02, 0.045, 0.062, 0.074], 0.001);
        chain.finger_rest.ring = row([0.02, 0.042, 0.057, 0.067], -0.002);
        chain.finger_rest.little = row([0.02, 0.036, 0.048, 0.056], -0.005);
        // The thumb rests half way out from the fingers, in the +Z half plane,
        // so the model's own rest spread is not the same as a flat hand's. Its
        // three segments are collinear, because this rig's thumb lies in the
        // palm plane: an observation of it must read as straight, and an
        // in-plane bend has no anatomical axis to take a sign from.
        chain.finger_rest.thumb = crate::arm::FingerJointRestReferences {
            metacarpal: Some(finger_binding(wrist + Vec3::new(0.012, 0.0, 0.022))),
            proximal: Some(finger_binding(wrist + Vec3::new(0.03, 0.0, 0.035))),
            intermediate: None,
            distal: Some(finger_binding(wrist + Vec3::new(0.048, 0.0, 0.048))),
        };
        chain
    }

    // Reconstruct the hierarchy the compositor writes, including the clavicle.
    // Checking only the IK solution missed the later double shoulder rotation.

    /// A solved pose for the articulated chain, with the observation's fingers
    /// applied at `weight`.
    fn resolved_fingers(
        chain: &ArmChainBinding,
        fingers: HandFingerPose,
        weight: f32,
    ) -> ResolvedFingerPose {
        observed_finger_deltas(chain, fingers, weight, 0.0).expect("finger deltas")
    }

    /// Make actual tracking input from a rig. Tips continue the terminal rest
    /// segment, and the thumb's model-placed CMC is written to a position no
    /// joint derives from, so a test that ignores it proves the point.
    fn rest_observation(
        chain: &ArmChainBinding,
        arm_rotation: Quat,
        hand_rotation: Quat,
    ) -> ArmTrackingTarget {
        use vtuber_core::arm_tracking::{ArmLandmarks, HandWorldLandmarks, PoseWorldLandmark};
        let point = |v: Vec3| PoseWorldLandmark {
            meters: [v.x, -v.y, -v.z],
            visibility: Some(1.0),
            presence: Some(1.0),
        };
        let wrist = chain.rest.wrist.position;
        let mut landmarks = [point(Vec3::ZERO); 21];
        let finger =
            |bone: Option<crate::arm::FingerJointRestBinding>| bone.unwrap().rest.position - wrist;
        for (start, row) in [
            (5, chain.finger_rest.index),
            (9, chain.finger_rest.middle),
            (13, chain.finger_rest.ring),
            (17, chain.finger_rest.little),
        ] {
            let [a, b, c] = [
                finger(row.proximal),
                finger(row.intermediate),
                finger(row.distal),
            ];
            for (i, position) in [a, b, c, c + (c - b)].into_iter().enumerate() {
                landmarks[start + i] = point(hand_rotation * position);
            }
        }
        let thumb = chain.finger_rest.thumb;
        let [a, b, c] = [
            finger(thumb.metacarpal),
            finger(thumb.proximal),
            finger(thumb.distal),
        ];
        for (i, position) in [a, b, c, c + (c - b)].into_iter().enumerate() {
            landmarks[1 + i] = point(hand_rotation * position);
        }
        // The Hand Landmarker places the CMC with its hand model, not by
        // detecting it. Park it away from the rig so any ray measured from it
        // would be visibly wrong.
        landmarks[1] = point(hand_rotation * (wrist + Vec3::new(0.09, -0.06, 0.11)));
        let arm = ArmLandmarks {
            shoulder: point(Vec3::ZERO),
            elbow: point(
                arm_rotation * (chain.rest.elbow.position - chain.rest.upper_arm.position),
            ),
            wrist: point(arm_rotation * (wrist - chain.rest.upper_arm.position)),
            hand: Some(HandWorldLandmarks {
                landmarks,
                handedness_score: None,
            }),
        };
        vtuber_tracking::arm_tracking::retarget_arm_landmarks(
            arm,
            vtuber_tracking::arm_tracking::measure_arm_reference(arm).unwrap(),
        )
    }

    fn straight_fingers() -> HandFingerPose {
        rest_observation(&articulated_chain(), Quat::IDENTITY, Quat::IDENTITY)
            .fingers
            .unwrap()
    }

    #[test]
    fn finger_opening_keeps_its_rest_branch_at_partial_weight() {
        let joint = Some(finger_binding(Vec3::ZERO));
        let next = Some(finger_binding(-Vec3::X));
        let a = opening_delta(joint, next, -0.001, Vec3::X, Vec3::Z, 0.5).unwrap();
        let b = opening_delta(joint, next, 0.001, Vec3::X, Vec3::Z, 0.5).unwrap();
        assert!(a.delta.angle_between(b.delta) < 0.005);
        let full = opening_delta(joint, next, 0.0, Vec3::X, Vec3::Z, 1.0).unwrap();
        assert!((full.delta * -Vec3::X).distance(Vec3::X) < 1.0e-6);
    }

    #[test]
    fn a_straight_hand_leaves_every_finger_at_rest() {
        let chain = articulated_chain();
        let pose = resolved_fingers(&chain, straight_fingers(), 1.0);
        // The thumb base is never driven from an observation, so it keeps the
        // rig's own rest pose just like the four fingers.
        assert_eq!(pose.thumb.metacarpal.unwrap().delta, Quat::IDENTITY);
        assert!(pose.thumb.intermediate.is_none());
        for finger in [
            &pose.index,
            &pose.middle,
            &pose.ring,
            &pose.little,
            &pose.thumb,
        ] {
            for joint in [
                finger.metacarpal,
                finger.proximal,
                finger.intermediate,
                finger.distal,
            ]
            .into_iter()
            .flatten()
            {
                let delta = joint;
                assert!(
                    delta.delta.angle_between(Quat::IDENTITY) < 1.0e-5,
                    "a straight finger must stay at its rest pose: {:?}",
                    delta.delta
                );
            }
        }
    }

    #[test]
    fn each_finger_bends_by_its_own_observed_amount() {
        let chain = articulated_chain();
        let mut observed = straight_fingers();
        // A peace sign: the index and middle stay up, the ring and little fold.
        observed.fingers[0] = [0.0; 3];
        observed.fingers[1] = [0.0; 3];
        observed.fingers[2] = [1.0, 1.2, 0.6];
        observed.fingers[3] = [0.8, 0.9, 0.5];
        let pose = resolved_fingers(&chain, observed, 1.0);

        for finger in [&pose.index, &pose.middle] {
            for joint in [finger.proximal, finger.intermediate, finger.distal] {
                let delta = joint.expect("the chain has every joint");
                assert!(
                    delta.delta.angle_between(Quat::IDENTITY) < 1.0e-5,
                    "an extended finger must not be curled"
                );
            }
        }
        // The folded fingers differ from each other, so this is not one shared
        // curl amount applied to all of them.
        let ring = pose.ring.proximal.expect("ring proximal");
        let little = pose.little.proximal.expect("little proximal");
        assert!(ring.delta.angle_between(Quat::IDENTITY) > 0.5);
        assert!(little.delta.angle_between(Quat::IDENTITY) > 0.2);
        assert!(
            (ring.delta.angle_between(Quat::IDENTITY) - little.delta.angle_between(Quat::IDENTITY))
                .abs()
                > 0.05,
            "each finger must keep its own curl"
        );
    }

    #[test]
    fn cmc_uses_the_two_source_axes_and_preserves_authored_rest() {
        let mut chain = articulated_chain();
        chain.side = crate::arm::ArmSide::Right;
        let wrist = chain.rest.wrist.position;
        chain
            .finger_rest
            .index
            .proximal
            .as_mut()
            .unwrap()
            .rest
            .position = wrist + Vec3::new(0.022178, -0.080917, 0.010979);
        chain
            .finger_rest
            .little
            .proximal
            .as_mut()
            .unwrap()
            .rest
            .position = wrist + Vec3::new(-0.019501, -0.071168, -0.003387);
        let base = chain.finger_rest.thumb.metacarpal.as_mut().unwrap();
        base.rest.global_rotation = Quat::from_euler(EulerRot::XYZ, 0.3, -0.4, 0.6);
        let origin = base.rest.position;
        let rest = base.rest.global_rotation;
        chain
            .finger_rest
            .thumb
            .proximal
            .as_mut()
            .unwrap()
            .rest
            .position = origin + Vec3::new(0.0165, -0.0292, -0.0127);
        for (flex, abduct) in [
            (0.0, 0.0),
            (0.4, 0.0),
            (0.0, -0.3),
            (0.4, -0.3),
            (2.0, -2.0),
        ] {
            let delta = crate::thumb::cmc_delta(&chain, [flex, abduct], 1.0)
                .unwrap()
                .delta;
            let expected = Quat::from_axis_angle(
                Vec3::new(-0.042399, -0.665286, 0.745384).normalize(),
                flex.clamp(-0.78, 0.7),
            ) * Quat::from_axis_angle(
                Vec3::new(0.495557, 0.731736, 0.467959).normalize(),
                abduct.clamp(-0.5, 0.78),
            );
            assert!((rest * delta * rest.inverse()).angle_between(expected) < 1.0e-3);
            let mut mirrored = chain;
            mirrored.side = crate::arm::ArmSide::Left;
            let reflect_point = |v: Vec3| Vec3::new(-v.x, v.y, v.z);
            let reflect_rotation = |q: Quat| Quat::from_xyzw(q.x, -q.y, -q.z, q.w);
            mirrored.rest.wrist.position = reflect_point(wrist);
            for bone in [
                &mut mirrored.finger_rest.index.proximal,
                &mut mirrored.finger_rest.little.proximal,
                &mut mirrored.finger_rest.thumb.metacarpal,
                &mut mirrored.finger_rest.thumb.proximal,
            ]
            .into_iter()
            .flatten()
            {
                bone.rest.position = reflect_point(bone.rest.position);
                bone.rest.global_rotation = reflect_rotation(bone.rest.global_rotation);
            }
            let reflected = crate::thumb::cmc_delta(&mirrored, [flex, abduct], 1.0)
                .unwrap()
                .delta;
            assert!(reflected.angle_between(reflect_rotation(delta)) < 1.0e-3);
            assert!(
                crate::thumb::cmc_delta(&chain, [flex, abduct], 0.0)
                    .unwrap()
                    .delta
                    .angle_between(Quat::IDENTITY)
                    < 1.0e-3
            );
        }
        // Check the source model's coordinate signs as physical motion,
        // not only equality to the source quaternion formula. Opposition
        // moves the right thumb ulnarly and volarly from its authored rest.
        let shaft = Vec3::new(0.0165, -0.0292, -0.0127);
        let opposed =
            crate::thumb::cmc_delta(&chain, [-25.0_f32.to_radians(), 20.0_f32.to_radians()], 1.0)
                .unwrap();
        let moved = rest * opposed.delta * rest.inverse() * shaft;
        assert!(moved.x < shaft.x);
        let palm = rest_palm_normal(&chain).unwrap();
        assert!(moved.dot(palm) > shaft.dot(palm));
    }

    #[test]
    fn the_thumb_is_not_folded_by_a_four_finger_axis() {
        // The thumb's rest chain runs diagonally and its joints do not share the
        // four fingers' geometry, so its flexion must still resolve to a real
        // rotation on its own bones rather than a missing or identity delta.
        let chain = articulated_chain();
        let mut observed = straight_fingers();
        observed.thumb = [0.5, 0.7];
        let pose = resolved_fingers(&chain, observed, 1.0);
        let proximal = pose.thumb.proximal.expect("thumb proximal");
        let distal = pose.thumb.distal.expect("thumb distal");
        assert!(proximal.delta.angle_between(Quat::IDENTITY) > 0.1);
        assert!(distal.delta.angle_between(Quat::IDENTITY) > 0.1);
        assert!(
            pose.thumb.intermediate.is_none(),
            "VRM has no thumb intermediate"
        );
    }

    #[test]
    fn thumb_spread_cannot_replace_the_authored_base_direction() {
        let mut chain = articulated_chain();
        // Non-flat thumb like Sapphy: fixed poses must not flatten its rest
        // elevation or apply their 35–70 degree opening at the MCP.
        chain
            .finger_rest
            .thumb
            .proximal
            .as_mut()
            .unwrap()
            .rest
            .position
            .y -= 0.01;
        chain
            .finger_rest
            .thumb
            .distal
            .as_mut()
            .unwrap()
            .rest
            .position
            .y -= 0.016;
        let mut observed = straight_fingers();
        observed.thumb = [0.0, 0.0];
        for spread in [0.35, 0.6, 0.96, 1.22] {
            observed.thumb_spread = spread;
            let pose = resolved_fingers(&chain, observed, 1.0);
            assert!(
                pose.thumb
                    .metacarpal
                    .unwrap()
                    .delta
                    .angle_between(Quat::IDENTITY)
                    < 1.0e-3
            );
            assert!(
                pose.thumb
                    .proximal
                    .unwrap()
                    .delta
                    .angle_between(Quat::IDENTITY)
                    < 1e-6
            );
            assert!(
                pose.thumb
                    .distal
                    .unwrap()
                    .delta
                    .angle_between(Quat::IDENTITY)
                    < 1e-6
            );
        }
    }

    #[test]
    fn tracking_restores_the_thumb_base_after_a_default_pose_curl() {
        use bevy::prelude::*;

        let mut chain = articulated_chain();
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_systems(PostUpdate, crate::arm_pose::apply_default_arm_pose);
        let root = app
            .world_mut()
            .spawn((
                crate::ActiveAvatar,
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();
        let spawn = |app: &mut App, parent, position| {
            app.world_mut()
                .spawn((
                    Transform::from_translation(position),
                    bevy_vrm1::prelude::RestTransform(Transform::from_translation(position)),
                    GlobalTransform::IDENTITY,
                    ChildOf(parent),
                ))
                .id()
        };
        let upper = spawn(&mut app, root, Vec3::ZERO);
        let lower = spawn(&mut app, upper, Vec3::ZERO);
        let hand = spawn(&mut app, lower, chain.rest.wrist.position);
        let mut thumb = chain.finger_rest.thumb;
        let cmc = thumb.metacarpal.unwrap().rest.position;
        let mcp = thumb.proximal.unwrap().rest.position;
        let ip = thumb.distal.unwrap().rest.position;
        let metacarpal = spawn(&mut app, hand, cmc - chain.rest.wrist.position);
        let proximal = spawn(&mut app, metacarpal, mcp - cmc);
        let distal = spawn(&mut app, proximal, ip - mcp);
        thumb.metacarpal.as_mut().unwrap().entity = metacarpal;
        thumb.proximal.as_mut().unwrap().entity = proximal;
        thumb.distal.as_mut().unwrap().entity = distal;
        chain.finger_rest.thumb = thumb;

        let generation = crate::AvatarGeneration(1);
        let pose = ResolvedArmPose {
            upper_arm: upper,
            lower_arm: lower,
            upper_arm_delta: Quat::IDENTITY,
            lower_arm_delta: Quat::IDENTITY,
            hand: None,
            shoulder: None,
            fingers: ResolvedFingerPose {
                thumb: ResolvedFingerJointPose {
                    metacarpal: Some(ResolvedBoneDelta {
                        entity: metacarpal,
                        delta: Quat::from_rotation_y(0.2),
                    }),
                    ..default()
                },
                ..default()
            },
        };
        app.world_mut().entity_mut(root).insert((
            crate::AvatarBinding::head_only(root, root, generation),
            crate::DynamicArmTargets {
                generation: Some(generation),
                source_seq: None,
                left: Some(pose),
                right: None,
            },
        ));
        app.update();
        assert!(
            app.world()
                .get::<GlobalTransform>(proximal)
                .unwrap()
                .translation()
                .distance(mcp)
                > 0.001
        );

        let tracked = ResolvedArmPose {
            fingers: ResolvedFingerPose {
                thumb: resolved_fingers(&chain, straight_fingers(), 1.0).thumb,
                ..default()
            },
            ..pose
        };
        app.world_mut()
            .entity_mut(root)
            .insert(crate::DynamicArmTargets {
                generation: Some(generation),
                left: Some(tracked),
                ..default()
            });
        app.update();
        near(
            app.world()
                .get::<GlobalTransform>(proximal)
                .unwrap()
                .translation(),
            mcp,
        );
        assert_eq!(
            app.world().get::<Transform>(metacarpal).unwrap().rotation,
            Quat::IDENTITY
        );
    }

    #[test]
    fn thumb_motion_never_reaches_the_thumb_base() {
        // A real thumb's base barely moves when the thumb moves: the visible
        // articulation is at the two joints. The observed base came from the
        // Hand Landmarker's model-placed CMC, which travels with the whole
        // thumb, so driving the metacarpal from it made the thumb's root swing
        // and pushed the tip past straight. The base must stay at rest for any
        // observation.
        let chain = articulated_chain();
        for (curl, spread) in [
            (0.0, 0.0),
            (0.6, 0.0),
            (0.0, 0.5),
            (0.6, 0.5),
            (1.2, -0.5),
            (0.3, 0.9),
        ] {
            let mut observed = straight_fingers();
            observed.thumb = [curl, curl];
            observed.thumb_spread += spread;
            let pose = resolved_fingers(&chain, observed, 1.0);
            assert!(
                pose.thumb.metacarpal.unwrap().delta == Quat::IDENTITY,
                "the thumb base must stay neutral: curl {curl} spread {spread}"
            );
            let applied = pose
                .thumb
                .proximal
                .expect("the thumb has a knuckle")
                .delta
                .angle_between(Quat::IDENTITY);
            assert!(
                applied < 3.15,
                "the knuckle stays within one turn: {applied}"
            );
        }
    }

    #[test]
    fn thumb_opening_is_not_reassigned_to_the_mcp_joint() {
        // Keep base opening as a recognition feature. Its absence from the
        // rig's allowed motion must not move the MCP sideways in its place.
        let chain = articulated_chain();
        let normal = rest_palm_normal(&chain).unwrap();
        let origin = chain.finger_rest.thumb.metacarpal.unwrap().rest.position;
        let baseline = rest_observation(&chain, Quat::IDENTITY, Quat::IDENTITY);

        for (opening, elevation) in [(0.35, 0.0), (0.0, 0.6), (0.35, 0.6), (-0.35, -0.6)] {
            // Keep the CMC, palm, arm and four fingers fixed, and rotate the
            // whole thumb about the CMC to separate opening from flexion.
            let mut observed_chain = chain;
            let segment = observed_chain
                .finger_rest
                .thumb
                .proximal
                .unwrap()
                .rest
                .position
                - origin;
            let axis = segment.normalize().cross(normal).normalize();
            let movement =
                Quat::from_axis_angle(normal, opening) * Quat::from_axis_angle(axis, elevation);
            for joint in [
                &mut observed_chain.finger_rest.thumb.proximal,
                &mut observed_chain.finger_rest.thumb.distal,
            ] {
                let joint = joint.as_mut().unwrap();
                joint.rest.position = origin + movement * (joint.rest.position - origin);
            }
            let target = rest_observation(&observed_chain, Quat::IDENTITY, Quat::IDENTITY);
            let observed = target.fingers.expect("the thumb is observed");
            near(Vec3::from(target.wrist), Vec3::from(baseline.wrist));
            near(
                Vec3::from(target.palm_normal.unwrap()),
                Vec3::from(baseline.palm_normal.unwrap()),
            );

            for weight in [0.0, 0.5, 1.0] {
                let fingers = observed_finger_deltas(&chain, observed, weight, 0.0);
                if weight == 0.0 {
                    assert!(fingers.is_none());
                }
                let fingers =
                    fingers.unwrap_or_else(|| crate::arm_pose::resolve_finger_pose(&chain, 0.0));
                if weight > 0.0 {
                    // The observed path drives the two real joints only. The
                    // virtual path is a separate pose and is not under test.
                    assert!(
                        fingers.thumb.metacarpal.unwrap().delta == Quat::IDENTITY,
                        "an oblique thumb must leave its base neutral"
                    );
                    let moved = fingers
                        .thumb
                        .proximal
                        .expect("the thumb has a knuckle")
                        .delta
                        .angle_between(Quat::IDENTITY);
                    if elevation == 0.0 {
                        assert!(moved < 1.0e-5, "opening must not rotate the MCP: {moved}");
                    } else {
                        assert!(
                            moved > 1.0e-3,
                            "flexion must still reach the knuckle: {moved}"
                        );
                    }
                }
                // The four fingers are untouched by any of this.
                for finger in [
                    &fingers.index,
                    &fingers.middle,
                    &fingers.ring,
                    &fingers.little,
                ] {
                    for joint in [finger.proximal, finger.intermediate, finger.distal] {
                        assert!(joint.unwrap().delta.angle_between(Quat::IDENTITY) < 1.0e-5);
                    }
                }
            }
        }
    }

    #[test]
    fn local_fingers_ignore_arm_swing_and_forearm_pronation() {
        let chain = articulated_chain();
        let neutral = observed_finger_deltas(&chain, straight_fingers(), 1.0, 0.0).unwrap();
        for (arm_rotation, twist) in [
            (
                Quat::from_rotation_y(0.7) * Quat::from_rotation_z(-0.4),
                0.0,
            ),
            (Quat::IDENTITY, 1.0),
            (Quat::from_rotation_z(0.45), -0.8),
        ] {
            let target = rest_observation(
                &chain,
                arm_rotation,
                arm_rotation * Quat::from_rotation_x(twist),
            );
            let fingers =
                observed_finger_deltas(&chain, target.fingers.unwrap(), 1.0, 0.0).unwrap();
            for (a, b) in [
                (&fingers.thumb, &neutral.thumb),
                (&fingers.index, &neutral.index),
                (&fingers.middle, &neutral.middle),
                (&fingers.ring, &neutral.ring),
                (&fingers.little, &neutral.little),
            ] {
                for (a, b) in [a.metacarpal, a.proximal, a.intermediate, a.distal]
                    .into_iter()
                    .zip([b.metacarpal, b.proximal, b.intermediate, b.distal])
                {
                    match (a, b) {
                        (Some(a), Some(b)) => assert!(a.delta.angle_between(b.delta) < 1.0e-3),
                        (None, None) => (),
                        _ => panic!("finger binding changed"),
                    }
                }
            }
        }
    }

    #[test]
    fn selected_fist_curls_into_both_palms_in_a_palms_down_t_pose() {
        use vtuber_core::arm_tracking::{
            ArmLandmarks, HandWorldLandmarks, PoseArmFrame, PoseArmObservation, PoseWorldLandmark,
        };
        use vtuber_core::{FrameSeq, MonoTimeNs};
        use vtuber_tracking::arm_tracking::{
            ArmTrackingProfile, ArmTrackingState, step_arm_tracking,
        };

        let point = |v: Vec3| PoseWorldLandmark {
            meters: [v.x, -v.y, -v.z],
            visibility: Some(1.0),
            presence: Some(1.0),
        };
        let mut state = ArmTrackingState::new();
        let profile = ArmTrackingProfile::default();
        for reflected in [false, true] {
            let sign = if reflected { -1.0 } else { 1.0 };
            let mut chain = articulated_chain();
            chain.side = if reflected {
                crate::arm::ArmSide::Right
            } else {
                crate::arm::ArmSide::Left
            };
            for bone in [
                &mut chain.rest.upper_arm,
                &mut chain.rest.elbow,
                &mut chain.rest.wrist,
            ] {
                bone.position.x *= sign;
            }
            for row in [
                &mut chain.finger_rest.thumb,
                &mut chain.finger_rest.index,
                &mut chain.finger_rest.middle,
                &mut chain.finger_rest.ring,
                &mut chain.finger_rest.little,
            ] {
                for joint in [
                    &mut row.metacarpal,
                    &mut row.proximal,
                    &mut row.intermediate,
                    &mut row.distal,
                ]
                .into_iter()
                .flatten()
                {
                    joint.rest.position.x *= sign;
                }
            }
            let wrist = chain.rest.wrist.position;
            let mut landmarks = [point(wrist); 21];
            for (start, row) in [
                (5, chain.finger_rest.index),
                (9, chain.finger_rest.middle),
                (13, chain.finger_rest.ring),
                (17, chain.finger_rest.little),
            ] {
                let mut position = row.proximal.unwrap().rest.position;
                landmarks[start] = point(position);
                // The left hand points +X with thumb +Z in the VRM T-pose.
                // Both palms face DOWN (-Y), regardless of the cross-product sign.
                for (step, heading) in [1.4_f32, 2.9, 3.8].into_iter().enumerate() {
                    position += Vec3::new(sign * heading.cos(), -heading.sin(), 0.0) * 0.02;
                    landmarks[start + step + 1] = point(position);
                }
            }
            let thumb = chain.finger_rest.thumb;
            let a = thumb.proximal.unwrap().rest.position;
            let b = thumb.distal.unwrap().rest.position;
            for (i, position) in [thumb.metacarpal.unwrap().rest.position, a, b, b + (b - a)]
                .into_iter()
                .enumerate()
            {
                landmarks[1 + i] = point(position);
            }
            let arm = ArmLandmarks {
                shoulder: point(chain.rest.upper_arm.position),
                elbow: point(chain.rest.elbow.position),
                wrist: point(wrist),
                hand: Some(HandWorldLandmarks {
                    landmarks,
                    handedness_score: None,
                }),
            };
            state.reset();
            let mut control = None;
            for seq in 1..=90 {
                let now = MonoTimeNs(seq * 33_333_333);
                let frame = PoseArmFrame {
                    source_seq: FrameSeq(seq),
                    captured_at: now,
                    inference_finished_at: now,
                    observation: Some(PoseArmObservation {
                        hips: None,
                        left: arm,
                        right: arm,
                    }),
                };
                (state, control) = step_arm_tracking(&state, Some(&frame), now, &profile);
            }
            let targets = control.unwrap().targets;
            let target = if reflected {
                targets.right
            } else {
                targets.left
            }
            .unwrap();
            let pose = observed_finger_deltas(&chain, target.fingers.unwrap(), 1.0, 0.0).unwrap();
            let joint = chain.finger_rest.index.proximal.unwrap();
            let ray = (chain.finger_rest.index.intermediate.unwrap().rest.position
                - joint.rest.position)
                .normalize();
            let rotation = joint.rest.global_rotation;
            let bent = rotation * pose.index.proximal.unwrap().delta * rotation.inverse() * ray;
            assert!(
                bent.y < -0.5,
                "a selected fist must curl palmward, not backward: {bent:?}"
            );
            assert!(
                pose.thumb
                    .metacarpal
                    .unwrap()
                    .delta
                    .angle_between(Quat::IDENTITY)
                    > 0.1
            );
        }
    }

    #[test]
    fn both_hands_bend_palmward_with_non_identity_rest_axes() {
        for reflected in [false, true] {
            let mut chain = articulated_chain();
            let reflect = |v: Vec3| {
                if reflected {
                    Vec3::new(-v.x, v.y, v.z)
                } else {
                    v
                }
            };
            chain.side = if reflected {
                crate::arm::ArmSide::Right
            } else {
                crate::arm::ArmSide::Left
            };
            for bone in [
                &mut chain.rest.upper_arm,
                &mut chain.rest.elbow,
                &mut chain.rest.wrist,
            ] {
                bone.position = reflect(bone.position);
            }
            for row in [
                &mut chain.finger_rest.thumb,
                &mut chain.finger_rest.index,
                &mut chain.finger_rest.middle,
                &mut chain.finger_rest.ring,
                &mut chain.finger_rest.little,
            ] {
                let mut parent_rotation = chain.rest.wrist.global_rotation;
                for (i, joint) in [
                    &mut row.metacarpal,
                    &mut row.proximal,
                    &mut row.intermediate,
                    &mut row.distal,
                ]
                .into_iter()
                .enumerate()
                {
                    if let Some(joint) = joint {
                        joint.rest.position = reflect(joint.rest.position);
                        joint.rest.global_rotation = Quat::from_rotation_x(0.4 + i as f32 * 0.3)
                            * Quat::from_rotation_z(-0.6);
                        joint.rest.local_rotation =
                            parent_rotation.inverse() * joint.rest.global_rotation;
                        parent_rotation = joint.rest.global_rotation;
                    }
                }
            }
            let mut observed = rest_observation(&chain, Quat::IDENTITY, Quat::IDENTITY)
                .fingers
                .unwrap();
            let normal = rest_palm_normal(&chain).unwrap();
            let curl = 0.4 * normal.dot(Vec3::Y).signum();
            observed.fingers = [[curl; 3]; 4];
            observed.thumb = [curl; 2];
            let pose = observed_finger_deltas(&chain, observed, 1.0, 0.0).unwrap();
            for (row, deltas) in [
                (chain.finger_rest.thumb, pose.thumb),
                (chain.finger_rest.index, pose.index),
                (chain.finger_rest.middle, pose.middle),
                (chain.finger_rest.ring, pose.ring),
                (chain.finger_rest.little, pose.little),
            ] {
                for (joint, next, previous, delta) in [
                    (
                        row.proximal,
                        row.intermediate.or(row.distal),
                        row.metacarpal,
                        deltas.proximal,
                    ),
                    (
                        row.intermediate,
                        row.distal,
                        row.proximal,
                        deltas.intermediate,
                    ),
                    (
                        row.distal,
                        None,
                        row.intermediate.or(row.proximal),
                        deltas.distal,
                    ),
                ] {
                    let Some(joint) = joint else {
                        continue;
                    };
                    let ray = next
                        .map(|next| next.rest.position - joint.rest.position)
                        .unwrap_or_else(|| joint.rest.position - previous.unwrap().rest.position)
                        .normalize();
                    let rotation = joint.rest.global_rotation;
                    let bent = rotation * delta.unwrap().delta * rotation.inverse() * ray;
                    assert!(
                        bent.dot(Vec3::Y) > 0.3,
                        "both hands must bend toward the physical palm: {bent:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn proximal_spread_and_curl_reach_independent_rest_relative_directions() {
        let chain = articulated_chain();
        let joint = chain.finger_rest.index.proximal.unwrap();
        let ray = (chain.finger_rest.index.intermediate.unwrap().rest.position
            - joint.rest.position)
            .normalize();
        let normal = rest_palm_normal(&chain).unwrap();
        for (spread, curl) in [(0.35, 0.0), (0.0, 0.6), (0.35, 0.6)] {
            let mut observed = straight_fingers();
            observed.spread[0] += spread;
            observed.fingers[0][0] = curl;
            let delta = observed_finger_deltas(&chain, observed, 1.0, 0.0)
                .unwrap()
                .index
                .proximal
                .unwrap()
                .delta;
            let expected =
                Quat::from_axis_angle(normal, -spread) * ray * curl.cos() + normal * curl.sin();
            let rotation = joint.rest.global_rotation;
            near(rotation * delta * rotation.inverse() * ray, expected);
        }
    }

    #[test]
    fn loss_returns_finger_hinges_to_the_configured_curl_without_moving_cmc() {
        let chain = articulated_chain();
        let baseline = crate::arm_pose::resolve_finger_pose(&chain, 0.3);
        let observed = straight_fingers();
        let near_loss = observed_finger_deltas(&chain, observed, 1.0e-5, 0.3).unwrap();
        for (a, b) in [
            (baseline.index.proximal, near_loss.index.proximal),
            (baseline.index.intermediate, near_loss.index.intermediate),
            (baseline.index.distal, near_loss.index.distal),
            (baseline.thumb.proximal, near_loss.thumb.proximal),
            (baseline.thumb.distal, near_loss.thumb.distal),
        ] {
            let (a, b) = (a.unwrap().delta, b.unwrap().delta);
            assert!((a * Vec3::X).distance(b * Vec3::X) < 1.0e-5);
            assert!((a * Vec3::Y).distance(b * Vec3::Y) < 1.0e-5);
        }
        assert_eq!(near_loss.thumb.metacarpal.unwrap().delta, Quat::IDENTITY);
        assert_eq!(
            observed_finger_deltas(&chain, observed, 1.0, 0.0),
            observed_finger_deltas(&chain, observed, 1.0, 0.3)
        );
    }

    #[test]
    fn default_and_lost_fingers_curl_palmward_on_both_sides() {
        for reflected in [false, true] {
            let mut chain = articulated_chain();
            if reflected {
                chain.side = crate::arm::ArmSide::Right;
                for bone in [
                    &mut chain.rest.upper_arm,
                    &mut chain.rest.elbow,
                    &mut chain.rest.wrist,
                ] {
                    bone.position.x = -bone.position.x;
                }
                for row in [
                    &mut chain.finger_rest.thumb,
                    &mut chain.finger_rest.index,
                    &mut chain.finger_rest.middle,
                    &mut chain.finger_rest.ring,
                    &mut chain.finger_rest.little,
                ] {
                    for joint in [
                        &mut row.metacarpal,
                        &mut row.proximal,
                        &mut row.intermediate,
                        &mut row.distal,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        joint.rest.position.x = -joint.rest.position.x;
                    }
                }
            }
            let joint = chain.finger_rest.index.proximal.unwrap();
            let ray = (chain.finger_rest.index.intermediate.unwrap().rest.position
                - joint.rest.position)
                .normalize();
            for delta in [
                crate::arm_pose::resolve_finger_pose(&chain, 0.3)
                    .index
                    .proximal
                    .unwrap()
                    .delta,
                observed_finger_deltas(&chain, straight_fingers(), 1.0e-5, 0.3)
                    .unwrap()
                    .index
                    .proximal
                    .unwrap()
                    .delta,
            ] {
                let bent =
                    joint.rest.global_rotation * delta * joint.rest.global_rotation.inverse() * ray;
                assert!(bent.y < -0.01, "default finger bent backward: {bent:?}");
            }
        }
    }

    #[test]
    fn the_finger_channel_weight_scales_every_joint() {
        let chain = articulated_chain();
        let mut observed = straight_fingers();
        observed.fingers[0] = [0.6, 0.8, 0.4];
        let full = resolved_fingers(&chain, observed, 1.0);
        let half = resolved_fingers(&chain, observed, 0.5);
        for (a, b) in [
            (full.index.proximal, half.index.proximal),
            (full.index.intermediate, half.index.intermediate),
            (full.index.distal, half.index.distal),
        ] {
            let a = a.expect("index joint");
            let b = b.expect("index joint");
            let (full_angle, half_angle) = (
                a.delta.angle_between(Quat::IDENTITY),
                b.delta.angle_between(Quat::IDENTITY),
            );
            assert!(
                (half_angle - full_angle * 0.5).abs() < 1.0e-3,
                "half weight must halve the rotation: {half_angle} vs {full_angle}"
            );
        }
        // A zero weight is the virtual channel: no finger rotation at all.
        assert_eq!(
            observed_finger_deltas(&chain, straight_fingers(), 0.0, 0.0),
            None
        );
    }

    #[test]
    fn finger_deltas_are_skipped_where_the_model_has_no_bone() {
        // A chain with only the index bound must not invent the other fingers.
        let mut chain = articulated_chain();
        chain.finger_rest.middle = crate::arm::FingerJointRestReferences::default();
        chain.finger_rest.ring = crate::arm::FingerJointRestReferences::default();
        chain.finger_rest.little.intermediate = None;
        chain.finger_rest.little.distal = None;
        chain.finger_rest.thumb = crate::arm::FingerJointRestReferences::default();
        let mut observed = straight_fingers();
        observed.fingers[0] = [0.5, 0.5, 0.5];
        observed.thumb = [0.5, 0.5];
        let pose = resolved_fingers(&chain, observed, 1.0);
        assert!(pose.index.proximal.is_some());
        for missing in [&pose.middle, &pose.ring, &pose.little, &pose.thumb] {
            assert!(missing.proximal.is_none());
            assert!(missing.intermediate.is_none());
            assert!(missing.distal.is_none());
            assert!(missing.metacarpal.is_none());
        }
    }
}
