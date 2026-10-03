//! Pure adapter from engine-neutral tracking targets to the existing arm IK.
//!
//! This module never writes Transforms. Live integration must route its output
//! through the existing arm compositor, not register another bone writer.

use bevy::prelude::{Quat, Vec3};
use nalgebra::{Quaternion, UnitQuaternion};
use vtuber_core::arm_tracking::{ArmTrackingTarget, HandFingerPose};
use vtuber_tracking::filter::{
    damped::{DEFAULT_MAX_DT_SEC, RotationSpring, ScalarSpring},
    exponential::shortest_angle_delta,
    time::bounded_dt,
};

use crate::arm::{
    ArmChainBinding, ArmIkError, ArmIkInput, ArmIkSolution, ArmIkTarget, ArmRestGeometry,
    FingerJointRestReferences, solve_two_bone_arm,
};
use crate::arm::{FOREARM_ROLL_LIMIT_RAD, rest_palm_normal};
use crate::arm_pipeline::ArmPipelineError;
use crate::arm_pose::{
    ResolvedArmPose, ResolvedBoneDelta, ResolvedFingerJointPose, ResolvedFingerPose,
};

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

/// Converts an observed solve into rest-relative local joint rotations.
/// The shoulder centre is observed, but clavicle/scapular articulation is not;
/// their authored rest pose is retained. The wrist is reset to its rest pose
/// because independent wrist articulation is not present in the input. Forearm
/// axial rotation is already in `solution` and is not copied to the wrist joint.
pub fn resolved_tracked_arm_pose(
    chain: &ArmChainBinding,
    solution: ArmIkSolution,
    fingers: Option<ResolvedFingerPose>,
) -> Result<ResolvedArmPose, ArmPipelineError> {
    let upper_model = normalized_finite(
        solution.upper_arm_global_rotation * chain.rest.upper_arm.global_rotation.inverse(),
    )
    .ok_or(ArmPipelineError::DegenerateSolvedPose)?;
    // Tracking supplies the shoulder joint centre, not separate clavicle or
    // scapular rotations. Keep the authored girdle pose instead of inventing
    // a coupling that also moves the upper-arm joint centre after the solve.
    let shoulder = chain.shoulder.map(|entity| ResolvedBoneDelta {
        entity,
        delta: Quat::IDENTITY,
    });
    let upper_arm_delta =
        crate::arm::conjugated_rest_delta(upper_model, chain.rest.upper_arm.global_rotation)
            .map_err(ArmPipelineError::Solve)?;
    let lower_arm_delta = normalized_finite(solution.lower_arm_delta)
        .ok_or(ArmPipelineError::DegenerateSolvedPose)?;
    Ok(ResolvedArmPose {
        upper_arm: chain.upper_arm,
        lower_arm: chain.lower_arm,
        upper_arm_delta,
        lower_arm_delta,
        hand: Some(ResolvedBoneDelta {
            entity: chain.hand,
            delta: Quat::IDENTITY,
        }),
        shoulder,
        fingers: fingers.unwrap_or_else(|| crate::arm_pose::resolve_finger_pose(chain, 0.0)),
    })
}

/// Converts hand-local articulation to the rig's rest-relative finger rotations.
/// The palm frame is built from the same wrist/index/little rays as tracking.
/// No solved arm orientation is read: arm swing, shoulder motion and the
/// forearm rotation from `align_palm_twist` have already been removed by the observation's
/// hand-local frame, before its independent smoothing.
#[must_use]
pub fn observed_finger_deltas(
    chain: &ArmChainBinding,
    observed: HandFingerPose,
    weight: f32,
) -> Option<ResolvedFingerPose> {
    if !weight.is_finite() || weight <= f32::EPSILON {
        return None;
    }
    let weight = weight.clamp(0.0, 1.0);
    let normal = rest_palm_normal(chain)?;
    let wrist = chain.rest.wrist.position;
    let index =
        crate::arm::finite_normalized(chain.finger_rest.index.proximal?.rest.position - wrist)?;
    let little =
        crate::arm::finite_normalized(chain.finger_rest.little.proximal?.rest.position - wrist)?;
    let forward = crate::arm::finite_normalized(index + little)?;
    let across = forward.cross(normal);
    let rest = &chain.finger_rest;
    let [index_curl, middle_curl, ring_curl, little_curl] = observed.fingers;
    let [index_spread, middle_spread, ring_spread, little_spread] = observed.spread;
    let [thumb_curl, thumb_ip] = observed.thumb;
    let four = |finger, curl, spread: f32| {
        three_joint_deltas(
            finger,
            curl,
            across * spread.sin() + forward * spread.cos(),
            normal,
            weight,
        )
    };
    Some(ResolvedFingerPose {
        thumb: thumb_deltas(&rest.thumb, [thumb_curl, thumb_ip], normal, weight),
        index: four(&rest.index, index_curl, index_spread),
        middle: four(&rest.middle, middle_curl, middle_spread),
        ring: four(&rest.ring, ring_curl, ring_spread),
        little: four(&rest.little, little_curl, little_spread),
    })
}

/// The thumb's two observed joints, on the rig's `metacarpal, proximal,
/// distal` bones.
///
/// There is no observed base rotation, so the metacarpal keeps the rig's rest
/// pose. Driving it from a Hand Landmarker ray was what made the thumb's root
/// travel whenever the thumb moved and drove its tip past straight: the ray's
/// origin is the model-placed CMC, which rides along with the whole thumb.
/// Authored poses contain flexion amounts, not a replacement for the model's
/// thumb orientation. Keep its rest elevation and opening: applying a CMC
/// opening at the MCP instead displaced the skinned thumb root. Thumb spread
/// is a recognition feature only while the CMC is held at rest.
fn thumb_deltas(
    finger: &FingerJointRestReferences,
    [mcp, ip]: [f32; 2],
    normal: Vec3,
    weight: f32,
) -> ResolvedFingerJointPose {
    ResolvedFingerJointPose {
        // None means "do not write", and leaves the preceding virtual/default
        // curl on this bone. Write neutral to restore the authored thumb base.
        metacarpal: finger.metacarpal.map(|joint| ResolvedBoneDelta {
            entity: joint.entity,
            delta: Quat::IDENTITY,
        }),
        proximal: crate::arm_pose::resolve_finger_joint(
            finger.proximal,
            finger.distal,
            finger.metacarpal,
            mcp * weight,
            normal,
        ),
        intermediate: None,
        distal: observed_joint_bend(finger.distal, None, finger.proximal, ip, normal, weight),
    }
}

fn three_joint_deltas(
    finger: &FingerJointRestReferences,
    [mcp, pip, dip]: [f32; 3],
    direction: Vec3,
    normal: Vec3,
    weight: f32,
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
            (mcp - rest_elevation) * weight,
            normal,
        )?;
        let opening = opening_delta(
            finger.proximal,
            finger.intermediate,
            direction,
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
        ),
        distal: observed_joint_bend(
            finger.distal,
            None,
            finger.intermediate,
            dip,
            normal,
            weight,
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
        (angle - rest_angle) * weight,
        normal,
    )
}

/// Compare the same segment in the rest palm frame and rotate about its normal.
fn opening_delta(
    joint: Option<crate::arm::FingerJointRestBinding>,
    next: Option<crate::arm::FingerJointRestBinding>,
    observed_ray: Vec3,
    normal: Vec3,
    weight: f32,
) -> Option<ResolvedBoneDelta> {
    let joint = joint?;
    let rest_ray = next?.rest.position - joint.rest.position;
    let in_plane = |ray: Vec3| crate::arm::finite_normalized(ray - normal * ray.dot(normal));
    let rest = in_plane(rest_ray)?;
    let observed = in_plane(observed_ray)?;
    let angle = normal.dot(rest.cross(observed)).atan2(rest.dot(observed)) * weight;
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

/// Solves an observed arm with the existing constant-time two-bone implementation.
///
/// This solve deliberately omits virtual-hand swivel, torso lag, shoulder
/// trim. Those modifiers need tracked-source semantics
/// before being enabled, otherwise they can move a hand away from its target.
/// The compositor filters the solved ball/hinge angles after this solve and
/// reconstructs the rigid chain; it never reapplies a raw Cartesian target.
/// Missing/occluded observations are handled by tracking, not an implicit fallback.
pub fn solve_tracked_arm(
    chain: &ArmChainBinding,
    target: ArmTrackingTarget,
    tracking_to_rest: Quat,
) -> Result<ArmIkSolution, ArmIkError> {
    let target = tracked_arm_ik_target(chain.rest, target, tracking_to_rest);
    let solution = solve_two_bone_arm(ArmIkInput::from_chain(chain, target))?;
    arm_from_joints(
        chain,
        solution.upper_arm_global_rotation,
        elbow_flexion(chain, &solution).ok_or(ArmIkError::DegenerateGeometry)?,
    )
    .ok_or(ArmIkError::DegenerateGeometry)
}

/// Holzbaur's thoracohumeral elevation plane: lateral = 0, anterior = 90
/// degrees, cross-body limit = 130 (Xu et al., 2012, doi:10.1016/j.jbiomech.2012.08.018).
/// This is a directional limit, not a limit on the axial rotation of the humerus.
/// Work in the parent-compensated model/rest frame, so anterior follows the
/// chest rather than the camera when the torso turns.
fn constrain_shoulder_rotation(chain: &ArmChainBinding, rotation: Quat) -> Option<Quat> {
    let rest = chain.rest;
    let rest_direction = (rest.elbow.position - rest.upper_arm.position).try_normalize()?;
    let direction = rotation * rest.upper_arm.global_rotation.inverse() * rest_direction;
    let side = match chain.side {
        crate::arm::ArmSide::Left => 1.0,
        crate::arm::ArmSide::Right => -1.0,
    };
    let lateral = side * direction.x;
    let anterior = direction.z;
    let min_plane = -90.0_f32.to_radians();
    let max_plane = 130.0_f32.to_radians();
    let center = (min_plane + max_plane) * 0.5;
    let (sin_center, cos_center) = center.sin_cos();
    // Center the angular chart on the allowed sector. In particular, depth
    // noise at +/-180 degrees must select the same anterior crossing limit.
    let plane = (anterior * cos_center - lateral * sin_center)
        .atan2(lateral * cos_center + anterior * sin_center)
        + center;
    let limited = plane.clamp(min_plane, max_plane);
    if plane == limited {
        return Some(rotation);
    }
    let (sin_plane, cos_plane) = limited.sin_cos();
    let horizontal = lateral.hypot(anterior);
    let bounded = Vec3::new(
        side * horizontal * cos_plane,
        direction.y,
        horizontal * sin_plane,
    );
    // Preserve elevation and transport the existing hinge with the shortest
    // swing. Unlike a yaw of the entire joint frame, this correction tends to
    // identity at the arm-down/up singularity, without an arbitrary hold band.
    // The centered chart bounds this correction to 70 degrees, so the two
    // directions cannot be antiparallel. Keep even tiny corrections instead
    // of from_rotation_arc's near-parallel identity approximation.
    let cross = direction.cross(bounded);
    let swing =
        Quat::from_xyzw(cross.x, cross.y, cross.z, 1.0 + direction.dot(bounded)).normalize();
    Some((swing * rotation).normalize())
}

/// Render-clock time constant of the forearm pronation filter.
///
/// The roll is not observed directly: it is the angle between the observed and
/// the reference palm normals projected perpendicular to the forearm axis, so
/// landmark noise is amplified when the palm is edge-on to that axis, and the
/// measured value swings by tens of degrees from frame to frame. The palm
/// normal is already smoothed on the render clock; the roll it derives gets
/// the same treatment so a jittering landmark cannot spin the wrist.
const PALM_ROLL_TIME_CONSTANT_SEC: f32 = 0.10;

// Critically damped rotation-vector response, following the quaternion spring
// formulation in https://theorangeduck.com/page/spring-roll-call . No angular
// deadband: small intentional movement and a return to neutral still converge.
const PROXIMAL_RESPONSE_SEC: f32 = 0.15;

fn spring_rotation(rotation: Quat) -> UnitQuaternion<f32> {
    UnitQuaternion::from_quaternion(Quaternion::new(
        rotation.w, rotation.x, rotation.y, rotation.z,
    ))
}

fn avatar_rotation(rotation: UnitQuaternion<f32>) -> Quat {
    let q = rotation.quaternion();
    Quat::from_xyzw(q.i, q.j, q.k, q.w)
}

/// One arm's proximal pose control and independent forearm pronation.
///
/// The shoulder is a ball joint, the elbow is one flexion hinge, and pronation
/// stays downstream. Filter those degrees of freedom, then reconstruct both
/// fixed-length bones together; independently following noisy Cartesian joint
/// positions made a still raised arm continuously change its elbow angle.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TrackedArmFilter {
    upper: Option<RotationSpring>,
    elbow: Option<ScalarSpring>,
    pub(crate) roll: PalmRollFilter,
}

impl TrackedArmFilter {
    pub(crate) fn stabilize(
        &mut self,
        chain: &ArmChainBinding,
        solution: &mut ArmIkSolution,
        tracking_to_rest: Quat,
        dt_sec: f32,
    ) -> Option<()> {
        let dt = bounded_dt(dt_sec, DEFAULT_MAX_DT_SEC);
        // Store the arm in the canonical observation frame. A turning torso
        // must not become a new joint target or get delayed by this filter.
        let measured_upper = tracking_to_rest.inverse()
            * constrain_shoulder_rotation(chain, solution.upper_arm_global_rotation)?;
        let upper = if let Some(upper) = self.upper.as_mut() {
            avatar_rotation(upper.step_world(
                spring_rotation(measured_upper),
                dt,
                PROXIMAL_RESPONSE_SEC,
            ))
        } else {
            self.upper = Some(RotationSpring::new(spring_rotation(measured_upper)));
            measured_upper
        };
        let upper = constrain_shoulder_rotation(chain, tracking_to_rest * upper)?;
        // A spherical interpolation can leave the allowed sector even when
        // its endpoints are valid. Retain the bounded joint in the spring,
        // rather than applying a separate correction to the displayed bones.
        self.upper.as_mut()?.value = spring_rotation(tracking_to_rest.inverse() * upper);
        let measured = elbow_flexion(chain, solution)?;
        let elbow = self
            .elbow
            .get_or_insert_with(|| ScalarSpring::new(measured));
        elbow.step(measured, dt, PROXIMAL_RESPONSE_SEC);
        elbow.value = elbow.value.clamp(0.0, crate::arm::ELBOW_FLEXION_LIMIT_RAD);
        *solution = arm_from_joints(chain, upper, elbow.value)?;
        Some(())
    }
}

fn elbow_flexion(chain: &ArmChainBinding, solution: &ArmIkSolution) -> Option<f32> {
    let upper = crate::arm::finite_normalized(solution.elbow - chain.rest.upper_arm.position)?;
    let lower = crate::arm::finite_normalized(solution.wrist - solution.elbow)?;
    Some(upper.cross(lower).length().atan2(upper.dot(lower)))
}

/// Blend the already-filtered shoulder ball joint and elbow hinge back to the
/// resolved initial pose. Cartesian wrist interpolation can pass through the
/// shoulder and force a fold/plane inversion between two valid skeletons.
/// Reconstruct FK with the model's immutable lengths and hinge instead.
pub(crate) fn blend_arm_joints(
    chain: &ArmChainBinding,
    observed: &ArmIkSolution,
    neutral: &ResolvedArmPose,
    weight: f32,
) -> Option<ArmIkSolution> {
    let rest = chain.rest;
    let upper_segment = rest.elbow.position - rest.upper_arm.position;
    let lower_segment = rest.wrist.position - rest.elbow.position;
    let neutral_lower =
        rest.elbow.global_rotation * neutral.lower_arm_delta * rest.elbow.global_rotation.inverse();
    let neutral_flexion = upper_segment.angle_between(neutral_lower * lower_segment);
    let flexion = neutral_flexion + (elbow_flexion(chain, observed)? - neutral_flexion) * weight;
    let upper = (rest.upper_arm.global_rotation * neutral.upper_arm_delta)
        .slerp(observed.upper_arm_global_rotation, weight)
        .normalize();
    arm_from_joints(chain, upper, flexion)
}

fn arm_from_joints(
    chain: &ArmChainBinding,
    upper_global: Quat,
    elbow_angle: f32,
) -> Option<ArmIkSolution> {
    let input = crate::arm::ArmIkInput::from_chain(
        chain,
        crate::arm::ArmIkTarget {
            wrist: chain.rest.wrist.position,
            elbow_pole: chain.rest.elbow.position,
        },
    );
    let upper_global = constrain_shoulder_rotation(chain, upper_global)?;
    let pose = crate::skeleton::from_joints(input.skeleton_rest(), upper_global, elbow_angle, 0.0)?;
    Some(input.solution_from_skeleton(pose))
}
/// Render-clock low-pass state for forearm pronation, in radians.
///
/// The measured angle is a bounded joint coordinate relative to neutral, not
/// a free-spinning phase. Project it into the joint's range before the
/// critically damped second-order step (the same one the tracking channels
/// use). `value` is also what
/// [`align_palm_twist`] applied last, so the filter keeps running across ticks
/// whether or not a new camera frame arrived.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PalmRollFilter {
    spring: ScalarSpring,
}

impl PalmRollFilter {
    /// Advances the filter toward `measured` by one render tick and returns
    /// the filtered roll. Non-finite input holds the current value.
    pub fn step(&mut self, measured: f32, dt_sec: f32) -> f32 {
        if !measured.is_finite() || !dt_sec.is_finite() || dt_sec <= 0.0 {
            return self.spring.value;
        }
        // Bound the target and the retained state in the same joint space.
        // Clamping only the bone output let the filter retain whole turns,
        // leaving the hand stuck at a limit after the observation returned.
        // Use neutral as the angular origin. Lifting near the prior value at
        // +90 would turn a -120 observation into +240 and pin the wrong limit.
        let target = shortest_angle_delta(0.0, measured)
            .clamp(-FOREARM_ROLL_LIMIT_RAD, FOREARM_ROLL_LIMIT_RAD);
        self.spring
            .step(target, dt_sec, PALM_ROLL_TIME_CONSTANT_SEC);
        self.spring.value = self
            .spring
            .value
            .clamp(-FOREARM_ROLL_LIMIT_RAD, FOREARM_ROLL_LIMIT_RAD);
        if self.spring.value.abs() >= FOREARM_ROLL_LIMIT_RAD {
            self.spring.velocity = 0.0;
        }
        self.spring.value
    }

    /// Drops the roll back to neutral, for a side that has no observation.
    pub fn reset(&mut self) {
        self.spring = ScalarSpring::default();
    }
}

/// One resolved forearm pronation/supination and neutral wrist output.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PalmTwist {
    /// Neutral wrist delta. Forearm pronation is inherited through FK.
    pub hand: Option<Quat>,
    /// Forearm axial coordinate, bounded by [`FOREARM_ROLL_LIMIT_RAD`].
    pub forearm_roll: f32,
}

/// Applies observed pronation/supination to the forearm's longitudinal axis.
/// The shared two-bone IK defines the humeral orientation and fixed elbow
/// flexion plane. The remaining signed palm rotation is a radioulnar DOF.
/// Its bounded, filtered coordinate never changes the shoulder or elbow
/// positions; the hand inherits it through the existing hierarchy. No axial
/// hand-local rotation is added. Missing palm geometry returns `None`.
#[must_use]
pub fn align_palm_twist(
    chain: &ArmChainBinding,
    solution: &mut ArmIkSolution,
    palm_normal: [f32; 3],
    tracking_to_rest: Quat,
    weight: f32,
    roll_filter: &mut PalmRollFilter,
    dt_sec: f32,
) -> Option<PalmTwist> {
    if !weight.is_finite() || weight <= f32::EPSILON || !tracking_to_rest.is_finite() {
        return None;
    }
    let weight = weight.clamp(0.0, 1.0);
    let rest_normal = rest_palm_normal(chain)?;
    let axis = crate::arm::finite_normalized(solution.wrist - solution.elbow)?;
    let observed = crate::arm::finite_normalized(tracking_to_rest * Vec3::from(palm_normal))?;
    let rest_hand = chain.rest.wrist.global_rotation;
    let current = crate::arm::finite_normalized(
        hand_global(chain, solution) * (rest_hand.inverse() * rest_normal),
    )?;
    let observed_perp = crate::arm::finite_normalized(observed - axis * observed.dot(axis))?;
    let current_perp = crate::arm::finite_normalized(current - axis * current.dot(axis))?;
    let measured = axis
        .dot(current_perp.cross(observed_perp))
        .atan2(current_perp.dot(observed_perp));
    if !measured.is_finite() {
        return None;
    }
    let filtered = roll_filter.step(measured, dt_sec);
    let forearm_roll = filtered * weight;
    // The hand follows the forearm rigidly. Axial rotation belongs to the
    // radioulnar joint; adding a hand-local roll invents a third wrist DOF.
    let hand = Some(Quat::IDENTITY);
    roll_forearm(chain, solution, forearm_roll);
    Some(PalmTwist { hand, forearm_roll })
}

/// The hand bone's model-space orientation with its authored rest-relative pose.
fn hand_global(chain: &ArmChainBinding, solution: &ArmIkSolution) -> Quat {
    solution.lower_arm_global_rotation
        * (chain.rest.elbow.global_rotation.inverse() * chain.rest.wrist.global_rotation)
}

/// Rolls the solved forearm about its own long axis in model space.
///
/// The elbow and wrist lie on that axis, so the roll moves no bone position and
/// the hand follows the forearm rigidly. Rebuilds the lower-arm global rotation,
/// rest-relative delta, and local rotation exactly as [`solve_two_bone_arm`]
/// does, so the shared conversion still applies. Returns whether the roll was
/// written.
pub(crate) fn roll_forearm(
    chain: &ArmChainBinding,
    solution: &mut ArmIkSolution,
    angle: f32,
) -> bool {
    if !angle.is_finite() || angle.abs() <= 1.0e-6 {
        return false;
    }
    let input = crate::arm::ArmIkInput::from_chain(
        chain,
        crate::arm::ArmIkTarget {
            wrist: solution.wrist,
            elbow_pole: solution.elbow,
        },
    );
    let rest = input.skeleton_rest();
    let Some(joints) = crate::skeleton::joint_coordinates(rest, solution.skeleton_pose()) else {
        return false;
    };
    let Some(pose) = crate::skeleton::from_joints(
        rest,
        solution.upper_arm_global_rotation,
        joints.x,
        joints.y + angle,
    ) else {
        return false;
    };
    *solution = input.solution_from_skeleton(pose);
    true
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
    use crate::arm::rest_palm_normal;
    use crate::arm::{ArmPoseProfile, RestSpaceBonePose};

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
            fingers: None,
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

    #[test]
    fn existing_solver_preserves_bone_lengths_and_reaches_observed_target() {
        let rest = geometry(1.0);
        let solution = solve_tracked_arm(&chain(), target(), Quat::IDENTITY).unwrap();
        let mapped = tracked_arm_ik_target(rest, target(), Quat::IDENTITY);
        near(solution.wrist, mapped.wrist);
        assert!(((solution.elbow - rest.upper_arm.position).length() - 0.4).abs() < 1.0e-5);
        assert!(((solution.wrist - solution.elbow).length() - 0.3).abs() < 1.0e-5);
    }

    #[test]
    fn local_rotations_reconstruct_the_same_elbow_and_wrist() {
        let rest = geometry(1.0);
        let solution = solve_tracked_arm(&chain(), target(), Quat::IDENTITY).unwrap();
        let elbow = rest.upper_arm.position + solution.upper_arm_local_rotation * (Vec3::X * 0.4);
        let wrist = elbow
            + (solution.upper_arm_local_rotation * solution.lower_arm_local_rotation)
                * (Vec3::X * 0.3);
        near(elbow, solution.elbow);
        near(wrist, solution.wrist);
    }

    #[test]
    fn current_parent_rotation_is_removed_exactly_once() {
        let rest = geometry(1.0);
        let parent_rest = Quat::from_rotation_z(0.2);
        let parent_current = Quat::from_rotation_y(0.8) * parent_rest;
        let view_to_model = Quat::from_rotation_x(-0.1);
        let tracking_to_rest =
            tracking_to_rest_rotation(parent_rest, parent_current, view_to_model);
        let solution = solve_tracked_arm(&chain(), target(), tracking_to_rest).unwrap();
        let displayed_offset =
            (parent_current * parent_rest.inverse()) * (solution.wrist - rest.upper_arm.position);
        near(
            displayed_offset,
            view_to_model * Vec3::new(0.4, -0.3, 0.5) * rest.total_arm_length,
        );
    }

    #[test]
    fn unreachable_target_uses_existing_solver_reach_constraint() {
        let rest = geometry(1.0);
        let far = ArmTrackingTarget {
            wrist: [3.0, 0.0, 0.0],
            elbow_pole: [0.5, -0.5, 0.0],
            palm_normal: None,
            fingers: None,
        };
        let solution = solve_tracked_arm(&chain(), far, Quat::IDENTITY).unwrap();
        assert!(solution.solved_reach < rest.total_arm_length);
        assert!(solution.solved_reach > 0.69);
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

    #[test]
    fn shoulder_plane_preserves_valid_poses_and_brings_medial_elbows_forward() {
        for side in [crate::arm::ArmSide::Left, crate::arm::ArmSide::Right] {
            let sign = if side == crate::arm::ArmSide::Left {
                1.0
            } else {
                -1.0
            };
            let mut chain = chain();
            chain.side = side;
            chain.rest.elbow.position.x = chain.rest.upper_arm.position.x + sign * 0.4;
            chain.rest.wrist.position.x = chain.rest.upper_arm.position.x + sign * 0.7;
            chain.rest.upper_arm.global_rotation =
                Quat::from_rotation_y(1.1) * Quat::from_rotation_z(-0.2);
            chain.rest.elbow.global_rotation =
                Quat::from_rotation_x(-0.7) * Quat::from_rotation_z(0.4);
            for degrees in [-80.0_f32, 0.0, 90.0, 120.0, 150.0, 179.99, 180.01, 195.0] {
                let (sin, cos) = degrees.to_radians().sin_cos();
                let direction = Vec3::new(sign * cos, -0.7, sin).normalize();
                let upper = Quat::from_axis_angle(direction, 0.5)
                    * Quat::from_rotation_arc(sign * Vec3::X, direction)
                    * chain.rest.upper_arm.global_rotation;
                let bounded = constrain_shoulder_rotation(&chain, upper).unwrap();
                let solution = arm_from_joints(&chain, upper, 1.2).unwrap();
                let actual = (solution.elbow - chain.rest.upper_arm.position).normalize();
                assert!((actual.y - direction.y).abs() < 1.0e-6);
                if degrees <= 130.0 {
                    assert_eq!(
                        bounded, upper,
                        "valid swing and axial roll must be untouched"
                    );
                } else {
                    assert!(actual.z > 0.0);
                    assert!((actual.z.atan2(sign * actual.x).to_degrees() - 130.0).abs() < 0.001);
                }
                assert!((elbow_flexion(&chain, &solution).unwrap() - 1.2).abs() < 1.0e-5);
                let pose = resolved_tracked_arm_pose(&chain, solution, None).unwrap();
                let [origin, elbow, wrist] = composed_joints(&chain, pose);
                near(elbow, solution.elbow);
                near(wrist, solution.wrist);
                assert!((origin.distance(elbow) - 0.4).abs() < 1.0e-6);
                assert!((elbow.distance(wrist) - 0.3).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn shoulder_crossing_is_continuous_through_arm_down_and_depth_sign_change() {
        let chain = chain();
        let mut previous = None::<Quat>;
        // Cross the 90-degree lowered position without a finite shoulder
        // yaw appearing when the elevation plane changes from 0 to 180.
        for step in 0..=400 {
            let descent = (88.0 + step as f32 * 0.01).to_radians();
            let rotation = Quat::from_rotation_z(-descent);
            let bounded = constrain_shoulder_rotation(&chain, rotation).unwrap();
            if let Some(previous) = previous {
                assert!(previous.angle_between(bounded) < 0.002);
            }
            previous = Some(bounded);
        }
        let crossing = |z| {
            let direction = Vec3::new(-0.6, -0.8, z).normalize();
            constrain_shoulder_rotation(&chain, Quat::from_rotation_arc(Vec3::X, direction))
                .unwrap()
        };
        assert!(crossing(-0.00001).angle_between(crossing(0.00001)) < 0.001);
    }

    #[test]
    fn shoulder_filter_and_loss_blend_keep_crossings_inside_the_joint_limit() {
        let chain = chain();
        let neutral = arm_from_joints(&chain, Quat::from_rotation_z(-0.6), 0.8).unwrap();
        let neutral = resolved_tracked_arm_pose(&chain, neutral, None).unwrap();
        let mut filter = TrackedArmFilter::default();
        for tick in 0..360 {
            // Unconstrained IK observations, including a small depth sign
            // change. Supply them before the production joint filter.
            let descent = if tick < 180 {
                tick as f32
            } else {
                (360 - tick) as f32
            };
            let raw = Quat::from_rotation_y(if tick % 2 == 0 { 0.001 } else { -0.001 })
                * Quat::from_rotation_z(-descent.to_radians());
            let input = ArmIkInput::from_chain(
                &chain,
                ArmIkTarget {
                    wrist: chain.rest.wrist.position,
                    elbow_pole: chain.rest.elbow.position,
                },
            );
            let mut solution = input.solution_from_skeleton(
                crate::skeleton::from_joints(input.skeleton_rest(), raw, 0.8, 0.0).unwrap(),
            );
            filter
                .stabilize(&chain, &mut solution, Quat::IDENTITY, 1.0 / 60.0)
                .unwrap();
            let retained = avatar_rotation(filter.upper.unwrap().value);
            assert!(retained.angle_between(solution.upper_arm_global_rotation) < 0.001);
            for weight in [1.0, 0.75, 0.5, 0.25, 0.0] {
                let blended = blend_arm_joints(&chain, &solution, &neutral, weight).unwrap();
                let direction = (blended.elbow - chain.rest.upper_arm.position).normalize();
                if direction.x < -1.0e-5 {
                    assert!(
                        direction.z >= -direction.x * 50.0_f32.to_radians().tan() - 1.0e-5,
                        "tick {tick}, weight {weight}, direction {direction:?}"
                    );
                }
                assert!((elbow_flexion(&chain, &blended).unwrap() - 0.8).abs() < 1.0e-5);
            }
        }
    }

    #[test]
    fn tracked_conversion_keeps_the_solved_deltas_without_a_clavicle() {
        let chain = chain();
        let solution = solve_two_bone_arm(ArmIkInput::from_chain(
            &chain,
            tracked_arm_ik_target(chain.rest, target(), Quat::IDENTITY),
        ))
        .unwrap();
        let tracked = resolved_tracked_arm_pose(&chain, solution, None).unwrap();
        let profile = ArmPoseProfile {
            finger_curl_radians: 0.0,
            ..ArmPoseProfile::default()
        };
        let mut shared = crate::arm_pose::resolved_from_solution(&chain, &solution, profile)
            .unwrap()
            .unwrap();
        shared.hand = Some(ResolvedBoneDelta {
            entity: chain.hand,
            delta: Quat::IDENTITY,
        });
        assert_eq!(tracked, shared);
        assert!(tracked.upper_arm_delta.is_finite());
        assert!(tracked.lower_arm_delta.is_finite());
    }

    fn finger_binding(position: Vec3) -> crate::arm::FingerJointRestBinding {
        crate::arm::FingerJointRestBinding {
            entity: bevy::prelude::Entity::from_raw_u32(3).unwrap(),
            rest: bone(position),
        }
    }

    /// The default test chain plus authored index/little-finger rest positions
    /// whose cross product points along +Y, i.e. a palm facing up.
    fn palm_chain() -> ArmChainBinding {
        let mut chain = chain();
        let wrist = chain.rest.wrist.position;
        chain.finger_rest.index.proximal =
            Some(finger_binding(wrist + Vec3::new(0.05, 0.0, 0.003)));
        chain.finger_rest.little.proximal =
            Some(finger_binding(wrist + Vec3::new(0.05, 0.0, -0.003)));
        chain
    }

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

    fn with_clavicle(mut chain: ArmChainBinding) -> ArmChainBinding {
        chain.shoulder = Some(bevy::prelude::Entity::from_raw_u32(20).unwrap());
        chain.rest.shoulder = Some(bone(chain.rest.upper_arm.position - Vec3::X * 0.08));
        chain
    }

    // Reconstruct the hierarchy the compositor writes, including the clavicle.
    // Checking only the IK solution missed the later double shoulder rotation.
    fn composed_joints(chain: &ArmChainBinding, pose: ResolvedArmPose) -> [Vec3; 3] {
        let rest = chain.rest;
        let shoulder = pose
            .shoulder
            .zip(rest.shoulder)
            .map_or(Quat::IDENTITY, |(s, r)| {
                r.global_rotation * s.delta * r.global_rotation.inverse()
            });
        let upper_origin = rest.shoulder.map_or(rest.upper_arm.position, |r| {
            r.position + shoulder * (rest.upper_arm.position - r.position)
        });
        let upper = shoulder
            * rest.upper_arm.global_rotation
            * pose.upper_arm_delta
            * rest.upper_arm.global_rotation.inverse();
        let lower = rest.elbow.global_rotation
            * pose.lower_arm_delta
            * rest.elbow.global_rotation.inverse();
        let elbow = upper_origin + upper * (rest.elbow.position - rest.upper_arm.position);
        let wrist = elbow + upper * lower * (rest.wrist.position - rest.elbow.position);
        [upper_origin, elbow, wrist]
    }

    #[test]
    fn proximal_noise_is_damped_through_clavicle_palm_and_finger_composition() {
        let chain = with_clavicle(articulated_chain());
        let upper = Quat::from_rotation_z(0.4) * Quat::from_rotation_x(0.5);
        let elbow = 90.0_f32.to_radians();
        let mut filter = TrackedArmFilter::default();
        let mut first = arm_from_joints(&chain, upper, elbow).unwrap();
        filter
            .stabilize(&chain, &mut first, Quat::IDENTITY, 1.0 / 60.0)
            .unwrap();
        let baseline = resolved_tracked_arm_pose(&chain, first, None).unwrap();
        let positions = composed_joints(&chain, baseline);
        let mut last_roll = 0.0;
        for tick in 0..240 {
            let sign = if (tick / 2) % 2 == 0 { 1.0 } else { -1.0 };
            let mut measured = arm_from_joints(
                &chain,
                Quat::from_rotation_y(sign * 2.0_f32.to_radians()) * upper,
                elbow + sign * 4.0_f32.to_radians(),
            )
            .unwrap();
            filter
                .stabilize(&chain, &mut measured, Quat::IDENTITY, 1.0 / 60.0)
                .unwrap();
            let axis = (measured.wrist - measured.elbow).normalize();
            let palm = Quat::from_axis_angle(axis, 0.6)
                * solved_palm_normal(&chain, &measured, Quat::IDENTITY);
            let twist = align_palm_twist(
                &chain,
                &mut measured,
                palm.to_array(),
                Quat::IDENTITY,
                1.0,
                &mut filter.roll,
                1.0 / 60.0,
            )
            .unwrap();
            last_roll = twist.forearm_roll;
            let mut fingers = straight_fingers();
            fingers.fingers[2] = [0.8, 0.9, 0.4];
            let fingers = observed_finger_deltas(&chain, fingers, 1.0).unwrap();
            let pose = resolved_tracked_arm_pose(&chain, measured, Some(fingers)).unwrap();
            for (got, expected) in composed_joints(&chain, pose).into_iter().zip(positions) {
                assert!(
                    got.distance(expected) < 0.003,
                    "damped joint noise moved the skeleton: {got:?} vs {expected:?}"
                );
            }
            assert!(
                pose.fingers
                    .ring
                    .proximal
                    .unwrap()
                    .delta
                    .angle_between(Quat::IDENTITY)
                    > 0.3
            );
            assert_eq!(pose.fingers.thumb.metacarpal.unwrap().delta, Quat::IDENTITY);
        }
        assert!(
            last_roll > 0.5,
            "stabilizing the arm must not freeze palm rotation"
        );
    }

    #[test]
    fn clavicle_composition_preserves_both_solved_bone_directions() {
        let chain = with_clavicle(articulated_chain());
        let solution = arm_from_joints(&chain, Quat::from_rotation_z(0.6), 1.1).unwrap();
        let pose = resolved_tracked_arm_pose(&chain, solution, None).unwrap();
        assert_eq!(pose.shoulder.unwrap().delta, Quat::IDENTITY);
        let [origin, elbow, wrist] = composed_joints(&chain, pose);
        near(
            elbow - origin,
            solution.elbow - chain.rest.upper_arm.position,
        );
        near(wrist - elbow, solution.wrist - solution.elbow);
        assert!((origin.distance(elbow) - chain.rest.upper_arm_length).abs() < 1.0e-5);
        assert!((elbow.distance(wrist) - chain.rest.forearm_length).abs() < 1.0e-5);
    }

    #[test]
    fn proximal_intent_follows_motion_without_damping_the_parent_compensation() {
        let chain = articulated_chain();
        let start = Quat::from_rotation_z(0.3);
        let end = Quat::from_rotation_z(0.8) * Quat::from_rotation_x(0.4);
        let mut filter = TrackedArmFilter::default();
        let mut solved = arm_from_joints(&chain, start, 0.8).unwrap();
        filter
            .stabilize(&chain, &mut solved, Quat::IDENTITY, 1.0 / 60.0)
            .unwrap();
        for _ in 0..120 {
            solved = arm_from_joints(&chain, end, 1.5).unwrap();
            filter
                .stabilize(&chain, &mut solved, Quat::IDENTITY, 1.0 / 60.0)
                .unwrap();
        }
        assert!(solved.upper_arm_global_rotation.angle_between(end) < 0.001);
        let angle = (solved.elbow - chain.rest.upper_arm.position)
            .angle_between(solved.wrist - solved.elbow);
        assert!((angle - 1.5).abs() < 0.001);
        let retained_upper = solved.upper_arm_global_rotation;
        let retained_elbow = angle;
        // A torso turn is removed before composition. It must take effect
        // immediately, rather than becoming delayed shoulder muscle intent.
        let parent = Quat::from_rotation_y(0.8) * Quat::from_rotation_z(0.2);
        let tracking_to_rest = tracking_to_rest_rotation(Quat::IDENTITY, parent, Quat::IDENTITY);
        solved = arm_from_joints(&chain, tracking_to_rest * end, 1.5).unwrap();
        filter
            .stabilize(&chain, &mut solved, tracking_to_rest, 1.0 / 60.0)
            .unwrap();
        assert!(
            (parent * solved.upper_arm_global_rotation)
                .dot(retained_upper)
                .abs()
                > 1.0 - 1.0e-6
        );
        let angle = (solved.elbow - chain.rest.upper_arm.position)
            .angle_between(solved.wrist - solved.elbow);
        assert!((angle - retained_elbow).abs() < 0.0001);
    }

    /// A solved pose for the articulated chain, with the observation's fingers
    /// applied at `weight`.
    fn resolved_fingers(
        chain: &ArmChainBinding,
        fingers: HandFingerPose,
        weight: f32,
    ) -> ResolvedFingerPose {
        observed_finger_deltas(chain, fingers, weight).expect("finger deltas")
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
            assert_eq!(pose.thumb.metacarpal.unwrap().delta, Quat::IDENTITY);
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
            crate::DefaultArmPose {
                generation,
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
        let solution = solve_tracked_arm(&chain, baseline, Quat::IDENTITY).unwrap();

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
                let fingers = observed_finger_deltas(&chain, observed, weight);
                if weight == 0.0 {
                    assert!(fingers.is_none());
                }
                let pose = resolved_tracked_arm_pose(&chain, solution, fingers).unwrap();
                if weight > 0.0 {
                    // The observed path drives the two real joints only. The
                    // virtual path is a separate pose and is not under test.
                    assert!(
                        pose.fingers.thumb.metacarpal.unwrap().delta == Quat::IDENTITY,
                        "an oblique thumb must leave its base neutral"
                    );
                    let moved = pose
                        .fingers
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
                    &pose.fingers.index,
                    &pose.fingers.middle,
                    &pose.fingers.ring,
                    &pose.fingers.little,
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
        let neutral = observed_finger_deltas(&chain, straight_fingers(), 1.0).unwrap();
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
            let mut solution = solve_tracked_arm(&chain, target, Quat::IDENTITY).unwrap();
            let positions = (solution.elbow, solution.wrist);
            let hand_delta = align_palm_twist(
                &chain,
                &mut solution,
                target.palm_normal.unwrap(),
                Quat::IDENTITY,
                1.0,
                &mut PalmRollFilter::default(),
                1.0,
            )
            .and_then(|twist| twist.hand);
            let fingers = observed_finger_deltas(&chain, target.fingers.unwrap(), 1.0).unwrap();
            let pose = resolved_tracked_arm_pose(&chain, solution, Some(fingers)).unwrap();
            for (a, b) in [
                (&pose.fingers.thumb, &neutral.thumb),
                (&pose.fingers.index, &neutral.index),
                (&pose.fingers.middle, &neutral.middle),
                (&pose.fingers.ring, &neutral.ring),
                (&pose.fingers.little, &neutral.little),
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
            if twist != 0.0 {
                // These observations are rigid rotations of a straight rest arm,
                // so their bend plane carries no information and the palm's
                // angle is not split into a wrist and a humeral part. The
                // remaining share must still reach the hand, which is all this
                // test needs: whether it lands at the wrist's limit or inside
                // it is the separate palm-twist tests' business.
                assert_eq!(hand_delta.unwrap(), Quat::IDENTITY);
            }
            near(solution.elbow, positions.0);
            near(solution.wrist, positions.1);
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
            let pose = observed_finger_deltas(&chain, target.fingers.unwrap(), 1.0).unwrap();
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
            assert_eq!(pose.thumb.metacarpal.unwrap().delta, Quat::IDENTITY);
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
            let pose = observed_finger_deltas(&chain, observed, 1.0).unwrap();
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
            let delta = observed_finger_deltas(&chain, observed, 1.0)
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
            observed_finger_deltas(&chain, straight_fingers(), 0.0),
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

    /// The hand's model-space palm normal after the solve and hand delta.
    fn solved_palm_normal(chain: &ArmChainBinding, solution: &ArmIkSolution, hand: Quat) -> Vec3 {
        let rest_hand = chain.rest.wrist.global_rotation;
        let hand_global = solution.lower_arm_global_rotation
            * (chain.rest.elbow.global_rotation.inverse() * rest_hand)
            * hand;
        (hand_global * (rest_hand.inverse() * rest_palm_normal(chain).unwrap())).normalize()
    }

    /// A bent solve: the pole gives the elbow's bend plane, which the palm
    /// reference is measured against.
    fn bent_target() -> ArmTrackingTarget {
        ArmTrackingTarget {
            wrist: [0.9, -0.05, 0.05],
            elbow_pole: [0.5, 0.4, 0.0],
            palm_normal: None,
            fingers: None,
        }
    }

    /// The oriented solve plus the palm normal it shows with no wrist roll.
    ///
    /// The bend plane puts the palm here, so a test observation is a roll of
    /// this normal around the solved forearm axis, which is exactly the wrist's
    /// own pronation.
    fn oriented() -> (ArmChainBinding, ArmIkSolution, Vec3) {
        let chain = palm_chain();
        let solution = solve_two_bone_arm(ArmIkInput::from_chain(
            &chain,
            tracked_arm_ik_target(chain.rest, bent_target(), Quat::IDENTITY),
        ))
        .unwrap();
        let reference = solved_palm_normal(&chain, &solution, Quat::IDENTITY);
        (chain, solution, reference)
    }

    fn perpendicular(value: Vec3, axis: Vec3) -> Vec3 {
        (value - axis * value.dot(axis)).normalize()
    }

    #[test]
    fn fixed_hinge_ik_points_the_forearm_at_its_solved_wrist() {
        let chain = palm_chain();
        let solution = solve_tracked_arm(&chain, bent_target(), Quat::IDENTITY).unwrap();
        near(
            crate::arm::finite_normalized(
                solution.lower_arm_global_rotation
                    * (chain.rest.elbow.global_rotation.inverse()
                        * (chain.rest.wrist.position - chain.rest.elbow.position)),
            )
            .unwrap(),
            crate::arm::finite_normalized(solution.wrist - solution.elbow).unwrap(),
        );
    }

    #[test]
    fn downstream_forearm_rotation_cannot_exceed_the_skeletal_joint_limit() {
        for sign in [-1.0, 1.0] {
            let (chain, mut solution, _) = oriented();
            let base = solution;
            assert!(roll_forearm(
                &chain,
                &mut solution,
                sign * FOREARM_ROLL_LIMIT_RAD
            ));
            assert!(roll_forearm(
                &chain,
                &mut solution,
                sign * FOREARM_ROLL_LIMIT_RAD
            ));
            let input = ArmIkInput::from_chain(
                &chain,
                tracked_arm_ik_target(chain.rest, bent_target(), Quat::IDENTITY),
            );
            let joints =
                crate::skeleton::joint_coordinates(input.skeleton_rest(), solution.skeleton_pose())
                    .unwrap();
            assert!((joints.y - sign * FOREARM_ROLL_LIMIT_RAD).abs() < 1.0e-5);
            near(solution.elbow, base.elbow);
            near(solution.wrist, base.wrist);
            assert_eq!(solution.upper_arm_delta, base.upper_arm_delta);
        }
    }

    #[test]
    fn a_palm_twist_turns_the_forearm_without_swinging_the_upper_arm() {
        // The recorded symptom: a small palm rotation swung the whole upper arm
        // because the demanded roll was measured against the direction-only
        // solve's arbitrary twist, and everything past the wrist's range was
        // rolled into the humerus. The bend plane now fixes the humeral twist,
        // so the palm turns the forearm and the wrist inherits it through FK.
        let (chain, base, reference) = oriented();
        let axis = crate::arm::finite_normalized(base.wrist - base.elbow).unwrap();
        let expected_upper = base.upper_arm_delta;
        let mut previous: Option<(Quat, Quat)> = None;
        let mut worst = 0.0_f32;
        let mut saturated = 0;
        for degrees in (0..=120_i32).chain((0..120_i32).rev()) {
            let observed = Quat::from_axis_angle(axis, (degrees as f32).to_radians())
                .mul_vec3(reference)
                .to_array();
            let mut solution = base;
            let twist = align_palm_twist(
                &chain,
                &mut solution,
                observed,
                Quat::IDENTITY,
                1.0,
                &mut PalmRollFilter::default(),
                1.0,
            )
            .expect("every step of a wrist twist is usable");
            assert!(
                solution
                    .upper_arm_delta
                    .angle_between(expected_upper)
                    .to_degrees()
                    < 1.0e-3,
                "a palm twist must not move the upper arm (at {degrees} degrees)"
            );
            if (twist.forearm_roll.abs() - FOREARM_ROLL_LIMIT_RAD).abs() < 0.1_f32.to_radians() {
                saturated += 1;
            }
            let current = (
                solution.lower_arm_delta,
                twist.hand.unwrap_or(Quat::IDENTITY),
            );
            if let Some(previous) = previous {
                worst = worst
                    .max(previous.0.angle_between(current.0))
                    .max(previous.1.angle_between(current.1));
            }
            previous = Some(current);
        }
        assert!(
            saturated > 0,
            "the sweep must reach the wrist's limit and hold there"
        );
        // A degree of wrist twist must not move either bone by anything near a
        // half turn.
        assert!(
            worst < 5.0_f32.to_radians(),
            "a wrist twist must stay continuous, worst step was {} degrees",
            worst.to_degrees()
        );
    }

    #[test]
    fn palm_twist_rotates_the_wrist_to_the_observed_plane() {
        let (chain, mut solution, reference) = oriented();
        let axis = crate::arm::finite_normalized(solution.wrist - solution.elbow).unwrap();
        let observed = Quat::from_axis_angle(axis, 60.0_f32.to_radians()) * reference;
        let twist = align_palm_twist(
            &chain,
            &mut solution,
            observed.to_array(),
            Quat::IDENTITY,
            1.0,
            &mut PalmRollFilter::default(),
            1.0,
        )
        .expect("hand roll");
        let hand = twist.hand.expect("hand share");

        // The forearm and the hand each take a share; together they land the
        // palm plane on the observation.
        let expected = perpendicular(observed, axis);
        let actual = perpendicular(solved_palm_normal(&chain, &solution, hand), axis);
        assert!(
            actual.dot(expected) > 1.0 - 1.0e-4,
            "hand plane must match the observation: {actual:?} != {expected:?}"
        );
        assert!(
            (twist.forearm_roll - 60.0_f32.to_radians()).abs() < 1.0e-3,
            "the wrist must show exactly the observed pronation: {} deg",
            twist.forearm_roll.to_degrees()
        );
    }

    #[test]
    fn pronation_moves_the_forearm_without_adding_an_axial_wrist_joint() {
        let (chain, mut solution, reference) = oriented();
        let axis = crate::arm::finite_normalized(solution.wrist - solution.elbow).unwrap();
        let observed = Quat::from_axis_angle(axis, 60.0_f32.to_radians()) * reference;
        let forearm_before = solution.lower_arm_delta;
        let hand = align_palm_twist(
            &chain,
            &mut solution,
            observed.to_array(),
            Quat::IDENTITY,
            1.0,
            &mut PalmRollFilter::default(),
            1.0,
        )
        .expect("hand roll")
        .hand
        .expect("hand share");
        assert_ne!(
            solution.lower_arm_delta, forearm_before,
            "the forearm must take a share of the roll"
        );
        assert_eq!(hand, Quat::IDENTITY, "wrist has no independent axial DOF");
    }

    #[test]
    fn palm_twist_weight_zero_is_a_no_op() {
        let chain = palm_chain();
        let mut solution = solve_two_bone_arm(ArmIkInput::from_chain(
            &chain,
            tracked_arm_ik_target(chain.rest, target(), Quat::IDENTITY),
        ))
        .unwrap();
        let before = solution;
        assert_eq!(
            align_palm_twist(
                &chain,
                &mut solution,
                [0.0, 0.0, 1.0],
                Quat::IDENTITY,
                0.0,
                &mut PalmRollFilter::default(),
                1.0,
            ),
            None
        );
        assert_eq!(solution, before);
    }

    #[test]
    fn palm_twist_scales_monotonically_with_the_weight() {
        let (chain, mut full, reference) = oriented();
        let axis = crate::arm::finite_normalized(full.wrist - full.elbow).unwrap();
        let observed = Quat::from_axis_angle(axis, 60.0_f32.to_radians()) * reference;
        let full_hand = align_palm_twist(
            &chain,
            &mut full,
            observed.to_array(),
            Quat::IDENTITY,
            1.0,
            &mut PalmRollFilter::default(),
            1.0,
        )
        .unwrap()
        .hand
        .unwrap();
        let (_, mut half, _) = oriented();
        let half_hand = align_palm_twist(
            &chain,
            &mut half,
            observed.to_array(),
            Quat::IDENTITY,
            0.5,
            &mut PalmRollFilter::default(),
            1.0,
        )
        .unwrap()
        .hand
        .unwrap();

        let expected = perpendicular(observed, axis);
        let error = |solution: &ArmIkSolution, hand: Quat| {
            let normal = perpendicular(solved_palm_normal(&chain, solution, hand), axis);
            1.0 - normal.dot(expected)
        };
        let full_error = error(&full, full_hand);
        let half_error = error(&half, half_hand);
        assert!(full_error < 1.0e-4, "full weight must align: {full_error}");
        assert!(
            half_error > full_error && half_error < 1.0,
            "half weight must sit between the default roll and full alignment: \
             {half_error} vs {full_error}"
        );
    }

    #[test]
    fn palm_twist_is_a_no_op_without_usable_geometry() {
        let plain = chain();
        let mut base = solve_tracked_arm(&plain, target(), Quat::IDENTITY).unwrap();
        assert_eq!(
            align_palm_twist(
                &plain,
                &mut base,
                [0.0, 0.0, 1.0],
                Quat::IDENTITY,
                1.0,
                &mut PalmRollFilter::default(),
                1.0,
            ),
            None
        );

        let chain = palm_chain();
        let mut base = solve_two_bone_arm(ArmIkInput::from_chain(
            &chain,
            tracked_arm_ik_target(chain.rest, target(), Quat::IDENTITY),
        ))
        .unwrap();
        let axis = crate::arm::finite_normalized(base.wrist - base.elbow).unwrap();
        assert_eq!(
            align_palm_twist(
                &chain,
                &mut base,
                axis.to_array(),
                Quat::IDENTITY,
                1.0,
                &mut PalmRollFilter::default(),
                1.0,
            ),
            None
        );
    }

    #[test]
    fn a_bounded_forearm_roll_returns_after_crossing_the_angle_wrap() {
        let mut filter = PalmRollFilter::default();
        let dt = 1.0 / 60.0;
        // A projected observation winds through a whole turn. It may reach
        // the joint limit, but its hidden state must never retain that turn.
        for tick in 0..720 {
            let measured = shortest_angle_delta(0.0, tick as f32 * std::f32::consts::TAU / 360.0);
            assert!(filter.step(measured, dt).abs() <= FOREARM_ROLL_LIMIT_RAD);
        }
        for _ in 0..120 {
            filter.step(0.0, dt);
        }
        assert!(
            filter.spring.value.abs() < 0.001,
            "the roll stayed stuck after returning to neutral: {}",
            filter.spring.value
        );
        for _ in 0..120 {
            filter.step(-45.0_f32.to_radians(), dt);
        }
        assert!((filter.spring.value + 45.0_f32.to_radians()).abs() < 0.001);
        for _ in 0..120 {
            filter.step(120.0_f32.to_radians(), dt);
        }
        assert!((filter.spring.value - FOREARM_ROLL_LIMIT_RAD).abs() < 0.001);
        for _ in 0..120 {
            filter.step(-120.0_f32.to_radians(), dt);
        }
        assert!(
            (filter.spring.value + FOREARM_ROLL_LIMIT_RAD).abs() < 0.001,
            "the previous limit must not rewrite a negative joint observation as a positive turn"
        );
    }

    #[test]
    fn the_roll_filter_suppresses_tick_noise_but_still_follows_a_turn() {
        // The roll comes from a projection of noisy hand landmarks, so an
        // alternating measurement must not spin the wrist every render tick.
        let dt = 1.0 / 60.0;
        let mut filter = PalmRollFilter::default();
        let mut previous = 0.0_f32;
        let mut worst_noise = 0.0_f32;
        // A landmark that alternates by 20 degrees every tick.
        for tick in 0..180_u32 {
            let noisy: f32 = 50.0 + if tick % 2 == 0 { 10.0 } else { -10.0 };
            let applied = filter.step(noisy.to_radians(), dt);
            worst_noise = worst_noise.max((applied - previous).abs());
            previous = applied;
        }
        assert!(
            worst_noise < 5.0_f32.to_radians(),
            "tick noise must not reach the wrist: worst step {} deg",
            worst_noise.to_degrees()
        );

        // A sustained turn of the same size still reaches the wrist within the
        // filter's time constant.
        let mut settled = PalmRollFilter::default();
        for _ in 0..30 {
            settled.step(45.0_f32.to_radians(), dt);
        }
        let reached = settled.step(45.0_f32.to_radians(), dt).abs().to_degrees();
        assert!(
            reached > 40.0,
            "a sustained roll must settle at the measurement: {reached} deg"
        );
    }
}
