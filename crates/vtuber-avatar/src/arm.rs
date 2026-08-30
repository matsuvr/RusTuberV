//! Rest-space arm-chain data used by the model-adaptive default pose.
//!
//! This module contains the immutable references, pure IK solver, and
//! measurements produced during avatar binding. It performs no ECS writes, so
//! later pose systems do not need to rediscover the hierarchy every frame.

use bevy::prelude::*;

/// The side of a humanoid arm chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmSide {
    /// The model's left arm.
    Left,
    /// The model's right arm.
    Right,
}

/// Entity references for one finger's authored joints.
///
/// These are references only; immutable rest poses for available joints are
/// cached separately in [`FingerRestReferences`]. The optional metacarpal is
/// retained because VRM 1.0 exposes it for the thumb's relaxation chain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FingerJointReferences {
    /// Optional metacarpal joint (normally present for the thumb).
    pub metacarpal: Option<Entity>,
    /// Proximal joint.
    pub proximal: Option<Entity>,
    /// Intermediate joint.
    pub intermediate: Option<Entity>,
    /// Distal joint.
    pub distal: Option<Entity>,
}

impl FingerJointReferences {
    /// Returns whether at least one joint is available.
    #[must_use]
    pub const fn is_present(self) -> bool {
        self.metacarpal.is_some()
            || self.proximal.is_some()
            || self.intermediate.is_some()
            || self.distal.is_some()
    }
}

/// Optional finger references for one arm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FingerReferences {
    /// Thumb joints.
    pub thumb: FingerJointReferences,
    /// Index-finger joints.
    pub index: FingerJointReferences,
    /// Middle-finger joints.
    pub middle: FingerJointReferences,
    /// Ring-finger joints.
    pub ring: FingerJointReferences,
    /// Little-finger joints.
    pub little: FingerJointReferences,
}

impl FingerReferences {
    /// Returns whether any authored finger joint was resolved.
    #[must_use]
    pub const fn has_any(self) -> bool {
        self.thumb.is_present()
            || self.index.is_present()
            || self.middle.is_present()
            || self.ring.is_present()
            || self.little.is_present()
    }
}

/// One authored finger joint with its immutable rest-space pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FingerJointRestBinding {
    /// Finger joint entity.
    pub entity: Entity,
    /// Immutable rest-space pose.
    pub rest: RestSpaceBonePose,
}

/// Optional rest-space finger data for one arm.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FingerRestReferences {
    /// Thumb joints.
    pub thumb: FingerJointRestReferences,
    /// Index-finger joints.
    pub index: FingerJointRestReferences,
    /// Middle-finger joints.
    pub middle: FingerJointRestReferences,
    /// Ring-finger joints.
    pub ring: FingerJointRestReferences,
    /// Little-finger joints.
    pub little: FingerJointRestReferences,
}

impl FingerRestReferences {
    /// Returns whether at least one finger joint has valid rest data.
    #[must_use]
    pub const fn has_any(self) -> bool {
        self.thumb.has_any()
            || self.index.has_any()
            || self.middle.has_any()
            || self.ring.has_any()
            || self.little.has_any()
    }
}

/// Optional rest-space data for the joints of one finger.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FingerJointRestReferences {
    /// Metacarpal joint.
    pub metacarpal: Option<FingerJointRestBinding>,
    /// Proximal joint.
    pub proximal: Option<FingerJointRestBinding>,
    /// Intermediate joint.
    pub intermediate: Option<FingerJointRestBinding>,
    /// Distal joint.
    pub distal: Option<FingerJointRestBinding>,
}

impl FingerJointRestReferences {
    /// Returns whether at least one joint has valid rest data.
    #[must_use]
    pub const fn has_any(self) -> bool {
        self.metacarpal.is_some()
            || self.proximal.is_some()
            || self.intermediate.is_some()
            || self.distal.is_some()
    }
}

/// Candidate entity references read from the VRM root during binding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArmChainReferences {
    /// Optional shoulder entity.
    pub shoulder: Option<Entity>,
    /// Upper-arm entity.
    pub upper_arm: Option<Entity>,
    /// Lower-arm entity.
    pub lower_arm: Option<Entity>,
    /// Hand entity, used as the wrist target/origin.
    pub hand: Option<Entity>,
    /// Optional authored finger joints.
    pub fingers: FingerReferences,
}

/// Rest-space pose for one bone.
///
/// `position` and `global_rotation` come from the immutable
/// `RestGlobalTransform`. `local_rotation` comes from `RestTransform` and is
/// retained for converting future model-space rotations back into local
/// rest-relative deltas without assuming identity bone rotations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RestSpaceBonePose {
    /// Bone origin in the model/rest global space.
    pub position: Vec3,
    /// Bone orientation in the model/rest global space.
    pub global_rotation: Quat,
    /// Authored local rest orientation.
    pub local_rotation: Quat,
}

/// Immutable rest-space geometry for a complete arm chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmRestGeometry {
    /// Optional shoulder pose.
    pub shoulder: Option<RestSpaceBonePose>,
    /// Upper-arm origin and rest orientation.
    pub upper_arm: RestSpaceBonePose,
    /// Elbow origin and rest orientation (the lower-arm origin).
    pub elbow: RestSpaceBonePose,
    /// Wrist origin and rest orientation (the hand origin).
    pub wrist: RestSpaceBonePose,
    /// Rest distance from upper-arm origin to elbow.
    pub upper_arm_length: f32,
    /// Rest distance from elbow to wrist.
    pub forearm_length: f32,
    /// Sum of the two rest arm lengths.
    pub total_arm_length: f32,
}

/// Optional-feature capabilities of a resolved arm chain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArmChainCapabilities {
    /// Whether a valid shoulder reference and rest pose were resolved.
    pub has_shoulder: bool,
    /// Whether at least one valid authored finger reference was resolved.
    pub has_fingers: bool,
}

/// A complete, validated arm chain and its immutable rest-space data.
///
/// The chain is present only when upper arm, lower arm, and hand references
/// all have usable rest data. Optional shoulder and finger data remain
/// explicit capabilities rather than making the avatar binding fail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmChainBinding {
    /// Which side this chain belongs to.
    pub side: ArmSide,
    /// Optional shoulder entity.
    pub shoulder: Option<Entity>,
    /// Upper-arm entity.
    pub upper_arm: Entity,
    /// Lower-arm entity.
    pub lower_arm: Entity,
    /// Hand entity.
    pub hand: Entity,
    /// Optional authored finger entities.
    pub fingers: FingerReferences,
    /// Optional authored finger rest-space poses.
    pub finger_rest: FingerRestReferences,
    /// Immutable rest-space positions/orientations and lengths.
    pub rest: ArmRestGeometry,
    /// Optional shoulder/finger capability flags.
    pub capabilities: ArmChainCapabilities,
}

/// Initial geometry-derived parameters for the default relaxed arm pose.
///
/// The values are intentionally kept in one typed profile so later per-model
/// tuning can validate and replace them without scattering pose constants
/// through the solver. The model basis is VRM's conventional +Y-up, +Z
/// forward basis; therefore -Z is the small rearward elbow-pole offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmPoseProfile {
    /// Angle by which the rest lateral arm direction is lowered toward -Y.
    pub arm_drop_radians: f32,
    /// Desired wrist reach as a fraction of the total arm length.
    pub reach_ratio: f32,
    /// Forward hand offset as a fraction of the total arm length.
    pub forward_hand_offset_ratio: f32,
    /// Rearward elbow-pole offset as a fraction of the total arm length.
    pub elbow_pole_offset_ratio: f32,
    /// Weak shoulder-follow strength.
    pub shoulder_follow_weight: f32,
    /// Relaxed finger curl angle.
    pub finger_curl_radians: f32,
}

impl Default for ArmPoseProfile {
    fn default() -> Self {
        Self {
            arm_drop_radians: 70.0_f32.to_radians(),
            reach_ratio: 0.99,
            forward_hand_offset_ratio: 0.081,
            elbow_pole_offset_ratio: 0.05,
            shoulder_follow_weight: 0.18,
            finger_curl_radians: 10.0_f32.to_radians(),
        }
    }
}

impl ArmPoseProfile {
    /// Validates profile values before they are used to construct a target.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.arm_drop_radians.is_finite()
            && self.arm_drop_radians >= 0.0
            && self.arm_drop_radians <= std::f32::consts::FRAC_PI_2
            && self.reach_ratio.is_finite()
            && self.reach_ratio > 0.0
            && self.reach_ratio <= 1.0
            && self.forward_hand_offset_ratio.is_finite()
            && self.forward_hand_offset_ratio.abs() <= 1.0
            && self.elbow_pole_offset_ratio.is_finite()
            && self.elbow_pole_offset_ratio >= 0.0
            && self.elbow_pole_offset_ratio <= 1.0
            && self.shoulder_follow_weight.is_finite()
            && self.shoulder_follow_weight >= 0.0
            && self.shoulder_follow_weight <= 1.0
            && self.finger_curl_radians.is_finite()
            && self.finger_curl_radians >= 0.0
            && self.finger_curl_radians <= std::f32::consts::FRAC_PI_2
    }
}

/// Version of the persisted per-model arm-profile override format.
pub const ARM_POSE_PROFILE_OVERRIDE_VERSION: u32 = 1;

/// Typed, versioned per-model profile override.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmPoseProfileOverride {
    /// Persisted schema version.
    pub schema_version: u32,
    /// Override for [`ArmPoseProfile::arm_drop_radians`].
    pub arm_drop_radians: f32,
    /// Override for [`ArmPoseProfile::reach_ratio`].
    pub reach_ratio: f32,
    /// Override for [`ArmPoseProfile::forward_hand_offset_ratio`].
    pub forward_hand_offset_ratio: f32,
    /// Override for [`ArmPoseProfile::elbow_pole_offset_ratio`].
    pub elbow_pole_offset_ratio: f32,
    /// Override for [`ArmPoseProfile::shoulder_follow_weight`].
    pub shoulder_follow_weight: f32,
    /// Override for [`ArmPoseProfile::finger_curl_radians`].
    pub finger_curl_radians: f32,
}

impl ArmPoseProfileOverride {
    /// Creates a version-one override from a validated profile.
    #[must_use]
    pub fn from_profile(profile: ArmPoseProfile) -> Self {
        Self {
            schema_version: ARM_POSE_PROFILE_OVERRIDE_VERSION,
            arm_drop_radians: profile.arm_drop_radians,
            reach_ratio: profile.reach_ratio,
            forward_hand_offset_ratio: profile.forward_hand_offset_ratio,
            elbow_pole_offset_ratio: profile.elbow_pole_offset_ratio,
            shoulder_follow_weight: profile.shoulder_follow_weight,
            finger_curl_radians: profile.finger_curl_radians,
        }
    }

    /// Validates and converts a persisted override into runtime profile data.
    pub fn into_profile(self) -> Result<ArmPoseProfile, ArmPoseProfileOverrideError> {
        if self.schema_version != ARM_POSE_PROFILE_OVERRIDE_VERSION {
            return Err(ArmPoseProfileOverrideError::UnsupportedVersion {
                version: self.schema_version,
            });
        }
        let profile = ArmPoseProfile {
            arm_drop_radians: self.arm_drop_radians,
            reach_ratio: self.reach_ratio,
            forward_hand_offset_ratio: self.forward_hand_offset_ratio,
            elbow_pole_offset_ratio: self.elbow_pole_offset_ratio,
            shoulder_follow_weight: self.shoulder_follow_weight,
            finger_curl_radians: self.finger_curl_radians,
        };
        if !profile.is_valid() {
            return Err(ArmPoseProfileOverrideError::OutOfRangeOrNonFinite);
        }
        Ok(profile)
    }
}

/// Validation failures for persisted arm-profile overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmPoseProfileOverrideError {
    /// The persisted schema is not supported.
    UnsupportedVersion {
        /// Encountered schema version.
        version: u32,
    },
    /// One or more values are non-finite or outside the bounded profile.
    OutOfRangeOrNonFinite,
}

impl std::fmt::Display for ArmPoseProfileOverrideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion { version } => {
                write!(f, "unsupported arm pose profile version {version}")
            }
            Self::OutOfRangeOrNonFinite => {
                f.write_str("arm pose profile override is out of range or non-finite")
            }
        }
    }
}

impl std::error::Error for ArmPoseProfileOverrideError {}

/// Desired wrist and elbow-pole positions in model/rest space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmIkTarget {
    /// Desired wrist origin in model/rest space.
    pub wrist: Vec3,
    /// A model-space point that determines the elbow bend side.
    pub elbow_pole: Vec3,
}

/// Inputs for the pure analytic two-bone solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmIkInput {
    /// Upper-arm origin in model/rest space.
    pub shoulder: Vec3,
    /// Rest elbow origin in model/rest space.
    pub rest_elbow: Vec3,
    /// Rest wrist origin in model/rest space.
    pub rest_wrist: Vec3,
    /// Rest upper-arm length.
    pub upper_arm_length: f32,
    /// Rest forearm length.
    pub forearm_length: f32,
    /// Desired wrist and elbow-pole positions.
    pub target: ArmIkTarget,
    /// Authored upper-arm local rest orientation.
    pub upper_arm_rest_rotation: Quat,
    /// Authored lower-arm local rest orientation.
    pub lower_arm_rest_rotation: Quat,
    /// Authored upper-arm model/rest global orientation.
    pub upper_arm_rest_global_rotation: Quat,
    /// Authored lower-arm model/rest global orientation.
    pub lower_arm_rest_global_rotation: Quat,
}

impl ArmIkInput {
    /// Creates solver input from cached immutable arm geometry.
    #[must_use]
    pub fn from_geometry(geometry: ArmRestGeometry, target: ArmIkTarget) -> Self {
        Self {
            shoulder: geometry.upper_arm.position,
            rest_elbow: geometry.elbow.position,
            rest_wrist: geometry.wrist.position,
            upper_arm_length: geometry.upper_arm_length,
            forearm_length: geometry.forearm_length,
            target,
            upper_arm_rest_rotation: geometry.upper_arm.local_rotation,
            lower_arm_rest_rotation: geometry.elbow.local_rotation,
            upper_arm_rest_global_rotation: geometry.upper_arm.global_rotation,
            lower_arm_rest_global_rotation: geometry.elbow.global_rotation,
        }
    }
}

/// Pure solver output for a two-bone arm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmIkSolution {
    /// The target distance after valid two-bone reach clamping.
    pub solved_reach: f32,
    /// Solved elbow origin in model/rest space.
    pub elbow: Vec3,
    /// Solved wrist origin in model/rest space.
    pub wrist: Vec3,
    /// Solved upper-arm model/rest global orientation.
    pub upper_arm_global_rotation: Quat,
    /// Solved lower-arm model/rest global orientation.
    pub lower_arm_global_rotation: Quat,
    /// Solved upper-arm local orientation after applying the rest-relative delta.
    pub upper_arm_local_rotation: Quat,
    /// Solved lower-arm local orientation after applying the rest-relative delta.
    pub lower_arm_local_rotation: Quat,
    /// Upper-arm rest-relative local rotation delta.
    pub upper_arm_delta: Quat,
    /// Lower-arm rest-relative local rotation delta.
    pub lower_arm_delta: Quat,
}

/// Input errors for the analytic arm solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmIkError {
    /// At least one position, length, or orientation was non-finite.
    NonFiniteInput,
    /// A bone length or orientation was too close to zero to solve safely.
    DegenerateGeometry,
    /// The default profile contains an invalid parameter.
    InvalidProfile,
}

impl std::fmt::Display for ArmIkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteInput => f.write_str("arm IK input contains a non-finite value"),
            Self::DegenerateGeometry => f.write_str("arm IK geometry is degenerate"),
            Self::InvalidProfile => f.write_str("arm pose profile is invalid"),
        }
    }
}

impl std::error::Error for ArmIkError {}

/// Builds the default hand-down target from one side's rest geometry.
pub fn default_arm_target(
    chain: &ArmChainBinding,
    profile: ArmPoseProfile,
) -> Result<ArmIkTarget, ArmIkError> {
    if !profile.is_valid() {
        return Err(ArmIkError::InvalidProfile);
    }
    let rest_direction =
        finite_normalized(chain.rest.elbow.position - chain.rest.upper_arm.position)
            .ok_or(ArmIkError::DegenerateGeometry)?;
    let down = -Vec3::Y;
    let angle_to_down = rest_direction.dot(down).clamp(-1.0, 1.0).acos();
    let drop = profile.arm_drop_radians.min(angle_to_down);
    let drop_axis =
        stable_perpendicular(rest_direction, down).ok_or(ArmIkError::DegenerateGeometry)?;
    let dropped_direction = Quat::from_axis_angle(drop_axis, drop) * rest_direction;
    let total = chain.rest.total_arm_length;
    if !total.is_finite() || total <= ARM_IK_EPSILON {
        return Err(ArmIkError::DegenerateGeometry);
    }

    let target = ArmIkTarget {
        wrist: chain.rest.upper_arm.position
            + dropped_direction * (total * profile.reach_ratio)
            + Vec3::Z * (total * profile.forward_hand_offset_ratio),
        elbow_pole: chain.rest.elbow.position
            + Vec3::NEG_Z * (total * profile.elbow_pole_offset_ratio),
    };
    if !target.wrist.is_finite() || !target.elbow_pole.is_finite() {
        return Err(ArmIkError::NonFiniteInput);
    }
    Ok(target)
}

/// Solves a deterministic constant-time analytic two-bone arm IK problem.
///
/// The target is clamped into the valid annulus with an epsilon margin. A
/// pole that is near-zero or collinear with the target falls back first to
/// the authored rest-elbow plane and then to a stable world axis. The output
/// rotations are rest-relative local deltas obtained by conjugating the
/// model-space direction changes with each bone's authored rest-global
/// orientation.
pub fn solve_two_bone_arm(input: ArmIkInput) -> Result<ArmIkSolution, ArmIkError> {
    validate_input(input)?;

    let upper_length = input.upper_arm_length;
    let forearm_length = input.forearm_length;
    let rest_wrist_direction = finite_normalized(input.rest_wrist - input.shoulder)
        .ok_or(ArmIkError::DegenerateGeometry)?;
    let target_vector = input.target.wrist - input.shoulder;
    let target_direction = finite_normalized(target_vector).unwrap_or(rest_wrist_direction);
    let target_distance = target_vector.length();
    if !target_distance.is_finite() {
        return Err(ArmIkError::NonFiniteInput);
    }

    let min_reach = (upper_length - forearm_length).abs() + ARM_IK_EPSILON;
    let max_reach = (upper_length + forearm_length) - ARM_IK_EPSILON;
    if min_reach >= max_reach {
        return Err(ArmIkError::DegenerateGeometry);
    }
    let solved_reach = target_distance.clamp(min_reach, max_reach);
    let wrist = input.shoulder + target_direction * solved_reach;

    let target_pole_vector = input.target.elbow_pole - input.shoulder;
    let pole_direction = project_to_plane(target_pole_vector, target_direction)
        .and_then(finite_normalized)
        .or_else(|| {
            project_to_plane(input.rest_elbow - input.shoulder, target_direction)
                .and_then(finite_normalized)
        })
        .or_else(|| stable_perpendicular(target_direction, Vec3::Y))
        .ok_or(ArmIkError::DegenerateGeometry)?;

    let cosine = ((upper_length * upper_length) + (solved_reach * solved_reach)
        - (forearm_length * forearm_length))
        / (2.0 * upper_length * solved_reach);
    let cosine = cosine.clamp(-1.0, 1.0);
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
    let elbow = input.shoulder
        + target_direction * (cosine * upper_length)
        + pole_direction * (sine * upper_length);
    let upper_direction =
        finite_normalized(elbow - input.shoulder).ok_or(ArmIkError::DegenerateGeometry)?;
    let lower_direction = finite_normalized(wrist - elbow).ok_or(ArmIkError::DegenerateGeometry)?;
    let rest_upper_direction = finite_normalized(input.rest_elbow - input.shoulder)
        .ok_or(ArmIkError::DegenerateGeometry)?;
    let rest_lower_direction = finite_normalized(input.rest_wrist - input.rest_elbow)
        .ok_or(ArmIkError::DegenerateGeometry)?;

    let upper_model_delta = rotation_arc(rest_upper_direction, upper_direction);
    let lower_model_delta = rotation_arc(rest_lower_direction, lower_direction);
    let upper_global =
        normalized_or_identity(upper_model_delta * input.upper_arm_rest_global_rotation)?;
    let lower_global =
        normalized_or_identity(lower_model_delta * input.lower_arm_rest_global_rotation)?;
    let upper_delta =
        conjugated_rest_delta(upper_model_delta, input.upper_arm_rest_global_rotation)?;
    // The lower bone is a child of the upper bone. Its local delta must first
    // cancel the model-space rotation already applied to the upper parent;
    // otherwise applying both local deltas would rotate the forearm twice.
    let lower_local_model_delta = upper_model_delta.inverse() * lower_model_delta;
    let lower_delta = conjugated_rest_delta(
        lower_local_model_delta,
        input.lower_arm_rest_global_rotation,
    )?;
    let upper_local = normalized_or_identity(input.upper_arm_rest_rotation * upper_delta)?;
    let lower_local = normalized_or_identity(input.lower_arm_rest_rotation * lower_delta)?;

    Ok(ArmIkSolution {
        solved_reach,
        elbow,
        wrist,
        upper_arm_global_rotation: upper_global,
        lower_arm_global_rotation: lower_global,
        upper_arm_local_rotation: upper_local,
        lower_arm_local_rotation: lower_local,
        upper_arm_delta: upper_delta,
        lower_arm_delta: lower_delta,
    })
}

const ARM_IK_EPSILON: f32 = 1.0e-4;

fn validate_input(input: ArmIkInput) -> Result<(), ArmIkError> {
    let vectors = [
        input.shoulder,
        input.rest_elbow,
        input.rest_wrist,
        input.target.wrist,
        input.target.elbow_pole,
    ];
    if vectors.iter().any(|value| !value.is_finite()) {
        return Err(ArmIkError::NonFiniteInput);
    }
    let rotations = [
        input.upper_arm_rest_rotation,
        input.lower_arm_rest_rotation,
        input.upper_arm_rest_global_rotation,
        input.lower_arm_rest_global_rotation,
    ];
    if rotations.iter().any(|value| !value.is_finite()) {
        return Err(ArmIkError::NonFiniteInput);
    }
    if input.upper_arm_length <= ARM_IK_EPSILON
        || input.forearm_length <= ARM_IK_EPSILON
        || !input.upper_arm_length.is_finite()
        || !input.forearm_length.is_finite()
        || rotations
            .iter()
            .any(|value| value.length_squared() <= ARM_IK_EPSILON)
    {
        return Err(ArmIkError::DegenerateGeometry);
    }
    Ok(())
}

pub(crate) fn finite_normalized(value: Vec3) -> Option<Vec3> {
    let length_squared = value.length_squared();
    if value.is_finite() && length_squared.is_finite() && length_squared > ARM_IK_EPSILON {
        Some(value.normalize())
    } else {
        None
    }
}

fn normalized_or_identity(value: Quat) -> Result<Quat, ArmIkError> {
    if !value.is_finite() || value.length_squared() <= ARM_IK_EPSILON {
        return Err(ArmIkError::DegenerateGeometry);
    }
    Ok(value.normalize())
}

fn project_to_plane(value: Vec3, plane_normal: Vec3) -> Option<Vec3> {
    if !value.is_finite() || !plane_normal.is_finite() {
        return None;
    }
    Some(value - plane_normal * value.dot(plane_normal))
}

fn stable_perpendicular(first: Vec3, second: Vec3) -> Option<Vec3> {
    let cross = first.cross(second);
    if let Some(normalized) = finite_normalized(cross) {
        return Some(normalized);
    }
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    axes.into_iter()
        .filter(|axis| first.dot(*axis).abs() < 0.9)
        .find_map(|axis| finite_normalized(first.cross(axis)))
}

pub(crate) fn rotation_arc(from: Vec3, to: Vec3) -> Quat {
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 1.0 - ARM_IK_EPSILON {
        return Quat::IDENTITY;
    }
    if dot < -1.0 + ARM_IK_EPSILON {
        let axis = stable_perpendicular(from, Vec3::Y)
            .or_else(|| stable_perpendicular(from, Vec3::X))
            .unwrap_or(Vec3::Z);
        return Quat::from_axis_angle(axis, std::f32::consts::PI);
    }
    let cross = from.cross(to);
    let scale = (2.0 * (1.0 + dot)).sqrt();
    let inverse_scale = 1.0 / scale;
    Quat::from_xyzw(
        cross.x * inverse_scale,
        cross.y * inverse_scale,
        cross.z * inverse_scale,
        scale * 0.5,
    )
    .normalize()
}

pub(crate) fn conjugated_rest_delta(
    model_delta: Quat,
    rest_global: Quat,
) -> Result<Quat, ArmIkError> {
    normalized_or_identity(rest_global.inverse() * model_delta * rest_global)
}
