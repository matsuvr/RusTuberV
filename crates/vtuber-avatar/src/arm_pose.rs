//! Runtime composition for the model-adaptive default arm pose.
//!
//! Binding resolves a typed, rest-relative pose once. This module applies that
//! pose after animation and direct body tracking without changing immutable
//! rest components or accumulating the delta from one frame to the next.

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::arm::{
    ArmChainBinding, ArmIkInput, ArmPoseProfile, ArmPoseProfileOverride,
    ArmPoseProfileOverrideError, FingerJointRestBinding, FingerJointRestReferences,
    FingerRestReferences, solve_two_bone_arm,
};
use crate::arm_pipeline::DynamicArmProfileOverride;
use crate::binding::AvatarBinding;
use crate::lifecycle::{ActiveAvatar, AvatarGeneration};
use crate::load::AvatarAssetId;

const ROTATION_MATCH_EPSILON: f32 = 1.0e-6;
const SHOULDER_FOLLOW_MAX_RADIANS: f32 = 5.0_f32.to_radians();
/// Share of a shoulder model-space rotation inherited by the upper arm.
///
/// The hierarchy already carries the shoulder rotation rigidly to its
/// children; this small extra share changes the relative bend so the arm
/// does not rotate as a single rigid stick.
pub(crate) const SHOULDER_TO_UPPER_SHARE: f32 = 0.3;
/// Share of a shoulder model-space rotation inherited by the forearm.
///
/// Smaller than the upper-arm share so motion visibly decays toward the
/// terminal bones instead of stopping at the shoulder.
pub(crate) const SHOULDER_TO_LOWER_SHARE: f32 = 0.15;
/// Normal default-pose transition duration.
pub const DEFAULT_ARM_TRANSITION_SECONDS: f32 = 0.25;
/// Slower return-to-default transition duration.
pub const DEFAULT_ARM_RETURN_SECONDS: f32 = 0.6;

/// In-memory per-model override store.
///
/// The key is the stable imported model identity/content hash. The store is a
/// resource so unloading and reloading an avatar does not lose its override,
/// while a different model ID cannot inherit it.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct ArmPoseOverrideStore {
    overrides: HashMap<String, ArmPoseProfileOverride>,
    dynamic_profiles: HashMap<String, DynamicArmProfileOverride>,
}

impl ArmPoseOverrideStore {
    /// Stores a bounded, versioned override for one model identity.
    pub fn set(
        &mut self,
        model_id: impl Into<String>,
        profile: ArmPoseProfileOverride,
    ) -> Result<(), ArmPoseOverrideStoreError> {
        let model_id = model_id.into();
        if model_id.is_empty() {
            return Err(ArmPoseOverrideStoreError::EmptyModelId);
        }
        profile
            .into_profile()
            .map_err(ArmPoseOverrideStoreError::InvalidProfile)?;
        self.overrides.insert(model_id, profile);
        Ok(())
    }

    /// Stores a bounded, versioned dynamic arm profile.
    pub fn set_dynamic_profile(
        &mut self,
        model_id: impl Into<String>,
        profile: crate::arm_pipeline::DynamicArmProfileOverride,
    ) -> Result<(), ArmPoseOverrideStoreError> {
        let model_id = model_id.into();
        if model_id.is_empty() {
            return Err(ArmPoseOverrideStoreError::EmptyModelId);
        }
        profile
            .into_profile()
            .map_err(ArmPoseOverrideStoreError::InvalidDynamicProfile)?;
        self.dynamic_profiles.insert(model_id, profile);
        Ok(())
    }

    /// Returns the validated legacy static override for a model identity.
    #[must_use]
    pub fn profile_for(&self, model_id: &AvatarAssetId) -> Option<ArmPoseProfile> {
        self.overrides
            .get(&model_id.0)
            .and_then(|profile| profile.into_profile().ok())
    }

    /// Returns the validated dynamic arm profile for a model identity. A
    /// missing entry means the automatic defaults apply; a different model's
    /// entry can never be read through another identity.
    #[must_use]
    pub fn dynamic_profile_for(
        &self,
        model_id: &AvatarAssetId,
    ) -> Option<crate::arm_pipeline::DynamicArmProfile> {
        self.dynamic_profiles
            .get(&model_id.0)
            .and_then(|profile| profile.into_profile().ok())
    }

    /// Removes a model's legacy override so automatic defaults apply.
    pub fn reset(&mut self, model_id: &AvatarAssetId) -> bool {
        self.overrides.remove(&model_id.0).is_some()
    }

    /// Removes a model's dynamic profile so automatic defaults apply.
    pub fn reset_dynamic_profile(&mut self, model_id: &AvatarAssetId) -> bool {
        self.dynamic_profiles.remove(&model_id.0).is_some()
    }

    /// Iterates over validated dynamic profiles for persistence.
    pub fn dynamic_entries(
        &self,
    ) -> impl Iterator<Item = (&str, &crate::arm_pipeline::DynamicArmProfileOverride)> {
        self.dynamic_profiles
            .iter()
            .map(|(model_id, profile)| (model_id.as_str(), profile))
    }

    /// Imports persisted dynamic profiles, retaining only valid entries.
    pub fn import_dynamic_entries<I>(&mut self, entries: I) -> usize
    where
        I: IntoIterator<Item = (String, crate::arm_pipeline::DynamicArmProfileOverride)>,
    {
        let mut accepted = 0;
        for (model_id, profile) in entries {
            if self.set_dynamic_profile(model_id, profile).is_ok() {
                accepted += 1;
            }
        }
        accepted
    }

    /// Returns the number of stored model overrides.
    #[must_use]
    pub fn len(&self) -> usize {
        self.overrides.len()
    }

    /// Returns whether no model overrides are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    /// Iterates over validated entries for application settings persistence.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &ArmPoseProfileOverride)> {
        self.overrides
            .iter()
            .map(|(model_id, profile)| (model_id.as_str(), profile))
    }

    /// Imports entries from a persistence layer, retaining only valid entries.
    ///
    /// The caller can persist the returned map in its chosen application
    /// settings format without exposing unvalidated values to the avatar.
    pub fn import_entries<I>(&mut self, entries: I) -> usize
    where
        I: IntoIterator<Item = (String, ArmPoseProfileOverride)>,
    {
        let mut accepted = 0;
        for (model_id, profile) in entries {
            if self.set(model_id, profile).is_ok() {
                accepted += 1;
            }
        }
        accepted
    }
}

/// Errors returned when a model override cannot be stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmPoseOverrideStoreError {
    /// The stable model identity was empty.
    EmptyModelId,
    /// The profile version or values are invalid.
    InvalidProfile(ArmPoseProfileOverrideError),
    /// The dynamic profile version or values are invalid.
    InvalidDynamicProfile(crate::arm_pipeline::DynamicArmProfileOverrideError),
}

impl std::fmt::Display for ArmPoseOverrideStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModelId => f.write_str("model identity is empty"),
            Self::InvalidProfile(error) => error.fmt(f),
            Self::InvalidDynamicProfile(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ArmPoseOverrideStoreError {}

/// Requests that the active avatar re-resolve its model-specific default pose.
///
/// The application settings layer owns persistence and writes the validated
/// override into [`ArmPoseOverrideStore`] before sending this message. The
/// avatar side then reads the immutable cached binding geometry and routes the
/// new target through the existing compositor.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct ArmPoseProfileChange {
    /// Stable model identity whose profile changed.
    pub model_id: AvatarAssetId,
    /// Whether this change is a reset back to the automatic profile.
    pub return_to_default: bool,
}

/// A resolved default pose for one complete arm chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedArmPose {
    /// Upper-arm entity to receive the local rest-relative delta.
    pub upper_arm: Entity,
    /// Lower-arm entity to receive the local rest-relative delta.
    pub lower_arm: Entity,
    /// Upper-arm local rest-relative rotation.
    pub upper_arm_delta: Quat,
    /// Lower-arm local rest-relative rotation.
    pub lower_arm_delta: Quat,
    /// Optional weak shoulder-follow correction.
    pub shoulder: Option<ResolvedBoneDelta>,
    /// Authored finger curl corrections.
    pub fingers: ResolvedFingerPose,
}

/// One optional bone's local rest-relative correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedBoneDelta {
    /// Bone entity receiving the correction.
    pub entity: Entity,
    /// Local rest-relative correction.
    pub delta: Quat,
}

/// Resolved weak curl corrections for one arm's finger joints.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ResolvedFingerPose {
    /// Thumb corrections.
    pub thumb: ResolvedFingerJointPose,
    /// Index-finger corrections.
    pub index: ResolvedFingerJointPose,
    /// Middle-finger corrections.
    pub middle: ResolvedFingerJointPose,
    /// Ring-finger corrections.
    pub ring: ResolvedFingerJointPose,
    /// Little-finger corrections.
    pub little: ResolvedFingerJointPose,
}

/// Resolved corrections for the joints of one finger.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ResolvedFingerJointPose {
    /// Metacarpal correction.
    pub metacarpal: Option<ResolvedBoneDelta>,
    /// Proximal correction.
    pub proximal: Option<ResolvedBoneDelta>,
    /// Intermediate correction.
    pub intermediate: Option<ResolvedBoneDelta>,
    /// Distal correction.
    pub distal: Option<ResolvedBoneDelta>,
}

/// The typed default arm pose resolved for one avatar generation.
///
/// Each side is independently optional. An incomplete or degenerate arm chain
/// therefore leaves that side untouched without preventing the avatar from
/// becoming ready.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct DefaultArmPose {
    /// Avatar generation this pose belongs to.
    pub generation: AvatarGeneration,
    /// Resolved left-arm pose, when the complete chain is usable.
    pub left: Option<ResolvedArmPose>,
    /// Resolved right-arm pose, when the complete chain is usable.
    pub right: Option<ResolvedArmPose>,
}

impl DefaultArmPose {
    /// Resolves both default arm poses from immutable binding geometry.
    #[must_use]
    pub fn from_chains(
        generation: AvatarGeneration,
        left: Option<ArmChainBinding>,
        right: Option<ArmChainBinding>,
    ) -> Self {
        Self::from_chains_with_profile(generation, left, right, ArmPoseProfile::default())
    }

    /// Resolves both default arm poses using an explicit bounded profile.
    #[must_use]
    pub fn from_chains_with_profile(
        generation: AvatarGeneration,
        left: Option<ArmChainBinding>,
        right: Option<ArmChainBinding>,
        profile: ArmPoseProfile,
    ) -> Self {
        Self {
            generation,
            left: left.and_then(|chain| resolve_chain(chain, profile)),
            right: right.and_then(|chain| resolve_chain(chain, profile)),
        }
    }
}

/// Independently blendable left/right arm-pose transition state.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct ArmPoseBlendState {
    /// Avatar generation this transition belongs to.
    pub generation: AvatarGeneration,
    /// Left-arm source transition.
    pub left: Option<ArmPoseBlendSide>,
    /// Right-arm source transition.
    pub right: Option<ArmPoseBlendSide>,
}

impl ArmPoseBlendState {
    /// Creates a normal frame-rate-independent transition from neutral to the
    /// resolved default pose.
    #[must_use]
    pub fn from_default(default_pose: &DefaultArmPose) -> Self {
        Self {
            generation: default_pose.generation,
            left: default_pose.left.map(|target| {
                ArmPoseBlendSide::new(neutral_pose(target), target, DEFAULT_ARM_TRANSITION_SECONDS)
            }),
            right: default_pose.right.map(|target| {
                ArmPoseBlendSide::new(neutral_pose(target), target, DEFAULT_ARM_TRANSITION_SECONDS)
            }),
        }
    }

    /// Advances both sides by a finite monotonic time delta.
    pub fn advance(&mut self, delta_seconds: f32) {
        if !delta_seconds.is_finite() || delta_seconds < 0.0 {
            return;
        }
        if let Some(left) = &mut self.left {
            left.advance(delta_seconds);
        }
        if let Some(right) = &mut self.right {
            right.advance(delta_seconds);
        }
    }

    /// Returns the current left-arm pose.
    #[must_use]
    pub fn current_left(&self) -> Option<ResolvedArmPose> {
        self.left.map(ArmPoseBlendSide::current)
    }

    /// Returns the current right-arm pose.
    #[must_use]
    pub fn current_right(&self) -> Option<ResolvedArmPose> {
        self.right.map(ArmPoseBlendSide::current)
    }

    /// Starts an independently blendable left-arm transition.
    pub fn transition_left(&mut self, target: ResolvedArmPose, duration_seconds: f32) {
        self.left = Some(ArmPoseBlendSide::new(
            self.current_left().unwrap_or_else(|| neutral_pose(target)),
            target,
            duration_seconds,
        ));
    }

    /// Starts an independently blendable right-arm transition.
    pub fn transition_right(&mut self, target: ResolvedArmPose, duration_seconds: f32) {
        self.right = Some(ArmPoseBlendSide::new(
            self.current_right().unwrap_or_else(|| neutral_pose(target)),
            target,
            duration_seconds,
        ));
    }

    /// Returns the left arm toward its default target using the slower return
    /// profile.
    pub fn return_left_to_default(&mut self, target: ResolvedArmPose) {
        self.transition_left(target, DEFAULT_ARM_RETURN_SECONDS);
    }

    /// Returns the right arm toward its default target using the slower return
    /// profile.
    pub fn return_right_to_default(&mut self, target: ResolvedArmPose) {
        self.transition_right(target, DEFAULT_ARM_RETURN_SECONDS);
    }

    /// Replaces the resolved default target while preserving each side's
    /// current compositor output as the transition source.
    pub fn transition_to_default(&mut self, default_pose: &DefaultArmPose, duration_seconds: f32) {
        self.generation = default_pose.generation;
        self.left = transition_side(self.left, default_pose.left, duration_seconds);
        self.right = transition_side(self.right, default_pose.right, duration_seconds);
    }
}

fn transition_side(
    current: Option<ArmPoseBlendSide>,
    target: Option<ResolvedArmPose>,
    duration_seconds: f32,
) -> Option<ArmPoseBlendSide> {
    target.map(|target| {
        let from = current
            .map(ArmPoseBlendSide::current)
            .unwrap_or_else(|| neutral_pose(target));
        ArmPoseBlendSide::new(from, target, duration_seconds)
    })
}

/// One side of an arm-pose source transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmPoseBlendSide {
    from: ResolvedArmPose,
    target: ResolvedArmPose,
    elapsed_seconds: f32,
    duration_seconds: f32,
}

impl ArmPoseBlendSide {
    /// Creates a side transition from an arbitrary resolved source pose.
    #[must_use]
    pub fn new(from: ResolvedArmPose, target: ResolvedArmPose, duration_seconds: f32) -> Self {
        Self {
            from,
            target,
            elapsed_seconds: 0.0,
            duration_seconds: if duration_seconds.is_finite() {
                duration_seconds.max(0.0)
            } else {
                0.0
            },
        }
    }

    /// Advances this side by a finite delta.
    pub fn advance(&mut self, delta_seconds: f32) {
        if delta_seconds.is_finite() && delta_seconds >= 0.0 {
            self.elapsed_seconds =
                (self.elapsed_seconds + delta_seconds).min(self.duration_seconds);
        }
    }

    /// Returns the shortest-arc interpolated pose at the current time.
    #[must_use]
    pub fn current(self) -> ResolvedArmPose {
        let amount = if self.duration_seconds <= f32::EPSILON {
            1.0
        } else {
            (self.elapsed_seconds / self.duration_seconds).clamp(0.0, 1.0)
        };
        blend_pose(self.from, self.target, amount)
    }
}

fn neutral_pose(pose: ResolvedArmPose) -> ResolvedArmPose {
    ResolvedArmPose {
        upper_arm_delta: Quat::IDENTITY,
        lower_arm_delta: Quat::IDENTITY,
        shoulder: pose.shoulder.map(|bone| ResolvedBoneDelta {
            delta: Quat::IDENTITY,
            ..bone
        }),
        fingers: neutral_fingers(pose.fingers),
        ..pose
    }
}

fn neutral_fingers(fingers: ResolvedFingerPose) -> ResolvedFingerPose {
    ResolvedFingerPose {
        thumb: neutral_finger_joints(fingers.thumb),
        index: neutral_finger_joints(fingers.index),
        middle: neutral_finger_joints(fingers.middle),
        ring: neutral_finger_joints(fingers.ring),
        little: neutral_finger_joints(fingers.little),
    }
}

fn neutral_finger_joints(joints: ResolvedFingerJointPose) -> ResolvedFingerJointPose {
    ResolvedFingerJointPose {
        metacarpal: neutral_bone(joints.metacarpal),
        proximal: neutral_bone(joints.proximal),
        intermediate: neutral_bone(joints.intermediate),
        distal: neutral_bone(joints.distal),
    }
}

fn neutral_bone(bone: Option<ResolvedBoneDelta>) -> Option<ResolvedBoneDelta> {
    bone.map(|bone| ResolvedBoneDelta {
        delta: Quat::IDENTITY,
        ..bone
    })
}

fn blend_pose(from: ResolvedArmPose, target: ResolvedArmPose, amount: f32) -> ResolvedArmPose {
    ResolvedArmPose {
        upper_arm: target.upper_arm,
        lower_arm: target.lower_arm,
        upper_arm_delta: from
            .upper_arm_delta
            .slerp(target.upper_arm_delta, amount)
            .normalize(),
        lower_arm_delta: from
            .lower_arm_delta
            .slerp(target.lower_arm_delta, amount)
            .normalize(),
        shoulder: blend_bone(from.shoulder, target.shoulder, amount),
        fingers: blend_fingers(from.fingers, target.fingers, amount),
    }
}

fn blend_bone(
    from: Option<ResolvedBoneDelta>,
    target: Option<ResolvedBoneDelta>,
    amount: f32,
) -> Option<ResolvedBoneDelta> {
    let entity = target.or(from)?.entity;
    let from_delta = from.map_or(Quat::IDENTITY, |bone| bone.delta);
    let target_delta = target.map_or(Quat::IDENTITY, |bone| bone.delta);
    Some(ResolvedBoneDelta {
        entity,
        delta: from_delta.slerp(target_delta, amount).normalize(),
    })
}

fn blend_fingers(
    from: ResolvedFingerPose,
    target: ResolvedFingerPose,
    amount: f32,
) -> ResolvedFingerPose {
    ResolvedFingerPose {
        thumb: blend_finger_joints(from.thumb, target.thumb, amount),
        index: blend_finger_joints(from.index, target.index, amount),
        middle: blend_finger_joints(from.middle, target.middle, amount),
        ring: blend_finger_joints(from.ring, target.ring, amount),
        little: blend_finger_joints(from.little, target.little, amount),
    }
}

fn blend_finger_joints(
    from: ResolvedFingerJointPose,
    target: ResolvedFingerJointPose,
    amount: f32,
) -> ResolvedFingerJointPose {
    ResolvedFingerJointPose {
        metacarpal: blend_bone(from.metacarpal, target.metacarpal, amount),
        proximal: blend_bone(from.proximal, target.proximal, amount),
        intermediate: blend_bone(from.intermediate, target.intermediate, amount),
        distal: blend_bone(from.distal, target.distal, amount),
    }
}

fn resolve_chain(
    chain: ArmChainBinding,
    profile: crate::arm::ArmPoseProfile,
) -> Option<ResolvedArmPose> {
    // Binding-time resolution goes through the typed pipeline so the legacy
    // static source stays a single, explicitly demoted stage (Issue #176).
    let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
        chain.side,
        &chain.rest,
        None,
        None,
        None,
    );
    let input = crate::arm_pipeline::ArmPipelineInput::binding_time(&chain, &motion, profile);
    crate::arm_pipeline::resolve_arm_pose(
        &input,
        crate::arm_pipeline::ArmPoseSourceKind::LegacyStatic,
    )
    .ok()?
    .map(|(pose, _)| pose)
}

/// Stage 2..=4 of the arm pipeline: analytic two-bone solve plus the
/// shoulder-follow and finger stages that produce final rest-relative deltas.
///
/// The hand target is produced by an earlier pipeline stage; this function
/// never generates one itself. `Ok(None)` means the solved pose was degenerate
/// (non-finite or identity) and the side should stay untouched.
pub(crate) fn solve_stage(
    chain: &ArmChainBinding,
    profile: crate::arm::ArmPoseProfile,
    target: &crate::arm::ArmIkTarget,
    twist_relax: Option<&crate::arm_pipeline::TwistRelaxParams<'_>>,
) -> Result<Option<ResolvedArmPose>, crate::arm_pipeline::ArmPipelineError> {
    let input = ArmIkInput::from_geometry(chain.rest, *target);
    let mut solution =
        solve_two_bone_arm(input).map_err(crate::arm_pipeline::ArmPipelineError::Solve)?;
    // Stage 3b: the coronal descent limit keeps the arm out of the torso
    // regardless of which source produced the hand target. Raising the arm
    // and forward/backward swing stay free.
    crate::arm_pipeline::clamp_upper_arm_swing(
        &mut solution,
        &input,
        crate::arm_pipeline::MAX_ARM_DROP_RADIANS,
    );
    if let Some(params) = twist_relax {
        crate::arm_pipeline::relax_forearm_twist(&mut solution, params)?;
    }
    if !solution.upper_arm_delta.is_finite()
        || !solution.lower_arm_delta.is_finite()
        || solution.upper_arm_delta.length_squared() <= f32::EPSILON
        || solution.lower_arm_delta.length_squared() <= f32::EPSILON
    {
        return Ok(None);
    }

    let upper_model_delta = normalized_or_identity(
        solution.upper_arm_global_rotation * chain.rest.upper_arm.global_rotation.inverse(),
    )
    .ok_or(crate::arm_pipeline::ArmPipelineError::DegenerateSolvedPose)?;
    let shoulder = chain
        .shoulder
        .zip(chain.rest.shoulder)
        .and_then(|(entity, rest)| {
            weak_follow_delta(
                upper_model_delta,
                rest.global_rotation,
                profile.shoulder_follow_weight,
            )
            .map(|delta| ResolvedBoneDelta { entity, delta })
        });

    let (mut upper_arm_delta, mut lower_arm_delta) = (
        solution.upper_arm_delta.normalize(),
        solution.lower_arm_delta.normalize(),
    );
    // Downstream propagation: a small share of the shoulder's own
    // model-space rotation reaches the elbow and wrist so the bend follows
    // the shoulder instead of locking into a rigid stick.
    if let Some(shoulder_delta) = shoulder.zip(chain.rest.shoulder) {
        let (rest_shoulder, local) = (shoulder_delta.1, shoulder_delta.0.delta);
        let shoulder_model =
            rest_shoulder.global_rotation * local * rest_shoulder.global_rotation.inverse();
        (upper_arm_delta, lower_arm_delta) = propagate_shoulder_downstream(
            shoulder_model,
            chain.rest.upper_arm.global_rotation,
            chain.rest.elbow.global_rotation,
            upper_arm_delta,
            lower_arm_delta,
        );
    }

    Ok(Some(ResolvedArmPose {
        upper_arm: chain.upper_arm,
        lower_arm: chain.lower_arm,
        upper_arm_delta,
        lower_arm_delta,
        shoulder,
        fingers: resolve_finger_pose(chain.finger_rest, profile.finger_curl_radians),
    }))
}

fn normalized_or_identity(value: Quat) -> Option<Quat> {
    if value.is_finite() && value.length_squared() > f32::EPSILON {
        Some(value.normalize())
    } else {
        None
    }
}

fn weak_follow_delta(model_delta: Quat, rest_global: Quat, weight: f32) -> Option<Quat> {
    let model_delta = normalized_or_identity(model_delta)?;
    let (axis, angle) = model_delta.to_axis_angle();
    let angle = (angle * weight).min(SHOULDER_FOLLOW_MAX_RADIANS);
    let weak_model_delta = Quat::from_axis_angle(axis, angle);
    normalized_or_identity(rest_global.inverse() * weak_model_delta * rest_global)
}

/// Propagates a small share of a shoulder model-space rotation downstream
/// into the upper/lower local rest-relative deltas.
///
/// Every motion, however slight, must reach the terminal bones: without this
/// the solved elbow bend stays locked while the shoulder rotates and the arm
/// reads as a rigid stick. The shares decay toward the wrist
/// ([`SHOULDER_TO_UPPER_SHARE`] > [`SHOULDER_TO_LOWER_SHARE`]). Degenerate
/// input leaves the deltas untouched.
pub(crate) fn propagate_shoulder_downstream(
    shoulder_model_delta: Quat,
    upper_rest_global: Quat,
    lower_rest_global: Quat,
    upper_delta: Quat,
    lower_delta: Quat,
) -> (Quat, Quat) {
    let Some(upper_extra) = fractional_rest_delta(
        shoulder_model_delta,
        upper_rest_global,
        SHOULDER_TO_UPPER_SHARE,
    ) else {
        return (upper_delta, lower_delta);
    };
    let Some(lower_extra) = fractional_rest_delta(
        shoulder_model_delta,
        lower_rest_global,
        SHOULDER_TO_LOWER_SHARE,
    ) else {
        return (upper_delta, lower_delta);
    };
    let upper = (upper_extra * upper_delta).normalize();
    let lower = (lower_extra * lower_delta).normalize();
    if upper.is_finite() && lower.is_finite() {
        (upper, lower)
    } else {
        (upper_delta, lower_delta)
    }
}

fn fractional_rest_delta(model_delta: Quat, rest_global: Quat, share: f32) -> Option<Quat> {
    if !model_delta.is_finite() || !rest_global.is_finite() {
        return None;
    }
    let model_delta = model_delta.normalize();
    let fractional = Quat::IDENTITY.slerp(model_delta, share).normalize();
    normalized_or_identity(rest_global.inverse() * fractional * rest_global)
}

fn resolve_finger_pose(fingers: FingerRestReferences, curl_radians: f32) -> ResolvedFingerPose {
    ResolvedFingerPose {
        thumb: resolve_finger_joints(fingers.thumb, curl_radians),
        index: resolve_finger_joints(fingers.index, curl_radians),
        middle: resolve_finger_joints(fingers.middle, curl_radians),
        ring: resolve_finger_joints(fingers.ring, curl_radians),
        little: resolve_finger_joints(fingers.little, curl_radians),
    }
}

fn resolve_finger_joints(
    finger: FingerJointRestReferences,
    curl_radians: f32,
) -> ResolvedFingerJointPose {
    ResolvedFingerJointPose {
        metacarpal: resolve_finger_joint(finger.metacarpal, finger.proximal, None, curl_radians),
        proximal: resolve_finger_joint(
            finger.proximal,
            finger.intermediate,
            finger.metacarpal,
            curl_radians,
        ),
        intermediate: resolve_finger_joint(
            finger.intermediate,
            finger.distal,
            finger.proximal,
            curl_radians,
        ),
        distal: resolve_finger_joint(finger.distal, None, finger.intermediate, curl_radians),
    }
}

fn resolve_finger_joint(
    joint: Option<FingerJointRestBinding>,
    next: Option<FingerJointRestBinding>,
    previous: Option<FingerJointRestBinding>,
    curl_radians: f32,
) -> Option<ResolvedBoneDelta> {
    let joint = joint?;
    let segment = next
        .map(|next| next.rest.position - joint.rest.position)
        .or_else(|| previous.map(|previous| joint.rest.position - previous.rest.position))?;
    let segment_direction = finite_normalized(segment)?;
    // Prefer authored local Z/Y axes, projected off the finger segment. This
    // gives a bend axis for the common straight +X finger without assuming a
    // universal world Euler axis, while still handling unusual authored axes.
    let axis = [
        joint.rest.global_rotation * Vec3::Z,
        joint.rest.global_rotation * Vec3::Y,
        joint.rest.global_rotation * Vec3::X,
    ]
    .into_iter()
    .map(|candidate| candidate - segment_direction * candidate.dot(segment_direction))
    .find_map(finite_normalized)?;
    let model_delta = Quat::from_axis_angle(axis, curl_radians);
    let delta = normalized_or_identity(
        joint.rest.global_rotation.inverse() * model_delta * joint.rest.global_rotation,
    )?;
    Some(ResolvedBoneDelta {
        entity: joint.entity,
        delta,
    })
}

fn finite_normalized(value: Vec3) -> Option<Vec3> {
    let length_squared = value.length_squared();
    if value.is_finite() && length_squared.is_finite() && length_squared > f32::EPSILON {
        Some(value.normalize())
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct DefaultArmPoseBoneState {
    base: Quat,
    last_delta: Quat,
    initialized: bool,
}

impl Default for DefaultArmPoseBoneState {
    fn default() -> Self {
        Self {
            base: Quat::IDENTITY,
            last_delta: Quat::IDENTITY,
            initialized: false,
        }
    }
}

/// Applies the resolved default arm pose after animation and direct tracking.
///
/// The state detects an animation change by comparing the current transform to
/// the previous composed output. This makes the operation stable across
/// frames while still allowing an animation system to provide a new base pose.
/// The affected subtree is then propagated through its actual `ChildOf` path,
/// including intermediate nodes, before VRM gaze and constraints execute.
#[allow(clippy::type_complexity)]
pub fn apply_default_arm_pose(
    mut roots: Query<
        (
            &AvatarBinding,
            &DefaultArmPose,
            Option<&mut ArmPoseBlendState>,
            Option<&crate::arm_pipeline::DynamicArmTargets>,
        ),
        With<ActiveAvatar>,
    >,
    mut transforms: Query<(&mut Transform, &mut GlobalTransform)>,
    child_ofs: Query<&ChildOf>,
    children: Query<&Children>,
    time: Res<Time>,
    mut bone_states: Local<HashMap<Entity, DefaultArmPoseBoneState>>,
) {
    bone_states.retain(|entity, _| transforms.contains(*entity));

    for (binding, pose, blend_state, dynamic) in roots.iter_mut() {
        if pose.generation != binding.generation {
            continue;
        }

        // The Issue #168 virtual-hand source is authoritative per side when
        // its per-frame resolution exists for this generation; sides without
        // a dynamic resolution keep the static default pose/blend fallback.
        let mut resolved_poses: [Option<ResolvedArmPose>; 2] =
            if let Some(mut blend_state) = blend_state {
                if blend_state.generation != binding.generation {
                    continue;
                }
                blend_state.advance(time.delta_secs());
                [blend_state.current_left(), blend_state.current_right()]
            } else {
                [pose.left, pose.right]
            };
        if let Some(targets) = dynamic.filter(|t| t.generation == Some(binding.generation)) {
            if targets.left.is_some() {
                resolved_poses[0] = targets.left;
            }
            if targets.right.is_some() {
                resolved_poses[1] = targets.right;
            }
        }

        for resolved in resolved_poses.into_iter().flatten() {
            let mut any_changed = false;
            if let Some(shoulder) = resolved.shoulder {
                any_changed |= apply_delta(
                    shoulder.entity,
                    shoulder.delta,
                    &mut transforms,
                    &mut bone_states,
                );
            }
            any_changed |= apply_delta(
                resolved.upper_arm,
                resolved.upper_arm_delta,
                &mut transforms,
                &mut bone_states,
            );
            any_changed |= apply_delta(
                resolved.lower_arm,
                resolved.lower_arm_delta,
                &mut transforms,
                &mut bone_states,
            );
            for finger in [
                resolved.fingers.thumb.metacarpal,
                resolved.fingers.thumb.proximal,
                resolved.fingers.thumb.intermediate,
                resolved.fingers.thumb.distal,
                resolved.fingers.index.metacarpal,
                resolved.fingers.index.proximal,
                resolved.fingers.index.intermediate,
                resolved.fingers.index.distal,
                resolved.fingers.middle.metacarpal,
                resolved.fingers.middle.proximal,
                resolved.fingers.middle.intermediate,
                resolved.fingers.middle.distal,
                resolved.fingers.ring.metacarpal,
                resolved.fingers.ring.proximal,
                resolved.fingers.ring.intermediate,
                resolved.fingers.ring.distal,
                resolved.fingers.little.metacarpal,
                resolved.fingers.little.proximal,
                resolved.fingers.little.intermediate,
                resolved.fingers.little.distal,
            ]
            .into_iter()
            .flatten()
            {
                any_changed |= apply_delta(
                    finger.entity,
                    finger.delta,
                    &mut transforms,
                    &mut bone_states,
                );
            }
            if !any_changed {
                continue;
            }

            let mut computed = HashMap::new();
            let mut visiting = HashSet::new();
            if refresh_parent_global(
                resolved
                    .shoulder
                    .map(|bone| bone.entity)
                    .unwrap_or(resolved.upper_arm),
                &mut transforms,
                &child_ofs,
                &mut computed,
                &mut visiting,
            )
            .is_none()
            {
                continue;
            }
            let refresh_root = resolved
                .shoulder
                .map(|bone| bone.entity)
                .unwrap_or(resolved.upper_arm);
            let root_parent_global = child_ofs
                .get(refresh_root)
                .ok()
                .and_then(|child_of| computed.get(&child_of.parent()).copied())
                .unwrap_or(GlobalTransform::IDENTITY);
            refresh_subtree(
                refresh_root,
                root_parent_global,
                &mut transforms,
                &children,
                &mut HashSet::new(),
            );
        }
    }
}

/// Re-resolves the active model's default pose after a validated settings
/// update. This system never traverses or reconstructs the arm hierarchy: it
/// uses the immutable `ArmBinding` geometry cached during binding and hands the
/// result to the existing generation-scoped compositor state.
pub fn apply_arm_pose_profile_changes(
    mut changes: MessageReader<ArmPoseProfileChange>,
    overrides: Res<ArmPoseOverrideStore>,
    mut roots: Query<
        (
            &AvatarBinding,
            &AvatarAssetId,
            &mut DefaultArmPose,
            &mut ArmPoseBlendState,
        ),
        With<ActiveAvatar>,
    >,
) {
    for change in changes.read() {
        for (binding, model_id, mut pose, mut blend_state) in roots.iter_mut() {
            if model_id != &change.model_id {
                continue;
            }

            let profile = overrides.profile_for(model_id).unwrap_or_default();
            let resolved = DefaultArmPose::from_chains_with_profile(
                binding.generation,
                binding.left_arm,
                binding.right_arm,
                profile,
            );
            let duration = if change.return_to_default {
                DEFAULT_ARM_RETURN_SECONDS
            } else {
                DEFAULT_ARM_TRANSITION_SECONDS
            };
            blend_state.transition_to_default(&resolved, duration);
            *pose = resolved;
        }
    }
}

fn apply_delta(
    entity: Entity,
    delta: Quat,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform)>,
    bone_states: &mut HashMap<Entity, DefaultArmPoseBoneState>,
) -> bool {
    if !delta.is_finite() || delta.length_squared() <= f32::EPSILON {
        return false;
    }
    let Ok((mut transform, _global)) = transforms.get_mut(entity) else {
        return false;
    };

    let state = bone_states.entry(entity).or_default();
    let expected_previous = state.base * state.last_delta;
    let animation_changed = !state.initialized
        || !transform.rotation.is_finite()
        || transform.rotation.dot(expected_previous).abs() < 1.0 - ROTATION_MATCH_EPSILON;
    let base = if animation_changed {
        finite_normalized_or(transform.rotation, Quat::IDENTITY)
    } else {
        state.base
    };
    let delta = delta.normalize();
    let output = finite_normalized_or(base * delta, base);
    state.base = base;
    state.last_delta = delta;
    state.initialized = true;
    if output == transform.rotation {
        // The pose is already applied bit-for-bit; reporting no change lets
        // the caller skip the ancestor/subtree global-transform refresh that
        // would only rewrite identical values.
        return false;
    }
    transform.rotation = output;
    true
}

fn finite_normalized_or(value: Quat, fallback: Quat) -> Quat {
    if value.is_finite() && value.length_squared() > f32::EPSILON {
        value.normalize()
    } else {
        fallback
    }
}

fn refresh_parent_global(
    entity: Entity,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform)>,
    child_ofs: &Query<&ChildOf>,
    computed: &mut HashMap<Entity, GlobalTransform>,
    visiting: &mut HashSet<Entity>,
) -> Option<GlobalTransform> {
    if let Some(global) = computed.get(&entity) {
        return Some(*global);
    }
    if !visiting.insert(entity) {
        return None;
    }

    let parent_global = match child_ofs.get(entity) {
        Ok(child_of) => {
            refresh_parent_global(child_of.parent(), transforms, child_ofs, computed, visiting)?
        }
        Err(_) => GlobalTransform::IDENTITY,
    };
    let Ok((transform, mut global)) = transforms.get_mut(entity) else {
        visiting.remove(&entity);
        return None;
    };
    *global = parent_global.mul_transform(*transform);
    let result = *global;
    computed.insert(entity, result);
    visiting.remove(&entity);
    Some(result)
}

fn refresh_subtree(
    entity: Entity,
    parent_global: GlobalTransform,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform)>,
    children: &Query<&Children>,
    visited: &mut HashSet<Entity>,
) {
    if !visited.insert(entity) {
        return;
    }
    let current_global = {
        let Ok((transform, mut global)) = transforms.get_mut(entity) else {
            return;
        };
        *global = parent_global.mul_transform(*transform);
        *global
    };

    if let Ok(child_entities) = children.get(entity) {
        for child in child_entities.iter() {
            refresh_subtree(child, current_global, transforms, children, visited);
        }
    }
}
