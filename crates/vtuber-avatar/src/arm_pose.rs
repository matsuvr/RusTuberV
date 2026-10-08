//! Runtime composition for the model-adaptive default arm pose.
//!
//! The constrained runtime admits a rest-relative path. This module applies that
//! pose after animation and direct body tracking without changing immutable
//! rest components or accumulating the delta from one frame to the next.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::arm::{
    ArmChainBinding, ArmPoseProfile, ArmPoseProfileOverride, ArmPoseProfileOverrideError,
    FingerJointRestBinding, FingerJointRestReferences,
};
use crate::arm_pipeline::DynamicArmProfileOverride;
use crate::binding::AvatarBinding;
use crate::lifecycle::ActiveAvatar;
use crate::load::AvatarAssetId;

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

    /// Returns the validated rest-pose override for a model identity.
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

    /// Removes a model's rest-pose override so automatic defaults apply.
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

    /// Returns the number of rest-pose overrides, excluding dynamic profiles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.overrides.len()
    }

    /// Returns whether no rest-pose overrides are stored, ignoring dynamic profiles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    /// Iterates over validated rest-pose overrides for settings persistence.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &ArmPoseProfileOverride)> {
        self.overrides
            .iter()
            .map(|(model_id, profile)| (model_id.as_str(), profile))
    }

    /// Imports entries from a persistence layer, retaining only valid entries.
    ///
    /// Returns the number of successfully accepted entries, including
    /// replacements. Stored static overrides can be read with [`Self::entries`].
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
    /// Optional hand (wrist) local rest-relative rotation.
    ///
    /// Only an explicit hand target writes this; the default and virtual poses
    /// leave the hand at its authored rest-relative pose.
    pub hand: Option<ResolvedBoneDelta>,
    /// Optional shoulder-girdle rotation.
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

pub(crate) fn resolve_finger_pose(
    chain: &ArmChainBinding,
    curl_radians: f32,
) -> ResolvedFingerPose {
    let Some(normal) = crate::arm::rest_palm_normal(chain) else {
        return ResolvedFingerPose::default();
    };
    let fingers = chain.finger_rest;
    let curl_radians = signed_finger_curl(chain.side, curl_radians);
    ResolvedFingerPose {
        thumb: crate::thumb::flexion_deltas(chain, [curl_radians; 2]),
        index: resolve_finger_joints(fingers.index, curl_radians, normal),
        middle: resolve_finger_joints(fingers.middle, curl_radians, normal),
        ring: resolve_finger_joints(fingers.ring, curl_radians, normal),
        little: resolve_finger_joints(fingers.little, curl_radians, normal),
    }
}

/// The palm-frame normal reverses across the anatomical sides. Positive
/// profile curl therefore needs the same handed sign as observed finger flexion.
pub(crate) fn signed_finger_curl(side: crate::arm::ArmSide, curl: f32) -> f32 {
    match side {
        crate::arm::ArmSide::Left => -curl,
        crate::arm::ArmSide::Right => curl,
    }
}

fn resolve_finger_joints(
    finger: FingerJointRestReferences,
    curl_radians: f32,
    normal: Vec3,
) -> ResolvedFingerJointPose {
    ResolvedFingerJointPose {
        metacarpal: finger.metacarpal.map(|joint| ResolvedBoneDelta {
            entity: joint.entity,
            delta: Quat::IDENTITY,
        }),
        proximal: resolve_finger_joint(
            finger.proximal,
            finger.intermediate.or(finger.distal),
            finger.metacarpal,
            curl_radians,
            normal,
        ),
        intermediate: resolve_finger_joint(
            finger.intermediate,
            finger.distal,
            finger.proximal,
            curl_radians,
            normal,
        ),
        distal: resolve_finger_joint(
            finger.distal,
            None,
            finger.intermediate.or(finger.proximal),
            curl_radians,
            normal,
        ),
    }
}
/// Builds the rest-relative rotation for one finger joint's flexion.
///
/// Observed signed flexion and virtual curl use the same rest palm normal.
/// Both paths conjugate the model-space rotation into the joint's rest frame.
pub(crate) fn resolve_finger_joint(
    joint: Option<FingerJointRestBinding>,
    next: Option<FingerJointRestBinding>,
    previous: Option<FingerJointRestBinding>,
    curl_radians: f32,
    bend_toward: Vec3,
) -> Option<ResolvedBoneDelta> {
    let joint = joint?;
    let segment = next
        .map(|next| next.rest.position - joint.rest.position)
        .or_else(|| previous.map(|previous| joint.rest.position - previous.rest.position))?;
    let delta = crate::skeleton::hinge_delta(
        joint.rest.global_rotation,
        segment,
        bend_toward,
        curl_radians,
    )?;
    Some(ResolvedBoneDelta {
        entity: joint.entity,
        delta,
    })
}

/// Writes the admitted joint path relative to immutable authored locals.
/// Animation cannot add an unchecked arm rotation after the constrained solve.
/// The admitted subtree is propagated before its dependent helpers are evaluated.
pub fn apply_default_arm_pose(
    roots: Query<
        (
            &AvatarBinding,
            &crate::arm_pipeline::DynamicArmTargets,
            Option<&crate::node_constraints::NodeConstraintBindings>,
        ),
        With<ActiveAvatar>,
    >,
    mut transforms: Query<(&mut Transform, &mut GlobalTransform)>,
    rests: Query<&bevy_vrm1::prelude::RestTransform>,
    child_ofs: Query<&ChildOf>,
    children: Query<&Children>,
) {
    use crate::node_constraints::NodeConstraintKind;
    for (binding, targets, constraints) in &roots {
        if targets.generation != Some(binding.generation) {
            continue;
        }
        let resolved_poses = [targets.left, targets.right];
        let mut controlled = Vec::new();

        for resolved in resolved_poses.into_iter().flatten() {
            controlled.extend([resolved.upper_arm, resolved.lower_arm]);
            let refresh_root = resolved
                .shoulder
                .map(|b| b.entity)
                .unwrap_or(resolved.upper_arm);
            // The candidate FK uses immutable local offsets, including helper
            // joints. Animation may not translate/scale those links afterward.
            let mut stack = vec![refresh_root];
            let mut any_changed = false;
            while let Some(bone) = stack.pop() {
                if let (Ok(rest), Ok((mut current, _))) =
                    (rests.get(bone), transforms.get_mut(bone))
                    && *current != **rest
                {
                    *current = **rest;
                    any_changed = true;
                }
                if let Ok(descendants) = children.get(bone) {
                    stack.extend(descendants.iter());
                }
            }
            if let Some(shoulder) = resolved.shoulder {
                controlled.push(shoulder.entity);
                any_changed |=
                    apply_delta(shoulder.entity, shoulder.delta, &mut transforms, &rests);
            }
            any_changed |= apply_delta(
                resolved.upper_arm,
                resolved.upper_arm_delta,
                &mut transforms,
                &rests,
            );
            any_changed |= apply_delta(
                resolved.lower_arm,
                resolved.lower_arm_delta,
                &mut transforms,
                &rests,
            );
            if let Some(hand) = resolved.hand {
                controlled.push(hand.entity);
                any_changed |= apply_delta(hand.entity, hand.delta, &mut transforms, &rests);
            }
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
                controlled.push(finger.entity);
                any_changed |= apply_delta(finger.entity, finger.delta, &mut transforms, &rests);
            }
            if !any_changed {
                continue;
            }

            let refresh_root = resolved
                .shoulder
                .map(|bone| bone.entity)
                .unwrap_or(resolved.upper_arm);
            if let Some(global) =
                crate::skeleton::refresh_global(refresh_root, &mut transforms, &child_ofs, None)
            {
                crate::skeleton::refresh_subtree(refresh_root, global, &mut transforms, &children);
            }
        }
        // Upstream constraints ran before the admitted arm pose. Evaluate its
        // dependent helper branches again from this frame's final sources;
        // otherwise reset sleeves/twist bones remain in T-pose. A constraint
        // may never change an ancestor of an admitted anatomical joint.
        for &source in &controlled {
            let Some(destinations) = constraints.and_then(|c| c.0.get(&source)) else {
                continue;
            };
            let (Ok(rest), Ok((local, global))) = (rests.get(source), transforms.get(source))
            else {
                continue;
            };
            let source_rest = rest.rotation;
            let source_rotation = local.rotation;
            let source_position = global.translation();
            let delta = source_rest.inverse() * source_rotation;
            for constraint in destinations {
                let dest = constraint.destination;
                if controlled.iter().any(|&bone| {
                    let mut ancestor = bone;
                    loop {
                        if ancestor == dest {
                            return true;
                        }
                        let Ok(parent) = child_ofs.get(ancestor) else {
                            return false;
                        };
                        ancestor = parent.parent();
                    }
                }) {
                    continue;
                }
                let Ok(rest) = rests.get(dest) else {
                    continue;
                };
                let target = if let NodeConstraintKind::Roll(axis) = constraint.kind {
                    // VRMC_node_constraint: source local -> parent -> helper
                    // local, then remove swing and retain only axial twist.
                    let in_parent = source_rest * delta * source_rest.inverse();
                    let in_dest = rest.rotation.inverse() * in_parent * rest.rotation;
                    rest.rotation
                        * Quat::from_rotation_arc(axis, in_dest * axis).inverse()
                        * in_dest
                } else if let NodeConstraintKind::Aim(axis) = constraint.kind {
                    let Ok(parent) = child_ofs.get(dest) else {
                        continue;
                    };
                    let (Ok((_, parent_global)), Ok((_, global))) =
                        (transforms.get(parent.parent()), transforms.get(dest))
                    else {
                        continue;
                    };
                    let Some(direction) = (source_position - global.translation()).try_normalize()
                    else {
                        continue;
                    };
                    let parent_rotation = parent_global.rotation();
                    let from = parent_rotation * rest.rotation * axis;
                    parent_rotation.inverse()
                        * Quat::from_rotation_arc(from, direction)
                        * parent_rotation
                        * rest.rotation
                } else {
                    rest.rotation * delta
                };
                let Ok((mut local, _)) = transforms.get_mut(dest) else {
                    continue;
                };
                local.rotation = rest.rotation.slerp(target, constraint.weight).normalize();
                if let Some(global) =
                    crate::skeleton::refresh_global(dest, &mut transforms, &child_ofs, None)
                {
                    crate::skeleton::refresh_subtree(dest, global, &mut transforms, &children);
                }
            }
        }
    }
}

fn apply_delta(
    entity: Entity,
    delta: Quat,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform)>,
    rests: &Query<&bevy_vrm1::prelude::RestTransform>,
) -> bool {
    let Ok(rest) = rests.get(entity) else {
        return false;
    };
    let Ok((mut transform, _)) = transforms.get_mut(entity) else {
        return false;
    };
    let output = (rest.rotation * delta).normalize();
    if output == transform.rotation {
        return false;
    }
    transform.rotation = output;
    true
}
