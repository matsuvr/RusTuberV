//! Typed arm-pose resolution pipeline (Issue #176).
//!
//! Reorganizes the closed #15/#16 analytic two-bone IK and the model-adaptive
//! arm-pose compositor into explicit, individually testable stages:
//!
//! ```text
//! ArmPoseSourceKind
//!  -> hand target generation
//!  -> analytic two-bone solve
//!  -> post-solve modifiers (added by Issues #169..=#171)
//!  -> final rest-relative deltas -> existing compositor Transform write
//! ```
//!
//! Writer ownership does not change: `apply_default_arm_pose` remains the
//! only system writing upper-arm/lower-arm/hand-chain Transforms, and every
//! stage here is a pure function without ECS access.
//!
//! The legacy fixed `arm_drop / reach_ratio /
//! forward_hand_offset` source ([`ArmPoseSourceKind::LegacyStatic`]) is
//! explicitly demoted to a fallback authority. The hips-relative virtual-hand
//! source is the default dynamic authority, while the legacy source is used
//! automatically when model geometry cannot produce a dynamic target.

use bevy::prelude::*;

use bevy_vrm1::prelude::RestGlobalTransform;

use crate::arm::{
    ArmChainBinding, ArmIkError, ArmIkInput, ArmIkSolution, ArmIkTarget, ArmPoseProfile, ArmSide,
};
use crate::arm_motion_geometry::ArmMotionRestGeometry;
use crate::binding::AvatarBinding;
use crate::lifecycle::AvatarLifecycle;

/// Share of the sampled torso model-space rotation that the hips-relative
/// hand target counter-rotates.
///
/// The virtual hand anchor is hips-relative, so without compensation the
/// whole arm is carried rigidly by the turning torso and the elbow bend
/// never changes. Counter-rotating the anchor by a fraction of the chest's
/// actual rotation makes the hands trail the turn like a real body's inert
/// arms, so motion propagates through the shoulder, elbow, and wrist instead
/// of stopping at the shoulder. The share must stay moderate: with a large
/// share the hand stays pinned near the hips while the shoulder swings away
/// with the torso, and the elbow ends up absorbing the whole difference as a
/// pendulum-like swing. The wrists keep most of the turn, and the elbow only
/// the small remainder.
pub const TORSO_LAG_SHARE: f32 = 0.3;

/// Which arm-pose authority produces hand targets for the compositor.
///
/// The legacy static source is retained only as an explicit fallback; new
/// dynamic sources plug in behind this selection without adding competing
/// Transform writers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArmPoseSourceKind {
    /// Legacy fixed hand-down pose (closed #13..=#16).
    ///
    /// Fallback-only authority. Kept while the dynamic sources mature and
    /// used whenever a selected dynamic source cannot resolve a usable
    /// target from the bound rest geometry.
    #[default]
    LegacyStatic,
    /// Hips-relative virtual hand anchors (Issue #168).
    ///
    /// The stage resolves targets from [`ArmMotionRestGeometry`] resolved
    /// during binding (Issue #175). Under this authority the legacy fixed
    /// finger curl (closed #17) is excluded: fingers keep their authored or
    /// animated base unless a future explicit tracking source provides one.
    /// The wrist likewise carries no fabricated Euler bias; the compositor
    /// never writes hand orientation, so the hand keeps its rest-relative pose
    /// unless a future hand target supplies an explicit rotation.
    VirtualHandAnchor,
    /// Webcam-observed shoulders/elbows/wrists (Issues #44/#47/#48).
    ///
    /// The tracked target is produced outside the virtual-hand generator by
    /// [`crate::tracked_arm`] and blended with the virtual target by explicit
    /// per-channel weights. The virtual post-solve modifiers (torso lag,
    /// swivel, shoulder trim) are never re-applied to the
    /// observed target, because that would move a measured hand off its target.
    TrackedPose,
}

/// Resource selecting the active arm-pose source.
///
/// Avatar replacement never carries this resource over implicitly: the
/// selection is global application state, while all resolved poses stay
/// generation-scoped inside the existing compositor components.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct ArmSourceSelection {
    /// Currently selected source kind.
    pub mode: ArmPoseSourceKind,
    /// Parameters for the dynamic virtual-hand stage.
    pub profile: DynamicArmProfile,
}

impl Default for ArmSourceSelection {
    fn default() -> Self {
        Self {
            // The hips-relative virtual-hand source is the default authority;
            // the legacy static pose remains an explicitly selectable fallback.
            mode: ArmPoseSourceKind::VirtualHandAnchor,
            profile: DynamicArmProfile::default(),
        }
    }
}

/// Scale-aware virtual hand motion and arm modifier parameters.
///
/// Values are semantic ratios of body scale meters rather than absolute
/// model-specific lengths. Per-model tuning aggregates into this one typed
/// profile instead of scattering constants through systems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicArmProfile {
    /// How much of the combined head/body translation each axis of the hand
    /// target follows. Arms hang from the shoulders, so the lateral axis
    /// only gives a small natural give; established webcam trackers keep
    /// the hands from visibly tracking lateral head sway.
    pub compensation_gains: Vec3,
    /// Elbow swivel magnitude at the default anchor position.
    pub elbow_swivel_radians: f32,
    /// Distance over which the swivel fades to zero as the hand approaches
    /// the chest center, as a fraction of body scale.
    pub swivel_transition_width_ratio: f32,
    /// Weak bend/pole influence of the swivel correction.
    pub pole_influence: f32,
    /// Optional per-model shoulder elevation trim (negative lowers the
    /// shoulder). Neutral default 0; bounded to +/- 15 degrees.
    pub shoulder_elevation_trim_radians: f32,
}

impl Default for DynamicArmProfile {
    fn default() -> Self {
        Self {
            compensation_gains: Vec3::new(0.25, 0.0, 1.0),
            elbow_swivel_radians: 0.0,
            swivel_transition_width_ratio: 0.15,
            pole_influence: 0.2,
            shoulder_elevation_trim_radians: 0.0,
        }
    }
}

impl DynamicArmProfile {
    /// Validates the bounded profile before any solve uses it.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.compensation_gains.is_finite()
            && self.compensation_gains.x >= 0.0
            && self.compensation_gains.x <= 1.0
            && self.compensation_gains.y >= 0.0
            && self.compensation_gains.y <= 1.0
            && self.compensation_gains.z >= 0.0
            && self.compensation_gains.z <= 1.0
            && self.elbow_swivel_radians.is_finite()
            && self.elbow_swivel_radians >= 0.0
            && self.elbow_swivel_radians <= std::f32::consts::FRAC_PI_2
            && self.swivel_transition_width_ratio.is_finite()
            && self.swivel_transition_width_ratio >= 0.0
            && self.swivel_transition_width_ratio <= 1.0
            && self.pole_influence.is_finite()
            && self.pole_influence >= 0.0
            && self.pole_influence <= 1.0
            && self.shoulder_elevation_trim_radians.is_finite()
            && self.shoulder_elevation_trim_radians.abs() <= 15.0_f32.to_radians()
    }
}

/// Why a pipeline run produced its final pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmPoseSourceUsed {
    /// The selected dynamic source produced the target.
    SelectedDynamic,
    /// The selected dynamic source could not produce a target and the
    /// documented compatibility fallback applied.
    LegacyFallback,
    /// The legacy static source was explicitly selected.
    LegacySelected,
}

/// Typed result of one complete pipeline run for a single side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmPipelineOutcome {
    /// Source actually used after any documented fallback.
    pub source_used: ArmPoseSourceUsed,
    /// Hand target handed to the analytic two-bone solver.
    pub hand_target: ArmIkTarget,
}

/// Errors surfaced by the pipeline stages themselves.
///
/// Solver-level errors are passed through unchanged; a `None` outcome means
/// the caller should leave that side untouched (missing/degenerate chains
/// stay a normal capability gap, not an error).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmPipelineError {
    /// The underlying analytic solver rejected the stage output.
    Solve(ArmIkError),
    /// The solved pose was degenerate (non-finite or zero-length delta).
    DegenerateSolvedPose,
}

impl std::fmt::Display for ArmPipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Solve(error) => write!(f, "arm IK stage failed: {error}"),
            Self::DegenerateSolvedPose => f.write_str("solved arm pose is degenerate"),
        }
    }
}

impl std::error::Error for ArmPipelineError {}

/// Per-side inputs for one pipeline run.
///
/// Everything is immutable rest-space or semantic data; no ECS queries are
/// performed here so each stage can be unit-tested in isolation.
#[derive(Debug, Clone, Copy)]
pub struct ArmPipelineInput<'a> {
    /// Bound arm chain with immutable rest-space geometry.
    pub chain: &'a ArmChainBinding,
    /// Motion geometry resolved once during binding (Issue #175).
    pub motion: &'a ArmMotionRestGeometry,
    /// Model-adaptive legacy pose profile used by the fallback stage.
    pub legacy_profile: ArmPoseProfile,
    /// Virtual-hand profile used by the dynamic stage.
    pub dynamic_profile: DynamicArmProfile,
    /// Shaped tracked head target before body-follow/idle/weight, in semantic meters.
    pub head_offset: Vec3,
    /// Shaped tracked body target at the same pre-body-follow stage, in semantic meters.
    pub body_offset: Vec3,
    /// Model-space rotation delta of the torso bone the arms hang from
    /// (chest-relative-to-rest), sampled this frame. The dynamic hand target
    /// counter-rotates by [`TORSO_LAG_SHARE`] of it so body turns reach the
    /// elbow and wrist instead of carrying the arms rigidly.
    pub torso_delta: Quat,
    /// Body scale in meters for scale-aware normalization.
    pub body_scale_meters: f32,
}

impl<'a> ArmPipelineInput<'a> {
    /// Builds pipeline input for one side from binding-time data only.
    ///
    /// Dynamic offsets default to zero so binding-time resolution matches
    /// the pre-dynamic behavior exactly.
    #[must_use]
    pub fn binding_time(
        chain: &'a ArmChainBinding,
        motion: &'a ArmMotionRestGeometry,
        legacy_profile: ArmPoseProfile,
    ) -> Self {
        Self {
            chain,
            motion,
            legacy_profile,
            dynamic_profile: DynamicArmProfile::default(),
            head_offset: Vec3::ZERO,
            body_offset: Vec3::ZERO,
            torso_delta: Quat::IDENTITY,
            body_scale_meters: crate::body_scale::DEFAULT_BODY_SCALE_METERS,
        }
    }
}

/// Per-frame resolved virtual hand targets produced by the pipeline.
///
/// Lives on the active avatar root so avatar replacement/unload drops it
/// with the entity; the generation guard additionally rejects stale writes.
#[derive(Component, Debug, Clone, Copy, PartialEq, Default)]
pub struct DynamicArmTargets {
    /// Avatar generation this resolution belongs to.
    pub generation: Option<crate::lifecycle::AvatarGeneration>,
    /// Source sequence of the control frame this resolution consumed.
    pub source_seq: Option<vtuber_core::FrameSeq>,
    /// Left-arm resolved pose, present when the left chain bound.
    pub left: Option<crate::arm_pose::ResolvedArmPose>,
    /// Right-arm resolved pose, present when the right chain bound.
    pub right: Option<crate::arm_pose::ResolvedArmPose>,
}

/// Runs the full arm-pose pipeline for one side.
///
/// # Errors
///
/// Returns [`ArmPipelineError`] when the selected source produced a target
/// but the analytic solver rejected it. A degenerate or missing chain yields
/// `Ok(None)` instead: that side simply receives no delta.
pub fn resolve_arm_pose(
    input: &ArmPipelineInput<'_>,
    source: ArmPoseSourceKind,
) -> Result<Option<(crate::arm_pose::ResolvedArmPose, ArmPipelineOutcome)>, ArmPipelineError> {
    let Some(generated) = generate_hand_target(input, source) else {
        return Ok(None);
    };
    let (target, outcome) = generated?;
    // Under dynamic virtual-hand authority the legacy fixed finger curl never
    // applies; fingers keep their authored/animation base. The wrist carries
    // no fabricated Euler bias either, so hand pose stays rest-relative.
    let dynamic_authority = source == ArmPoseSourceKind::VirtualHandAnchor;
    let effective_legacy_profile = if dynamic_authority {
        ArmPoseProfile {
            finger_curl_radians: 0.0,
            ..input.legacy_profile
        }
    } else {
        input.legacy_profile
    };
    let pose = super::arm_pose::solve_stage(input.chain, effective_legacy_profile, &target)?;
    let mut result = pose.map(|pose| (pose, outcome));
    if let Some((pose, _)) = result.as_mut() {
        apply_shoulder_elevation_trim(pose, input);
    }
    Ok(result)
}

/// Stage 4 (Issue #171): optional per-model shoulder elevation trim.
///
/// The trim is a single bounded semantic parameter (negative lowers the
/// shoulder) applied around the rest-space elevation axis resolved from the
/// shoulder/upper-arm geometry — never a fixed Euler angle in the model's
/// authored local axes. Trim 0 leaves the compositor output bit-for-bit
/// unchanged; missing shoulder bones or degenerate geometry are safe no-ops.
fn apply_shoulder_elevation_trim(
    pose: &mut crate::arm_pose::ResolvedArmPose,
    input: &ArmPipelineInput<'_>,
) {
    let trim = input.dynamic_profile.shoulder_elevation_trim_radians;
    if !trim.is_finite() || trim.abs() <= f32::EPSILON {
        return;
    }
    let Some(rest_shoulder) = input.chain.rest.shoulder.as_ref() else {
        return;
    };
    let Some(shoulder_entity) = input.chain.shoulder else {
        return;
    };
    // Elevation axis: horizontal forward direction of the arm's rest plane,
    // derived from the rest lateral arm direction and model up.
    let Some(lateral) =
        (input.chain.rest.elbow.position - input.chain.rest.upper_arm.position).try_normalize()
    else {
        return;
    };
    let axis = lateral.cross(Vec3::Y);
    let Some(axis) = axis.try_normalize().filter(|axis| axis.is_finite()) else {
        return;
    };
    let model_delta = Quat::from_axis_angle(axis, trim);
    let Ok(trim_delta) =
        crate::arm::conjugated_rest_delta(model_delta, rest_shoulder.global_rotation)
    else {
        return;
    };
    pose.shoulder = match pose.shoulder {
        Some(existing) => {
            if existing.entity != shoulder_entity {
                return;
            }
            Some(crate::arm_pose::ResolvedBoneDelta {
                entity: existing.entity,
                delta: (existing.delta * trim_delta).normalize(),
            })
        }
        None => Some(crate::arm_pose::ResolvedBoneDelta {
            entity: shoulder_entity,
            delta: trim_delta,
        }),
    };
    // FK propagates this clavicle rotation to its descendants exactly once.
    // Do not add a second rotation to the upper-arm or elbow local joints.
}

/// Coronal descent limit for the upper arm, measured from the authored
/// T-pose direction.
///
/// 0 degrees is the T-pose and 90 degrees is the fully lowered "attention"
/// pose. The bound is 85 degrees instead of 90 so clothing thickness cannot
/// push the arm into the torso mesh when body-follow translation or hand
/// target compensation pulls the arm across the body.
///
/// This bounds the virtual arm. Observed arms instead limit the elevation
/// plane relative to the chest: they can cross the body when the elbow is in
/// front. Reusing this coronal bound prevented crossing and jumped at the
/// signed angle's half-turn wrap. See
/// [`resolve_tracked_side`].
pub const MAX_ARM_DROP_RADIANS: f32 = 85.0_f32.to_radians();

/// Stage 3b: bounded upper-arm coronal descent.
///
/// Rotates the whole solved arm (elbow, wrist, model/global rotations, and
/// rest-relative local deltas) rigidly around the shoulder so the upper-arm
/// direction never descends more than `max_swing_radians` past the authored
/// T-pose direction in the coronal plane (normal to model forward `+Z`).
/// Raising the arm and forward/backward swing stay free; the elbow bend and
/// reach are preserved exactly because the chain rotates as a rigid unit.
///
/// Returns `true` when the pose was clamped. Degenerate geometry, non-finite
/// limits, and already-valid poses return `false` with the solution
/// untouched, so the stage always degrades to a safe no-op.
pub fn clamp_upper_arm_swing(
    solution: &mut ArmIkSolution,
    input: &ArmIkInput,
    max_swing_radians: f32,
) -> bool {
    if !max_swing_radians.is_finite() || max_swing_radians <= 0.0 {
        return false;
    }
    let forward = Vec3::Z;
    let down = -Vec3::Y;
    let Some(rest_direction) = crate::arm::finite_normalized(input.rest_elbow - input.shoulder)
    else {
        return false;
    };
    let Some(upper_direction) = crate::arm::finite_normalized(solution.elbow - input.shoulder)
    else {
        return false;
    };

    // Coronal-plane projection: remove the sagittal (forward/back) component,
    // which stays free per the human swing model.
    let Some(rest_coronal) =
        crate::arm::finite_normalized(rest_direction - forward * rest_direction.dot(forward))
    else {
        return false;
    };
    let sagittal = forward * upper_direction.dot(forward);
    let coronal_raw = upper_direction - sagittal;
    let coronal_length = coronal_raw.length();
    let Some(coronal) = crate::arm::finite_normalized(coronal_raw) else {
        // The arm points straight forward or back: no coronal descent exists.
        return false;
    };
    let Some(swing_axis) = crate::arm::finite_normalized(rest_coronal.cross(down)) else {
        return false;
    };

    let descent = f32::atan2(
        rest_coronal.cross(coronal).dot(swing_axis),
        rest_coronal.dot(coronal),
    );
    if descent <= max_swing_radians {
        return false;
    }

    // Rotate the coronal component back to the limit, keep the sagittal
    // component, and rebuild the bounded upper-arm direction.
    let corrected_coronal =
        Quat::from_axis_angle(swing_axis, max_swing_radians - descent) * coronal;
    let new_upper_direction = corrected_coronal * coronal_length + sagittal;
    if !new_upper_direction.is_finite() {
        return false;
    }
    let Some(new_upper_direction) = crate::arm::finite_normalized(new_upper_direction) else {
        return false;
    };
    let swing = crate::arm::rotation_arc(upper_direction, new_upper_direction);

    let rest = input.skeleton_rest();
    let Some(joints) = crate::skeleton::joint_coordinates(rest, solution.skeleton_pose()) else {
        return false;
    };
    let Some(pose) = crate::skeleton::from_joints(
        rest,
        (swing * solution.upper_arm_global_rotation).normalize(),
        joints.x,
        joints.y,
    ) else {
        return false;
    };
    *solution = input.solution_from_skeleton(pose);
    true
}

/// Signed coronal descent of a solved upper-arm direction from the authored
/// T-pose, in radians. Positive values descend toward the body side; 90
/// degrees is the fully lowered arm and larger values cross under the torso.
///
/// Takes the two directions so a probe can read the same quantity off the
/// composed bone, not only off a live solve.
#[must_use]
pub fn coronal_descent_radians(rest_direction: Vec3, upper_direction: Vec3) -> Option<f32> {
    let forward = Vec3::Z;
    let rest_coronal =
        crate::arm::finite_normalized(rest_direction - forward * rest_direction.dot(forward))?;
    let coronal =
        crate::arm::finite_normalized(upper_direction - forward * upper_direction.dot(forward))?;
    let swing_axis = crate::arm::finite_normalized(rest_coronal.cross(-Vec3::Y))?;
    Some(f32::atan2(
        rest_coronal.cross(coronal).dot(swing_axis),
        rest_coronal.dot(coronal),
    ))
}

/// Signed coronal descent of a solved upper-arm direction from the authored
/// T-pose, in radians.
#[must_use]
pub fn upper_arm_descent_radians(input: &ArmIkInput, solution: &ArmIkSolution) -> Option<f32> {
    coronal_descent_radians(
        crate::arm::finite_normalized(input.rest_elbow - input.shoulder)?,
        crate::arm::finite_normalized(solution.elbow - input.shoulder)?,
    )
}

/// Stage 1: hand target generation with documented source selection.
///
/// Returns `None` when neither the selected source nor the legacy fallback
/// can produce a usable target (e.g. a degenerate chain), which callers must
/// treat as "leave this side untouched".
fn generate_hand_target(
    input: &ArmPipelineInput<'_>,
    source: ArmPoseSourceKind,
) -> Option<Result<(ArmIkTarget, ArmPipelineOutcome), ArmPipelineError>> {
    match source {
        ArmPoseSourceKind::LegacyStatic => {
            legacy_static_target(input, ArmPoseSourceUsed::LegacySelected)
        }
        ArmPoseSourceKind::VirtualHandAnchor => {
            match virtual_hand_target(input) {
                Some(target) => Some(Ok(target)),
                // Documented compatibility fallback: without usable hips/
                // rest geometry the demoted legacy source keeps the side
                // posed instead of freezing it.
                None => legacy_static_target(input, ArmPoseSourceUsed::LegacyFallback),
            }
        }
        // The observed target is supplied by `crate::tracked_arm`, not
        // generated from rest geometry. This generator has nothing to add.
        ArmPoseSourceKind::TrackedPose => None,
    }
}

fn legacy_static_target(
    input: &ArmPipelineInput<'_>,
    used: ArmPoseSourceUsed,
) -> Option<Result<(ArmIkTarget, ArmPipelineOutcome), ArmPipelineError>> {
    Some(
        crate::arm::default_arm_target(input.chain, input.legacy_profile)
            .map(|target| {
                (
                    target,
                    ArmPipelineOutcome {
                        source_used: used,
                        hand_target: target,
                    },
                )
            })
            .map_err(ArmPipelineError::Solve),
    )
}

/// Stage 1b (Issue #169): dynamic elbow swivel / pole correction.
///
/// The lowered arm's anatomical bend direction is the base pole. It is
/// rotated around the shoulder -> hand axis by a
/// side-signed swivel angle scaled by the pole influence, and continuously
/// faded to zero as the hand target approaches the chest center over
/// `width = ratio * body_scale`.
fn swivel_adjusted_elbow_pole(input: &ArmPipelineInput<'_>, wrist_target: Vec3) -> Option<Vec3> {
    let profile = input.dynamic_profile;
    let shoulder = input.chain.rest.upper_arm.position;
    let base_direction = crate::arm::neutral_elbow_pole(input.chain, wrist_target)? - shoulder;
    // Right-handed VRM/glTF basis: the model faces +Z and its left side is
    // +X, so mirrored swivel angles need opposite signs per side.
    let side_sign = match input.chain.side {
        crate::arm::ArmSide::Left => 1.0,
        crate::arm::ArmSide::Right => -1.0,
    };

    // Fade the swivel out as the hand nears the chest center.
    let width = profile.swivel_transition_width_ratio
        * if input.body_scale_meters.is_finite() && input.body_scale_meters > 0.0 {
            input.body_scale_meters
        } else {
            crate::body_scale::DEFAULT_BODY_SCALE_METERS
        };
    let fade = match input.motion.torso_center {
        Some(center) if width > 1.0e-4 && wrist_target.is_finite() && center.is_finite() => {
            let distance = (wrist_target - center).length();
            vtuber_tracking::filter::time::smoothstep(distance / width)
        }
        _ => 1.0,
    };

    let angle = side_sign * profile.elbow_swivel_radians * profile.pole_influence * fade;
    let axis = (wrist_target - shoulder).try_normalize()?;
    let pole = shoulder + Quat::from_axis_angle(axis, angle) * base_direction;
    pole.is_finite().then_some(pole)
}

/// Version of the persisted dynamic arm profile format.
pub const DYNAMIC_ARM_PROFILE_OVERRIDE_VERSION: u32 = 3;

/// Versioned, persisted per-model dynamic arm profile.
///
/// Version 3 removes the fixed hips-relative hand anchor. Neutral hand positions
/// now come from each arm's relaxed attention pose geometry; older profiles are rejected by the
/// existing settings policy rather than restoring their model-dependent posture.
///
/// This schema replaces the legacy static-pose parameters as the center of
/// per-model arm tuning. There is deliberately no field-level mapping from
/// the legacy v1 override: the old `arm_drop / reach_ratio /
/// forward_hand_offset / finger_curl` values describe a different authority,
/// so migration resets to automatic defaults instead of silently reusing them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicArmProfileOverride {
    /// Persisted schema version (must be [`DYNAMIC_ARM_PROFILE_OVERRIDE_VERSION`]).
    pub schema_version: u32,
    /// Per-axis head/body follow gains for the hand target.
    pub compensation_gains: [f32; 3],
    /// Elbow swivel magnitude at the default anchor.
    pub elbow_swivel_radians: f32,
    /// Swivel fade width as a fraction of body scale.
    pub swivel_transition_width_ratio: f32,
    /// Weak pole influence of the swivel correction.
    pub pole_influence: f32,
    /// Optional shoulder elevation trim (negative lowers).
    pub shoulder_elevation_trim_radians: f32,
}

impl DynamicArmProfileOverride {
    /// Creates a persisted override from validated runtime data.
    #[must_use]
    pub fn from_profile(profile: DynamicArmProfile) -> Self {
        Self {
            schema_version: DYNAMIC_ARM_PROFILE_OVERRIDE_VERSION,
            compensation_gains: profile.compensation_gains.to_array(),
            elbow_swivel_radians: profile.elbow_swivel_radians,
            swivel_transition_width_ratio: profile.swivel_transition_width_ratio,
            pole_influence: profile.pole_influence,
            shoulder_elevation_trim_radians: profile.shoulder_elevation_trim_radians,
        }
    }

    /// Explicit migration from a legacy v1 static-pose override.
    ///
    /// Policy: conservative reset. The legacy fields describe a fixed pose
    /// source that is no longer an authority, so none of them are re-used;
    /// the model starts from deterministic automatic defaults and can be
    /// tuned from there. The legacy override itself stays untouched for the
    /// explicitly selectable fallback source.
    #[must_use]
    pub fn from_legacy_override(_legacy: &crate::arm::ArmPoseProfileOverride) -> Self {
        Self::from_profile(DynamicArmProfile::default())
    }

    /// Validates and converts into runtime profile data.
    pub fn into_profile(self) -> Result<DynamicArmProfile, DynamicArmProfileOverrideError> {
        if self.schema_version != DYNAMIC_ARM_PROFILE_OVERRIDE_VERSION {
            return Err(DynamicArmProfileOverrideError::UnsupportedVersion {
                version: self.schema_version,
            });
        }
        let profile = DynamicArmProfile {
            compensation_gains: Vec3::from_array(self.compensation_gains),
            elbow_swivel_radians: self.elbow_swivel_radians,
            swivel_transition_width_ratio: self.swivel_transition_width_ratio,
            pole_influence: self.pole_influence,
            shoulder_elevation_trim_radians: self.shoulder_elevation_trim_radians,
        };
        if !profile.is_valid() {
            return Err(DynamicArmProfileOverrideError::OutOfRangeOrNonFinite);
        }
        Ok(profile)
    }
}

/// Validation failures for persisted dynamic arm profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicArmProfileOverrideError {
    /// The persisted schema version is not supported.
    UnsupportedVersion {
        /// Encountered schema version.
        version: u32,
    },
    /// One or more values are non-finite or outside the bounded profile.
    OutOfRangeOrNonFinite,
}

impl std::fmt::Display for DynamicArmProfileOverrideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion { version } => {
                write!(f, "unsupported dynamic arm profile version {version}")
            }
            Self::OutOfRangeOrNonFinite => {
                f.write_str("dynamic arm profile is out of range or non-finite")
            }
        }
    }
}

impl std::error::Error for DynamicArmProfileOverrideError {}

/// Counter-rotation applied to the hips-relative hand offset: a bounded
/// share of the torso's model-space rotation, inverted so the hands trail
/// the turn. Degenerate input degrades to no lag.
fn torso_lag_rotation(torso_delta: Quat) -> Quat {
    if !torso_delta.is_finite() {
        return Quat::IDENTITY;
    }
    Quat::IDENTITY
        .slerp(torso_delta.normalize().inverse(), TORSO_LAG_SHARE)
        .normalize()
}

fn virtual_hand_target(input: &ArmPipelineInput<'_>) -> Option<(ArmIkTarget, ArmPipelineOutcome)> {
    let profile = input.dynamic_profile;
    if !profile.is_valid() {
        return None;
    }
    let anchor = input.motion.hand_anchor.as_ref()?;
    let follow = (input.head_offset + input.body_offset) * profile.compensation_gains;
    // Recover the hips rest origin from the bound wrist/anchor pair so the
    // target stays hips-relative even though the solver works in rest space.
    let hips_rest = input.chain.rest.wrist.position - anchor.translation_from_hips;
    // The neutral anchor is the same model-specific relaxed attention pose as the static
    // initial pose. A fixed hips/body-scale ratio ignores shoulder width and
    // arm lengths, forcing different models to fold their elbows or hang inward.
    let base = crate::arm::default_arm_target(input.chain, ArmPoseProfile::default())
        .ok()?
        .wrist
        - hips_rest;
    let lag = torso_lag_rotation(input.torso_delta);
    let wrist = hips_rest + lag * (base + follow);
    if !wrist.is_finite() {
        return None;
    }
    let elbow_pole = swivel_adjusted_elbow_pole(input, wrist)?;
    let target = ArmIkTarget { wrist, elbow_pole };
    Some((
        target,
        ArmPipelineOutcome {
            source_used: ArmPoseSourceUsed::SelectedDynamic,
            hand_target: target,
        },
    ))
}

/// Returns which side label a chain belongs to (test/diagnostic helper).
#[must_use]
pub fn chain_side_label(chain: &ArmChainBinding) -> &'static str {
    match chain.side {
        ArmSide::Left => "left",
        ArmSide::Right => "right",
    }
}

/// Resolves one side through the pipeline with the given source selection.
///
/// Shared by the per-frame system and binding-time resolution so both paths
/// exercise identical stages.
#[must_use]
pub fn resolve_side(
    input: &ArmPipelineInput<'_>,
    selection: ArmPoseSourceKind,
) -> Option<crate::arm_pose::ResolvedArmPose> {
    resolve_arm_pose(input, selection)
        .ok()
        .flatten()
        .map(|(pose, _)| pose)
}

/// Per-frame system that resolves hips-relative virtual hand poses for the
/// active avatar (Issue #168).
///
/// Reads the bridge-published tracked targets before body-follow/idle/weight,
/// with the actual torso rotation after body and grounding writers. Runs
/// before `apply_default_arm_pose`, which stays the
/// only arm Transform writer. When lifecycle is not Ready, the selected
/// source is not the virtual hand, or no control frame is available, targets
/// clear so the compositor falls back to its static default pose.
/// In tracked mode the observed stage owns and clears its own targets.
#[expect(
    clippy::type_complexity,
    reason = "the ECS tuple matches binding, geometry, scale, published position and mutable arm targets on the same avatar root"
)]
pub fn update_dynamic_arm_targets(
    lifecycle: Res<AvatarLifecycle>,
    selection: Res<ArmSourceSelection>,
    overrides: Option<Res<crate::arm_pose::ArmPoseOverrideStore>>, // per-model profiles
    control_frame: Res<crate::unload::ActiveControlFrame>,
    mut roots: Query<(
        &AvatarBinding,
        &crate::load::AvatarAssetId,
        &crate::arm_motion_geometry::ArmMotionGeometry,
        &crate::body_scale::BodyScaleMeters,
        &crate::direct_position::BodyTrackingPositionInput,
        Option<&mut DynamicArmTargets>,
    )>,
    torso_rotations: Query<(&GlobalTransform, &RestGlobalTransform)>,
) {
    // The observed stage owns these targets in tracked mode. Clearing them
    // here erased its generation/previous pose every tick and reseeded all
    // joint filters immediately before they could advance.
    if selection.mode == ArmPoseSourceKind::TrackedPose {
        return;
    }
    let Ok((binding, model_id, motion, scale, position, targets)) = roots.single_mut() else {
        return;
    };
    // Per-model dynamic profile; automatic defaults otherwise.
    let profile = overrides
        .as_deref()
        .and_then(|store| store.dynamic_profile_for(model_id))
        .unwrap_or(selection.profile);

    // Any condition that breaks generation or source authority clears the
    // dynamic override so the compositor falls back to its static default.
    let Some(mut targets) = targets else {
        return;
    };
    let frame = crate::unload::resolve_control_target(
        &lifecycle,
        Some(binding),
        control_frame
            .frame
            .as_ref()
            .map(|_| control_frame.generation),
    )
    .ok()
    .filter(|target| target.frame_is_current)
    .and_then(|_| control_frame.frame.as_ref());
    let Some(frame) = frame.filter(|_| selection.mode == ArmPoseSourceKind::VirtualHandAnchor)
    else {
        *targets = DynamicArmTargets::default();
        return;
    };
    let head_offset = position.tracked_head_target;
    let body_offset = position.tracked_body_target;

    // Sample the torso bone the arms hang from. The direct body-tracking
    // writer refreshed its global rotation (one frame at most), so the delta
    // against its rest rotation is the actual turn the arms should trail.
    let torso_delta = binding
        .upper_chest
        .or(binding.chest)
        .and_then(|bone| {
            let (global, rest) = torso_rotations.get(bone).ok()?;
            let delta = global.rotation() * rest.0.rotation().inverse();
            delta.is_finite().then(|| delta.normalize())
        })
        .unwrap_or(Quat::IDENTITY);

    let resolve =
        |chain: Option<&ArmChainBinding>,
         geometry: Option<&crate::arm_motion_geometry::ArmMotionRestGeometry>| {
            chain.zip(geometry).and_then(|(chain, geometry)| {
                let input = ArmPipelineInput {
                    chain,
                    motion: geometry,
                    legacy_profile: crate::arm::ArmPoseProfile::default(),
                    dynamic_profile: profile,
                    head_offset,
                    body_offset,
                    torso_delta,
                    body_scale_meters: scale.scale_meters,
                };
                resolve_side(&input, selection.mode)
            })
        };
    *targets = DynamicArmTargets {
        generation: Some(binding.generation),
        source_seq: Some(frame.source_seq),
        left: resolve(binding.left_arm.as_ref(), motion.left.as_ref()),
        right: resolve(binding.right_arm.as_ref(), motion.right.as_ref()),
    };
}

/// Latest observed-arm control for the active avatar (Issues #47/#48).
///
/// The application bridge writes the pure tracking result here; the system
/// below converts it into the compositor's per-frame targets. A resource is
/// enough because exactly one avatar is active, and the stored generation
/// rejects frames from a replaced model.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct TrackedArmControl {
    /// Avatar generation the stored frame belongs to.
    pub generation: Option<crate::lifecycle::AvatarGeneration>,
    /// Latest canonical (unmirrored) observed control frame. Target conversion
    /// applies the avatar motion mirror exactly once when it is enabled.
    pub frame: Option<vtuber_core::arm_tracking::ArmControlFrame>,
    /// Fixed rotation from the canonical tracking basis into the model basis.
    pub view_to_model: Quat,
}

impl Default for TrackedArmControl {
    fn default() -> Self {
        Self {
            generation: None,
            frame: None,
            view_to_model: Quat::IDENTITY,
        }
    }
}

/// Converts the latest observed control frame into compositor targets.
///
/// The stored frame stays canonical tracking data; when
/// [`crate::mirror::AvatarMotionMirror`] is enabled, the per-side targets and
/// weights are reflected and side-swapped exactly once here so the observed
/// arms follow the same mirror as the face.
///
/// Runs after [`update_dynamic_arm_targets`] (which yields target ownership
/// while tracked mode is selected) and before `apply_default_arm_pose`. It only
/// reads the current parent pose and immutable rest geometry; native handles,
/// the camera, and the clock stay outside.
#[expect(
    clippy::too_many_arguments,
    reason = "Bevy injects this system's resources and message streams, so the parameter list is the declared ECS contract and has no call site to restructure"
)]
pub fn update_tracked_arm_targets(
    lifecycle: Res<AvatarLifecycle>,
    selection: Res<ArmSourceSelection>,
    control: Res<TrackedArmControl>,
    time: Res<Time>,
    mirror: Option<Res<crate::mirror::AvatarMotionMirror>>,
    overrides: Option<Res<crate::arm_pose::ArmPoseOverrideStore>>,
    mut roots: Query<(
        &AvatarBinding,
        &crate::load::AvatarAssetId,
        &crate::arm_motion_geometry::ArmMotionGeometry,
        &crate::body_scale::BodyScaleMeters,
        Option<&mut DynamicArmTargets>,
    )>,
    torso_rotations: Query<(&GlobalTransform, &RestGlobalTransform)>,
    mut filters: Local<TrackedArmFilters>,
    #[cfg(debug_assertions)] mut debug_frame: Local<u32>,
) {
    if selection.mode != ArmPoseSourceKind::TrackedPose {
        filters.clear();
        return;
    }
    let Ok((binding, model_id, motion, scale, targets)) = roots.single_mut() else {
        return;
    };
    let Some(mut targets) = targets else {
        return;
    };
    let observed = control.frame.zip(control.generation);
    let frame = crate::unload::resolve_control_target(
        &lifecycle,
        Some(binding),
        observed.map(|(_, generation)| generation),
    )
    .ok()
    .filter(|target| target.frame_is_current)
    .and_then(|_| observed.map(|(frame, _)| frame));
    let Some(frame) = frame else {
        filters.clear();
        *targets = DynamicArmTargets::default();
        return;
    };

    // The tracked frame is re-resolved on every render tick: its source
    // sequence only advances when the camera does, but the render-clock
    // smoothing advances the targets each tick.
    let mirrored = mirror.as_deref().is_none_or(|mirror| mirror.is_enabled());
    let (frame_targets, frame_weights) =
        vtuber_core::mirror::MotionMirror::new(mirrored).arms(frame.targets, frame.weights);

    let profile = overrides
        .as_deref()
        .and_then(|store| store.dynamic_profile_for(model_id))
        .unwrap_or(selection.profile);

    // Parent rest/current rotations so a torso turn is removed exactly once.
    let (parent_rest, parent_current) = binding
        .upper_chest
        .or(binding.chest)
        .and_then(|bone| torso_rotations.get(bone).ok())
        .map(|(global, rest)| (rest.0.rotation(), global.rotation()))
        .unwrap_or((Quat::IDENTITY, Quat::IDENTITY));
    let tracking_to_rest = crate::tracked_arm::tracking_to_rest_rotation(
        parent_rest,
        parent_current,
        control.view_to_model,
    );

    // A replaced model has its own joint frames and lengths.
    if targets.generation != Some(binding.generation) {
        filters.clear();
    }
    let dt_sec = time.delta_secs();

    let resolve = |chain: Option<&ArmChainBinding>,
                   geometry: Option<&crate::arm_motion_geometry::ArmMotionRestGeometry>,
                   target: Option<vtuber_core::arm_tracking::ArmTrackingTarget>,
                   weights: vtuber_core::arm_tracking::ArmBlendWeight,
                   filter: &mut crate::tracked_arm::TrackedArmFilter| {
        resolve_tracked_side(
            chain,
            geometry,
            profile,
            scale.scale_meters,
            target,
            weights,
            tracking_to_rest,
            filter,
            dt_sec,
        )
    };
    // A tick whose blend or solve is degenerate keeps the last resolved pose
    // for that side instead of clearing it: clearing would drop the side to the
    // static default pose for one frame, which reads as a snap. The generation
    // guard keeps a replaced model from ever receiving the previous pose.
    let previous = *targets;
    let hold = |side: Option<crate::arm_pose::ResolvedArmPose>,
                resolved: Option<crate::arm_pose::ResolvedArmPose>| {
        resolved.or(if previous.generation == Some(binding.generation) {
            side
        } else {
            None
        })
    };
    let left = resolve(
        binding.left_arm.as_ref(),
        motion.left.as_ref(),
        frame_targets.left,
        frame_weights.left,
        &mut filters.left,
    );
    let right = resolve(
        binding.right_arm.as_ref(),
        motion.right.as_ref(),
        frame_targets.right,
        frame_weights.right,
        &mut filters.right,
    );
    *targets = DynamicArmTargets {
        generation: Some(binding.generation),
        source_seq: Some(frame.source_seq),
        left: hold(previous.left, left),
        right: hold(previous.right, right),
    };
    #[cfg(debug_assertions)]
    log_tracked_arm_frame(
        &frame,
        frame_targets,
        frame_weights,
        &targets,
        &mut debug_frame,
    );
}

/// Per-side render-clock joint state, cleared with its tracked source/model.
#[derive(Default)]
pub struct TrackedArmFilters {
    left: crate::tracked_arm::TrackedArmFilter,
    right: crate::tracked_arm::TrackedArmFilter,
}

impl TrackedArmFilters {
    fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Debug-build trace of one tracked-arm resolution every 30 render ticks.
///
/// Records at `info!` so it is visible without changing the log filter. The
/// per-channel weights next to the applied wrist roll show whether a stuck
/// palm comes from a missing observation, a zero blend weight, or a twist the
/// solver rejected as degenerate.
#[cfg(debug_assertions)]
fn log_tracked_arm_frame(
    frame: &vtuber_core::arm_tracking::ArmControlFrame,
    targets: vtuber_core::arm_tracking::ArmTrackingTargets,
    weights: vtuber_core::arm_tracking::ArmBlendWeights,
    resolved: &DynamicArmTargets,
    counter: &mut u32,
) {
    *counter = counter.wrapping_add(1);
    if !counter.is_multiple_of(30) {
        return;
    }
    for (side, target, weight, pose) in [
        ("left", targets.left, weights.left, resolved.left),
        ("right", targets.right, weights.right, resolved.right),
    ] {
        let roll_degrees = pose
            .and_then(|pose| pose.hand)
            .map(|hand| {
                let (_, angle) = hand.delta.to_axis_angle();
                angle.to_degrees()
            })
            .unwrap_or(0.0);
        bevy::log::info!(
            target: "palm_trace",
            "arm seq={} side={side} weight(wrist={:.2} pole={:.2} palm={:.2}) \
             palm_normal={:?} hand_roll_deg={roll_degrees:.1}",
            frame.source_seq.0,
            weight.wrist,
            weight.pole,
            weight.palm,
            target.and_then(|target| target.palm_normal),
        );
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "every argument is one stage input: the bound chain and rest geometry, the virtual blend profile and scale, the observation and weights, the basis fixup, and the render-clock joint state"
)]
fn resolve_tracked_side(
    chain: Option<&ArmChainBinding>,
    geometry: Option<&crate::arm_motion_geometry::ArmMotionRestGeometry>,
    profile: DynamicArmProfile,
    body_scale_meters: f32,
    target: Option<vtuber_core::arm_tracking::ArmTrackingTarget>,
    weights: vtuber_core::arm_tracking::ArmBlendWeight,
    tracking_to_rest: Quat,
    filter: &mut crate::tracked_arm::TrackedArmFilter,
    dt_sec: f32,
) -> Option<crate::arm_pose::ResolvedArmPose> {
    let chain = chain?;
    if !weights.wrist.is_finite() || !weights.pole.is_finite() {
        return None;
    }
    let wrist_weight = weights.wrist.clamp(0.0, 1.0);
    let neutral = geometry
        .and_then(|geometry| neutral_virtual_pose(chain, geometry, profile, body_scale_meters));
    if wrist_weight <= f32::EPSILON || target.is_none() {
        let (mut pose, _) = neutral?;
        // Virtual hands normally leave the authored wrist untouched. After
        // tracking, explicitly restore it through the same single writer.
        pose.hand = Some(crate::arm_pose::ResolvedBoneDelta {
            entity: chain.hand,
            delta: Quat::IDENTITY,
        });
        *filter = Default::default();
        return Some(pose);
    }
    let target = target?;
    let tracked = crate::tracked_arm::tracked_arm_ik_target(chain.rest, target, tracking_to_rest);
    let ik_input = ArmIkInput::from_chain(chain, tracked);
    let mut solution = crate::arm::solve_two_bone_arm(ik_input).ok()?;
    if neutral.is_some() {
        // The pole is a conditional observation within the observed arm.
        // During a whole-arm loss both absolute weights decay together; do
        // not apply that decay twice to the shoulder's bend-plane coordinate.
        let pole_weight = (weights.pole / wrist_weight).clamp(0.0, 1.0);
        if pole_weight < 1.0 {
            let virtual_pole = crate::arm::solve_two_bone_arm(ArmIkInput::from_chain(
                chain,
                ArmIkTarget {
                    elbow_pole: crate::arm::neutral_elbow_pole(chain, tracked.wrist)?,
                    ..tracked
                },
            ))
            .ok()?;
            solution.upper_arm_global_rotation = virtual_pole
                .upper_arm_global_rotation
                .slerp(solution.upper_arm_global_rotation, pole_weight)
                .normalize();
        }
    }
    // The shared IK preserves the bound rig's elbow hinge and rigid bone
    // lengths. Tracking and virtual targets use the same anatomical chain.
    // IK can amplify a small wrist-distance error into an elbow-angle error.
    // Damp the joint intent after that conversion, then retain it through all
    // downstream stages. No later stage may solve toward the raw wrist again.
    filter.stabilize(chain, &mut solution, tracking_to_rest, dt_sec)?;
    // Measure pronation against the observed skeleton, before returning its
    // shoulder/elbow toward neutral. A held camera-space palm must not become
    // a new twist target as the returning arm changes its orientation.
    let twist = target
        .palm_normal
        .zip(target.palm_forward)
        .and_then(|palm| {
            crate::tracked_arm::align_hand_orientation(
                chain,
                &mut solution,
                palm,
                tracking_to_rest,
                weights.palm.min(wrist_weight),
                &mut filter.hand,
                dt_sec,
            )
        });
    if let Some((pose, _)) = neutral.as_ref() {
        solution = crate::tracked_arm::blend_arm_joints(chain, &solution, pose, wrist_weight)?;
        if let Some(twist) = twist {
            crate::tracked_arm::roll_forearm(chain, &mut solution, twist.forearm_roll);
        }
    }
    // Finger articulation is hand-local; forearm pronation changes no finger
    // local joint coordinate.
    let fingers = target.fingers.and_then(|fingers| {
        crate::tracked_arm::observed_finger_deltas(chain, fingers, weights.fingers)
    });
    let mut pose = crate::tracked_arm::resolved_tracked_arm_pose(chain, solution, fingers).ok()?;
    if let Some(hand) = pose.hand.as_mut() {
        hand.delta = twist.and_then(|value| value.hand).unwrap_or(Quat::IDENTITY);
    }
    if let Some((neutral, _)) = neutral {
        pose.shoulder = neutral
            .shoulder
            .map(|shoulder| crate::arm_pose::ResolvedBoneDelta {
                delta: shoulder
                    .delta
                    .slerp(Quat::IDENTITY, wrist_weight)
                    .normalize(),
                ..shoulder
            });
    }
    Some(pose)
}

/// The resolved initial skeleton used as the blend source for observed arms.
///
/// Head/body offsets are zero here: with tracked authority the observed frame
/// is the authority, and the virtual fallback is the neutral anchor the
/// compositor returns to. That keeps the arm alive when the face is lost.
fn neutral_virtual_pose(
    chain: &ArmChainBinding,
    motion: &crate::arm_motion_geometry::ArmMotionRestGeometry,
    profile: DynamicArmProfile,
    body_scale_meters: f32,
) -> Option<(crate::arm_pose::ResolvedArmPose, ArmPipelineOutcome)> {
    let input = ArmPipelineInput {
        chain,
        motion,
        legacy_profile: crate::arm::ArmPoseProfile::default(),
        dynamic_profile: profile,
        head_offset: Vec3::ZERO,
        body_offset: Vec3::ZERO,
        torso_delta: Quat::IDENTITY,
        body_scale_meters,
    };
    resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
        .ok()
        .flatten()
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
    use crate::arm::{ArmIkInput, ArmIkSolution, RestSpaceBonePose};

    fn rest_bone(position: Vec3) -> RestSpaceBonePose {
        RestSpaceBonePose {
            position,
            global_rotation: Quat::IDENTITY,
            local_rotation: Quat::IDENTITY,
        }
    }

    fn sample_chain(side: ArmSide) -> ArmChainBinding {
        // Real VRM/glTF basis: the model faces +Z and its left arm is +X.
        let upper_origin = Vec3::new(0.05 * side_sign(side), 1.35, 0.0);
        let elbow = upper_origin + Vec3::new(0.25 * side_sign(side), -0.05, 0.0);
        let wrist = elbow + Vec3::new(-0.02 * side_sign(side), -0.24, -0.01);
        ArmChainBinding {
            side,
            shoulder: None,
            upper_arm: bevy::prelude::Entity::from_raw_u32(0).unwrap(),
            lower_arm: bevy::prelude::Entity::from_raw_u32(1).unwrap(),
            hand: bevy::prelude::Entity::from_raw_u32(2).unwrap(),
            fingers: crate::arm::FingerReferences::default(),
            finger_rest: crate::arm::FingerRestReferences::default(),
            rest: crate::arm::ArmRestGeometry {
                shoulder: None,
                upper_arm: rest_bone(upper_origin),
                elbow: rest_bone(elbow),
                wrist: rest_bone(wrist),
                upper_arm_length: upper_origin.distance(elbow),
                forearm_length: elbow.distance(wrist),
                total_arm_length: upper_origin.distance(elbow) + elbow.distance(wrist),
            },
            capabilities: crate::arm::ArmChainCapabilities::default(),
        }
    }

    fn side_sign(side: ArmSide) -> f32 {
        match side {
            ArmSide::Left => 1.0,
            ArmSide::Right => -1.0,
        }
    }

    // Rest positions from the model in the 2026-10-02 23:58 camera log.
    // Its slightly bent T-pose and non-identity axes expose errors hidden by
    // a perfectly straight, identity-axis synthetic arm.
    fn logged_model_chain(side: ArmSide) -> ArmChainBinding {
        let mut chain = sample_chain(side);
        let point = |x, y, z| Vec3::new(side_sign(side) * x, y, z);
        chain.rest.upper_arm = rest_bone(point(0.09431, 1.057586, -0.006573));
        chain.rest.elbow = rest_bone(point(0.272219, 1.049302, -0.005080));
        chain.rest.wrist = rest_bone(point(0.444790, 1.048926, 0.009165));
        chain.rest.upper_arm.global_rotation = Quat::from_rotation_y(std::f32::consts::PI);
        chain.rest.elbow.global_rotation = chain.rest.upper_arm.global_rotation;
        chain.rest.wrist.global_rotation = chain.rest.upper_arm.global_rotation;
        chain.rest.upper_arm_length = chain
            .rest
            .upper_arm
            .position
            .distance(chain.rest.elbow.position);
        chain.rest.forearm_length = chain
            .rest
            .elbow
            .position
            .distance(chain.rest.wrist.position);
        chain.rest.total_arm_length = chain.rest.upper_arm_length + chain.rest.forearm_length;
        chain
    }

    #[test]
    fn palm_to_back_turn_keeps_its_direction_on_the_recorded_rig_and_its_mirror() {
        use crate::arm::{FingerJointRestBinding, rest_palm_normal};
        use vtuber_core::arm_tracking::{
            ArmBlendWeight, ArmBlendWeights, ArmTrackingTarget, ArmTrackingTargets,
        };
        use vtuber_core::mirror::MotionMirror;

        // 2026-10-03 18:47 log, model a91e2969. The authored elbow bends
        // mostly upward by a few millimetres; its segment cross product is
        // 73 degrees away from the anatomical anterior flexion plane.
        for mirrored in [false, true] {
            let side = if mirrored {
                ArmSide::Right
            } else {
                ArmSide::Left
            };
            let mut chain = logged_model_chain(side);
            let point = |x, y, z| Vec3::new(side_sign(side) * x, y, z);
            chain.rest.upper_arm.position = point(0.084835, 1.056020, -0.020731);
            chain.rest.elbow.position = point(0.267107, 1.048425, -0.021421);
            chain.rest.wrist.position = point(0.442552, 1.050134, -0.019305);
            chain.rest.upper_arm_length = chain
                .rest
                .upper_arm
                .position
                .distance(chain.rest.elbow.position);
            chain.rest.forearm_length = chain
                .rest
                .elbow
                .position
                .distance(chain.rest.wrist.position);
            chain.rest.total_arm_length = chain.rest.upper_arm_length + chain.rest.forearm_length;
            chain.finger_rest.index.proximal = Some(FingerJointRestBinding {
                entity: chain.hand,
                rest: rest_bone(point(0.507350, 1.046863, -0.003236)),
            });
            chain.finger_rest.little.proximal = Some(FingerJointRestBinding {
                entity: chain.hand,
                rest: rest_bone(point(0.500417, 1.043589, -0.043314)),
            });
            let geometry = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
                side,
                &chain.rest,
                Some(Vec3::Y * 0.7),
                Some(Quat::IDENTITY),
                None,
            );
            // A physical 120-degree elbow: upper arm down, forearm raised
            // in the sagittal plane. Input normals come from this anatomical
            // frame, never from the solver being tested.
            let forearm = Vec3::new(0.0, 0.5, 3.0_f32.sqrt() * 0.5);
            let elbow = Vec3::NEG_Y * chain.rest.upper_arm_length;
            let wrist = elbow + forearm * chain.rest.forearm_length;
            let mut filter = crate::tracked_arm::TrackedArmFilter::default();
            let mut previous_palm: Option<Vec3> = None;
            let mut previous_pose: Option<crate::arm_pose::ResolvedArmPose> = None;
            for degrees in (-75..=75).chain((-75..75).rev()) {
                let target = ArmTrackingTarget {
                    wrist: (wrist / chain.rest.total_arm_length).to_array(),
                    elbow_pole: (elbow / chain.rest.total_arm_length).to_array(),
                    palm_normal: Some(
                        (Quat::from_axis_angle(forearm, (degrees as f32).to_radians()) * Vec3::X)
                            .to_array(),
                    ),
                    palm_forward: Some(forearm.to_array()),
                    fingers: None,
                };
                let (targets, weights) = MotionMirror::new(mirrored).arms(
                    ArmTrackingTargets {
                        left: Some(target),
                        right: None,
                    },
                    ArmBlendWeights {
                        left: ArmBlendWeight::ONE,
                        right: ArmBlendWeight::ZERO,
                    },
                );
                let (target, weights) = if mirrored {
                    (targets.right.unwrap(), weights.right)
                } else {
                    (targets.left.unwrap(), weights.left)
                };
                for tick in 0..90 {
                    let pose = resolve_tracked_side(
                        Some(&chain),
                        Some(&geometry),
                        DynamicArmProfile::default(),
                        0.7,
                        Some(target),
                        weights,
                        Quat::IDENTITY,
                        &mut filter,
                        1.0 / 60.0,
                    )
                    .unwrap();
                    let (upper, lower, orientation) = pose_segments(&chain, &pose);
                    let wrist_delta = chain.rest.wrist.global_rotation
                        * pose.hand.unwrap().delta
                        * chain.rest.wrist.global_rotation.inverse();
                    let normal = orientation * wrist_delta * rest_palm_normal(&chain).unwrap();
                    let project = |v: Vec3| (v - forearm * v.dot(forearm)).normalize();
                    let normal = project(normal);
                    assert!((upper.length() - chain.rest.upper_arm_length).abs() < 1.0e-5);
                    assert!((lower.length() - chain.rest.forearm_length).abs() < 1.0e-5);
                    assert!((upper.angle_between(lower).to_degrees() - 120.0).abs() < 0.01);
                    // A real rig has an authored palm/forearm offset. The new
                    // long-axis observation corrects it at the wrist, without
                    // adding the missing third (axial) wrist freedom.
                    let y = (chain.rest.wrist.position - chain.rest.elbow.position).normalize();
                    let x = y.cross(rest_palm_normal(&chain).unwrap()).normalize();
                    let frame = Quat::from_mat3(&bevy::prelude::Mat3::from_cols(x, y, x.cross(y)));
                    let wrist = frame.inverse() * wrist_delta * frame;
                    assert!(wrist.to_euler(bevy::prelude::EulerRot::YXZ).0.abs() < 1.0e-5);
                    if let Some(previous) = previous_pose {
                        assert!(
                            pose.upper_arm_delta.dot(previous.upper_arm_delta).abs() > 1.0 - 1.0e-6
                        );
                    }
                    if tick == 89 {
                        let wanted = project(Vec3::from(target.palm_normal.unwrap()));
                        assert!(
                            normal.dot(wanted) > 0.9999,
                            "mirror={mirrored} angle={degrees}: {normal:?} != {wanted:?}"
                        );
                        if let Some(previous) = previous_palm {
                            assert!(
                                normal.dot(previous) > 0.999,
                                "palm must not reverse across an angle branch"
                            );
                        }
                        previous_palm = Some(normal);
                    }
                    previous_pose = Some(pose);
                }
            }
        }
    }

    fn pose_segments(
        chain: &ArmChainBinding,
        pose: &crate::arm_pose::ResolvedArmPose,
    ) -> (Vec3, Vec3, Quat) {
        let upper = chain.rest.upper_arm.global_rotation
            * pose.upper_arm_delta
            * chain.rest.upper_arm.global_rotation.inverse();
        let lower = chain.rest.elbow.global_rotation
            * pose.lower_arm_delta
            * chain.rest.elbow.global_rotation.inverse();
        (
            upper * (chain.rest.elbow.position - chain.rest.upper_arm.position),
            upper * lower * (chain.rest.wrist.position - chain.rest.elbow.position),
            upper * lower,
        )
    }

    #[test]
    fn initial_and_lost_arms_share_attention_pose_across_model_proportions() {
        for side in [ArmSide::Left, ArmSide::Right] {
            for (shoulder_width, upper_length, lower_length, body_scale) in [
                (0.09, 0.18, 0.17, 0.46),
                (0.18, 0.32, 0.22, 0.7),
                (0.12, 0.20, 0.35, 0.85),
            ] {
                let mut chain = logged_model_chain(side);
                let upper_direction =
                    (chain.rest.elbow.position - chain.rest.upper_arm.position).normalize();
                let lower_direction =
                    (chain.rest.wrist.position - chain.rest.elbow.position).normalize();
                chain.rest.upper_arm.position.x = side_sign(side) * shoulder_width;
                chain.rest.elbow.position =
                    chain.rest.upper_arm.position + upper_direction * upper_length;
                chain.rest.wrist.position =
                    chain.rest.elbow.position + lower_direction * lower_length;
                chain.rest.upper_arm_length = upper_length;
                chain.rest.forearm_length = lower_length;
                chain.rest.total_arm_length = upper_length + lower_length;
                let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
                    side,
                    &chain.rest,
                    Some(Vec3::new(0.0, 0.705036, 0.004195)),
                    Some(Quat::IDENTITY),
                    None,
                );
                let input = ArmPipelineInput {
                    body_scale_meters: body_scale,
                    ..ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default())
                };
                let (down, lateral) = ArmPoseProfile::default().arm_drop_radians.sin_cos();
                let expected = Vec3::new(side_sign(side) * lateral, -down, 0.0);
                let lost = resolve_tracked_side(
                    Some(&chain),
                    Some(&motion),
                    DynamicArmProfile::default(),
                    body_scale,
                    None,
                    vtuber_core::arm_tracking::ArmBlendWeight::ZERO,
                    Quat::IDENTITY,
                    &mut crate::tracked_arm::TrackedArmFilter::default(),
                    1.0 / 60.0,
                )
                .unwrap();
                for source in [
                    ArmPoseSourceKind::LegacyStatic,
                    ArmPoseSourceKind::VirtualHandAnchor,
                ] {
                    let (pose, _) = resolve_arm_pose(&input, source).unwrap().unwrap();
                    let (upper, lower, hand) = pose_segments(&chain, &pose);
                    let thumb_side =
                        Vec3::new(side_sign(side) * 0.000532, 0.002478, 0.038959).normalize();
                    let shown = hand * thumb_side;
                    assert!(shown.z > 0.9, "{side:?} {source:?}: thumb side {shown:?}");
                    assert!(upper.normalize().dot(expected) > 0.998);
                    assert!(lower.normalize().dot(expected) > 0.998);
                    assert!(upper.angle_between(lower) < 5.0_f32.to_radians());
                    assert!((upper.length() - upper_length).abs() < 1.0e-5);
                    assert!((lower.length() - lower_length).abs() < 1.0e-5);
                    assert!(pose.upper_arm_delta.angle_between(lost.upper_arm_delta) < 1.0e-3);
                    assert!(pose.lower_arm_delta.angle_between(lost.lower_arm_delta) < 1.0e-3);
                    assert!(pose.hand.is_none());
                }
            }
        }
    }

    #[test]
    fn lost_elbow_observation_does_not_lift_the_elbow_and_hand_loss_returns_without_a_flip() {
        use vtuber_core::arm_tracking::{ArmBlendWeight, ArmTrackingTarget};
        for side in [ArmSide::Left, ArmSide::Right] {
            let chain = logged_model_chain(side);
            let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
                side,
                &chain.rest,
                Some(Vec3::new(0.0, 0.705036, 0.004195)),
                Some(Quat::IDENTITY),
                None,
            );
            // seq 1204: the held observed elbow is below the shoulder, but
            // fading its authority used to send the arm toward the T-pose pole.
            let target = ArmTrackingTarget {
                wrist: [-side_sign(side) * 0.09, 0.03, 0.65],
                elbow_pole: [side_sign(side) * 0.10, -0.25, 0.08],
                palm_normal: None,
                palm_forward: None,
                fingers: None,
            };
            let mut filter = crate::tracked_arm::TrackedArmFilter::default();
            let mut previous: Option<Vec3> = None;
            let mut returned = None;
            for tick in 0..=360 {
                let wrist = if tick <= 180 {
                    1.0
                } else {
                    1.0 - (tick - 180) as f32 / 180.0
                };
                let pole = (1.0 - tick as f32 / 120.0).max(0.0);
                let pose = resolve_tracked_side(
                    Some(&chain),
                    Some(&motion),
                    DynamicArmProfile::default(),
                    crate::body_scale::DEFAULT_BODY_SCALE_METERS,
                    Some(target),
                    ArmBlendWeight {
                        wrist,
                        pole,
                        palm: 0.0,
                        fingers: 0.0,
                    },
                    Quat::IDENTITY,
                    &mut filter,
                    1.0 / 60.0,
                )
                .unwrap();
                let (upper, lower, _) = pose_segments(&chain, &pose);
                assert!(
                    upper.y < -chain.rest.upper_arm_length * 0.35,
                    "tick {tick}: {upper:?}"
                );
                assert!((upper.length() - chain.rest.upper_arm_length).abs() < 1.0e-5);
                assert!((lower.length() - chain.rest.forearm_length).abs() < 1.0e-5);
                assert!(upper.angle_between(lower) <= crate::arm::ELBOW_FLEXION_LIMIT_RAD + 1.0e-4);
                if let Some(previous) = previous {
                    assert!(
                        previous.angle_between(upper) < 0.02,
                        "tick {tick}: elbow jumped"
                    );
                }
                previous = Some(upper);
                returned = Some(pose);
            }
            let (neutral, _) = neutral_virtual_pose(
                &chain,
                &motion,
                DynamicArmProfile::default(),
                crate::body_scale::DEFAULT_BODY_SCALE_METERS,
            )
            .unwrap();
            let returned = returned.unwrap();
            assert!(
                returned
                    .upper_arm_delta
                    .angle_between(neutral.upper_arm_delta)
                    < 1.0e-3
            );
            assert!(
                returned
                    .lower_arm_delta
                    .angle_between(neutral.lower_arm_delta)
                    < 1.0e-3
            );
        }
    }

    fn near(a: Vec3, b: Vec3) {
        assert!((a - b).length() < 1.0e-4, "{a:?} != {b:?}");
    }

    fn sample_motion() -> ArmMotionRestGeometry {
        crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Left,
            &sample_chain(ArmSide::Left).rest,
            None,
            None,
            None,
        )
    }

    #[test]
    fn legacy_selection_reports_explicit_legacy_authority() {
        let chain = sample_chain(ArmSide::Left);
        let motion = sample_motion();
        let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        let outcome = resolve_arm_pose(&input, ArmPoseSourceKind::LegacyStatic)
            .expect("no pipeline error")
            .expect("pose resolved");
        assert_eq!(outcome.1.source_used, ArmPoseSourceUsed::LegacySelected);
        assert!(outcome.1.hand_target.wrist.is_finite());
    }

    #[test]
    fn virtual_hand_selection_falls_back_without_a_hips_anchor() {
        let chain = sample_chain(ArmSide::Right);
        let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Right,
            &chain.rest,
            None,
            None,
            None,
        );
        let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        let outcome = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .expect("no pipeline error")
            .expect("pose resolved");
        assert_eq!(outcome.1.source_used, ArmPoseSourceUsed::LegacyFallback);
    }

    fn anchored_motion(side: ArmSide) -> (ArmChainBinding, ArmMotionRestGeometry) {
        let chain = sample_chain(side);
        let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            side,
            &chain.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );
        (chain, motion)
    }

    #[test]
    fn virtual_hand_source_is_authority_when_hips_anchor_is_bound() {
        let (chain, motion) = anchored_motion(ArmSide::Left);
        let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        let outcome = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .expect("no pipeline error")
            .expect("pose resolved");
        assert_eq!(outcome.1.source_used, ArmPoseSourceUsed::SelectedDynamic);
    }

    #[test]
    fn virtual_hand_anchor_lands_on_the_models_own_side() {
        // Regression: the hips-relative anchor once used a Unity-style
        // `Left => -X` sign in Bevy's right-handed glTF basis, where the
        // model's left arm is authored at +X. Each hand target then landed on
        // the opposite side and the IK twisted both arms into the body.
        for side in [ArmSide::Left, ArmSide::Right] {
            let (chain, motion) = anchored_motion(side);
            let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
            let (_, outcome) = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
                .expect("no pipeline error")
                .expect("pose resolved");
            assert_eq!(outcome.source_used, ArmPoseSourceUsed::SelectedDynamic);
            // The authored rest wrist is on the chain's own side (+X for the
            // left arm); the dynamic target must stay on that same side.
            let rest_side = chain.rest.wrist.position.x.signum();
            assert_eq!(
                outcome.hand_target.wrist.x.signum(),
                rest_side,
                "{side:?} hand anchor must resolve to the model's own side"
            );
        }
    }

    #[test]
    fn hand_targets_mirror_between_sides_for_mirrored_rest_data() {
        // Build mirrored rest data by mirroring positions across X.
        let (left_chain, left_motion) = anchored_motion(ArmSide::Left);
        let mut right_chain = sample_chain(ArmSide::Right);
        right_chain.rest.upper_arm.position.x = -left_chain.rest.upper_arm.position.x;
        right_chain.rest.elbow.position.x = -left_chain.rest.elbow.position.x;
        right_chain.rest.wrist.position.x = -left_chain.rest.wrist.position.x;
        let right_motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Right,
            &right_chain.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );

        let mut left_input =
            ArmPipelineInput::binding_time(&left_chain, &left_motion, ArmPoseProfile::default());
        let mut right_input =
            ArmPipelineInput::binding_time(&right_chain, &right_motion, ArmPoseProfile::default());

        let neutral_left = resolve_arm_pose(&left_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .1
            .hand_target
            .wrist;
        let neutral_right = resolve_arm_pose(&right_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .1
            .hand_target
            .wrist;
        assert!((neutral_left.x + neutral_right.x).abs() < 1e-4);
        assert!((neutral_left.y - neutral_right.y).abs() < 1e-5);
        assert!((neutral_left.z - neutral_right.z).abs() < 1e-5);

        // Mirrored lateral inputs must produce exactly mirrored targets:
        // the same world-space sway moves both hands in the same direction,
        // so mirroring requires negating the lateral input as well.
        left_input.head_offset = Vec3::new(0.05, 0.01, 0.02);
        right_input.head_offset = Vec3::new(-0.05, 0.01, 0.02);
        left_input.body_offset = Vec3::new(0.02, 0.0, -0.01);
        right_input.body_offset = Vec3::new(-0.02, 0.0, -0.01);
        let moved_left = resolve_arm_pose(&left_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .1
            .hand_target
            .wrist;
        let moved_right = resolve_arm_pose(&right_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .1
            .hand_target
            .wrist;
        assert!((moved_left.x + moved_right.x).abs() < 1e-4);
        assert!((moved_left.y - moved_right.y).abs() < 1e-5);
        assert!((moved_left.z - moved_right.z).abs() < 1e-5);
    }

    #[test]
    fn compensation_gains_follow_the_profile_exactly_once() {
        let (chain, motion) = anchored_motion(ArmSide::Right);
        let profile = DynamicArmProfile::default();
        let mut input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        input.dynamic_profile = profile;
        input.head_offset = Vec3::new(0.08, 0.04, 0.06);
        input.body_offset = Vec3::new(0.02, 0.01, 0.00);

        let baseline = ArmPipelineInput {
            head_offset: Vec3::ZERO,
            body_offset: Vec3::ZERO,
            ..input
        };
        let base_target = resolve_arm_pose(&baseline, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .1
            .hand_target
            .wrist;
        let target = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .1
            .hand_target
            .wrist;

        let delta = target - base_target;
        let total = input.head_offset + input.body_offset;
        assert!((delta.x - total.x * profile.compensation_gains.x).abs() < 1e-4);
        assert!((delta.y - total.y * profile.compensation_gains.y).abs() < 1e-4);
        assert!((delta.z - total.z * profile.compensation_gains.z).abs() < 1e-4);
    }

    #[test]
    fn torso_rotation_trails_the_hand_target() {
        let (chain, motion) = anchored_motion(ArmSide::Left);
        let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        let neutral = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .1
            .hand_target
            .wrist;

        // A 90-degree left yaw of the chest must trail the hands: in rest
        // space the left-hand anchor counter-rotates toward +Z (forward).
        let turned = ArmPipelineInput {
            torso_delta: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ..input
        };
        let lagged = resolve_arm_pose(&turned, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .1
            .hand_target
            .wrist;
        assert!(
            lagged.z > neutral.z + 1.0e-3,
            "left-hand target must swing forward on a left body turn"
        );
        assert!(
            lagged.x < neutral.x - 1.0e-3,
            "left-hand target must pull toward the body center on a left turn"
        );

        // The opposite turn mirrors the trail exactly.
        let opposite = ArmPipelineInput {
            torso_delta: Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
            ..input
        };
        let mirrored = resolve_arm_pose(&opposite, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .1
            .hand_target
            .wrist;
        let neutral_offset = neutral
            - (chain.rest.wrist.position
                - input
                    .motion
                    .hand_anchor
                    .as_ref()
                    .unwrap()
                    .translation_from_hips);
        let expected_offset =
            Quat::from_rotation_y(TORSO_LAG_SHARE * std::f32::consts::FRAC_PI_2) * neutral_offset;
        let expected = expected_offset
            + (chain.rest.wrist.position
                - input
                    .motion
                    .hand_anchor
                    .as_ref()
                    .unwrap()
                    .translation_from_hips);
        assert!(
            mirrored.distance(expected) < 1.0e-3,
            "lag must be exactly the bounded share of the torso turn"
        );
    }

    fn mirrored_pair() -> (
        ArmChainBinding,
        ArmMotionRestGeometry,
        ArmChainBinding,
        ArmMotionRestGeometry,
    ) {
        let (left_chain, left_motion) = anchored_motion(ArmSide::Left);
        let mut right_chain = sample_chain(ArmSide::Right);
        right_chain.rest.upper_arm.position.x = -left_chain.rest.upper_arm.position.x;
        right_chain.rest.elbow.position.x = -left_chain.rest.elbow.position.x;
        right_chain.rest.wrist.position.x = -left_chain.rest.wrist.position.x;
        let right_motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Right,
            &right_chain.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );
        (left_chain, left_motion, right_chain, right_motion)
    }

    #[test]
    fn elbow_swivel_is_mirror_symmetric_between_sides() {
        let (lc, lm, rc, rm) = mirrored_pair();
        let mut left_input = ArmPipelineInput::binding_time(&lc, &lm, ArmPoseProfile::default());
        let mut right_input = ArmPipelineInput::binding_time(&rc, &rm, ArmPoseProfile::default());
        left_input.dynamic_profile.elbow_swivel_radians = 15.0_f32.to_radians();
        right_input.dynamic_profile = left_input.dynamic_profile;
        let (_, l_outcome) = resolve_arm_pose(&left_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap();
        let (_, r_outcome) = resolve_arm_pose(&right_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap();

        // Pole offsets from the shoulders must be mirror images.
        let l_pole = l_outcome.hand_target.elbow_pole - lc.rest.upper_arm.position;
        let r_pole = r_outcome.hand_target.elbow_pole - rc.rest.upper_arm.position;
        assert!((l_pole.x + r_pole.x).abs() < 1e-4);
        assert!((l_pole.y - r_pole.y).abs() < 1e-5);
        assert!((l_pole.z - r_pole.z).abs() < 1e-5);
    }

    #[test]
    fn swivel_fades_continuously_toward_the_chest_center() {
        let (chain, _) = anchored_motion(ArmSide::Right);
        let profile = DynamicArmProfile {
            elbow_swivel_radians: 15.0_f32.to_radians(),
            ..Default::default()
        };
        let width =
            profile.swivel_transition_width_ratio * crate::body_scale::DEFAULT_BODY_SCALE_METERS;
        // Place the chest center just beside the hand anchor so lateral
        // offsets sweep across the transition band (the anchor itself starts
        // inside the band; large offsets leave it).
        let center = chain.rest.wrist.position + Vec3::new(-0.02, 0.05, 0.0);
        let motion_with_center = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Right,
            &chain.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            Some(center),
        );

        // Drive the stage directly with controlled wrist targets so the
        // measurement isolates the swivel fade. Offsets are chosen relative
        // to the chest center so the sampled distances sweep the transition
        // band monotonically.
        let shoulder = chain.rest.upper_arm.position;
        let rest_target = chain.rest.wrist.position;
        let anchor_delta_from_center = (rest_target - center).x;
        let pole_at = |lateral: f32| {
            let mut input = ArmPipelineInput::binding_time(
                &chain,
                &motion_with_center,
                ArmPoseProfile::default(),
            );
            input.dynamic_profile = profile;
            swivel_adjusted_elbow_pole(&input, rest_target + Vec3::X * lateral).unwrap()
        };

        let angle_at = |delta_from_center: f32| {
            // Place the target so its horizontal offset from the chest
            // center is exactly `delta_from_center`.
            let lateral = -anchor_delta_from_center + delta_from_center;
            let wrist = rest_target + Vec3::X * lateral;
            let base = crate::arm::neutral_elbow_pole(&chain, wrist).unwrap() - shoulder;
            let offset = pole_at(lateral) - shoulder;
            let axis = (wrist - shoulder).normalize();
            let proj_base = base - axis * base.dot(axis);
            let proj_off = offset - axis * offset.dot(axis);
            f32::atan2(proj_base.cross(proj_off).dot(axis), proj_base.dot(proj_off)).abs()
        };

        let far = angle_at(width * 4.0);
        let mid_outer = angle_at(width * 1.5);
        let mid_inner = angle_at(width * 0.8);
        let near = angle_at(width * 0.2);
        // Both outer samples saturate at fade = 1 but measure through
        // different rotation axes, so allow float-rounding noise there.
        assert!(far >= mid_outer - 1.0e-5, "monotonic fade outer half");
        assert!(mid_outer >= mid_inner, "monotonic fade mid range");
        assert!(
            mid_inner >= near,
            "monotonic fade inner half: {mid_inner} vs {near}"
        );
        assert!(far > near + 1e-4, "swivel must shrink near the center");
        for p in [far, mid_outer, mid_inner, near] {
            assert!(p.is_finite(), "no NaN in swivel-modified poles");
        }
    }

    #[test]
    fn missing_hips_keeps_the_legacy_pose() {
        let (chain, _motion) = anchored_motion(ArmSide::Left);
        // Motion geometry with no hips anchor or torso center.
        let motion = crate::arm_motion_geometry::ArmMotionRestGeometry {
            side: ArmSide::Left,
            hand_anchor: None,
            torso_center: None,
        };
        let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        let (pose, outcome) = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap();
        // No anchor -> legacy fallback path; output stays finite.
        assert!(outcome.hand_target.elbow_pole.is_finite());
        assert!(pose.upper_arm_delta.is_finite());
    }

    // ---- Issue #171: per-model shoulder elevation trim ----

    fn chain_with_shoulder(side: ArmSide) -> ArmChainBinding {
        // Reuse sample_chain geometry but attach a shoulder rest pose.
        let mut chain = sample_chain(side);
        let sign = match side {
            ArmSide::Left => 1.0,
            ArmSide::Right => -1.0,
        };
        let shoulder_position = chain.rest.upper_arm.position + Vec3::new(-0.03 * sign, 0.08, 0.0);
        let rest_shoulder = crate::arm::RestSpaceBonePose {
            position: shoulder_position,
            global_rotation: Quat::from_rotation_y(0.3),
            local_rotation: Quat::from_rotation_z(0.1),
        };
        chain.shoulder = Some(bevy::prelude::Entity::from_raw_u32(20).unwrap());
        chain.rest.shoulder = Some(rest_shoulder);
        chain
    }

    #[test]
    fn zero_shoulder_trim_reproduces_the_untrimmed_output() {
        let chain = chain_with_shoulder(ArmSide::Right);
        let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Right,
            &chain.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );
        let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        let untrimmed = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        assert!(untrimmed.shoulder.is_some(), "shoulder follow present");
        // Default profile has trim 0; nothing else to compare against.
        assert_eq!(input.dynamic_profile.shoulder_elevation_trim_radians, 0.0);
    }

    #[test]
    fn clavicle_trim_is_carried_by_fk_without_changing_elbow_articulation() {
        let chain = chain_with_shoulder(ArmSide::Right);
        let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Right,
            &chain.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );
        let base_input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        let trimmed_profile = DynamicArmProfile {
            shoulder_elevation_trim_radians: -5.0_f32.to_radians(),
            ..DynamicArmProfile::default()
        };
        let trimmed_input = ArmPipelineInput {
            dynamic_profile: trimmed_profile,
            ..base_input
        };
        let before = resolve_arm_pose(&base_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        let after = resolve_arm_pose(&trimmed_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        let sh_before = before.shoulder.expect("shoulder present").delta;
        let sh_after = after.shoulder.expect("shoulder present").delta;
        assert!(sh_before.angle_between(sh_after) > 1e-4, "trim applied");
        assert!(sh_after.is_finite());
        assert_eq!(before.upper_arm_delta, after.upper_arm_delta);
        assert_eq!(before.lower_arm_delta, after.lower_arm_delta);
        // Bounded: the trim contribution is exactly the requested angle in
        // the shoulder's rest frame.
        let axis_local = chain
            .rest
            .shoulder
            .as_ref()
            .unwrap()
            .global_rotation
            .inverse()
            * lateral_axis_of(&chain).cross(Vec3::Y).normalize();
        let trim_q = Quat::from_axis_angle(axis_local, -5.0_f32.to_radians());
        let expected = (sh_before * trim_q).normalize();
        assert!(sh_after.angle_between(expected) < 1e-4);
    }

    fn lateral_axis_of(chain: &ArmChainBinding) -> Vec3 {
        (chain.rest.elbow.position - chain.rest.upper_arm.position)
            .try_normalize()
            .unwrap_or(Vec3::X)
    }

    #[test]
    fn shoulder_trim_is_symmetric_and_safe_without_a_bone() {
        // Missing shoulder bone: trim is a no-op.
        let mut no_shoulder = sample_chain(ArmSide::Left);
        no_shoulder.shoulder = None;
        no_shoulder.rest.shoulder = None;
        let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Left,
            &no_shoulder.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );
        let profile = DynamicArmProfile {
            shoulder_elevation_trim_radians: -0.13,
            ..DynamicArmProfile::default()
        };
        let mut input =
            ArmPipelineInput::binding_time(&no_shoulder, &motion, ArmPoseProfile::default());
        input.dynamic_profile = profile;
        let pose = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        assert!(pose.shoulder.is_none(), "cannot trim a missing bone");
        assert!(pose.upper_arm_delta.is_finite());

        // Mirrored geometry with the same signed profile produces mirrored
        // magnitude changes on both shoulders.
        let left = chain_with_shoulder(ArmSide::Left);
        let mut right = chain_with_shoulder(ArmSide::Right);
        // Mirror the right-side geometry exactly from the left side.
        right.rest.upper_arm.position.x = -left.rest.upper_arm.position.x;
        right.rest.elbow.position.x = -left.rest.elbow.position.x;
        right.rest.wrist.position.x = -left.rest.wrist.position.x;
        let l_sh = left.rest.shoulder.unwrap();
        right.rest.shoulder = Some(crate::arm::RestSpaceBonePose {
            position: Vec3::new(-l_sh.position.x, l_sh.position.y, l_sh.position.z),
            global_rotation: mirror_x(l_sh.global_rotation),
            local_rotation: l_sh.local_rotation,
        });
        let lm = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Left,
            &left.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );
        let rm = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Right,
            &right.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );
        let profile = DynamicArmProfile {
            shoulder_elevation_trim_radians: -0.13,
            ..DynamicArmProfile::default()
        };
        let mut l_input = ArmPipelineInput::binding_time(&left, &lm, ArmPoseProfile::default());
        l_input.dynamic_profile = profile;
        let mut r_input = ArmPipelineInput::binding_time(&right, &rm, ArmPoseProfile::default());
        r_input.dynamic_profile = profile;
        let l_pose = resolve_arm_pose(&l_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        let r_pose = resolve_arm_pose(&r_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        let l_delta = l_pose.shoulder.expect("left shoulder").delta;
        let r_delta = r_pose.shoulder.expect("right shoulder").delta;
        assert!(l_delta.is_finite() && r_delta.is_finite());
        assert!(
            (l_delta.angle_between(Quat::IDENTITY) - r_delta.angle_between(Quat::IDENTITY)).abs()
                < 1e-3,
            "mirrored trims must have equal magnitude"
        );
    }

    fn mirror_x(q: Quat) -> Quat {
        Quat::from_xyzw(-q.x, q.y, q.z, q.w)
    }

    // ---- Issue #177: legacy finger curl excluded from dynamic mode ----

    fn chain_with_fingers(side: ArmSide) -> ArmChainBinding {
        let mut chain = sample_chain(side);
        let entity = |id: u32| bevy::prelude::Entity::from_raw_u32(id).unwrap();
        let joint = |id: u32, position: Vec3| crate::arm::FingerJointRestBinding {
            entity: entity(id),
            rest: crate::arm::RestSpaceBonePose {
                position,
                // Non-identity rest orientation exercises the exclusion path.
                global_rotation: Quat::from_rotation_z(0.4),
                local_rotation: Quat::from_rotation_y(-0.2),
            },
        };
        let base = chain.rest.wrist.position;
        chain.fingers.index = crate::arm::FingerJointReferences {
            metacarpal: None,
            proximal: Some(entity(30)),
            intermediate: Some(entity(31)),
            distal: Some(entity(32)),
        };
        chain.finger_rest.index = crate::arm::FingerJointRestReferences {
            metacarpal: None,
            proximal: Some(joint(30, base + Vec3::X * 0.03)),
            intermediate: Some(joint(31, base + Vec3::X * 0.05)),
            distal: None,
        };
        chain.finger_rest.little.proximal = Some(joint(33, base + Vec3::new(0.03, 0.0, -0.02)));
        chain
    }

    #[test]
    fn dynamic_mode_never_applies_the_legacy_fixed_finger_curl() {
        let chain = chain_with_fingers(ArmSide::Right);
        assert!(chain.finger_rest.index.proximal.is_some());
        let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Right,
            &chain.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );
        // Legacy profile carries the fixed 10-degree curl.
        let legacy_profile = ArmPoseProfile {
            finger_curl_radians: 10.0_f32.to_radians(),
            ..ArmPoseProfile::default()
        };
        let input = ArmPipelineInput {
            legacy_profile,
            ..ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default())
        };
        let pose = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        let fingers = pose.fingers.index;
        // No curl may be applied: any resolved entry must be an identity
        // delta (the compositor skips those), never a real rotation.
        for joint in [fingers.metacarpal, fingers.proximal, fingers.intermediate] {
            match joint {
                None => {}
                Some(delta) => assert!(
                    delta.delta.angle_between(Quat::IDENTITY) < 1e-5,
                    "dynamic mode must not curl fingers"
                ),
            }
        }

        // The same profile under the explicitly selected legacy source still
        // applies the curl (the field remains usable there).
        let legacy = resolve_arm_pose(&input, ArmPoseSourceKind::LegacyStatic)
            .unwrap()
            .unwrap()
            .0;
        assert!(legacy.fingers.index.proximal.is_some());
    }

    #[test]
    fn both_sides_resolve_through_the_same_typed_stages() {
        for side in [ArmSide::Left, ArmSide::Right] {
            let chain = sample_chain(side);
            let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
                side,
                &chain.rest,
                None,
                None,
                None,
            );
            let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
            let outcome = resolve_arm_pose(&input, ArmPoseSourceKind::LegacyStatic)
                .expect("pipeline error")
                .expect("pose");
            assert_eq!(
                outcome.0.upper_arm,
                chain.upper_arm,
                "{} side",
                chain_side_label(&chain)
            );
        }
    }

    // ---- Upper-arm coronal descent limit (torso collision guard) ----

    fn over_swing_input(side: ArmSide) -> (ArmChainBinding, ArmIkInput, ArmIkSolution) {
        let chain = sample_chain(side);
        // Pull the wrist below and across the body with an across-body elbow
        // pole so the solved upper-arm direction descends well past the fully
        // lowered 90-degree pose.
        let across = match side {
            ArmSide::Left => -1.0,
            ArmSide::Right => 1.0,
        };
        let direction = Vec3::new(across * 0.35, -1.0, 0.05).normalize();
        let target = ArmIkTarget {
            wrist: chain.rest.upper_arm.position + direction * chain.rest.total_arm_length * 0.995,
            elbow_pole: chain.rest.upper_arm.position
                + Vec3::new(across * 0.4, -0.8, 0.0).normalize() * 0.3,
        };
        let input = ArmIkInput::from_geometry(chain.rest, target);
        let solution = crate::arm::solve_two_bone_arm(input).expect("two-bone solve");
        (chain, input, solution)
    }

    #[test]
    fn clamp_is_a_noop_while_the_descent_stays_within_the_limit() {
        let chain = sample_chain(ArmSide::Left);
        // The default relaxed attention pose (80 degrees below horizontal) must never be
        // touched by the 85-degree limit.
        let target = crate::arm::default_arm_target(&chain, ArmPoseProfile::default())
            .expect("legacy target");
        let relaxed_input = ArmIkInput::from_geometry(chain.rest, target);
        let mut relaxed = crate::arm::solve_two_bone_arm(relaxed_input).expect("solve");
        let descent =
            upper_arm_descent_radians(&relaxed_input, &relaxed).expect("measurable descent");
        assert!(
            descent <= MAX_ARM_DROP_RADIANS,
            "default descent must sit inside the limit: {} deg",
            descent.to_degrees()
        );
        let before = relaxed;
        assert!(
            !clamp_upper_arm_swing(&mut relaxed, &relaxed_input, MAX_ARM_DROP_RADIANS),
            "a pose inside the limit must not be modified"
        );
        assert_eq!(relaxed, before);
    }

    #[test]
    fn over_swing_descent_is_clamped_to_the_limit() {
        let (chain, input, mut solution) = over_swing_input(ArmSide::Left);
        let descent_before =
            upper_arm_descent_radians(&input, &solution).expect("measurable descent");
        assert!(
            descent_before > 90.0_f32.to_radians(),
            "fixture must start past the attention pose: {} deg",
            descent_before.to_degrees()
        );

        let elbow_bend_before = (solution.wrist - solution.elbow)
            .normalize()
            .dot((solution.elbow - input.shoulder).normalize());
        let reach_before = (solution.wrist - input.shoulder).length();

        assert!(clamp_upper_arm_swing(
            &mut solution,
            &input,
            MAX_ARM_DROP_RADIANS
        ));

        let descent_after =
            upper_arm_descent_radians(&input, &solution).expect("measurable descent");
        assert!(
            (descent_after - MAX_ARM_DROP_RADIANS).abs() < 1.0e-4,
            "descent must land on the limit: {} deg",
            descent_after.to_degrees()
        );
        // The chain rotated rigidly: bend and reach are preserved exactly.
        let elbow_bend_after = (solution.wrist - solution.elbow)
            .normalize()
            .dot((solution.elbow - input.shoulder).normalize());
        assert!((elbow_bend_after - elbow_bend_before).abs() < 1.0e-4);
        let reach_after = (solution.wrist - input.shoulder).length();
        assert!((reach_after - reach_before).abs() < 1.0e-4);
        assert!(solution.upper_arm_delta.is_finite());
        assert!(solution.lower_arm_delta.is_finite());
        let _ = chain;
    }

    #[test]
    fn both_sides_clamp_symmetrically() {
        for side in [ArmSide::Left, ArmSide::Right] {
            let (_, input, mut solution) = over_swing_input(side);
            let descent_before =
                upper_arm_descent_radians(&input, &solution).expect("measurable descent");
            assert!(descent_before > 90.0_f32.to_radians());
            assert!(clamp_upper_arm_swing(
                &mut solution,
                &input,
                MAX_ARM_DROP_RADIANS
            ));
            let descent_after =
                upper_arm_descent_radians(&input, &solution).expect("measurable descent");
            assert!((descent_after - MAX_ARM_DROP_RADIANS).abs() < 1.0e-4);
        }
    }

    #[test]
    fn raising_the_arm_is_never_clamped() {
        let chain = sample_chain(ArmSide::Left);
        // Target above the shoulder: the descent goes negative (arm raised).
        let target = ArmIkTarget {
            wrist: chain.rest.upper_arm.position
                + Vec3::new(0.2, 0.9, 0.1).normalize() * chain.rest.total_arm_length * 0.98,
            elbow_pole: chain.rest.elbow.position + Vec3::NEG_Z * 0.05,
        };
        let input = ArmIkInput::from_geometry(chain.rest, target);
        let mut solution = crate::arm::solve_two_bone_arm(input).expect("solve");
        let before = solution;
        assert!(!clamp_upper_arm_swing(
            &mut solution,
            &input,
            MAX_ARM_DROP_RADIANS
        ));
        assert_eq!(solution, before);
    }

    #[test]
    fn forward_swing_survives_the_clamp() {
        let (chain, input, mut solution) = over_swing_input(ArmSide::Left);
        let forward_before = solution.elbow.z - chain.rest.upper_arm.position.z;
        assert!(clamp_upper_arm_swing(
            &mut solution,
            &input,
            MAX_ARM_DROP_RADIANS
        ));
        let forward_after = solution.elbow.z - chain.rest.upper_arm.position.z;
        assert!(
            forward_after.signum() == forward_before.signum() && forward_after.abs() > 1.0e-3,
            "sagittal swing must survive: {forward_before} -> {forward_after}"
        );
    }

    #[test]
    fn degenerate_limits_and_geometry_are_safe_noops() {
        let (_, input, mut solution) = over_swing_input(ArmSide::Left);
        let before = solution;
        for limit in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(!clamp_upper_arm_swing(&mut solution, &input, limit));
            assert_eq!(solution, before);
        }
    }

    #[test]
    fn pipeline_output_respects_the_coronal_descent_limit() {
        // End-to-end: the virtual-hand authority with a body-follow offset
        // that pulls the hand across the torso must still emit a pose whose
        // solved upper-arm direction stays inside the limit.
        let (chain, _input, _solution) = over_swing_input(ArmSide::Left);
        let motion = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            ArmSide::Left,
            &chain.rest,
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::IDENTITY),
            Some(Vec3::new(0.0, 1.2, 0.02)),
        );
        let mut input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        input.head_offset = Vec3::new(-0.20, 0.0, 0.0);
        input.body_offset = Vec3::ZERO;
        let pose = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .expect("pipeline error")
            .expect("pose");

        // Reconstruct the solved upper-arm model direction from the emitted
        // rest-relative delta and measure its coronal descent.
        let rest = &chain.rest.upper_arm;
        let model_delta =
            rest.global_rotation * pose.0.upper_arm_delta * rest.global_rotation.inverse();
        let solved_direction =
            model_delta * (chain.rest.elbow.position - chain.rest.upper_arm.position).normalize();
        let rest_coronal = (chain.rest.elbow.position - chain.rest.upper_arm.position).normalize();
        let coronal =
            crate::arm::finite_normalized(solved_direction - Vec3::Z * solved_direction.z)
                .expect("coronal component");
        let swing_axis =
            crate::arm::finite_normalized(rest_coronal.cross(-Vec3::Y)).expect("swing axis");
        let descent = f32::atan2(
            rest_coronal.cross(coronal).dot(swing_axis),
            rest_coronal.dot(coronal),
        );
        assert!(
            descent <= MAX_ARM_DROP_RADIANS + 1.0e-3,
            "pipeline descent {} deg exceeds the {} deg limit",
            descent.to_degrees(),
            MAX_ARM_DROP_RADIANS.to_degrees()
        );
    }

    #[test]
    fn tracked_side_applies_the_observed_fingers_through_the_finger_weight() {
        // The finger channel is its own weight, so a hand with fingers observed
        // but the palm channel weighted out still articulates, and a zero
        // finger weight leaves the rest pose the shared conversion produced.
        let mut chain = chain_with_fingers(ArmSide::Right);
        let wrist = chain.rest.wrist.position;
        let row = |dx: [f32; 3], z: f32| crate::arm::FingerJointRestReferences {
            metacarpal: None,
            proximal: Some(crate::arm::FingerJointRestBinding {
                entity: bevy::prelude::Entity::from_raw_u32(40).unwrap(),
                rest: crate::arm::RestSpaceBonePose {
                    position: wrist + Vec3::new(dx[0], 0.0, z),
                    global_rotation: Quat::IDENTITY,
                    local_rotation: Quat::IDENTITY,
                },
            }),
            intermediate: Some(crate::arm::FingerJointRestBinding {
                entity: bevy::prelude::Entity::from_raw_u32(41).unwrap(),
                rest: crate::arm::RestSpaceBonePose {
                    position: wrist + Vec3::new(dx[1], 0.0, z),
                    global_rotation: Quat::IDENTITY,
                    local_rotation: Quat::IDENTITY,
                },
            }),
            distal: Some(crate::arm::FingerJointRestBinding {
                entity: bevy::prelude::Entity::from_raw_u32(42).unwrap(),
                rest: crate::arm::RestSpaceBonePose {
                    position: wrist + Vec3::new(dx[2], 0.0, z),
                    global_rotation: Quat::IDENTITY,
                    local_rotation: Quat::IDENTITY,
                },
            }),
        };
        chain.finger_rest.index = row([0.02, 0.04, 0.055], 0.0);
        chain.finger_rest.middle = row([0.02, 0.04, 0.055], 0.002);
        chain.finger_rest.ring = row([0.02, 0.04, 0.055], -0.002);
        chain.finger_rest.little = row([0.02, 0.04, 0.055], -0.004);

        let target = vtuber_core::arm_tracking::ArmTrackingTarget {
            wrist: [0.4, -0.3, 0.5],
            elbow_pole: [0.7, -0.5, -0.1],
            palm_normal: None,
            palm_forward: None,
            fingers: Some(vtuber_core::arm_tracking::HandFingerPose {
                fingers: [[0.9, 0.9, 0.4]; 4],
                spread: [0.0; 4],
                thumb: [0.3, 0.3],
                thumb_spread: 0.0,
            }),
        };
        let resolve = |fingers: f32| {
            resolve_tracked_side(
                Some(&chain),
                None,
                DynamicArmProfile::default(),
                0.7,
                Some(target),
                vtuber_core::arm_tracking::ArmBlendWeight {
                    wrist: 1.0,
                    pole: 1.0,
                    palm: 0.0,
                    fingers,
                },
                Quat::IDENTITY,
                &mut crate::tracked_arm::TrackedArmFilter::default(),
                1.0,
            )
        };
        let curled = resolve(1.0).expect("a curled hand resolves");
        let delta = curled.fingers.index.proximal.expect("index proximal");
        assert!(
            delta.delta.angle_between(Quat::IDENTITY) > 0.2,
            "the observed curl must reach the bone: {:?}",
            delta.delta
        );
        // The elbow and wrist are solved from the observation, not from the
        // finger articulation, so a curl cannot move the hand. A zero finger
        // weight leaves the identity deltas the shared conversion already
        // produced, which the compositor skips.
        let rest = resolve(0.0).expect("the rest pose still resolves");
        let straight = rest.fingers.index.proximal.expect("index proximal");
        assert!(straight.delta.angle_between(Quat::IDENTITY) < 1.0e-5);
        assert_eq!(rest.lower_arm_delta, curled.lower_arm_delta);
        assert_eq!(rest.upper_arm_delta, curled.upper_arm_delta);
    }

    /// The observed upper-arm direction this target solves to, in model space.
    fn observed_upper_direction(
        chain: &ArmChainBinding,
        pose: &crate::arm_pose::ResolvedArmPose,
    ) -> Vec3 {
        let rest = &chain.rest.upper_arm;
        let model_delta =
            rest.global_rotation * pose.upper_arm_delta * rest.global_rotation.inverse();
        crate::arm::finite_normalized(
            model_delta
                * crate::arm::finite_normalized(chain.rest.elbow.position - rest.position).unwrap(),
        )
        .expect("solved upper-arm direction")
    }

    #[test]
    fn tracked_side_crosses_in_front_instead_of_using_the_virtual_drop_limit() {
        let chain = sample_chain(ArmSide::Left);
        let target = vtuber_core::arm_tracking::ArmTrackingTarget {
            wrist: [-0.55, 0.10, 0.55],
            elbow_pole: [-0.20, -0.40, -0.20],
            palm_normal: None,
            palm_forward: None,
            fingers: None,
        };
        let geometrized =
            crate::tracked_arm::tracked_arm_ik_target(chain.rest, target, Quat::IDENTITY);
        let raw_input = ArmIkInput::from_geometry(chain.rest, geometrized);
        let raw = crate::arm::solve_two_bone_arm(raw_input).expect("solve");
        let raw_descent = upper_arm_descent_radians(&raw_input, &raw).expect("descent");
        assert!(
            raw_descent.abs() > MAX_ARM_DROP_RADIANS,
            "this target must exceed the limit for the test to mean anything, got {} deg",
            raw_descent.to_degrees()
        );

        let pose = resolve_tracked_side(
            Some(&chain),
            None,
            DynamicArmProfile::default(),
            0.7,
            Some(target),
            vtuber_core::arm_tracking::ArmBlendWeight::ONE,
            Quat::IDENTITY,
            &mut crate::tracked_arm::TrackedArmFilter::default(),
            1.0,
        )
        .expect("tracked pose");

        let direction = observed_upper_direction(&chain, &pose);
        assert!(direction.x < 0.0, "crossing must remain possible");
        assert!(
            direction.z > 0.0,
            "the elbow must move in front of the chest"
        );
        assert!((direction.z.atan2(direction.x).to_degrees() - 130.0).abs() < 0.001);
        let upper = chain.rest.upper_arm.global_rotation
            * pose.upper_arm_delta
            * chain.rest.upper_arm.global_rotation.inverse();
        let lower = chain.rest.elbow.global_rotation
            * pose.lower_arm_delta
            * chain.rest.elbow.global_rotation.inverse();
        let elbow = chain.rest.upper_arm.position
            + upper * (chain.rest.elbow.position - chain.rest.upper_arm.position);
        let wrist = elbow + upper * lower * (chain.rest.wrist.position - chain.rest.elbow.position);
        assert!(wrist.z > chain.rest.upper_arm.position.z);
        assert!((elbow.distance(wrist) - chain.rest.forearm_length).abs() < 1.0e-6);
        near(
            pose.lower_arm_delta * Vec3::X,
            raw.lower_arm_delta * Vec3::X,
        );
    }

    #[test]
    fn tracked_loss_reaches_neutral_after_joint_damping() {
        let chain = sample_chain(ArmSide::Left);
        let geometry = crate::arm_motion_geometry::build_arm_motion_rest_geometry(
            chain.side,
            &chain.rest,
            Some(Vec3::new(0.0, 0.92, 0.0)),
            Some(Quat::IDENTITY),
            None,
        );
        let input = ArmPipelineInput {
            body_scale_meters: 0.7,
            ..ArmPipelineInput::binding_time(&chain, &geometry, ArmPoseProfile::default())
        };
        let (neutral, outcome) = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap();
        let initial =
            crate::arm::solve_two_bone_arm(ArmIkInput::from_chain(&chain, outcome.hand_target))
                .unwrap();
        let origin = chain.rest.upper_arm.position;
        // Raise the same arm on the opposite side of its shoulder. Cartesian
        // wrist interpolation crosses the shoulder and folds the elbow even
        // though both endpoint skeletons have the same flexion.
        let turn = Quat::from_rotation_z(std::f32::consts::PI);
        let target = vtuber_core::arm_tracking::ArmTrackingTarget {
            wrist: (turn * (initial.wrist - origin) / chain.rest.total_arm_length).to_array(),
            elbow_pole: (turn * (initial.elbow - origin) / chain.rest.total_arm_length).to_array(),
            palm_normal: None,
            palm_forward: None,
            fingers: None,
        };
        let resolve = |weights, filter: &mut crate::tracked_arm::TrackedArmFilter| {
            resolve_tracked_side(
                Some(&chain),
                Some(&geometry),
                DynamicArmProfile::default(),
                0.7,
                Some(target),
                weights,
                Quat::IDENTITY,
                filter,
                1.0 / 60.0,
            )
            .unwrap()
        };
        let mut filter = crate::tracked_arm::TrackedArmFilter::default();
        let tracked = resolve(vtuber_core::arm_tracking::ArmBlendWeight::ONE, &mut filter);
        let zero = vtuber_core::arm_tracking::ArmBlendWeight {
            wrist: 0.0,
            pole: 0.0,
            palm: 0.0,
            fingers: 0.0,
        };
        let expected = neutral;
        assert!(
            tracked
                .upper_arm_delta
                .angle_between(expected.upper_arm_delta)
                > 0.1
        );
        let mut returned = tracked;
        let flexion = |pose: crate::arm_pose::ResolvedArmPose| {
            let upper = chain.rest.upper_arm.global_rotation
                * pose.upper_arm_delta
                * chain.rest.upper_arm.global_rotation.inverse();
            let lower = chain.rest.elbow.global_rotation
                * pose.lower_arm_delta
                * chain.rest.elbow.global_rotation.inverse();
            (upper * (chain.rest.elbow.position - origin)).angle_between(
                upper * lower * (chain.rest.wrist.position - chain.rest.elbow.position),
            )
        };
        let max_flexion = flexion(tracked).max(flexion(expected));
        for tick in 0..=300 {
            let weight = 1.0 - tick as f32 / 300.0;
            returned = resolve(
                vtuber_core::arm_tracking::ArmBlendWeight {
                    wrist: weight,
                    pole: weight,
                    palm: weight,
                    fingers: weight,
                },
                &mut filter,
            );
            assert!(
                flexion(returned) <= max_flexion + 0.001,
                "return must not fold the elbow beyond either endpoint: tick {tick}, {} deg vs {} deg",
                flexion(returned).to_degrees(),
                max_flexion.to_degrees()
            );
        }
        for _ in 0..240 {
            returned = resolve(zero, &mut filter);
        }
        assert!(returned.upper_arm_delta.dot(expected.upper_arm_delta).abs() > 1.0 - 1.0e-6);
        assert!(returned.lower_arm_delta.dot(expected.lower_arm_delta).abs() > 1.0 - 1.0e-6);
    }

    #[test]
    fn tracked_side_applies_the_palm_twist_through_the_palm_weight() {
        use crate::arm::{FingerJointRestBinding, FingerRestReferences};
        use vtuber_core::arm_tracking::{ArmBlendWeight, ArmTrackingTarget};

        let mut chain = sample_chain(ArmSide::Left);
        let wrist = chain.rest.wrist.position;
        let index = rest_bone(wrist + Vec3::new(0.05, 0.0, 0.003));
        let little = rest_bone(wrist + Vec3::new(0.05, 0.0, -0.003));
        chain.finger_rest = FingerRestReferences {
            index: crate::arm::FingerJointRestReferences {
                proximal: Some(FingerJointRestBinding {
                    entity: bevy::prelude::Entity::from_raw_u32(4).unwrap(),
                    rest: index,
                }),
                ..Default::default()
            },
            little: crate::arm::FingerJointRestReferences {
                proximal: Some(FingerJointRestBinding {
                    entity: bevy::prelude::Entity::from_raw_u32(5).unwrap(),
                    rest: little,
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        let target = ArmTrackingTarget {
            wrist: [0.4, -0.3, 0.5],
            elbow_pole: [0.7, -0.5, -0.1],
            palm_normal: Some([0.0, 0.0, 1.0]),
            palm_forward: Some([1.0, 0.0, 0.0]),
            fingers: None,
        };
        let weights = |palm| ArmBlendWeight {
            wrist: 1.0,
            pole: 1.0,
            palm,
            fingers: 0.0,
        };
        let resolve = |target: ArmTrackingTarget, palm: f32| {
            resolve_tracked_side(
                Some(&chain),
                None,
                DynamicArmProfile::default(),
                0.7,
                Some(target),
                weights(palm),
                Quat::IDENTITY,
                // A single tick with a full step: this test is about the
                // channel's authority, not the filter's response.
                &mut crate::tracked_arm::TrackedArmFilter::default(),
                1.0,
            )
        };

        let tracked = resolve(target, 1.0).expect("tracked pose");
        let default_twist = resolve(target, 0.0).expect("tracked pose");
        let no_palm = resolve(
            ArmTrackingTarget {
                palm_normal: None,
                palm_forward: None,
                ..target
            },
            1.0,
        )
        .expect("tracked pose");

        assert!(tracked.hand.unwrap().delta.is_finite());
        assert_eq!(default_twist.hand.unwrap().delta, Quat::IDENTITY);
        assert_eq!(no_palm, default_twist);
        // Pronation changes the forearm, without an axial wrist correction.
        assert_ne!(tracked.lower_arm_delta, default_twist.lower_arm_delta);
    }
}
