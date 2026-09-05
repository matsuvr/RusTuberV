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
/// of stopping at the shoulder.
pub const TORSO_LAG_SHARE: f32 = 0.6;

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

/// Scale-aware virtual hand anchor and arm modifier parameters.
///
/// Values are semantic ratios of body scale meters rather than absolute
/// model-specific lengths. Per-model tuning aggregates into this one typed
/// profile instead of scattering constants through systems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicArmProfile {
    /// Hips-relative hand anchor offset as fractions of body scale.
    ///
    /// The lateral component is mirrored per side from the magnitude.
    pub hand_anchor_ratio: Vec3,
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
    /// Fraction of the solved forearm twist removed by the relaxer.
    pub twist_relax_weight: f32,
    /// Share of the removed twist kept off the parent bone.
    pub twist_parent_child_crossfade: f32,
    /// Optional per-model shoulder elevation trim (negative lowers the
    /// shoulder). Neutral default 0; bounded to +/- 15 degrees.
    pub shoulder_elevation_trim_radians: f32,
}

impl Default for DynamicArmProfile {
    fn default() -> Self {
        Self {
            hand_anchor_ratio: Vec3::new(
                0.215 / crate::body_scale::DEFAULT_BODY_SCALE_METERS,
                -0.150 / crate::body_scale::DEFAULT_BODY_SCALE_METERS,
                0.0,
            ),
            compensation_gains: Vec3::new(0.25, 0.0, 1.0),
            elbow_swivel_radians: 15.0_f32.to_radians(),
            swivel_transition_width_ratio: 0.15,
            pole_influence: 0.2,
            twist_relax_weight: 0.7,
            twist_parent_child_crossfade: 0.9,
            shoulder_elevation_trim_radians: 0.0,
        }
    }
}

impl DynamicArmProfile {
    /// Validates the bounded profile before any solve uses it.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.hand_anchor_ratio.is_finite()
            && self.hand_anchor_ratio.x.abs() <= 2.0
            && self.hand_anchor_ratio.y.abs() <= 2.0
            && self.hand_anchor_ratio.z.abs() <= 2.0
            && self.compensation_gains.is_finite()
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
            && self.twist_relax_weight.is_finite()
            && self.twist_relax_weight >= 0.0
            && self.twist_relax_weight <= 1.0
            && self.twist_parent_child_crossfade.is_finite()
            && self.twist_parent_child_crossfade >= 0.0
            && self.twist_parent_child_crossfade <= 1.0
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
    /// Semantic head translation offset (meters), used by dynamic sources.
    pub head_offset: Vec3,
    /// Semantic root/body compensation offset (meters), dynamic sources.
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
    let relax_params = if dynamic_authority {
        Some(TwistRelaxParams {
            chain: input.chain,
            motion: input.motion,
            weight: input.dynamic_profile.twist_relax_weight,
            crossfade: input.dynamic_profile.twist_parent_child_crossfade,
        })
    } else {
        None
    };
    let pose = super::arm_pose::solve_stage(
        input.chain,
        effective_legacy_profile,
        &target,
        relax_params.as_ref(),
    )?;
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
    // Downstream propagation: the trim's model-space rotation also reaches
    // the elbow and wrist in small decaying shares so the arm bends with the
    // shoulder instead of rotating as a rigid stick.
    (pose.upper_arm_delta, pose.lower_arm_delta) =
        crate::arm_pose::propagate_shoulder_downstream(
            model_delta,
            input.chain.rest.upper_arm.global_rotation,
            input.chain.rest.elbow.global_rotation,
            pose.upper_arm_delta,
            pose.lower_arm_delta,
        );
}

/// Parameters for the post-solve forearm twist relaxer (Issue #170).
#[derive(Debug, Clone, Copy)]
pub struct TwistRelaxParams<'a> {
    /// Bound arm chain providing immutable rest orientations.
    pub chain: &'a ArmChainBinding,
    /// Binding-time motion geometry carrying the forearm twist axis.
    pub motion: &'a ArmMotionRestGeometry,
    /// Fraction of the solved relative twist removed from the forearm.
    pub weight: f32,
    /// Share of the removed twist kept off the parent bone.
    pub crossfade: f32,
}

/// Rest-relative swing/twist decomposition of a quaternion around an axis.
///
/// Returns `(swing, twist)` with `q ~= swing * twist`, both finite normalized
/// quaternions and the twist angle taken on the shortest arc. Returns `None`
/// for degenerate input instead of producing NaNs.
#[must_use]
pub fn decompose_swing_twist(q: Quat, axis: Vec3) -> Option<(Quat, Quat)> {
    let q = if q.w < 0.0 { -q } else { q };
    let axis = axis.try_normalize()?;
    if !q.is_finite() || axis.length_squared() < 0.5 {
        return None;
    }
    let d = q.xyz().dot(axis);
    let angle = 2.0 * f32::atan2(d, q.w);
    let twist = Quat::from_axis_angle(axis, angle);
    let swing = q * twist.inverse();
    if swing.is_finite() && twist.is_finite() && swing.length_squared() > f32::EPSILON {
        Some((swing.normalize(), twist.normalize()))
    } else {
        None
    }
}

/// Stage 3 (Issue #170): forearm swing-twist relaxer.
///
/// Decomposes the solved lower-arm rotation relative to its parent into
/// swing and twist around the rest-space forearm axis, removes
/// `weight * crossfade` of the relative twist from the forearm, and
/// redistributes `weight * (1 - crossfade)` onto the upper arm as bounded
/// compensation. Weight 0 leaves the solution untouched; missing or
/// degenerate twist geometry is a safe no-op.
pub fn relax_forearm_twist(
    solution: &mut crate::arm::ArmIkSolution,
    params: &TwistRelaxParams<'_>,
) -> Result<(), ArmPipelineError> {
    let profile_valid = params.weight.is_finite()
        && (0.0..=1.0).contains(&params.weight)
        && params.crossfade.is_finite()
        && (0.0..=1.0).contains(&params.crossfade);
    let Some(twist_info) = params.motion.forearm_twist.as_ref().filter(|t| t.usable()) else {
        return Ok(());
    };
    if !profile_valid || params.weight <= f32::EPSILON {
        return Ok(());
    }
    let axis_model = twist_info.direction;

    let rest = &params.chain.rest;
    // Model-space deltas of upper arm and of the lower arm relative to it.
    let upper_model_delta =
        solution.upper_arm_global_rotation * rest.upper_arm.global_rotation.inverse();
    let lower_model_delta =
        solution.lower_arm_global_rotation * rest.elbow.global_rotation.inverse();
    let relative = upper_model_delta.inverse() * lower_model_delta;

    let Some((_, _twist)) = decompose_swing_twist(relative, axis_model) else {
        return Ok(());
    };
    // Signed relative twist angle around the axis (shortest arc).
    let r = if relative.w < 0.0 {
        -relative
    } else {
        relative
    };
    let signed_angle = 2.0 * f32::atan2(r.xyz().dot(axis_model), r.w);

    let reduce = signed_angle * params.weight * params.crossfade;
    let compensate = signed_angle * params.weight * (1.0 - params.crossfade);
    if !reduce.is_finite() || !compensate.is_finite() {
        return Err(ArmPipelineError::DegenerateSolvedPose);
    }

    let reduce_q = Quat::from_axis_angle(axis_model, -reduce);
    let compensate_q = Quat::from_axis_angle(axis_model, compensate);

    let upper_corrected = compensate_q * upper_model_delta;
    let lower_relative_corrected = relative * reduce_q;

    let upper_rest = rest.upper_arm.global_rotation;
    let lower_rest = rest.elbow.global_rotation;
    let normalize = |q: Quat| {
        if q.is_finite() && q.length_squared() > f32::EPSILON {
            Ok(q.normalize())
        } else {
            Err(ArmPipelineError::DegenerateSolvedPose)
        }
    };
    solution.upper_arm_global_rotation = normalize(upper_corrected * upper_rest)?;
    solution.lower_arm_global_rotation = normalize(lower_relative_corrected * lower_rest)?;
    solution.upper_arm_delta = crate::arm::conjugated_rest_delta(upper_corrected, upper_rest)
        .map_err(ArmPipelineError::Solve)?;
    solution.lower_arm_delta =
        crate::arm::conjugated_rest_delta(lower_relative_corrected, lower_rest)
            .map_err(ArmPipelineError::Solve)?;
    solution.upper_arm_local_rotation = rest.upper_arm.local_rotation * solution.upper_arm_delta;
    solution.lower_arm_local_rotation = rest.elbow.local_rotation * solution.lower_arm_delta;
    Ok(())
}

/// Coronal descent limit for the upper arm, measured from the authored
/// T-pose direction.
///
/// 0 degrees is the T-pose and 90 degrees is the fully lowered "attention"
/// pose. The bound is 85 degrees instead of 90 so clothing thickness cannot
/// push the arm into the torso mesh when body-follow translation or hand
/// target compensation pulls the arm across the body.
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

    // Carry the solved chain rigidly: compose the swing onto the existing
    // model-space deltas so the elbow bend and twist allocation survive
    // exactly, then rebuild the rest-relative local deltas.
    let upper_rest_global = input.upper_arm_rest_global_rotation;
    let lower_rest_global = input.lower_arm_rest_global_rotation;
    let upper_model = solution.upper_arm_global_rotation * upper_rest_global.inverse();
    let lower_model = solution.lower_arm_global_rotation * lower_rest_global.inverse();
    let upper_model_new = (swing * upper_model).normalize();
    let lower_model_new = (swing * lower_model).normalize();
    let Ok(upper_delta) = crate::arm::conjugated_rest_delta(upper_model_new, upper_rest_global)
    else {
        return false;
    };
    let lower_local_model = upper_model_new.inverse() * lower_model_new;
    let Ok(lower_delta) = crate::arm::conjugated_rest_delta(lower_local_model, lower_rest_global)
    else {
        return false;
    };

    solution.elbow = input.shoulder + swing * (solution.elbow - input.shoulder);
    solution.wrist = input.shoulder + swing * (solution.wrist - input.shoulder);
    solution.upper_arm_global_rotation = upper_model_new * upper_rest_global;
    solution.lower_arm_global_rotation = lower_model_new * lower_rest_global;
    solution.upper_arm_delta = upper_delta;
    solution.lower_arm_delta = lower_delta;
    solution.upper_arm_local_rotation = input.upper_arm_rest_rotation * upper_delta;
    solution.lower_arm_local_rotation = input.lower_arm_rest_rotation * lower_delta;
    true
}

/// Signed coronal descent of a solved upper-arm direction from the authored
/// T-pose, in radians. Positive values descend toward the body side; 90
/// degrees is the fully lowered arm and larger values cross under the torso.
#[must_use]
pub fn upper_arm_descent_radians(input: &ArmIkInput, solution: &ArmIkSolution) -> Option<f32> {
    let forward = Vec3::Z;
    let rest_direction = crate::arm::finite_normalized(input.rest_elbow - input.shoulder)?;
    let upper_direction = crate::arm::finite_normalized(solution.elbow - input.shoulder)?;
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

/// Stage 1 (dynamic): hips-relative virtual hand target (Issue #168).
///
/// The anchor base is a scale-aware ratio of body scale mirrored per side;
/// the combined head/body translation follows each axis by the typed
/// compensation gains. The elbow pole is subsequently refined by the dynamic
/// swivel stage.
/// Returns `None` when no hips-relative anchor was bound, leaving the
/// documented fallback in charge.
/// Canonicalizes a rest bend-plane normal into a stable rearward pole
/// direction.
///
/// Cross-product normals are anti-mirrored between sides; flipping whichever
/// hemisphere points forward makes both sides share the legacy rearward pole
/// convention, so mirrored models get mirrored (not inverted) poles.
fn canonical_pole_direction(normal: Vec3) -> Vec3 {
    if normal.dot(Vec3::NEG_Z) < 0.0 {
        -normal
    } else {
        normal
    }
}

/// Stage 1b (Issue #169): dynamic elbow swivel / pole correction.
///
/// The rest bend-plane normal from the binding-time elbow reference is the
/// base bend direction. It is rotated around the shoulder -> hand axis by a
/// side-signed swivel angle scaled by the pole influence, and continuously
/// faded to zero as the hand target approaches the chest center over
/// `width = ratio * body_scale`. Degenerate geometry degrades to the
/// unmodified rest-plane pole instead of producing NaNs or flipping.
fn swivel_adjusted_elbow_pole(
    input: &ArmPipelineInput<'_>,
    wrist_target: Vec3,
    total_arm_length: f32,
) -> Option<Vec3> {
    let profile = input.dynamic_profile;
    let magnitude = total_arm_length * input.legacy_profile.elbow_pole_offset_ratio;
    if !magnitude.is_finite() || magnitude < 0.0 {
        return None;
    }
    let base_direction = match input.motion.elbow_reference.as_ref() {
        Some(reference) => canonical_pole_direction(reference.normal),
        // No usable reference plane: keep the legacy rearward pole policy.
        None => Vec3::NEG_Z,
    };
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
            let t = (distance / width).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }
        _ => 1.0,
    };

    let angle = side_sign * profile.elbow_swivel_radians * profile.pole_influence * fade;
    let direction = if angle.abs() <= f32::EPSILON {
        base_direction
    } else {
        let axis = (wrist_target - input.chain.rest.upper_arm.position)
            .try_normalize()
            .filter(|axis| axis.is_finite())?;
        let rotated = Quat::from_axis_angle(axis, angle) * base_direction;
        if rotated.is_finite() && rotated.length_squared() > f32::EPSILON {
            rotated.normalize()
        } else {
            base_direction
        }
    };

    let pole = input.chain.rest.elbow.position + direction * magnitude;
    pole.is_finite().then_some(pole)
}

/// Version of the persisted dynamic arm profile format.
pub const DYNAMIC_ARM_PROFILE_OVERRIDE_VERSION: u32 = 2;

/// Versioned, persisted per-model dynamic arm profile.
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
    /// Hips-relative hand anchor offsets as body-scale fractions.
    pub hand_anchor_ratio: [f32; 3],
    /// Per-axis head/body follow gains for the hand target.
    pub compensation_gains: [f32; 3],
    /// Elbow swivel magnitude at the default anchor.
    pub elbow_swivel_radians: f32,
    /// Swivel fade width as a fraction of body scale.
    pub swivel_transition_width_ratio: f32,
    /// Weak pole influence of the swivel correction.
    pub pole_influence: f32,
    /// Forearm twist relax weight.
    pub twist_relax_weight: f32,
    /// Twist redistribution crossfade between parent and child.
    pub twist_parent_child_crossfade: f32,
    /// Optional shoulder elevation trim (negative lowers).
    pub shoulder_elevation_trim_radians: f32,
}

impl DynamicArmProfileOverride {
    /// Creates a persisted override from validated runtime data.
    #[must_use]
    pub fn from_profile(profile: DynamicArmProfile) -> Self {
        Self {
            schema_version: DYNAMIC_ARM_PROFILE_OVERRIDE_VERSION,
            hand_anchor_ratio: profile.hand_anchor_ratio.to_array(),
            compensation_gains: profile.compensation_gains.to_array(),
            elbow_swivel_radians: profile.elbow_swivel_radians,
            swivel_transition_width_ratio: profile.swivel_transition_width_ratio,
            pole_influence: profile.pole_influence,
            twist_relax_weight: profile.twist_relax_weight,
            twist_parent_child_crossfade: profile.twist_parent_child_crossfade,
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
            hand_anchor_ratio: Vec3::from_array(self.hand_anchor_ratio),
            compensation_gains: Vec3::from_array(self.compensation_gains),
            elbow_swivel_radians: self.elbow_swivel_radians,
            swivel_transition_width_ratio: self.swivel_transition_width_ratio,
            pole_influence: self.pole_influence,
            twist_relax_weight: self.twist_relax_weight,
            twist_parent_child_crossfade: self.twist_parent_child_crossfade,
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
    // Right-handed VRM/glTF basis: the model faces +Z and its left side is
    // +X (glTF defines -X as right). Anchoring each hand with an inverted
    // sign drives the arms across the body, so the left side must resolve to
    // the +X lateral component.
    let side_sign = match input.chain.side {
        crate::arm::ArmSide::Left => 1.0,
        crate::arm::ArmSide::Right => -1.0,
    };
    let scale = if input.body_scale_meters.is_finite() && input.body_scale_meters > 0.0 {
        input.body_scale_meters
    } else {
        crate::body_scale::DEFAULT_BODY_SCALE_METERS
    };
    let base = Vec3::new(
        side_sign * profile.hand_anchor_ratio.x.abs() * scale,
        profile.hand_anchor_ratio.y * scale,
        profile.hand_anchor_ratio.z * scale,
    );
    let follow = (input.head_offset + input.body_offset) * profile.compensation_gains;
    // Recover the hips rest origin from the bound wrist/anchor pair so the
    // target stays hips-relative even though the solver works in rest space.
    let hips_rest = input.chain.rest.wrist.position - anchor.translation_from_hips;
    let lag = torso_lag_rotation(input.torso_delta);
    let wrist = hips_rest + lag * (base + follow);
    if !wrist.is_finite() {
        return None;
    }
    let total = input.chain.rest.total_arm_length;
    if !total.is_finite() || total <= 1.0e-4 {
        return None;
    }
    let elbow_pole = swivel_adjusted_elbow_pole(input, wrist, total)?;
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
/// Runs after the position-input bridge so it consumes the same shaped
/// head/body channels, and before `apply_default_arm_pose`, which stays the
/// only arm Transform writer. When lifecycle is not Ready, the selected
/// source is not the virtual hand, or no control frame is available, targets
/// clear so the compositor falls back to its static default pose.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn update_dynamic_arm_targets(
    lifecycle: Res<AvatarLifecycle>,
    selection: Res<ArmSourceSelection>,
    overrides: Option<Res<crate::arm_pose::ArmPoseOverrideStore>>, // per-model profiles
    control_frame: Res<crate::unload::ActiveControlFrame>,
    mirror: Option<Res<crate::mirror::AvatarMotionMirror>>,
    body_profiles: Option<Res<crate::body_motion::BodyMotionProfiles>>,
    mut roots: Query<(
        &AvatarBinding,
        &crate::load::AvatarAssetId,
        &crate::arm_motion_geometry::ArmMotionGeometry,
        &crate::body_scale::BodyScaleMeters,
        Option<&mut DynamicArmTargets>,
    )>,
    torso_rotations: Query<(&GlobalTransform, &RestGlobalTransform)>,
) {
    let Ok((binding, model_id, motion, scale, targets)) = roots.single_mut() else {
        return;
    };
    // Per-model dynamic profile; automatic defaults otherwise.
    let profile = overrides
        .as_deref()
        .and_then(|store| store.dynamic_profile_for(model_id))
        .unwrap_or(selection.profile);

    // Any condition that breaks generation or source authority clears the
    // dynamic override so the compositor falls back to its static default.
    let active = lifecycle.state() == crate::lifecycle::AvatarLifecycleState::Ready
        && selection.mode == ArmPoseSourceKind::VirtualHandAnchor
        && control_frame.generation == binding.generation
        && control_frame.frame.is_some();

    let Some(mut targets) = targets else {
        return;
    };
    if !active {
        *targets = DynamicArmTargets::default();
        return;
    }
    // `active` already verified a frame exists; keep the invariant explicit
    // without panicking on runtime data.
    let Some(frame) = control_frame.frame.as_ref() else {
        *targets = DynamicArmTargets::default();
        return;
    };
    if targets.generation == Some(binding.generation)
        && targets.source_seq == Some(frame.source_seq)
    {
        // Same input frame: keep the existing resolution (idempotent).
        return;
    }
    let default_profiles = crate::body_motion::BodyMotionProfiles::default();
    let body_profiles = body_profiles.as_deref().unwrap_or(&default_profiles);
    let mirrored = mirror.as_deref().is_none_or(|mirror| mirror.is_enabled());
    let (head_offset, body_offset) =
        crate::body_motion::position_channels(frame, mirrored, body_profiles, scale.scale_meters)
            .unwrap_or((Vec3::ZERO, Vec3::ZERO));

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

    let resolve = |chain: Option<&ArmChainBinding>,
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

#[cfg(test)]
mod tests {
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
                total_arm_length: upper_origin.distance(wrist),
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
                - input.motion.hand_anchor.as_ref().unwrap().translation_from_hips);
        let expected_offset = Quat::from_rotation_y(TORSO_LAG_SHARE * std::f32::consts::FRAC_PI_2)
            * neutral_offset;
        let expected = expected_offset
            + (chain.rest.wrist.position
                - input.motion.hand_anchor.as_ref().unwrap().translation_from_hips);
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
        let left_input = ArmPipelineInput::binding_time(&lc, &lm, ArmPoseProfile::default());
        let right_input = ArmPipelineInput::binding_time(&rc, &rm, ArmPoseProfile::default());
        let (_, l_outcome) = resolve_arm_pose(&left_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap();
        let (_, r_outcome) = resolve_arm_pose(&right_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap();

        // Pole offsets from the rest elbow must be mirror images.
        let l_pole = l_outcome.hand_target.elbow_pole - lc.rest.elbow.position;
        let r_pole = r_outcome.hand_target.elbow_pole - rc.rest.elbow.position;
        assert!((l_pole.x + r_pole.x).abs() < 1e-4);
        assert!((l_pole.y - r_pole.y).abs() < 1e-5);
        assert!((l_pole.z - r_pole.z).abs() < 1e-5);
    }

    #[test]
    fn swivel_fades_continuously_toward_the_chest_center() {
        let (chain, _) = anchored_motion(ArmSide::Right);
        let profile = DynamicArmProfile::default();
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
        let base =
            canonical_pole_direction(motion_with_center.elbow_reference.as_ref().unwrap().normal);

        // Drive the stage directly with controlled wrist targets so the
        // measurement isolates the swivel fade. Offsets are chosen relative
        // to the chest center so the sampled distances sweep the transition
        // band monotonically.
        let shoulder = chain.rest.upper_arm.position;
        let rest_target = chain.rest.wrist.position;
        let anchor_delta_from_center = (rest_target - center).x;
        let pole_at = |lateral: f32| {
            let input = ArmPipelineInput::binding_time(
                &chain,
                &motion_with_center,
                ArmPoseProfile::default(),
            );
            swivel_adjusted_elbow_pole(
                &input,
                rest_target + Vec3::X * lateral,
                chain.rest.total_arm_length,
            )
            .unwrap()
        };

        let angle_at = |delta_from_center: f32| {
            // Place the target so its horizontal offset from the chest
            // center is exactly `delta_from_center`.
            let lateral = -anchor_delta_from_center + delta_from_center;
            let offset = pole_at(lateral) - chain.rest.elbow.position;
            let axis = ((rest_target + Vec3::X * lateral) - shoulder).normalize();
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
    fn degenerate_elbow_reference_keeps_a_safe_finite_pole() {
        let (chain, _motion) = anchored_motion(ArmSide::Left);
        // Motion geometry with degenerate reference plane and no centers.
        let motion = crate::arm_motion_geometry::ArmMotionRestGeometry {
            side: ArmSide::Left,
            hand_anchor: None,
            torso_center: None,
            forearm_twist: None,
            elbow_reference: None,
        };
        let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        let (pose, outcome) = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap();
        // No anchor -> legacy fallback path; output stays finite.
        assert!(outcome.hand_target.elbow_pole.is_finite());
        assert!(pose.upper_arm_delta.is_finite());
    }

    // ---- Issue #170: swing-twist decomposition and forearm relaxer ----

    #[test]
    fn swing_twist_decomposition_handles_pure_and_mixed_rotations() {
        let axis = Vec3::X;
        // Pure swing around Y.
        let pure_swing = Quat::from_rotation_y(0.6);
        let (s, t) = decompose_swing_twist(pure_swing, axis).unwrap();
        assert!(t.angle_between(Quat::IDENTITY) < 1e-4, "no twist component");
        assert!(pure_swing.angle_between(s * t) < 1e-5);
        // Pure twist around the axis.
        let pure_twist = Quat::from_axis_angle(axis, 0.8);
        let (_, t2) = decompose_swing_twist(pure_twist, axis).unwrap();
        assert!(pure_twist.angle_between(t2) < 1e-5);
        // Mixed rotation reconstructs exactly.
        let mixed = Quat::from_rotation_z(-0.4) * Quat::from_axis_angle(axis, 0.9);
        let (s3, t3) = decompose_swing_twist(mixed, axis).unwrap();
        assert!(mixed.angle_between(s3 * t3) < 1e-5);
        for q in [s, t, t2, s3, t3] {
            assert!(q.is_finite(), "finite normalized outputs");
            assert!((q.length() - 1.0).abs() < 1e-4);
        }
        // Near +/-180 degree twist stays on the shortest arc and finite.
        let flip = Quat::from_axis_angle(axis, std::f32::consts::PI - 1e-3);
        let (_, tf) = decompose_swing_twist(flip, axis).unwrap();
        assert!(tf.is_finite());
        assert!(flip.angle_between(tf) < 1e-3);
        // Degenerate inputs are rejected instead of producing NaNs.
        assert!(decompose_swing_twist(Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0), axis).is_none());
        assert!(decompose_swing_twist(Quat::IDENTITY, Vec3::ZERO).is_none());
    }

    #[test]
    fn zero_twist_relax_weight_reproduces_the_plain_solution() {
        let (chain, motion) = anchored_motion(ArmSide::Right);
        let target = crate::arm::ArmIkTarget {
            wrist: chain.rest.wrist.position + Vec3::new(0.05, -0.08, 0.06),
            elbow_pole: chain.rest.elbow.position + Vec3::NEG_Z * 0.03,
        };
        let plain = crate::arm_pose::solve_stage(&chain, ArmPoseProfile::default(), &target, None)
            .unwrap()
            .unwrap();
        let zero_params = TwistRelaxParams {
            chain: &chain,
            motion: &motion,
            weight: 0.0,
            crossfade: 0.9,
        };
        let relaxed_zero = crate::arm_pose::solve_stage(
            &chain,
            ArmPoseProfile::default(),
            &target,
            Some(&zero_params),
        )
        .unwrap()
        .unwrap();
        assert_eq!(plain, relaxed_zero, "weight=0 must be a no-op");
    }

    #[test]
    fn twist_relaxer_reduces_relative_forearm_twist_without_nan() {
        let (chain, motion) = anchored_motion(ArmSide::Right);
        let profile = DynamicArmProfile::default();
        let mut input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        input.dynamic_profile = profile;
        // A strongly twisted hand target via compensation offsets exercises
        // the relaxer on a large solved twist.
        input.head_offset = Vec3::new(0.12, -0.05, 0.18);
        let pose = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        assert!(pose.upper_arm_delta.is_finite() && pose.lower_arm_delta.is_finite());
        assert!((pose.upper_arm_delta.length() - 1.0).abs() < 1e-4);
        assert!((pose.lower_arm_delta.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn twist_relaxer_is_mirror_symmetric_across_sides() {
        let (lc, lm, rc, rm) = mirrored_pair();
        let left_input = ArmPipelineInput::binding_time(&lc, &lm, ArmPoseProfile::default());
        let right_input = ArmPipelineInput::binding_time(&rc, &rm, ArmPoseProfile::default());
        let l = resolve_arm_pose(&left_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        let r = resolve_arm_pose(&right_input, ArmPoseSourceKind::VirtualHandAnchor)
            .unwrap()
            .unwrap()
            .0;
        // Mirrored solutions must have equal twist magnitude on the forearm:
        // compare lower-arm delta angles to their respective rest axes.
        let angle_of = |chain: &ArmChainBinding, delta: Quat| {
            let axis_local = chain.rest.elbow.global_rotation.inverse() * chain_motion_axis(chain);
            decompose_swing_twist(delta, axis_local).map(|(_, t)| t.angle_between(Quat::IDENTITY))
        };
        let l_angle = angle_of(&lc, l.lower_arm_delta).expect("twist");
        let r_angle = angle_of(&rc, r.lower_arm_delta).expect("twist");
        assert!((l_angle - r_angle).abs() < 1e-3, "{l_angle} vs {r_angle}");
    }

    fn chain_motion_axis(chain: &ArmChainBinding) -> Vec3 {
        (chain.rest.wrist.position - chain.rest.elbow.position)
            .try_normalize()
            .unwrap_or(Vec3::X)
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
    fn nonzero_trim_propagates_weakly_to_the_elbow_and_wrist() {
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
        // The trim reaches the terminal bones in small decaying shares: the
        // elbow follows weakly and the wrist follows even more weakly, so the
        // arm bends with the shoulder instead of rotating as a rigid stick.
        let upper_change = before.upper_arm_delta.angle_between(after.upper_arm_delta);
        let lower_change = before.lower_arm_delta.angle_between(after.lower_arm_delta);
        let shoulder_change = sh_before.angle_between(sh_after);
        assert!(upper_change > 1e-4, "upper arm inherits part of the trim");
        assert!(lower_change > 1e-4, "forearm inherits part of the trim");
        assert!(
            upper_change < shoulder_change,
            "downstream share stays smaller than the shoulder change"
        );
        assert!(
            lower_change < upper_change,
            "motion decays toward the terminal bones"
        );
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
        // The legacy default drop (70 degrees from the T-pose) must never be
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
}
