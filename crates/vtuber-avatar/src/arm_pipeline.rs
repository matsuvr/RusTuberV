//! Rest/virtual hand goals and per-model profiles. All sources enter the same
//! bilateral constrained solve in `upper_limb_runtime`; this module generates
//! targets only and never applies post-solve corrections.

use bevy::prelude::*;

use crate::arm::{ArmChainBinding, ArmIkTarget, ArmPoseProfile};
use crate::arm_motion_geometry::ArmMotionRestGeometry;

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

/// Hand-target intent supplied to the shared bilateral constrained solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArmPoseSourceKind {
    /// Hips-relative virtual hand anchors (Issue #168).
    ///
    /// Uses [`ArmMotionRestGeometry`] resolved during binding (Issue #175).
    /// Position and elbow-pole intent enter the shared solver; unobserved
    /// wrist/finger articulation returns to the configured resting pose.
    #[default]
    VirtualHandAnchor,
    /// Webcam-observed shoulders/elbows/wrists (Issues #44/#47/#48).
    ///
    /// Observed wrist, elbow, palm and finger intent is supplied through
    /// [`TrackedArmControl`]. Per-channel weights set observation objectives
    /// and neutral return objectives before solving, never after the path.
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

/// Per-side inputs for one pipeline run.
///
/// Everything is immutable rest-space or semantic data; no ECS queries are
/// performed here so each stage can be unit-tested in isolation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArmPipelineInput<'a> {
    /// Bound arm chain with immutable rest-space geometry.
    pub chain: &'a ArmChainBinding,
    /// Motion geometry resolved once during binding (Issue #175).
    pub motion: &'a ArmMotionRestGeometry,
    /// Model-specific rest target and finger curl, shared by every source.
    pub pose_profile: ArmPoseProfile,
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

#[cfg(test)]
impl<'a> ArmPipelineInput<'a> {
    /// Builds pipeline input for one side from binding-time data only.
    ///
    /// Dynamic offsets default to zero so binding-time resolution matches
    /// the pre-dynamic behavior exactly.
    #[must_use]
    pub fn binding_time(
        chain: &'a ArmChainBinding,
        motion: &'a ArmMotionRestGeometry,
        pose_profile: ArmPoseProfile,
    ) -> Self {
        Self {
            chain,
            motion,
            pose_profile,
            dynamic_profile: DynamicArmProfile::default(),
            head_offset: Vec3::ZERO,
            body_offset: Vec3::ZERO,
            torso_delta: Quat::IDENTITY,
            body_scale_meters: crate::body_scale::DEFAULT_BODY_SCALE_METERS,
        }
    }
}

/// Bilateral FK poses admitted by the shared path, consumed by the compositor.
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
/// Rest-pose tuning remains in `ArmPoseProfile`; these fields describe
/// virtual-hand intent supplied to the same constrained solver.
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

pub(crate) fn virtual_hand_target(input: &ArmPipelineInput<'_>) -> Option<ArmIkTarget> {
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
    let target = crate::arm::default_arm_target(input.chain, input.pose_profile).ok()?;
    let base = target.wrist - hips_rest;
    let lag = torso_lag_rotation(input.torso_delta);
    let wrist = hips_rest + lag * (base + follow);
    if !wrist.is_finite() {
        return None;
    }
    let elbow_pole = swivel_adjusted_elbow_pole(input, wrist)?;
    let target = ArmIkTarget { wrist, elbow_pole };
    Some(target)
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

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use crate::arm::{ArmSide, RestSpaceBonePose};

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
    fn virtual_target_is_absent_without_hips_geometry() {
        let chain = sample_chain(ArmSide::Right);
        let motion = sample_motion();
        let input = ArmPipelineInput::binding_time(&chain, &motion, ArmPoseProfile::default());
        assert!(virtual_hand_target(&input).is_none());
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
        let outcome = virtual_hand_target(&input).expect("target resolved");
        assert!(outcome.wrist.is_finite());
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
            let outcome = virtual_hand_target(&input).expect("target resolved");
            // The authored rest wrist is on the chain's own side (+X for the
            // left arm); the dynamic target must stay on that same side.
            let rest_side = chain.rest.wrist.position.x.signum();
            assert_eq!(
                outcome.wrist.x.signum(),
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

        let neutral_left = virtual_hand_target(&left_input).unwrap().wrist;
        let neutral_right = virtual_hand_target(&right_input).unwrap().wrist;
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
        let moved_left = virtual_hand_target(&left_input).unwrap().wrist;
        let moved_right = virtual_hand_target(&right_input).unwrap().wrist;
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
        let base_target = virtual_hand_target(&baseline).unwrap().wrist;
        let target = virtual_hand_target(&input).unwrap().wrist;

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
        let neutral = virtual_hand_target(&input).unwrap().wrist;

        // A 90-degree left yaw of the chest must trail the hands: in rest
        // space the left-hand anchor counter-rotates toward +Z (forward).
        let turned = ArmPipelineInput {
            torso_delta: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ..input
        };
        let lagged = virtual_hand_target(&turned).unwrap().wrist;
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
        let mirrored = virtual_hand_target(&opposite).unwrap().wrist;
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
        let l_outcome = virtual_hand_target(&left_input).unwrap();
        let r_outcome = virtual_hand_target(&right_input).unwrap();

        // Pole offsets from the shoulders must be mirror images.
        let l_pole = l_outcome.elbow_pole - lc.rest.upper_arm.position;
        let r_pole = r_outcome.elbow_pole - rc.rest.upper_arm.position;
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
}

/// Signed upper-arm coronal descent used by existing pose diagnostics.
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
