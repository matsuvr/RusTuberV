use super::{BodyTracking, compute_additive_rotation};
use crate::prelude::*;
use crate::system_set::VrmSystemSets;
use crate::vrm::{RestGlobalTransform, RestTransform};
use bevy::app::{AnimationSystems, App};
use bevy::prelude::*;
use std::collections::HashMap;

const BONE_COUNT: usize = 6;
const HEAD: usize = 0;
const NECK: usize = 1;
const UPPER_CHEST: usize = 2;
const CHEST: usize = 3;
const SPINE: usize = 4;
const HIPS: usize = 5;

/// Calibrated semantic head pose supplied directly to [`BodyTracking`].
///
/// Angles are radians. Positive yaw turns toward image right, positive pitch
/// raises the chin, and positive roll is clockwise in the unmirrored image.
#[derive(Component, Debug, Clone, Copy, Reflect, Default, PartialEq)]
#[reflect(Component)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct BodyTrackingPoseInput {
    /// Calibrated yaw in radians.
    pub yaw_radians: f32,
    /// Calibrated pitch in radians.
    pub pitch_radians: f32,
    /// Calibrated roll in radians.
    pub roll_radians: f32,
    /// Confidence multiplier in the inclusive range `0.0..=1.0`.
    pub weight: f32,
    /// Whether tracking is currently active.
    pub active: bool,
}

/// Named per-bone weights in `head -> neck -> upperChest -> chest -> spine -> hips` order.
#[derive(Debug, Clone, Copy, Reflect, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct BodyBoneWeights {
    /// Head contribution.
    pub head: f32,
    /// Neck contribution.
    pub neck: f32,
    /// Upper-chest contribution.
    pub upper_chest: f32,
    /// Chest contribution.
    pub chest: f32,
    /// Spine contribution.
    pub spine: f32,
    /// Hips contribution. Small values let head motion propagate through the
    /// whole body; the legs follow the hips automatically as child bones.
    pub hips: f32,
}

impl BodyBoneWeights {
    const fn new(
        head: f32,
        neck: f32,
        upper_chest: f32,
        chest: f32,
        spine: f32,
        hips: f32,
    ) -> Self {
        Self {
            head,
            neck,
            upper_chest,
            chest,
            spine,
            hips,
        }
    }

    fn as_array(self) -> [f32; BONE_COUNT] {
        [
            self.head,
            self.neck,
            self.upper_chest,
            self.chest,
            self.spine,
            self.hips,
        ]
    }

    fn from_array(values: [f32; BONE_COUNT]) -> Self {
        Self::new(
            values[HEAD],
            values[NECK],
            values[UPPER_CHEST],
            values[CHEST],
            values[SPINE],
            values[HIPS],
        )
    }
}

/// Per-bone exponential smoothing half-lives in seconds.
#[derive(Debug, Clone, Copy, Reflect, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct BodyBoneHalfLives {
    /// Head half-life in seconds.
    pub head_seconds: f32,
    /// Neck half-life in seconds.
    pub neck_seconds: f32,
    /// Upper-chest half-life in seconds.
    pub upper_chest_seconds: f32,
    /// Chest half-life in seconds.
    pub chest_seconds: f32,
    /// Spine half-life in seconds.
    pub spine_seconds: f32,
    /// Hips half-life in seconds.
    pub hips_seconds: f32,
}

impl Default for BodyBoneHalfLives {
    fn default() -> Self {
        Self {
            head_seconds: 0.055,
            neck_seconds: 0.105,
            upper_chest_seconds: 0.180,
            chest_seconds: 0.285,
            spine_seconds: 0.450,
            hips_seconds: 0.650,
        }
    }
}

impl BodyBoneHalfLives {
    fn as_array(self) -> [f32; BONE_COUNT] {
        [
            self.head_seconds,
            self.neck_seconds,
            self.upper_chest_seconds,
            self.chest_seconds,
            self.spine_seconds,
            self.hips_seconds,
        ]
    }
}

/// Per-axis rotation limit for one humanoid bone, in radians.
#[derive(Debug, Clone, Copy, Reflect, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct BoneRotationLimit {
    /// Maximum absolute yaw in radians.
    pub yaw_radians: f32,
    /// Maximum absolute pitch in radians.
    pub pitch_radians: f32,
    /// Maximum absolute roll in radians.
    pub roll_radians: f32,
}

impl BoneRotationLimit {
    fn from_degrees(
        yaw: f32,
        pitch: f32,
        roll: f32,
    ) -> Self {
        Self {
            yaw_radians: yaw.to_radians(),
            pitch_radians: pitch.to_radians(),
            roll_radians: roll.to_radians(),
        }
    }
}

/// Rotation limits for the direct-pose humanoid chain.
#[derive(Debug, Clone, Copy, Reflect, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct BodyBoneRotationLimits {
    /// Head limits.
    pub head: BoneRotationLimit,
    /// Neck limits.
    pub neck: BoneRotationLimit,
    /// Upper-chest limits.
    pub upper_chest: BoneRotationLimit,
    /// Chest limits.
    pub chest: BoneRotationLimit,
    /// Spine limits.
    pub spine: BoneRotationLimit,
    /// Hips limits.
    pub hips: BoneRotationLimit,
}

impl Default for BodyBoneRotationLimits {
    fn default() -> Self {
        Self {
            head: BoneRotationLimit::from_degrees(45.0, 30.0, 25.0),
            neck: BoneRotationLimit::from_degrees(25.0, 20.0, 15.0),
            // Conservative torso pitch/roll limits avoid folding clothing and
            // shoulder rigs while still allowing visible upper-body follow.
            upper_chest: BoneRotationLimit::from_degrees(18.0, 8.0, 6.0),
            chest: BoneRotationLimit::from_degrees(12.0, 4.0, 0.0),
            spine: BoneRotationLimit::from_degrees(8.0, 0.0, 0.0),
            // Hips carry only a faint whisper of the head pose so the whole
            // body sways with it without the feet visibly sliding.
            hips: BoneRotationLimit::from_degrees(10.0, 4.0, 4.0),
        }
    }
}

impl BodyBoneRotationLimits {
    fn as_array(self) -> [BoneRotationLimit; BONE_COUNT] {
        [
            self.head,
            self.neck,
            self.upper_chest,
            self.chest,
            self.spine,
            self.hips,
        ]
    }
}

/// Axis distribution and response settings for direct-pose [`BodyTracking`].
#[derive(Component, Debug, Clone, Copy, Reflect, PartialEq)]
#[reflect(Component)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct BodyTrackingProfile {
    /// Distribution below the torso engagement threshold.
    pub small_yaw_weights: BodyBoneWeights,
    /// Distribution at and above full torso engagement.
    pub large_yaw_weights: BodyBoneWeights,
    /// Pitch distribution.
    pub pitch_weights: BodyBoneWeights,
    /// Roll distribution.
    pub roll_weights: BodyBoneWeights,
    /// Absolute yaw where torso engagement starts, in radians.
    pub yaw_body_engagement_start_radians: f32,
    /// Absolute yaw where torso engagement is full, in radians.
    pub yaw_body_engagement_full_radians: f32,
    /// Per-bone response half-lives.
    pub bone_half_lives: BodyBoneHalfLives,
    /// Per-bone rotation limits.
    pub bone_rotation_limits: BodyBoneRotationLimits,
}

impl Default for BodyTrackingProfile {
    fn default() -> Self {
        Self {
            small_yaw_weights: BodyBoneWeights::new(0.65, 0.35, 0.0, 0.0, 0.0, 0.0),
            large_yaw_weights: BodyBoneWeights::new(0.40, 0.22, 0.16, 0.10, 0.07, 0.05),
            pitch_weights: BodyBoneWeights::new(0.68, 0.25, 0.06, 0.01, 0.0, 0.0),
            roll_weights: BodyBoneWeights::new(0.72, 0.23, 0.05, 0.0, 0.0, 0.0),
            yaw_body_engagement_start_radians: 12.0_f32.to_radians(),
            yaw_body_engagement_full_radians: 45.0_f32.to_radians(),
            bone_half_lives: BodyBoneHalfLives::default(),
            bone_rotation_limits: BodyBoneRotationLimits::default(),
        }
    }
}

pub(super) fn register(app: &mut App) {
    app.register_type::<BodyTrackingPoseInput>()
        .register_type::<BodyTrackingProfile>()
        .register_type::<BodyBoneWeights>()
        .register_type::<BodyBoneHalfLives>()
        .register_type::<BoneRotationLimit>()
        .register_type::<BodyBoneRotationLimits>()
        .add_systems(
            PostUpdate,
            apply_direct_body_tracking
                .after(AnimationSystems)
                .before(VrmSystemSets::GazeControl)
                .before(VrmSystemSets::Constraints)
                .run_if(any_with_component::<BodyTrackingPoseInput>),
        );
}

fn smoothstep(
    edge0: f32,
    edge1: f32,
    value: f32,
) -> f32 {
    if !edge0.is_finite() || !edge1.is_finite() || !value.is_finite() || edge1 <= edge0 {
        return 0.0;
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp_weights(
    a: BodyBoneWeights,
    b: BodyBoneWeights,
    factor: f32,
) -> BodyBoneWeights {
    let factor = if factor.is_finite() {
        factor.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let a = a.as_array();
    let b = b.as_array();
    BodyBoneWeights::from_array(std::array::from_fn(|index| {
        a[index] + (b[index] - a[index]) * factor
    }))
}

fn normalize_available_weights(
    weights: BodyBoneWeights,
    available: [bool; BONE_COUNT],
) -> BodyBoneWeights {
    let mut values = weights.as_array();
    for (index, value) in values.iter_mut().enumerate() {
        if !available[index] || !value.is_finite() || *value <= 0.0 {
            *value = 0.0;
        }
    }
    let sum: f32 = values.iter().sum();
    if !sum.is_finite() || sum <= f32::EPSILON {
        return BodyBoneWeights::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    BodyBoneWeights::from_array(values.map(|value| value / sum))
}

fn half_life_alpha(
    half_life_seconds: f32,
    delta_seconds: f32,
) -> f32 {
    if !half_life_seconds.is_finite() || half_life_seconds <= 0.0 {
        return 1.0;
    }
    if !delta_seconds.is_finite() || delta_seconds <= 0.0 {
        return 0.0;
    }
    1.0 - (-std::f32::consts::LN_2 * delta_seconds / half_life_seconds).exp()
}

fn shortest_angle_delta(
    current: f32,
    target: f32,
) -> f32 {
    let current = if current.is_finite() { current } else { 0.0 };
    let target = if target.is_finite() { target } else { 0.0 };
    (target - current + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI
}

fn smooth_angle_half_life(
    current: f32,
    target: f32,
    half_life: f32,
    dt: f32,
) -> f32 {
    let current = if current.is_finite() { current } else { 0.0 };
    let target = if target.is_finite() { target } else { 0.0 };
    let alpha = half_life_alpha(half_life, dt);
    if alpha >= 1.0 {
        return target;
    }
    current + shortest_angle_delta(current, target) * alpha
}

fn clamp_angle(
    angle: f32,
    limit: f32,
) -> f32 {
    if !angle.is_finite() || !limit.is_finite() || limit <= 0.0 {
        return 0.0;
    }
    angle.clamp(-limit, limit)
}

fn sanitize_input(input: &BodyTrackingPoseInput) -> Vec3 {
    if !input.active {
        return Vec3::ZERO;
    }
    let weight = if input.weight.is_finite() {
        input.weight.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let finite_or_zero = |value: f32| if value.is_finite() { value } else { 0.0 };
    Vec3::new(
        finite_or_zero(input.yaw_radians) * weight,
        finite_or_zero(input.pitch_radians) * weight,
        finite_or_zero(input.roll_radians) * weight,
    )
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct DirectBoneState {
    base: Quat,
    last_delta: Quat,
    smoothed_angles: Vec3,
    initialized: bool,
}

impl Default for DirectBoneState {
    fn default() -> Self {
        Self {
            base: Quat::IDENTITY,
            last_delta: Quat::IDENTITY,
            smoothed_angles: Vec3::ZERO,
            initialized: false,
        }
    }
}

#[derive(Clone, Copy)]
struct DirectBoneEntry {
    index: usize,
    entity: Entity,
}

fn finite_normalized_or(
    value: Quat,
    fallback: Quat,
) -> Quat {
    let length_squared = value.length_squared();
    if value.is_finite() && length_squared.is_finite() && length_squared > f32::EPSILON {
        value.normalize()
    } else {
        fallback
    }
}

fn direct_tracking_target(
    angles: Vec3,
    root_rest_rotation: Quat,
    rest_tf: &RestTransform,
    rest_gtf: &RestGlobalTransform,
) -> Quat {
    let model_delta = Quat::from_euler(EulerRot::YXZ, angles.x, -angles.y, -angles.z);
    let bone_rest_model = root_rest_rotation.inverse() * rest_gtf.rotation();
    let local_delta = bone_rest_model.inverse() * model_delta * bone_rest_model;
    finite_normalized_or(rest_tf.rotation * local_delta, rest_tf.rotation)
}

pub(super) fn refresh_parent_global(
    root: Entity,
    parent: Entity,
    root_global: GlobalTransform,
    transforms: &mut Query<(&mut Transform, &mut GlobalTransform), Without<Vrm>>,
    child_ofs: &Query<&ChildOf>,
    computed: &mut HashMap<Entity, GlobalTransform>,
) -> Option<GlobalTransform> {
    if parent == root {
        return Some(root_global);
    }
    if let Some(global) = computed.get(&parent) {
        return Some(*global);
    }

    let mut path = Vec::new();
    let mut cursor = parent;
    let base = loop {
        if cursor == root {
            break root_global;
        }
        if let Some(global) = computed.get(&cursor) {
            break *global;
        }
        path.push(cursor);
        let Ok(child_of) = child_ofs.get(cursor) else {
            return transforms.get(parent).ok().map(|(_, global)| *global);
        };
        cursor = child_of.parent();
    };

    let mut parent_global = base;
    for entity in path.into_iter().rev() {
        let Ok((transform, mut global)) = transforms.get_mut(entity) else {
            return None;
        };
        *global = parent_global.mul_transform(*transform);
        parent_global = *global;
        computed.insert(entity, parent_global);
    }
    Some(parent_global)
}

/// Applies direct pose input to the humanoid upper-body chain and hips.
///
/// Applications normally use [`crate::prelude::VrmPlugin`], which registers
/// this system after Bevy animation and before VRM constraints. The function
/// is public so integration tests and custom schedules can verify that path.
pub fn apply_direct_body_tracking(
    vrms: Query<
        (
            Entity,
            &BodyTrackingPoseInput,
            Option<&BodyTrackingProfile>,
            &HeadBoneEntity,
            Option<&NeckBoneEntity>,
            Option<&UpperChestBoneEntity>,
            Option<&ChestBoneEntity>,
            Option<&SpineBoneEntity>,
            Option<&HipsBoneEntity>,
        ),
        With<BodyTracking>,
    >,
    root_globals: Query<&GlobalTransform, With<Vrm>>,
    mut transforms: Query<(&mut Transform, &mut GlobalTransform), Without<Vrm>>,
    child_ofs: Query<&ChildOf>,
    rests: Query<(&RestTransform, &RestGlobalTransform)>,
    time: Res<Time>,
    mut bone_states: Local<HashMap<Entity, DirectBoneState>>,
    mut root_rest_rotations: Local<HashMap<Entity, Quat>>,
) {
    let dt = time.delta_secs();
    let default_profile = BodyTrackingProfile::default();

    // Direct-pose state is local to this system rather than stored on model
    // entities. Drop entries as soon as an avatar (or one of its humanoid
    // bones) is despawned so repeated model replacement cannot retain stale
    // smoothing state indefinitely.
    bone_states.retain(|entity, _| transforms.contains(*entity));
    root_rest_rotations.retain(|entity, _| root_globals.contains(*entity));

    for (root, input, profile, head, neck, upper_chest, chest, spine, hips) in vrms.iter() {
        let Ok(root_global) = root_globals.get(root) else {
            continue;
        };
        let root_rest_rotation = *root_rest_rotations
            .entry(root)
            .or_insert(root_global.rotation());
        let profile = profile.unwrap_or(&default_profile);
        let pose = sanitize_input(input);
        let available = [
            true,
            neck.is_some(),
            upper_chest.is_some(),
            chest.is_some(),
            spine.is_some(),
            hips.is_some(),
        ];
        let engagement = smoothstep(
            profile.yaw_body_engagement_start_radians,
            profile.yaw_body_engagement_full_radians,
            pose.x.abs(),
        );
        let yaw_weights = normalize_available_weights(
            lerp_weights(
                profile.small_yaw_weights,
                profile.large_yaw_weights,
                engagement,
            ),
            available,
        )
        .as_array();
        let pitch_weights =
            normalize_available_weights(profile.pitch_weights, available).as_array();
        let roll_weights = normalize_available_weights(profile.roll_weights, available).as_array();
        let half_lives = profile.bone_half_lives.as_array();
        let limits = profile.bone_rotation_limits.as_array();

        let mut chain = Vec::with_capacity(BONE_COUNT);
        if let Some(hips) = hips {
            chain.push(DirectBoneEntry {
                index: HIPS,
                entity: hips.0,
            });
        }
        if let Some(spine) = spine {
            chain.push(DirectBoneEntry {
                index: SPINE,
                entity: spine.0,
            });
        }
        if let Some(chest) = chest {
            chain.push(DirectBoneEntry {
                index: CHEST,
                entity: chest.0,
            });
        }
        if let Some(upper_chest) = upper_chest {
            chain.push(DirectBoneEntry {
                index: UPPER_CHEST,
                entity: upper_chest.0,
            });
        }
        if let Some(neck) = neck {
            chain.push(DirectBoneEntry {
                index: NECK,
                entity: neck.0,
            });
        }
        chain.push(DirectBoneEntry {
            index: HEAD,
            entity: head.0,
        });

        let mut computed_globals = HashMap::with_capacity(BONE_COUNT * 2);
        for bone in chain {
            let Ok((rest_tf, rest_gtf)) = rests.get(bone.entity) else {
                continue;
            };
            let Some(parent) = child_ofs.get(bone.entity).ok().map(ChildOf::parent) else {
                continue;
            };
            let Some(parent_global) = refresh_parent_global(
                root,
                parent,
                *root_global,
                &mut transforms,
                &child_ofs,
                &mut computed_globals,
            ) else {
                continue;
            };

            let target_angles = Vec3::new(
                clamp_angle(
                    pose.x * yaw_weights[bone.index],
                    limits[bone.index].yaw_radians,
                ),
                clamp_angle(
                    pose.y * pitch_weights[bone.index],
                    limits[bone.index].pitch_radians,
                ),
                clamp_angle(
                    pose.z * roll_weights[bone.index],
                    limits[bone.index].roll_radians,
                ),
            );
            let state = bone_states.entry(bone.entity).or_default();
            state.smoothed_angles = Vec3::new(
                smooth_angle_half_life(
                    state.smoothed_angles.x,
                    target_angles.x,
                    half_lives[bone.index],
                    dt,
                ),
                smooth_angle_half_life(
                    state.smoothed_angles.y,
                    target_angles.y,
                    half_lives[bone.index],
                    dt,
                ),
                smooth_angle_half_life(
                    state.smoothed_angles.z,
                    target_angles.z,
                    half_lives[bone.index],
                    dt,
                ),
            );
            let tracking_target = direct_tracking_target(
                state.smoothed_angles,
                root_rest_rotation,
                rest_tf,
                rest_gtf,
            );

            let Ok((mut transform, mut global)) = transforms.get_mut(bone.entity) else {
                continue;
            };
            let expected_previous = state.base * state.last_delta;
            let animation_changed = !state.initialized
                || !transform.rotation.is_finite()
                || transform.rotation.dot(expected_previous).abs() < 0.999;
            let base = if animation_changed {
                finite_normalized_or(transform.rotation, rest_tf.rotation)
            } else {
                state.base
            };
            let delta =
                finite_normalized_or(rest_tf.rotation.inverse() * tracking_target, Quat::IDENTITY);
            let output = finite_normalized_or(
                compute_additive_rotation(base, rest_tf.rotation, tracking_target),
                base,
            );

            transform.rotation = output;
            *global = parent_global.mul_transform(*transform);
            state.base = base;
            state.last_delta = delta;
            state.initialized = true;
            computed_globals.insert(bone.entity, *global);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-5;

    fn sum(weights: BodyBoneWeights) -> f32 {
        weights.as_array().iter().sum()
    }

    #[test]
    fn default_axis_weights_sum_to_one() {
        let profile = BodyTrackingProfile::default();
        for weights in [
            profile.small_yaw_weights,
            profile.large_yaw_weights,
            profile.pitch_weights,
            profile.roll_weights,
        ] {
            assert!((sum(weights) - 1.0).abs() < EPSILON);
        }
    }

    #[test]
    fn yaw_engagement_uses_documented_boundaries_and_sign_symmetry() {
        let profile = BodyTrackingProfile::default();
        let engagement = |degrees: f32| {
            smoothstep(
                profile.yaw_body_engagement_start_radians,
                profile.yaw_body_engagement_full_radians,
                degrees.to_radians().abs(),
            )
        };
        assert_eq!(engagement(0.0), 0.0);
        assert_eq!(engagement(12.0), 0.0);
        assert_eq!(engagement(45.0), 1.0);
        assert_eq!(engagement(60.0), 1.0);
        assert!((engagement(28.5) - 0.5).abs() < EPSILON);
        assert_eq!(engagement(30.0), engagement(-30.0));
    }

    #[test]
    fn optional_bones_are_renormalized_without_activating_zero_weights() {
        let profile = BodyTrackingProfile::default();
        let cases = [
            [true, true, true, true, true, true],
            [true, true, false, true, true, true],
            [true, true, true, false, true, true],
            [true, true, false, false, true, true],
            [true, true, false, false, false, true],
            [true, true, true, true, false, true],
            [true, true, false, false, false, false],
        ];
        for available in cases {
            let normalized = normalize_available_weights(profile.large_yaw_weights, available);
            assert!((sum(normalized) - 1.0).abs() < EPSILON);
        }

        let full = normalize_available_weights(
            profile.large_yaw_weights,
            [true, true, true, true, true, true],
        );
        assert!(full.hips > 0.0, "large yaw must reach the hips");

        let small_without_torso = normalize_available_weights(
            profile.small_yaw_weights,
            [true, true, true, true, true, true],
        );
        assert_eq!(small_without_torso.upper_chest, 0.0);
        assert_eq!(small_without_torso.chest, 0.0);
        assert_eq!(small_without_torso.spine, 0.0);
        assert_eq!(small_without_torso.hips, 0.0);
    }

    #[test]
    fn axis_distribution_keeps_pitch_and_roll_out_of_zero_weight_bones() {
        let profile = BodyTrackingProfile::default();
        let pitch = normalize_available_weights(profile.pitch_weights, [true; BONE_COUNT]);
        let roll = normalize_available_weights(profile.roll_weights, [true; BONE_COUNT]);
        assert_eq!(pitch.spine, 0.0);
        assert_eq!(pitch.hips, 0.0);
        assert_eq!(roll.chest, 0.0);
        assert_eq!(roll.spine, 0.0);
        assert_eq!(roll.hips, 0.0);
        assert!((pitch.head - 0.68).abs() < EPSILON);
        assert!((roll.upper_chest - 0.05).abs() < EPSILON);
    }

    #[test]
    fn yaw_weight_interpolation_is_continuous() {
        let profile = BodyTrackingProfile::default();
        let start = profile.yaw_body_engagement_start_radians;
        let full = profile.yaw_body_engagement_full_radians;
        let center = (start + full) * 0.5;
        let before = lerp_weights(
            profile.small_yaw_weights,
            profile.large_yaw_weights,
            smoothstep(start, full, center - 1.0e-5),
        );
        let after = lerp_weights(
            profile.small_yaw_weights,
            profile.large_yaw_weights,
            smoothstep(start, full, center + 1.0e-5),
        );
        assert!((before.head - after.head).abs() < 1.0e-4);
        assert!((before.spine - after.spine).abs() < 1.0e-4);
    }

    #[test]
    fn zero_and_non_finite_weights_produce_finite_zeroes() {
        let zero = normalize_available_weights(
            BodyBoneWeights::new(0.0, f32::NAN, f32::INFINITY, -1.0, 0.0, 0.0),
            [true; BONE_COUNT],
        );
        assert_eq!(
            zero,
            BodyBoneWeights::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        );
        assert!(zero.as_array().iter().all(|value| value.is_finite()));
    }

    fn response_after(
        seconds: f32,
        fps: u32,
        half_life: f32,
    ) -> f32 {
        let dt = 1.0 / fps as f32;
        let mut current = 0.0;
        for _ in 0..(seconds * fps as f32).round() as u32 {
            current = smooth_angle_half_life(current, 1.0, half_life, dt);
        }
        current
    }

    #[test]
    fn half_life_response_is_frame_rate_independent() {
        for fps in [30, 60, 120] {
            let value = response_after(0.5, fps, 0.5);
            assert!((value - 0.5).abs() < 0.001, "fps={fps}, value={value}");
        }
    }

    #[test]
    fn head_converges_faster_than_spine() {
        let half_lives = BodyBoneHalfLives::default();
        let head = response_after(0.1, 60, half_lives.head_seconds);
        let spine = response_after(0.1, 60, half_lives.spine_seconds);
        assert!(head > spine);
    }

    #[test]
    fn zero_half_life_is_immediate_and_angles_take_shortest_path() {
        assert_eq!(smooth_angle_half_life(0.0, 1.0, 0.0, 1.0 / 60.0), 1.0);
        assert_eq!(smooth_angle_half_life(0.0, 1.0, 0.0, 0.0), 1.0);
        let current = 179.0_f32.to_radians();
        let target = -179.0_f32.to_radians();
        let delta = shortest_angle_delta(current, target);
        assert!(delta > 0.0);
        assert!((delta.to_degrees() - 2.0).abs() < 0.001);
    }

    #[test]
    fn inactive_and_non_finite_input_returns_finite_neutral() {
        let inactive = BodyTrackingPoseInput {
            yaw_radians: 1.0,
            pitch_radians: 1.0,
            roll_radians: 1.0,
            weight: 1.0,
            active: false,
        };
        assert_eq!(sanitize_input(&inactive), Vec3::ZERO);

        let invalid = BodyTrackingPoseInput {
            yaw_radians: f32::NAN,
            pitch_radians: f32::INFINITY,
            roll_radians: f32::NEG_INFINITY,
            weight: f32::NAN,
            active: true,
        };
        let sanitized = sanitize_input(&invalid);
        assert_eq!(sanitized, Vec3::ZERO);
        assert!(sanitized.is_finite());
    }

    #[test]
    fn tracking_loss_converges_to_neutral() {
        let mut current = 1.0;
        for _ in 0..240 {
            current = smooth_angle_half_life(current, 0.0, 0.1, 1.0 / 60.0);
        }
        assert!(current.abs() < 1.0e-5);
    }

    #[test]
    fn direct_rotation_preserves_semantic_axis_signs() {
        let rest_tf = RestTransform(Transform::IDENTITY);
        let rest_global = RestGlobalTransform(GlobalTransform::IDENTITY);
        let rotation = direct_tracking_target(
            Vec3::new(0.2, 0.1, 0.05),
            Quat::IDENTITY,
            &rest_tf,
            &rest_global,
        );
        let expected = Quat::from_euler(EulerRot::YXZ, 0.2, -0.1, -0.05);
        assert!(rotation.angle_between(expected) < EPSILON);
    }

    #[test]
    fn additive_rotation_preserves_animation_and_does_not_accumulate() {
        let rest = Quat::from_rotation_y(0.15);
        let animated = Quat::from_rotation_x(0.2) * rest;
        let tracking_target = rest * Quat::from_rotation_y(0.3);
        let first = compute_additive_rotation(animated, rest, tracking_target);
        let second = compute_additive_rotation(animated, rest, tracking_target);
        assert!(first.angle_between(second) < EPSILON);
        assert!(first.angle_between(tracking_target) > 0.01);

        let neutral = compute_additive_rotation(animated, rest, rest);
        assert!(neutral.angle_between(animated) < EPSILON);
    }
}
