use super::direct::refresh_parent_global;
use crate::prelude::*;
use crate::system_set::VrmSystemSets;
use bevy::app::{AnimationSystems, App};
use bevy::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

/// Minimum lever-arm height (meters) used for the bounded lean solve when the
/// rest chain is too degenerate to measure one.
const FALLBACK_LEVER_METERS: f32 = 0.3;

/// Lever arms shorter than this are treated as degenerate measurements.
const MIN_MEASURED_LEVER_METERS: f32 = 0.01;

/// Semantic positional input for the position-aware upper-body solve.
///
/// All offsets are meters in the engine-neutral camera-aligned frame: `x`
/// positive toward the unmirrored image right, `y` positive up, and `z`
/// positive away from the camera. Producers feed the residual virtual head
/// translation plus root/body compensation from a target-generation stage;
/// this component never carries Bevy world-space data.
///
/// Inserting this component is optional. Without it, [`BodyTracking`] keeps
/// its rotation-only behavior unchanged.
#[derive(Component, Debug, Clone, Copy, Reflect, Default, PartialEq)]
#[reflect(Component)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct BodyTrackingPositionInput {
    /// Residual translation kept at the head/neck chain, in meters.
    pub head_offset: Vec3,
    /// Root/body translation compensation, in meters.
    pub body_offset: Vec3,
    /// Confidence multiplier in the inclusive range `0.0..=1.0`.
    pub weight: f32,
    /// Whether positional tracking is currently active.
    pub active: bool,
}

/// Bounds for the position-aware upper-body solve.
#[derive(Component, Debug, Clone, Copy, Reflect, PartialEq)]
#[reflect(Component)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub struct BodyTrackingPositionProfile {
    /// Maximum total torso lean angle, in radians, distributed over the
    /// available spine/chest/upper-chest/neck bones. The analysis seed bounds
    /// body/root rotation near 15 degrees.
    pub max_lean_radians: f32,
    /// Maximum magnitude of the root/body translation offset, in meters.
    pub max_body_translation_meters: f32,
}

impl Default for BodyTrackingPositionProfile {
    fn default() -> Self {
        Self {
            max_lean_radians: 15.0_f32.to_radians(),
            max_body_translation_meters: 0.25,
        }
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

/// Returns the sanitized weighted `(head_offset, body_offset)` pair, or
/// `None` when positional tracking is inactive or carries no confidence.
fn sanitize_input(input: &BodyTrackingPositionInput) -> Option<(Vec3, Vec3)> {
    if !input.active {
        return None;
    }
    let weight = if input.weight.is_finite() {
        input.weight.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if weight <= 0.0 {
        return None;
    }
    let head = Vec3::new(
        finite_or_zero(input.head_offset.x),
        finite_or_zero(input.head_offset.y),
        finite_or_zero(input.head_offset.z),
    ) * weight;
    let body = Vec3::new(
        finite_or_zero(input.body_offset.x),
        finite_or_zero(input.body_offset.y),
        finite_or_zero(input.body_offset.z),
    ) * weight;
    Some((head, body))
}

/// Maps a semantic camera-frame offset into model/root space.
///
/// This is part of the single conversion point between the engine-neutral
/// semantic frame (`+x` image right, `+y` up, `+z` away from camera) and the
/// model rest space anchored by the immutable avatar-root rest rotation. With
/// an identity rest rotation the mapping is `(x, y, -z)`: a model facing the
/// camera has its face toward world `+Z`, so "away from camera" points along
/// `-Z`. VRM generation differences (for example a legacy Y=180 basis root)
/// are absorbed by the captured root rest rotation; no per-generation branch
/// exists here.
#[must_use]
pub fn semantic_offset_to_model(
    semantic: Vec3,
    root_rest_rotation: Quat,
) -> Vec3 {
    root_rest_rotation * Vec3::new(semantic.x, semantic.y, -semantic.z)
}

fn clamp_vector_length(
    vector: Vec3,
    limit: f32,
) -> Vec3 {
    if !limit.is_finite() || limit <= 0.0 {
        return Vec3::ZERO;
    }
    let length = vector.length();
    if !length.is_finite() {
        return Vec3::ZERO;
    }
    if length > limit {
        vector * (limit / length)
    } else {
        vector
    }
}

/// Computes the bounded torso lean angles `(alpha_about_forward,
/// beta_about_right)` in radians from a model-space head offset.
///
/// Rotating about the model forward axis (`+Z`) tilts the up-vector sideways:
/// a rotation of angle `alpha` about `+Z` moves a head at height `h`
/// approximately `-alpha * h` along `X`. Rotating about the model right axis
/// (`+X`) by `beta` moves it approximately `beta * h` along `Z`. Both angles
/// come from `atan2` against the measured lever arm and are clamped so the
/// magnitude never exceeds [`BodyTrackingPositionProfile::max_lean_radians`].
#[must_use]
pub fn lean_angles_model_space(
    head_offset_model: Vec3,
    lever_meters: f32,
    max_lean_radians: f32,
) -> (f32, f32) {
    let max = if max_lean_radians.is_finite() && max_lean_radians > 0.0 {
        max_lean_radians
    } else {
        return (0.0, 0.0);
    };
    let lever = if lever_meters.is_finite() && lever_meters > MIN_MEASURED_LEVER_METERS {
        lever_meters
    } else {
        FALLBACK_LEVER_METERS
    };
    let alpha = (-(head_offset_model.x).atan2(lever)).clamp(-max, max);
    let beta = head_offset_model.z.atan2(lever).clamp(-max, max);
    if !alpha.is_finite() || !beta.is_finite() {
        return (0.0, 0.0);
    }
    (alpha, beta)
}

pub(super) fn register(app: &mut App) {
    app.register_type::<BodyTrackingPositionInput>()
        .register_type::<BodyTrackingPositionProfile>()
        .add_systems(
            PostUpdate,
            apply_direct_body_position
                .after(AnimationSystems)
                .after(super::direct::apply_direct_body_tracking)
                .before(VrmSystemSets::GazeControl)
                .before(VrmSystemSets::Constraints)
                .run_if(any_with_component::<BodyTrackingPositionInput>),
        );
}

/// Applies the position-aware upper-body solve on top of direct body tracking.
///
/// The system runs immediately after
/// [`apply_direct_body_tracking`](super::direct::apply_direct_body_tracking)
/// and before gaze/constraints. It owns exactly two output channels:
///
/// 1. The avatar-root translation offset (root/body compensation), written
///    absolutely as `rest_translation + offset`, so re-evaluation never
///    accumulates.
/// 2. Bounded additive torso lean rotations distributed evenly over the
///    available spine/chest/upper-chest/neck bones (never the head),
///    premultiplied in world space on top of the rotations produced by the
///    direct-pose writer.
///
/// Head rotation remains exclusively owned by the direct-pose writer, hips
/// translation remains owned by the application breathing layer, and no
/// camera/projection property is touched.
///
/// Applications normally use [`crate::prelude::VrmPlugin`], which registers
/// this system in the correct order.
#[allow(clippy::too_many_arguments)]
pub fn apply_direct_body_position(
    vrms: Query<
        (
            Entity,
            &BodyTrackingPositionInput,
            Option<&BodyTrackingPositionProfile>,
            &HeadBoneEntity,
            Option<&NeckBoneEntity>,
            Option<&UpperChestBoneEntity>,
            Option<&ChestBoneEntity>,
            Option<&SpineBoneEntity>,
        ),
        With<BodyTracking>,
    >,
    mut root_transforms: Query<(&mut Transform, &mut GlobalTransform), With<Vrm>>,
    child_ofs: Query<&ChildOf>,
    mut transforms: Query<(&mut Transform, &mut GlobalTransform), Without<Vrm>>,
    rests: Query<&RestGlobalTransform>,
    time: Res<Time>,
    mut lean_bases: Local<HashMap<Entity, Quat>>,
    mut root_rest_data: Local<HashMap<Entity, (Quat, Vec3)>>,
    mut frame_stamp: Local<Option<Duration>>,
) {
    let now = time.elapsed();
    // Within one frame the system may be evaluated more than once (manual
    // re-runs in tests, retry schedules). Deltas must never accumulate: when
    // the frame stamp advances, the direct-pose writer ran in between and its
    // output becomes the new additive base; otherwise the stored base is
    // reused so a repeated evaluation reproduces the identical result.
    let frame_advanced = *frame_stamp != Some(now);
    *frame_stamp = Some(now);

    let default_profile = BodyTrackingPositionProfile::default();

    // State keyed by entity is dropped as soon as entities despawn so avatar
    // replacement cannot retain stale snapshots indefinitely.
    lean_bases.retain(|entity, _| transforms.contains(*entity));
    root_rest_data.retain(|entity, _| root_transforms.contains(*entity));

    for (root, input, profile, head, neck, upper_chest, chest, spine) in vrms.iter() {
        let profile = profile.unwrap_or(&default_profile);
        let sanitized = sanitize_input(input);

        // Capture the immutable root rest orientation and translation once.
        // Root transform data is read from the exclusive root query to keep
        // query access non-conflicting.
        let Ok((mut root_transform, mut root_global)) = root_transforms.get_mut(root) else {
            continue;
        };
        let (root_rest_rotation, root_rest_translation) =
            *root_rest_data.entry(root).or_insert_with(|| {
                let rotation = if root_transform.rotation.is_finite()
                    && root_transform.rotation.length_squared() > f32::EPSILON
                {
                    root_transform.rotation.normalize()
                } else {
                    Quat::IDENTITY
                };
                let translation = if root_transform.translation.is_finite() {
                    root_transform.translation
                } else {
                    Vec3::ZERO
                };
                (rotation, translation)
            });

        // --- Channel 1: root/body translation offset ---------------------
        let body_offset_model = match sanitized {
            Some((_, body)) => clamp_vector_length(
                semantic_offset_to_model(body, root_rest_rotation),
                profile.max_body_translation_meters,
            ),
            None => Vec3::ZERO,
        };

        let parent_global_before = child_ofs.get(root).ok().and_then(|child_of| {
            transforms
                .get(child_of.parent())
                .ok()
                .map(|(_, global)| *global)
        });
        root_transform.translation = root_rest_translation + body_offset_model;
        *root_global = match parent_global_before {
            Some(parent_global) => parent_global.mul_transform(*root_transform),
            None => GlobalTransform::from(*root_transform),
        };
        let root_global_after = *root_global;

        // --- Channel 2: bounded torso lean --------------------------------
        let head_offset_model = match sanitized {
            Some((head_residual, _)) => semantic_offset_to_model(head_residual, root_rest_rotation),
            None => Vec3::ZERO,
        };

        // Lever arm from immutable rest-global geometry: spine-to-head extent.
        let lever_meters = spine
            .and_then(|spine| rests.get(spine.0).ok())
            .and_then(|spine_gtf| {
                rests
                    .get(head.0)
                    .ok()
                    .map(|head_gtf| (head_gtf.translation() - spine_gtf.translation()).length())
            })
            .unwrap_or(FALLBACK_LEVER_METERS);

        let (alpha_total, beta_total) =
            lean_angles_model_space(head_offset_model, lever_meters, profile.max_lean_radians);

        // Distribute the lean across the available torso bones (never head).
        let mut chain: Vec<Entity> = Vec::with_capacity(4);
        if let Some(spine) = spine {
            chain.push(spine.0);
        }
        if let Some(chest) = chest {
            chain.push(chest.0);
        }
        if let Some(upper_chest) = upper_chest {
            chain.push(upper_chest.0);
        }
        if let Some(neck) = neck {
            chain.push(neck.0);
        }
        if chain.is_empty() {
            continue;
        }
        let share = 1.0 / chain.len() as f32;

        // World-space axes conjugated by the root rest rotation keep the lean
        // aligned with the model basis regardless of scene placement.
        let forward_world = (root_rest_rotation * Vec3::Z).normalize_or_zero();
        let right_world = (root_rest_rotation * Vec3::X).normalize_or_zero();

        let mut computed_globals = HashMap::with_capacity(chain.len() + 2);
        for entity in chain {
            let Some(parent) = child_ofs.get(entity).ok().map(ChildOf::parent) else {
                continue;
            };
            let Some(parent_global) = refresh_parent_global(
                root,
                parent,
                root_global_after,
                &mut transforms,
                &child_ofs,
                &mut computed_globals,
            ) else {
                continue;
            };
            let Ok((mut transform, mut global)) = transforms.get_mut(entity) else {
                continue;
            };

            let base = if frame_advanced {
                // Fresh evaluation: the direct-pose writer produced a new
                // rotation this frame; adopt it as the additive base.
                let entry = transform.rotation;
                if entry.is_finite() && entry.length_squared() > f32::EPSILON {
                    entry.normalize()
                } else {
                    Quat::IDENTITY
                }
            } else {
                // Same-frame re-evaluation: reuse the recorded base so the
                // output is bit-for-bit stable and deltas never accumulate.
                lean_bases.get(&entity).copied().unwrap_or(Quat::IDENTITY)
            };

            let alpha = alpha_total * share;
            let beta = beta_total * share;
            let lean_world = Quat::from_axis_angle(forward_world, alpha)
                * Quat::from_axis_angle(right_world, beta);
            let candidate = lean_world * base;
            let output = if candidate.is_finite()
                && candidate.length_squared().is_finite()
                && candidate.length_squared() > f32::EPSILON
            {
                candidate.normalize()
            } else {
                base
            };

            transform.rotation = output;
            *global = parent_global.mul_transform(*transform);
            lean_bases.insert(entity, base);
            computed_globals.insert(entity, *global);
        }

        // The head carries no lean itself, but downstream consumers inside
        // PostUpdate (direct look-at input reads the head world transform
        // before the VRM-spec propagation stage) need a fresh cached global.
        if let Some(parent) = child_ofs.get(head.0).ok().map(ChildOf::parent)
            && let Some(parent_global) = refresh_parent_global(
                root,
                parent,
                root_global_after,
                &mut transforms,
                &child_ofs,
                &mut computed_globals,
            )
            && let Ok((mut transform, mut global)) = transforms.get_mut(head.0)
        {
            *global = parent_global.mul_transform(*transform);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-5;
    const PI: f32 = std::f32::consts::PI;

    fn input(
        head: Vec3,
        body: Vec3,
    ) -> BodyTrackingPositionInput {
        BodyTrackingPositionInput {
            head_offset: head,
            body_offset: body,
            weight: 1.0,
            active: true,
        }
    }

    fn assert_close(
        actual: f32,
        expected: f32,
    ) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn semantic_mapping_flips_depth_and_preserves_lateral_axes() {
        let mapped = semantic_offset_to_model(Vec3::new(0.1, 0.2, 0.3), Quat::IDENTITY);
        assert_close(mapped.x, 0.1);
        assert_close(mapped.y, 0.2);
        assert_close(mapped.z, -0.3);
    }

    #[test]
    fn semantic_mapping_absorbs_root_rest_rotation() {
        // A legacy Y=180 basis root must flip lateral and depth axes without
        // any per-generation branch in the caller.
        let rotated = semantic_offset_to_model(Vec3::new(0.1, 0.2, 0.3), Quat::from_rotation_y(PI));
        assert!((rotated.x + 0.1).abs() < EPSILON);
        assert!((rotated.y - 0.2).abs() < EPSILON);
        assert!((rotated.z - 0.3).abs() < EPSILON);
    }

    #[test]
    fn lean_angles_move_head_toward_the_requested_model_offset() {
        let lever = 0.3;
        let (alpha, beta) = lean_angles_model_space(Vec3::new(0.06, 0.02, -0.03), lever, PI);
        // alpha about +Z moves the head by ~-alpha*h along X.
        let expected_alpha = -(0.06f32).atan2(lever);
        assert_close(alpha, expected_alpha);
        // beta about +X moves the head by ~beta*h along Z.
        let expected_beta = (-0.03f32).atan2(lever);
        assert_close(beta, expected_beta);

        // Simulated displacement at height `lever`.
        let dx = -alpha * lever;
        let dz = beta * lever;
        assert!((dx - 0.06).abs() < 1.0e-3);
        assert!((dz + 0.03).abs() < 1.0e-3);
    }

    #[test]
    fn lean_angles_are_clamped_to_the_profile_limit() {
        let max = 15.0_f32.to_radians();
        let (alpha, _) = lean_angles_model_space(Vec3::new(10.0, 0.0, 0.0), 0.3, max);
        assert_close(alpha, -max);
        let (_, beta) = lean_angles_model_space(Vec3::new(0.0, 0.0, -10.0), 0.3, max);
        assert_close(beta, -max);
    }

    #[test]
    fn degenerate_lever_arms_and_profiles_fail_closed() {
        let (alpha, beta) =
            lean_angles_model_space(Vec3::new(1.0, 0.0, 1.0), 0.0, 15.0_f32.to_radians());
        assert!(alpha.is_finite() && beta.is_finite());
        let (alpha, beta) = lean_angles_model_space(Vec3::ONE, 0.3, 0.0);
        assert_eq!((alpha, beta), (0.0, 0.0));
        let (alpha, beta) = lean_angles_model_space(Vec3::ONE, 0.3, f32::NAN);
        assert_eq!((alpha, beta), (0.0, 0.0));
    }

    #[test]
    fn clamped_vectors_never_exceed_the_limit_and_stay_finite() {
        let clamped = clamp_vector_length(Vec3::new(1.0, 0.0, 0.0), 0.25);
        assert_close(clamped.x, 0.25);
        assert_eq!(
            clamp_vector_length(Vec3::new(1.0, 2.0, 3.0), f32::NAN),
            Vec3::ZERO
        );
        assert_eq!(
            clamp_vector_length(Vec3::new(f32::NAN, 0.0, 0.0), 1.0),
            Vec3::ZERO
        );
    }

    #[test]
    fn inactive_or_zero_confidence_input_sanitizes_to_none() {
        let mut inactive = input(Vec3::ONE, Vec3::ONE);
        inactive.active = false;
        assert_eq!(sanitize_input(&inactive), None);

        let mut zero_weight = input(Vec3::ONE, Vec3::ONE);
        zero_weight.weight = 0.0;
        assert_eq!(sanitize_input(&zero_weight), None);
    }

    #[test]
    fn non_finite_offsets_are_zeroed_and_weights_scale_both_channels() {
        let mut noisy = input(
            Vec3::new(f32::NAN, 0.05, f32::INFINITY),
            Vec3::new(0.1, f32::NAN, -0.1),
        );
        noisy.weight = 0.5;
        let sanitized = sanitize_input(&noisy);
        let (head, body) = sanitized.expect("active input");
        assert_eq!(head.x, 0.0);
        assert_close(head.y, 0.025);
        assert_eq!(head.z, 0.0);
        assert_close(body.x, 0.05);
        assert_eq!(body.y, 0.0);
        assert_close(body.z, -0.05);
    }
}
