// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Tests for per-model arm tuning and deterministic pose transitions.

use bevy::prelude::*;
use vtuber_avatar::{
    ARM_POSE_PROFILE_OVERRIDE_VERSION, ActiveAvatar, ArmChainBinding, ArmChainCapabilities,
    ArmPoseBlendSide, ArmPoseBlendState, ArmPoseOverrideStore, ArmPoseProfile,
    ArmPoseProfileChange, ArmPoseProfileOverride, ArmPoseProfileOverrideError, ArmRestGeometry,
    ArmSide, AvatarAssetId, AvatarBinding, AvatarGeneration, DEFAULT_ARM_RETURN_SECONDS,
    DEFAULT_ARM_TRANSITION_SECONDS, DefaultArmPose, FingerReferences, FingerRestReferences,
    ResolvedArmPose, RestSpaceBonePose, apply_arm_pose_profile_changes,
};

const EPSILON: f32 = 1.0e-5;

fn pose(upper_arm_delta: Quat, lower_arm_delta: Quat) -> ResolvedArmPose {
    ResolvedArmPose {
        upper_arm: Entity::from_raw_u32(1).unwrap(),
        lower_arm: Entity::from_raw_u32(2).unwrap(),
        upper_arm_delta,
        lower_arm_delta,
        shoulder: None,
        fingers: Default::default(),
    }
}

fn rotation_close(actual: Quat, expected: Quat) -> bool {
    actual.dot(expected).abs() > 1.0 - EPSILON
}

fn simple_chain() -> ArmChainBinding {
    let rest = |position: Vec3| RestSpaceBonePose {
        position,
        global_rotation: Quat::IDENTITY,
        local_rotation: Quat::IDENTITY,
    };
    ArmChainBinding {
        side: ArmSide::Left,
        shoulder: None,
        upper_arm: Entity::from_raw_u32(11).unwrap(),
        lower_arm: Entity::from_raw_u32(12).unwrap(),
        hand: Entity::from_raw_u32(13).unwrap(),
        fingers: FingerReferences::default(),
        finger_rest: FingerRestReferences::default(),
        rest: ArmRestGeometry {
            shoulder: None,
            upper_arm: rest(Vec3::new(0.0, 1.0, 0.0)),
            elbow: rest(Vec3::new(0.5, 1.0, 0.0)),
            wrist: rest(Vec3::new(1.0, 1.0, 0.0)),
            upper_arm_length: 0.5,
            forearm_length: 0.5,
            total_arm_length: 1.0,
        },
        capabilities: ArmChainCapabilities::default(),
    }
}

#[test]
fn override_store_is_model_keyed_resettable_and_exportable() {
    let first = AvatarAssetId::new("sha256:first");
    let second = AvatarAssetId::new("sha256:second");
    let first_profile = ArmPoseProfile {
        arm_drop_radians: 0.55,
        ..Default::default()
    };
    let second_profile = ArmPoseProfile {
        arm_drop_radians: 0.85,
        ..Default::default()
    };

    let mut store = ArmPoseOverrideStore::default();
    store
        .set(
            first.0.clone(),
            ArmPoseProfileOverride::from_profile(first_profile),
        )
        .unwrap();
    store
        .set(
            second.0.clone(),
            ArmPoseProfileOverride::from_profile(second_profile),
        )
        .unwrap();

    assert_eq!(store.len(), 2);
    assert_eq!(store.profile_for(&first).unwrap(), first_profile);
    assert_eq!(store.profile_for(&second).unwrap(), second_profile);
    let exported: Vec<_> = store.entries().collect();
    assert_eq!(exported.len(), 2);
    assert!(exported.iter().any(|(id, _)| *id == first.0));

    // The resource remains usable across an avatar unload/reload boundary.
    assert!(store.reset(&first));
    assert!(store.profile_for(&first).is_none());
    assert_eq!(store.profile_for(&second).unwrap(), second_profile);
    assert!(!store.reset(&first));
}

#[test]
fn invalid_or_unknown_persisted_entries_are_rejected() {
    let mut store = ArmPoseOverrideStore::default();
    let valid = ArmPoseProfileOverride::from_profile(ArmPoseProfile::default());

    let mut unknown_version = valid;
    unknown_version.schema_version = ARM_POSE_PROFILE_OVERRIDE_VERSION + 1;
    assert_eq!(
        store.set("model", unknown_version),
        Err(vtuber_avatar::ArmPoseOverrideStoreError::InvalidProfile(
            ArmPoseProfileOverrideError::UnsupportedVersion {
                version: ARM_POSE_PROFILE_OVERRIDE_VERSION + 1
            }
        ))
    );

    let mut non_finite = valid;
    non_finite.finger_curl_radians = f32::NAN;
    assert!(matches!(
        store.set("model", non_finite),
        Err(vtuber_avatar::ArmPoseOverrideStoreError::InvalidProfile(
            ArmPoseProfileOverrideError::OutOfRangeOrNonFinite
        ))
    ));
    assert_eq!(
        store.set("", valid),
        Err(vtuber_avatar::ArmPoseOverrideStoreError::EmptyModelId)
    );

    let accepted = store.import_entries([
        ("valid".to_owned(), valid),
        ("bad-version".to_owned(), unknown_version),
        ("bad-number".to_owned(), non_finite),
    ]);
    assert_eq!(accepted, 1);
    assert_eq!(store.len(), 1);
    assert!(store.profile_for(&AvatarAssetId::new("valid")).is_some());
}

#[test]
fn model_profile_is_consumed_by_default_pose_resolution() {
    let generation = AvatarGeneration(9);
    let automatic = DefaultArmPose::from_chains(generation, Some(simple_chain()), None)
        .left
        .expect("simple chain should resolve");
    let tuned_profile = ArmPoseProfile {
        arm_drop_radians: 0.2,
        forward_hand_offset_ratio: -0.15,
        ..Default::default()
    };
    let tuned = DefaultArmPose::from_chains_with_profile(
        generation,
        Some(simple_chain()),
        None,
        tuned_profile,
    )
    .left
    .expect("tuned simple chain should resolve");
    assert!(!rotation_close(
        automatic.upper_arm_delta,
        tuned.upper_arm_delta
    ));
    assert!(tuned.upper_arm_delta.is_finite());
    assert!(tuned.lower_arm_delta.is_finite());
}

#[test]
fn active_profile_change_re_resolves_cached_geometry_through_the_compositor() {
    let generation = AvatarGeneration(9);
    let model_id = AvatarAssetId::new("sha256:active");
    let root = Entity::from_raw_u32(100).unwrap();
    let mut binding =
        AvatarBinding::head_only(root, Entity::from_raw_u32(101).unwrap(), generation);
    binding.left_arm = Some(simple_chain());
    let automatic = DefaultArmPose::from_chains(generation, binding.left_arm, None);
    let tuned_profile = ArmPoseProfile {
        arm_drop_radians: 0.2,
        forward_hand_offset_ratio: -0.15,
        ..Default::default()
    };

    let mut app = App::new();
    app.add_message::<ArmPoseProfileChange>()
        .init_resource::<ArmPoseOverrideStore>()
        .add_systems(Update, apply_arm_pose_profile_changes);
    app.world_mut().spawn((
        ActiveAvatar,
        model_id.clone(),
        binding,
        automatic,
        ArmPoseBlendState::from_default(&automatic),
    ));
    app.world_mut()
        .resource_mut::<ArmPoseOverrideStore>()
        .set(
            model_id.0.clone(),
            ArmPoseProfileOverride::from_profile(tuned_profile),
        )
        .unwrap();
    app.world_mut()
        .resource_mut::<Messages<ArmPoseProfileChange>>()
        .write(ArmPoseProfileChange {
            model_id: model_id.clone(),
            return_to_default: false,
        });

    app.update();

    let mut query = app
        .world_mut()
        .query::<(&DefaultArmPose, &ArmPoseBlendState)>();
    let (resolved, blend) = query.single(app.world()).unwrap();
    let tuned =
        DefaultArmPose::from_chains_with_profile(generation, binding.left_arm, None, tuned_profile);
    assert_ne!(resolved.left, automatic.left);
    assert_eq!(resolved.left, tuned.left);
    assert!(blend.current_left().is_some());
    assert!(blend.current_left().unwrap().upper_arm_delta.is_finite());

    app.world_mut()
        .resource_mut::<ArmPoseOverrideStore>()
        .reset(&model_id);
    app.world_mut()
        .resource_mut::<Messages<ArmPoseProfileChange>>()
        .write(ArmPoseProfileChange {
            model_id,
            return_to_default: true,
        });
    app.update();

    let mut query = app
        .world_mut()
        .query::<(&DefaultArmPose, &mut ArmPoseBlendState)>();
    let (resolved, mut blend) = query.single_mut(app.world_mut()).unwrap();
    assert_eq!(resolved.left, automatic.left);
    blend.advance(DEFAULT_ARM_RETURN_SECONDS);
    assert_eq!(blend.current_left(), automatic.left);
}

fn simulate_to(fps: u32, seconds: f32) -> ResolvedArmPose {
    let target = pose(Quat::from_rotation_y(1.2), Quat::from_rotation_x(-0.7));
    let default = DefaultArmPose {
        generation: AvatarGeneration(1),
        left: Some(target),
        right: None,
    };
    let mut state = ArmPoseBlendState::from_default(&default);
    let steps = ((fps as f32 * seconds).floor() as usize).max(1);
    let step = seconds / steps as f32;
    for _ in 0..steps {
        state.advance(step);
    }
    state.current_left().unwrap()
}

#[test]
fn transition_is_frame_rate_independent_at_30_60_and_120_fps() {
    let expected_half = pose(
        Quat::IDENTITY.slerp(Quat::from_rotation_y(1.2), 0.5),
        Quat::IDENTITY.slerp(Quat::from_rotation_x(-0.7), 0.5),
    );
    let at_30 = simulate_to(30, DEFAULT_ARM_TRANSITION_SECONDS / 2.0);
    let at_60 = simulate_to(60, DEFAULT_ARM_TRANSITION_SECONDS / 2.0);
    let at_120 = simulate_to(120, DEFAULT_ARM_TRANSITION_SECONDS / 2.0);
    for actual in [at_30, at_60, at_120] {
        assert!(rotation_close(
            actual.upper_arm_delta,
            expected_half.upper_arm_delta
        ));
        assert!(rotation_close(
            actual.lower_arm_delta,
            expected_half.lower_arm_delta
        ));
        assert!(actual.upper_arm_delta.is_finite());
        assert!(actual.lower_arm_delta.is_finite());
    }
    assert!(rotation_close(at_30.upper_arm_delta, at_60.upper_arm_delta));
    assert!(rotation_close(
        at_60.upper_arm_delta,
        at_120.upper_arm_delta
    ));
}

#[test]
fn transitions_use_shortest_arc_and_sanitize_invalid_time() {
    let from = pose(Quat::from_rotation_y(3.0), Quat::IDENTITY);
    let target = pose(Quat::from_rotation_y(-3.0), Quat::IDENTITY);
    let mut side = ArmPoseBlendSide::new(from, target, 1.0);
    side.advance(f32::NAN);
    side.advance(-1.0);
    assert!(rotation_close(
        side.current().upper_arm_delta,
        from.upper_arm_delta
    ));
    side.advance(0.5);
    let halfway = side.current().upper_arm_delta;
    assert!(halfway.is_finite());
    assert!(
        halfway
            .dot(Quat::from_rotation_y(std::f32::consts::PI))
            .abs()
            > 0.99
    );

    let mut zero_duration = ArmPoseBlendSide::new(from, target, f32::NAN);
    assert!(rotation_close(
        zero_duration.current().upper_arm_delta,
        target.upper_arm_delta
    ));
    zero_duration.advance(f32::INFINITY);
    assert!(rotation_close(
        zero_duration.current().upper_arm_delta,
        target.upper_arm_delta
    ));
}

#[test]
fn left_and_right_transitions_are_independent_and_return_is_slower() {
    let left_default = pose(Quat::from_rotation_y(0.8), Quat::IDENTITY);
    let right_default = pose(Quat::from_rotation_y(-0.6), Quat::IDENTITY);
    let mut state = ArmPoseBlendState::from_default(&DefaultArmPose {
        generation: AvatarGeneration(2),
        left: Some(left_default),
        right: Some(right_default),
    });
    state.advance(DEFAULT_ARM_TRANSITION_SECONDS);
    let new_left = pose(Quat::from_rotation_x(1.0), Quat::IDENTITY);
    state.return_left_to_default(new_left);
    state.advance(DEFAULT_ARM_RETURN_SECONDS / 2.0);

    let left = state.current_left().unwrap();
    let right = state.current_right().unwrap();
    assert!(!rotation_close(
        left.upper_arm_delta,
        new_left.upper_arm_delta
    ));
    assert!(rotation_close(
        right.upper_arm_delta,
        right_default.upper_arm_delta
    ));
    state.advance(DEFAULT_ARM_RETURN_SECONDS / 2.0);
    assert!(rotation_close(
        state.current_left().unwrap().upper_arm_delta,
        new_left.upper_arm_delta
    ));
}
