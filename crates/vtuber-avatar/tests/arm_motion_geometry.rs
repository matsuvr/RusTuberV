// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Arm motion rest geometry binding tests (Issue #175).
//!
//! Verifies that binding resolves the per-side motion
//! geometry exactly once from immutable rests, with mirror symmetry between
//! sides, explicit degeneracy status, and graceful degradation when optional
//! references are missing.

use bevy::asset::AssetApp;
use bevy::math::Vec3;
use bevy::prelude::*;
use bevy_vrm1::prelude::*;

use vtuber_avatar::arm_motion_geometry::ArmMotionGeometry;
use vtuber_avatar::bind::BindTriggered;
use vtuber_avatar::bind_humanoid_bones;
use vtuber_avatar::lifecycle::{
    AvatarLifecycle, AvatarLifecycleState, LoadAvatarRequest, LoadAvatarResult,
    ReplaceAvatarRequest, ReplaceAvatarResult, UnloadAvatarRequest, UnloadAvatarResult,
    apply_avatar_request_events,
};
use vtuber_avatar::unload::{ActiveControlFrame, despawn_unloading_avatar};

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<bevy_vrm1::prelude::VrmAsset>()
        .init_resource::<AvatarLifecycle>()
        .init_resource::<ActiveControlFrame>()
        .init_resource::<vtuber_avatar::GlobalBodyTrackingProfile>()
        .add_message::<LoadAvatarRequest>()
        .add_message::<LoadAvatarResult>()
        .add_message::<UnloadAvatarRequest>()
        .add_message::<UnloadAvatarResult>()
        .add_message::<ReplaceAvatarRequest>()
        .add_message::<ReplaceAvatarResult>()
        .add_systems(
            Update,
            (
                apply_avatar_request_events,
                despawn_unloading_avatar,
                bind_humanoid_bones,
            )
                .chain(),
        );
    app
}

/// Spawns a bone with distinct authored-local and rest-global transforms.
fn spawn_bone_at(app: &mut App, local: Vec3, global: Vec3, rotation: Quat) -> Entity {
    let local_transform = Transform::from_translation(local);
    app.world_mut()
        .spawn((
            local_transform,
            GlobalTransform::IDENTITY,
            RestTransform(local_transform),
            RestGlobalTransform(GlobalTransform::from(
                Transform::from_translation(global).with_rotation(rotation),
            )),
        ))
        .id()
}

fn load_and_bind(app: &mut App, root: Entity) {
    app.world_mut()
        .resource_mut::<Messages<LoadAvatarRequest>>()
        .write(LoadAvatarRequest { root });
    app.update();

    app.world_mut().entity_mut(root).insert(Initialized);
    app.world_mut().entity_mut(root).insert(BindTriggered);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .start_binding(root);
    app.update();
}

/// Builds a humanoid rig with hips, torso, and both arms at mirrored,
/// non-identity rest poses. Returns `(root, hand_global_positions)`.
fn build_full_rig(app: &mut App) -> (Entity, Vec3, Vec3) {
    let hips = spawn_bone_at(
        &mut *app,
        Vec3::ZERO,
        Vec3::new(0.0, 0.95, 0.0),
        Quat::IDENTITY,
    );
    let upper_chest = spawn_bone_at(
        &mut *app,
        Vec3::new(0.0, 0.12, 0.0),
        Vec3::new(0.0, 1.25, 0.0),
        Quat::IDENTITY,
    );
    let chest = spawn_bone_at(
        &mut *app,
        Vec3::new(0.0, 0.10, 0.0),
        Vec3::new(0.0, 1.15, 0.0),
        Quat::IDENTITY,
    );
    let head = spawn_bone_at(
        &mut *app,
        Vec3::new(0.0, 0.10, 0.0),
        Vec3::new(0.0, 1.55, 0.0),
        Quat::IDENTITY,
    );
    let spine = spawn_bone_at(
        &mut *app,
        Vec3::new(0.0, 0.08, 0.0),
        Vec3::new(0.0, 1.02, 0.0),
        Quat::IDENTITY,
    );

    // Left arm: shoulder-out, elbow-down; mirrored for the right side.
    let l_upper = spawn_bone_at(
        &mut *app,
        Vec3::new(0.12, 0.08, 0.0),
        Vec3::new(0.16, 1.42, 0.0),
        Quat::from_rotation_z(0.05),
    );
    let l_lower = spawn_bone_at(
        &mut *app,
        Vec3::new(0.25, -0.02, 0.0),
        Vec3::new(0.42, 1.38, 0.0),
        Quat::from_rotation_z(-0.3),
    );
    let l_hand = spawn_bone_at(
        &mut *app,
        Vec3::new(0.24, -0.01, 0.0),
        Vec3::new(0.66, 1.36, 0.01),
        Quat::from_rotation_x(0.15),
    );
    let r_upper = spawn_bone_at(
        &mut *app,
        Vec3::new(-0.12, 0.08, 0.0),
        Vec3::new(-0.16, 1.42, 0.0),
        Quat::from_rotation_z(-0.05),
    );
    let r_lower = spawn_bone_at(
        &mut *app,
        Vec3::new(-0.25, -0.02, 0.0),
        Vec3::new(-0.42, 1.38, 0.0),
        Quat::from_rotation_z(0.3),
    );
    let r_hand = spawn_bone_at(
        &mut *app,
        Vec3::new(-0.24, -0.01, 0.0),
        Vec3::new(-0.66, 1.36, 0.01),
        Quat::from_rotation_x(0.15),
    );

    let root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            HeadBoneEntity(head),
            HipsBoneEntity(hips),
            UpperChestBoneEntity(upper_chest),
            ChestBoneEntity(chest),
            SpineBoneEntity(spine),
        ))
        .id();
    app.world_mut().entity_mut(root).insert((
        LeftShoulderBoneEntity(l_upper),
        RightShoulderBoneEntity(r_upper),
        LeftUpperArmBoneEntity(l_upper),
        RightUpperArmBoneEntity(r_upper),
    ));
    app.world_mut().entity_mut(root).insert((
        LeftLowerArmBoneEntity(l_lower),
        RightLowerArmBoneEntity(r_lower),
        LeftHandBoneEntity(l_hand),
        RightHandBoneEntity(r_hand),
    ));

    for bone in [
        hips,
        upper_chest,
        chest,
        head,
        spine,
        l_upper,
        l_lower,
        l_hand,
        r_upper,
        r_lower,
        r_hand,
    ] {
        app.world_mut().entity_mut(bone).insert(ChildOf(root));
    }

    (
        root,
        Vec3::new(0.66, 1.36, 0.01),
        Vec3::new(-0.66, 1.36, 0.01),
    )
}

#[test]
fn binding_resolves_mirrored_motion_geometry_for_both_sides() {
    let mut app = test_app();
    let (root, left_hand, right_hand) = build_full_rig(&mut app);
    load_and_bind(&mut app, root);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();
    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Ready
    );

    let geometry = app
        .world()
        .get::<ArmMotionGeometry>(root)
        .expect("binding should insert arm motion geometry");
    let left = geometry.left.expect("left chain bound");
    let right = geometry.right.expect("right chain bound");

    // Hips rest origin is (0, 0.95, 0): anchors are hand minus hips.
    assert_relative_vec(
        left.hand_anchor.as_ref().unwrap().translation_from_hips,
        Vec3::new(left_hand.x, left_hand.y - 0.95, left_hand.z),
    );
    assert_relative_vec(
        right.hand_anchor.as_ref().unwrap().translation_from_hips,
        Vec3::new(right_hand.x, right_hand.y - 0.95, right_hand.z),
    );

    // Mirror symmetry across X.
    assert!(
        (left.hand_anchor.unwrap().translation_from_hips.x
            + right.hand_anchor.unwrap().translation_from_hips.x)
            .abs()
            < 1e-5
    );

    // Twist axes are usable unit vectors along each forearm.
    let lt = left.forearm_twist.as_ref().unwrap();
    let rt = right.forearm_twist.as_ref().unwrap();
    assert!(lt.usable() && rt.usable());
    assert!((lt.direction.length() - 1.0).abs() < 1e-5);
    assert!((rt.direction.length() - 1.0).abs() < 1e-5);

    // Elbow references sit on the elbow origins with unit normals.
    let le = left.elbow_reference.as_ref().unwrap();
    let re = right.elbow_reference.as_ref().unwrap();
    assert!((le.normal.length() - 1.0).abs() < 1e-5);
    assert!((re.normal.length() - 1.0).abs() < 1e-5);
    assert_relative_vec(le.point, Vec3::new(0.42, 1.38, 0.0));

    // Torso center prefers the upperChest reference.
    assert_relative_vec(left.torso_center.unwrap(), Vec3::new(0.0, 1.25, 0.0));
    assert_relative_vec(right.torso_center.unwrap(), Vec3::new(0.0, 1.25, 0.0));
}

fn assert_relative_vec(actual: Vec3, expected: Vec3) {
    assert!(
        actual.distance(expected) < 1e-5,
        "expected {expected:?}, got {actual:?}"
    );
}

#[test]
fn armless_models_bind_with_unavailable_geometry() {
    let mut app = test_app();

    let head = spawn_bone_at(
        &mut app,
        Vec3::ZERO,
        Vec3::new(0.0, 1.55, 0.0),
        Quat::IDENTITY,
    );
    let root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            HeadBoneEntity(head),
        ))
        .id();
    app.world_mut().entity_mut(head).insert(ChildOf(root));

    load_and_bind(&mut app, root);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();
    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Ready
    );

    let geometry = app
        .world()
        .get::<ArmMotionGeometry>(root)
        .expect("component should exist even without arms");
    assert!(geometry.left.is_none());
    assert!(geometry.right.is_none());
}
