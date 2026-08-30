// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for humanoid bone binding.

use bevy::prelude::*;
use bevy_vrm1::prelude::*;

use vtuber_avatar::DefaultArmPose;
use vtuber_avatar::arm::ArmSide;
use vtuber_avatar::bind::BindTriggered;
use vtuber_avatar::binding::{AvatarBinding, bind_humanoid_bones};
use vtuber_avatar::lifecycle::{ActiveAvatar, AvatarLifecycle, AvatarLifecycleState};

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<AvatarLifecycle>()
        .add_systems(Update, bind_humanoid_bones);
    app
}

fn spawn_bone(app: &mut App) -> Entity {
    app.world_mut()
        .spawn((
            Transform::IDENTITY,
            GlobalTransform::IDENTITY,
            RestTransform(Transform::IDENTITY),
            RestGlobalTransform(GlobalTransform::IDENTITY),
        ))
        .id()
}

fn spawn_rest_bone(
    app: &mut App,
    position: Vec3,
    local_rest_rotation: Quat,
    global_rest_rotation: Quat,
) -> Entity {
    let rest = Transform::from_translation(position).with_rotation(local_rest_rotation);
    let rest_global = GlobalTransform::from(
        Transform::from_translation(position).with_rotation(global_rest_rotation),
    );
    app.world_mut()
        .spawn((
            Transform::from_translation(position + Vec3::new(0.0, 0.25, 0.0)),
            GlobalTransform::from_translation(position + Vec3::new(0.0, 0.25, 0.0)),
            RestTransform(rest),
            RestGlobalTransform(rest_global),
        ))
        .id()
}

fn enter_binding(app: &mut App, root: Entity) {
    let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
    lifecycle.request_load(root).unwrap();
    lifecycle.start_binding(root);
}

fn spawn_complete_arm_root(app: &mut App, x: f32, y: f32) -> (Entity, Entity, Entity, Entity) {
    let upper_arm = spawn_rest_bone(app, Vec3::new(x, y, 0.0), Quat::IDENTITY, Quat::IDENTITY);
    let lower_arm = spawn_rest_bone(
        app,
        Vec3::new(x.signum() * (x.abs() + 0.5), y - 0.3, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let hand = spawn_rest_bone(
        app,
        Vec3::new(x.signum() * (x.abs() + 0.8), y - 0.6, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let head = spawn_bone(app);
    let root = app
        .world_mut()
        .spawn((
            ActiveAvatar,
            BindTriggered,
            HeadBoneEntity(head),
            LeftUpperArmBoneEntity(upper_arm),
            LeftLowerArmBoneEntity(lower_arm),
            LeftHandBoneEntity(hand),
        ))
        .id();
    (root, upper_arm, lower_arm, hand)
}

#[test]
fn humanoid_binding_head_only_ready() {
    let mut app = test_app();
    let head = spawn_bone(&mut app);
    let root = app
        .world_mut()
        .spawn((
            ActiveAvatar,
            BindTriggered,
            HeadBoneEntity(head),
            Visibility::Hidden,
        ))
        .id();

    enter_binding(&mut app, root);
    app.update();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);
    assert_eq!(lifecycle.active_root(), Some(root));

    let binding = app
        .world()
        .get::<AvatarBinding>(root)
        .expect("AvatarBinding should be cached on the root");
    assert_eq!(binding.head, head);
    assert!(binding.neck.is_none());
    assert!(binding.upper_chest.is_none());
    assert!(binding.chest.is_none());
    assert!(binding.spine.is_none());
    assert!(binding.left_upper_arm.is_none());
    assert!(binding.right_upper_arm.is_none());
    assert!(binding.left_eye.is_none());
    assert!(binding.right_eye.is_none());
    assert!(app.world().get::<BodyTracking>(root).is_some());
    assert!(app.world().get::<BodyTrackingPoseInput>(root).is_some());
    assert!(app.world().get::<BodyTrackingProfile>(root).is_some());
    // Regression (ADR-019 wiring): the position-aware upper-body solve only
    // runs when the root carries `BodyTrackingPositionInput`. Binding must
    // insert it, otherwise the head motion never reaches the torso and hips.
    assert!(app.world().get::<BodyTrackingPositionInput>(root).is_some());
    assert!(
        app.world()
            .get::<BodyTrackingPositionProfile>(root)
            .is_some()
    );
    // The inserted rotation profile must propagate head motion into the
    // torso: the pitch distribution must give the upper body a visible share.
    let profile = app
        .world()
        .get::<BodyTrackingProfile>(root)
        .expect("tracking profile should be inserted");
    let torso_pitch_share = profile.pitch_weights.upper_chest
        + profile.pitch_weights.chest
        + profile.pitch_weights.spine;
    assert!(
        torso_pitch_share >= 0.15,
        "torso must visibly follow head pitch: {torso_pitch_share}"
    );

    assert_eq!(
        app.world().get::<Visibility>(root),
        Some(&Visibility::Inherited)
    );
    let capabilities = app
        .world()
        .resource::<AvatarLifecycle>()
        .capabilities()
        .expect("successful binding should publish capabilities");
    assert!(capabilities.bones.head);
    assert!(!capabilities.bones.neck);
}

#[test]
fn humanoid_binding_waits_for_rest_global_transform() {
    let mut app = test_app();
    let head = app
        .world_mut()
        .spawn((Transform::IDENTITY, RestTransform(Transform::IDENTITY)))
        .id();
    app.world_mut().entity_mut(head).remove::<GlobalTransform>();
    let root = app
        .world_mut()
        .spawn((ActiveAvatar, BindTriggered, HeadBoneEntity(head)))
        .id();

    enter_binding(&mut app, root);
    app.update();
    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Binding
    );
    assert!(app.world().get::<AvatarBinding>(root).is_none());

    app.world_mut()
        .entity_mut(head)
        .insert(RestGlobalTransform(GlobalTransform::IDENTITY));
    app.update();

    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Ready
    );
    assert!(app.world().get::<BodyTrackingPoseInput>(root).is_some());
}

#[test]
fn humanoid_binding_missing_head_fails() {
    let mut app = test_app();
    let root = app.world_mut().spawn((ActiveAvatar, BindTriggered)).id();

    enter_binding(&mut app, root);
    app.update();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Failed);
    assert!(lifecycle.active_root().is_none());
    assert!(!app.world().entity(root).contains::<ActiveAvatar>());
    assert!(
        app.world().get::<AvatarBinding>(root).is_none(),
        "no binding should be cached when the head is missing"
    );
}

#[test]
fn humanoid_binding_despawned_head_fails() {
    let mut app = test_app();
    let head = spawn_bone(&mut app);
    let root = app
        .world_mut()
        .spawn((ActiveAvatar, BindTriggered, HeadBoneEntity(head)))
        .id();

    enter_binding(&mut app, root);
    app.world_mut().entity_mut(head).despawn();
    app.update();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Failed);
    assert!(lifecycle.active_root().is_none());
}

#[test]
fn humanoid_binding_caches_normal_symmetric_arm_chains() {
    let mut app = test_app();
    let head = spawn_bone(&mut app);
    let left_upper_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(0.4, 1.4, 0.0),
        Quat::from_rotation_x(0.2),
        Quat::from_rotation_y(0.3),
    );
    let left_lower_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(0.9, 1.1, 0.0),
        Quat::from_rotation_x(-0.2),
        Quat::from_rotation_y(0.4),
    );
    let left_hand = spawn_rest_bone(
        &mut app,
        Vec3::new(1.2, 0.8, 0.0),
        Quat::from_rotation_z(0.1),
        Quat::from_rotation_y(0.5),
    );
    let right_upper_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(-0.4, 1.4, 0.0),
        Quat::from_rotation_x(-0.2),
        Quat::from_rotation_y(-0.3),
    );
    let right_lower_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(-0.9, 1.1, 0.0),
        Quat::from_rotation_x(0.2),
        Quat::from_rotation_y(-0.4),
    );
    let right_hand = spawn_rest_bone(
        &mut app,
        Vec3::new(-1.2, 0.8, 0.0),
        Quat::from_rotation_z(-0.1),
        Quat::from_rotation_y(-0.5),
    );
    let root = app
        .world_mut()
        .spawn((
            ActiveAvatar,
            BindTriggered,
            HeadBoneEntity(head),
            LeftUpperArmBoneEntity(left_upper_arm),
            LeftLowerArmBoneEntity(left_lower_arm),
            LeftHandBoneEntity(left_hand),
            RightUpperArmBoneEntity(right_upper_arm),
            RightLowerArmBoneEntity(right_lower_arm),
            RightHandBoneEntity(right_hand),
        ))
        .id();

    enter_binding(&mut app, root);
    app.update();

    let binding = app.world().get::<AvatarBinding>(root).unwrap();
    let left = binding.left_arm.expect("left symmetric chain should bind");
    let right = binding
        .right_arm
        .expect("right symmetric chain should bind");
    let default_pose = app
        .world()
        .get::<DefaultArmPose>(root)
        .expect("complete chains should resolve a typed default pose");
    assert!(default_pose.left.is_some());
    assert!(default_pose.right.is_some());
    assert!((left.rest.upper_arm_length - right.rest.upper_arm_length).abs() < 1.0e-6);
    assert!((left.rest.forearm_length - right.rest.forearm_length).abs() < 1.0e-6);
    assert!((left.rest.total_arm_length - right.rest.total_arm_length).abs() < 1.0e-6);
    assert_eq!(
        left.rest.upper_arm.position,
        -right.rest.upper_arm.position * Vec3::new(1.0, -1.0, 1.0)
    );
    assert_eq!(
        left.rest.elbow.position,
        -right.rest.elbow.position * Vec3::new(1.0, -1.0, 1.0)
    );
    assert_eq!(
        left.rest.wrist.position,
        -right.rest.wrist.position * Vec3::new(1.0, -1.0, 1.0)
    );
}

#[test]
fn humanoid_binding_replacement_caches_fresh_arm_geometry_and_generation() {
    let mut app = test_app();
    let (root_a, upper_a, lower_a, hand_a) = spawn_complete_arm_root(&mut app, 0.4, 1.4);
    enter_binding(&mut app, root_a);
    app.update();

    let binding_a = *app.world().get::<AvatarBinding>(root_a).unwrap();
    let generation_a = binding_a.generation;
    let left_a = binding_a.left_arm.expect("avatar A arm should bind");
    assert_eq!(left_a.upper_arm, upper_a);
    assert_eq!(left_a.lower_arm, lower_a);
    assert_eq!(left_a.hand, hand_a);

    let (root_b, upper_b, lower_b, hand_b) = spawn_complete_arm_root(&mut app, 0.7, 1.8);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .request_replace(root_b)
        .expect("ready avatar should accept replacement");
    app.world_mut().entity_mut(root_a).remove::<ActiveAvatar>();
    app.world_mut().entity_mut(root_a).despawn();
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_unload();
    app.world_mut().entity_mut(root_b).insert(ActiveAvatar);
    {
        let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
        lifecycle.start_binding(root_b);
    }
    app.update();

    let binding_b = app.world().get::<AvatarBinding>(root_b).unwrap();
    let left_b = binding_b.left_arm.expect("avatar B arm should bind");
    assert_ne!(binding_b.generation, generation_a);
    assert_ne!(left_b.upper_arm, left_a.upper_arm);
    assert_eq!(left_b.upper_arm, upper_b);
    assert_eq!(left_b.lower_arm, lower_b);
    assert_eq!(left_b.hand, hand_b);
    assert_ne!(
        left_b.rest.upper_arm.position,
        left_a.rest.upper_arm.position
    );
    assert!(app.world().get_entity(root_a).is_err());
}

#[test]
fn humanoid_binding_caches_asymmetric_rest_space_arm_geometry() {
    let mut app = test_app();
    let head = spawn_bone(&mut app);
    let left_shoulder = spawn_rest_bone(
        &mut app,
        Vec3::new(0.2, 1.5, 0.0),
        Quat::from_rotation_y(0.2),
        Quat::from_rotation_z(0.1),
    );
    let left_upper_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(0.4, 1.4, 0.0),
        Quat::from_rotation_x(0.3),
        Quat::from_rotation_y(0.4),
    );
    let left_lower_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(0.9, 1.1, 0.0),
        Quat::from_rotation_x(-0.25),
        Quat::from_rotation_y(-0.2),
    );
    let left_hand = spawn_rest_bone(
        &mut app,
        Vec3::new(1.2, 0.8, 0.0),
        Quat::from_rotation_z(0.15),
        Quat::from_rotation_x(0.5),
    );
    let left_index_proximal = spawn_rest_bone(
        &mut app,
        Vec3::new(1.25, 0.75, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );

    let right_upper_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(-0.4, 1.4, 0.0),
        Quat::from_rotation_x(-0.3),
        Quat::from_rotation_y(-0.4),
    );
    let right_lower_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(-0.8, 1.05, 0.0),
        Quat::from_rotation_x(0.25),
        Quat::from_rotation_y(0.2),
    );
    let right_hand = spawn_rest_bone(
        &mut app,
        Vec3::new(-1.0, 0.7, 0.0),
        Quat::from_rotation_z(-0.15),
        Quat::from_rotation_x(-0.5),
    );

    let root = app
        .world_mut()
        .spawn((
            ActiveAvatar,
            BindTriggered,
            HeadBoneEntity(head),
            LeftShoulderBoneEntity(left_shoulder),
            LeftUpperArmBoneEntity(left_upper_arm),
            LeftLowerArmBoneEntity(left_lower_arm),
            LeftHandBoneEntity(left_hand),
            LeftIndexProximalBoneEntity(left_index_proximal),
            RightUpperArmBoneEntity(right_upper_arm),
            RightLowerArmBoneEntity(right_lower_arm),
            RightHandBoneEntity(right_hand),
        ))
        .id();

    enter_binding(&mut app, root);
    app.update();

    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Ready
    );
    let binding = app.world().get::<AvatarBinding>(root).unwrap();
    let left = binding
        .left_arm
        .expect("complete left arm should be cached");
    let right = binding
        .right_arm
        .expect("complete right arm should be cached");

    assert_eq!(left.side, ArmSide::Left);
    assert_eq!(left.shoulder, Some(left_shoulder));
    assert!(left.capabilities.has_shoulder);
    assert!(left.capabilities.has_fingers);
    assert_eq!(left.fingers.index.proximal, Some(left_index_proximal));
    assert_eq!(
        left.finger_rest.index.proximal.unwrap().entity,
        left_index_proximal
    );
    assert_eq!(
        left.finger_rest.index.proximal.unwrap().rest.position,
        Vec3::new(1.25, 0.75, 0.0)
    );
    assert_eq!(left.rest.upper_arm.position, Vec3::new(0.4, 1.4, 0.0));
    assert_eq!(left.rest.elbow.position, Vec3::new(0.9, 1.1, 0.0));
    assert_eq!(left.rest.wrist.position, Vec3::new(1.2, 0.8, 0.0));
    assert!((left.rest.upper_arm_length - 0.5830952).abs() < 1.0e-5);
    assert!((left.rest.forearm_length - 0.42426407).abs() < 1.0e-5);
    assert!((left.rest.total_arm_length - 1.0073593).abs() < 1.0e-5);
    assert!(
        left.rest
            .upper_arm
            .global_rotation
            .dot(Quat::from_rotation_y(0.4))
            .abs()
            > 0.999_99
    );
    assert!(
        left.rest
            .upper_arm
            .local_rotation
            .dot(Quat::from_rotation_x(0.3))
            .abs()
            > 0.999_99
    );

    assert_eq!(right.side, ArmSide::Right);
    assert!(!right.capabilities.has_shoulder);
    assert!(!right.capabilities.has_fingers);
    assert!((right.rest.upper_arm_length - 0.5315073).abs() < 1.0e-5);
    assert!((right.rest.forearm_length - 0.4031129).abs() < 1.0e-5);
}

#[test]
fn humanoid_binding_missing_optional_shoulder_and_fingers_keeps_complete_chain() {
    let mut app = test_app();
    let head = spawn_bone(&mut app);
    let upper_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(0.4, 1.2, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let lower_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(0.9, 1.0, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let hand = spawn_rest_bone(
        &mut app,
        Vec3::new(1.2, 0.8, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let root = app
        .world_mut()
        .spawn((
            ActiveAvatar,
            BindTriggered,
            HeadBoneEntity(head),
            LeftUpperArmBoneEntity(upper_arm),
            LeftLowerArmBoneEntity(lower_arm),
            LeftHandBoneEntity(hand),
        ))
        .id();

    enter_binding(&mut app, root);
    app.update();

    let binding = app.world().get::<AvatarBinding>(root).unwrap();
    let left = binding
        .left_arm
        .expect("required arm chain should be present");
    assert!(left.shoulder.is_none());
    assert!(!left.capabilities.has_shoulder);
    assert!(!left.capabilities.has_fingers);
    assert_eq!(left.fingers, Default::default());
}

#[test]
fn humanoid_binding_degenerate_or_nonfinite_rest_geometry_is_unavailable_only() {
    let mut app = test_app();
    let head = spawn_bone(&mut app);
    let upper_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(0.4, 1.2, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let lower_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(0.4, 1.2, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let hand = spawn_rest_bone(
        &mut app,
        Vec3::new(1.2, 0.8, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let right_upper_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(f32::NAN, 1.2, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let right_lower_arm = spawn_rest_bone(
        &mut app,
        Vec3::new(-0.4, 1.0, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let right_hand = spawn_rest_bone(
        &mut app,
        Vec3::new(-0.7, 0.8, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    );
    let root = app
        .world_mut()
        .spawn((
            ActiveAvatar,
            BindTriggered,
            HeadBoneEntity(head),
            LeftUpperArmBoneEntity(upper_arm),
            LeftLowerArmBoneEntity(lower_arm),
            LeftHandBoneEntity(hand),
            RightUpperArmBoneEntity(right_upper_arm),
            RightLowerArmBoneEntity(right_lower_arm),
            RightHandBoneEntity(right_hand),
        ))
        .id();

    enter_binding(&mut app, root);
    app.update();

    let binding = app.world().get::<AvatarBinding>(root).unwrap();
    assert!(binding.left_arm.is_none());
    assert!(binding.right_arm.is_none());
    assert_eq!(binding.left_upper_arm, Some(upper_arm));
    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Ready
    );
}

#[test]
fn humanoid_binding_missing_rest_transform_fails() {
    let mut app = test_app();
    let head = app.world_mut().spawn(Transform::IDENTITY).id();
    let root = app
        .world_mut()
        .spawn((ActiveAvatar, BindTriggered, HeadBoneEntity(head)))
        .id();

    enter_binding(&mut app, root);
    app.update();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Failed);
}

#[test]
fn humanoid_binding_optional_bones_cached() {
    let mut app = test_app();
    let head = spawn_bone(&mut app);
    let neck = spawn_bone(&mut app);
    let upper_chest = spawn_bone(&mut app);
    let chest = spawn_bone(&mut app);
    let spine = spawn_bone(&mut app);
    let left_upper_arm = spawn_bone(&mut app);
    let right_upper_arm = spawn_bone(&mut app);
    let left_rest_rotation = Quat::from_rotation_x(0.13);
    let right_rest_rotation = Quat::from_rotation_x(-0.17);
    app.world_mut().entity_mut(left_upper_arm).insert((
        Transform::from_rotation(Quat::from_rotation_y(0.4)),
        RestTransform(Transform::from_rotation(left_rest_rotation)),
    ));
    app.world_mut().entity_mut(right_upper_arm).insert((
        Transform::from_rotation(Quat::from_rotation_y(-0.3)),
        RestTransform(Transform::from_rotation(right_rest_rotation)),
    ));
    let left_eye = spawn_bone(&mut app);
    let right_eye = spawn_bone(&mut app);

    let root = app
        .world_mut()
        .spawn((
            ActiveAvatar,
            BindTriggered,
            HeadBoneEntity(head),
            NeckBoneEntity(neck),
            UpperChestBoneEntity(upper_chest),
            ChestBoneEntity(chest),
            SpineBoneEntity(spine),
            LeftUpperArmBoneEntity(left_upper_arm),
            RightUpperArmBoneEntity(right_upper_arm),
            LeftEyeBoneEntity(left_eye),
            RightEyeBoneEntity(right_eye),
        ))
        .id();

    enter_binding(&mut app, root);
    app.update();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);

    let binding = app
        .world()
        .get::<AvatarBinding>(root)
        .expect("AvatarBinding should be cached");
    assert_eq!(binding.head, head);
    assert_eq!(binding.neck, Some(neck));
    assert_eq!(binding.upper_chest, Some(upper_chest));
    assert_eq!(binding.chest, Some(chest));
    assert_eq!(binding.spine, Some(spine));
    assert_eq!(binding.left_upper_arm, Some(left_upper_arm));
    assert_eq!(binding.right_upper_arm, Some(right_upper_arm));
    assert_eq!(binding.left_eye, Some(left_eye));
    assert_eq!(binding.right_eye, Some(right_eye));

    let left_rotation = app
        .world()
        .get::<Transform>(left_upper_arm)
        .unwrap()
        .rotation;
    let right_rotation = app
        .world()
        .get::<Transform>(right_upper_arm)
        .unwrap()
        .rotation;
    let left_rest = app
        .world()
        .get::<RestTransform>(left_upper_arm)
        .expect("upper arm rest transform remains model-authored");
    let right_rest = app
        .world()
        .get::<RestTransform>(right_upper_arm)
        .expect("upper arm rest transform remains model-authored");
    assert!(
        left_rotation.dot(Quat::from_rotation_y(0.4)).abs() > 0.999_999,
        "binding must not write a default pose into the animated transform: actual={left_rotation:?}"
    );
    assert!(
        right_rotation.dot(Quat::from_rotation_y(-0.3)).abs() > 0.999_999,
        "binding must not write a default pose into the animated transform: actual={right_rotation:?}"
    );
    assert!(binding.left_arm.is_none());
    assert!(binding.right_arm.is_none());
    assert_eq!(left_rest.0.rotation, left_rest_rotation);
    assert_eq!(right_rest.0.rotation, right_rest_rotation);
}

#[test]
fn humanoid_binding_no_repeated_lookup_after_ready() {
    let mut app = test_app();
    let head = spawn_bone(&mut app);
    let left_upper_arm = spawn_bone(&mut app);
    let right_upper_arm = spawn_bone(&mut app);
    let root = app
        .world_mut()
        .spawn((
            ActiveAvatar,
            BindTriggered,
            HeadBoneEntity(head),
            LeftUpperArmBoneEntity(left_upper_arm),
            RightUpperArmBoneEntity(right_upper_arm),
        ))
        .id();

    enter_binding(&mut app, root);
    app.update();
    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Ready
    );

    let before = app.world().get::<AvatarBinding>(root).copied();
    let replacement_rotation = Quat::from_rotation_y(0.2);
    app.world_mut()
        .get_mut::<Transform>(left_upper_arm)
        .expect("bound left upper arm remains available")
        .rotation = replacement_rotation;

    // A second update must not invalidate the ready state, run binding again,
    // or change the cached binding.
    app.update();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);
    assert_eq!(
        app.world().get::<AvatarBinding>(root),
        before.as_ref(),
        "AvatarBinding should not change after the avatar is ready"
    );
    assert_eq!(
        app.world()
            .get::<Transform>(left_upper_arm)
            .unwrap()
            .rotation,
        replacement_rotation,
        "binding must not rewrite a ready avatar's transform"
    );
}
