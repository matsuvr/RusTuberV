// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Runtime tests for model-adaptive default arm-pose composition.

use bevy::prelude::*;
use vtuber_avatar::{
    ActiveAvatar, ArmChainBinding, ArmChainCapabilities, ArmIkInput, ArmPoseProfile,
    ArmRestGeometry, ArmSide, AvatarBinding, AvatarGeneration, DefaultArmPose,
    FingerJointRestBinding, FingerJointRestReferences, FingerReferences, FingerRestReferences,
    ResolvedArmPose, RestSpaceBonePose, apply_default_arm_pose, default_arm_target,
    solve_two_bone_arm,
};

const EPSILON: f32 = 1.0e-5;

#[derive(Clone, Copy)]
struct ArmChain {
    root: Entity,
    upper: Entity,
    helper: Entity,
    lower: Entity,
    hand: Entity,
    pose: ResolvedArmPose,
}

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_systems(PostUpdate, apply_default_arm_pose);
    app
}

fn spawn_child(app: &mut App, parent: Entity, transform: Transform) -> Entity {
    app.world_mut()
        .spawn((transform, GlobalTransform::IDENTITY, ChildOf(parent)))
        .id()
}

fn spawn_avatar(app: &mut App, base_rotation: Quat, delta: Quat) -> ArmChain {
    let generation = AvatarGeneration(7);
    let root = app
        .world_mut()
        .spawn((
            ActiveAvatar,
            Transform::from_translation(Vec3::new(0.2, 0.4, -0.1)),
            GlobalTransform::IDENTITY,
        ))
        .id();
    let upper = spawn_child(
        app,
        root,
        Transform::from_translation(Vec3::new(0.4, 1.2, 0.0)).with_rotation(base_rotation),
    );
    let helper = spawn_child(
        app,
        upper,
        Transform::from_translation(Vec3::new(0.1, 0.05, 0.0)),
    );
    let lower = spawn_child(
        app,
        helper,
        Transform::from_translation(Vec3::new(0.7, 0.0, 0.0)),
    );
    let hand = spawn_child(
        app,
        lower,
        Transform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
    );
    let pose = ResolvedArmPose {
        upper_arm: upper,
        lower_arm: lower,
        upper_arm_delta: delta,
        lower_arm_delta: delta.inverse(),
        shoulder: None,
        fingers: Default::default(),
    };
    app.world_mut().entity_mut(root).insert((
        AvatarBinding::head_only(root, root, generation),
        DefaultArmPose {
            generation,
            left: Some(pose),
            right: None,
        },
    ));
    ArmChain {
        root,
        upper,
        helper,
        lower,
        hand,
        pose,
    }
}

fn rotation_close(actual: Quat, expected: Quat) -> bool {
    actual.dot(expected).abs() > 1.0 - EPSILON
}

fn optional_correction_chain() -> ArmChainBinding {
    let rest_pose =
        |position: Vec3, global_rotation: Quat, local_rotation: Quat| RestSpaceBonePose {
            position,
            global_rotation,
            local_rotation,
        };
    let shoulder = Entity::from_raw_u32(10).unwrap();
    let upper = Entity::from_raw_u32(11).unwrap();
    let lower = Entity::from_raw_u32(12).unwrap();
    let hand = Entity::from_raw_u32(13).unwrap();
    let finger_proximal = Entity::from_raw_u32(14).unwrap();
    let finger_intermediate = Entity::from_raw_u32(15).unwrap();
    ArmChainBinding {
        side: ArmSide::Left,
        shoulder: Some(shoulder),
        upper_arm: upper,
        lower_arm: lower,
        hand,
        fingers: FingerReferences::default(),
        finger_rest: FingerRestReferences {
            index: FingerJointRestReferences {
                proximal: Some(FingerJointRestBinding {
                    entity: finger_proximal,
                    rest: rest_pose(
                        Vec3::new(1.45, 1.3, 0.0),
                        Quat::from_rotation_y(0.3),
                        Quat::from_rotation_y(0.2),
                    ),
                }),
                intermediate: Some(FingerJointRestBinding {
                    entity: finger_intermediate,
                    rest: rest_pose(
                        Vec3::new(1.55, 1.3, 0.0),
                        Quat::from_rotation_z(-0.2),
                        Quat::from_rotation_z(-0.1),
                    ),
                }),
                ..default()
            },
            ..default()
        },
        rest: ArmRestGeometry {
            shoulder: Some(rest_pose(
                Vec3::new(0.1, 1.5, 0.0),
                Quat::from_rotation_y(0.4),
                Quat::from_rotation_y(0.3),
            )),
            upper_arm: rest_pose(Vec3::new(0.3, 1.4, 0.0), Quat::IDENTITY, Quat::IDENTITY),
            elbow: rest_pose(Vec3::new(0.9, 1.4, 0.0), Quat::IDENTITY, Quat::IDENTITY),
            wrist: rest_pose(Vec3::new(1.4, 1.4, 0.0), Quat::IDENTITY, Quat::IDENTITY),
            upper_arm_length: 0.6,
            forearm_length: 0.5,
            total_arm_length: 1.1,
        },
        capabilities: ArmChainCapabilities {
            has_shoulder: true,
            has_fingers: true,
        },
    }
}

#[test]
fn animation_base_is_composed_without_accumulation() {
    let mut app = build_app();
    let base = Quat::from_rotation_y(0.25);
    let delta = Quat::from_rotation_z(0.4);
    let chain = spawn_avatar(&mut app, base, delta);

    app.update();
    let first = app.world().get::<Transform>(chain.upper).unwrap().rotation;
    assert!(rotation_close(first, base * delta));

    app.update();
    let second = app.world().get::<Transform>(chain.upper).unwrap().rotation;
    assert!(
        rotation_close(second, first),
        "default arm pose accumulated"
    );

    let new_base = Quat::from_rotation_x(-0.3);
    app.world_mut()
        .get_mut::<Transform>(chain.upper)
        .unwrap()
        .rotation = new_base;
    app.update();
    let after_animation_change = app.world().get::<Transform>(chain.upper).unwrap().rotation;
    assert!(rotation_close(after_animation_change, new_base * delta));
}

#[test]
fn successive_small_animation_updates_are_not_overwritten() {
    let mut app = build_app();
    let chain = spawn_avatar(&mut app, Quat::IDENTITY, Quat::IDENTITY);
    app.update();

    let first_animation = Quat::from_rotation_y(0.02);
    app.world_mut()
        .get_mut::<Transform>(chain.upper)
        .unwrap()
        .rotation = first_animation;
    app.update();
    assert!(rotation_close(
        app.world().get::<Transform>(chain.upper).unwrap().rotation,
        first_animation
    ));

    let second_animation = Quat::from_rotation_y(0.03);
    app.world_mut()
        .get_mut::<Transform>(chain.upper)
        .unwrap()
        .rotation = second_animation;
    app.update();
    assert!(rotation_close(
        app.world().get::<Transform>(chain.upper).unwrap().rotation,
        second_animation
    ));
}

#[test]
fn shoulder_follow_is_weak_clamped_and_finger_rest_axes_are_converted() {
    let chain = optional_correction_chain();
    let pose = DefaultArmPose::from_chains(AvatarGeneration(1), Some(chain), None);
    let resolved = pose.left.expect("complete chain should resolve");
    let shoulder = resolved.shoulder.expect("optional shoulder should resolve");
    let (_, shoulder_angle) = shoulder.delta.to_axis_angle();
    assert!(shoulder_angle > 0.0);
    assert!(shoulder_angle <= 5.0_f32.to_radians() + EPSILON);
    assert!(resolved.fingers.index.proximal.is_some());
    assert!(resolved.fingers.index.intermediate.is_some());
    assert!(resolved.fingers.thumb.proximal.is_none());
    let (finger_axis, finger_angle) = resolved
        .fingers
        .index
        .proximal
        .unwrap()
        .delta
        .to_axis_angle();
    assert!((finger_angle - 10.0_f32.to_radians()).abs() < EPSILON);
    assert!(finger_axis.dot(Vec3::X).abs() < 0.9);
    assert!(
        resolved
            .fingers
            .index
            .proximal
            .unwrap()
            .delta
            .dot(Quat::IDENTITY)
            .abs()
            < 0.99999
    );
}

#[test]
fn shoulder_follow_propagates_weakly_downstream_to_elbow_and_wrist() {
    let with_shoulder = optional_correction_chain();
    let mut without_shoulder = with_shoulder;
    without_shoulder.shoulder = None;
    without_shoulder.rest.shoulder = None;
    without_shoulder.capabilities.has_shoulder = false;
    let with = DefaultArmPose::from_chains(AvatarGeneration(1), Some(with_shoulder), None)
        .left
        .expect("complete chain should resolve");
    let without = DefaultArmPose::from_chains(AvatarGeneration(1), Some(without_shoulder), None)
        .left
        .expect("complete chain should resolve");
    let shoulder_angle = with
        .shoulder
        .expect("shoulder should resolve")
        .delta
        .angle_between(Quat::IDENTITY);
    assert!(shoulder_angle > 0.0);
    let upper_change = with
        .upper_arm_delta
        .angle_between(without.upper_arm_delta);
    let lower_change = with
        .lower_arm_delta
        .angle_between(without.lower_arm_delta);
    assert!(upper_change > 1.0e-6, "upper arm inherits part of the shoulder");
    assert!(lower_change > 1.0e-6, "forearm inherits part of the shoulder");
    assert!(
        upper_change < shoulder_angle,
        "downstream share stays smaller than the shoulder change"
    );
    assert!(
        lower_change < upper_change,
        "motion decays toward the terminal bones"
    );
}

#[test]
fn actual_child_of_path_propagates_intermediate_globals() {
    let mut app = build_app();
    let chain = spawn_avatar(
        &mut app,
        Quat::from_rotation_y(0.2),
        Quat::from_rotation_z(0.35),
    );
    app.update();

    let root_global = *app.world().get::<GlobalTransform>(chain.root).unwrap();
    let upper_transform = *app.world().get::<Transform>(chain.upper).unwrap();
    let expected_upper = root_global.mul_transform(upper_transform);
    let actual_upper = *app.world().get::<GlobalTransform>(chain.upper).unwrap();
    assert_eq!(actual_upper, expected_upper);

    let helper_transform = *app.world().get::<Transform>(chain.helper).unwrap();
    let expected_helper = expected_upper.mul_transform(helper_transform);
    let actual_helper = *app.world().get::<GlobalTransform>(chain.helper).unwrap();
    assert_eq!(actual_helper, expected_helper);

    let lower_transform = *app.world().get::<Transform>(chain.lower).unwrap();
    let expected_lower = expected_helper.mul_transform(lower_transform);
    let actual_lower = *app.world().get::<GlobalTransform>(chain.lower).unwrap();
    assert_eq!(actual_lower, expected_lower);

    let hand_transform = *app.world().get::<Transform>(chain.hand).unwrap();
    let expected_hand = expected_lower.mul_transform(hand_transform);
    let actual_hand = *app.world().get::<GlobalTransform>(chain.hand).unwrap();
    assert_eq!(actual_hand, expected_hand);
}

#[test]
fn solver_pose_composes_to_target_wrist_through_non_identity_rest_chain() {
    let mut app = build_app();
    let generation = AvatarGeneration(12);
    let root = app
        .world_mut()
        .spawn((ActiveAvatar, Transform::IDENTITY, GlobalTransform::IDENTITY))
        .id();

    let upper_rest_rotation = Quat::from_rotation_y(0.3);
    let lower_rest_rotation = Quat::from_rotation_x(-0.25);
    let upper_position = Vec3::new(0.3, 1.4, 0.0);
    let elbow_position = upper_position + Vec3::new(0.75, 0.0, 0.0);
    let wrist_position = elbow_position + Vec3::new(0.55, 0.0, 0.0);
    let lower_global_rotation = upper_rest_rotation * lower_rest_rotation;

    let upper = spawn_child(
        &mut app,
        root,
        Transform::from_translation(upper_position).with_rotation(upper_rest_rotation),
    );
    let helper = spawn_child(
        &mut app,
        upper,
        Transform::from_translation(
            upper_rest_rotation.inverse() * (elbow_position - upper_position),
        ),
    );
    let lower = spawn_child(
        &mut app,
        helper,
        Transform::from_rotation(lower_rest_rotation),
    );
    let hand = spawn_child(
        &mut app,
        lower,
        Transform::from_translation(
            lower_global_rotation.inverse() * (wrist_position - elbow_position),
        ),
    );

    let rest_pose =
        |position: Vec3, global_rotation: Quat, local_rotation: Quat| RestSpaceBonePose {
            position,
            global_rotation,
            local_rotation,
        };
    let chain = ArmChainBinding {
        side: ArmSide::Left,
        shoulder: None,
        upper_arm: upper,
        lower_arm: lower,
        hand,
        fingers: FingerReferences::default(),
        rest: ArmRestGeometry {
            shoulder: None,
            upper_arm: rest_pose(upper_position, upper_rest_rotation, upper_rest_rotation),
            elbow: rest_pose(elbow_position, lower_global_rotation, lower_rest_rotation),
            wrist: rest_pose(wrist_position, lower_global_rotation, Quat::IDENTITY),
            upper_arm_length: 0.75,
            forearm_length: 0.55,
            total_arm_length: 1.3,
        },
        capabilities: ArmChainCapabilities::default(),
        finger_rest: FingerRestReferences::default(),
    };
    let target = default_arm_target(&chain, ArmPoseProfile::default()).unwrap();
    let solution = solve_two_bone_arm(ArmIkInput::from_geometry(chain.rest, target)).unwrap();
    let pose = DefaultArmPose::from_chains(generation, Some(chain), None);
    assert!(pose.left.is_some());
    app.world_mut()
        .entity_mut(root)
        .insert((AvatarBinding::head_only(root, root, generation), pose));

    app.update();

    let actual_elbow = app
        .world()
        .get::<GlobalTransform>(lower)
        .unwrap()
        .translation();
    let actual_wrist = app
        .world()
        .get::<GlobalTransform>(hand)
        .unwrap()
        .translation();
    assert!(actual_elbow.distance(solution.elbow) < 1.0e-4);
    assert!(actual_wrist.distance(solution.wrist) < 1.0e-4);
    assert!(rotation_close(
        app.world()
            .get::<GlobalTransform>(upper)
            .unwrap()
            .rotation(),
        solution.upper_arm_global_rotation,
    ));
    assert!(rotation_close(
        app.world()
            .get::<GlobalTransform>(lower)
            .unwrap()
            .rotation(),
        solution.lower_arm_global_rotation,
    ));
    assert!(rotation_close(
        app.world().get::<GlobalTransform>(hand).unwrap().rotation(),
        solution.lower_arm_global_rotation,
    ));
}

#[test]
fn optional_shoulder_and_finger_corrections_compose_without_accumulation() {
    let mut app = build_app();
    let generation = AvatarGeneration(21);
    let root = app
        .world_mut()
        .spawn((ActiveAvatar, Transform::IDENTITY, GlobalTransform::IDENTITY))
        .id();
    let shoulder = spawn_child(&mut app, root, Transform::IDENTITY);
    let upper = spawn_child(&mut app, shoulder, Transform::IDENTITY);
    let lower = spawn_child(&mut app, upper, Transform::IDENTITY);
    let hand = spawn_child(&mut app, lower, Transform::IDENTITY);
    let finger = spawn_child(&mut app, hand, Transform::IDENTITY);
    let shoulder_delta = Quat::from_rotation_y(0.04);
    let finger_delta = Quat::from_rotation_x(0.1);
    let pose = DefaultArmPose {
        generation,
        left: Some(ResolvedArmPose {
            upper_arm: upper,
            lower_arm: lower,
            upper_arm_delta: Quat::IDENTITY,
            lower_arm_delta: Quat::IDENTITY,
            shoulder: Some(vtuber_avatar::ResolvedBoneDelta {
                entity: shoulder,
                delta: shoulder_delta,
            }),
            fingers: vtuber_avatar::ResolvedFingerPose {
                index: vtuber_avatar::ResolvedFingerJointPose {
                    proximal: Some(vtuber_avatar::ResolvedBoneDelta {
                        entity: finger,
                        delta: finger_delta,
                    }),
                    ..default()
                },
                ..default()
            },
        }),
        right: None,
    };
    app.world_mut()
        .entity_mut(root)
        .insert((AvatarBinding::head_only(root, root, generation), pose));

    app.update();
    let first_shoulder = app.world().get::<Transform>(shoulder).unwrap().rotation;
    let first_finger = app.world().get::<Transform>(finger).unwrap().rotation;
    assert!(rotation_close(first_shoulder, shoulder_delta));
    assert!(rotation_close(first_finger, finger_delta));
    app.update();
    assert!(rotation_close(
        app.world().get::<Transform>(shoulder).unwrap().rotation,
        first_shoulder
    ));
    assert!(rotation_close(
        app.world().get::<Transform>(finger).unwrap().rotation,
        first_finger
    ));
}

#[test]
fn generation_mismatch_and_missing_pose_are_safe_no_ops() {
    let mut app = build_app();
    let root = app.world_mut().spawn(ActiveAvatar).id();
    let upper = app
        .world_mut()
        .spawn((Transform::IDENTITY, GlobalTransform::IDENTITY))
        .id();
    let generation = AvatarGeneration(3);
    app.world_mut().entity_mut(root).insert((
        AvatarBinding::head_only(root, root, generation),
        DefaultArmPose {
            generation: AvatarGeneration(4),
            left: Some(ResolvedArmPose {
                upper_arm: upper,
                lower_arm: upper,
                upper_arm_delta: Quat::from_rotation_z(0.8),
                lower_arm_delta: Quat::IDENTITY,
                shoulder: None,
                fingers: Default::default(),
            }),
            right: None,
        },
    ));
    app.update();
    assert!(rotation_close(
        app.world().get::<Transform>(upper).unwrap().rotation,
        Quat::IDENTITY
    ));

    app.world_mut().entity_mut(root).insert(DefaultArmPose {
        generation,
        left: None,
        right: None,
    });
    app.update();
    assert!(rotation_close(
        app.world().get::<Transform>(upper).unwrap().rotation,
        Quat::IDENTITY
    ));
}

#[test]
fn replacement_starts_with_fresh_compositor_state() {
    let mut app = build_app();
    let first = spawn_avatar(
        &mut app,
        Quat::from_rotation_y(0.1),
        Quat::from_rotation_z(0.2),
    );
    app.update();
    app.world_mut().entity_mut(first.hand).despawn();
    app.world_mut().entity_mut(first.lower).despawn();
    app.world_mut().entity_mut(first.helper).despawn();
    app.world_mut().entity_mut(first.upper).despawn();
    app.world_mut().entity_mut(first.root).despawn();
    app.update();

    let second = spawn_avatar(
        &mut app,
        Quat::from_rotation_x(-0.45),
        Quat::from_rotation_z(-0.3),
    );
    app.update();
    let actual = app.world().get::<Transform>(second.upper).unwrap().rotation;
    assert!(rotation_close(
        actual,
        Quat::from_rotation_x(-0.45) * second.pose.upper_arm_delta
    ));
}
