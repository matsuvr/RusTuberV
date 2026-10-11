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
    ArmRestGeometry, ArmSide, AvatarBinding, AvatarGeneration, DynamicArmTargets, FingerReferences,
    FingerRestReferences, ResolvedArmPose, RestSpaceBonePose, apply_default_arm_pose,
    default_arm_target, solve_two_bone_arm,
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
        .spawn((
            transform,
            bevy_vrm1::prelude::RestTransform(transform),
            GlobalTransform::IDENTITY,
            ChildOf(parent),
        ))
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
        hand: None,
        shoulder: None,
        fingers: Default::default(),
    };
    app.world_mut().entity_mut(root).insert((
        AvatarBinding::head_only(root, root, generation),
        DynamicArmTargets {
            generation: Some(generation),
            source_seq: None,
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

#[test]
fn hand_driven_forearm_roll_distributes_skin_twist_without_changing_admitted_fk() {
    use vtuber_avatar::node_constraints::{
        NodeConstraintBindings, NodeConstraintDestination, NodeConstraintKind,
    };

    // The VRM1 twist reference uses hand-sourced X roll at weights 0.5/1,
    // and an elbow-sourced Y roll at weight 0.5. Exercise both mirrored arms,
    // arbitrary local rest axes, and a rigid intermediary before the hand.
    for side in [ArmSide::Left, ArmSide::Right] {
        for authored_axes in [false, true] {
            for (has_helpers, axial_helper, weight, on_joint_path) in [
                (false, true, 1.0, false),
                (true, true, 1.0, false),
                (true, false, 1.0, false),
                (true, true, 0.0, false),
                (true, true, 1.0, true),
            ] {
                let hand_roll = has_helpers && axial_helper && weight > 0.0 && !on_joint_path;
                let sign = if side == ArmSide::Left { 1.0 } else { -1.0 };
                let mut app = build_app();
                let arm = spawn_avatar(&mut app, Quat::IDENTITY, Quat::IDENTITY);
                let lower_rest = if authored_axes {
                    Quat::from_euler(EulerRot::XYZ, 0.3, -0.4, 0.2)
                } else {
                    Quat::IDENTITY
                };
                let hand_rest = Quat::from_rotation_x(0.25);
                let lower_position = Vec3::X * sign * 0.3;
                let wrist_offset = Vec3::X * sign * 0.25;
                let lower_tf =
                    Transform::from_translation(lower_position).with_rotation(lower_rest);
                app.world_mut().entity_mut(arm.lower).insert((
                    lower_tf,
                    bevy_vrm1::prelude::RestTransform(lower_tf),
                    ChildOf(arm.upper),
                ));
                let bridge = spawn_child(
                    &mut app,
                    arm.lower,
                    Transform::from_rotation(lower_rest.inverse()),
                );
                let hand_tf = Transform::from_translation(wrist_offset).with_rotation(hand_rest);
                app.world_mut().entity_mut(arm.hand).insert((
                    hand_tf,
                    bevy_vrm1::prelude::RestTransform(hand_tf),
                    ChildOf(bridge),
                ));
                let helper_rest = if authored_axes {
                    Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)
                } else {
                    Quat::IDENTITY
                };
                let roll_axis = if !axial_helper {
                    Vec3::Z
                } else if authored_axes {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                let mid = spawn_child(
                    &mut app,
                    bridge,
                    Transform::from_translation(wrist_offset * 0.5).with_rotation(helper_rest),
                );
                let end = spawn_child(
                    &mut app,
                    bridge,
                    Transform::from_translation(wrist_offset).with_rotation(helper_rest),
                );
                let elbow = spawn_child(
                    &mut app,
                    arm.upper,
                    Transform::from_translation(lower_position),
                );
                let rest = |position, rotation, local_rotation| RestSpaceBonePose {
                    position,
                    global_rotation: rotation,
                    local_rotation,
                };
                let chain = ArmChainBinding {
                    side,
                    shoulder: None,
                    upper_arm: arm.upper,
                    lower_arm: arm.lower,
                    hand: arm.hand,
                    fingers: Default::default(),
                    finger_rest: Default::default(),
                    capabilities: Default::default(),
                    rest: ArmRestGeometry {
                        shoulder: None,
                        upper_arm: rest(Vec3::ZERO, Quat::IDENTITY, Quat::IDENTITY),
                        elbow: rest(lower_position, lower_rest, lower_rest),
                        wrist: rest(lower_position + wrist_offset, hand_rest, hand_rest),
                        upper_arm_length: 0.3,
                        forearm_length: 0.25,
                        total_arm_length: 0.55,
                    },
                };
                app.world_mut()
                    .get_mut::<AvatarBinding>(arm.root)
                    .unwrap()
                    .left_arm = Some(chain);
                let roll = |destination, axis, weight| NodeConstraintDestination {
                    destination,
                    kind: NodeConstraintKind::Roll(axis),
                    weight,
                };
                let mut bindings =
                    std::collections::HashMap::from([(arm.lower, vec![roll(elbow, Vec3::Y, 0.5)])]);
                if has_helpers {
                    bindings.insert(
                        arm.hand,
                        if on_joint_path {
                            vec![roll(bridge, roll_axis, weight)]
                        } else {
                            vec![
                                roll(mid, roll_axis, 0.5 * weight),
                                roll(end, roll_axis, weight),
                            ]
                        },
                    );
                }
                app.world_mut()
                    .entity_mut(arm.root)
                    .insert(NodeConstraintBindings(bindings));

                // A flexed elbow and a full pronation/supination sweep. The
                // expected transforms are independent analytic rotations.
                let bend = Quat::from_rotation_y(sign * 1.2);
                let wrist_bend = Quat::from_rotation_z(0.2);
                for step in (0..=90).chain((0..=90).rev()) {
                    let angle = sign * (step as f32 - 45.0).to_radians() * 2.0;
                    let twist = Quat::from_rotation_x(angle);
                    let full_lower = bend * twist;
                    let mut pose = arm.pose;
                    pose.lower_arm_delta = lower_rest.inverse() * full_lower * lower_rest;
                    pose.hand = Some(vtuber_avatar::ResolvedBoneDelta {
                        entity: arm.hand,
                        delta: hand_rest.inverse() * wrist_bend * hand_rest,
                    });
                    app.world_mut()
                        .get_mut::<DynamicArmTargets>(arm.root)
                        .unwrap()
                        .left = Some(pose);
                    app.update();
                    let world = app.world();
                    let upper = world.get::<GlobalTransform>(arm.upper).unwrap();
                    let lower = world.get::<GlobalTransform>(arm.lower).unwrap();
                    let hand = world.get::<GlobalTransform>(arm.hand).unwrap();
                    let expected_elbow = upper.transform_point(lower_position);
                    let expected_wrist =
                        expected_elbow + upper.rotation() * full_lower * wrist_offset;
                    assert!(lower.translation().distance(expected_elbow) < 2.0e-6);
                    assert!(hand.translation().distance(expected_wrist) < 2.0e-6);
                    assert!(rotation_close(
                        hand.rotation(),
                        upper.rotation() * full_lower * wrist_bend * hand_rest
                    ));
                    assert!(
                        (hand.translation().distance(lower.translation()) - 0.25).abs() < 2.0e-6
                    );
                    if hand_roll {
                        assert!(
                            rotation_close(lower.rotation(), upper.rotation() * bend * lower_rest),
                            "forearm roll must enter the authored hand-driven helpers, not accumulate at the elbow"
                        );
                        assert!(rotation_close(
                            world.get::<Transform>(mid).unwrap().rotation,
                            Quat::from_rotation_x(angle * 0.5) * helper_rest
                        ));
                        assert!(rotation_close(
                            world.get::<Transform>(end).unwrap().rotation,
                            twist * helper_rest
                        ));
                        assert!(rotation_close(
                            world.get::<Transform>(elbow).unwrap().rotation,
                            Quat::from_rotation_y(sign * 0.6)
                        ));
                    } else {
                        assert!(rotation_close(
                            lower.rotation(),
                            upper.rotation() * full_lower * lower_rest
                        ));
                    }
                }
            }
        }
    }
}

#[test]
fn final_arm_pose_drives_roll_aim_and_copy_helpers_in_the_same_frame() {
    use vtuber_avatar::node_constraints::{
        NodeConstraintBindings, NodeConstraintDestination, NodeConstraintKind,
    };
    let mut app = build_app();
    let angle = 0.6;
    let chain = spawn_avatar(
        &mut app,
        Quat::from_rotation_y(0.4),
        Quat::from_rotation_y(angle),
    );
    let roll_rest = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let twist = spawn_child(&mut app, chain.upper, Transform::from_rotation(roll_rest));
    let copy_rest = Quat::from_rotation_x(-0.2);
    let copy = spawn_child(&mut app, chain.root, Transform::from_rotation(copy_rest));
    let sleeve = spawn_child(
        &mut app,
        chain.upper,
        Transform::from_rotation(Quat::from_rotation_z(0.3)),
    );
    let second_sleeve = spawn_child(&mut app, chain.upper, Transform::IDENTITY);
    app.world_mut()
        .entity_mut(chain.root)
        .insert(NodeConstraintBindings(std::collections::HashMap::from([
            (
                chain.upper,
                vec![
                    NodeConstraintDestination {
                        destination: twist,
                        kind: NodeConstraintKind::Roll(Vec3::X),
                        weight: 0.5,
                    },
                    NodeConstraintDestination {
                        destination: copy,
                        kind: NodeConstraintKind::Rotation,
                        weight: 1.0,
                    },
                    // This helper is an ancestor of the admitted elbow. Applying a
                    // constraint here would change bone positions after admission.
                    NodeConstraintDestination {
                        destination: chain.helper,
                        kind: NodeConstraintKind::Rotation,
                        weight: 1.0,
                    },
                ],
            ),
            (
                chain.lower,
                vec![sleeve, second_sleeve]
                    .into_iter()
                    .map(|destination| NodeConstraintDestination {
                        destination,
                        kind: NodeConstraintKind::Aim(Vec3::X),
                        weight: 1.0,
                    })
                    .collect(),
            ),
        ])));

    for _ in 0..3 {
        app.update();
        let world = app.world();
        assert!(rotation_close(
            world.get::<Transform>(twist).unwrap().rotation,
            roll_rest * Quat::from_rotation_x(angle * 0.5),
        ));
        assert!(rotation_close(
            world.get::<Transform>(copy).unwrap().rotation,
            copy_rest * Quat::from_rotation_y(angle),
        ));
        assert!(rotation_close(
            world.get::<Transform>(chain.helper).unwrap().rotation,
            Quat::IDENTITY,
        ));
        let elbow = world
            .get::<GlobalTransform>(chain.lower)
            .unwrap()
            .translation();
        for sleeve in [sleeve, second_sleeve] {
            let sleeve_global = world.get::<GlobalTransform>(sleeve).unwrap();
            let direction = (elbow - sleeve_global.translation()).normalize();
            assert!((sleeve_global.rotation() * Vec3::X).dot(direction) > 1.0 - EPSILON);
        }
    }
}

#[test]
fn admitted_pose_uses_authored_rest_without_accumulation() {
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
    assert!(rotation_close(after_animation_change, base * delta));
}

#[test]
fn animation_cannot_override_the_admitted_arm_pose() {
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
        Quat::IDENTITY
    ));

    let second_animation = Quat::from_rotation_y(0.03);
    app.world_mut()
        .get_mut::<Transform>(chain.upper)
        .unwrap()
        .rotation = second_animation;
    app.update();
    assert!(rotation_close(
        app.world().get::<Transform>(chain.upper).unwrap().rotation,
        Quat::IDENTITY
    ));
}

#[test]
fn arm_composition_preserves_spring_branch_pose_and_updates_its_attachment() {
    use bevy_vrm1::vrm::spring_bone::{SpringJoints, SpringRoot};

    let mut app = build_app();
    let chain = spawn_avatar(&mut app, Quat::IDENTITY, Quat::from_rotation_z(0.6));
    let ribbon = spawn_child(
        &mut app,
        chain.lower,
        Transform::from_translation(Vec3::new(0.15, -0.04, 0.0)),
    );
    let tip = spawn_child(
        &mut app,
        ribbon,
        Transform::from_translation(Vec3::new(0.0, -0.12, 0.0)),
    );
    app.world_mut().entity_mut(ribbon).insert(SpringRoot {
        joints: SpringJoints(vec![ribbon, tip]),
        ..default()
    });
    let ribbon_pose = Transform::from_translation(Vec3::new(0.15, -0.04, 0.0))
        .with_rotation(Quat::from_rotation_x(0.8));
    let tip_pose = Transform::from_translation(Vec3::new(0.0, -0.12, 0.0))
        .with_rotation(Quat::from_rotation_z(-0.3));
    *app.world_mut().get_mut::<Transform>(ribbon).unwrap() = ribbon_pose;
    *app.world_mut().get_mut::<Transform>(tip).unwrap() = tip_pose;

    // The rigid helper lies on the admitted elbow path; the ribbon is a
    // separate physics branch. Only the rigid link must return to authored rest.
    *app.world_mut().get_mut::<Transform>(chain.helper).unwrap() =
        Transform::from_translation(Vec3::splat(4.0)).with_rotation(Quat::from_rotation_y(0.9));
    for _ in 0..2 {
        app.update();
        let world = app.world();
        assert_eq!(*world.get::<Transform>(ribbon).unwrap(), ribbon_pose);
        assert_eq!(*world.get::<Transform>(tip).unwrap(), tip_pose);
        assert_eq!(
            *world.get::<Transform>(chain.helper).unwrap(),
            **world
                .get::<bevy_vrm1::prelude::RestTransform>(chain.helper)
                .unwrap(),
        );
        let expected_ribbon = world
            .get::<GlobalTransform>(chain.lower)
            .unwrap()
            .mul_transform(ribbon_pose);
        assert_eq!(
            *world.get::<GlobalTransform>(ribbon).unwrap(),
            expected_ribbon
        );
        assert_eq!(
            *world.get::<GlobalTransform>(tip).unwrap(),
            expected_ribbon.mul_transform(tip_pose),
        );
    }
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
    let solution =
        solve_two_bone_arm(ArmIkInput::from_geometry(chain.rest, target, chain.side)).unwrap();
    let pose = DynamicArmTargets {
        generation: Some(generation),
        source_seq: None,
        left: Some(ResolvedArmPose {
            upper_arm: upper,
            lower_arm: lower,
            upper_arm_delta: solution.upper_arm_delta,
            lower_arm_delta: solution.lower_arm_delta,
            hand: None,
            shoulder: None,
            fingers: Default::default(),
        }),
        right: None,
    };
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
    let pose = DynamicArmTargets {
        generation: Some(generation),
        source_seq: None,
        left: Some(ResolvedArmPose {
            upper_arm: upper,
            lower_arm: lower,
            upper_arm_delta: Quat::IDENTITY,
            lower_arm_delta: Quat::IDENTITY,
            hand: None,
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
        DynamicArmTargets {
            generation: Some(AvatarGeneration(4)),
            source_seq: None,
            left: Some(ResolvedArmPose {
                upper_arm: upper,
                lower_arm: upper,
                upper_arm_delta: Quat::from_rotation_z(0.8),
                lower_arm_delta: Quat::IDENTITY,
                hand: None,
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

    app.world_mut().entity_mut(root).insert(DynamicArmTargets {
        generation: Some(generation),
        source_seq: None,
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
