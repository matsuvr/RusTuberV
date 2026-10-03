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
        hand: None,
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

#[test]
fn camera_loss_returns_the_composed_skeleton_while_the_torso_turns() {
    use bevy_vrm1::prelude::{RestGlobalTransform, RestTransform};
    use vtuber_avatar::{
        ActiveControlFrame, ArmMotionGeometry, ArmPoseSourceKind, ArmSourceSelection,
        AvatarAssetId, AvatarLifecycle, AvatarMotionMirror, DynamicArmTargets, TrackedArmControl,
        build_arm_motion_rest_geometry, update_dynamic_arm_targets, update_tracked_arm_targets,
    };
    use vtuber_core::arm_tracking::{
        ArmBlendWeight, ArmBlendWeights, ArmControlFrame, ArmTrackingTarget, ArmTrackingTargets,
        HandFingerPose,
    };
    use vtuber_core::{FrameSeq, MonoTimeNs};
    use vtuber_tracking::loss_blend::{LossBlend, LossBlendProfile};

    for side in [ArmSide::Left, ArmSide::Right] {
        let sign = if side == ArmSide::Left { 1.0 } else { -1.0 };
        let mut app = build_app();
        app.init_resource::<ActiveControlFrame>().add_systems(
            PostUpdate,
            (update_dynamic_arm_targets, update_tracked_arm_targets)
                .chain()
                .before(apply_default_arm_pose),
        );
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        ));
        let mut mirror = AvatarMotionMirror::default();
        mirror.toggle();
        app.insert_resource(mirror);
        app.insert_resource(ArmSourceSelection {
            mode: ArmPoseSourceKind::TrackedPose,
            ..Default::default()
        });
        let bone = |position, rotation| RestSpaceBonePose {
            position,
            global_rotation: rotation,
            local_rotation: rotation,
        };
        let hips_rest = bone(Vec3::new(0.0, 0.95, 0.0), Quat::IDENTITY);
        let root = app
            .world_mut()
            .spawn((
                ActiveAvatar,
                Transform::from_translation(hips_rest.position),
                GlobalTransform::from_translation(hips_rest.position),
            ))
            .id();
        let spawn =
            |app: &mut App, parent, parent_rest: RestSpaceBonePose, mut rest: RestSpaceBonePose| {
                rest.local_rotation = parent_rest.global_rotation.inverse() * rest.global_rotation;
                let local = Transform::from_translation(
                    parent_rest.global_rotation.inverse() * (rest.position - parent_rest.position),
                )
                .with_rotation(rest.local_rotation);
                let global = GlobalTransform::from(
                    Transform::from_translation(rest.position).with_rotation(rest.global_rotation),
                );
                let entity = app
                    .world_mut()
                    .spawn((
                        local,
                        global,
                        RestTransform(local),
                        RestGlobalTransform(global),
                        ChildOf(parent),
                    ))
                    .id();
                (entity, rest)
            };
        let (chest, chest_rest) = spawn(
            &mut app,
            root,
            hips_rest,
            bone(Vec3::new(0.0, 1.20, 0.0), Quat::from_rotation_y(0.2)),
        );
        let (shoulder, shoulder_rest) = spawn(
            &mut app,
            chest,
            chest_rest,
            bone(
                Vec3::new(sign * 0.05, 1.35, 0.0),
                Quat::from_rotation_z(0.15),
            ),
        );
        let (upper, upper_rest) = spawn(
            &mut app,
            shoulder,
            shoulder_rest,
            bone(
                Vec3::new(sign * 0.16, 1.35, 0.0),
                Quat::from_rotation_y(0.4),
            ),
        );
        let (lower, lower_rest) = spawn(
            &mut app,
            upper,
            upper_rest,
            bone(
                Vec3::new(sign * 0.41, 1.35, 0.0),
                Quat::from_rotation_x(-0.3),
            ),
        );
        let (hand, hand_rest) = spawn(
            &mut app,
            lower,
            lower_rest,
            bone(
                Vec3::new(sign * 0.65, 1.35, 0.0),
                Quat::from_rotation_z(-0.2),
            ),
        );
        let mut fingers = FingerRestReferences::default();
        let mut finger_entities = Vec::new();
        for (joints, z) in [(&mut fingers.index, 0.015), (&mut fingers.little, -0.015)] {
            let mut parent = hand;
            let mut parent_rest = hand_rest;
            for (slot, x) in [
                (&mut joints.proximal, 0.02),
                (&mut joints.intermediate, 0.04),
                (&mut joints.distal, 0.055),
            ] {
                let (entity, rest) = spawn(
                    &mut app,
                    parent,
                    parent_rest,
                    bone(
                        hand_rest.position + Vec3::new(sign * x, 0.0, z),
                        hand_rest.global_rotation,
                    ),
                );
                *slot = Some(FingerJointRestBinding { entity, rest });
                finger_entities.push(entity);
                parent = entity;
                parent_rest = rest;
            }
        }
        let chain = ArmChainBinding {
            side,
            shoulder: Some(shoulder),
            upper_arm: upper,
            lower_arm: lower,
            hand,
            fingers: FingerReferences::default(),
            finger_rest: fingers,
            rest: ArmRestGeometry {
                shoulder: Some(shoulder_rest),
                upper_arm: upper_rest,
                elbow: lower_rest,
                wrist: hand_rest,
                upper_arm_length: 0.25,
                forearm_length: 0.24,
                total_arm_length: 0.49,
            },
            capabilities: ArmChainCapabilities::default(),
        };
        let geometry = build_arm_motion_rest_geometry(
            side,
            &chain.rest,
            Some(hips_rest.position),
            Some(Quat::IDENTITY),
            Some(chest_rest.position),
        );
        let (left, right) = if side == ArmSide::Left {
            (Some(chain), None)
        } else {
            (None, Some(chain))
        };
        let mut lifecycle = AvatarLifecycle::default();
        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
        let generation = lifecycle.current_generation();
        app.insert_resource(lifecycle);
        let mut binding = AvatarBinding::head_only(root, chest, generation);
        binding.chest = Some(chest);
        binding.left_arm = left;
        binding.right_arm = right;
        let motion = if side == ArmSide::Left {
            ArmMotionGeometry {
                left: Some(geometry),
                right: None,
            }
        } else {
            ArmMotionGeometry {
                left: None,
                right: Some(geometry),
            }
        };
        app.world_mut().entity_mut(root).insert((
            binding,
            DefaultArmPose::from_chains(generation, left, right),
            AvatarAssetId::new("sha256:loss-rest-rotation-model"),
            motion,
            vtuber_avatar::body_scale::BodyScaleMeters {
                generation,
                scale_meters: 0.7,
            },
            DynamicArmTargets::default(),
        ));
        let publish = |app: &mut App, target, weight| {
            let weights = ArmBlendWeight {
                wrist: weight,
                pole: weight,
                palm: weight,
                fingers: weight,
            };
            let (targets, weights) = if side == ArmSide::Left {
                (
                    ArmTrackingTargets {
                        left: target,
                        right: None,
                    },
                    ArmBlendWeights {
                        left: weights,
                        right: ArmBlendWeight::ZERO,
                    },
                )
            } else {
                (
                    ArmTrackingTargets {
                        left: None,
                        right: target,
                    },
                    ArmBlendWeights {
                        left: ArmBlendWeight::ZERO,
                        right: weights,
                    },
                )
            };
            app.insert_resource(TrackedArmControl {
                generation: Some(generation),
                view_to_model: Quat::IDENTITY,
                frame: Some(ArmControlFrame {
                    thorax: None,
                    source_seq: FrameSeq(1),
                    captured_at: MonoTimeNs(0),
                    produced_at: MonoTimeNs(0),
                    targets,
                    weights,
                }),
            });
        };
        publish(&mut app, None, 0.0);
        app.update();
        let entities: Vec<_> = [shoulder, upper, lower, hand]
            .into_iter()
            .chain(finger_entities)
            .collect();
        let initial: Vec<_> = entities
            .iter()
            .map(|&entity| *app.world().get::<Transform>(entity).unwrap())
            .collect();
        let turn = Quat::from_rotation_z(sign * 2.5);
        let position = |entity| {
            app.world()
                .get::<GlobalTransform>(entity)
                .unwrap()
                .translation()
        };
        let neutral_bend =
            (position(lower) - position(upper)).angle_between(position(hand) - position(lower));
        let wrist = turn * (position(hand) - position(upper));
        let pole = turn * (position(lower) - position(upper));
        let mut target = ArmTrackingTarget {
            wrist: (wrist / 0.49).to_array(),
            elbow_pole: (pole / 0.49).to_array(),
            palm_normal: None,
            palm_forward: None,
            fingers: Some(HandFingerPose {
                fingers: [[0.8, 0.9, 0.4]; 4],
                spread: [0.0; 4],
                thumb: [0.0; 2],
                thumb_spread: 0.0,
                thumb_cmc: [0.0; 2],
            }),
        };
        let solution = solve_two_bone_arm(ArmIkInput::from_chain(
            &chain,
            vtuber_avatar::tracked_arm_ik_target(chain.rest, target, Quat::IDENTITY),
        ))
        .unwrap();
        let palm_rest = (fingers.index.proximal.unwrap().rest.position - hand_rest.position)
            .cross(fingers.little.proximal.unwrap().rest.position - hand_rest.position)
            .normalize();
        let normal =
            solution.lower_arm_global_rotation * lower_rest.global_rotation.inverse() * palm_rest;
        let axis = (solution.wrist - solution.elbow).normalize();
        target.palm_normal = Some((Quat::from_axis_angle(axis, 1.0) * normal).to_array());
        let palm_forward = ((fingers.index.proximal.unwrap().rest.position - hand_rest.position)
            .normalize()
            + (fingers.little.proximal.unwrap().rest.position - hand_rest.position).normalize())
        .normalize();
        target.palm_forward = Some(
            (Quat::from_axis_angle(axis, 1.0)
                * solution.lower_arm_global_rotation
                * lower_rest.global_rotation.inverse()
                * palm_forward)
                .to_array(),
        );
        publish(&mut app, Some(target), 1.0);
        for _ in 0..120 {
            app.update();
        }
        assert!(
            !rotation_close(
                app.world().get::<Transform>(entities[4]).unwrap().rotation,
                initial[4].rotation
            ),
            "the test must first articulate the fingers"
        );
        let mut loss = LossBlend::new();
        let loss_profile = LossBlendProfile::default();
        loss.advance(MonoTimeNs(0), true, &loss_profile);
        loss.advance(MonoTimeNs(1_000_000_000), true, &loss_profile);
        for tick in 0..400 {
            loss.advance(
                MonoTimeNs(1_000_000_000 + tick * 16_666_667),
                false,
                &loss_profile,
            );
            let t = tick as f32 / 399.0;
            let torso = Transform::from_translation(chest_rest.position - hips_rest.position)
                .with_rotation(Quat::from_rotation_y(t * 0.5) * chest_rest.global_rotation);
            *app.world_mut().get_mut::<Transform>(chest).unwrap() = torso;
            *app.world_mut().get_mut::<GlobalTransform>(chest).unwrap() =
                GlobalTransform::from_translation(hips_rest.position).mul_transform(torso);
            publish(&mut app, Some(target), loss.weight());
            app.update();
            let position = |entity| {
                app.world()
                    .get::<GlobalTransform>(entity)
                    .unwrap()
                    .translation()
            };
            let u = position(lower) - position(upper);
            let l = position(hand) - position(lower);
            assert!((u.length() - 0.25).abs() < EPSILON);
            assert!((l.length() - 0.24).abs() < EPSILON);
            assert!(
                (u.angle_between(l) - neutral_bend).abs() < 0.05,
                "return must preserve the extended arm's carrying angle: side {side:?}, tick {tick}"
            );
            for (&entity, rest) in entities.iter().zip(&initial) {
                let local = app.world().get::<Transform>(entity).unwrap();
                assert_eq!(local.translation, rest.translation);
                assert_eq!(local.scale, rest.scale);
            }
            assert!(
                rotation_close(
                    app.world().get::<Transform>(hand).unwrap().rotation,
                    hand_rest.local_rotation
                ),
                "the wrist must inherit pronation once through FK"
            );
        }
        assert_eq!(loss.weight(), 0.0);
        for (&entity, rest) in entities.iter().zip(&initial) {
            assert!(
                rotation_close(
                    app.world().get::<Transform>(entity).unwrap().rotation,
                    rest.rotation
                ),
                "the complete chain must return to its actual initial local pose: {side:?}, {entity:?}"
            );
        }
    }
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
fn unobserved_clavicle_stays_at_rest_and_finger_axes_are_converted() {
    let mut chain = optional_correction_chain();
    let mut little = chain.finger_rest.index.proximal.unwrap();
    little.rest.position += Vec3::Z * 0.03;
    chain.finger_rest.little.proximal = Some(little);
    let pose = DefaultArmPose::from_chains(AvatarGeneration(1), Some(chain), None);
    let resolved = pose.left.expect("complete chain should resolve");
    let shoulder = resolved.shoulder.expect("optional shoulder should resolve");
    let (_, shoulder_angle) = shoulder.delta.to_axis_angle();
    assert!(shoulder_angle <= EPSILON);
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
fn clavicle_presence_does_not_add_artificial_elbow_articulation() {
    let with_shoulder = optional_correction_chain();
    let mut without_shoulder = with_shoulder;
    without_shoulder.shoulder = None;
    without_shoulder.rest.shoulder = None;
    without_shoulder.capabilities.has_shoulder = false;
    let with = DefaultArmPose::from_chains(AvatarGeneration(1), Some(with_shoulder), None)
        .left
        .unwrap();
    let without = DefaultArmPose::from_chains(AvatarGeneration(1), Some(without_shoulder), None)
        .left
        .unwrap();
    assert_eq!(with.shoulder.unwrap().delta, Quat::IDENTITY);
    assert!(rotation_close(
        with.upper_arm_delta,
        without.upper_arm_delta
    ));
    assert!(rotation_close(
        with.lower_arm_delta,
        without.lower_arm_delta
    ));
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
        DefaultArmPose {
            generation: AvatarGeneration(4),
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
