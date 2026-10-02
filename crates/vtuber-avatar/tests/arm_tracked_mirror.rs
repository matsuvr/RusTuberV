// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for the avatar motion mirror on observed arms.
//!
//! The tracked control frame stays canonical (unmirrored). When
//! `AvatarMotionMirror` is enabled, `update_tracked_arm_targets` must apply the
//! same X reflection and left/right swap to the observed targets and weights
//! that the face channels receive, exactly once.

use bevy::prelude::*;
use bevy_vrm1::prelude::{RestGlobalTransform, RestTransform};
use vtuber_avatar::{
    ArmChainBinding, ArmChainCapabilities, ArmMotionGeometry, ArmPipelineInput, ArmPoseProfile,
    ArmPoseSourceKind, ArmRestGeometry, ArmSide, ArmSourceSelection, AvatarAssetId, AvatarBinding,
    AvatarGeneration, AvatarLifecycle, AvatarMotionMirror, DynamicArmProfile, DynamicArmTargets,
    RestSpaceBonePose, TrackedArmControl, build_arm_motion_rest_geometry, resolve_arm_pose,
    update_dynamic_arm_targets, update_tracked_arm_targets,
};
use vtuber_core::arm_tracking::{
    ArmBlendWeight, ArmBlendWeights, ArmControlFrame, ArmTrackingTarget, ArmTrackingTargets,
};
use vtuber_core::{FrameSeq, MonoTimeNs};

struct MirrorRig {
    root: Entity,
    right_upper: Entity,
}

fn rest_bone(position: Vec3) -> RestSpaceBonePose {
    RestSpaceBonePose {
        position,
        global_rotation: Quat::IDENTITY,
        local_rotation: Quat::IDENTITY,
    }
}

fn chain(side: ArmSide, upper: Entity, lower: Entity) -> ArmChainBinding {
    let sign = match side {
        ArmSide::Left => 1.0_f32,
        ArmSide::Right => -1.0,
    };
    let shoulder_pos = Vec3::new(0.04 * sign, 1.30, 0.0);
    let upper_origin = Vec3::new(0.16 * sign, 1.32, 0.0);
    let elbow = upper_origin + Vec3::new(0.24 * sign, -0.04, 0.0);
    let wrist = elbow + Vec3::new(-0.02 * sign, -0.25, -0.01);
    ArmChainBinding {
        side,
        shoulder: None,
        upper_arm: upper,
        lower_arm: lower,
        hand: lower,
        fingers: vtuber_avatar::FingerReferences::default(),
        finger_rest: vtuber_avatar::FingerRestReferences::default(),
        rest: ArmRestGeometry {
            shoulder: Some(rest_bone(shoulder_pos)),
            upper_arm: rest_bone(upper_origin),
            elbow: rest_bone(elbow),
            wrist: rest_bone(wrist),
            upper_arm_length: upper_origin.distance(elbow),
            forearm_length: elbow.distance(wrist),
            total_arm_length: upper_origin.distance(wrist),
        },
        capabilities: ArmChainCapabilities {
            has_shoulder: true,
            has_fingers: false,
        },
    }
}

fn build_app(mirror_enabled: bool) -> (App, MirrorRig) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<AvatarLifecycle>()
        .init_resource::<ArmSourceSelection>()
        .init_resource::<TrackedArmControl>()
        .init_resource::<vtuber_avatar::ActiveControlFrame>()
        .add_systems(
            Update,
            (update_dynamic_arm_targets, update_tracked_arm_targets).chain(),
        );

    let mut mirror = AvatarMotionMirror::default();
    if !mirror_enabled {
        mirror.toggle();
    }
    app.insert_resource(mirror);

    app.insert_resource(ArmSourceSelection {
        mode: ArmPoseSourceKind::TrackedPose,
        ..Default::default()
    });

    let root = app
        .world_mut()
        .spawn((Transform::IDENTITY, GlobalTransform::IDENTITY))
        .id();
    let spawn_bone = |app: &mut App, parent: Entity, offset: Vec3| {
        app.world_mut()
            .spawn((
                Transform::from_translation(offset),
                GlobalTransform::IDENTITY,
                RestTransform(Transform::from_translation(offset)),
                RestGlobalTransform(GlobalTransform::from_translation(offset)),
                ChildOf(parent),
            ))
            .id()
    };
    let chest = spawn_bone(&mut app, root, Vec3::Y * 0.26);
    let left_upper = spawn_bone(&mut app, chest, Vec3::new(0.12, 0.10, 0.0));
    let left_lower = spawn_bone(&mut app, left_upper, Vec3::new(0.24, -0.04, 0.0));
    let right_upper = spawn_bone(&mut app, chest, Vec3::new(-0.12, 0.10, 0.0));
    let right_lower = spawn_bone(&mut app, right_upper, Vec3::new(-0.24, -0.04, 0.0));

    let left_arm = chain(ArmSide::Left, left_upper, left_lower);
    let right_arm = chain(ArmSide::Right, right_upper, right_lower);
    let motion = ArmMotionGeometry {
        left: Some(build_arm_motion_rest_geometry(
            ArmSide::Left,
            &left_arm.rest,
            Some(Vec3::new(0.0, 0.92, 0.0)),
            Some(Quat::IDENTITY),
            Some(Vec3::new(0.0, 1.20, 0.02)),
        )),
        right: Some(build_arm_motion_rest_geometry(
            ArmSide::Right,
            &right_arm.rest,
            Some(Vec3::new(0.0, 0.92, 0.0)),
            Some(Quat::IDENTITY),
            Some(Vec3::new(0.0, 1.20, 0.02)),
        )),
    };

    let mut lifecycle = AvatarLifecycle::default();
    lifecycle.request_load(root).expect("load request");
    lifecycle.start_binding(root);
    lifecycle.finish_ready();
    let generation = lifecycle.current_generation();
    app.insert_resource(lifecycle);

    let binding = AvatarBinding {
        root,
        head: chest,
        neck: None,
        upper_chest: None,
        chest: Some(chest),
        spine: None,
        left_upper_arm: Some(left_upper),
        right_upper_arm: Some(right_upper),
        left_arm: Some(left_arm),
        right_arm: Some(right_arm),
        left_eye: None,
        right_eye: None,
        generation,
    };
    app.world_mut().entity_mut(root).insert((
        binding,
        AvatarAssetId::new("sha256:tracked-mirror-model"),
        motion,
        vtuber_avatar::body_scale::BodyScaleMeters {
            generation,
            scale_meters: 0.7,
        },
        DynamicArmTargets::default(),
    ));

    (app, MirrorRig { root, right_upper })
}

fn control_frame(
    seq: u64,
    targets: ArmTrackingTargets,
    weights: ArmBlendWeights,
) -> ArmControlFrame {
    ArmControlFrame {
        source_seq: FrameSeq(seq),
        captured_at: MonoTimeNs(0),
        produced_at: MonoTimeNs(0),
        targets,
        weights,
    }
}

fn set_control(app: &mut App, generation: AvatarGeneration, frame: ArmControlFrame) {
    let mut control = app.world_mut().resource_mut::<TrackedArmControl>();
    control.generation = Some(generation);
    control.frame = Some(frame);
}

fn resolved_targets(app: &App, rig: &MirrorRig) -> DynamicArmTargets {
    *app.world()
        .get::<DynamicArmTargets>(rig.root)
        .expect("dynamic targets present")
}

fn observed_target() -> ArmTrackingTarget {
    ArmTrackingTarget {
        wrist: [0.30, -0.24, 0.05],
        elbow_pole: [0.16, -0.10, 0.02],
        palm_normal: Some([0.10, 0.20, -0.90]),
        fingers: None,
    }
}

#[test]
fn the_live_stage_order_keeps_palm_roll_state_and_returns_from_a_turn() {
    let (mut app, rig) = build_app(false);
    let generation = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        std::time::Duration::from_secs_f64(1.0 / 60.0),
    ));
    let chain = {
        let mut binding = app.world_mut().get_mut::<AvatarBinding>(rig.root).unwrap();
        let chain = binding.left_arm.as_mut().unwrap();
        let wrist = chain.rest.wrist.position;
        chain.finger_rest.index.proximal = Some(vtuber_avatar::FingerJointRestBinding {
            entity: chain.upper_arm,
            rest: rest_bone(wrist + Vec3::new(0.05, 0.0, 0.005)),
        });
        chain.finger_rest.little.proximal = Some(vtuber_avatar::FingerJointRestBinding {
            entity: chain.lower_arm,
            rest: rest_bone(wrist + Vec3::new(0.05, 0.0, -0.005)),
        });
        *chain
    };
    let publish = |app: &mut App, seq, palm_normal| {
        let target = ArmTrackingTarget {
            palm_normal,
            ..observed_target()
        };
        set_control(
            app,
            generation,
            control_frame(
                seq,
                ArmTrackingTargets {
                    left: Some(target),
                    right: None,
                },
                ArmBlendWeights {
                    left: ArmBlendWeight::ONE,
                    right: ArmBlendWeight::ZERO,
                },
            ),
        );
    };
    let palm = |pose: vtuber_avatar::ResolvedArmPose| {
        let upper = chain.rest.upper_arm.global_rotation
            * pose.upper_arm_delta
            * chain.rest.upper_arm.global_rotation.inverse();
        let lower = upper * chain.rest.elbow.global_rotation * pose.lower_arm_delta;
        let hand = lower
            * chain.rest.elbow.global_rotation.inverse()
            * chain.rest.wrist.global_rotation
            * pose.hand.map_or(Quat::IDENTITY, |h| h.delta);
        let rest_normal = (chain.finger_rest.index.proximal.unwrap().rest.position
            - chain.rest.wrist.position)
            .cross(
                chain.finger_rest.little.proximal.unwrap().rest.position
                    - chain.rest.wrist.position,
            )
            .normalize();
        (hand * (chain.rest.wrist.global_rotation.inverse() * rest_normal)).normalize()
    };
    publish(&mut app, 1, None);
    app.update();
    let baseline = resolved_targets(&app, &rig).left.unwrap();
    let normal = palm(baseline);
    let u = chain.rest.upper_arm.global_rotation
        * baseline.upper_arm_delta
        * chain.rest.upper_arm.global_rotation.inverse();
    let l = chain.rest.elbow.global_rotation
        * baseline.lower_arm_delta
        * chain.rest.elbow.global_rotation.inverse();
    let axis = (u * l * (chain.rest.wrist.position - chain.rest.elbow.position)).normalize();
    let turned = Quat::from_axis_angle(axis, 60.0_f32.to_radians()) * normal;
    publish(&mut app, 2, Some(turned.to_array()));
    for _ in 0..120 {
        app.update();
    }
    let pose = resolved_targets(&app, &rig).left.unwrap();
    assert!(
        palm(pose).dot(turned) > 0.999,
        "the composed hand must reach the observed palm, got {:?}",
        palm(pose)
    );
    assert_eq!(
        pose.hand.unwrap().delta,
        Quat::IDENTITY,
        "pronation belongs to the forearm"
    );
    assert!(pose.upper_arm_delta.dot(baseline.upper_arm_delta).abs() > 1.0 - 1.0e-6);
    publish(&mut app, 3, Some(normal.to_array()));
    for _ in 0..120 {
        app.update();
    }
    let returned = resolved_targets(&app, &rig).left.unwrap();
    assert!(
        palm(returned).dot(normal) > 0.999,
        "the back of the hand must not remain reversed"
    );
    app.world_mut().resource_mut::<TrackedArmControl>().frame = None;
    app.update();
    assert_eq!(resolved_targets(&app, &rig), DynamicArmTargets::default());
}

#[test]
fn enabled_mirror_equals_a_manually_mirrored_frame_with_the_mirror_disabled() {
    let weight = ArmBlendWeight {
        wrist: 0.25,
        pole: 0.5,
        palm: 0.75,
        fingers: 0.0,
    };
    let (mut app, rig) = build_app(true);
    let generation = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();

    set_control(
        &mut app,
        generation,
        control_frame(
            1,
            ArmTrackingTargets {
                left: Some(observed_target()),
                right: None,
            },
            ArmBlendWeights {
                left: weight,
                right: ArmBlendWeight::ZERO,
            },
        ),
    );
    app.update();

    let mirrored = resolved_targets(&app, &rig);
    assert_eq!(mirrored.generation, Some(generation));
    assert_eq!(mirrored.source_seq, Some(FrameSeq(1)));
    let neutral_left = mirrored
        .left
        .expect("the unobserved arm keeps its initial pose");
    let mirrored_right = mirrored.right.expect("right arm resolved");

    app.world_mut()
        .resource_mut::<AvatarMotionMirror>()
        .toggle();
    set_control(
        &mut app,
        generation,
        control_frame(
            2,
            ArmTrackingTargets {
                left: None,
                right: Some(observed_target().mirrored()),
            },
            ArmBlendWeights {
                left: ArmBlendWeight::ZERO,
                right: weight,
            },
        ),
    );
    app.update();

    let manual = resolved_targets(&app, &rig);
    assert_eq!(manual.left, Some(neutral_left));
    assert_eq!(
        manual.right,
        Some(mirrored_right),
        "the enabled mirror must resolve exactly like the manually mirrored frame"
    );
    assert_eq!(
        mirrored_right.upper_arm, rig.right_upper,
        "the mirrored target must resolve on the avatar's right arm"
    );
}

#[test]
fn opposed_bend_planes_resolve_and_return_instead_of_freezing() {
    let (mut app, rig) = build_app(false);
    let generation = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        std::time::Duration::from_secs_f64(1.0 / 60.0),
    ));

    set_control(
        &mut app,
        generation,
        control_frame(
            1,
            ArmTrackingTargets {
                left: Some(observed_target()),
                right: None,
            },
            ArmBlendWeights {
                left: ArmBlendWeight::ONE,
                right: ArmBlendWeight::ZERO,
            },
        ),
    );
    app.update();
    let first = resolved_targets(&app, &rig)
        .left
        .expect("the first frame resolves");

    // The virtual blend source, computed exactly as the system does, then a
    // tracked target whose converted pole is reflected through the converted
    // wrist. Cartesian pole blending used to treat the whole return as
    // degenerate and freeze; shoulder joint interpolation remains defined.
    let (opposed, neutral) = {
        let binding = app.world().get::<AvatarBinding>(rig.root).expect("binding");
        let chain = binding.left_arm.as_ref().expect("left chain");
        let motion = app
            .world()
            .get::<ArmMotionGeometry>(rig.root)
            .expect("motion geometry")
            .left
            .as_ref()
            .expect("left motion");
        let input = ArmPipelineInput {
            chain,
            motion,
            legacy_profile: ArmPoseProfile::default(),
            dynamic_profile: DynamicArmProfile::default(),
            head_offset: Vec3::ZERO,
            body_offset: Vec3::ZERO,
            torso_delta: Quat::IDENTITY,
            body_scale_meters: 0.7,
        };
        let (neutral, outcome) = resolve_arm_pose(&input, ArmPoseSourceKind::VirtualHandAnchor)
            .expect("pipeline error")
            .expect("virtual target");
        let virtual_target = outcome.hand_target;
        let total = chain.rest.total_arm_length;
        let to_tracking = |p: Vec3| ((p - chain.rest.upper_arm.position) / total).to_array();
        (
            ArmTrackingTarget {
                wrist: to_tracking(virtual_target.wrist),
                elbow_pole: to_tracking(virtual_target.wrist * 2.0 - virtual_target.elbow_pole),
                palm_normal: None,
                fingers: None,
            },
            neutral,
        )
    };

    set_control(
        &mut app,
        generation,
        control_frame(
            2,
            ArmTrackingTargets {
                left: Some(opposed),
                right: None,
            },
            ArmBlendWeights {
                left: ArmBlendWeight {
                    wrist: 0.5,
                    pole: 0.25,
                    palm: 0.0,
                    fingers: 0.0,
                },
                right: ArmBlendWeight::ZERO,
            },
        ),
    );
    for _ in 0..120 {
        app.update();
    }
    let middle = resolved_targets(&app, &rig);
    assert_eq!(middle.source_seq, Some(FrameSeq(2)));
    let middle = middle.left.unwrap();
    assert!(middle.upper_arm_delta.is_finite());
    assert!(middle.lower_arm_delta.is_finite());
    assert!(middle.upper_arm_delta.angle_between(first.upper_arm_delta) > 0.01);
    set_control(
        &mut app,
        generation,
        control_frame(
            3,
            ArmTrackingTargets {
                left: Some(opposed),
                right: None,
            },
            ArmBlendWeights::default(),
        ),
    );
    app.update();
    let returned = resolved_targets(&app, &rig).left.unwrap();
    assert!(returned.upper_arm_delta.dot(neutral.upper_arm_delta).abs() > 0.999999);
    assert!(returned.lower_arm_delta.dot(neutral.lower_arm_delta).abs() > 0.999999);
}

#[test]
fn disabled_mirror_keeps_the_canonical_side_assignment() {
    let (mut app, rig) = build_app(false);
    let generation = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();

    set_control(
        &mut app,
        generation,
        control_frame(
            1,
            ArmTrackingTargets {
                left: Some(observed_target()),
                right: None,
            },
            ArmBlendWeights {
                left: ArmBlendWeight::ONE,
                right: ArmBlendWeight::ZERO,
            },
        ),
    );
    app.update();

    let targets = resolved_targets(&app, &rig);
    assert!(
        targets.left.is_some(),
        "without the mirror the observed left arm stays on the left"
    );
    let right = targets
        .right
        .expect("the unobserved right arm keeps its initial pose");
    assert_eq!(right.upper_arm, rig.right_upper);
}
