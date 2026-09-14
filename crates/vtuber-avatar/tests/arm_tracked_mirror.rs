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
    ArmChainBinding, ArmChainCapabilities, ArmMotionGeometry, ArmPoseSourceKind, ArmRestGeometry,
    ArmSide, ArmSourceSelection, AvatarAssetId, AvatarBinding, AvatarGeneration, AvatarLifecycle,
    AvatarMotionMirror, DynamicArmTargets, RestSpaceBonePose, TrackedArmControl,
    build_arm_motion_rest_geometry, update_tracked_arm_targets,
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

fn build_app(mirror_enabled: bool, generation: AvatarGeneration) -> (App, MirrorRig) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<AvatarLifecycle>()
        .init_resource::<ArmSourceSelection>()
        .init_resource::<TrackedArmControl>()
        .add_systems(Update, update_tracked_arm_targets);

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

    let mut lifecycle = AvatarLifecycle::default();
    lifecycle.request_load(root).expect("load request");
    lifecycle.start_binding(root);
    lifecycle.finish_ready();
    app.insert_resource(lifecycle);

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
    }
}

#[test]
fn enabled_mirror_equals_a_manually_mirrored_frame_with_the_mirror_disabled() {
    let generation = AvatarGeneration(7);
    let weight = ArmBlendWeight {
        wrist: 0.25,
        pole: 0.5,
    };
    let (mut app, rig) = build_app(true, generation);

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
    assert!(
        mirrored.left.is_none(),
        "the observed left arm must resolve on the avatar's right arm"
    );
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
    assert!(manual.left.is_none());
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
fn disabled_mirror_keeps_the_canonical_side_assignment() {
    let generation = AvatarGeneration(11);
    let (mut app, rig) = build_app(false, generation);

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
    assert!(targets.right.is_none());
}
