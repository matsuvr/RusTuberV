// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Position-aware upper-body solve integration tests (Issue #167).

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_vrm1::prelude::*;
use vtuber_avatar::{
    BodyTrackingPoseInput, BodyTrackingPositionInput, BodyTrackingPositionProfile,
    apply_direct_body_position, apply_direct_body_tracking,
};

const EPSILON: f32 = 1.0e-4;
const FRAME_MILLIS: u64 = 16;

#[derive(Clone, Copy)]
struct Rig {
    root: Entity,
    spine: Entity,
    chest: Entity,
    #[allow(dead_code)]
    upper_chest: Option<Entity>,
    #[allow(dead_code)]
    neck: Option<Entity>,
    head: Entity,
}

#[derive(Resource)]
struct RigResource {
    rig: Rig,
}

fn live_input(head_offset: Vec3, body_offset: Vec3) -> BodyTrackingPositionInput {
    BodyTrackingPositionInput {
        head_offset,
        body_offset,
        weight: 1.0,
        active: true,
        ..Default::default()
    }
}

/// Builds a synthetic humanoid rig: root -> spine -> chest -> [upperChest] ->
/// [neck] -> head, each bone resting 0.15 m above its parent with identity
/// rotations. Rest globals accumulate so the spine-to-head lever arm is
/// measurable from immutable rest geometry (0.6 m with all bones present).
fn build_rig(
    with_upper_chest: bool,
    with_neck: bool,
    position_input: BodyTrackingPositionInput,
) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<vtuber_avatar::direct_position::TorsoLeanBases>()
        .insert_resource(TimeUpdateStrategy::ManualInstant(instant_at(0)))
        .add_systems(
            PostUpdate,
            (
                vtuber_avatar::direct_position::restore_torso_lean,
                apply_direct_body_tracking,
                apply_direct_body_position,
            )
                .chain(),
        );

    let root = app
        .world_mut()
        .spawn((
            Vrm,
            Transform::IDENTITY,
            GlobalTransform::IDENTITY,
            BodyTracking::default(),
            BodyTrackingPoseInput {
                yaw_radians: 0.0,
                pitch_radians: 0.0,
                roll_radians: 0.0,
                weight: 1.0,
                active: true,
            },
            BodyTrackingPositionProfile::default(),
            position_input,
        ))
        .id();

    let mut height = 0.0_f32;
    let mut parent = root;
    let spawn_bone = |app: &mut App, parent: Entity, height: f32| {
        let local = Transform::from_translation(Vec3::new(0.0, 0.15, 0.0));
        let global = Transform::from_translation(Vec3::new(0.0, height, 0.0));
        app.world_mut()
            .spawn((
                local,
                GlobalTransform::IDENTITY,
                RestTransform(local),
                RestGlobalTransform(GlobalTransform::from(global)),
                ChildOf(parent),
            ))
            .id()
    };

    let spine = spawn_bone(&mut app, parent, height + 0.15);
    height += 0.15;
    parent = spine;
    let chest = spawn_bone(&mut app, parent, height + 0.15);
    height += 0.15;
    parent = chest;
    let upper_chest = with_upper_chest.then(|| {
        let entity = spawn_bone(&mut app, parent, height + 0.15);
        height += 0.15;
        parent = entity;
        entity
    });
    let _ = &mut parent;
    let neck = with_neck.then(|| {
        let entity = spawn_bone(&mut app, parent, height + 0.15);
        height += 0.15;
        parent = entity;
        entity
    });
    let head = spawn_bone(&mut app, parent, height + 0.15);

    let mut root_entity = app.world_mut().entity_mut(root);
    root_entity.insert((
        HeadBoneEntity(head),
        ChestBoneEntity(chest),
        SpineBoneEntity(spine),
    ));
    if let Some(upper_chest) = upper_chest {
        root_entity.insert(UpperChestBoneEntity(upper_chest));
    }
    if let Some(neck) = neck {
        root_entity.insert(NeckBoneEntity(neck));
    }

    let rig = Rig {
        root,
        spine,
        chest,
        upper_chest,
        neck,
        head,
    };
    app.insert_resource(RigResource { rig });
    app
}

/// Advances the manual clock to `tick` and runs one full schedule pass.
fn instant_at(millis: u64) -> Instant {
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *BASE.get_or_init(Instant::now) + Duration::from_millis(millis)
}

fn tick(app: &mut App, tick_millis: u64) {
    if let Some(mut strategy) = app.world_mut().get_resource_mut::<TimeUpdateStrategy>() {
        *strategy = TimeUpdateStrategy::ManualInstant(instant_at(tick_millis));
    }
    app.update();
}

fn rig_of(app: &App) -> Rig {
    app.world().resource::<RigResource>().rig
}

fn head_world_position(app: &App, head: Entity) -> Vec3 {
    app.world()
        .get::<GlobalTransform>(head)
        .unwrap()
        .translation()
}

fn root_translation(app: &App, root: Entity) -> Vec3 {
    app.world().get::<Transform>(root).unwrap().translation
}

fn bone_rotation(app: &App, entity: Entity) -> Quat {
    app.world().get::<Transform>(entity).unwrap().rotation
}

fn assert_relative_vec(actual: Vec3, expected: Vec3) {
    assert!(
        actual.distance(expected) < EPSILON,
        "expected {expected:?}, got {actual:?}"
    );
}

fn assert_relative_quat(actual: Quat, expected: Quat) {
    assert!(
        actual.angle_between(expected) < EPSILON,
        "expected {expected:?}, got {actual:?}"
    );
}

#[test]
fn tracked_arms_preserve_head_follow_through_the_torso() {
    use vtuber_avatar::{AvatarBinding, AvatarGeneration, TrackedArmControl};
    use vtuber_core::arm_tracking::{ArmControlFrame, ThoraxTarget};

    for thorax in [
        None,
        Some(ThoraxTarget {
            rotation: Quat::IDENTITY.to_array(),
            shoulder_offsets: [[0.0; 3]; 2],
            weight: 1.0,
        }),
    ] {
        let mut app = build_rig(
            true,
            true,
            live_input(Vec3::new(0.06, 0.0, 0.0), Vec3::ZERO),
        );
        let rig = rig_of(&app);
        let generation = AvatarGeneration(1);
        app.world_mut().entity_mut(rig.root).insert((
            AvatarBinding::head_only(rig.root, rig.head, generation),
            vtuber_avatar::pose::natural_body_tracking_profile(),
            BodyTrackingPoseInput {
                yaw_radians: 0.5,
                pitch_radians: 0.25,
                roll_radians: 0.25,
                weight: 1.0,
                active: true,
            },
        ));
        app.insert_resource(vtuber_avatar::ArmSourceSelection {
            mode: vtuber_avatar::ArmPoseSourceKind::TrackedPose,
            ..Default::default()
        });
        app.insert_resource(TrackedArmControl {
            generation: Some(generation),
            frame: Some(ArmControlFrame {
                thorax,
                source_seq: vtuber_core::FrameSeq(1),
                captured_at: vtuber_core::MonoTimeNs(0),
                produced_at: vtuber_core::MonoTimeNs(0),
                targets: Default::default(),
                weights: Default::default(),
            }),
            ..Default::default()
        });
        for frame in 0..180 {
            tick(&mut app, frame * FRAME_MILLIS);
        }
        for bone in [rig.spine, rig.chest, rig.upper_chest.unwrap()] {
            assert!(
                bone_rotation(&app, bone).angle_between(Quat::IDENTITY) > 0.01,
                "head follow stopped at {bone:?}, thorax={thorax:?}"
            );
        }
    }
}

#[test]
fn observed_chest_composes_with_head_follow_without_turning_head_twice() {
    use vtuber_avatar::{AvatarBinding, AvatarGeneration, AvatarMotionMirror, TrackedArmControl};
    use vtuber_core::arm_tracking::{ArmControlFrame, ThoraxTarget};
    for (head_yaw, chest_yaw) in [(0.25, 0.0), (0.0, 0.25), (0.25, -0.2)] {
        let mut app = build_rig(true, true, live_input(Vec3::ZERO, Vec3::ZERO));
        let mut face_only = build_rig(true, true, live_input(Vec3::ZERO, Vec3::ZERO));
        let face_rig = rig_of(&face_only);
        face_only
            .world_mut()
            .get_mut::<BodyTrackingPoseInput>(face_rig.root)
            .unwrap()
            .yaw_radians = head_yaw;
        app.insert_resource(vtuber_avatar::ArmSourceSelection {
            mode: vtuber_avatar::ArmPoseSourceKind::TrackedPose,
            ..Default::default()
        });
        let rig = rig_of(&app);
        let generation = AvatarGeneration(1);
        app.world_mut()
            .entity_mut(rig.root)
            .insert(AvatarBinding::head_only(rig.root, rig.head, generation));
        app.world_mut()
            .get_mut::<BodyTrackingPoseInput>(rig.root)
            .unwrap()
            .yaw_radians = head_yaw;
        let mut mirror = AvatarMotionMirror::default();
        mirror.toggle();
        app.insert_resource(mirror);
        app.insert_resource(TrackedArmControl {
            generation: Some(generation),
            view_to_model: Quat::IDENTITY,
            frame: Some(ArmControlFrame {
                thorax: Some(ThoraxTarget {
                    rotation: Quat::from_rotation_y(chest_yaw).to_array(),
                    shoulder_offsets: [[0.0; 3]; 2],
                    weight: 1.0,
                }),
                source_seq: vtuber_core::FrameSeq(1),
                captured_at: vtuber_core::MonoTimeNs(0),
                produced_at: vtuber_core::MonoTimeNs(0),
                targets: Default::default(),
                weights: Default::default(),
            }),
        });
        for frame in 0..180 {
            tick(&mut app, frame * FRAME_MILLIS);
            tick(&mut face_only, frame * FRAME_MILLIS);
        }
        let chest = app
            .world()
            .get::<GlobalTransform>(rig.upper_chest.unwrap())
            .unwrap()
            .rotation();
        let head = app
            .world()
            .get::<GlobalTransform>(rig.head)
            .unwrap()
            .rotation();
        let follow = face_only
            .world()
            .get::<GlobalTransform>(face_rig.upper_chest.unwrap())
            .unwrap()
            .rotation();
        assert!(chest.dot(Quat::from_rotation_y(chest_yaw) * follow).abs() > 1.0 - 1.0e-6);
        assert!(head.dot(Quat::from_rotation_y(head_yaw)).abs() > 1.0 - 1.0e-6);
    }
}

#[test]
fn lateral_target_keeps_root_x_and_produces_torso_lean() {
    let mut app = build_rig(
        true,
        true,
        live_input(Vec3::new(0.06, 0.0, 0.0), Vec3::ZERO),
    );
    tick(&mut app, FRAME_MILLIS);
    let rig = rig_of(&app);

    let translation = root_translation(&app, rig.root);
    assert!(
        translation.x.abs() < EPSILON && translation.y.abs() < EPSILON,
        "root must not follow the lateral target: {translation:?}"
    );

    // Torso bones lean; the head bone itself carries no local lean.
    assert!(bone_rotation(&app, rig.spine).angle_between(Quat::IDENTITY) > EPSILON);
    assert!(bone_rotation(&app, rig.chest).angle_between(Quat::IDENTITY) > EPSILON);
    assert_relative_quat(bone_rotation(&app, rig.head), Quat::IDENTITY);

    // Head world position moved toward image right (+X under identity rest).
    let head_position = head_world_position(&app, rig.head);
    assert!(
        head_position.x > 0.005,
        "head should displace laterally: {head_position:?}"
    );
    assert!(head_position.is_finite());
}

#[test]
fn depth_and_vertical_targets_move_the_root_per_compensation() {
    let body_offset = Vec3::new(0.0, 0.04, -0.08);
    let mut app = build_rig(true, true, live_input(Vec3::ZERO, body_offset));
    tick(&mut app, FRAME_MILLIS);
    let rig = rig_of(&app);

    // Semantic +Z (away from camera) maps to model -Z under identity rest.
    let translation = root_translation(&app, rig.root);
    assert_relative_vec(translation, Vec3::new(0.0, body_offset.y, -body_offset.z));
}

#[test]
fn repeated_evaluation_of_the_same_inputs_does_not_accumulate() {
    let input = live_input(Vec3::new(0.05, 0.01, -0.02), Vec3::new(0.0, 0.03, -0.04));
    let mut app = build_rig(true, true, input);
    let rig = rig_of(&app);

    // Two render frames re-evaluating the identical control channels.
    tick(&mut app, FRAME_MILLIS);
    let head_steady = head_world_position(&app, rig.head);
    let root_steady = root_translation(&app, rig.root);
    let spine_steady = bone_rotation(&app, rig.spine);

    tick(&mut app, FRAME_MILLIS * 2);
    assert_relative_vec(head_world_position(&app, rig.head), head_steady);
    assert_relative_vec(root_translation(&app, rig.root), root_steady);
    assert_relative_quat(bone_rotation(&app, rig.spine), spine_steady);
}

#[test]
fn literal_same_tick_reevaluation_is_bit_stable() {
    let input = live_input(Vec3::new(0.05, 0.01, -0.02), Vec3::new(0.0, 0.03, -0.04));
    let mut app = build_rig(true, true, input);
    let rig = rig_of(&app);

    tick(&mut app, FRAME_MILLIS);
    let head_first = head_world_position(&app, rig.head);
    let spine_first = bone_rotation(&app, rig.spine);

    // Re-run at the SAME clock instant: the system must reproduce the exact
    // same result rather than stacking another lean delta.
    tick(&mut app, FRAME_MILLIS);
    assert_relative_vec(head_world_position(&app, rig.head), head_first);
    assert_relative_quat(bone_rotation(&app, rig.spine), spine_first);
}

#[test]
fn large_lean_does_not_become_the_next_animation_base() {
    let mut app = build_rig(
        false,
        false,
        live_input(Vec3::new(0.3, 0.0, 0.0), Vec3::ZERO),
    );
    let rig = rig_of(&app);
    tick(&mut app, FRAME_MILLIS);
    let steady = head_world_position(&app, rig.head);
    for frame in 2..=120 {
        tick(&mut app, FRAME_MILLIS * frame);
        assert_relative_vec(head_world_position(&app, rig.head), steady);
    }
    *app.world_mut()
        .get_mut::<BodyTrackingPositionInput>(rig.root)
        .unwrap() = Default::default();
    tick(&mut app, FRAME_MILLIS * 121);
    assert_relative_quat(bone_rotation(&app, rig.spine), Quat::IDENTITY);
    assert_relative_quat(bone_rotation(&app, rig.chest), Quat::IDENTITY);
}

#[test]
fn lean_uses_model_axes_with_rotated_root_and_authored_joint_frames() {
    let input = live_input(Vec3::new(0.08, 0.0, 0.035), Vec3::ZERO);
    let mut baseline = build_rig(true, true, input);
    tick(&mut baseline, FRAME_MILLIS);
    let expected = head_world_position(&baseline, rig_of(&baseline).head);
    for yaw in [0.6, std::f32::consts::PI] {
        let mut app = build_rig(true, true, input);
        let rig = rig_of(&app);
        let placement = Quat::from_rotation_y(yaw);
        app.world_mut()
            .get_mut::<Transform>(rig.root)
            .unwrap()
            .rotation = placement;
        *app.world_mut()
            .get_mut::<GlobalTransform>(rig.root)
            .unwrap() = GlobalTransform::from(Transform::from_rotation(placement));
        let mut parent = *app.world().get::<GlobalTransform>(rig.root).unwrap();
        for entity in [
            Some(rig.spine),
            Some(rig.chest),
            rig.upper_chest,
            rig.neck,
            Some(rig.head),
        ]
        .into_iter()
        .flatten()
        {
            let mut local = *app.world().get::<Transform>(entity).unwrap();
            local.rotation = Quat::from_rotation_y(0.4);
            let global = parent.mul_transform(local);
            app.world_mut().entity_mut(entity).insert((
                local,
                global,
                RestTransform(local),
                RestGlobalTransform(global),
            ));
            parent = global;
        }
        for frame in 1..=20 {
            tick(&mut app, FRAME_MILLIS * frame);
            assert_relative_vec(head_world_position(&app, rig.head), placement * expected);
        }
    }
}

#[test]
fn rotation_only_semantics_are_preserved_without_position_input() {
    let mut app = build_rig(true, true, BodyTrackingPositionInput::default());
    let rig = rig_of(&app);
    app.world_mut()
        .entity_mut(rig.root)
        .remove::<BodyTrackingPositionInput>();
    tick(&mut app, FRAME_MILLIS);

    for entity in [rig.spine, rig.chest, rig.head] {
        assert_relative_quat(bone_rotation(&app, entity), Quat::IDENTITY);
    }
    assert_relative_vec(root_translation(&app, rig.root), Vec3::ZERO);
}

#[test]
fn inactive_channels_go_inert_without_breaking_the_pose() {
    let mut inactive = live_input(Vec3::new(0.06, 0.0, 0.0), Vec3::new(0.0, 0.05, 0.0));
    inactive.active = false;
    let mut app = build_rig(true, true, inactive);
    let rig = rig_of(&app);
    tick(&mut app, FRAME_MILLIS);

    assert_relative_vec(root_translation(&app, rig.root), Vec3::ZERO);
    for entity in [rig.spine, rig.chest, rig.head] {
        assert_relative_quat(bone_rotation(&app, entity), Quat::IDENTITY);
    }
}

#[test]
fn missing_optional_bones_degrade_safely_with_finite_output() {
    // Only spine + chest available; lean redistributes over two bones.
    let mut app = build_rig(
        false,
        false,
        live_input(Vec3::new(0.06, 0.0, 0.0), Vec3::ZERO),
    );
    tick(&mut app, FRAME_MILLIS);
    let rig = rig_of(&app);

    assert!(bone_rotation(&app, rig.spine).is_finite());
    assert!(bone_rotation(&app, rig.chest).is_finite());
    assert!(bone_rotation(&app, rig.head).is_finite());
    let head_position = head_world_position(&app, rig.head);
    assert!(head_position.x > 0.005);
    assert!(head_position.is_finite());
}

#[test]
fn output_is_deterministic_across_identical_runs() {
    let run = || -> (Vec3, Quat, Vec3) {
        let mut app = build_rig(
            true,
            true,
            live_input(Vec3::new(0.04, 0.02, -0.03), Vec3::new(0.0, 0.02, -0.03)),
        );
        let rig = rig_of(&app);
        tick(&mut app, FRAME_MILLIS);
        let _ = head_world_position(&app, rig.head);
        tick(&mut app, FRAME_MILLIS);
        (
            head_world_position(&app, rig.head),
            bone_rotation(&app, rig.spine),
            root_translation(&app, rig.root),
        )
    };
    let (_, spine_a, _) = run();
    let (_, spine_b, _) = run();
    assert_relative_quat(spine_a, spine_b);
}

#[test]
fn large_and_non_finite_offsets_stay_bounded_and_finite() {
    let huge_head = Vec3::new(f32::NAN, 5.0, -50.0);
    let huge_body = Vec3::new(100.0, f32::INFINITY, 0.0);
    let mut app = build_rig(true, true, live_input(huge_head, huge_body));
    tick(&mut app, FRAME_MILLIS);
    let rig = rig_of(&app);

    let translation = root_translation(&app, rig.root);
    assert!(translation.is_finite());
    assert!(translation.length() <= 0.25 + EPSILON);
    assert!(head_world_position(&app, rig.head).is_finite());
}
