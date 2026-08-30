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
use bevy_vrm1::vrm::body_tracking::{apply_direct_body_position, apply_direct_body_tracking};

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
        .insert_resource(TimeUpdateStrategy::ManualInstant(instant_at(0)))
        .add_systems(
            PostUpdate,
            (apply_direct_body_tracking, apply_direct_body_position).chain(),
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
