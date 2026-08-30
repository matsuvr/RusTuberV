// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Direct-pose `bevy_vrm1::BodyTracking` integration tests.

use bevy::prelude::*;
use bevy_vrm1::prelude::*;
use bevy_vrm1::vrm::body_tracking::apply_direct_body_tracking;

const EPSILON: f32 = 1.0e-4;

#[derive(Clone, Copy)]
struct Chain {
    root: Entity,
    head: Entity,
    neck: Option<Entity>,
    upper_chest: Option<Entity>,
    chest: Option<Entity>,
    spine: Option<Entity>,
    intermediate: Option<Entity>,
    left_eye: Entity,
}

fn instant_profile() -> BodyTrackingProfile {
    BodyTrackingProfile {
        bone_half_lives: BodyBoneHalfLives {
            head_seconds: 0.0,
            neck_seconds: 0.0,
            upper_chest_seconds: 0.0,
            chest_seconds: 0.0,
            spine_seconds: 0.0,
        },
        ..Default::default()
    }
}

fn spawn_bone(app: &mut App, parent: Entity, rotation: Quat) -> Entity {
    app.world_mut()
        .spawn((
            Transform::from_rotation(rotation),
            GlobalTransform::IDENTITY,
            RestTransform(Transform::IDENTITY),
            RestGlobalTransform(GlobalTransform::IDENTITY),
            ChildOf(parent),
        ))
        .id()
}

fn build_app(with_upper_chest: bool, with_intermediate: bool, animated_head: Quat) -> (App, Chain) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_systems(PostUpdate, apply_direct_body_tracking);

    let root = app
        .world_mut()
        .spawn((
            Vrm,
            Transform::IDENTITY,
            GlobalTransform::IDENTITY,
            BodyTracking::default(),
            BodyTrackingPoseInput {
                yaw_radians: 1.0,
                pitch_radians: 0.0,
                roll_radians: 0.0,
                weight: 1.0,
                active: true,
            },
            instant_profile(),
        ))
        .id();
    let spine = spawn_bone(&mut app, root, Quat::IDENTITY);
    let intermediate = with_intermediate.then(|| spawn_bone(&mut app, spine, Quat::IDENTITY));
    let chest_parent = intermediate.unwrap_or(spine);
    let chest = spawn_bone(&mut app, chest_parent, Quat::IDENTITY);
    let upper_chest = with_upper_chest.then(|| spawn_bone(&mut app, chest, Quat::IDENTITY));
    let neck_parent = upper_chest.unwrap_or(chest);
    let neck = spawn_bone(&mut app, neck_parent, Quat::IDENTITY);
    let head = spawn_bone(&mut app, neck, animated_head);
    let left_eye_rest = Quat::from_rotation_z(0.07);
    let left_eye = spawn_bone(&mut app, head, left_eye_rest);

    let mut root_entity = app.world_mut().entity_mut(root);
    root_entity.insert((
        HeadBoneEntity(head),
        NeckBoneEntity(neck),
        ChestBoneEntity(chest),
        SpineBoneEntity(spine),
        LeftEyeBoneEntity(left_eye),
    ));
    if let Some(upper_chest) = upper_chest {
        root_entity.insert(UpperChestBoneEntity(upper_chest));
    }

    (
        app,
        Chain {
            root,
            head,
            neck: Some(neck),
            upper_chest,
            chest: Some(chest),
            spine: Some(spine),
            intermediate,
            left_eye,
        },
    )
}

fn local_yaw(app: &App, entity: Entity) -> f32 {
    app.world()
        .get::<Transform>(entity)
        .unwrap()
        .rotation
        .to_euler(EulerRot::YXZ)
        .0
}

#[test]
fn large_yaw_uses_all_five_bones_with_documented_weights() {
    let (mut app, chain) = build_app(true, false, Quat::IDENTITY);
    app.update();

    assert!((local_yaw(&app, chain.head) - 0.42).abs() < EPSILON);
    assert!((local_yaw(&app, chain.neck.unwrap()) - 0.23).abs() < EPSILON);
    assert!((local_yaw(&app, chain.upper_chest.unwrap()) - 0.17).abs() < EPSILON);
    assert!((local_yaw(&app, chain.chest.unwrap()) - 0.11).abs() < EPSILON);
    assert!((local_yaw(&app, chain.spine.unwrap()) - 0.07).abs() < EPSILON);
}

#[test]
fn missing_upper_chest_renormalizes_and_does_not_panic() {
    let (mut app, chain) = build_app(false, false, Quat::IDENTITY);
    app.update();

    assert!(chain.upper_chest.is_none());
    for entity in [
        chain.head,
        chain.neck.unwrap(),
        chain.chest.unwrap(),
        chain.spine.unwrap(),
    ] {
        assert!(
            app.world()
                .get::<Transform>(entity)
                .unwrap()
                .rotation
                .is_finite()
        );
    }
    let remaining_sum = 0.42 + 0.23 + 0.11 + 0.07;
    assert!((local_yaw(&app, chain.head) - 0.42 / remaining_sum).abs() < EPSILON);
}

#[test]
fn intermediate_node_receives_fresh_global_transform() {
    let (mut app, chain) = build_app(true, true, Quat::IDENTITY);
    app.update();

    let intermediate = chain.intermediate.unwrap();
    let spine_global = app
        .world()
        .get::<GlobalTransform>(chain.spine.unwrap())
        .unwrap();
    let intermediate_global = app.world().get::<GlobalTransform>(intermediate).unwrap();
    assert!(
        spine_global
            .rotation()
            .angle_between(intermediate_global.rotation())
            < EPSILON
    );
    assert!(
        app.world()
            .get::<GlobalTransform>(chain.head)
            .unwrap()
            .rotation()
            .is_finite()
    );
}

#[test]
fn animation_base_is_preserved_and_tracking_delta_does_not_accumulate() {
    let animated = Quat::from_rotation_x(0.2);
    let (mut app, chain) = build_app(false, false, animated);
    app.world_mut()
        .get_mut::<BodyTrackingPoseInput>(chain.root)
        .unwrap()
        .yaw_radians = 0.1;
    app.update();

    let first = app.world().get::<Transform>(chain.head).unwrap().rotation;
    let expected = animated * Quat::from_rotation_y(0.1 * (0.65 / (0.65 + 0.35)));
    assert!(first.angle_between(expected) < EPSILON);

    app.update();
    let second = app.world().get::<Transform>(chain.head).unwrap().rotation;
    assert!(first.angle_between(second) < EPSILON);
}

#[test]
fn body_tracking_does_not_write_eye_bones() {
    let (mut app, chain) = build_app(true, false, Quat::IDENTITY);
    let before = app
        .world()
        .get::<Transform>(chain.left_eye)
        .unwrap()
        .rotation;
    app.update();
    let after = app
        .world()
        .get::<Transform>(chain.left_eye)
        .unwrap()
        .rotation;
    assert!(before.angle_between(after) < EPSILON);
}
