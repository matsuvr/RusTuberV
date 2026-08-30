// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for the hips-relative virtual hand source (Issue #168).
//!
//! Verifies that the existing arm-pose compositor stays the single Transform
//! writer while preferring the per-frame `DynamicArmTargets` resolution, and
//! that repeated runs and generation changes never accumulate deltas or leak
//! state between avatar generations.

use bevy::prelude::*;
use std::collections::HashMap;
use vtuber_avatar::{
    ActiveAvatar, AvatarBinding, AvatarGeneration, DefaultArmPose, DynamicArmTargets,
    ResolvedArmPose, apply_default_arm_pose,
};

#[derive(Clone, Copy)]
struct Chain {
    upper: Entity,
    lower: Entity,
}

fn spawn_child(app: &mut App, parent: Entity, transform: Transform) -> Entity {
    app.world_mut()
        .spawn((transform, GlobalTransform::IDENTITY, ChildOf(parent)))
        .id()
}

fn pose_for(chain: &Chain, delta: Quat) -> ResolvedArmPose {
    ResolvedArmPose {
        upper_arm: chain.upper,
        lower_arm: chain.lower,
        upper_arm_delta: delta,
        lower_arm_delta: Quat::IDENTITY,
        shoulder: None,
        fingers: Default::default(),
    }
}

/// Spawns a minimal avatar with a static default pose and an (empty) dynamic
/// target component, mirroring what avatar binding inserts.
fn spawn_avatar(app: &mut App, generation: AvatarGeneration) -> Chain {
    let root = app
        .world_mut()
        .spawn((ActiveAvatar, GlobalTransform::IDENTITY))
        .id();
    let upper = spawn_child(
        app,
        root,
        Transform::from_translation(Vec3::new(0.4, 1.2, 0.0)),
    );
    let lower = spawn_child(
        app,
        upper,
        Transform::from_translation(Vec3::new(0.7, 0.0, 0.0)),
    );
    let chain = Chain { upper, lower };
    let static_pose = pose_for(&chain, Quat::from_rotation_z(0.1));
    app.world_mut().entity_mut(root).insert((
        AvatarBinding::head_only(root, root, generation),
        DefaultArmPose {
            generation,
            left: Some(static_pose),
            right: None,
        },
        DynamicArmTargets {
            generation: Some(generation),
            source_seq: None,
            left: None,
            right: None,
        },
    ));
    chain
}

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_systems(PostUpdate, apply_default_arm_pose);
    app
}

fn upper_rotation(app: &App, chain: &Chain) -> Quat {
    app.world().get::<Transform>(chain.upper).unwrap().rotation
}

#[test]
fn fresh_dynamic_targets_override_the_static_default_pose() {
    let mut app = build_app();
    let generation = AvatarGeneration(3);
    let chain = spawn_avatar(&mut app, generation);

    // No dynamic resolution yet: static default pose applies.
    app.update();
    let static_rotation = upper_rotation(&app, &chain);
    assert!(
        static_rotation.angle_between(Quat::IDENTITY * Quat::from_rotation_z(0.1)) < 1e-5,
        "fallback path must keep applying the static pose"
    );

    // A dynamic resolution for this generation becomes authoritative.
    let dynamic = pose_for(&chain, Quat::from_rotation_x(0.4));
    let targets = app
        .world_mut()
        .query_filtered::<Entity, With<ActiveAvatar>>()
        .single(app.world())
        .unwrap();
    app.world_mut()
        .entity_mut(targets)
        .insert(DynamicArmTargets {
            generation: Some(generation),
            source_seq: Some(vtuber_core_placeholder_seq()),
            left: Some(dynamic),
            right: None,
        });

    app.update();
    let applied = upper_rotation(&app, &chain);
    let expected = Quat::IDENTITY * Quat::from_rotation_x(0.4);
    assert!(
        applied.angle_between(expected) < 1e-5,
        "dynamic target must override the static default"
    );
}

#[test]
fn repeated_updates_with_identical_targets_do_not_accumulate() {
    let mut app = build_app();
    let generation = AvatarGeneration(5);
    let chain = spawn_avatar(&mut app, generation);
    let dynamic = pose_for(&chain, Quat::from_rotation_x(0.4));
    let targets = app
        .world_mut()
        .query_filtered::<Entity, With<ActiveAvatar>>()
        .single(app.world())
        .unwrap();
    app.world_mut()
        .entity_mut(targets)
        .insert(DynamicArmTargets {
            generation: Some(generation),
            source_seq: None,
            left: Some(dynamic),
            right: None,
        });

    app.update();
    let first = rotation_map(&app, &chain);
    app.update();
    let second = rotation_map(&app, &chain);

    assert_eq!(first, second, "identical input must not accumulate");
}

#[test]
fn stale_generation_targets_are_ignored_in_favor_of_the_static_pose() {
    let mut app = build_app();
    let generation = AvatarGeneration(9);
    let chain = spawn_avatar(&mut app, generation);

    // Targets tagged with a previous generation must never apply.
    let stale = pose_for(&chain, Quat::from_rotation_x(0.4));
    let targets = app
        .world_mut()
        .query_filtered::<Entity, With<ActiveAvatar>>()
        .single(app.world())
        .unwrap();
    app.world_mut()
        .entity_mut(targets)
        .insert(DynamicArmTargets {
            generation: Some(AvatarGeneration(generation.0 - 1)),
            source_seq: None,
            left: Some(stale),
            right: None,
        });

    app.update();
    let applied = upper_rotation(&app, &chain);
    assert!(
        applied.angle_between(Quat::IDENTITY * Quat::from_rotation_z(0.1)) < 1e-5,
        "generation-isolated compositor must fall back to the static pose"
    );
}

fn rotation_map(app: &App, chain: &Chain) -> HashMap<Entity, Quat> {
    let mut map = HashMap::new();
    for entity in [chain.upper, chain.lower] {
        let rotation = app.world().get::<Transform>(entity).unwrap().rotation;
        map.insert(entity, rotation);
    }
    map
}

fn vtuber_core_placeholder_seq() -> vtuber_core::FrameSeq {
    vtuber_core::FrameSeq(0)
}
