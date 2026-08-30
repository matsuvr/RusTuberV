// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Integration tests for avatar unload cleanup and stale control rejection.

use bevy::asset::AssetApp;
use bevy::prelude::*;
use bevy_vrm1::prelude::*;

use vtuber_avatar::bind::BindTriggered;
use vtuber_avatar::binding::{AvatarBinding, bind_humanoid_bones};
use vtuber_avatar::lifecycle::{
    ActiveAvatar, AvatarLifecycle, AvatarLifecycleState, LoadAvatarRequest, LoadAvatarResult,
    ReplaceAvatarRequest, ReplaceAvatarResult, UnloadAvatarRequest, UnloadAvatarResult,
    apply_avatar_request_events,
};
use vtuber_avatar::unload::{
    ActiveControlFrame, ControlFrameError, apply_active_control_frame, despawn_unloading_avatar,
};
use vtuber_core::types::{
    AvatarControlFrame, ExpressionCoefficients, FrameSeq, HeadPose, MonoTimeNs, TrackingState,
};

fn dummy_frame() -> AvatarControlFrame {
    AvatarControlFrame {
        source_seq: FrameSeq(1),
        captured_at: MonoTimeNs(0),
        produced_at: MonoTimeNs(0),
        confidence: 1.0,
        state: TrackingState::Tracking,
        head: HeadPose::default(),
        head_translation: vtuber_core::HeadTranslationSignal::UNAVAILABLE,
        gaze: vtuber_core::GazeSignal::UNAVAILABLE,
        expressions: ExpressionCoefficients::default(),
        detailed_face: None,
    }
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

fn spawn_avatar_root(app: &mut App, head: Entity) -> Entity {
    let root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            HeadBoneEntity(head),
        ))
        .id();
    // Make the head a descendant of the root so recursive despawn removes it.
    app.world_mut().entity_mut(head).insert(ChildOf(root));
    root
}

fn load_root(app: &mut App, root: Entity) {
    app.world_mut()
        .resource_mut::<Messages<LoadAvatarRequest>>()
        .write(LoadAvatarRequest { root });
    app.update();
}

fn bind_root(app: &mut App, root: Entity) {
    app.world_mut().entity_mut(root).insert(BindTriggered);
    let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
    lifecycle.start_binding(root);
    // Run bind_humanoid_bones to create AvatarBinding.
    app.update();
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();
}

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<bevy_vrm1::prelude::VrmAsset>()
        .init_resource::<AvatarLifecycle>()
        .init_resource::<ActiveControlFrame>()
        .add_message::<LoadAvatarRequest>()
        .add_message::<LoadAvatarResult>()
        .add_message::<UnloadAvatarRequest>()
        .add_message::<UnloadAvatarResult>()
        .add_message::<ReplaceAvatarRequest>()
        .add_message::<ReplaceAvatarResult>()
        .add_systems(
            Update,
            (
                apply_avatar_request_events,
                despawn_unloading_avatar,
                bind_humanoid_bones,
            )
                .chain(),
        );
    app
}

#[test]
fn avatar_unload_cleanup() {
    let mut app = test_app();

    // --- Load and bind avatar A ---
    let head_a = spawn_bone(&mut app);
    let root_a = spawn_avatar_root(&mut app, head_a);
    load_root(&mut app, root_a);
    bind_root(&mut app, root_a);

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);
    assert_eq!(lifecycle.active_root(), Some(root_a));
    let generation_a = lifecycle.current_generation();
    assert_ne!(generation_a, vtuber_avatar::lifecycle::AvatarGeneration(0));

    let binding_a = app
        .world()
        .get::<AvatarBinding>(root_a)
        .copied()
        .expect("avatar A should be bound");
    assert_eq!(binding_a.generation, generation_a);

    // Current frame applies to avatar A.
    let mut active = ActiveControlFrame {
        generation: generation_a,
        frame: Some(dummy_frame()),
    };
    let frame_ref = apply_active_control_frame(
        app.world().resource::<AvatarLifecycle>(),
        &active,
        app.world().get::<AvatarBinding>(root_a),
    )
    .expect("frame should apply to avatar A");
    assert!(frame_ref.is_some());

    // --- Unload avatar A ---
    app.world_mut()
        .resource_mut::<Messages<UnloadAvatarRequest>>()
        .write(UnloadAvatarRequest);
    app.update();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::NoAvatar);
    assert!(lifecycle.active_root().is_none());
    assert!(!lifecycle.has_active_generation());

    // No VRM/avatar entity remains in the world.
    assert!(!app.world().entities().contains(root_a));
    assert!(!app.world().entities().contains(head_a));
    let mut active_query = app
        .world_mut()
        .query_filtered::<Entity, With<ActiveAvatar>>();
    assert_eq!(active_query.iter(app.world()).count(), 0);

    // Control cache is cleared by lifecycle change.
    assert!(app.world().resource::<ActiveControlFrame>().frame.is_none());

    // --- Double unload is idempotent ---
    app.world_mut()
        .resource_mut::<Messages<UnloadAvatarRequest>>()
        .write(UnloadAvatarRequest);
    app.update();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::NoAvatar);

    // --- Load avatar B ---
    let head_b = spawn_bone(&mut app);
    let root_b = spawn_avatar_root(&mut app, head_b);
    load_root(&mut app, root_b);
    bind_root(&mut app, root_b);

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);
    assert_eq!(lifecycle.active_root(), Some(root_b));
    let generation_b = lifecycle.current_generation();
    assert_ne!(generation_b, generation_a);

    // Simulate an old control frame arriving after avatar A was unloaded.
    active.generation = generation_a;
    active.frame = Some(dummy_frame());

    let binding_b = app
        .world()
        .get::<AvatarBinding>(root_b)
        .copied()
        .expect("avatar B should be bound");
    assert_eq!(binding_b.generation, generation_b);

    let result = apply_active_control_frame(
        app.world().resource::<AvatarLifecycle>(),
        &active,
        app.world().get::<AvatarBinding>(root_b),
    );
    assert!(
        matches!(
            result,
            Err(ControlFrameError::StaleGeneration {
                frame_generation,
                binding_generation,
            }) if frame_generation == generation_a && binding_generation == generation_b
        ),
        "stale frame targeting avatar A should be rejected for avatar B, got {result:?}"
    );
}
