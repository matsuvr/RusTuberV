// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Edge-case model synthetic tests.
//!
//! These tests exercise the avatar lifecycle with models that have unusual
//! bone or expression configurations:
//!
//! - No head bone → typed error, lifecycle enters `Failed`.
//! - No neck bone → binding succeeds, `BonePresence.neck == false`.
//! - No expression map → binding succeeds, empty expression capabilities.
//! - No eye bones → binding succeeds, gaze mode reflects missing eyes.
//!
//! Each test constructs a synthetic Bevy world that mimics what `bevy_vrm1`
//! would produce for the corresponding model configuration.

use bevy::asset::AssetApp;
use bevy::prelude::*;
use bevy_vrm1::prelude::*;

use vtuber_avatar::bind::BindTriggered;
use vtuber_avatar::binding::{AvatarBinding, bind_humanoid_bones};
use vtuber_avatar::capabilities::{BlinkMode, MouthMode, SelectedGazeBackend};
use vtuber_avatar::lifecycle::{
    ActiveAvatar, AvatarLifecycle, AvatarLifecycleState, LoadAvatarRequest, LoadAvatarResult,
    ReplaceAvatarRequest, ReplaceAvatarResult, UnloadAvatarRequest, UnloadAvatarResult,
    apply_avatar_request_events,
};
use vtuber_avatar::unload::{ActiveControlFrame, despawn_unloading_avatar};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<bevy_vrm1::prelude::VrmAsset>()
        .init_resource::<AvatarLifecycle>()
        .init_resource::<ActiveControlFrame>()
        .init_resource::<vtuber_avatar::GlobalBodyTrackingProfile>()
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

fn load_and_bind(app: &mut App, root: Entity) {
    app.world_mut()
        .resource_mut::<Messages<LoadAvatarRequest>>()
        .write(LoadAvatarRequest { root });
    app.update();

    app.world_mut().entity_mut(root).insert(Initialized);
    app.world_mut().entity_mut(root).insert(BindTriggered);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .start_binding(root);
    app.update();
}

// ---------------------------------------------------------------------------
// No-head model
// ---------------------------------------------------------------------------

/// A model without a head bone must produce a typed error and enter `Failed`.
#[test]
fn no_head_model_fails_with_typed_error() {
    let mut app = test_app();

    // Root without HeadBoneEntity.
    let root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            ActiveAvatar,
            BindTriggered,
        ))
        .id();

    load_and_bind(&mut app, root);

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Failed);
    assert!(lifecycle.active_root().is_none());
    assert!(lifecycle.capabilities().is_none());
    assert!(
        app.world().get::<AvatarBinding>(root).is_none(),
        "no binding should exist for a headless model"
    );
    assert!(
        !app.world().entity(root).contains::<ActiveAvatar>(),
        "ActiveAvatar marker should be removed on failure"
    );
}

/// After a headless-model failure, a new load must succeed.
#[test]
fn no_head_model_recovery_with_valid_model() {
    let mut app = test_app();

    // First model: no head.
    let bad_root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            ActiveAvatar,
            BindTriggered,
        ))
        .id();

    load_and_bind(&mut app, bad_root);
    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Failed
    );

    // Second model: valid head.
    let head = spawn_bone(&mut app);
    let good_root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            HeadBoneEntity(head),
        ))
        .id();
    app.world_mut().entity_mut(head).insert(ChildOf(good_root));

    load_and_bind(&mut app, good_root);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);
    assert_eq!(lifecycle.active_root(), Some(good_root));
}

// ---------------------------------------------------------------------------
// No-neck model
// ---------------------------------------------------------------------------

/// A model without a neck bone binds successfully with `neck == None`.
#[test]
fn no_neck_model_binds_successfully() {
    let mut app = test_app();

    let head = spawn_bone(&mut app);
    let root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            HeadBoneEntity(head),
            // No NeckBoneEntity.
        ))
        .id();
    app.world_mut().entity_mut(head).insert(ChildOf(root));

    load_and_bind(&mut app, root);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);

    let binding = app
        .world()
        .get::<AvatarBinding>(root)
        .expect("binding should succeed without neck");
    assert!(binding.neck.is_none());
    assert_eq!(binding.head, head);

    let caps = lifecycle
        .capabilities()
        .cloned()
        .expect("capabilities should be populated");
    assert!(!caps.bones.neck, "BonePresence.neck should be false");
    assert!(caps.bones.head);
}

// ---------------------------------------------------------------------------
// No-expression model
// ---------------------------------------------------------------------------

/// A model without an expression map binds successfully with empty expression
/// capabilities.
#[test]
fn no_expression_model_binds_with_empty_capabilities() {
    let mut app = test_app();

    let head = spawn_bone(&mut app);
    let root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            HeadBoneEntity(head),
            // No ExpressionEntityMap.
        ))
        .id();
    app.world_mut().entity_mut(head).insert(ChildOf(root));

    load_and_bind(&mut app, root);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);

    let caps = lifecycle
        .capabilities()
        .cloned()
        .expect("capabilities should be populated even without expressions");
    assert_eq!(caps.blink, BlinkMode::None);
    assert_eq!(caps.mouth, MouthMode::None);
    assert!(!caps.look_directions.any());
    assert!(caps.unknown_expressions.is_empty());
    assert!(!caps.is_fully_supported());
}

// ---------------------------------------------------------------------------
// No-eye-bones model
// ---------------------------------------------------------------------------

/// A model without eye bones binds successfully. Gaze mode falls back to
/// expression-only or none, depending on look-direction expressions.
#[test]
fn no_eye_bones_model_binds_with_degraded_gaze() {
    let mut app = test_app();

    let head = spawn_bone(&mut app);
    let root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            HeadBoneEntity(head),
            // No LeftEyeBoneEntity or RightEyeBoneEntity.
        ))
        .id();
    app.world_mut().entity_mut(head).insert(ChildOf(root));

    load_and_bind(&mut app, root);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);

    let caps = lifecycle
        .capabilities()
        .cloned()
        .expect("capabilities should be populated");
    assert!(!caps.bones.left_eye);
    assert!(!caps.bones.right_eye);

    // Without eye bones and without look-direction expressions, gaze is None.
    assert_eq!(caps.gaze_backend, SelectedGazeBackend::None);
}

// ---------------------------------------------------------------------------
// Minimal capable model (head + blink only)
// ---------------------------------------------------------------------------

/// A model with just a head bone and blink expression satisfies the MVP
/// capability gate.
#[test]
fn minimal_capable_model_head_and_blink() {
    let mut app = test_app();

    let head = spawn_bone(&mut app);
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
    app.world_mut().entity_mut(head).insert(ChildOf(root));

    load_and_bind(&mut app, root);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);

    let caps = lifecycle
        .capabilities()
        .cloned()
        .expect("capabilities should be populated");
    assert!(caps.bones.head);
    assert!(!caps.bones.neck);
    assert!(!caps.bones.left_eye);
    assert!(!caps.bones.right_eye);
    assert_eq!(caps.blink, BlinkMode::None);
    assert_eq!(caps.mouth, MouthMode::None);
    assert_eq!(caps.gaze_backend, SelectedGazeBackend::None);
    assert!(!caps.spring_bone);
    assert!(!caps.is_fully_supported());
    assert!(caps.summary().contains("Bones: head"));
}

// ---------------------------------------------------------------------------
// Full-featured model
// ---------------------------------------------------------------------------

/// A model with all optional bones and a complete expression map produces
/// a fully-supported capability snapshot.
#[test]
fn full_featured_model_all_bones_and_expressions() {
    let mut app = test_app();

    let head = spawn_bone(&mut app);
    let neck = spawn_bone(&mut app);
    let upper_chest = spawn_bone(&mut app);
    let chest = spawn_bone(&mut app);
    let spine = spawn_bone(&mut app);
    let left_eye = spawn_bone(&mut app);
    let right_eye = spawn_bone(&mut app);

    let root = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            HeadBoneEntity(head),
            NeckBoneEntity(neck),
            UpperChestBoneEntity(upper_chest),
            ChestBoneEntity(chest),
            SpineBoneEntity(spine),
            LeftEyeBoneEntity(left_eye),
            RightEyeBoneEntity(right_eye),
        ))
        .id();

    for bone in [head, neck, upper_chest, chest, spine, left_eye, right_eye] {
        app.world_mut().entity_mut(bone).insert(ChildOf(root));
    }

    load_and_bind(&mut app, root);
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);

    let binding = app
        .world()
        .get::<AvatarBinding>(root)
        .expect("full model should bind");
    assert_eq!(binding.neck, Some(neck));
    assert_eq!(binding.upper_chest, Some(upper_chest));
    assert_eq!(binding.chest, Some(chest));
    assert_eq!(binding.spine, Some(spine));
    assert_eq!(binding.left_eye, Some(left_eye));
    assert_eq!(binding.right_eye, Some(right_eye));

    let caps = lifecycle
        .capabilities()
        .cloned()
        .expect("capabilities should be populated");
    assert!(caps.bones.head);
    assert!(caps.bones.neck);
    assert!(caps.bones.left_eye);
    assert!(caps.bones.right_eye);
    assert!(caps.bones.upper_chest);
    assert!(caps.bones.chest);
    assert!(caps.bones.spine);
    assert_eq!(caps.gaze_backend, SelectedGazeBackend::Bone);
}
