// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Full-lifecycle integration test.
//!
//! Exercises the complete avatar lifecycle through the ECS system pipeline:
//! load → Initialized → binding → Ready → replace → unload → failure.
//!
//! No real VRM asset is loaded; the test synthesises the components that
//! `bevy_vrm1` would insert (`Initialized`, bone entities, `ExpressionEntityMap`)
//! to drive the avatar systems through every state transition.

use bevy::asset::AssetApp;
use bevy::prelude::*;
use bevy_vrm1::prelude::*;
use bevy_vrm1::vrm::spring_bone::{
    SpringCenterNode, SpringColliders, SpringJointState, VrmSpringBonePlugin,
};

use vtuber_avatar::bind::BindTriggered;
use vtuber_avatar::binding::{AvatarBinding, bind_humanoid_bones};
use vtuber_avatar::lifecycle::{
    ActiveAvatar, AvatarGeneration, AvatarLifecycle, AvatarLifecycleState, LoadAvatarRequest,
    LoadAvatarResult, ReplaceAvatarRequest, ReplaceAvatarResult, UnloadAvatarRequest,
    UnloadAvatarResult, apply_avatar_request_events,
};
use vtuber_avatar::unload::{
    ActiveControlFrame, apply_active_control_frame, despawn_unloading_avatar,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .add_plugins(VrmSpringBonePlugin)
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

/// Spawns a synthetic VRM root with the given bone entities and optional
/// expression map, mimicking what `bevy_vrm1` produces after initialization.
fn spawn_avatar_root(
    app: &mut App,
    head: Entity,
    neck: Option<Entity>,
    left_eye: Option<Entity>,
    right_eye: Option<Entity>,
) -> Entity {
    let mut entity = app.world_mut().spawn((
        Transform::default(),
        GlobalTransform::default(),
        Visibility::Hidden,
        VrmHandle(Handle::default()),
        HeadBoneEntity(head),
    ));

    if let Some(neck) = neck {
        entity.insert(NeckBoneEntity(neck));
    }
    if let Some(le) = left_eye {
        entity.insert(LeftEyeBoneEntity(le));
    }
    if let Some(re) = right_eye {
        entity.insert(RightEyeBoneEntity(re));
    }

    let root = entity.id();

    // Make bones descendants so recursive despawn cleans them up.
    app.world_mut().entity_mut(head).insert(ChildOf(root));
    if let Some(neck) = neck {
        app.world_mut().entity_mut(neck).insert(ChildOf(root));
    }
    if let Some(le) = left_eye {
        app.world_mut().entity_mut(le).insert(ChildOf(root));
    }
    if let Some(re) = right_eye {
        app.world_mut().entity_mut(re).insert(ChildOf(root));
    }

    root
}

fn load_root(app: &mut App, root: Entity) {
    app.world_mut()
        .resource_mut::<Messages<LoadAvatarRequest>>()
        .write(LoadAvatarRequest { root });
    app.update();
}

/// Loads a root and drives it through Initialized → binding in one shot.
fn load_and_bind(app: &mut App, root: Entity) {
    load_root(app, root);
    simulate_initialized_and_bind(app, root);
}

/// Simulates `bevy_vrm1` adding `Initialized` and then runs the bind system.
fn simulate_initialized_and_bind(app: &mut App, root: Entity) {
    app.world_mut().entity_mut(root).insert(Initialized);
    app.world_mut().entity_mut(root).insert(BindTriggered);

    let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
    lifecycle.start_binding(root);

    app.update();
}

fn finish_ready(app: &mut App) {
    app.world_mut()
        .resource_mut::<AvatarLifecycle>()
        .finish_ready();
}

fn assert_ready(app: &App, expected_root: Entity) {
    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);
    assert_eq!(lifecycle.active_root(), Some(expected_root));
}

fn count_active_markers(app: &mut App) -> usize {
    app.world_mut()
        .query_filtered::<Entity, With<ActiveAvatar>>()
        .iter(app.world())
        .count()
}

fn count_spring_joint_states(app: &mut App) -> usize {
    app.world_mut()
        .query::<&SpringJointState>()
        .iter(app.world())
        .count()
}

#[derive(Clone, Copy, Debug)]
enum SyntheticGeneration {
    Vrm0,
    Vrm1,
}

fn spawn_generation_avatar_root(app: &mut App, generation: SyntheticGeneration) -> Entity {
    let head = spawn_bone(app);
    if matches!(generation, SyntheticGeneration::Vrm0) {
        let terminal_source_transform = Transform::from_translation(Vec3::Y);
        app.world_mut().entity_mut(head).insert((
            terminal_source_transform,
            GlobalTransform::from(terminal_source_transform),
        ));
    }
    let root = app
        .world_mut()
        .spawn((
            Transform::IDENTITY,
            GlobalTransform::IDENTITY,
            Visibility::Hidden,
            Vrm,
            VrmHandle(Handle::default()),
            VrmCoordinateBasis(match generation {
                SyntheticGeneration::Vrm0 => CoordinateBasis::Vrm0Y180,
                SyntheticGeneration::Vrm1 => CoordinateBasis::Vrm1Identity,
            }),
            HeadBoneEntity(head),
            ExpressionEntityMap(bevy::platform::collections::HashMap::new()),
        ))
        .id();

    let parent = if matches!(generation, SyntheticGeneration::Vrm0) {
        app.world_mut()
            .spawn((
                VrmBasisRoot,
                VrmCoordinateBasis(CoordinateBasis::Vrm0Y180)
                    .transform()
                    .expect("VRM 0.x synthetic basis exists"),
                GlobalTransform::IDENTITY,
                ChildOf(root),
            ))
            .id()
    } else {
        root
    };
    app.world_mut().entity_mut(head).insert(ChildOf(parent));
    app.world_mut().entity_mut(head).insert(SpringRoot {
        joints: SpringJoints(vec![head]),
        colliders: SpringColliders::default(),
        center_node: SpringCenterNode::default(),
        terminal_length: match generation {
            SyntheticGeneration::Vrm0 => Some(0.07),
            SyntheticGeneration::Vrm1 => None,
        },
    });
    root
}

fn assert_generation_state(app: &mut App, root: Entity, generation: SyntheticGeneration) {
    let expected_basis = match generation {
        SyntheticGeneration::Vrm0 => CoordinateBasis::Vrm0Y180,
        SyntheticGeneration::Vrm1 => CoordinateBasis::Vrm1Identity,
    };
    assert_eq!(
        app.world().get::<VrmCoordinateBasis>(root).unwrap().0,
        expected_basis
    );
    assert_eq!(count_active_markers(app), 1);
    let basis_count = app
        .world_mut()
        .query_filtered::<Entity, With<VrmBasisRoot>>()
        .iter(app.world())
        .count();
    assert_eq!(
        basis_count,
        if matches!(generation, SyntheticGeneration::Vrm0) {
            1
        } else {
            0
        }
    );
    assert_eq!(
        app.world_mut()
            .query::<&SpringRoot>()
            .iter(app.world())
            .count(),
        1
    );
    assert_eq!(
        count_spring_joint_states(app),
        if matches!(generation, SyntheticGeneration::Vrm0) {
            1
        } else {
            0
        },
        "VRM 0.x owns one finite terminal SpringJointState; VRM 1.0 has no legacy terminal"
    );
    assert_eq!(
        app.world_mut()
            .query::<&ExpressionEntityMap>()
            .iter(app.world())
            .count(),
        1
    );
}

fn assert_avatar_owned_state_empty(app: &mut App) {
    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::NoAvatar);
    assert!(lifecycle.active_root().is_none());
    assert!(lifecycle.pending_root().is_none());
    assert!(lifecycle.capabilities().is_none());
    assert!(!lifecycle.has_active_generation());
    assert_eq!(count_active_markers(app), 0);
    assert_eq!(app.world_mut().query::<&Vrm>().iter(app.world()).count(), 0);
    assert_eq!(
        app.world_mut()
            .query::<&VrmCoordinateBasis>()
            .iter(app.world())
            .count(),
        0
    );
    assert_eq!(
        app.world_mut()
            .query::<&VrmBasisRoot>()
            .iter(app.world())
            .count(),
        0
    );
    assert_eq!(
        app.world_mut()
            .query::<&SpringRoot>()
            .iter(app.world())
            .count(),
        0
    );
    assert_eq!(
        app.world_mut()
            .query::<&SpringJointState>()
            .iter(app.world())
            .count(),
        0
    );
    assert_eq!(
        app.world_mut()
            .query::<&ExpressionEntityMap>()
            .iter(app.world())
            .count(),
        0
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Walks the full lifecycle: load → bind → Ready → replace → new Ready → unload.
#[test]
fn full_lifecycle_load_replace_unload() {
    let mut app = test_app();

    // --- Avatar A: head + neck + eyes ---
    let head_a = spawn_bone(&mut app);
    let neck_a = spawn_bone(&mut app);
    let le_a = spawn_bone(&mut app);
    let re_a = spawn_bone(&mut app);
    let root_a = spawn_avatar_root(&mut app, head_a, Some(neck_a), Some(le_a), Some(re_a));

    load_root(&mut app, root_a);
    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Loading
    );
    assert_eq!(count_active_markers(&mut app), 1);

    simulate_initialized_and_bind(&mut app, root_a);
    finish_ready(&mut app);
    assert_ready(&app, root_a);

    let gen_a = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();
    let binding_a = app
        .world()
        .get::<AvatarBinding>(root_a)
        .copied()
        .expect("avatar A should be bound");
    assert_eq!(binding_a.generation, gen_a);
    assert_eq!(binding_a.neck, Some(neck_a));
    assert_eq!(binding_a.left_eye, Some(le_a));
    assert_eq!(binding_a.right_eye, Some(re_a));

    // Capabilities should reflect the full bone set.
    let caps = app
        .world()
        .resource::<AvatarLifecycle>()
        .capabilities()
        .cloned()
        .expect("capabilities should be populated");
    assert!(caps.bones.neck);
    assert!(caps.bones.left_eye);
    assert!(caps.bones.right_eye);

    // --- Replace with Avatar B: head only ---
    let head_b = spawn_bone(&mut app);
    let root_b = spawn_avatar_root(&mut app, head_b, None, None, None);

    app.world_mut()
        .resource_mut::<Messages<ReplaceAvatarRequest>>()
        .write(ReplaceAvatarRequest { root: root_b });
    app.update();

    // despawn_unloading_avatar runs in the same frame, so the lifecycle
    // transitions through Unloading → Loading for the pending root.
    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
    assert_eq!(lifecycle.active_root(), Some(root_b));

    // Avatar A's root should be despawned by despawn_unloading_avatar.
    assert!(!app.world().entities().contains(root_a));
    assert!(!app.world().entities().contains(head_a));

    let gen_b = lifecycle.current_generation();
    assert_ne!(gen_a, gen_b, "generation must change on replace");

    // Drive avatar B through binding.
    simulate_initialized_and_bind(&mut app, root_b);
    finish_ready(&mut app);
    assert_ready(&app, root_b);

    let binding_b = app
        .world()
        .get::<AvatarBinding>(root_b)
        .copied()
        .expect("avatar B should be bound");
    assert_eq!(binding_b.generation, gen_b);
    assert!(binding_b.neck.is_none());
    assert!(binding_b.left_eye.is_none());

    // Only one active marker.
    assert_eq!(count_active_markers(&mut app), 1);

    // --- Unload avatar B ---
    app.world_mut()
        .resource_mut::<Messages<UnloadAvatarRequest>>()
        .write(UnloadAvatarRequest);
    app.update();

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::NoAvatar);
    assert!(lifecycle.active_root().is_none());
    assert!(!lifecycle.has_active_generation());
    assert!(!app.world().entities().contains(root_b));
    assert_eq!(count_active_markers(&mut app), 0);
}

/// Failure during binding, then recovery with a new load.
#[test]
fn full_lifecycle_failure_then_recovery() {
    let mut app = test_app();

    // --- Avatar A: missing head → binding fails ---
    // load_and_bind inserts ActiveAvatar (via load_root), then BindTriggered and
    // Initialized (via simulate_initialized_and_bind). The bind system detects
    // the missing head in the same frame and transitions to Failed.
    let root_a = app
        .world_mut()
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            VrmHandle(Handle::default()),
            // No HeadBoneEntity → binding will fail.
        ))
        .id();

    load_and_bind(&mut app, root_a);

    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Failed);
    assert!(lifecycle.active_root().is_none());
    assert!(!app.world().entity(root_a).contains::<ActiveAvatar>());

    // --- Recovery: load avatar B successfully ---
    let head_b = spawn_bone(&mut app);
    let root_b = spawn_avatar_root(&mut app, head_b, None, None, None);

    load_root(&mut app, root_b);
    assert_eq!(
        app.world().resource::<AvatarLifecycle>().state(),
        AvatarLifecycleState::Loading,
        "load should be accepted from Failed state"
    );

    simulate_initialized_and_bind(&mut app, root_b);
    finish_ready(&mut app);
    assert_ready(&app, root_b);
    assert_eq!(count_active_markers(&mut app), 1);
}

/// Rapid replace coalescing: two replace requests while unloading.
#[test]
fn full_lifecycle_rapid_replace_coalesces() {
    let mut app = test_app();

    // Load and ready avatar A.
    let head_a = spawn_bone(&mut app);
    let root_a = spawn_avatar_root(&mut app, head_a, None, None, None);
    load_root(&mut app, root_a);
    simulate_initialized_and_bind(&mut app, root_a);
    finish_ready(&mut app);
    assert_ready(&app, root_a);
    let gen_a = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();

    // First replace → B, and second replace → C in the same frame (coalescing).
    // Both messages are read by apply_avatar_request_events in one pass:
    // the first transitions Ready → Unloading with pending = root_b,
    // the second coalesces in Unloading → Unloading with pending = root_c.
    let head_b = spawn_bone(&mut app);
    let root_b = spawn_avatar_root(&mut app, head_b, None, None, None);
    let head_c = spawn_bone(&mut app);
    let root_c = spawn_avatar_root(&mut app, head_c, None, None, None);

    {
        let mut messages = app
            .world_mut()
            .resource_mut::<Messages<ReplaceAvatarRequest>>();
        messages.write(ReplaceAvatarRequest { root: root_b });
        messages.write(ReplaceAvatarRequest { root: root_c });
    }
    app.update();

    // After coalescing + despawn in the same frame, lifecycle is Loading for C.
    let lifecycle = app.world().resource::<AvatarLifecycle>();
    assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
    assert_eq!(lifecycle.active_root(), Some(root_c));

    let gen_c = lifecycle.current_generation();
    assert_ne!(gen_a, gen_c);

    // A should be despawned.
    assert!(!app.world().entities().contains(root_a));

    // Drive C through binding.
    simulate_initialized_and_bind(&mut app, root_c);
    finish_ready(&mut app);
    assert_ready(&app, root_c);

    let binding_c = app
        .world()
        .get::<AvatarBinding>(root_c)
        .copied()
        .expect("avatar C should be bound");
    assert_eq!(binding_c.generation, gen_c);
    assert_eq!(count_active_markers(&mut app), 1);
}

/// Verifies that the single-active-avatar invariant holds at every step.
#[test]
fn full_lifecycle_single_active_invariant() {
    let mut app = test_app();

    // No avatar → 0 markers.
    assert_eq!(count_active_markers(&mut app), 0);

    // Load A → 1 marker.
    let head_a = spawn_bone(&mut app);
    let root_a = spawn_avatar_root(&mut app, head_a, None, None, None);
    load_root(&mut app, root_a);
    assert_eq!(count_active_markers(&mut app), 1);

    simulate_initialized_and_bind(&mut app, root_a);
    finish_ready(&mut app);
    assert_eq!(count_active_markers(&mut app), 1);

    // Replace → during Unloading, old marker is removed before new one is added.
    let head_b = spawn_bone(&mut app);
    let root_b = spawn_avatar_root(&mut app, head_b, None, None, None);
    app.world_mut()
        .resource_mut::<Messages<ReplaceAvatarRequest>>()
        .write(ReplaceAvatarRequest { root: root_b });
    app.update();

    // After despawn_unloading_avatar runs, old root is gone and new root
    // becomes active.
    assert_eq!(count_active_markers(&mut app), 1);

    simulate_initialized_and_bind(&mut app, root_b);
    finish_ready(&mut app);
    assert_eq!(count_active_markers(&mut app), 1);

    // Unload → 0 markers.
    app.world_mut()
        .resource_mut::<Messages<UnloadAvatarRequest>>()
        .write(UnloadAvatarRequest);
    app.update();
    assert_eq!(count_active_markers(&mut app), 0);
}

/// Control frames are rejected across avatar generations.
#[test]
fn full_lifecycle_control_frame_generation_boundary() {
    let mut app = test_app();

    // Load and ready avatar A.
    let head_a = spawn_bone(&mut app);
    let root_a = spawn_avatar_root(&mut app, head_a, None, None, None);
    load_root(&mut app, root_a);
    simulate_initialized_and_bind(&mut app, root_a);
    finish_ready(&mut app);
    let gen_a = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();

    // Apply a valid frame to A.
    let active = ActiveControlFrame {
        generation: gen_a,
        frame: Some(vtuber_core::types::AvatarControlFrame {
            source_seq: vtuber_core::types::FrameSeq(1),
            captured_at: vtuber_core::types::MonoTimeNs(0),
            produced_at: vtuber_core::types::MonoTimeNs(0),
            confidence: 1.0,
            state: vtuber_core::types::TrackingState::Tracking,
            head: vtuber_core::types::HeadPose::default(),
            head_translation: vtuber_core::HeadTranslationSignal::UNAVAILABLE,
            gaze: vtuber_core::GazeSignal::UNAVAILABLE,
            expressions: vtuber_core::types::ExpressionCoefficients::default(),
            detailed_face: None,
        }),
    };
    let result = apply_active_control_frame(
        app.world().resource::<AvatarLifecycle>(),
        &active,
        app.world().get::<AvatarBinding>(root_a),
    );
    assert!(result.is_ok(), "frame should apply to avatar A");

    // Replace with B.
    let head_b = spawn_bone(&mut app);
    let root_b = spawn_avatar_root(&mut app, head_b, None, None, None);
    app.world_mut()
        .resource_mut::<Messages<ReplaceAvatarRequest>>()
        .write(ReplaceAvatarRequest { root: root_b });
    app.update();
    simulate_initialized_and_bind(&mut app, root_b);
    finish_ready(&mut app);
    let gen_b = app
        .world()
        .resource::<AvatarLifecycle>()
        .current_generation();

    // Old frame targeting A must be rejected for B.
    let result = apply_active_control_frame(
        app.world().resource::<AvatarLifecycle>(),
        &active,
        app.world().get::<AvatarBinding>(root_b),
    );
    assert!(
        matches!(
            result,
            Err(vtuber_avatar::unload::ControlFrameError::StaleGeneration { .. })
        ),
        "frame from avatar A must be rejected for avatar B"
    );

    // New frame targeting B should succeed.
    let active_b = ActiveControlFrame {
        generation: gen_b,
        frame: Some(vtuber_core::types::AvatarControlFrame {
            source_seq: vtuber_core::types::FrameSeq(2),
            ..active.frame.unwrap().clone()
        }),
    };
    let result = apply_active_control_frame(
        app.world().resource::<AvatarLifecycle>(),
        &active_b,
        app.world().get::<AvatarBinding>(root_b),
    );
    assert!(result.is_ok(), "frame should apply to avatar B");
}

/// Snapshot reflects the lifecycle at every phase.
#[test]
fn full_lifecycle_snapshot_reflects_state() {
    let mut app = test_app();

    // NoAvatar
    let snap = app.world().resource::<AvatarLifecycle>().snapshot();
    assert_eq!(snap.state, AvatarLifecycleState::NoAvatar);
    assert!(snap.active_root.is_none());
    assert!(snap.capabilities.is_none());
    assert_eq!(snap.generation, AvatarGeneration::default());

    // Load A
    let head_a = spawn_bone(&mut app);
    let root_a = spawn_avatar_root(&mut app, head_a, None, None, None);
    load_root(&mut app, root_a);

    let snap = app.world().resource::<AvatarLifecycle>().snapshot();
    assert_eq!(snap.state, AvatarLifecycleState::Loading);
    assert_eq!(snap.active_root, Some(root_a));
    assert!(snap.capabilities.is_none());

    // Ready A
    simulate_initialized_and_bind(&mut app, root_a);
    finish_ready(&mut app);

    let snap = app.world().resource::<AvatarLifecycle>().snapshot();
    assert_eq!(snap.state, AvatarLifecycleState::Ready);
    assert_eq!(snap.active_root, Some(root_a));
    assert!(snap.capabilities.is_some());

    // Unload
    app.world_mut()
        .resource_mut::<Messages<UnloadAvatarRequest>>()
        .write(UnloadAvatarRequest);
    app.update();

    let snap = app.world().resource::<AvatarLifecycle>().snapshot();
    assert_eq!(snap.state, AvatarLifecycleState::NoAvatar);
    assert!(snap.active_root.is_none());
    assert!(snap.capabilities.is_none());
}

/// Runs the revised Issue #31 transition matrix twice in one process. The
/// synthetic roots carry the generation-specific runtime components that the
/// real loader owns, so stale basis, expression, and SpringBone state is
/// checked directly after every unload and replacement.
#[test]
fn generation_transition_matrix_has_no_stale_avatar_state() {
    let sequences = vec![
        vec![SyntheticGeneration::Vrm0],
        vec![SyntheticGeneration::Vrm1],
        vec![SyntheticGeneration::Vrm0, SyntheticGeneration::Vrm0],
        vec![SyntheticGeneration::Vrm1, SyntheticGeneration::Vrm1],
        vec![
            SyntheticGeneration::Vrm1,
            SyntheticGeneration::Vrm0,
            SyntheticGeneration::Vrm1,
        ],
        vec![
            SyntheticGeneration::Vrm0,
            SyntheticGeneration::Vrm1,
            SyntheticGeneration::Vrm0,
        ],
    ];

    let mut app = test_app();
    for _repetition in 0..2 {
        for sequence in &sequences {
            let mut previous_root = None;
            for (index, generation) in sequence.iter().copied().enumerate() {
                let root = spawn_generation_avatar_root(&mut app, generation);
                if index == 0 {
                    load_root(&mut app, root);
                } else {
                    app.world_mut()
                        .resource_mut::<Messages<ReplaceAvatarRequest>>()
                        .write(ReplaceAvatarRequest { root });
                    app.update();
                    assert!(
                        previous_root
                            .is_some_and(|previous| !app.world().entities().contains(previous)),
                        "the previous avatar root must be gone after replacement"
                    );
                    assert_eq!(
                        count_spring_joint_states(&mut app),
                        if matches!(generation, SyntheticGeneration::Vrm0) {
                            1
                        } else {
                            0
                        },
                        "replacement must not retain SpringJointState from the previous avatar"
                    );
                }

                assert_generation_state(&mut app, root, generation);
                simulate_initialized_and_bind(&mut app, root);
                finish_ready(&mut app);
                assert_ready(&app, root);
                assert_generation_state(&mut app, root, generation);
                previous_root = Some(root);
            }

            app.world_mut()
                .resource_mut::<Messages<UnloadAvatarRequest>>()
                .write(UnloadAvatarRequest);
            app.update();
            assert_avatar_owned_state_empty(&mut app);
        }
    }
}
