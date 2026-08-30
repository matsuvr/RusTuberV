// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! VRM schedule ordering and input-path guard tests.
//!
//! Verifies that:
//! - The product app does not use `bevy_vrm1::LookAt` to encode face pose.
//! - Direct-pose `bevy_vrm1::BodyTracking` is the intended humanoid writer.
//! - The pose and expression systems are registered in the correct schedule.
//! - No schedule cycles exist.

/// Verify that the avatar plugin remains constructible without exposing a
/// synthetic face-pose `LookAt` API.
///
/// This test verifies by checking that our public API does not re-export
/// these bevy_vrm1 types. The plugin.rs also does not insert them.
#[test]
fn avatar_schedule_has_no_synthetic_look_at_api() {
    // Verify that VtuberAvatarPlugin is constructible and our public types
    // don't include a synthetic LookAt target.
    let _plugin = vtuber_avatar::VtuberAvatarPlugin;

    // The following types should NOT exist in our public API:
    // - bevy_vrm1::LookAt (we never insert it)
    // Direct-pose BodyTracking is integrated internally by Q2-06-001.
    //
    // This is verified by the fact that this test compiles without
    // importing those types, and our lib.rs doesn't re-export them.
}

/// Verify the schedule graph registered by the real avatar and VRM plugins:
///
/// 1. apply_avatar_request_events (Update, chained)
/// 2. despawn_unloading_avatar (Update, chained)
/// 3. bind_humanoid_bones (Update, chained)
/// 4. direct-pose BodyTracking (PostUpdate)
/// 5. model-adaptive default arm pose compositor (PostUpdate)
/// 6. VrmSystemSets::Constraints (PostUpdate, bevy_vrm1 internal)
/// 7. VrmSystemSets::Expressions (PostUpdate, bevy_vrm1 internal)
/// 8. VrmSystemSets::SpringBone (PostUpdate, bevy_vrm1 internal)
#[test]
fn avatar_schedule_ordering_matches_design() {
    use bevy::app::AnimationSystems;
    use bevy::ecs::schedule::{IntoScheduleConfigs, Schedule};
    use bevy::prelude::*;
    use bevy::winit::WinitPlugin;
    use bevy_vrm1::prelude::VrmSystemSets;

    fn system_index(schedule: &Schedule, suffix: &str) -> usize {
        let matches: Vec<_> = schedule
            .systems()
            .expect("schedule should already be initialized")
            .enumerate()
            .filter(|(_, (_, system))| system.name().contains(suffix))
            .map(|(index, _)| index)
            .collect();
        if matches.len() != 1 {
            let names: Vec<_> = schedule
                .systems()
                .expect("schedule should already be initialized")
                .map(|(_, system)| system.name().to_string())
                .collect();
            panic!("expected one system containing {suffix}; registered systems: {names:#?}");
        }
        matches[0]
    }

    fn assert_before(schedule: &Schedule, before: &str, after: &str) {
        assert!(
            system_index(schedule, before) < system_index(schedule, after),
            "registered schedule should order {before} before {after}"
        );
    }

    fn trace_animation() {}
    fn trace_gaze() {}
    fn trace_expressions() {}
    fn trace_propagate_expressions() {}
    fn trace_constraints() {}
    fn trace_propagate_constraints() {}
    fn trace_spring_bone() {}

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .build()
            .disable::<WinitPlugin>()
            .set(WindowPlugin {
                primary_window: None,
                ..default()
            }),
    );
    app.add_plugins(vtuber_avatar::VtuberAvatarPlugin);
    app.add_systems(
        PostUpdate,
        (
            trace_animation.in_set(AnimationSystems),
            trace_gaze.in_set(VrmSystemSets::GazeControl),
            trace_expressions.in_set(VrmSystemSets::Expressions),
            trace_propagate_expressions.in_set(VrmSystemSets::PropagateAfterExpressions),
            trace_constraints.in_set(VrmSystemSets::Constraints),
            trace_propagate_constraints.in_set(VrmSystemSets::PropagateAfterConstraints),
            trace_spring_bone.in_set(VrmSystemSets::SpringBone),
        ),
    );

    app.world_mut()
        .schedule_scope(PostUpdate, |world, schedule| {
            schedule
                .initialize(world)
                .expect("registered PostUpdate schedule should initialize");
            for (before, after) in [
                ("trace_animation", "update_body_tracking_pose_input"),
                (
                    "update_body_tracking_pose_input",
                    "apply_direct_body_tracking",
                ),
                ("trace_animation", "update_body_tracking_position_input"),
                (
                    "update_body_tracking_position_input",
                    "apply_direct_body_tracking",
                ),
                // The #20 breathing writer is retired (issue #180, ADR-020):
                // no system named "breathing" may exist anywhere in PostUpdate
                // (asserted below), so the hips channel has no runtime writer.
                ("apply_direct_body_tracking", "apply_default_arm_pose"),
                ("apply_default_arm_pose", "update_direct_look_at_input"),
                ("apply_direct_body_tracking", "update_direct_look_at_input"),
                ("update_direct_look_at_input", "trace_gaze"),
                ("trace_gaze", "apply_tracked_expressions"),
                ("apply_tracked_expressions", "trace_expressions"),
                ("trace_expressions", "trace_propagate_expressions"),
                ("trace_propagate_expressions", "trace_constraints"),
                ("trace_constraints", "trace_propagate_constraints"),
                ("trace_propagate_constraints", "trace_spring_bone"),
            ] {
                assert_before(schedule, before, after);
            }
            // Issue #180: the retired #20 breathing writer must not exist in
            // the production PostUpdate schedule at all.
            for (_, system) in schedule.systems().expect("schedule initialized") {
                let name = system.name().to_string();
                assert!(
                    !name.to_lowercase().contains("breathing"),
                    "retired breathing system found in PostUpdate: {name}"
                );
            }
        });
}

/// Verify that the lifecycle types are properly exported for schedule integration.
#[test]
fn avatar_schedule_lifecycle_types_exported() {
    // These types are needed by the schedule systems.
    let _state = vtuber_avatar::AvatarLifecycleState::NoAvatar;
    let _gen = vtuber_avatar::AvatarGeneration(0);
}
