// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Public-construction smoke tests and registered PostUpdate ordering checks.
//!
//! These tests construct the public avatar plugin and lifecycle values,
//! initialize the real PostUpdate schedule, and inspect the listed system
//! ordering constraints and absence of a system name containing "breathing".
//! They do not prove the absence of public APIs or exercise pose application.

/// Verifies that downstream code can construct the public avatar plugin.
#[test]
fn avatar_plugin_is_publicly_constructible() {
    let _plugin = vtuber_avatar::VtuberAvatarPlugin;
}

/// Checks the listed ordering edges in the real avatar/VRM PostUpdate schedule.
///
/// The checks cover animation before body inputs, body writers before arm
/// target generation and composition, arm composition before direct gaze, and
/// tracked expressions between gaze control and the upstream expression set.
/// Schedule initialization also detects cycles in this PostUpdate schedule.
/// Update-stage lifecycle ordering is not inspected by this test.
#[test]
fn avatar_post_update_schedule_orders_body_arms_gaze_and_expressions() {
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
                // The retired always-on breathing writer is absent (ADR-020):
                // below we reject any PostUpdate system name containing
                // "breathing". This does not rule out other hips writers.
                ("apply_direct_body_tracking", "apply_default_arm_pose"),
                ("apply_default_arm_pose", "update_direct_look_at_input"),
                ("apply_direct_body_tracking", "update_direct_look_at_input"),
                ("update_direct_look_at_input", "trace_gaze"),
                ("trace_gaze", "apply_tracked_expressions"),
                ("apply_tracked_expressions", "trace_expressions"),
                // Both arm target stages sample the torso bone the arms hang
                // from, so they must run after every writer of that bone's
                // global rotation. Reading it before either writer would use
                // the pose the previous frame left behind.
                ("apply_direct_body_tracking", "update_dynamic_arm_targets"),
                ("apply_direct_body_position", "update_dynamic_arm_targets"),
                ("apply_direct_body_tracking", "update_tracked_arm_targets"),
                ("apply_direct_body_position", "update_tracked_arm_targets"),
                ("apply_direct_body_position", "apply_default_arm_pose"),
                ("update_dynamic_arm_targets", "update_tracked_arm_targets"),
                ("update_tracked_arm_targets", "apply_default_arm_pose"),
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
