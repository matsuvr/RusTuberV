//! Binding trigger for initialized VRM roots.
//!
//! Watches the active avatar root for the `Initialized` marker, validates the
//! asset load state, and drives the lifecycle into `Binding` or `Failed`.
//! Stale initializations on roots that are no longer active are ignored, and
//! each root can trigger the bind path at most once.

use bevy::asset::LoadState;
use bevy::prelude::*;
use bevy_vrm1::prelude::*;
use std::time::Instant;

use crate::lifecycle::{
    ActiveAvatar, AvatarLifecycle, AvatarLifecycleFailure, AvatarLifecycleState,
};
use crate::load::ExpectedVrmGeneration;

/// Marker preventing a root from triggering the bind path more than once.
///
/// This is inserted by [`observe_initialized`] the first time `Initialized` is
/// observed on the active root. It defends against duplicate triggers if the
/// same root is ever re-scanned.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindTriggered;

/// Observes `Initialized` on the active root, detects load failure and timeout,
/// and transitions the lifecycle.
///
/// Only the entity currently tracked as the active root can trigger a
/// transition. Roots that are no longer active (for example, a replaced model
/// whose asset finishes loading late) are ignored.
#[allow(clippy::type_complexity)]
pub(crate) fn observe_initialized(
    mut commands: Commands,
    mut lifecycle: ResMut<AvatarLifecycle>,
    asset_server: Res<AssetServer>,
    active_loading_handles: Query<(Entity, &VrmHandle), With<ActiveAvatar>>,
    newly_initialized: Query<
        (
            Entity,
            Option<&ExpectedVrmGeneration>,
            Option<&VrmCoordinateBasis>,
        ),
        (
            With<ActiveAvatar>,
            Added<Initialized>,
            Without<BindTriggered>,
        ),
    >,
) {
    // Detect timeout or explicit asset load failure for the currently loading
    // root before considering successful initialization.
    if lifecycle.state() == AvatarLifecycleState::Loading {
        let now = Instant::now();
        if lifecycle.is_load_timed_out(now) {
            if let Some(entity) = lifecycle.active_root() {
                error!(
                    "avatar load timeout: root={entity:?} had not reached Initialized before the deadline; lifecycle={:?}",
                    lifecycle.state()
                );
                commands.entity(entity).remove::<ActiveAvatar>();
            }
            lifecycle.fail(AvatarLifecycleFailure::AssetLoadFailed);
            return;
        }

        for (entity, handle) in active_loading_handles.iter() {
            if lifecycle.active_root() != Some(entity) {
                continue;
            }
            match asset_server.load_state(handle.0.id()) {
                LoadState::Failed(error) => {
                    error!(
                        "VRM asset load failed: root={entity:?} path={:?} error={error:?} lifecycle={:?}",
                        handle.0.path(),
                        lifecycle.state()
                    );
                    commands.entity(entity).remove::<ActiveAvatar>();
                    lifecycle.fail(AvatarLifecycleFailure::AssetLoadFailed);
                    return;
                }
                LoadState::Loaded | LoadState::Loading | LoadState::NotLoaded => {
                    // Keep waiting. `NotLoaded` is unusual after spawning a
                    // handle, but is treated as "not yet failed" until timeout.
                }
            }
        }
    }

    // Transition to Binding once Initialized is observed exactly once on the
    // active root. The `Without<BindTriggered>` filter and `Added<Initialized>`
    // filter together ensure this path runs at most once per root.
    for (entity, expected_generation, coordinate_basis) in newly_initialized.iter() {
        if lifecycle.active_root() != Some(entity) {
            continue;
        }
        if lifecycle.state() != AvatarLifecycleState::Loading {
            continue;
        }
        if let Some(expected) = expected_generation {
            let detected = coordinate_basis.map(|basis| match basis.0 {
                CoordinateBasis::Vrm0Y180 => ExpectedVrmGeneration::Vrm0,
                CoordinateBasis::Vrm1Identity => ExpectedVrmGeneration::Vrm1,
            });
            if detected != Some(*expected) {
                commands.entity(entity).remove::<ActiveAvatar>();
                lifecycle.fail(AvatarLifecycleFailure::GenerationMismatch {
                    expected: *expected,
                    detected,
                });
                continue;
            }
        }
        commands.entity(entity).insert(BindTriggered);
        lifecycle.start_binding(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::AssetApp;

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<bevy_vrm1::prelude::VrmAsset>()
            .init_resource::<AvatarLifecycle>()
            .add_systems(Update, observe_initialized);
        app
    }

    fn spawn_active_root(app: &mut App) -> Entity {
        app.world_mut()
            .spawn((
                ActiveAvatar,
                VrmHandle(Handle::default()),
                Transform::default(),
                GlobalTransform::default(),
                Visibility::Hidden,
            ))
            .id()
    }

    #[test]
    fn avatar_initialized_once() {
        let mut app = test_app();
        let root = spawn_active_root(&mut app);
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .unwrap();

        // Simulate bevy_vrm1 adding Initialized to the active root.
        app.world_mut().entity_mut(root).insert(Initialized);
        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Binding);
        assert_eq!(lifecycle.active_root(), Some(root));

        let world = app.world();
        assert!(world.entity(root).contains::<BindTriggered>());

        // A second update must not re-trigger binding or panic.
        app.update();
        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Binding);
        assert_eq!(lifecycle.active_root(), Some(root));
    }

    #[test]
    fn handle_removed_before_initialized_still_enters_binding() {
        let mut app = test_app();
        let root = spawn_active_root(&mut app);
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .unwrap();

        // This is the ordering used by the pinned bevy_vrm1 revision: the
        // handle is removed when the runtime components are attached, and the
        // Initialized marker arrives in a later update.
        app.world_mut().entity_mut(root).remove::<VrmHandle>();
        app.world_mut().entity_mut(root).insert(Initialized);
        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Binding);
        assert_eq!(lifecycle.active_root(), Some(root));
        assert!(lifecycle.failure().is_none());
        assert!(app.world().entity(root).contains::<BindTriggered>());
    }

    #[test]
    fn handleless_root_waits_for_initialized_without_failing_early() {
        let mut app = test_app();
        let root = spawn_active_root(&mut app);
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .request_load(root)
            .unwrap();
        app.world_mut().entity_mut(root).remove::<VrmHandle>();

        // The handle-less interval is expected while bevy_vrm1 finishes
        // spawning the runtime hierarchy. A normal update must keep waiting.
        app.update();
        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
        assert_eq!(lifecycle.active_root(), Some(root));
        assert!(lifecycle.failure().is_none());
    }

    #[test]
    fn avatar_initialized_ignored_when_not_active_root() {
        let mut app = test_app();
        let stale_root = spawn_active_root(&mut app);
        let active_root = spawn_active_root(&mut app);

        {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            lifecycle.request_load(active_root).unwrap();
        }

        // Both roots receive Initialized, but only the active one may trigger
        // binding. This exercises stale-request late-completion defense.
        app.world_mut().entity_mut(stale_root).remove::<VrmHandle>();
        app.world_mut()
            .entity_mut(active_root)
            .remove::<VrmHandle>();
        app.world_mut().entity_mut(stale_root).insert(Initialized);
        app.world_mut().entity_mut(active_root).insert(Initialized);
        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Binding);
        assert_eq!(lifecycle.active_root(), Some(active_root));

        let world = app.world();
        assert!(!world.entity(stale_root).contains::<BindTriggered>());
        assert!(world.entity(active_root).contains::<BindTriggered>());
    }

    #[test]
    fn avatar_load_timeout_transitions_to_failed() {
        let mut app = test_app();
        let root = spawn_active_root(&mut app);

        {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            lifecycle.request_load(root).unwrap();
            lifecycle.set_load_started_for_test(Some(
                Instant::now() - std::time::Duration::from_secs(60),
            ));
        }

        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Failed);
        assert!(lifecycle.active_root().is_none());
        assert!(!app.world().entity(root).contains::<ActiveAvatar>());
    }

    #[test]
    fn expected_and_runtime_generation_must_match() {
        for (expected, detected, should_bind) in [
            (ExpectedVrmGeneration::Vrm0, CoordinateBasis::Vrm0Y180, true),
            (
                ExpectedVrmGeneration::Vrm1,
                CoordinateBasis::Vrm1Identity,
                true,
            ),
            (
                ExpectedVrmGeneration::Vrm0,
                CoordinateBasis::Vrm1Identity,
                false,
            ),
            (
                ExpectedVrmGeneration::Vrm1,
                CoordinateBasis::Vrm0Y180,
                false,
            ),
        ] {
            let mut app = test_app();
            let root = spawn_active_root(&mut app);
            app.world_mut().entity_mut(root).insert((
                expected,
                VrmCoordinateBasis(detected),
                Initialized,
            ));
            app.world_mut()
                .resource_mut::<AvatarLifecycle>()
                .request_load(root)
                .unwrap();
            app.update();

            let lifecycle = app.world().resource::<AvatarLifecycle>();
            if should_bind {
                assert_eq!(lifecycle.state(), AvatarLifecycleState::Binding);
                assert!(lifecycle.failure().is_none());
            } else {
                assert_eq!(lifecycle.state(), AvatarLifecycleState::Failed);
                assert!(matches!(
                    lifecycle.failure(),
                    Some(AvatarLifecycleFailure::GenerationMismatch { .. })
                ));
            }
        }
    }
}
