//! Avatar lifecycle domain.
//!
//! This module defines the state machine that governs a single active VRM
//! avatar at a time. It exposes request events, typed results, and a
//! read-only snapshot for the UI. No `bevy_vrm1` types are used here.

use crate::capabilities::AvatarCapabilities;
use crate::load::ExpectedVrmGeneration;
use bevy::ecs::message::Message;
use bevy::prelude::*;
use std::time::{Duration, Instant};

/// Monotonically increasing identifier for an avatar instance.
///
/// Control frames are tagged with the generation of the avatar they target.
/// When a new avatar is loaded or replaces the current one, the generation
/// changes and frames targeting a previous generation are rejected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct AvatarGeneration(pub u64);

/// Lifecycle state of the single active avatar slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AvatarLifecycleState {
    /// No avatar is loaded and no request is in flight.
    #[default]
    NoAvatar,
    /// A load request has been accepted and the model is being loaded.
    Loading,
    /// The VRM root has initialized and bones/expressions are being bound.
    Binding,
    /// The avatar is bound and ready to receive tracking data.
    Ready,
    /// The current avatar is being unloaded before another can take its place.
    Unloading,
    /// The previous load or binding attempt failed.
    Failed,
}

/// Marker component for the single active avatar root entity.
///
/// At most one entity should carry this marker at any time. The
/// [`AvatarLifecycle`] resource is the authority that maintains this
/// invariant.
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ActiveAvatar;

/// Errors returned by lifecycle request validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AvatarRequestError {
    /// The request cannot be handled in the current lifecycle state.
    InvalidState {
        /// State the lifecycle was in when the request was rejected.
        current: AvatarLifecycleState,
    },
    /// An unload or replace was requested but no avatar is active.
    NoActiveAvatar,
}

impl std::fmt::Display for AvatarRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidState { current } => {
                write!(f, "invalid request in lifecycle state {current:?}")
            }
            Self::NoActiveAvatar => write!(f, "no active avatar to unload"),
        }
    }
}

impl std::error::Error for AvatarRequestError {}

/// Shorthand for request validation results.
pub type AvatarRequestResult = Result<(), AvatarRequestError>;

/// Read-only snapshot of the avatar lifecycle for UI consumption.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AvatarLifecycleSnapshot {
    /// Current lifecycle state.
    pub state: AvatarLifecycleState,
    /// Entity of the active avatar root, if any.
    pub active_root: Option<Entity>,
    /// Entity waiting to replace the current avatar during a replace request.
    pub pending_root: Option<Entity>,
    /// Capability snapshot of the active avatar, if binding has completed.
    pub capabilities: Option<AvatarCapabilities>,
    /// Generation of the active or in-progress avatar instance.
    pub generation: AvatarGeneration,
    /// Failure reason when the lifecycle is in `Failed`.
    pub failure: Option<AvatarLifecycleFailure>,
}

/// Request to load a new avatar into the active slot.
///
/// The request must target the entity that will become the VRM root. It is
/// rejected unless the lifecycle is currently in [`AvatarLifecycleState::NoAvatar`]
/// or [`AvatarLifecycleState::Failed`].
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadAvatarRequest {
    /// Entity that will carry the loaded VRM.
    pub root: Entity,
}

/// Request to unload the currently active avatar.
///
/// Rejected unless the lifecycle is currently in [`AvatarLifecycleState::Ready`].
#[derive(Message, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnloadAvatarRequest;

/// Request to replace the currently active avatar with a new one.
///
/// When the lifecycle is in [`AvatarLifecycleState::Ready`], the current avatar
/// is put into [`AvatarLifecycleState::Unloading`] and the new root is queued.
/// When no avatar is active, this behaves like [`LoadAvatarRequest`].
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplaceAvatarRequest {
    /// Entity that will carry the replacement VRM.
    pub root: Entity,
}

/// Result of a [`LoadAvatarRequest`].
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadAvatarResult {
    /// The load request was accepted and the lifecycle entered `Loading`.
    Accepted {
        /// Entity that will become the active avatar root.
        root: Entity,
    },
    /// The load request was rejected.
    Rejected(AvatarRequestError),
}

/// Result of an [`UnloadAvatarRequest`].
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnloadAvatarResult {
    /// The unload request was accepted and the lifecycle entered `Unloading`.
    Accepted,
    /// The unload request was rejected.
    Rejected(AvatarRequestError),
}

/// Result of a [`ReplaceAvatarRequest`].
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplaceAvatarResult {
    /// The replace request was accepted.
    Accepted {
        /// Entity that will become the active avatar root.
        new_root: Entity,
    },
    /// The replace request was rejected.
    Rejected(AvatarRequestError),
}

/// Reasons the lifecycle can enter [`AvatarLifecycleState::Failed`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AvatarLifecycleFailure {
    /// A required humanoid bone was missing.
    MissingRequiredBone {
        /// Humanoid bone name, e.g. `head`.
        bone: &'static str,
    },
    /// Binding did not complete within the deadline.
    BindingTimeout,
    /// A required bone has an unusable rest orientation.
    InvalidRestOrientation {
        /// Humanoid bone name, e.g. `head`.
        bone: &'static str,
    },
    /// The VRM asset failed to load.
    AssetLoadFailed,
    /// Runtime generation disagreed with app-side preflight.
    GenerationMismatch {
        /// Generation recorded by app-side inspection.
        expected: ExpectedVrmGeneration,
        /// Generation discovered by the runtime boundary.
        detected: Option<ExpectedVrmGeneration>,
    },
}

impl std::fmt::Display for AvatarLifecycleFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingRequiredBone { bone } => {
                write!(f, "required humanoid bone is missing: {bone}")
            }
            Self::BindingTimeout => f.write_str("avatar binding timed out"),
            Self::InvalidRestOrientation { bone } => {
                write!(f, "humanoid bone has invalid rest orientation: {bone}")
            }
            Self::AssetLoadFailed => f.write_str("VRM asset failed to load"),
            Self::GenerationMismatch { expected, detected } => {
                write!(
                    f,
                    "VRM generation mismatch: expected {expected:?}, detected {detected:?}"
                )
            }
        }
    }
}

/// Maximum time to wait for a VRM asset to initialize before treating it as failed.
const LOAD_TIMEOUT: Duration = Duration::from_secs(30);

/// Resource that owns the single active avatar lifecycle.
///
/// This resource maintains the invariant that at most one avatar is active at
/// a time. It validates request events and drives internal state transitions.
#[derive(Resource, Debug, Clone, PartialEq, Eq)]
pub struct AvatarLifecycle {
    state: AvatarLifecycleState,
    active_root: Option<Entity>,
    pending_root: Option<Entity>,
    load_started: Option<Instant>,
    capabilities: Option<AvatarCapabilities>,
    failure: Option<AvatarLifecycleFailure>,
    current_generation: AvatarGeneration,
    next_generation: u64,
}

impl Default for AvatarLifecycle {
    fn default() -> Self {
        Self {
            state: AvatarLifecycleState::NoAvatar,
            active_root: None,
            pending_root: None,
            load_started: None,
            capabilities: None,
            failure: None,
            current_generation: AvatarGeneration::default(),
            next_generation: 1,
        }
    }
}

impl AvatarLifecycle {
    /// Creates a new lifecycle in [`AvatarLifecycleState::NoAvatar`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current lifecycle state.
    #[must_use]
    pub fn state(&self) -> AvatarLifecycleState {
        self.state
    }

    /// Entity of the active avatar root, if any.
    #[must_use]
    pub fn active_root(&self) -> Option<Entity> {
        self.active_root
    }

    /// Entity queued to replace the current avatar, if any.
    #[must_use]
    pub fn pending_root(&self) -> Option<Entity> {
        self.pending_root
    }

    /// Instant when the current `Loading` state began, if any.
    #[must_use]
    pub fn load_started(&self) -> Option<Instant> {
        self.load_started
    }

    /// Capability snapshot of the currently active avatar, if binding has completed.
    #[must_use]
    pub fn capabilities(&self) -> Option<&AvatarCapabilities> {
        self.capabilities.as_ref()
    }

    /// Failure reason for the current failed load, if any.
    #[must_use]
    pub fn failure(&self) -> Option<&AvatarLifecycleFailure> {
        self.failure.as_ref()
    }

    /// Sets or clears the capability snapshot.
    pub(crate) fn set_capabilities(&mut self, caps: Option<AvatarCapabilities>) {
        self.capabilities = caps;
    }

    /// Returns `true` if the current load has exceeded the timeout deadline.
    #[must_use]
    pub fn is_load_timed_out(&self, now: Instant) -> bool {
        self.load_started
            .is_some_and(|started| now >= started + LOAD_TIMEOUT)
    }

    /// Current avatar generation.
    ///
    /// Frames and bindings are correlated by this value. It changes whenever
    /// a new avatar is loaded or replaces the current one.
    #[must_use]
    pub fn current_generation(&self) -> AvatarGeneration {
        self.current_generation
    }

    /// Returns `true` if a non-zero generation is assigned to an in-progress
    /// or active avatar.
    #[must_use]
    pub fn has_active_generation(&self) -> bool {
        self.current_generation.0 != 0
            && self.state != AvatarLifecycleState::NoAvatar
            && self.state != AvatarLifecycleState::Failed
    }

    /// Returns a read-only snapshot for UI consumption.
    #[must_use]
    pub fn snapshot(&self) -> AvatarLifecycleSnapshot {
        AvatarLifecycleSnapshot {
            state: self.state,
            active_root: self.active_root,
            pending_root: self.pending_root,
            capabilities: self.capabilities.clone(),
            generation: self.current_generation,
            failure: self.failure.clone(),
        }
    }

    /// Validates and starts a load request.
    ///
    /// Allowed from `NoAvatar` or `Failed`. Transitions to `Loading`.
    pub fn request_load(&mut self, root: Entity) -> AvatarRequestResult {
        match self.state {
            AvatarLifecycleState::NoAvatar | AvatarLifecycleState::Failed => {
                self.state = AvatarLifecycleState::Loading;
                self.active_root = Some(root);
                self.pending_root = None;
                self.load_started = Some(Instant::now());
                self.capabilities = None;
                self.failure = None;
                self.current_generation = AvatarGeneration(self.next_generation);
                self.next_generation += 1;
                Ok(())
            }
            _ => Err(AvatarRequestError::InvalidState {
                current: self.state,
            }),
        }
    }

    /// Validates and starts an unload request.
    ///
    /// Allowed from `Ready`. Transitions to `Unloading` while preserving
    /// `active_root` until [`Self::finish_unload`] is called.
    pub fn request_unload(&mut self) -> AvatarRequestResult {
        match self.state {
            AvatarLifecycleState::Ready => {
                self.state = AvatarLifecycleState::Unloading;
                self.pending_root = None;
                self.capabilities = None;
                self.failure = None;
                Ok(())
            }
            AvatarLifecycleState::NoAvatar
            | AvatarLifecycleState::Loading
            | AvatarLifecycleState::Binding
            | AvatarLifecycleState::Unloading
            | AvatarLifecycleState::Failed => Err(AvatarRequestError::InvalidState {
                current: self.state,
            }),
        }
    }

    /// Validates and starts a replace request.
    ///
    /// From `Ready`, the current avatar moves to `Unloading` and the new root
    /// is queued. From `Unloading`, the pending root is updated to the latest
    /// request (coalescing). From `NoAvatar` or `Failed`, this behaves like a
    /// load.
    pub fn request_replace(&mut self, root: Entity) -> AvatarRequestResult {
        match self.state {
            AvatarLifecycleState::Ready | AvatarLifecycleState::Unloading => {
                // A new avatar instance gets a fresh generation even when
                // coalescing rapid replacements. Frames targeting an older
                // generation will be rejected by the binding that eventually
                // becomes Ready.
                self.current_generation = AvatarGeneration(self.next_generation);
                self.next_generation += 1;
                self.state = AvatarLifecycleState::Unloading;
                self.pending_root = Some(root);
                self.capabilities = None;
                self.failure = None;
                Ok(())
            }
            AvatarLifecycleState::NoAvatar | AvatarLifecycleState::Failed => {
                self.request_load(root)
            }
            _ => Err(AvatarRequestError::InvalidState {
                current: self.state,
            }),
        }
    }

    /// Transitions from `Loading` to `Binding` for the given root.
    ///
    /// Called by internal systems once the VRM asset has spawned and the
    /// `Initialized` marker is observed. The root must match the currently
    /// tracked active root; stale initializations are ignored.
    pub fn start_binding(&mut self, root: Entity) {
        if self.state == AvatarLifecycleState::Loading && self.active_root == Some(root) {
            self.state = AvatarLifecycleState::Binding;
            self.load_started = None;
        }
    }

    /// Transitions from `Binding` to `Ready`.
    ///
    /// Called by internal systems once all required bones and expressions have
    /// been bound.
    pub fn finish_ready(&mut self) {
        if self.state == AvatarLifecycleState::Binding {
            self.state = AvatarLifecycleState::Ready;
        }
    }

    /// Completes an unload.
    ///
    /// From `Unloading`, clears the active root. If a replacement was queued,
    /// the lifecycle moves directly to `Loading` for the pending root.
    /// Otherwise it returns to `NoAvatar`.
    pub fn finish_unload(&mut self) {
        if self.state != AvatarLifecycleState::Unloading {
            return;
        }

        if let Some(pending) = self.pending_root.take() {
            self.state = AvatarLifecycleState::Loading;
            self.active_root = Some(pending);
            self.load_started = Some(Instant::now());
            self.capabilities = None;
            self.failure = None;
        } else {
            self.state = AvatarLifecycleState::NoAvatar;
            self.active_root = None;
            self.load_started = None;
            self.capabilities = None;
            self.failure = None;
        }
    }

    /// Records a failure and clears any in-progress avatar.
    pub fn fail(&mut self, error: AvatarLifecycleFailure) {
        self.state = AvatarLifecycleState::Failed;
        self.active_root = None;
        self.pending_root = None;
        self.load_started = None;
        self.capabilities = None;
        self.failure = Some(error);
    }
}

#[cfg(test)]
impl AvatarLifecycle {
    /// Sets the load start timestamp for tests. Not part of the public contract.
    pub fn set_load_started_for_test(&mut self, instant: Option<Instant>) {
        self.load_started = instant;
    }
}

/// System that applies request events and emits typed results.
///
/// This system also inserts and removes the [`ActiveAvatar`] marker to keep
/// the ECS view consistent with the resource invariant.
#[allow(clippy::too_many_arguments)]
pub fn apply_avatar_request_events(
    mut commands: Commands,
    mut lifecycle: ResMut<AvatarLifecycle>,
    mut load_requests: MessageReader<LoadAvatarRequest>,
    mut load_results: MessageWriter<LoadAvatarResult>,
    mut unload_requests: MessageReader<UnloadAvatarRequest>,
    mut unload_results: MessageWriter<UnloadAvatarResult>,
    mut replace_requests: MessageReader<ReplaceAvatarRequest>,
    mut replace_results: MessageWriter<ReplaceAvatarResult>,
) {
    for request in load_requests.read() {
        match lifecycle.request_load(request.root) {
            Ok(()) => {
                commands.entity(request.root).insert(ActiveAvatar);
                load_results.write(LoadAvatarResult::Accepted { root: request.root });
            }
            Err(error) => {
                load_results.write(LoadAvatarResult::Rejected(error));
            }
        }
    }

    for _request in unload_requests.read() {
        let root = lifecycle.active_root();
        match lifecycle.request_unload() {
            Ok(()) => {
                if let Some(root) = root {
                    commands.entity(root).remove::<ActiveAvatar>();
                }
                unload_results.write(UnloadAvatarResult::Accepted);
            }
            Err(error) => {
                unload_results.write(UnloadAvatarResult::Rejected(error));
            }
        }
    }

    for request in replace_requests.read() {
        let previous_state = lifecycle.state();
        let previous_active = lifecycle.active_root();
        match lifecycle.request_replace(request.root) {
            Ok(()) => {
                let new_state = lifecycle.state();
                match (previous_state, new_state) {
                    (AvatarLifecycleState::Ready, AvatarLifecycleState::Unloading) => {
                        // Old active avatar is being replaced. Remove its marker
                        // now; the new root stays inactive until the unload completes.
                        if let Some(old) = previous_active {
                            commands.entity(old).remove::<ActiveAvatar>();
                        }
                    }
                    (
                        AvatarLifecycleState::NoAvatar | AvatarLifecycleState::Failed,
                        AvatarLifecycleState::Loading,
                    ) => {
                        // No previous active avatar; the new root is the loading
                        // active root immediately.
                        commands.entity(request.root).insert(ActiveAvatar);
                    }
                    (AvatarLifecycleState::Unloading, AvatarLifecycleState::Unloading) => {
                        // Coalesced replacement while already unloading. The old
                        // active avatar is already being unloaded and the new root
                        // remains pending.
                    }
                    _ => {}
                }
                replace_results.write(ReplaceAvatarResult::Accepted {
                    new_root: request.root,
                });
            }
            Err(error) => {
                replace_results.write(ReplaceAvatarResult::Rejected(error));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LookDirectionSet;
    use crate::PendingAvatarLoad;
    use crate::capabilities::{
        BlinkMode, BonePresence, DeclaredLookAtType, MouthMode, SelectedGazeBackend,
    };
    use crate::load::{
        AvatarAssetId, AvatarLoadRequestId, ImportedAvatar, LoadImportedAvatarRequest,
        LoadImportedAvatarResult, UserAssetPath, handle_load_imported_avatar_requests,
    };
    use crate::unload::despawn_unloading_avatar;

    fn entity(id: u32) -> Entity {
        Entity::from_raw_u32(id).expect("test entity index is valid")
    }

    #[test]
    fn lifecycle_state_load_from_no_avatar() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = entity(1);

        assert_eq!(lifecycle.state(), AvatarLifecycleState::NoAvatar);
        assert!(lifecycle.request_load(root).is_ok());
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
        assert_eq!(lifecycle.active_root(), Some(root));
    }

    #[test]
    fn lifecycle_state_rejects_invalid_request_order() {
        let mut lifecycle = AvatarLifecycle::new();

        // Unload with no avatar is invalid.
        assert!(matches!(
            lifecycle.request_unload(),
            Err(AvatarRequestError::InvalidState {
                current: AvatarLifecycleState::NoAvatar,
            })
        ));

        // Replace while loading is invalid.
        let root = entity(1);
        lifecycle.request_load(root).unwrap();
        assert!(matches!(
            lifecycle.request_replace(entity(2)),
            Err(AvatarRequestError::InvalidState {
                current: AvatarLifecycleState::Loading,
            })
        ));
    }

    #[test]
    fn lifecycle_state_active_avatar_is_unique() {
        let mut lifecycle = AvatarLifecycle::new();
        let first = entity(1);
        let second = entity(2);

        lifecycle.request_load(first).unwrap();
        lifecycle.start_binding(first);
        lifecycle.finish_ready();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);
        assert_eq!(lifecycle.active_root(), Some(first));

        // A second load while one is ready is rejected.
        assert!(lifecycle.request_load(second).is_err());
        assert_eq!(lifecycle.active_root(), Some(first));

        // A replace is accepted and queues the new root.
        assert!(lifecycle.request_replace(second).is_ok());
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Unloading);
        assert_eq!(lifecycle.pending_root(), Some(second));
        assert_eq!(lifecycle.active_root(), Some(first));

        lifecycle.finish_unload();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
        assert_eq!(lifecycle.active_root(), Some(second));
        assert!(lifecycle.pending_root().is_none());
    }

    #[test]
    fn lifecycle_state_replace_from_no_avatar_behaves_like_load() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = entity(7);

        assert!(lifecycle.request_replace(root).is_ok());
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
        assert_eq!(lifecycle.active_root(), Some(root));
        assert!(lifecycle.pending_root().is_none());
    }

    #[test]
    fn lifecycle_state_unload_returns_to_no_avatar() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = entity(3);

        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
        assert!(lifecycle.request_unload().is_ok());
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Unloading);
        assert_eq!(lifecycle.active_root(), Some(root));

        lifecycle.finish_unload();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::NoAvatar);
        assert!(lifecycle.active_root().is_none());
    }

    #[test]
    fn lifecycle_state_failure_clears_progress() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = entity(4);

        lifecycle.request_load(root).unwrap();
        lifecycle.fail(AvatarLifecycleFailure::AssetLoadFailed);
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Failed);
        assert_eq!(
            lifecycle.failure(),
            Some(&AvatarLifecycleFailure::AssetLoadFailed)
        );
        assert!(lifecycle.active_root().is_none());

        // A new load can be attempted after failure.
        let next = entity(5);
        assert!(lifecycle.request_load(next).is_ok());
        assert_eq!(lifecycle.active_root(), Some(next));
        assert!(lifecycle.failure().is_none());
    }

    #[test]
    fn lifecycle_state_snapshot_is_read_only() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = entity(6);

        lifecycle.request_load(root).unwrap();
        let snapshot = lifecycle.snapshot();
        assert_eq!(snapshot.state, AvatarLifecycleState::Loading);
        assert_eq!(snapshot.active_root, Some(root));
        assert!(snapshot.pending_root.is_none());

        // Mutating the snapshot does not affect the resource.
        let mut cloned = snapshot;
        cloned.state = AvatarLifecycleState::Ready;
        cloned.active_root = Some(entity(99));
        assert_eq!(cloned.state, AvatarLifecycleState::Ready);
        assert_eq!(cloned.active_root, Some(entity(99)));
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
        assert_eq!(lifecycle.active_root(), Some(root));
    }

    fn sample_capabilities() -> AvatarCapabilities {
        AvatarCapabilities {
            bones: BonePresence {
                head: true,
                neck: true,
                left_eye: true,
                right_eye: true,
                upper_chest: false,
                chest: false,
                spine: false,
            },
            blink: BlinkMode::PerEye,
            mouth: MouthMode::Full,
            declared_look_at: DeclaredLookAtType::Expression,
            gaze_backend: SelectedGazeBackend::Expression,
            gaze_fallback: None,
            look_directions: LookDirectionSet::default(),
            spring_bone: true,
            unknown_expressions: vec![],
            perfect_sync: Default::default(),
        }
    }

    #[test]
    fn avatar_capability_snapshot_lifecycle_update() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = entity(10);

        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        assert!(lifecycle.capabilities().is_none());

        let caps = sample_capabilities();
        lifecycle.set_capabilities(Some(caps.clone()));
        lifecycle.finish_ready();

        assert_eq!(lifecycle.state(), AvatarLifecycleState::Ready);
        let snapshot = lifecycle.snapshot();
        assert_eq!(snapshot.capabilities, Some(caps));

        // Model replacement clears the old snapshot.
        lifecycle.request_replace(entity(11)).unwrap();
        assert!(lifecycle.capabilities().is_none());
        assert!(lifecycle.snapshot().capabilities.is_none());
    }

    #[test]
    fn avatar_capability_snapshot_clears_on_failure() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = entity(12);

        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.set_capabilities(Some(sample_capabilities()));
        lifecycle.fail(AvatarLifecycleFailure::BindingTimeout);

        assert!(lifecycle.capabilities().is_none());
        assert!(lifecycle.snapshot().capabilities.is_none());
    }

    #[test]
    fn avatar_capability_snapshot_clears_on_unload() {
        let mut lifecycle = AvatarLifecycle::new();
        let root = entity(13);

        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.set_capabilities(Some(sample_capabilities()));
        lifecycle.finish_ready();
        assert!(lifecycle.capabilities().is_some());

        lifecycle.request_unload().unwrap();
        assert!(lifecycle.capabilities().is_none());

        lifecycle.finish_unload();
        assert!(lifecycle.capabilities().is_none());
        assert_eq!(lifecycle.state(), AvatarLifecycleState::NoAvatar);
    }

    fn test_app() -> App {
        use bevy::asset::AssetApp;

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<bevy_vrm1::prelude::VrmAsset>()
            .init_resource::<AvatarLifecycle>()
            .add_message::<LoadImportedAvatarRequest>()
            .add_message::<LoadImportedAvatarResult>()
            .add_message::<LoadAvatarRequest>()
            .add_message::<LoadAvatarResult>()
            .add_message::<ReplaceAvatarRequest>()
            .add_message::<ReplaceAvatarResult>()
            .add_message::<UnloadAvatarRequest>()
            .add_message::<UnloadAvatarResult>()
            .add_systems(
                Update,
                (
                    handle_load_imported_avatar_requests,
                    apply_avatar_request_events,
                    despawn_unloading_avatar,
                )
                    .chain(),
            );
        app
    }

    fn test_app_without_despawn() -> App {
        use bevy::asset::AssetApp;

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<bevy_vrm1::prelude::VrmAsset>()
            .init_resource::<AvatarLifecycle>()
            .add_message::<LoadImportedAvatarRequest>()
            .add_message::<LoadImportedAvatarResult>()
            .add_message::<LoadAvatarRequest>()
            .add_message::<LoadAvatarResult>()
            .add_message::<ReplaceAvatarRequest>()
            .add_message::<ReplaceAvatarResult>()
            .add_message::<UnloadAvatarRequest>()
            .add_message::<UnloadAvatarResult>()
            .add_systems(
                Update,
                (
                    handle_load_imported_avatar_requests,
                    apply_avatar_request_events,
                )
                    .chain(),
            );
        app
    }

    fn import_request(request_id: AvatarLoadRequestId, id: &str) -> LoadImportedAvatarRequest {
        let id = AvatarAssetId::new(id);
        let asset_path = UserAssetPath::avatar_model_path(&id).expect("test path is valid");
        LoadImportedAvatarRequest {
            request_id,
            imported: ImportedAvatar::new(
                id,
                asset_path,
                "Test Model",
                crate::load::ExpectedVrmGeneration::Vrm1,
            ),
        }
    }

    fn load_to_ready(app: &mut App, request_id: AvatarLoadRequestId, id: &str) -> Entity {
        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(request_id, id));
        app.update();

        let active_root = app
            .world()
            .resource::<AvatarLifecycle>()
            .active_root()
            .expect("active root after load");
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(active_root);
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();
        active_root
    }

    #[test]
    fn avatar_replace_transitions_through_unloading() {
        // Without the despawn system, the lifecycle stays in Unloading so we
        // can observe the intermediate state and marker cleanup.
        let mut app = test_app_without_despawn();
        let first_root = load_to_ready(&mut app, 1, "first");

        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(2, "second"));
        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Unloading);
        let active_root = lifecycle.active_root().expect("old root remains active");
        let pending_root = lifecycle.pending_root().expect("new root is pending");
        assert_eq!(active_root, first_root);
        assert_ne!(active_root, pending_root);

        let world = app.world();
        assert!(!world.entity(active_root).contains::<ActiveAvatar>());
        assert!(!world.entity(pending_root).contains::<ActiveAvatar>());
    }

    #[test]
    fn avatar_replace_despawns_old_and_activates_new() {
        let mut app = test_app();
        let first_root = load_to_ready(&mut app, 1, "first");

        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(2, "second"));
        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
        let second_root = lifecycle.active_root().expect("new active root");
        assert_ne!(second_root, first_root);
        assert!(lifecycle.pending_root().is_none());

        let world = app.world();
        assert!(!world.entities().contains(first_root));
        assert!(world.entity(second_root).contains::<ActiveAvatar>());
    }

    #[test]
    fn avatar_replace_coalesces_to_latest() {
        let mut app = test_app();
        let first_root = load_to_ready(&mut app, 1, "first");

        // Two replacement requests in the same update: only the latest root
        // should survive.
        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(2, "second"));
        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(3, "third"));
        app.update();

        let second_root = app
            .world()
            .resource::<AvatarLifecycle>()
            .active_root()
            .expect("replacement active");
        assert_ne!(second_root, first_root);

        let world = app.world_mut();
        assert!(!world.entities().contains(first_root));
        assert!(world.entity(second_root).contains::<ActiveAvatar>());

        let mut query = world.query::<(Entity, &PendingAvatarLoad)>();
        let pending: Vec<_> = query.iter(world).collect();
        assert_eq!(pending.len(), 1);
        let (pending_entity, pending_load) = pending[0];
        assert_eq!(pending_entity, second_root);
        assert_eq!(pending_load.request_id, 3);
    }

    #[test]
    fn avatar_replace_rapid_is_deterministic() {
        let mut app = test_app();
        let first_root = load_to_ready(&mut app, 1, "first");

        for i in 2..=10 {
            app.world_mut()
                .resource_mut::<Messages<LoadImportedAvatarRequest>>()
                .write(import_request(i, &format!("model{i}")));
        }
        app.update();

        let active_root = app.world().resource::<AvatarLifecycle>().active_root();

        let mut query = app.world_mut().query::<(Entity, &PendingAvatarLoad)>();
        let pending: Vec<_> = query.iter(app.world()).collect();
        assert_eq!(pending.len(), 1);
        let (pending_root, pending_load) = pending[0];
        assert_eq!(pending_load.request_id, 10);
        assert_eq!(active_root, Some(pending_root));

        let world = app.world_mut();
        assert!(!world.entities().contains(first_root));
        assert!(world.entity(pending_root).contains::<ActiveAvatar>());

        // At no point were two roots active simultaneously.
        let mut query = world.query_filtered::<Entity, With<ActiveAvatar>>();
        let active_count = query.iter(world).count();
        assert_eq!(active_count, 1);
    }

    #[test]
    fn avatar_replace_failure_leaves_empty() {
        let mut app = test_app();
        let first_root = load_to_ready(&mut app, 1, "first");

        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(2, "second"));
        app.update();

        let second_root = app
            .world()
            .resource::<AvatarLifecycle>()
            .active_root()
            .expect("replacement root active");

        // In real code the failing system removes the marker before calling
        // fail(); simulate that here so the test reflects the cleanup contract.
        app.world_mut()
            .entity_mut(second_root)
            .remove::<ActiveAvatar>();
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .fail(AvatarLifecycleFailure::AssetLoadFailed);

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Failed);
        assert!(lifecycle.active_root().is_none());
        assert!(lifecycle.pending_root().is_none());
        assert!(lifecycle.capabilities().is_none());

        let world = app.world();
        assert!(!world.entities().contains(first_root));
        assert!(!world.entity(second_root).contains::<ActiveAvatar>());
    }
}
