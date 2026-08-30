//! Spawning VRM roots from import results.
//!
//! This module bridges the application import domain (`ImportedAvatar`) with
//! `bevy_vrm1` by spawning a dedicated root entity, inserting a [`VrmHandle`],
//! and routing the request through the avatar lifecycle.
//!
//! The only asset paths accepted here use the `user://` scheme registered by
//! the application. Absolute filesystem paths are never passed directly to
//! [`AssetServer`].

use bevy::asset::AssetPath;
use bevy::prelude::*;
use bevy_vrm1::prelude::VrmHandle;
use std::fmt;

use crate::lifecycle::{
    AvatarLifecycle, AvatarLifecycleState, AvatarRequestError, LoadAvatarRequest,
    ReplaceAvatarRequest,
};

/// Stable identifier for an imported avatar asset (typically a SHA-256 hex digest).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Component)]
pub struct AvatarAssetId(pub String);

impl AvatarAssetId {
    /// Creates a new asset identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// A typed path that is guaranteed to reference the application-managed `user`
/// asset source.
///
/// Construction rejects empty paths, missing or wrong schemes, and paths that
/// escape the asset source root. This prevents absolute filesystem paths from
/// reaching [`AssetServer`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Component)]
pub struct UserAssetPath {
    /// The validated `user://...` path string.
    path: String,
    /// The parsed Bevy asset path, produced once at construction.
    asset_path: AssetPath<'static>,
}

impl UserAssetPath {
    /// Asset source scheme used for user-imported models.
    pub const SCHEME: &'static str = "user://";

    /// Validates and owns a `user://` asset path.
    ///
    /// # Errors
    ///
    /// Returns [`AssetPathError`] when the path is empty, uses the wrong scheme,
    /// has invalid syntax, or escapes the asset source root.
    pub fn new(path: impl Into<String>) -> Result<Self, AssetPathError> {
        let path = path.into();

        if !path.starts_with(Self::SCHEME) {
            return if path.contains("://") {
                Err(AssetPathError::WrongScheme)
            } else {
                Err(AssetPathError::MissingScheme)
            };
        }

        let after_scheme = &path[Self::SCHEME.len()..];
        if after_scheme.is_empty() {
            return Err(AssetPathError::EmptyPath);
        }
        if after_scheme.starts_with('/') || after_scheme.starts_with('\\') {
            return Err(AssetPathError::EscapesSourceRoot);
        }

        let asset_path = AssetPath::try_parse(&path)
            .map_err(|e| AssetPathError::InvalidSyntax(format!("{e}")))?
            .into_owned();
        if asset_path.is_unapproved() {
            return Err(AssetPathError::EscapesSourceRoot);
        }

        Ok(Self { path, asset_path })
    }

    /// Returns the canonical `user://avatars/<id>/model.vrm` path for an asset.
    ///
    /// # Errors
    ///
    /// Returns [`AssetPathError`] if the identifier produces an invalid path.
    pub fn avatar_model_path(id: &AvatarAssetId) -> Result<Self, AssetPathError> {
        Self::new(format!("{}avatars/{}/model.vrm", Self::SCHEME, id.0))
    }

    /// Returns the underlying path string, including the `user://` scheme.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.path
    }

    /// Converts this typed path into a Bevy [`AssetPath`].
    ///
    /// This cannot fail because the path was validated and parsed once at
    /// construction time; the parsed value is stored in the newtype.
    #[must_use]
    pub fn into_asset_path(self) -> AssetPath<'static> {
        self.asset_path
    }
}

/// Errors that can occur when constructing a [`UserAssetPath`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetPathError {
    /// The path does not contain an asset source scheme.
    MissingScheme,
    /// The path uses a scheme other than `user://`.
    WrongScheme,
    /// The path is empty after the `user://` scheme.
    EmptyPath,
    /// The path cannot be parsed as a Bevy asset path.
    InvalidSyntax(String),
    /// The path escapes the asset source root.
    EscapesSourceRoot,
}

impl fmt::Display for AssetPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingScheme => write!(f, "asset path is missing the `user://` scheme"),
            Self::WrongScheme => write!(f, "asset path does not use the `user://` scheme"),
            Self::EmptyPath => write!(f, "asset path is empty after `user://`"),
            Self::InvalidSyntax(reason) => {
                write!(f, "asset path has invalid syntax: {reason}")
            }
            Self::EscapesSourceRoot => write!(f, "asset path escapes the asset source root"),
        }
    }
}

impl std::error::Error for AssetPathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // The original parse error is stored as a string; the typed variant
            // above is the public contract.
            Self::InvalidSyntax(_) => None,
            _ => None,
        }
    }
}

impl AssetPathError {
    /// Stable string code for UI mapping and logs.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::MissingScheme => "AVATAR_PATH_MISSING_SCHEME",
            Self::WrongScheme => "AVATAR_PATH_WRONG_SCHEME",
            Self::EmptyPath => "AVATAR_PATH_EMPTY",
            Self::InvalidSyntax(_) => "AVATAR_PATH_INVALID_SYNTAX",
            Self::EscapesSourceRoot => "AVATAR_PATH_ESCAPES_ROOT",
        }
    }
}

/// The engine-facing description of an imported avatar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Component)]
pub enum ExpectedVrmGeneration {
    /// The preflight source was VRM 0.x.
    Vrm0,
    /// The preflight source was VRM 1.0.
    Vrm1,
}

/// The engine-facing description of an imported avatar.
#[derive(Clone, Debug, PartialEq, Eq, Component)]
pub struct ImportedAvatar {
    /// Stable asset identifier.
    pub id: AvatarAssetId,
    /// Typed `user://` path to the imported model.
    pub asset_path: UserAssetPath,
    /// User-facing model name.
    pub name: String,
    /// Generation established by app-side preflight.
    pub expected_generation: ExpectedVrmGeneration,
}

impl ImportedAvatar {
    /// Creates a new imported avatar descriptor.
    #[must_use]
    pub fn new(
        id: AvatarAssetId,
        asset_path: UserAssetPath,
        name: impl Into<String>,
        expected_generation: ExpectedVrmGeneration,
    ) -> Self {
        Self {
            id,
            asset_path,
            name: name.into(),
            expected_generation,
        }
    }
}

/// Opaque identifier used to correlate import requests with their results.
pub type AvatarLoadRequestId = u64;

/// Request to spawn and load a VRM root from an import result.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct LoadImportedAvatarRequest {
    /// Client-supplied correlation identifier.
    pub request_id: AvatarLoadRequestId,
    /// Imported avatar to load.
    pub imported: ImportedAvatar,
}

/// Marker that links a spawned root entity to its originating request.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingAvatarLoad {
    /// Request identifier that produced this root.
    pub request_id: AvatarLoadRequestId,
}

/// Result of a [`LoadImportedAvatarRequest`].
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub enum LoadImportedAvatarResult {
    /// The request was accepted, the root entity was spawned, and the lifecycle
    /// was notified.
    Accepted {
        /// Request identifier from the original request.
        request_id: AvatarLoadRequestId,
        /// Entity that will carry the loaded VRM.
        root: Entity,
    },
    /// The request was rejected before spawning a root.
    Rejected {
        /// Request identifier from the original request.
        request_id: AvatarLoadRequestId,
        /// Reason for rejection.
        error: LoadImportedAvatarError,
    },
}

/// Reasons a [`LoadImportedAvatarRequest`] can be rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadImportedAvatarError {
    /// The `user://` path is invalid.
    InvalidAssetPath(AssetPathError),
    /// The lifecycle is not in a state that can accept the request.
    Lifecycle(AvatarRequestError),
}

impl fmt::Display for LoadImportedAvatarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAssetPath(error) => write!(f, "invalid avatar asset path: {error}"),
            Self::Lifecycle(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for LoadImportedAvatarError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidAssetPath(error) => Some(error),
            Self::Lifecycle(error) => Some(error),
        }
    }
}

impl LoadImportedAvatarError {
    /// Stable string code for UI mapping and logs.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidAssetPath(_) => "LOAD_IMPORTED_AVATAR_INVALID_PATH",
            Self::Lifecycle(_) => "LOAD_IMPORTED_AVATAR_INVALID_LIFECYCLE",
        }
    }
}

/// Reads [`LoadImportedAvatarRequest`]s, spawns a dedicated root entity with a
/// [`VrmHandle`], and routes the entity through the avatar lifecycle.
///
/// - If no avatar is active, a [`LoadAvatarRequest`] is emitted.
/// - If an avatar is currently ready, a [`ReplaceAvatarRequest`] is emitted.
/// - If a replacement is already unloading, the previously spawned pending root
///   is discarded and a new [`ReplaceAvatarRequest`] is emitted for the latest
///   request (coalescing).
/// - Otherwise the request is rejected and no root is spawned.
pub(crate) fn handle_load_imported_avatar_requests(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    lifecycle: Res<AvatarLifecycle>,
    mut requests: MessageReader<LoadImportedAvatarRequest>,
    mut results: MessageWriter<LoadImportedAvatarResult>,
    mut load_requests: MessageWriter<LoadAvatarRequest>,
    mut replace_requests: MessageWriter<ReplaceAvatarRequest>,
) {
    // The lifecycle resource will not be updated until `apply_avatar_request_events`
    // runs later in the frame. Track the effective state locally so that multiple
    // import requests in a single frame do not spawn orphaned roots. Also track
    // the latest pending replacement root spawned this frame so continuous replace
    // requests can be coalesced to the latest one.
    let mut effective_state = lifecycle.state();
    let mut latest_pending_root: Option<Entity> = None;

    for request in requests.read() {
        let asset_path = match request.imported.asset_path.as_str().parse_asset_path() {
            Ok(path) => path,
            Err(error) => {
                results.write(LoadImportedAvatarResult::Rejected {
                    request_id: request.request_id,
                    error: LoadImportedAvatarError::InvalidAssetPath(error),
                });
                continue;
            }
        };

        let event = match effective_state {
            AvatarLifecycleState::NoAvatar | AvatarLifecycleState::Failed => {
                effective_state = AvatarLifecycleState::Loading;
                Some(LoadOrReplace::Load)
            }
            AvatarLifecycleState::Ready => {
                effective_state = AvatarLifecycleState::Unloading;
                Some(LoadOrReplace::Replace)
            }
            AvatarLifecycleState::Unloading => {
                // Coalesce to the latest request: discard the previously spawned
                // pending root. The lifecycle resource will be updated to point
                // to the new root by the `ReplaceAvatarRequest` emitted below.
                let root_to_discard = latest_pending_root.or_else(|| lifecycle.pending_root());
                if let Some(old) = root_to_discard
                    && let Ok(mut entity_commands) = commands.get_entity(old)
                {
                    entity_commands.despawn();
                }
                // effective_state stays Unloading; latest_pending_root is updated
                // after the new root is spawned.
                Some(LoadOrReplace::Replace)
            }
            AvatarLifecycleState::Loading | AvatarLifecycleState::Binding => None,
        };

        let Some(event) = event else {
            results.write(LoadImportedAvatarResult::Rejected {
                request_id: request.request_id,
                error: LoadImportedAvatarError::Lifecycle(AvatarRequestError::InvalidState {
                    current: lifecycle.state(),
                }),
            });
            continue;
        };

        let root = commands
            .spawn((
                PendingAvatarLoad {
                    request_id: request.request_id,
                },
                request.imported.id.clone(),
                request.imported.expected_generation,
                VrmHandle(asset_server.load(asset_path)),
                Transform::default(),
                GlobalTransform::default(),
                Visibility::Hidden,
            ))
            .id();

        latest_pending_root = Some(root);

        match event {
            LoadOrReplace::Load => {
                load_requests.write(LoadAvatarRequest { root });
            }
            LoadOrReplace::Replace => {
                replace_requests.write(ReplaceAvatarRequest { root });
            }
        }

        results.write(LoadImportedAvatarResult::Accepted {
            request_id: request.request_id,
            root,
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoadOrReplace {
    Load,
    Replace,
}

/// Internal helper that parses a validated `user://` path without panicking.
trait ParseUserAssetPath {
    /// Parses the path and returns a typed [`AssetPathError`].
    fn parse_asset_path(&self) -> Result<AssetPath<'static>, AssetPathError>;
}

impl ParseUserAssetPath for str {
    fn parse_asset_path(&self) -> Result<AssetPath<'static>, AssetPathError> {
        AssetPath::try_parse(self)
            .map_err(|e| AssetPathError::InvalidSyntax(format!("{e}")))
            .map(AssetPath::into_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::{
        ActiveAvatar, AvatarLifecycle, LoadAvatarResult, ReplaceAvatarResult, UnloadAvatarRequest,
        UnloadAvatarResult, apply_avatar_request_events,
    };

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
                ExpectedVrmGeneration::Vrm1,
            ),
        }
    }

    fn read_results(app: &App) -> Vec<LoadImportedAvatarResult> {
        let messages = app.world().resource::<Messages<LoadImportedAvatarResult>>();
        let mut cursor = messages.get_cursor();
        cursor.read(messages).cloned().collect()
    }

    #[test]
    fn avatar_load_request_spawns_one_root() {
        let mut app = test_app();
        let request = import_request(1, "abc123");
        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(request);

        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
        let root = lifecycle.active_root().expect("active root should exist");

        let mut query =
            app.world_mut()
                .query::<(&PendingAvatarLoad, &VrmHandle, &AvatarAssetId, &Visibility)>();
        let world = app.world();
        let (pending, _, id, visibility) = query.get(world, root).expect("root components");
        assert_eq!(pending.request_id, 1);
        assert_eq!(id.0, "abc123");
        assert_eq!(visibility, &Visibility::Hidden);

        assert!(world.entity(root).contains::<ActiveAvatar>());

        let results = read_results(&app);
        assert_eq!(results.len(), 1);
        assert!(matches!(
            results[0],
            LoadImportedAvatarResult::Accepted {
                request_id: 1,
                root: r,
            } if r == root
        ));
    }

    #[test]
    fn avatar_load_request_rejected_during_loading() {
        let mut app = test_app();
        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(1, "first"));
        app.update();

        // A second request while the first is still loading must be rejected.
        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(2, "second"));
        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);

        let mut query = app.world_mut().query::<&PendingAvatarLoad>();
        assert_eq!(query.iter(app.world()).count(), 1);

        let results = read_results(&app);
        let rejected = results.iter().find(|r| {
            matches!(
                r,
                LoadImportedAvatarResult::Rejected {
                    request_id: 2,
                    error: LoadImportedAvatarError::Lifecycle(
                        AvatarRequestError::InvalidState { .. }
                    ),
                }
            )
        });
        assert!(rejected.is_some(), "second request should be rejected");
    }

    #[test]
    fn avatar_load_request_replaces_ready_avatar() {
        let mut app = test_app();
        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(1, "first"));
        app.update();

        // Simulate successful initialization and binding of the first avatar.
        let active_root = app
            .world()
            .resource::<AvatarLifecycle>()
            .active_root()
            .expect("active root should exist");
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .start_binding(active_root);
        app.world_mut()
            .resource_mut::<AvatarLifecycle>()
            .finish_ready();

        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(import_request(2, "second"));
        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Unloading);
        let pending_root = lifecycle
            .pending_root()
            .expect("replacement root should be pending");
        let active_root = lifecycle
            .active_root()
            .expect("previous avatar should still be active");
        assert_ne!(pending_root, active_root);

        let world = app.world();
        // During unloading, neither the old active root nor the pending
        // replacement root carries the active marker. The marker is promoted
        // to the replacement only after the old root is despawned.
        assert!(!world.entity(active_root).contains::<ActiveAvatar>());
        assert!(!world.entity(pending_root).contains::<ActiveAvatar>());
    }

    #[test]
    fn avatar_load_request_missing_asset_does_not_panic() {
        let mut app = test_app();
        let request = import_request(42, "does-not-exist");
        app.world_mut()
            .resource_mut::<Messages<LoadImportedAvatarRequest>>()
            .write(request);

        // The user source is not registered in the test, and the file does not
        // exist. This must not panic.
        app.update();

        let lifecycle = app.world().resource::<AvatarLifecycle>();
        assert_eq!(lifecycle.state(), AvatarLifecycleState::Loading);
    }

    #[test]
    fn avatar_load_request_rejects_absolute_path_conversion() {
        let err = UserAssetPath::new("C:/models/avatar.vrm").expect_err("absolute path rejected");
        assert!(matches!(
            err,
            AssetPathError::MissingScheme | AssetPathError::WrongScheme
        ));

        let err =
            UserAssetPath::new("user:///C:/models/avatar.vrm").expect_err("rooted path rejected");
        assert!(matches!(err, AssetPathError::EscapesSourceRoot));
    }
}
