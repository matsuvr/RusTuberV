//! Desktop entry point for RusTuberV.

#![cfg_attr(windows, windows_subsystem = "windows")]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bevy::asset::io::{AssetSourceBuilder, AssetSourceBuilders};
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use vtuber_app::import;
use vtuber_app::inference_runtime::InferenceProjectRoot;
use vtuber_app::orchestrator::Orchestrator;
use vtuber_app::settings::AppSettings;
use vtuber_app::tracking_file::{TRACKING_PROFILE_FILE_NAME, load_tracking_profile};
use vtuber_app::ui::UiShellPlugin;
use vtuber_avatar::{ArmPoseSourceKind, ArmSourceSelection, StartupModelPath, VtuberAvatarPlugin};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Failed to start RusTuberV: {error}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
enum StartupError {
    MissingModelPath,
    NonUnicodeAssetRoot(PathBuf),
    CreateManagedRoot(std::io::Error),
    Tracking(vtuber_app::tracking_file::TrackingProfileFileError),
    Settings(vtuber_app::settings::SettingsError),
    Import(import::ModelImportError),
}

impl fmt::Display for StartupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingModelPath => f.write_str("--model requires a path"),
            Self::NonUnicodeAssetRoot(path) => {
                write!(f, "asset source root is not valid UTF-8: {path:?}")
            }
            Self::CreateManagedRoot(error) => {
                write!(f, "could not create managed asset directory: {error}")
            }
            Self::Tracking(error) => write!(f, "could not load tracking profile: {error}"),
            Self::Settings(error) => write!(f, "could not load settings: {error}"),
            Self::Import(error) => write!(f, "could not import CLI model: {error}"),
        }
    }
}

impl std::error::Error for StartupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CreateManagedRoot(error) => Some(error),
            Self::Tracking(error) => Some(error),
            Self::Settings(error) => Some(error),
            Self::Import(error) => Some(error),
            Self::MissingModelPath | Self::NonUnicodeAssetRoot(_) => None,
        }
    }
}

fn run() -> Result<(), StartupError> {
    let model_path = parse_model_arg(std::env::args_os().skip(1))?;
    let managed_root = managed_asset_root();
    std::fs::create_dir_all(&managed_root).map_err(StartupError::CreateManagedRoot)?;

    // User-tunable tracking distribution: a missing file is seeded with the
    // generated template so every knob is visible for editing.
    let tracking = load_tracking_profile(&resource_root().join(TRACKING_PROFILE_FILE_NAME))
        .map_err(StartupError::Tracking)?;

    // An unreadable, malformed, or newer settings document is reported instead
    // of being replaced with the current defaults.
    let settings = AppSettings::load_default().map_err(StartupError::Settings)?;

    // Import CLI model through the managed asset source so that the same
    // `user://avatars/<sha256>/model.vrm` path invariant is used.
    let startup_model = model_path
        .map(|path| import::import_vrm(path, &managed_root, import::DEFAULT_SIZE_LIMIT))
        .transpose()
        .map_err(StartupError::Import)?;

    // Register the `user` asset source BEFORE DefaultPlugins so that
    // `user://avatars/<sha256>/model.vrm` resolves to
    // `<managed_root>/avatars/<sha256>/model.vrm`.
    let mut sources = AssetSourceBuilders::default();
    sources.insert(
        "user",
        AssetSourceBuilder::platform_default(asset_source_root(&managed_root)?, None),
    );

    let mut app = App::new();
    app.insert_resource(sources)
        .add_plugins(DefaultPlugins)
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            SystemInformationDiagnosticsPlugin,
        ))
        .add_plugins(EguiPlugin::default())
        .add_plugins(VtuberAvatarPlugin)
        .insert_resource(tracking.body)
        .insert_resource(ArmSourceSelection {
            // The hips-relative virtual-hand source is the documented default
            // authority (arm_pipeline.rs); the enum's `#[default]` names the
            // fallback-only static pose, which must not win at startup.
            mode: ArmPoseSourceKind::VirtualHandAnchor,
            profile: tracking.arm,
        })
        .insert_resource(settings)
        .insert_resource(InferenceProjectRoot(resource_root()))
        .add_plugins(UiShellPlugin)
        .insert_resource(Orchestrator::new(managed_root));

    if let Some(imported) = startup_model {
        app.insert_resource(StartupModelPath(Some(imported.id.clone())));
        app.world_mut()
            .resource_mut::<Orchestrator>()
            .queue_imported_model(imported);
    }

    // Dev-only synthetic tracking: generates AvatarControlFrame values from
    // sine waves so the avatar apply path can be verified without a camera.
    #[cfg(feature = "dev-synthetic-input")]
    {
        use vtuber_app::synthetic_tracking::{SyntheticTrackingSource, synthetic_tracking_system};
        app.init_resource::<SyntheticTrackingSource>()
            .add_systems(Update, synthetic_tracking_system);
        bevy::log::warn!("dev-synthetic-input enabled: using synthetic tracking source");
    }

    app.run();
    Ok(())
}

/// Locates packaged model resources without depending on the process cwd.
fn resource_root() -> PathBuf {
    let mut candidates = Vec::new();
    if let Ok(executable) = std::env::current_exe()
        && let Some(parent) = executable.parent()
    {
        candidates.push(parent.to_path_buf());
        candidates.push(parent.join("resources"));
    }
    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir);
    }
    candidates
        .into_iter()
        .find(|root| root.join("assets/models/manifest.toml").is_file())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Returns the application-managed asset root directory.
///
/// On Windows this is `%APPDATA%\RusTuberV`. Falls back to
/// `.vtuber` in the current directory when the platform directories
/// crate cannot determine a suitable location.
fn managed_asset_root() -> PathBuf {
    if let Some(proj_dirs) = directories::ProjectDirs::from("", "", "RusTuberV") {
        proj_dirs.data_dir().to_path_buf()
    } else {
        PathBuf::from(".vtuber")
    }
}

/// Parses the optional `--model <path>` command-line argument.
///
/// Accepts both absolute paths and paths relative to the workspace root.
/// When omitted, no default model is loaded.
fn parse_model_arg(
    args: impl IntoIterator<Item = OsString>,
) -> Result<Option<PathBuf>, StartupError> {
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--model" {
            return args
                .next()
                .map(PathBuf::from)
                .map(Some)
                .ok_or(StartupError::MissingModelPath);
        }
    }
    Ok(None)
}

/// Bevy's platform asset source requires UTF-8; preserve OS paths until this boundary.
fn asset_source_root(path: &Path) -> Result<&str, StartupError> {
    path.to_str()
        .ok_or_else(|| StartupError::NonUnicodeAssetRoot(path.to_path_buf()))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use std::error::Error;

    #[test]
    fn model_argument_is_optional_and_unrelated_arguments_are_unchanged() {
        assert_eq!(parse_model_arg([]).unwrap(), None);
        assert_eq!(parse_model_arg([OsString::from("--other")]).unwrap(), None);
        assert_eq!(
            parse_model_arg(["--other", "--model", "モデル.vrm"].map(OsString::from)).unwrap(),
            Some(PathBuf::from("モデル.vrm"))
        );
        assert!(matches!(
            parse_model_arg([OsString::from("--model")]),
            Err(StartupError::MissingModelPath)
        ));
    }

    #[test]
    fn startup_error_preserves_the_io_cause() {
        let error = StartupError::CreateManagedRoot(std::io::Error::from(
            std::io::ErrorKind::PermissionDenied,
        ));
        assert!(
            error
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .is_some()
        );
    }

    #[cfg(unix)]
    #[test]
    fn os_model_path_is_preserved_and_utf8_only_asset_boundary_rejects_it() {
        use std::os::unix::ffi::OsStringExt;
        let path = PathBuf::from(OsString::from_vec(vec![b'a', 0xff, b'b']));
        let parsed =
            parse_model_arg([OsString::from("--model"), path.clone().into_os_string()]).unwrap();
        assert_eq!(parsed, Some(path.clone()));
        assert!(
            matches!(asset_source_root(&path), Err(StartupError::NonUnicodeAssetRoot(rejected)) if rejected == path)
        );
    }
}
