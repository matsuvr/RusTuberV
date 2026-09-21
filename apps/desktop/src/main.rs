// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
//! Desktop entry point for RusTuberV.

#![cfg_attr(windows, windows_subsystem = "windows")]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::path::PathBuf;

use bevy::asset::io::{AssetSourceBuilder, AssetSourceBuilders};
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use vtuber_app::import;
use vtuber_app::inference_runtime::InferenceProjectRoot;
use vtuber_app::orchestrator::Orchestrator;
use vtuber_app::settings::ArmPoseSettings;
use vtuber_app::tracking_file::{TRACKING_PROFILE_FILE_NAME, load_tracking_profile};
use vtuber_app::ui::UiShellPlugin;
use vtuber_avatar::{ArmPoseSourceKind, ArmSourceSelection, StartupModelPath, VtuberAvatarPlugin};

fn main() {
    let managed_root = managed_asset_root();
    std::fs::create_dir_all(&managed_root).ok();

    // User-tunable tracking distribution: a missing file is seeded with the
    // generated template so every knob is visible for editing.
    let tracking = match load_tracking_profile(&resource_root().join(TRACKING_PROFILE_FILE_NAME)) {
        Ok(tracking) => tracking,
        Err(error) => {
            eprintln!("Failed to load tracking profile: {error}");
            return;
        }
    };

    // Import CLI model through the managed asset source so that the same
    // `user://avatars/<sha256>/model.vrm` path invariant is used.
    let startup_model = parse_model_arg().and_then(|path| {
        match import::import_vrm(&path, &managed_root, import::DEFAULT_SIZE_LIMIT) {
            Ok(model) => Some(model),
            Err(e) => {
                eprintln!("Failed to import CLI model: {e}");
                None
            }
        }
    });

    // Register the `user` asset source BEFORE DefaultPlugins so that
    // `user://avatars/<sha256>/model.vrm` resolves to
    // `<managed_root>/avatars/<sha256>/model.vrm`.
    let mut sources = AssetSourceBuilders::default();
    sources.insert(
        "user",
        AssetSourceBuilder::platform_default(
            // Invariant: the managed root is built from joined UTF-8 path
            // components under the user profile directory.
            #[allow(clippy::expect_used)]
            managed_root.to_str().expect("managed root is valid UTF-8"),
            None,
        ),
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
        .insert_resource(ArmPoseSettings::load_default())
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
fn parse_model_arg() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--model" {
            return args.next();
        }
    }
    None
}
