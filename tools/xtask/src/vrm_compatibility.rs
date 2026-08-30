//! Compatibility runner for VRM 0.x/1.0 fixtures.
//!
//! Loads each fixture model in a headless Bevy app, waits for `Initialized`,
//! and writes a compatibility report. This is intentionally separate from the
//! main app so that the runner can be executed as `cargo xtask vrm-compat`.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use bevy::app::{App, PluginGroup, Startup, Update};
use bevy::asset::AssetPlugin;
use bevy::log::LogPlugin;
use bevy::prelude::{AssetServer, Commands, DefaultPlugins, Handle, Res, ResMut, Resource};
use bevy::render::RenderPlugin;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::utils::default;
use bevy::window::{ExitCondition, WindowPlugin};
use bevy::winit::WinitPlugin;
use bevy_vrm1::prelude::*;
use vtuber_app::import::inspect_vrm;
use vtuber_avatar::compatibility::{VrmCompatibilityPlugin, VrmCompatibilityReport};

/// Exit code used when at least one fixture fails the gate.
pub const EXIT_COMPAT_FAIL: i32 = 2;
/// Maximum number of Bevy updates to wait for initialization.
pub const MAX_INIT_FRAMES: usize = 600;

/// Result of inspecting one model.
#[derive(Debug, Clone)]
pub struct CompatibilityResult {
    /// Path to the model.
    pub path: PathBuf,
    /// File size in bytes captured before loading.
    pub file_size: u64,
    /// SHA-256 captured before loading.
    pub sha256: String,
    /// Preflight summary, if the file is a valid VRM 0.x/1.0 model.
    pub preflight: Result<vtuber_app::import::VrmInspectionSummary, String>,
    /// Runtime compatibility report, if the model loaded.
    pub runtime: Option<VrmCompatibilityReport>,
    /// Runner failure unrelated to the model's preflight result.
    pub runner_error: Option<String>,
}

/// Run compatibility checks against every `.vrm` file in `fixture_dir`.
///
/// Returns `Ok(results)` on completion. A non-zero process exit code is the
/// caller's responsibility.
pub fn run(fixture_dir: impl AsRef<Path>) -> Result<Vec<CompatibilityResult>, String> {
    let fixture_dir = fixture_dir.as_ref();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(fixture_dir)
        .map_err(|e| format!("failed to read fixture dir: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("vrm"))
        })
        .collect();
    entries.sort();

    let mut results = Vec::with_capacity(entries.len());
    for path in entries {
        let result = match run_single(&path) {
            Ok(result) => result,
            Err(error) => {
                let message = format!("runner error: {error}");
                CompatibilityResult {
                    path: path.clone(),
                    file_size: 0,
                    sha256: String::new(),
                    preflight: Err(message.clone()),
                    runtime: None,
                    runner_error: Some(message),
                }
            }
        };
        results.push(result);
    }
    Ok(results)
}

/// Run the gate against a single model.
pub fn run_single(path: &Path) -> Result<CompatibilityResult, String> {
    let (file_size, sha256) = fingerprint(path)?;
    let preflight = inspect_vrm(path).map_err(|e| format!("{e}"));

    // If preflight fails, there is no point asking bevy_vrm1 to load it.
    let (runtime, runner_error) = if preflight.is_ok() {
        match load_and_inspect(path) {
            Ok(report) => (Some(report), None),
            Err(error) => (None, Some(format!("runner error: {error}"))),
        }
    } else {
        (None, None)
    };

    Ok(CompatibilityResult {
        path: path.to_path_buf(),
        file_size,
        sha256,
        preflight,
        runtime,
        runner_error,
    })
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn fingerprint(path: &Path) -> Result<(u64, String), String> {
    let file_size = std::fs::metadata(path)
        .map_err(|error| format!("failed to stat {}: {error}", path.display()))?
        .len();
    let mut file = File::open(path)
        .map_err(|error| format!("failed to open {} for hashing: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok((file_size, format!("{:X}", hasher.finalize())))
}

fn load_and_inspect(path: &Path) -> Result<VrmCompatibilityReport, String> {
    let mut app = App::new();

    // Use DefaultPlugins but disable the window so the runner stays headless.
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin { ..default() })
            .disable::<PipelinedRenderingPlugin>()
            // This runner creates one short-lived Bevy app per model. Keep
            // it headless and avoid recreating process-global Winit/log
            // state between fixture entries.
            .disable::<WinitPlugin>()
            .disable::<LogPlugin>()
            .set(AssetPlugin {
                file_path: std::env::current_dir()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| ".".to_string()),
                ..default()
            }),
    )
    .add_plugins(VrmPlugin)
    .add_plugins(VrmCompatibilityPlugin)
    .insert_resource(VrmCompatibilityReport::default())
    .insert_resource(ModelPath(path.to_string_lossy().to_string()))
    .add_systems(Startup, spawn_model)
    .add_systems(Update, tick_timeout);

    let report = wait_for_report(&mut app)?;
    Ok(report)
}

#[derive(Resource, Debug, Clone)]
struct ModelPath(String);

#[derive(Resource, Debug, Clone, Default)]
struct TimeoutFrames(usize);

fn spawn_model(mut commands: Commands, asset_server: Res<AssetServer>, model: Res<ModelPath>) {
    let handle: Handle<VrmAsset> = asset_server.load(model.0.clone());
    commands.spawn(VrmHandle(handle));
    commands.insert_resource(TimeoutFrames(0));
}

fn tick_timeout(mut timeout: ResMut<TimeoutFrames>) {
    timeout.0 += 1;
}

fn wait_for_report(app: &mut App) -> Result<VrmCompatibilityReport, String> {
    // Make sure plugin `finish` and `cleanup` run once before the first update.
    app.finish();
    app.cleanup();

    for _ in 0..MAX_INIT_FRAMES {
        app.update();
        if let Some(report) = app.world().get_resource::<VrmCompatibilityReport>()
            && report.initialized
        {
            return Ok(report.clone());
        }
        if app
            .world()
            .get_resource::<TimeoutFrames>()
            .is_some_and(|t| t.0 >= MAX_INIT_FRAMES)
        {
            break;
        }
    }
    Err(format!(
        "model did not initialize within {MAX_INIT_FRAMES} frames"
    ))
}
