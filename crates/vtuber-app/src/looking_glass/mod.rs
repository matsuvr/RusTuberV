//! Optional, GPU-only Looking Glass output. No Bridge library is linked.
//!
//! Raw calibration I/O is confined here; optics contains pure calculations and
//! render owns Bevy entities. See `docs/looking-glass.md` for setup and scope.

mod devices;
mod optics;
mod render;
mod view_array;

pub use devices::{DevicePreset, DisplayFormat, OutputPreferences, QuiltPreset};

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use serde::Deserialize;

/// Failure to load the explicitly requested Looking Glass configuration.
#[derive(Debug, thiserror::Error)]
pub enum LookingGlassError {
    /// The command-line option needs a configuration path.
    #[error("--looking-glass requires a configuration file path")]
    MissingPath,
    /// A configuration or calibration file could not be read.
    #[error("could not read {path}: {source}")]
    Read {
        /// File being read.
        path: PathBuf,
        /// Original filesystem error.
        source: std::io::Error,
    },
    /// A document does not match the expected JSON schema.
    #[error("invalid JSON in {path}: {source}")]
    Json {
        /// File being parsed.
        path: PathBuf,
        /// Original JSON error.
        source: serde_json::Error,
    },
    /// Values cannot describe the requested projection or pixel layout.
    #[error("invalid Looking Glass configuration: {0}")]
    Invalid(&'static str),
}

#[derive(Deserialize)]
struct SettingsFile {
    calibration: PathBuf,
    columns: u32,
    rows: u32,
    view_width: u32,
    view_height: u32,
    depth_scale: f32,
}

#[derive(Resource, Clone)]
struct OutputConfig {
    layout: optics::Layout,
    calibration: optics::Calibration,
    depth_scale: f32,
}

#[derive(Resource, Clone)]
enum ActiveOutput {
    LightField(OutputConfig),
    Hld { screen: UVec2, safe_area: Vec2 },
}

impl ActiveOutput {
    fn screen(&self) -> UVec2 {
        match self {
            Self::LightField(config) => {
                UVec2::new(config.calibration.width, config.calibration.height)
            }
            Self::Hld { screen, .. } => *screen,
        }
    }

    fn layout(&self) -> optics::Layout {
        match self {
            Self::LightField(config) => config.layout,
            Self::Hld { screen, .. } => optics::Layout {
                columns: 1,
                rows: 1,
                view_width: screen.x,
                view_height: screen.y,
            },
        }
    }
}

/// Looking Glass settings and current output status presented by the UI.
#[derive(Clone, Debug, Default)]
pub struct LookingGlassViewModel {
    /// Presets loaded from the bundled JSON.
    pub devices: Vec<DevicePreset>,
    /// Last successfully saved selection.
    pub preferences: OutputPreferences,
    /// Whether an auxiliary output window exists.
    pub running: bool,
    /// Configuration or output error, if any.
    pub error: Option<String>,
}

/// Pending output command dispatched by the application.
#[derive(Resource, Default)]
pub struct LookingGlassControl {
    pending: Option<Option<OutputPreferences>>,
    startup: Option<ActiveOutput>,
}

impl LookingGlassControl {
    /// Start with an explicitly selected device and its own calibration.
    pub fn start(&mut self, preferences: OutputPreferences) {
        self.pending = Some(Some(preferences));
    }
    /// Release every output camera and target.
    pub fn stop(&mut self) {
        self.pending = Some(None);
    }
}

/// Install the settings-controlled output, with optional CLI startup configuration.
///
/// Call after the desktop's Bevy and avatar plugins, before `App::run`.
/// Without the option no output cameras, windows or images are allocated.
pub fn configure(
    app: &mut App,
    args: impl IntoIterator<Item = OsString>,
) -> Result<(), LookingGlassError> {
    let startup = if let Some(path) = config_argument(args)? {
        let settings: SettingsFile = read_json(&path)?;
        let mut calibration_path = path.clone();
        calibration_path.pop();
        calibration_path.push(&settings.calibration);
        let raw: optics::RawCalibration = read_json(&calibration_path)?;
        Some(ActiveOutput::LightField(output_config(settings, raw)?))
    } else {
        None
    };
    let preferences = app
        .world()
        .get_resource::<crate::settings::AppSettings>()
        .map(|settings| settings.looking_glass().clone())
        .unwrap_or_default();
    app.init_resource::<crate::ui_model::UiViewModel>();
    app.world_mut()
        .resource_mut::<crate::ui_model::UiViewModel>()
        .looking_glass = LookingGlassViewModel {
        devices: devices::catalog()?,
        preferences,
        ..default()
    };
    app.insert_resource(LookingGlassControl {
        startup,
        ..default()
    });
    render::install(app);
    Ok(())
}

fn selected_output(
    preset: &DevicePreset,
    preferences: &OutputPreferences,
) -> Result<ActiveOutput, LookingGlassError> {
    match preset.format {
        DisplayFormat::LightField => preset
            .light_field_config(read_json(&preferences.calibration_path)?)
            .map(ActiveOutput::LightField),
        DisplayFormat::Hld => {
            let safe_area = preset
                .safe_area
                .ok_or(LookingGlassError::Invalid("missing HLD safe area"))?;
            Ok(ActiveOutput::Hld {
                screen: UVec2::from_array(preset.screen),
                safe_area: Vec2::from_array(safe_area),
            })
        }
        DisplayFormat::Standalone => Err(LookingGlassError::Invalid(
            "musubi has no live desktop video input",
        )),
    }
}

fn config_argument(
    args: impl IntoIterator<Item = OsString>,
) -> Result<Option<PathBuf>, LookingGlassError> {
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--looking-glass" {
            return args
                .next()
                .map(PathBuf::from)
                .map(Some)
                .ok_or(LookingGlassError::MissingPath);
        }
    }
    Ok(None)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LookingGlassError> {
    let bytes = std::fs::read(path).map_err(|source| LookingGlassError::Read {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| LookingGlassError::Json {
        path: path.to_owned(),
        source,
    })
}

fn output_config(
    settings: SettingsFile,
    raw: optics::RawCalibration,
) -> Result<OutputConfig, LookingGlassError> {
    if !settings.depth_scale.is_finite() || settings.depth_scale < 0.0 {
        return Err(LookingGlassError::Invalid(
            "depth_scale must be finite and nonnegative",
        ));
    }
    Ok(OutputConfig {
        layout: optics::Layout::new(
            settings.columns,
            settings.rows,
            settings.view_width,
            settings.view_height,
        )?,
        calibration: optics::Calibration::new(raw)?,
        depth_scale: settings.depth_scale,
    })
}

#[cfg(test)]
mod tests;
