//! Optional, GPU-only Looking Glass output. No Bridge library is linked.
//!
//! File/argument I/O is confined here; optics contains pure calculations and
//! render owns Bevy entities. See `docs/looking-glass.md` for the prototype.

mod optics;
mod render;

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

/// Install the optional output only when `--looking-glass <config.json>` is present.
///
/// Call after the desktop's Bevy and avatar plugins, before `App::run`.
/// Without the option this performs no file I/O and installs no resources or systems.
pub fn configure(
    app: &mut App,
    args: impl IntoIterator<Item = OsString>,
) -> Result<(), LookingGlassError> {
    let Some(path) = config_argument(args)? else {
        return Ok(());
    };
    let settings: SettingsFile = read_json(&path)?;
    let mut calibration_path = path.clone();
    calibration_path.pop();
    calibration_path.push(&settings.calibration);
    let raw: optics::RawCalibration = read_json(&calibration_path)?;
    let config = output_config(settings, raw)?;
    render::install(app, config);
    Ok(())
}

fn config_argument(
    args: impl IntoIterator<Item = OsString>,
) -> Result<Option<PathBuf>, LookingGlassError> {
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--looking-glass" {
            return args.next().map(PathBuf::from).map(Some)
                .ok_or(LookingGlassError::MissingPath);
        }
    }
    Ok(None)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LookingGlassError> {
    let bytes = std::fs::read(path).map_err(|source| LookingGlassError::Read {
        path: path.to_owned(), source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| LookingGlassError::Json {
        path: path.to_owned(), source,
    })
}

fn output_config(settings: SettingsFile, raw: optics::RawCalibration) -> Result<OutputConfig, LookingGlassError> {
    if !settings.depth_scale.is_finite() || settings.depth_scale < 0.0 {
        return Err(LookingGlassError::Invalid("depth_scale must be finite and nonnegative"));
    }
    Ok(OutputConfig {
        layout: optics::Layout::new(settings.columns, settings.rows, settings.view_width, settings.view_height)?,
        calibration: optics::Calibration::new(raw)?,
        depth_scale: settings.depth_scale,
    })
}

#[cfg(test)]
mod tests;
