//! Manufacturer presets. Lens calibration is always supplied by the actual unit.

use serde::{Deserialize, Serialize};

use super::{LookingGlassError, OutputConfig, optics};

/// Saved selection, without automatically starting an output at application launch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputPreferences {
    /// Stable identifier in the bundled device catalog.
    pub device_id: String,
    /// Raw calibration JSON belonging to this physical display.
    pub calibration_path: std::path::PathBuf,
}

impl Default for OutputPreferences {
    fn default() -> Self {
        Self {
            device_id: "portrait".into(),
            calibration_path: Default::default(),
        }
    }
}

/// Display transport, independent of device branding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplayFormat {
    /// Calibrated multiview interlacing.
    LightField,
    /// Single portrait image on a physical holographic volume.
    Hld,
    /// Stored media only; no desktop video input.
    Standalone,
}

/// Reference quilt geometry, including deliberately stretched view tiles.
#[derive(Clone, Debug, Deserialize)]
pub struct QuiltPreset {
    /// Number of columns.
    pub columns: u32,
    /// Number of rows.
    pub rows: u32,
    /// Total quilt width.
    pub width: u32,
    /// Total quilt height.
    pub height: u32,
}

/// Device-specific data shared by UI and renderer.
#[derive(Clone, Debug, Deserialize)]
pub struct DevicePreset {
    /// Stable selection key.
    pub id: String,
    /// Manufacturer model name and orientation.
    pub name: String,
    /// Display protocol.
    pub format: DisplayFormat,
    /// Native desktop dimensions in the required OS orientation.
    pub screen: [u32; 2],
    /// Physical display aspect (not the tile aspect).
    pub aspect: f32,
    /// Reference light field layout; absent for HLDs.
    pub quilt: Option<QuiltPreset>,
    /// Fraction of the image reserved for HLD subject placement.
    pub safe_area: Option<[f32; 2]>,
    /// Whether this is retained for owners of discontinued hardware.
    pub legacy: bool,
    /// Official specification provenance.
    pub sources: Vec<String>,
}

#[derive(Deserialize)]
struct Catalog {
    devices: Vec<DevicePreset>,
}

pub(super) fn catalog() -> Result<Vec<DevicePreset>, LookingGlassError> {
    let catalog: Catalog =
        serde_json::from_str(include_str!("devices.json")).map_err(|source| {
            LookingGlassError::Json {
                path: "embedded:looking_glass/devices.json".into(),
                source,
            }
        })?;
    Ok(catalog.devices)
}

impl DevicePreset {
    pub(super) fn light_field_config(
        &self,
        raw: optics::RawCalibration,
    ) -> Result<OutputConfig, LookingGlassError> {
        let quilt = self.quilt.as_ref().ok_or(LookingGlassError::Invalid(
            "this device does not use quilts",
        ))?;
        if quilt.columns == 0 || quilt.rows == 0 {
            return Err(LookingGlassError::Invalid("empty quilt grid"));
        }
        // Some manufacturer presets (65 inch) have unused edge pixels.
        // Views occupy floor(total / tiles) pixels, as in the official quilt.
        let layout = optics::Layout::new(
            quilt.columns,
            quilt.rows,
            quilt.width / quilt.columns,
            quilt.height / quilt.rows,
        )?;
        let calibration = optics::Calibration::new(raw)?;
        if [calibration.width, calibration.height] != self.screen {
            return Err(LookingGlassError::Invalid(
                "visual.json screen size does not match the selected model/orientation",
            ));
        }
        Ok(OutputConfig {
            layout,
            calibration,
            depth_scale: 1.0,
        })
    }
}
