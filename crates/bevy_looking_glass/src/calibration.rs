//! Calibration JSON parsing and normalization, without filesystem or ECS access.
use crate::{
    LookingGlassError,
    optics::{InterlaceUniforms, ViewLayout as Layout},
};
use bevy::prelude::*;
use serde::Deserialize;

/// Parse the raw visual.json schema belonging to a physical display.
pub fn parse_calibration(bytes: &[u8]) -> Result<Calibration, LookingGlassError> {
    Calibration::new(serde_json::from_slice(bytes)?)
}

#[derive(Deserialize)]
pub(super) struct Value {
    pub(crate) value: f32,
}

#[derive(Deserialize)]
pub(crate) struct RawCalibration {
    pitch: Value,
    slope: Value,
    center: Value,
    #[serde(rename = "DPI")]
    dpi: Value,
    #[serde(rename = "screenW")]
    width: Value,
    #[serde(rename = "screenH")]
    height: Value,
    #[serde(rename = "viewCone")]
    view_cone: Value,
    #[serde(rename = "invView")]
    inv_view: Value,
    #[serde(rename = "flipImageX")]
    flip_x: Value,
    #[serde(rename = "flipImageY")]
    flip_y: Value,
    #[serde(rename = "flipSubp")]
    flip_subp: Value,
    // Legacy stripe panels do not carry a cell table or a cell-pattern mode.
    #[serde(rename = "subpixelCells", default)]
    cells: Vec<RawCell>,
    #[serde(rename = "CellPatternMode", default)]
    cell_mode: Option<Value>,
}

#[derive(Deserialize)]
struct RawCell {
    #[serde(rename = "ROffsetX")]
    rx: f32,
    #[serde(rename = "ROffsetY")]
    ry: f32,
    #[serde(rename = "GOffsetX")]
    gx: f32,
    #[serde(rename = "GOffsetY")]
    gy: f32,
    #[serde(rename = "BOffsetX")]
    bx: f32,
    #[serde(rename = "BOffsetY")]
    by: f32,
}

/// Validated lens calibration. Construct with [`parse_calibration`].
#[derive(Clone, Debug)]
pub struct Calibration {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) view_cone: f32,
    pub(crate) optics: Vec4,
    pub(crate) flags: UVec4,
    // Two aligned vectors per RGB cell: (rx,ry,gx,gy), (bx,by,0,0).
    pub(crate) cells: Vec<Vec4>,
}

impl Calibration {
    pub(crate) fn new(raw: RawCalibration) -> Result<Self, LookingGlassError> {
        let w = raw.width.value;
        let h = raw.height.value;
        let scalars = [
            w,
            h,
            raw.pitch.value,
            raw.slope.value,
            raw.center.value,
            raw.dpi.value,
            raw.view_cone.value,
        ];
        if scalars.into_iter().any(|v| !v.is_finite())
            || w < 1.0
            || h < 1.0
            || w >= u32::MAX as f32
            || h >= u32::MAX as f32
            || w.fract() != 0.0
            || h.fract() != 0.0
            || raw.dpi.value <= 0.0
            || raw.slope.value == 0.0
            || raw.pitch.value <= 0.0
            || raw.view_cone.value <= 0.0
            || raw.view_cone.value >= 180.0
        {
            return Err(LookingGlassError::Invalid(
                "visual.json contains invalid screen or lens geometry",
            ));
        }
        let flag_values = [
            raw.inv_view.value,
            raw.flip_x.value,
            raw.flip_y.value,
            raw.flip_subp.value,
        ];
        if flag_values.into_iter().any(|v| v != 0.0 && v != 1.0) {
            return Err(LookingGlassError::Invalid(
                "visual.json orientation flags must be 0 or 1",
            ));
        }
        let mode = raw.cell_mode.map_or(0.0, |v| v.value);
        if !mode.is_finite() || mode.fract() != 0.0 || !(0.0..=4.0).contains(&mode) {
            return Err(LookingGlassError::Invalid(
                "this interlacer implements CellPatternMode 0 through 4",
            ));
        }
        let count = u32::try_from(raw.cells.len())
            .map_err(|_| LookingGlassError::Invalid("cell table exceeds u32 indexing"))?;
        let mut cells = Vec::new();
        for cell in raw.cells {
            let rg = Vec4::new(cell.rx / w, cell.ry / h, cell.gx / w, cell.gy / h);
            let b = Vec4::new(cell.bx / w, cell.by / h, 0.0, 0.0);
            if !rg.is_finite() || !b.is_finite() {
                return Err(LookingGlassError::Invalid(
                    "subpixel offsets must be finite",
                ));
            }
            cells.extend([rg, b]);
        }
        // A GPU storage binding cannot be empty. With count=0 the shader never
        // reads this padding; it evaluates the RGB-stripe equation instead.
        if cells.is_empty() {
            cells.push(Vec4::ZERO);
        }
        let sign = if raw.flip_x.value == 1.0 { -1.0 } else { 1.0 };
        let optics = Vec4::new(
            raw.pitch.value * w / raw.dpi.value * (1.0 / raw.slope.value).atan().cos(),
            h / (w * raw.slope.value) * sign,
            raw.center.value,
            sign / (3.0 * w),
        );
        if !optics.is_finite() {
            return Err(LookingGlassError::Invalid(
                "lens normalization is not finite",
            ));
        }
        Ok(Self {
            width: w as u32,
            height: h as u32,
            view_cone: raw.view_cone.value.to_radians(),
            optics,
            flags: UVec4::new(
                count,
                mode as u32,
                raw.inv_view.value as u32,
                raw.flip_x.value as u32
                    | ((raw.flip_y.value as u32) << 1)
                    | ((raw.flip_subp.value as u32) << 2),
            ),
            cells,
        })
    }

    pub(crate) fn uniforms(&self, layout: Layout) -> InterlaceUniforms {
        InterlaceUniforms {
            optics: self.optics,
            screen: Vec4::new(self.width as f32, self.height as f32, 0.0, 0.0),
            grid: UVec4::new(
                layout.columns,
                layout.rows,
                layout.view_width,
                layout.view_height,
            ),
            flags: self.flags,
        }
    }
}

impl Calibration {
    /// Native panel resolution, in physical pixels.
    pub fn screen_size(&self) -> UVec2 {
        UVec2::new(self.width, self.height)
    }
    /// Calibrated horizontal view cone in radians.
    pub fn view_cone(&self) -> f32 {
        self.view_cone
    }
}
