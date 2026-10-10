//! Pure quilt geometry, raw-calibration conversion, and off-axis projection.

use bevy::camera::{CameraProjection, SubCameraView};
use bevy::math::Vec3A;
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use serde::Deserialize;

use super::LookingGlassError;

#[derive(Clone, Copy)]
pub(super) struct Layout {
    pub columns: u32,
    pub rows: u32,
    pub view_width: u32,
    pub view_height: u32,
}

impl Layout {
    pub fn new(columns: u32, rows: u32, view_width: u32, view_height: u32) -> Result<Self, LookingGlassError> {
        let products = [columns.checked_mul(rows), columns.checked_mul(view_width), rows.checked_mul(view_height)];
        if [columns, rows, view_width, view_height].contains(&0)
            || products.into_iter().any(|value| value.is_none())
            || columns.saturating_mul(rows) < 2
        {
            return Err(LookingGlassError::Invalid("quilt needs at least two views, positive dimensions, and u32-sized products"));
        }
        Ok(Self { columns, rows, view_width, view_height })
    }

    pub fn count(self) -> u32 { self.columns * self.rows }
    pub fn width(self) -> u32 { self.columns * self.view_width }
    pub fn height(self) -> u32 { self.rows * self.view_height }

    // Looking Glass view zero occupies the bottom-left tile. Bevy's 2D world
    // has +Y up; the interlacer converts the texture's top-left UV convention.
    pub fn tile_center(self, index: u32) -> Vec3 {
        Vec3::new(
            (index % self.columns) as f32 * self.view_width as f32 + self.view_width as f32 * 0.5 - self.width() as f32 * 0.5,
            (index / self.columns) as f32 * self.view_height as f32 + self.view_height as f32 * 0.5 - self.height() as f32 * 0.5,
            0.0,
        )
    }
}

#[derive(Deserialize)]
pub(super) struct Value { pub value: f32 }

#[derive(Deserialize)]
pub(super) struct RawCalibration {
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

#[derive(Clone)]
pub(super) struct Calibration {
    pub width: u32,
    pub height: u32,
    pub view_cone: f32,
    pub optics: Vec4,
    pub flags: UVec4,
    // Two aligned vectors per RGB cell: (rx,ry,gx,gy), (bx,by,0,0).
    pub cells: Vec<Vec4>,
}

impl Calibration {
    pub fn new(raw: RawCalibration) -> Result<Self, LookingGlassError> {
        let w = raw.width.value;
        let h = raw.height.value;
        let scalars = [w, h, raw.pitch.value, raw.slope.value, raw.center.value, raw.dpi.value, raw.view_cone.value];
        if scalars.into_iter().any(|v| !v.is_finite())
            || w < 1.0 || h < 1.0 || w >= u32::MAX as f32 || h >= u32::MAX as f32
            || w.fract() != 0.0 || h.fract() != 0.0
            || raw.dpi.value <= 0.0 || raw.slope.value == 0.0 || raw.pitch.value <= 0.0
            || raw.view_cone.value <= 0.0 || raw.view_cone.value >= 180.0
        {
            return Err(LookingGlassError::Invalid("visual.json contains invalid screen or lens geometry"));
        }
        let flag_values = [raw.inv_view.value, raw.flip_x.value, raw.flip_y.value, raw.flip_subp.value];
        if flag_values.into_iter().any(|v| v != 0.0 && v != 1.0) {
            return Err(LookingGlassError::Invalid("visual.json orientation flags must be 0 or 1"));
        }
        let mode = raw.cell_mode.map_or(0.0, |v| v.value);
        if !mode.is_finite() || mode.fract() != 0.0 || !(0.0..=4.0).contains(&mode) {
            return Err(LookingGlassError::Invalid("this interlacer implements CellPatternMode 0 through 4"));
        }
        let count = u32::try_from(raw.cells.len()).map_err(|_| LookingGlassError::Invalid("cell table exceeds u32 indexing"))?;
        let mut cells = Vec::new();
        for cell in raw.cells {
            let rg = Vec4::new(cell.rx / w, cell.ry / h, cell.gx / w, cell.gy / h);
            let b = Vec4::new(cell.bx / w, cell.by / h, 0.0, 0.0);
            if !rg.is_finite() || !b.is_finite() {
                return Err(LookingGlassError::Invalid("subpixel offsets must be finite"));
            }
            cells.extend([rg, b]);
        }
        // A GPU storage binding cannot be empty. With count=0 the shader never
        // reads this padding; it evaluates the RGB-stripe equation instead.
        if cells.is_empty() { cells.push(Vec4::ZERO); }
        let sign = if raw.flip_x.value == 1.0 { -1.0 } else { 1.0 };
        let optics = Vec4::new(
            raw.pitch.value * w / raw.dpi.value * (1.0 / raw.slope.value).atan().cos(),
            h / (w * raw.slope.value) * sign,
            raw.center.value,
            sign / (3.0 * w),
        );
        if !optics.is_finite() {
            return Err(LookingGlassError::Invalid("lens normalization is not finite"));
        }
        Ok(Self {
            width: w as u32, height: h as u32,
            view_cone: raw.view_cone.value.to_radians(), optics,
            flags: UVec4::new(count, mode as u32, raw.inv_view.value as u32,
                raw.flip_x.value as u32 | ((raw.flip_y.value as u32) << 1) | ((raw.flip_subp.value as u32) << 2)),
            cells,
        })
    }

    pub fn uniforms(&self, layout: Layout) -> InterlaceUniforms {
        InterlaceUniforms {
            optics: self.optics,
            screen: Vec4::new(self.width as f32, self.height as f32, 0.0, 0.0),
            grid: UVec4::new(layout.columns, layout.rows, layout.view_width, layout.view_height),
            flags: self.flags,
        }
    }
}

#[derive(Clone, Debug, ShaderType)]
pub(super) struct InterlaceUniforms {
    pub optics: Vec4,
    pub screen: Vec4,
    pub grid: UVec4,
    pub flags: UVec4,
}

#[derive(Clone, Debug)]
pub(super) struct OffAxisProjection {
    pub perspective: PerspectiveProjection,
    pub offset_over_focus: f32,
}

impl CameraProjection for OffAxisProjection {
    fn get_clip_from_view(&self) -> Mat4 {
        let mut matrix = self.perspective.get_clip_from_view();
        matrix.z_axis.x -= self.offset_over_focus * matrix.x_axis.x;
        matrix
    }
    fn get_clip_from_view_for_sub(&self, sub: &SubCameraView) -> Mat4 {
        // Crop the optical projection, rather than letting the tile's pixel
        // aspect replace the panel aspect in PerspectiveProjection's helper.
        let full = sub.full_size.as_vec2();
        let size = sub.size.as_vec2();
        let scale = full / size;
        let shift = (full - size - 2.0 * sub.offset) / size;
        let mut crop = Mat4::from_scale(Vec3::new(scale.x, scale.y, 1.0));
        crop.w_axis.x = shift.x;
        crop.w_axis.y = -shift.y;
        crop * self.get_clip_from_view()
    }
    fn update(&mut self, _width: f32, _height: f32) {
        // Optical aspect is the panel's, not the intentionally stretched tile's.
    }
    fn far(&self) -> f32 { self.perspective.far }
    fn get_frustum_corners(&self, z_near: f32, z_far: f32) -> [Vec3A; 8] {
        self.perspective.get_frustum_corners(z_near, z_far).map(|mut p| {
            p.x += self.offset_over_focus * p.z;
            p
        })
    }
}

pub(super) fn view_pose(
    center: Transform,
    focus_distance: f32,
    perspective: PerspectiveProjection,
    index: u32,
    count: u32,
    view_cone: f32,
    depth_scale: f32,
) -> (Transform, OffAxisProjection) {
    let ratio = (2.0 * index as f32 / (count - 1) as f32 - 1.0)
        * (view_cone * 0.5).tan() * depth_scale;
    let mut transform = center;
    transform.translation += center.rotation * Vec3::X * (ratio * focus_distance);
    (transform, OffAxisProjection { perspective, offset_over_focus: ratio })
}
