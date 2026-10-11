//! Pure view layout and off-axis projection calculations.

use bevy::camera::{CameraProjection, SubCameraView};
use bevy::math::Vec3A;
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

use super::LookingGlassError;

/// Validated virtual quilt grid and per-view resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewLayout {
    pub(crate) columns: u32,
    pub(crate) rows: u32,
    pub(crate) view_width: u32,
    pub(crate) view_height: u32,
}

impl ViewLayout {
    /// Construct a layout with at least two views and nonzero, nonoverflowing dimensions.
    pub fn new(
        columns: u32,
        rows: u32,
        view_width: u32,
        view_height: u32,
    ) -> Result<Self, LookingGlassError> {
        let products = [
            columns.checked_mul(rows),
            columns.checked_mul(view_width),
            rows.checked_mul(view_height),
        ];
        if [columns, rows, view_width, view_height].contains(&0)
            || products.into_iter().any(|value| value.is_none())
            || columns.saturating_mul(rows) < 2
        {
            return Err(LookingGlassError::Invalid(
                "quilt needs at least two views, positive dimensions, and u32-sized products",
            ));
        }
        Ok(Self {
            columns,
            rows,
            view_width,
            view_height,
        })
    }

    /// Total number of views.
    pub fn count(self) -> u32 {
        self.columns * self.rows
    }
    /// Virtual quilt width in pixels.
    pub fn width(self) -> u32 {
        self.columns * self.view_width
    }
    /// Virtual quilt height in pixels.
    pub fn height(self) -> u32 {
        self.rows * self.view_height
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
    fn far(&self) -> f32 {
        self.perspective.far
    }
    fn get_frustum_corners(&self, z_near: f32, z_far: f32) -> [Vec3A; 8] {
        self.perspective
            .get_frustum_corners(z_near, z_far)
            .map(|mut p| {
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
    let ratio = if count > 1 {
        (2.0 * index as f32 / (count - 1) as f32 - 1.0) * (view_cone * 0.5).tan() * depth_scale
    } else {
        0.0 // HLD uses one centered camera without angular parallax.
    };
    let mut transform = center;
    transform.translation += center.rotation * Vec3::X * (ratio * focus_distance);
    (
        transform,
        OffAxisProjection {
            perspective,
            offset_over_focus: ratio,
        },
    )
}

impl ViewLayout {
    /// Number of columns in the virtual quilt.
    pub fn columns(self) -> u32 {
        self.columns
    }
    /// Number of rows in the virtual quilt.
    pub fn rows(self) -> u32 {
        self.rows
    }
    /// Resolution of one view, independent of panel aspect ratio.
    pub fn view_size(self) -> UVec2 {
        UVec2::new(self.view_width, self.view_height)
    }
}
