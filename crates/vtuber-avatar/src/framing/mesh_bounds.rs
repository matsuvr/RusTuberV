//! Geometry validation and world AABBs shared by framing targets.
use bevy::camera::primitives::MeshAabb;
use bevy::mesh::VertexAttributeValues;
use bevy::prelude::*;

/// A finite world-space axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WorldBounds {
    min: Vec3,
    max: Vec3,
}

impl WorldBounds {
    pub(crate) fn new(min: Vec3, max: Vec3) -> Option<Self> {
        if !min.is_finite() || !max.is_finite() || min.x > max.x || min.y > max.y || min.z > max.z {
            return None;
        }
        Some(Self { min, max })
    }

    pub(crate) fn min(self) -> Vec3 {
        self.min
    }

    pub(crate) fn max(self) -> Vec3 {
        self.max
    }

    pub(crate) fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub(crate) fn corners(self) -> [Vec3; 8] {
        let min = self.min;
        let max = self.max;
        [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(max.x, max.y, max.z),
        ]
    }

    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// Validates geometry and transforms its local AABB into a finite world AABB.
pub(super) fn mesh_world_bounds(
    mesh: &Mesh,
    global_transform: &GlobalTransform,
) -> Option<WorldBounds> {
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(values))
            if values.iter().flatten().all(|value| value.is_finite()) => {}
        _ => return None,
    }
    let bounds = mesh.compute_aabb()?;
    let local_bounds = WorldBounds::new(bounds.min().into(), bounds.max().into())?;
    let mut corners = local_bounds.corners().into_iter();
    let first = global_transform.transform_point(corners.next()?);
    if !first.is_finite() {
        return None;
    }
    let mut min = first;
    let mut max = first;
    for corner in corners {
        let point = global_transform.transform_point(corner);
        if !point.is_finite() {
            return None;
        }
        min = min.min(point);
        max = max.max(point);
    }
    WorldBounds::new(min, max)
}
