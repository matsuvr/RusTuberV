//! MyoArmRight_v0.01 joint geometry (MyoHub 93b0ca8, Apache-2.0).
//! Preserve radius orientation and offset-induced wrist direction, projecting
//! the resulting reach onto the VRM lower arm's fixed length.

use bevy::prelude::{Mat3, Quat, Vec3};

use crate::arm::ArmSide;

pub(crate) fn elbow_geometry(upper: Vec3, side: ArmSide) -> Option<(Vec3, Vec3)> {
    let (map, source_h) = source_mapping(upper, side)?;
    Some((
        map * source_h,
        (map * Vec3::new(0.0184, -0.2535, 0.045)).normalize(),
    ))
}

fn source_mapping(upper: Vec3, side: ArmSide) -> Option<(Mat3, Vec3)> {
    let u = upper.try_normalize()?;
    let h = u.cross(Vec3::Z).try_normalize()?;
    let destination = Mat3::from_cols(u, h, u.cross(h));
    // Source coordinates: +X anterior, +Y proximal, +Z right-lateral.
    let polar = |v: Vec3| {
        let v = Vec3::new(-v.z, v.y, v.x);
        if side == ArmSide::Left {
            Vec3::new(-v.x, v.y, v.z)
        } else {
            v
        }
    };
    let axial = |v: Vec3| {
        let v = Vec3::new(-v.z, v.y, v.x);
        if side == ArmSide::Left {
            Vec3::new(v.x, -v.y, -v.z)
        } else {
            v
        }
    };
    let source_u = polar(Vec3::new(0.0061, -0.2904, -0.0123)).normalize();
    let source_h = axial(Vec3::new(0.0494004, 0.0366003, 0.998108)).normalize();
    let perpendicular = (source_h - source_u * source_u.dot(source_h)).normalize();
    let source = Mat3::from_cols(source_u, perpendicular, source_u.cross(perpendicular));
    let map = destination * source.transpose();
    let polar_map = Mat3::from_cols(polar(Vec3::X), polar(Vec3::Y), polar(Vec3::Z));
    // Return the map of source polar vectors. The elbow axis has already
    // received the axial reflection before this inverse polar conversion.
    Some((map * polar_map, polar_map.transpose() * source_h))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RadiusGeometry {
    pub origin: Vec3,
    pub wrist: Vec3,
    pub axis: Vec3,
}

impl RadiusGeometry {
    pub fn from_arm(upper: Vec3, side: ArmSide) -> Option<Self> {
        let (map, _) = source_mapping(upper, side)?;
        Some(Self {
            origin: map * Vec3::new(0.0004, -0.0115, 0.02),
            wrist: map * Vec3::new(0.018, -0.242, 0.025),
            // The adapter uses distal-positive roll; reflecting this polar
            // axis therefore reverses the roll coordinate between sides.
            axis: (map * -Vec3::new(-0.017161, 0.992666, -0.119668)).normalize(),
        })
    }

    /// Bounds for R'(r) and R''(r), with respect to the roll coordinate.
    /// Both vectors of the corrective shortest arc stay within asin(o/w)
    /// of the rotated radius, so its quaternion denominator stays positive.
    pub fn derivative_bounds(self) -> (f32, f32) {
        let o = self.origin.length();
        let w = self.wrist.length();
        let u1 = w / (w - o);
        let u2 = u1 + 3.0 * u1 * u1;
        let m = 2.0 * (1.0 - (o / w).powi(2)).sqrt();
        let q1 = std::f32::consts::SQRT_2 * (1.0 + u1) / m;
        let q2 = std::f32::consts::SQRT_2 * (1.0 + u2 + 2.0 * u1) / m + 3.0 * q1 * q1;
        let angular = 2.0 * q1;
        (
            angular + 1.0,
            2.0 * q2 + 2.0 * q1 * q1 + 2.0 * angular + 1.0,
        )
    }

    /// Compose source radius roll with the shortest correction to its offset
    /// wrist direction. The caller retains the VRM's fixed lower-arm length.
    pub fn rotation(self, roll: f32) -> Option<Quat> {
        let neutral = (self.origin + self.wrist).try_normalize()?;
        let radius = Quat::from_axis_angle(self.axis, roll);
        let direction = (self.origin + radius * self.wrist).try_normalize()?;
        Some((crate::skeleton::minimal_arc(radius * neutral, direction)? * radius).normalize())
    }
}

/// Dot product in the source model after reversing its proximal roll axis to
/// match the adapter's distal-positive convention. This is measured geometry,
/// not a tunable fraction shared with the wrist.
pub(crate) fn radius_axial_projection() -> f32 {
    let radius_axis = Vec3::new(-0.017161, 0.992666, -0.119668).normalize();
    let ulna = Vec3::new(0.0004 + 0.018, -0.0115 - 0.242, 0.02 + 0.025).normalize();
    -radius_axis.dot(ulna)
}
