//! MyoArmRight_v0.01 joint geometry (MyoHub 93b0ca8, Apache-2.0).
//! The VRM ulna centreline has fixed length. Only the representable axial
//! component of the radius orientation is transferred to that single bone.

use bevy::prelude::{Mat3, Vec3};

use crate::arm::ArmSide;

pub(crate) fn elbow_geometry(upper: Vec3, side: ArmSide) -> Option<(Vec3, Vec3)> {
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
    let radius_origin = Vec3::new(0.0004, -0.0115, 0.02);
    let wrist_from_radius = Vec3::new(0.018, -0.242, 0.025);
    Some((
        map * source_h,
        (map * polar(radius_origin + wrist_from_radius)).normalize(),
    ))
}

/// Dot product in the source model after reversing its proximal roll axis to
/// match the adapter's distal-positive convention. This is measured geometry,
/// not a tunable fraction shared with the wrist.
pub(crate) fn radius_axial_projection() -> f32 {
    let radius_axis = Vec3::new(-0.017161, 0.992666, -0.119668).normalize();
    let ulna = Vec3::new(0.0004 + 0.018, -0.0115 - 0.242, 0.02 + 0.025).normalize();
    -radius_axis.dot(ulna)
}
