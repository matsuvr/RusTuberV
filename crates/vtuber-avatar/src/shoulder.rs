//! Thoracohumeral globe coordinates, independent of authored bone axes.
//!
//! The neutral humerus points down and the flexed forearm points anteriorly.
//! A shortest swing from that neutral direction defines zero axial rotation.
//! Axial rotation is positive internally on either anatomical side. See the
//! upper-limb ADR for the relation to the ISB Y-X-Y coordinates.

use bevy::prelude::{Mat3, Quat, Vec3};

use crate::arm::{ArmChainBinding, ArmIkInput, ArmIkTarget, ArmSide};

/// Globe angles in radians. The elevation plane is undefined at either pole.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShoulderCoordinates {
    pub plane: Option<f32>,
    pub elevation: f32,
    pub axial: f32,
}

fn sign(chain: &ArmChainBinding) -> f32 {
    match chain.side {
        ArmSide::Left => 1.0,
        ArmSide::Right => -1.0,
    }
}

pub(crate) fn neutral(chain: &ArmChainBinding) -> Option<Quat> {
    let rest = chain.rest;
    let upper = (rest.elbow.position - rest.upper_arm.position).try_normalize()?;
    let input = ArmIkInput::from_chain(
        chain,
        ArmIkTarget {
            wrist: rest.wrist.position,
            elbow_pole: rest.elbow.position,
        },
    );
    let hinge = (input.elbow_axis - upper * upper.dot(input.elbow_axis)).try_normalize()?;
    let source = Mat3::from_cols(upper, hinge, upper.cross(hinge));
    // Both elbows flex toward +Z from -Y. The hinge is an axial vector:
    // reflection across X leaves its X component unchanged.
    let axis = Vec3::NEG_X;
    let destination = Mat3::from_cols(Vec3::NEG_Y, axis, Vec3::NEG_Y.cross(axis));
    Some(
        (Quat::from_mat3(&(destination * source.transpose())) * rest.upper_arm.global_rotation)
            .normalize(),
    )
}

// At the upper pole globe axial rotation and plane cannot be recovered
// independently. Return absence instead of inventing a hinge orientation.
fn swing(direction: Vec3) -> Option<Quat> {
    let cross = Vec3::NEG_Y.cross(direction);
    let q = Quat::from_xyzw(cross.x, cross.y, cross.z, 1.0 - direction.y);
    (q.length_squared() > f32::EPSILON * f32::EPSILON).then(|| q.normalize())
}

pub(crate) fn coordinates(chain: &ArmChainBinding, rotation: Quat) -> Option<ShoulderCoordinates> {
    if !rotation.is_finite() || rotation.length_squared() <= f32::EPSILON {
        return None;
    }
    let relative = (rotation * neutral(chain)?.inverse()).normalize();
    let direction = relative * Vec3::NEG_Y;
    let horizontal = direction.x.hypot(direction.z);
    let plane = (horizontal > 8.0 * f32::EPSILON).then(|| {
        let center = 20.0_f32.to_radians();
        let raw = direction.z.atan2(sign(chain) * direction.x);
        center + (raw - center).sin().atan2((raw - center).cos())
    });
    let twist = swing(direction)?.inverse() * relative;
    let twist = if twist.w < 0.0 { -twist } else { twist };
    Some(ShoulderCoordinates {
        plane,
        elevation: horizontal.atan2(-direction.y),
        axial: 2.0 * (-sign(chain) * twist.y).atan2(twist.w),
    })
}

pub(crate) fn from_coordinates(
    chain: &ArmChainBinding,
    angles: ShoulderCoordinates,
) -> Option<Quat> {
    let direction = if let Some(plane) = angles.plane {
        let (s, c) = plane.sin_cos();
        let (e, down) = angles.elevation.sin_cos();
        Vec3::new(sign(chain) * e * c, -down, e * s)
    } else if angles.elevation <= 8.0 * f32::EPSILON {
        Vec3::NEG_Y
    } else {
        return None;
    };
    Some(
        (swing(direction)?
            * Quat::from_axis_angle(-sign(chain) * Vec3::Y, angles.axial)
            * neutral(chain)?)
        .normalize(),
    )
}

/// Holzbaur/Xu thoracohumeral ranges, never bone-local Euler limits.
pub(crate) fn constrain(chain: &ArmChainBinding, rotation: Quat) -> Option<Quat> {
    let angles = coordinates(chain, rotation)?;
    let plane = angles
        .plane
        .map(|p| p.clamp(-90.0_f32.to_radians(), 130.0_f32.to_radians()));
    let axial = angles
        .axial
        .clamp(-90.0_f32.to_radians(), 20.0_f32.to_radians());
    if plane == angles.plane && axial == angles.axial {
        return Some(rotation);
    }
    from_coordinates(
        chain,
        ShoulderCoordinates {
            plane,
            axial,
            ..angles
        },
    )
}
