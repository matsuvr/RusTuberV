//! Shoulder-girdle kinematics: De Sapio/Holzbaur 2006 Table I and Eq. 9,
//! also Maier 2024 supplement B.1. Axes/offsets: MyoArm 93b0ca8 (Apache-2.0).
//! SC and AC rotations are expressed relative to the thorax (the source's
//! inverse phantom joints cancel the preceding orientation, not translation).

use bevy::prelude::{Quat, Vec2, Vec3};

use crate::arm::{ArmChainBinding, ArmSide};

#[derive(Clone, Copy, Debug)]
pub(crate) struct GirdlePose {
    /// Rotation of the VRM's single claviscapular link in model/rest space.
    pub rotation: Quat,
    pub centre: Vec3,
}

/// [SC protraction, SC elevation, AC internal rotation, AC upward rotation,
/// AC posterior tilt], radians. Plane and elevation are thoracohumeral globe
/// coordinates, not local rotations of the upper-arm bone.
pub(crate) fn rhythm(plane: f32, elevation: f32) -> [f32; 5] {
    [
        0.120 * plane - 0.242 * elevation,
        -0.046 * plane + 0.123 * elevation,
        0.140 * plane - 0.049 * elevation,
        -0.079 * plane + 0.396 * elevation,
        -0.028 * plane + 0.184 * elevation,
    ]
}

fn source_centre([protract, elevate, internal, upward, tilt]: [f32; 5]) -> Vec3 {
    let turn = |axis: Vec3, angle| Quat::from_axis_angle(axis.normalize(), angle);
    let clavicle = turn(Vec3::new(0.0153, 0.989299, -0.1451), protract)
        * turn(Vec3::new(-0.994473, 0.0, -0.104997), elevate);
    let scapula = turn(Vec3::new(0.157095, 0.947269, -0.279291), internal)
        * turn(Vec3::new(-0.754084, 0.297594, 0.585487), upward)
        * turn(Vec3::new(0.6377, 0.1186, 0.7611), tilt);
    clavicle * Vec3::new(-0.01433, 0.02007, 0.1355) + scapula * Vec3::new(-0.00955, -0.034, 0.009)
}

/// Derivative bound of the same shortest-arc shoulder map over a joint segment.
pub(crate) fn rotation_bound(chain: &ArmChainBinding, a: [f32; 9], b: [f32; 9]) -> f32 {
    if chain.shoulder.is_none() || chain.rest.shoulder.is_none() {
        return 0.0;
    }
    let [ap, ae, _, _, _, _, _, asp, ase] = a;
    let [bp, be, _, _, _, _, _, bsp, bse] = b;
    let p = (ap - bp).abs();
    let e = (ae - be).abs();
    let clavicle = Vec3::new(-0.01433, 0.02007, 0.1355).length();
    let scapula = Vec3::new(-0.00955, -0.034, 0.009).length();
    let ac = (0.140 + 0.079 + 0.028) * p + (0.049 + 0.396 + 0.184) * e;
    let speed =
        (clavicle * ((asp - bsp).abs() + (ase - bse).abs()) + scapula * ac) / (clavicle - scapula);
    let rest = source_centre([0.0; 5]);
    let mut midpoint = rhythm((ap + bp) * 0.5, (ae + be) * 0.5);
    midpoint[0] = (asp + bsp) * 0.5;
    midpoint[1] = (ase + bse) * 0.5;
    let direction = source_centre(midpoint);
    // Normalization costs 1/(|clavicle|-|scapula|). The arc's differential
    // costs sec(theta/2); include every direction in this interval, not just
    // the midpoint. An interval reaching the antipodal singularity cannot
    // be certified with this chart.
    let theta = rest.angle_between(direction) + speed * 0.5;
    if theta >= std::f32::consts::PI {
        f32::INFINITY
    } else {
        speed / (theta * 0.5).cos()
    }
}

/// Operator-norm bound on the second derivative of the clavicle rotation
/// along a linear joint segment. Normalization of v costs |v''|/m +
/// 3|v'|²/m². The shortest-arc quaternion is normalize([r×u,1+r·u]).
/// Differentiating q*x*q^-1 then costs 2|q''| + 2|q'|².
pub(crate) fn acceleration_bound(chain: &ArmChainBinding, a: [f32; 9], b: [f32; 9]) -> f32 {
    if chain.shoulder.is_none() || chain.rest.shoulder.is_none() {
        return 0.0;
    }
    let [ap, ae, _, _, _, _, _, asp, ase] = a;
    let [bp, be, _, _, _, _, _, bsp, bse] = b;
    let p = (ap - bp).abs();
    let e = (ae - be).abs();
    let c = Vec3::new(-0.01433, 0.02007, 0.1355).length();
    let s = Vec3::new(-0.00955, -0.034, 0.009).length();
    let sc = (asp - bsp).abs() + (ase - bse).abs();
    let ac = (0.140 + 0.079 + 0.028) * p + (0.049 + 0.396 + 0.184) * e;
    let u1 = (c * sc + s * ac) / (c - s);
    let u2 = (c * sc * sc + s * ac * ac) / (c - s) + 3.0 * u1 * u1;
    let rest = source_centre([0.0; 5]);
    let mut mid = rhythm((ap + bp) * 0.5, (ae + be) * 0.5);
    mid[0] = (asp + bsp) * 0.5;
    mid[1] = (ase + bse) * 0.5;
    let theta = rest.angle_between(source_centre(mid)) + u1 * 0.5;
    if theta >= std::f32::consts::PI {
        return f32::INFINITY;
    }
    let m = 2.0 * (theta * 0.5).cos();
    let q1 = std::f32::consts::SQRT_2 * u1 / m;
    let q2 = std::f32::consts::SQRT_2 * u2 / m + 3.0 * q1 * q1;
    2.0 * q2 + 2.0 * q1 * q1
}

/// Collapse the source's two links to the VRM's fixed-length shoulder link.
/// The source GH direction is retained, and its varying radial distance is
/// projected onto the authored SC-to-GH sphere. No bone is added or stretched.
pub(crate) fn forward(
    chain: &ArmChainBinding,
    plane: f32,
    elevation: f32,
    observed_sc: Option<Vec2>,
) -> Option<GirdlePose> {
    let Some(origin) = chain.rest.shoulder.filter(|_| chain.shoulder.is_some()) else {
        return Some(GirdlePose {
            rotation: Quat::IDENTITY,
            centre: chain.rest.upper_arm.position,
        });
    };
    let offset = chain.rest.upper_arm.position - origin.position;
    let authored = offset.try_normalize()?;
    let polar = |v: Vec3| {
        let x = if chain.side == ArmSide::Left {
            v.z
        } else {
            -v.z
        };
        Vec3::new(x, v.y, v.x)
    };
    // VRM T-pose definition 1.5 already places the shoulders at their relaxed,
    // lowest position, despite raised arms. Bind that link to the source's
    // arm-down zero; subtracting the raised-arm rhythm lowers it a second time.
    let rest = polar(source_centre([0.0; 5])).try_normalize()?;
    let mut joints = rhythm(plane, elevation);
    if let Some(sc) = observed_sc {
        joints[0] = sc.x;
        joints[1] = sc.y;
    }
    let direction = polar(source_centre(joints)).try_normalize()?;
    let alignment = crate::skeleton::minimal_arc(rest, authored)?;
    let rotation =
        (alignment * crate::skeleton::minimal_arc(rest, direction)? * alignment.inverse())
            .normalize();
    Some(GirdlePose {
        rotation,
        centre: origin.position + rotation * offset,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn vrm_t_pose_shoulders_are_the_arm_down_girdle_reference() {
        for side in [ArmSide::Left, ArmSide::Right] {
            let chain = crate::upper_limb::tests::chain(side);
            let down = forward(&chain, 0.0, 0.0, None).unwrap();
            assert!(down.rotation.dot(Quat::IDENTITY).abs() > 1.0 - f32::EPSILON);
            assert!(down.centre.distance(chain.rest.upper_arm.position) < 1.0e-6);
            let raised = forward(&chain, 0.0, std::f32::consts::FRAC_PI_2, None).unwrap();
            assert!(raised.centre.y > down.centre.y);
            let pivot = chain.rest.shoulder.unwrap().position;
            assert!((raised.centre.distance(pivot) - down.centre.distance(pivot)).abs() < 1.0e-6);
        }
    }

    #[test]
    fn full_rhythm_changes_with_plane_and_has_no_double_parent_rotation() {
        let pi = std::f32::consts::FRAC_PI_2;
        let frontal = rhythm(0.0, pi);
        let sagittal = rhythm(pi, pi);
        for ((front, side), b) in frontal
            .into_iter()
            .zip(sagittal)
            .zip([0.120, -0.046, 0.140, -0.079, -0.028])
        {
            assert!((side - front - b * pi).abs() < 1.0e-6);
        }
        let neutral = source_centre([0.0; 5]);
        assert!(neutral.distance(Vec3::new(-0.02388, -0.01393, 0.1445)) < 1.0e-7);
        // Rotating SC moves AC, while the absolute scapular orientation stays
        // fixed. Its offset must not inherit the clavicle rotation again.
        let q = Quat::from_axis_angle(Vec3::new(0.0153, 0.989299, -0.1451).normalize(), -0.4);
        let expected =
            q * Vec3::new(-0.01433, 0.02007, 0.1355) + Vec3::new(-0.00955, -0.034, 0.009);
        assert!(source_centre([-0.4, 0.0, 0.0, 0.0, 0.0]).distance(expected) < 1.0e-7);
    }
}
