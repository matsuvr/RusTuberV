//! Full-rotation k-DOP shoulder ROM fitted to four CMU range-of-motion clips.
//! This empirical admissible region is not a population maximum or a personal
//! passive-tissue measurement. Fixed source-model ranges and skin collision
//! constrain the elbow/radioulnar DOFs; the CMU shoulder fit does not claim to.

use crate::arm::{ArmChainBinding, ArmSide};
use bevy::prelude::{Quat, Vec3};

mod data {
    include!("data/cmu_shoulder_rom.rs");
}

/// Conservative angular distance to the ROM boundary in the source's right
/// anatomical T-reference. A full rotation couples direction and axial roll.
pub(crate) fn shoulder_margins(
    chain: &ArmChainBinding,
    girdle: Quat,
    upper: Quat,
) -> Option<([f32; 27], f32)> {
    let reference = crate::shoulder::from_coordinates(
        chain,
        crate::shoulder::ShoulderCoordinates {
            plane: Some(0.0),
            elevation: std::f32::consts::FRAC_PI_2,
            axial: 0.0,
        },
    )?;
    let mut q = (girdle.inverse() * upper * reference.inverse()).normalize();
    if chain.side == ArmSide::Left {
        q = Quat::from_xyzw(q.x, -q.y, -q.z, q.w);
    }
    if q.w < 0.0 {
        q = -q;
    }
    let (axis, angle) = q.to_axis_angle();
    let v = axis * angle;
    // The SO(3) logarithm derivative is bounded by pi/2 on this principal
    // chart. Keep distance to the pi boundary as well: path certification
    // must not silently wrap its coordinates at that boundary.
    let chart_margin = std::f32::consts::PI - angle;
    let mut margins = [chart_margin; 27];
    let planes = data::SHOULDER_ROM
        .into_iter()
        .flat_map(|[x, y, z, min, max]| {
            let value = Vec3::new(x, y, z).dot(v);
            [
                (value - min) / std::f32::consts::FRAC_PI_2,
                (max - value) / std::f32::consts::FRAC_PI_2,
            ]
        });
    for (margin, value) in margins.iter_mut().skip(1).zip(planes) {
        *margin = value;
    }
    Some((margins, chart_margin))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;
    use crate::upper_limb::tests::{chain, state};

    #[test]
    fn independent_ranges_admit_combinations_rejected_by_full_rotation_data() {
        let left = chain(ArmSide::Left);
        let right = chain(ArmSide::Right);
        let mut rejected = 0;
        let mut accepted = 0;
        let mut pose_dependent = 0;
        for p in [-1.5, 0.0, 0.8, 1.5, 2.2] {
            for e in [0.17, 0.7, 1.57, 2.5, 3.0] {
                let mut decisions = Vec::new();
                for a in [-1.5, -0.8, 0.0, 0.34] {
                    let state = state(p, e, a, 1.0);
                    assert!(state.valid());
                    let l = state.forward(&left).unwrap();
                    let r = state.forward(&right).unwrap();
                    assert!((l.joint_margin - r.joint_margin).abs() < 3.0e-6);
                    decisions.push(l.joint_margin >= 0.0);
                    if l.joint_margin < 0.0 {
                        rejected += 1;
                    } else {
                        accepted += 1;
                    }
                }
                if decisions.iter().any(|v| *v) && decisions.iter().any(|v| !*v) {
                    pose_dependent += 1;
                }
            }
        }
        assert!(
            rejected > 0 && accepted > 0 && pose_dependent > 0,
            "{accepted}/{rejected}/{pose_dependent}"
        );
        assert!(
            state(0.0, 0.17, 0.0, 0.0)
                .forward(&left)
                .unwrap()
                .joint_margin
                > 0.0
        );
    }
}
