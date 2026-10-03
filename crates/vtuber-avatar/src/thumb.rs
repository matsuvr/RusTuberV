//! CMC axes from MyoHub/myo_sim 93b0ca8 (Apache-2.0), myoarm_r_chain.xml.
//! Map the model's metacarpal shaft/palm frame to the avatar's authored shaft,
//! preserving its rest opening/elevation. See THIRD_PARTY_NOTICES.md.

use bevy::prelude::{Mat3, Quat, Vec3};

use crate::arm::{ArmChainBinding, ArmSide, rest_palm_normal};
use crate::arm_pose::ResolvedBoneDelta;

fn frame(shaft: Vec3, normal: Vec3) -> Option<Mat3> {
    let y = shaft.try_normalize()?;
    let x = y.cross(normal).try_normalize()?;
    Some(Mat3::from_cols(x, y, x.cross(y)))
}

pub(crate) fn cmc_delta(
    chain: &ArmChainBinding,
    [flexion, abduction]: [f32; 2],
    weight: f32,
) -> Option<ResolvedBoneDelta> {
    let thumb = chain.finger_rest.thumb;
    let joint = thumb.metacarpal?;
    let shaft = thumb.proximal?.rest.position - joint.rest.position;
    let target = frame(shaft, rest_palm_normal(chain)?)?;
    // All positions are differences inside the source hand. Index/little MCP
    // coordinates include capitate + metacarpal + proximal joint offsets.
    let index = Vec3::new(0.022178, -0.080917, 0.010979);
    let little = Vec3::new(-0.019501, -0.071168, -0.003387);
    let source = frame(Vec3::new(0.0165, -0.0292, -0.0127), index.cross(little))?;
    let axis = |value: Vec3| {
        let value = source.transpose() * value;
        // The palm normal is axial. In this shaft frame a reflected axial
        // vector changes X/Y sign, whereas its Z component is unchanged.
        let value = if chain.side == ArmSide::Left {
            Vec3::new(-value.x, -value.y, value.z)
        } else {
            value
        };
        (target * value).normalize()
    };
    let flexion = Quat::from_axis_angle(
        axis(Vec3::new(-0.042399, -0.665286, 0.745384)),
        flexion.clamp(-0.78, 0.7) * weight,
    );
    let abduction = Quat::from_axis_angle(
        axis(Vec3::new(0.495557, 0.731736, 0.467959)),
        abduction.clamp(-0.5, 0.78) * weight,
    );
    Some(ResolvedBoneDelta {
        entity: joint.entity,
        delta: crate::skeleton::rest_delta(flexion * abduction, joint.rest.global_rotation),
    })
}
