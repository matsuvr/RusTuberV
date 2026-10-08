//! Thumb axes from MyoHub/myo_sim 93b0ca8 (Apache-2.0), myoarm_r_chain.xml.
//! Map the model's metacarpal shaft/palm frame to the avatar's authored shaft,
//! preserving its rest opening/elevation. See THIRD_PARTY_NOTICES.md.

use bevy::prelude::{Mat3, Quat, Vec3};

use crate::arm::{ArmChainBinding, ArmSide, rest_palm_normal};
use crate::arm_pose::{ResolvedBoneDelta, ResolvedFingerJointPose};

fn frame(shaft: Vec3, normal: Vec3) -> Option<Mat3> {
    let y = shaft.try_normalize()?;
    let x = y.cross(normal).try_normalize()?;
    Some(Mat3::from_cols(x, y, x.cross(y)))
}

fn mapped_axis(
    chain: &ArmChainBinding,
    shaft: Vec3,
    source_shaft: Vec3,
    axis: Vec3,
) -> Option<Vec3> {
    let target = frame(shaft, rest_palm_normal(chain)?)?;
    // All positions are differences inside the source hand. Index/little MCP
    // coordinates include capitate + metacarpal + proximal joint offsets.
    let index = Vec3::new(0.022178, -0.080917, 0.010979);
    let little = Vec3::new(-0.019501, -0.071168, -0.003387);
    let source = frame(source_shaft, index.cross(little))?;
    let value = source.transpose() * axis;
    // The palm normal is axial. In this shaft frame a reflected axial
    // vector changes X/Y sign, whereas its Z component is unchanged.
    let value = if chain.side == ArmSide::Left {
        Vec3::new(-value.x, -value.y, value.z)
    } else {
        value
    };
    (target * value).try_normalize()
}

pub(crate) fn cmc_delta(
    chain: &ArmChainBinding,
    [flexion, abduction]: [f32; 2],
    weight: f32,
) -> Option<ResolvedBoneDelta> {
    let thumb = chain.finger_rest.thumb;
    let joint = thumb.metacarpal?;
    let shaft = thumb.proximal?.rest.position - joint.rest.position;
    let axis = |value| mapped_axis(chain, shaft, Vec3::new(0.0165, -0.0292, -0.0127), value);
    let flexion = Quat::from_axis_angle(
        axis(Vec3::new(-0.042399, -0.665286, 0.745384))?,
        flexion.clamp(-0.78, 0.7) * weight,
    );
    let abduction = Quat::from_axis_angle(
        axis(Vec3::new(0.495557, 0.731736, 0.467959))?,
        abduction.clamp(-0.5, 0.78) * weight,
    );
    Some(ResolvedBoneDelta {
        entity: joint.entity,
        delta: crate::skeleton::rest_delta(flexion * abduction, joint.rest.global_rotation),
    })
}

/// MCP and IP flex in the thumb's own oblique plane, not the four fingers'
/// palm-normal plane. Axes are immutable rest-space hinges carried by FK.
/// The source model uses negative flexion on both anatomical sides; remove
/// the tracking palm-normal sign before applying those reflected axes.
pub(crate) fn flexion_deltas(chain: &ArmChainBinding, angles: [f32; 2]) -> ResolvedFingerJointPose {
    let thumb = chain.finger_rest.thumb;
    let bend = |joint: Option<crate::arm::FingerJointRestBinding>, axis, angle: f32, limit| {
        let joint = joint?;
        let shaft = thumb.distal?.rest.position - thumb.proximal?.rest.position;
        let axis = mapped_axis(chain, shaft, Vec3::new(0.014, -0.0259, -0.0101), axis)?;
        let flexion = crate::arm_pose::signed_finger_curl(chain.side, angle).clamp(0.0, limit);
        Some(ResolvedBoneDelta {
            entity: joint.entity,
            delta: crate::skeleton::rest_delta(
                Quat::from_axis_angle(axis, -flexion),
                joint.rest.global_rotation,
            ),
        })
    };
    let [mcp, ip] = angles;
    ResolvedFingerJointPose {
        metacarpal: thumb.metacarpal.map(|joint| ResolvedBoneDelta {
            entity: joint.entity,
            delta: Quat::IDENTITY,
        }),
        proximal: bend(
            thumb.proximal,
            Vec3::new(-0.084295, -0.203488, 0.975442),
            mcp,
            std::f32::consts::FRAC_PI_4,
        ),
        intermediate: None,
        distal: bend(
            thumb.distal,
            Vec3::new(-0.050102, -0.479623, 0.876043),
            ip,
            1.309,
        ),
    }
}
