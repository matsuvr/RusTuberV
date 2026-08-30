//! Dynamic arm motion rest geometry (Body Motion 6/11 prep, Issue #175).
//!
//! Resolves, exactly once at bind time, the immutable rest-space data the
//! dynamic arm solve needs beyond [`crate::arm::ArmRestGeometry`]:
//!
//! - the hips-relative virtual hand anchor frame,
//! - a chest/torso center reference position,
//! - the forearm longitudinal twist axis with an explicit degeneracy status,
//! - a stable elbow pole/swivel reference plane expressed in model space.
//!
//! Everything is built from immutable `RestGlobalTransform` data; rendered
//! or current transforms are never consulted, and no VRM0/VRM1 branch
//! exists. Missing optional references or degenerate measurements become
//! explicit `None` capabilities instead of binding failures.

use bevy::prelude::*;

use crate::arm::{ArmRestGeometry, ArmSide};

/// Forearm segments shorter than this cannot define a twist axis.
const MIN_SEGMENT_METERS: f32 = 0.01;

/// Cross-product magnitude below this marks degenerate plane construction.
const MIN_CROSS_MAGNITUDE: f32 = 1.0e-4;

/// Hips-relative virtual hand anchor frame.
///
/// Translation is meters from the hips rest origin; rotation maps the hips
/// rest frame into the hand rest frame. Both are constant model-space values
/// resolved from immutable rest data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HipsAnchorFrame {
    /// Hand rest origin relative to the hips rest origin (meters).
    pub translation_from_hips: Vec3,
    /// Hand rest orientation relative to the hips rest orientation.
    pub rotation_from_hips: Quat,
}

/// Longitudinal twist axis of the forearm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForearmTwistAxisInfo {
    /// Unit axis along elbow -> wrist in rest global space.
    pub direction: Vec3,
    /// Whether the segment was long enough to measure reliably.
    pub valid: bool,
}

impl ForearmTwistAxisInfo {
    /// Whether the axis may be used for twist decomposition.
    #[must_use]
    pub const fn usable(&self) -> bool {
        self.valid
    }
}

/// Stable elbow pole/swivel reference plane in rest global space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElbowSwivelReference {
    /// Plane normal (unit length).
    pub normal: Vec3,
    /// A point on the plane (the elbow origin).
    pub point: Vec3,
}

/// Per-side arm motion rest geometry resolved once at bind time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmMotionRestGeometry {
    /// Which side this geometry belongs to.
    pub side: ArmSide,
    /// Hips-relative hand anchor; `None` when no hips rest reference exists.
    pub hand_anchor: Option<HipsAnchorFrame>,
    /// Chest/torso center reference; `None` when no usable reference exists.
    pub torso_center: Option<Vec3>,
    /// Forearm twist axis; `None` only for non-finite input.
    pub forearm_twist: Option<ForearmTwistAxisInfo>,
    /// Elbow swivel reference plane; `None` when even fallbacks degenerate.
    pub elbow_reference: Option<ElbowSwivelReference>,
}

/// Component holding both sides' resolved motion geometry.
#[derive(Component, Debug, Clone, Copy, PartialEq, Default)]
pub struct ArmMotionGeometry {
    /// Left-side geometry, present when the left chain bound.
    pub left: Option<ArmMotionRestGeometry>,
    /// Right-side geometry, present when the right chain bound.
    pub right: Option<ArmMotionRestGeometry>,
}

/// Builds the per-side motion geometry from immutable rest data.
///
/// * `hand_anchor` requires a hips rest reference (`hips_position`,
///   `hips_rotation`) and is otherwise `None`.
/// * `forearm_twist.direction` is the normalized elbow -> wrist axis; the
///   info carries `valid = false` when the segment is too short to trust but
///   still exposes the measured direction.
/// * `elbow_reference.normal` is the rest bend-plane normal (cross of the
///   upper-arm and forearm axes); straight arms fall back to an orthogonal
///   of the upper-arm axis around model up, keeping the value deterministic.
///
/// All outputs are finite or `None`; nothing here depends on animated state.
#[must_use]
pub fn build_arm_motion_rest_geometry(
    side: ArmSide,
    rest: &ArmRestGeometry,
    hips_position: Option<Vec3>,
    hips_rotation: Option<Quat>,
    torso_center: Option<Vec3>,
) -> ArmMotionRestGeometry {
    let finite_or_none = |value: Vec3| if value.is_finite() { Some(value) } else { None };

    let hand_anchor = match (hips_position, hips_rotation) {
        (Some(hips_position), Some(hips_rotation)) => {
            let translation = rest.wrist.position - hips_position;
            let rotation = hips_rotation.inverse() * rest.wrist.global_rotation;
            if translation.is_finite() && rotation.is_finite() {
                Some(HipsAnchorFrame {
                    translation_from_hips: translation,
                    rotation_from_hips: rotation,
                })
            } else {
                None
            }
        }
        _ => None,
    };

    let torso_center = torso_center.and_then(finite_or_none);

    let forearm_delta = rest.wrist.position - rest.elbow.position;
    let forearm_twist = if !forearm_delta.is_finite() {
        None
    } else {
        let length = forearm_delta.length();
        let valid = length.is_finite() && length > MIN_SEGMENT_METERS;
        let direction = if valid {
            forearm_delta / length
        } else {
            Vec3::X
        };
        Some(ForearmTwistAxisInfo { direction, valid })
    };

    let upper_arm_delta = rest.elbow.position - rest.upper_arm.position;
    let elbow_reference = if !upper_arm_delta.is_finite() || !forearm_delta.is_finite() {
        None
    } else {
        let upper_length = upper_arm_delta.length();
        let upper_axis = if upper_length > MIN_SEGMENT_METERS && upper_length.is_finite() {
            Some(upper_arm_delta / upper_length)
        } else {
            None
        };

        let bend_normal = match upper_axis {
            Some(axis) => {
                let normal = axis.cross(forearm_delta);
                if normal.length_squared() > MIN_CROSS_MAGNITUDE * MIN_CROSS_MAGNITUDE {
                    normal.normalize_or_zero()
                } else {
                    let candidate = axis.cross(Vec3::Y);
                    if candidate.length_squared() > MIN_CROSS_MAGNITUDE * MIN_CROSS_MAGNITUDE {
                        candidate.normalize_or_zero()
                    } else {
                        axis.cross(Vec3::Z).normalize_or_zero()
                    }
                }
            }
            None => Vec3::ZERO,
        };

        match (upper_axis, bend_normal) {
            (Some(_), normal) if normal.is_finite() && normal != Vec3::ZERO => {
                Some(ElbowSwivelReference {
                    normal,
                    point: rest.elbow.position,
                })
            }
            _ => None,
        }
    };

    ArmMotionRestGeometry {
        side,
        hand_anchor,
        torso_center,
        forearm_twist,
        elbow_reference,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rest_geometry(elbow_to_wrist: Vec3, shoulder_offset: Vec3) -> ArmRestGeometry {
        let upper_origin = Vec3::new(0.05, 1.35, 0.0) + shoulder_offset;
        let elbow = upper_origin + Vec3::new(0.25, -0.05, 0.0);
        let wrist = elbow + elbow_to_wrist;
        ArmRestGeometry {
            shoulder: None,
            upper_arm: crate::arm::RestSpaceBonePose {
                position: upper_origin,
                global_rotation: Quat::IDENTITY,
                local_rotation: Quat::IDENTITY,
            },
            elbow: crate::arm::RestSpaceBonePose {
                position: elbow,
                global_rotation: Quat::IDENTITY,
                local_rotation: Quat::IDENTITY,
            },
            wrist: crate::arm::RestSpaceBonePose {
                position: wrist,
                global_rotation: Quat::IDENTITY,
                local_rotation: Quat::IDENTITY,
            },
            upper_arm_length: upper_origin.distance(elbow),
            forearm_length: elbow.distance(wrist),
            total_arm_length: upper_origin.distance(wrist),
        }
    }

    fn assert_close(actual: Vec3, expected: Vec3) {
        assert!(
            actual.distance(expected) < 1e-5,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn assert_scalar_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    fn sample_rest() -> ArmRestGeometry {
        rest_geometry(Vec3::new(-0.02, -0.24, -0.01), Vec3::ZERO)
    }

    #[test]
    fn hand_anchor_is_expressed_relative_to_the_hips_frame() {
        let hips_position = Vec3::new(0.0, 0.95, 0.0);
        let hips_rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let mut rest = sample_rest();
        rest.wrist.global_rotation = Quat::from_rotation_x(0.2);

        let geometry = build_arm_motion_rest_geometry(
            ArmSide::Right,
            &rest,
            Some(hips_position),
            Some(hips_rotation),
            None,
        );
        let anchor = geometry.hand_anchor.expect("anchor");

        assert_close(
            anchor.translation_from_hips,
            rest.wrist.position - hips_position,
        );
        assert!(anchor.rotation_from_hips.is_finite());
        assert!(
            anchor
                .rotation_from_hips
                .angle_between(hips_rotation.inverse() * rest.wrist.global_rotation)
                < 1e-6
        );
    }

    #[test]
    fn missing_hips_reference_leaves_anchor_unavailable_without_failing() {
        let geometry =
            build_arm_motion_rest_geometry(ArmSide::Left, &sample_rest(), None, None, None);
        assert!(geometry.hand_anchor.is_none());
        assert!(geometry.forearm_twist.is_some());
    }

    #[test]
    fn mirror_symmetry_between_sides_for_mirrored_rest_data() {
        let hips_position = Vec3::new(0.0, 0.95, 0.0);
        let left = sample_rest();
        let mut right = sample_rest();
        right.upper_arm.position.x = -left.upper_arm.position.x;
        right.elbow.position.x = -left.elbow.position.x;
        right.wrist.position.x = -left.wrist.position.x;

        let left_geometry = build_arm_motion_rest_geometry(
            ArmSide::Left,
            &left,
            Some(hips_position),
            Some(Quat::IDENTITY),
            None,
        );
        let right_geometry = build_arm_motion_rest_geometry(
            ArmSide::Right,
            &right,
            Some(hips_position),
            Some(Quat::IDENTITY),
            None,
        );

        let l = left_geometry.hand_anchor.expect("left anchor");
        let r = right_geometry.hand_anchor.expect("right anchor");
        assert_scalar_close(l.translation_from_hips.x, -r.translation_from_hips.x);
        assert_scalar_close(l.translation_from_hips.y, r.translation_from_hips.y);
        assert_scalar_close(l.translation_from_hips.z, r.translation_from_hips.z);
    }

    #[test]
    fn forearm_twist_axis_follows_elbow_to_wrist_and_flags_degeneracy() {
        let good = build_arm_motion_rest_geometry(ArmSide::Left, &sample_rest(), None, None, None);
        let twist = good.forearm_twist.expect("twist");
        assert!(twist.usable());
        let rest = sample_rest();
        let expected = (rest.wrist.position - rest.elbow.position)
            .try_normalize()
            .unwrap();
        assert_close(twist.direction, expected);

        let short = rest_geometry(Vec3::new(0.001, -0.002, 0.0), Vec3::ZERO);
        let degenerate = build_arm_motion_rest_geometry(ArmSide::Left, &short, None, None, None);
        let info = degenerate.forearm_twist.expect("info still present");
        assert!(!info.usable());
    }

    #[test]
    fn elbow_reference_uses_bend_plane_and_survives_straight_arms() {
        let bent = build_arm_motion_rest_geometry(ArmSide::Left, &sample_rest(), None, None, None);
        let reference = bent.elbow_reference.expect("reference");
        assert_close(reference.point, sample_rest().elbow.position);
        assert!((reference.normal.length() - 1.0).abs() < 1e-5);

        let straight = rest_geometry(Vec3::new(0.0, -0.26, 0.0), Vec3::ZERO);
        let straight_geometry =
            build_arm_motion_rest_geometry(ArmSide::Left, &straight, None, None, None);
        let straight_ref = straight_geometry
            .elbow_reference
            .expect("fallback reference");
        assert_scalar_close(straight_ref.normal.length(), 1.0);
    }

    #[test]
    fn torso_center_passes_through_only_when_finite() {
        let geometry = build_arm_motion_rest_geometry(
            ArmSide::Left,
            &sample_rest(),
            None,
            None,
            Some(Vec3::new(0.0, 1.15, 0.005)),
        );
        assert_scalar_close(geometry.torso_center.expect("center").y, 1.15);

        let none = build_arm_motion_rest_geometry(
            ArmSide::Left,
            &sample_rest(),
            None,
            None,
            Some(Vec3::new(f32::NAN, 1.0, 0.0)),
        );
        assert!(none.torso_center.is_none());
    }

    #[test]
    fn identical_inputs_are_deterministic() {
        let a = build_arm_motion_rest_geometry(
            ArmSide::Right,
            &sample_rest(),
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::from_rotation_y(0.3)),
            Some(Vec3::new(0.0, 1.15, 0.0)),
        );
        let b = build_arm_motion_rest_geometry(
            ArmSide::Right,
            &sample_rest(),
            Some(Vec3::new(0.0, 0.95, 0.0)),
            Some(Quat::from_rotation_y(0.3)),
            Some(Vec3::new(0.0, 1.15, 0.0)),
        );
        assert_eq!(a, b);
    }
}
