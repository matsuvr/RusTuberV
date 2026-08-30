//! Axis-selective root compensation and virtual head/body target generation
//! (`DESIGN.md` §6, Issue #165).
//!
//! The shaped neutral-relative head translation (Issue #164) is split into a
//! pure, engine-neutral [`VirtualBodyTargets`] state:
//!
//! * **X** stays in the upper-body lean: the default policy routes zero X to
//!   root/body compensation.
//! * **Y** is routed mostly to the root/body side while a small residual
//!   remains at the head/neck chain, bounded by an explicit body-scale-aware
//!   cap.
//! * **Z** is routed mostly to the root/body side to prevent over-stretch of
//!   the neck and arms.
//!
//! Everything here is pure data: no Bevy `Entity`, `Transform`, camera
//! `Transform`, `Projection`, or FOV API is touched, and the same inputs and
//! profile always produce identical outputs. Head rotation passes through
//! unchanged; body/root rotation limits live in
//! [`VirtualBodyProfile`] so a later solver (Issue #167) can consume them.

use std::f32::consts::PI;

use vtuber_core::types::{HeadPose, HeadTranslationSignal, HeadTranslationState};

/// Body-scale-aware typed profile for virtual head/body target generation.
///
/// All magnitudes are expressed as ratios of the avatar's body scale or as
/// radians, never as scattered magic constants. Gains are fractions of the
/// shaped translation routed to root/body compensation per axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirtualBodyProfile {
    /// Fraction of shaped X translation routed to root/body compensation.
    ///
    /// The default keeps X out of the root entirely so horizontal motion
    /// survives as upper-body lean.
    pub x_root_gain: f32,
    /// Fraction of shaped Y translation routed to root/body compensation.
    pub y_root_gain: f32,
    /// Fraction of shaped Z translation routed to root/body compensation.
    pub z_root_gain: f32,
    /// Maximum head-target Y residual as a ratio of body scale.
    ///
    /// The default keeps roughly 5 cm of Y residual on a 0.7 m-scale avatar,
    /// which normalizes to about `0.0714`.
    pub y_residual_cap_ratio: f32,
    /// Upper bound on body/root rotation magnitude consumed by the later
    /// solver, in radians. The default is approximately 15 degrees.
    pub max_body_rotation_rad: f32,
}

impl Default for VirtualBodyProfile {
    fn default() -> Self {
        Self {
            x_root_gain: 0.0,
            y_root_gain: 0.8,
            z_root_gain: 0.9,
            y_residual_cap_ratio: 0.05 / 0.7,
            max_body_rotation_rad: 15.0 * PI / 180.0,
        }
    }
}

impl VirtualBodyProfile {
    /// Validates that all fields are finite and inside their contract ranges.
    ///
    /// # Errors
    ///
    /// Returns [`VirtualBodyProfileError`] when a gain is outside `[0, 1]`,
    /// the Y residual cap ratio is not strictly positive, or the rotation
    /// limit is not inside `(0, PI]`.
    pub fn validate(&self) -> Result<(), VirtualBodyProfileError> {
        for (name, gain) in [
            ("x_root_gain", self.x_root_gain),
            ("y_root_gain", self.y_root_gain),
            ("z_root_gain", self.z_root_gain),
        ] {
            if !gain.is_finite() || !(0.0..=1.0).contains(&gain) {
                return Err(VirtualBodyProfileError::GainOutOfRange { name, value: gain });
            }
        }
        if !self.y_residual_cap_ratio.is_finite() || self.y_residual_cap_ratio <= 0.0 {
            return Err(VirtualBodyProfileError::NonPositiveResidualCap {
                value: self.y_residual_cap_ratio,
            });
        }
        if !self.max_body_rotation_rad.is_finite()
            || self.max_body_rotation_rad <= 0.0
            || self.max_body_rotation_rad > PI
        {
            return Err(VirtualBodyProfileError::RotationLimitOutOfRange {
                value: self.max_body_rotation_rad,
            });
        }
        Ok(())
    }
}

/// Errors produced while validating a [`VirtualBodyProfile`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VirtualBodyProfileError {
    /// A root-compensation gain was non-finite or outside `[0, 1]`.
    GainOutOfRange {
        /// Offending field name.
        name: &'static str,
        /// Offending value.
        value: f32,
    },
    /// The head-target Y residual cap ratio was zero, negative, or non-finite.
    NonPositiveResidualCap {
        /// Offending value.
        value: f32,
    },
    /// The body/root rotation limit was non-finite or outside `(0, PI]`.
    RotationLimitOutOfRange {
        /// Offending value.
        value: f32,
    },
}

/// Neutral-relative translation offset in meters.
///
/// Axes follow the engine-neutral contract: positive X toward the unmirrored
/// image right, positive Y up, positive Z away from the camera.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TranslationMeters {
    /// X component in meters.
    pub x: f32,
    /// Y component in meters.
    pub y: f32,
    /// Z component in meters.
    pub z: f32,
}

impl TranslationMeters {
    const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// Pure virtual head target produced from one shaped control frame.
///
/// The translation is only the residual kept at the head/neck chain after
/// axis-selective routing; the full tracked head rotation is preserved
/// unchanged for the later position-aware solver (Issue #167).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirtualHeadTarget {
    /// Residual translation retained at the head/neck chain, in meters.
    pub translation: TranslationMeters,
    /// Tracked head rotation relative to calibrated neutral, passed through.
    pub rotation: HeadPose,
    /// Availability of the translation observation behind this target.
    pub state: HeadTranslationState,
}

/// Per-axis root/body translation compensation in meters.
///
/// Each axis can be tested independently; with the default profile X is
/// always zero and Y/Z follow the configured gains.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyTranslationCompensation {
    /// Root/body X compensation in meters (zero under the default policy).
    pub x: f32,
    /// Root/body Y compensation in meters.
    pub y: f32,
    /// Root/body Z compensation in meters.
    pub z: f32,
}

/// Pure target state consumed by the later upper-body solver (Issue #167).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirtualBodyTargets {
    /// Virtual head target: residual translation plus tracked rotation.
    pub head: VirtualHeadTarget,
    /// Root/body translation compensation per axis.
    pub body_compensation: BodyTranslationCompensation,
}

impl VirtualBodyTargets {
    /// A zeroed target set carrying no observation.
    pub const ZERO_UNAVAILABLE: Self = Self {
        head: VirtualHeadTarget {
            translation: TranslationMeters::ZERO,
            rotation: HeadPose {
                yaw_rad: 0.0,
                pitch_rad: 0.0,
                roll_rad: 0.0,
            },
            state: HeadTranslationState::Unavailable,
        },
        body_compensation: BodyTranslationCompensation {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    };
}

/// Splits one shaped head translation into virtual head and body targets.
///
/// Per-axis behavior:
///
/// * X: `body.x = shaped.x * x_root_gain`; the head target keeps the rest.
///   Under the default policy (`x_root_gain == 0`) nothing is compensated at
///   the root and the whole X survives as upper-body lean.
/// * Y: the head residual is `shaped.y * (1 - y_root_gain)` clamped to
///   `±y_residual_cap_ratio * body_scale`; the body side receives the exact
///   remainder so `head + body == shaped` holds on every axis even when the
///   cap absorbs excess motion instead of over-stretching the neck.
/// * Z: `body.z = shaped.z * z_root_gain`; the head target keeps the rest.
///   No extra cap is applied because the upstream soft-cap already bounds Z.
///
/// Unavailable observations zero the translation side while the head
/// rotation is still passed through: rotation authority is independent of
/// translation availability. Invalid body scale or any non-finite input
/// degrades the translation to unavailable rather than propagating NaN/Inf.
#[must_use]
pub fn build_virtual_body_targets(
    shaped: &HeadTranslationSignal,
    head_rotation: HeadPose,
    profile: &VirtualBodyProfile,
    body_scale_meters: f32,
) -> VirtualBodyTargets {
    let pass_through = |state| VirtualBodyTargets {
        head: VirtualHeadTarget {
            translation: TranslationMeters::ZERO,
            rotation: head_rotation,
            state,
        },
        ..VirtualBodyTargets::ZERO_UNAVAILABLE
    };

    if profile.validate().is_err() {
        return pass_through(HeadTranslationState::Unavailable);
    }
    if !shaped.is_available() {
        return pass_through(HeadTranslationState::Unavailable);
    }
    if !body_scale_meters.is_finite() || body_scale_meters <= 0.0 {
        return pass_through(HeadTranslationState::Unavailable);
    }

    let shaped_x = shaped.x_meters;
    let shaped_y = shaped.y_meters;
    let shaped_z = shaped.z_meters;

    let body_x = shaped_x * profile.x_root_gain;
    let head_x = shaped_x - body_x;

    let uncapped_head_y = shaped_y * (1.0 - profile.y_root_gain);
    let cap_meters = profile.y_residual_cap_ratio * body_scale_meters;
    let head_y = uncapped_head_y.clamp(-cap_meters, cap_meters);
    let body_y = shaped_y - head_y;

    let body_z = shaped_z * profile.z_root_gain;
    let head_z = shaped_z - body_z;

    let head_translation = TranslationMeters {
        x: head_x,
        y: head_y,
        z: head_z,
    };
    if !head_translation.is_finite()
        || !body_x.is_finite()
        || !body_y.is_finite()
        || !body_z.is_finite()
    {
        return pass_through(HeadTranslationState::Unavailable);
    }

    VirtualBodyTargets {
        head: VirtualHeadTarget {
            translation: head_translation,
            rotation: head_rotation,
            state: shaped.state,
        },
        body_compensation: BodyTranslationCompensation {
            x: body_x,
            y: body_y,
            z: body_z,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn tracked(x: f32, y: f32, z: f32) -> HeadTranslationSignal {
        HeadTranslationSignal::tracked(x, y, z)
    }

    #[test]
    fn default_profile_is_valid_and_matches_project_defaults() {
        let profile = VirtualBodyProfile::default();
        assert_eq!(profile.validate(), Ok(()));
        assert_eq!(profile.x_root_gain, 0.0);
        assert_relative_eq!(
            profile.max_body_rotation_rad,
            15.0f32.to_radians(),
            epsilon = 1e-6
        );
        assert_relative_eq!(profile.y_residual_cap_ratio * 0.7, 0.05, epsilon = 1e-5);
    }

    #[test]
    fn default_policy_routes_no_x_to_the_root() {
        let profile = VirtualBodyProfile::default();
        let targets = build_virtual_body_targets(
            &tracked(0.10, 0.02, 0.05),
            HeadPose::default(),
            &profile,
            1.0,
        );

        assert_eq!(targets.body_compensation.x, 0.0);
        assert_relative_eq!(targets.head.translation.x, 0.10, epsilon = 1e-6);
    }

    #[test]
    fn y_and_z_are_distributed_per_configured_gains() {
        let profile = VirtualBodyProfile::default();
        let targets = build_virtual_body_targets(
            &tracked(0.0, 0.10, 0.20),
            HeadPose::default(),
            &profile,
            1.0,
        );

        assert_relative_eq!(targets.body_compensation.y, 0.08, epsilon = 1e-6);
        assert_relative_eq!(targets.head.translation.y, 0.02, epsilon = 1e-6);
        assert_relative_eq!(targets.body_compensation.z, 0.18, epsilon = 1e-6);
        assert_relative_eq!(targets.head.translation.z, 0.02, epsilon = 1e-6);
    }

    #[test]
    fn axes_are_independent_of_each_other() {
        let profile = VirtualBodyProfile::default();
        let baseline =
            build_virtual_body_targets(&tracked(0.0, 0.0, 0.0), HeadPose::default(), &profile, 1.0);
        let moved_y = build_virtual_body_targets(
            &tracked(0.0, 0.08, 0.0),
            HeadPose::default(),
            &profile,
            1.0,
        );

        assert_relative_eq!(moved_y.head.translation.x, baseline.head.translation.x);
        assert_relative_eq!(moved_y.body_compensation.x, baseline.body_compensation.x);
        assert_relative_eq!(moved_y.body_compensation.z, baseline.body_compensation.z);
        assert_relative_eq!(moved_y.head.translation.z, baseline.head.translation.z);
        assert_relative_eq!(moved_y.body_compensation.y, 0.064, epsilon = 1e-6);

        let x_profile = VirtualBodyProfile {
            x_root_gain: 0.5,
            ..VirtualBodyProfile::default()
        };
        let moved_x = build_virtual_body_targets(
            &tracked(0.10, 0.0, 0.0),
            HeadPose::default(),
            &x_profile,
            1.0,
        );
        assert_relative_eq!(moved_x.body_compensation.x, 0.05, epsilon = 1e-6);
        assert_relative_eq!(moved_x.body_compensation.y, 0.0);
        assert_relative_eq!(moved_x.body_compensation.z, 0.0);
    }

    #[test]
    fn head_y_residual_is_capped_by_body_scale() {
        let profile = VirtualBodyProfile::default();
        let targets = build_virtual_body_targets(
            &tracked(0.0, -0.40, 0.0),
            HeadPose::default(),
            &profile,
            0.7,
        );

        assert_relative_eq!(
            targets.head.translation.y,
            -(profile.y_residual_cap_ratio * 0.7),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            targets.body_compensation.y + targets.head.translation.y,
            -0.40,
            epsilon = 1e-6
        );
    }

    #[test]
    fn translation_conservation_holds_on_every_axis() {
        let profile = VirtualBodyProfile::default();
        let targets = build_virtual_body_targets(
            &tracked(-0.12, 0.30, 0.25),
            HeadPose::default(),
            &profile,
            1.4,
        );

        assert_relative_eq!(
            targets.head.translation.x + targets.body_compensation.x,
            -0.12,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            targets.head.translation.y + targets.body_compensation.y,
            0.30,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            targets.head.translation.z + targets.body_compensation.z,
            0.25,
            epsilon = 1e-6
        );
    }

    #[test]
    fn rotation_passes_through_unchanged() {
        let profile = VirtualBodyProfile::default();
        let rotation = HeadPose {
            yaw_rad: 0.21,
            pitch_rad: -0.13,
            roll_rad: 0.07,
        };
        let targets =
            build_virtual_body_targets(&tracked(0.01, 0.01, 0.01), rotation, &profile, 1.0);
        assert_eq!(targets.head.rotation, rotation);
        let rotation_only = build_virtual_body_targets(
            &HeadTranslationSignal::UNAVAILABLE,
            rotation,
            &profile,
            1.0,
        );
        assert_eq!(rotation_only.head.rotation, rotation);
        assert!(matches!(
            rotation_only.head.state,
            HeadTranslationState::Unavailable
        ));
    }

    #[test]
    fn unavailable_input_zeroes_translation_but_keeps_rotation() {
        let profile = VirtualBodyProfile::default();
        let rotation = HeadPose {
            yaw_rad: -0.4,
            pitch_rad: 0.2,
            roll_rad: 0.1,
        };
        let targets = build_virtual_body_targets(
            &HeadTranslationSignal::UNAVAILABLE,
            rotation,
            &profile,
            1.0,
        );
        assert_eq!(targets.head.translation, TranslationMeters::ZERO);
        assert_eq!(
            targets.body_compensation,
            BodyTranslationCompensation::default()
        );
        assert_eq!(targets.head.state, HeadTranslationState::Unavailable);
        assert_eq!(targets.head.rotation, rotation);
    }

    #[test]
    fn zero_unavailable_constant_has_no_observation_and_zero_motion() {
        assert_eq!(
            VirtualBodyTargets::ZERO_UNAVAILABLE.head.translation,
            TranslationMeters::ZERO
        );
        assert_eq!(
            VirtualBodyTargets::ZERO_UNAVAILABLE.body_compensation,
            BodyTranslationCompensation::default()
        );
        assert!(!matches!(
            VirtualBodyTargets::ZERO_UNAVAILABLE.head.state,
            HeadTranslationState::Tracked | HeadTranslationState::Degraded
        ));
    }

    #[test]
    fn degraded_state_is_preserved_through_splitting() {
        let profile = VirtualBodyProfile::default();
        let targets = build_virtual_body_targets(
            &HeadTranslationSignal::degraded(0.02, 0.02, 0.02),
            HeadPose::default(),
            &profile,
            1.0,
        );
        assert_eq!(targets.head.state, HeadTranslationState::Degraded);
        assert_relative_eq!(targets.body_compensation.y, 0.016, epsilon = 1e-6);
    }

    #[test]
    fn invalid_body_scale_never_publishes_targets() {
        let profile = VirtualBodyProfile::default();
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let targets = build_virtual_body_targets(
                &tracked(0.10, 0.10, 0.10),
                HeadPose::default(),
                &profile,
                scale,
            );
            assert_eq!(targets.head.state, HeadTranslationState::Unavailable);
            assert_eq!(
                targets.body_compensation,
                BodyTranslationCompensation::default()
            );
        }
    }

    #[test]
    fn output_is_deterministic_for_identical_inputs() {
        let profile = VirtualBodyProfile::default();
        let a = build_virtual_body_targets(
            &tracked(0.09, -0.22, 0.17),
            HeadPose {
                yaw_rad: 0.3,
                pitch_rad: 0.1,
                roll_rad: -0.2,
            },
            &profile,
            0.85,
        );
        let b = build_virtual_body_targets(
            &tracked(0.09, -0.22, 0.17),
            HeadPose {
                yaw_rad: 0.3,
                pitch_rad: 0.1,
                roll_rad: -0.2,
            },
            &profile,
            0.85,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn profile_rejects_out_of_range_fields() {
        assert_eq!(VirtualBodyProfile::default().validate(), Ok(()));

        let bad_gain = VirtualBodyProfile {
            y_root_gain: 1.2,
            ..VirtualBodyProfile::default()
        };
        assert_eq!(
            bad_gain.validate(),
            Err(VirtualBodyProfileError::GainOutOfRange {
                name: "y_root_gain",
                value: 1.2,
            })
        );

        let bad_cap = VirtualBodyProfile {
            y_residual_cap_ratio: 0.0,
            ..VirtualBodyProfile::default()
        };
        assert_eq!(
            bad_cap.validate(),
            Err(VirtualBodyProfileError::NonPositiveResidualCap { value: 0.0 })
        );

        let bad_rotation = VirtualBodyProfile {
            max_body_rotation_rad: PI + 0.1,
            ..VirtualBodyProfile::default()
        };
        assert_eq!(
            bad_rotation.validate(),
            Err(VirtualBodyProfileError::RotationLimitOutOfRange { value: PI + 0.1 })
        );
    }

    #[test]
    fn invalid_profiles_fail_closed_instead_of_panicking() {
        let bad = VirtualBodyProfile {
            z_root_gain: f32::NAN,
            ..VirtualBodyProfile::default()
        };
        let targets =
            build_virtual_body_targets(&tracked(0.05, 0.05, 0.05), HeadPose::default(), &bad, 1.0);
        assert_eq!(targets, VirtualBodyTargets::ZERO_UNAVAILABLE);
    }

    #[test]
    fn negative_inputs_split_symmetrically() {
        let profile = VirtualBodyProfile::default();
        let targets = build_virtual_body_targets(
            &tracked(-0.06, -0.06, -0.06),
            HeadPose::default(),
            &profile,
            1.0,
        );
        assert_eq!(targets.body_compensation.x, 0.0);
        assert_relative_eq!(targets.head.translation.x, -0.06, epsilon = 1e-6);
        assert_relative_eq!(targets.body_compensation.y, -0.048, epsilon = 1e-6);
        assert_relative_eq!(targets.body_compensation.z, -0.054, epsilon = 1e-6);
    }
}
