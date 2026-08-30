//! Direct-pose body-tracking bridge.
//!
//! Calibrated semantic yaw/pitch/roll values are forwarded unchanged to the
//! dependency-owned `BodyTrackingPoseInput`. All bone distribution, rest-space
//! conversion, filtering, and additive composition lives in `bevy_vrm1`.

pub mod system;
pub use system::{
    PoseApplyMetrics, reset_pose_metrics_on_lifecycle_change, update_body_tracking_pose_input,
};

use bevy_vrm1::prelude::{
    BodyBoneRotationLimits, BodyBoneWeights, BodyTrackingProfile, BoneRotationLimit,
};

fn limit_degrees(yaw: f32, pitch: f32, roll: f32) -> BoneRotationLimit {
    BoneRotationLimit {
        yaw_radians: yaw.to_radians(),
        pitch_radians: pitch.to_radians(),
        roll_radians: roll.to_radians(),
    }
}

/// Head-to-torso rotation distribution tuned for human-like propagation.
pub fn natural_body_tracking_profile() -> BodyTrackingProfile {
    BodyTrackingProfile {
        pitch_weights: BodyBoneWeights {
            head: 0.58,
            neck: 0.20,
            upper_chest: 0.12,
            chest: 0.06,
            spine: 0.04,
        },
        roll_weights: BodyBoneWeights {
            head: 0.60,
            neck: 0.20,
            upper_chest: 0.12,
            chest: 0.05,
            spine: 0.03,
        },
        bone_rotation_limits: BodyBoneRotationLimits {
            head: limit_degrees(45.0, 30.0, 25.0),
            neck: limit_degrees(25.0, 20.0, 15.0),
            upper_chest: limit_degrees(18.0, 10.0, 8.0),
            chest: limit_degrees(12.0, 6.0, 5.0),
            spine: limit_degrees(8.0, 5.0, 4.0),
        },
        ..BodyTrackingProfile::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-5;

    fn sum(weights: BodyBoneWeights) -> f32 {
        weights.head + weights.neck + weights.upper_chest + weights.chest + weights.spine
    }

    #[test]
    fn every_axis_distribution_sums_to_one() {
        let profile = natural_body_tracking_profile();
        for weights in [
            profile.small_yaw_weights,
            profile.large_yaw_weights,
            profile.pitch_weights,
            profile.roll_weights,
        ] {
            assert!((sum(weights) - 1.0).abs() < EPSILON);
        }
    }

    #[test]
    fn pitch_and_roll_reach_the_torso_visibly() {
        let profile = natural_body_tracking_profile();
        let library_default = BodyTrackingProfile::default();
        for weights in [profile.pitch_weights, profile.roll_weights] {
            let torso_share = weights.upper_chest + weights.chest + weights.spine;
            assert!(
                torso_share >= 0.15,
                "torso share must be visible: {torso_share}"
            );
        }
        let default_pitch_torso = library_default.pitch_weights.upper_chest
            + library_default.pitch_weights.chest
            + library_default.pitch_weights.spine;
        let tuned_pitch_torso = profile.pitch_weights.upper_chest
            + profile.pitch_weights.chest
            + profile.pitch_weights.spine;
        assert!(tuned_pitch_torso > default_pitch_torso * 2.0);
    }

    #[test]
    fn yaw_policy_matches_the_library_default() {
        let profile = natural_body_tracking_profile();
        let library_default = BodyTrackingProfile::default();
        assert_eq!(profile.small_yaw_weights, library_default.small_yaw_weights);
        assert_eq!(profile.large_yaw_weights, library_default.large_yaw_weights);
        assert_eq!(
            profile.yaw_body_engagement_start_radians,
            library_default.yaw_body_engagement_start_radians
        );
        assert_eq!(
            profile.yaw_body_engagement_full_radians,
            library_default.yaw_body_engagement_full_radians
        );
    }

    #[test]
    fn torso_pitch_and_roll_limits_allow_the_tuned_distribution() {
        let profile = natural_body_tracking_profile();
        let limits = profile.bone_rotation_limits;
        // Each torso bone's largest single-axis contribution must fit inside
        // its per-bone limit so the measured pose sum survives unclamped for
        // any physiologically realistic input (60 degrees).
        let torso_limit_cases = [
            (
                limits.upper_chest.pitch_radians,
                profile.pitch_weights.upper_chest,
            ),
            (limits.chest.pitch_radians, profile.pitch_weights.chest),
            (limits.spine.pitch_radians, profile.pitch_weights.spine),
            (
                limits.upper_chest.roll_radians,
                profile.roll_weights.upper_chest,
            ),
            (limits.chest.roll_radians, profile.roll_weights.chest),
            (limits.spine.roll_radians, profile.roll_weights.spine),
        ];
        for (limit, share) in torso_limit_cases {
            assert!(
                60.0_f32.to_radians() * share <= limit,
                "torso limit {limit} must fit a 60-degree share {share}"
            );
        }
    }

    #[test]
    fn construction_is_deterministic() {
        assert_eq!(
            natural_body_tracking_profile(),
            natural_body_tracking_profile()
        );
    }
}
