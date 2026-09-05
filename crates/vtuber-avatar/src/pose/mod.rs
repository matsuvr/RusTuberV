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
    BodyBoneHalfLives, BodyBoneRotationLimits, BodyBoneWeights, BodyTrackingProfile,
    BoneRotationLimit,
};

fn limit_degrees(yaw: f32, pitch: f32, roll: f32) -> BoneRotationLimit {
    BoneRotationLimit {
        yaw_radians: yaw.to_radians(),
        pitch_radians: pitch.to_radians(),
        roll_radians: roll.to_radians(),
    }
}

/// Head-to-body rotation distribution tuned for human-like propagation.
///
/// In a human neck the head and neck rotate as one unit: the neck carries a
/// large share of every head rotation at nearly the same speed. The
/// distribution below follows the common practice of webcam VTuber trackers
/// (kalidoface/kalidokit drives the neck with a ~0.7 dampener of the head
/// rotation, other rigs use a 40/60 neck/head split): the neck share is
/// roughly two thirds of the head share on every axis and its smoothing
/// half-life stays close to the head's, while the torso keeps a visible,
/// slower share so large turns propagate into the chest, shoulders, and
/// spine instead of kinking at the neck. The hips keep a faint share with
/// the slowest half-life so every rotation propagates through the whole
/// body to the legs and feet, like a real body shifting its weight. Torso
/// engagement starts early (8 degrees) and is complete by 35 degrees,
/// mirroring the low "body stiffness" default of established trackers.
pub fn natural_body_tracking_profile() -> BodyTrackingProfile {
    BodyTrackingProfile {
        small_yaw_weights: BodyBoneWeights {
            head: 0.60,
            neck: 0.40,
            upper_chest: 0.0,
            chest: 0.0,
            spine: 0.0,
            hips: 0.0,
        },
        large_yaw_weights: BodyBoneWeights {
            head: 0.34,
            neck: 0.21,
            upper_chest: 0.19,
            chest: 0.12,
            spine: 0.10,
            hips: 0.04,
        },
        pitch_weights: BodyBoneWeights {
            head: 0.47,
            neck: 0.26,
            upper_chest: 0.13,
            chest: 0.07,
            spine: 0.04,
            hips: 0.03,
        },
        roll_weights: BodyBoneWeights {
            head: 0.47,
            neck: 0.26,
            upper_chest: 0.13,
            chest: 0.07,
            spine: 0.04,
            hips: 0.03,
        },
        yaw_body_engagement_start_radians: 8.0_f32.to_radians(),
        yaw_body_engagement_full_radians: 35.0_f32.to_radians(),
        bone_half_lives: BodyBoneHalfLives {
            head_seconds: 0.055,
            neck_seconds: 0.075,
            upper_chest_seconds: 0.180,
            chest_seconds: 0.285,
            spine_seconds: 0.450,
            hips_seconds: 0.650,
        },
        bone_rotation_limits: BodyBoneRotationLimits {
            head: limit_degrees(45.0, 30.0, 25.0),
            neck: limit_degrees(30.0, 24.0, 20.0),
            upper_chest: limit_degrees(18.0, 10.0, 10.0),
            chest: limit_degrees(12.0, 6.0, 5.0),
            spine: limit_degrees(8.0, 5.0, 4.0),
            hips: limit_degrees(10.0, 4.0, 4.0),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-5;

    fn sum(weights: BodyBoneWeights) -> f32 {
        weights.head
            + weights.neck
            + weights.upper_chest
            + weights.chest
            + weights.spine
            + weights.hips
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
    fn hips_receive_a_faint_share_of_large_motion() {
        let profile = natural_body_tracking_profile();
        // Small yaw keeps the body planted; large yaw, pitch, and roll all
        // reach the hips so the legs follow as child bones.
        assert_eq!(profile.small_yaw_weights.hips, 0.0);
        assert!(profile.large_yaw_weights.hips > 0.0);
        assert!(profile.pitch_weights.hips > 0.0);
        assert!(profile.roll_weights.hips > 0.0);
        // Hips respond slowest so the sway reads as weight shift.
        assert!(
            profile.bone_half_lives.hips_seconds > profile.bone_half_lives.spine_seconds
        );
    }

    #[test]
    fn yaw_policy_keeps_the_neck_on_the_head_axis_and_engages_the_torso_early() {
        let profile = natural_body_tracking_profile();
        // Small yaw: neck carries about two thirds of the head share so the
        // head and neck rotate as one unit, matching common tracker practice.
        assert_eq!(
            profile.small_yaw_weights,
            BodyBoneWeights {
                head: 0.60,
                neck: 0.40,
                upper_chest: 0.0,
                chest: 0.0,
                spine: 0.0,
                hips: 0.0,
            }
        );
        // Large yaw: the torso takes a visible share so big head turns turn
        // the chest and shoulders instead of kinking at the neck, and the
        // hips keep a faint share for whole-body propagation.
        assert_eq!(
            profile.large_yaw_weights,
            BodyBoneWeights {
                head: 0.34,
                neck: 0.21,
                upper_chest: 0.19,
                chest: 0.12,
                spine: 0.10,
                hips: 0.04,
            }
        );
        // Torso engagement starts early and completes by 35 degrees, the
        // low "body stiffness" default of established trackers.
        assert_eq!(
            profile.yaw_body_engagement_start_radians,
            8.0_f32.to_radians()
        );
        assert_eq!(
            profile.yaw_body_engagement_full_radians,
            35.0_f32.to_radians()
        );
    }

    #[test]
    fn neck_tracks_close_to_the_head_on_every_axis() {
        let profile = natural_body_tracking_profile();
        let head = profile.bone_half_lives.head_seconds;
        let neck = profile.bone_half_lives.neck_seconds;
        assert!(
            neck <= head * 2.0,
            "neck smoothing must stay near the head timing: head={head}, neck={neck}"
        );
        // At the physiological 60-degree maximum the unclamped neck share
        // must fit inside the neck rotation limits.
        let neck_limits = profile.bone_rotation_limits.neck;
        let yaw_share = profile.small_yaw_weights.neck;
        let pitch_share = profile.pitch_weights.neck;
        let roll_share = profile.roll_weights.neck;
        assert!(60.0_f32.to_radians() * yaw_share <= neck_limits.yaw_radians);
        assert!(60.0_f32.to_radians() * pitch_share <= neck_limits.pitch_radians);
        assert!(60.0_f32.to_radians() * roll_share <= neck_limits.roll_radians);
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
            (limits.hips.yaw_radians, profile.large_yaw_weights.hips),
            (limits.hips.pitch_radians, profile.pitch_weights.hips),
            (limits.hips.roll_radians, profile.roll_weights.hips),
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
