//! User-tunable tracking distribution document.
//!
//! [`TrackingProfileDocument`] mirrors the fields of the rotation profile
//! ([`BodyTrackingProfile`]) and the arm hand-follow gains in human-readable
//! units: angles are degrees and smoothing half-lives are seconds. Every
//! field is optional, so a file only needs to carry the values being tuned;
//! absent values keep the built-in default from
//! [`crate::pose::natural_body_tracking_profile`].
//!
//! File I/O lives in the application layer; this module only defines the
//! document schema, the default template, and the merge into runtime
//! profiles.

use bevy::prelude::*;
use bevy_vrm1::prelude::{
    BodyBoneHalfLives, BodyBoneRotationLimits, BodyBoneWeights, BodyTrackingProfile,
    BoneRotationLimit,
};
use serde::{Deserialize, Serialize};

use crate::arm_pipeline::DynamicArmProfile;

/// Schema version of [`TrackingProfileDocument`].
pub const TRACKING_PROFILE_SCHEMA_VERSION: u32 = 1;

/// Rounds a radian-to-degree conversion so the generated template carries
/// clean numbers instead of float round-trip noise.
fn clean_degrees(value: f32) -> f32 {
    (value * 10_000.0).round() / 10_000.0
}

/// Per-bone weights over the `head/neck/upperChest/chest/spine` chain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingWeights {
    /// Head share.
    pub head: Option<f32>,
    /// Neck share.
    pub neck: Option<f32>,
    /// Upper-chest share.
    pub upper_chest: Option<f32>,
    /// Chest share.
    pub chest: Option<f32>,
    /// Spine share.
    pub spine: Option<f32>,
}

impl TrackingWeights {
    fn merge_into(self, weights: &mut BodyBoneWeights) {
        if let Some(value) = self.head {
            weights.head = value;
        }
        if let Some(value) = self.neck {
            weights.neck = value;
        }
        if let Some(value) = self.upper_chest {
            weights.upper_chest = value;
        }
        if let Some(value) = self.chest {
            weights.chest = value;
        }
        if let Some(value) = self.spine {
            weights.spine = value;
        }
    }

    fn from_weights(weights: BodyBoneWeights) -> Self {
        Self {
            head: Some(weights.head),
            neck: Some(weights.neck),
            upper_chest: Some(weights.upper_chest),
            chest: Some(weights.chest),
            spine: Some(weights.spine),
        }
    }
}

/// Torso engagement boundaries in degrees.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingEngagement {
    /// Absolute yaw where torso engagement starts.
    pub start: Option<f32>,
    /// Absolute yaw where torso engagement is full.
    pub full: Option<f32>,
}

/// Per-bone smoothing half-lives in seconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingHalfLives {
    /// Head half-life.
    pub head: Option<f32>,
    /// Neck half-life.
    pub neck: Option<f32>,
    /// Upper-chest half-life.
    pub upper_chest: Option<f32>,
    /// Chest half-life.
    pub chest: Option<f32>,
    /// Spine half-life.
    pub spine: Option<f32>,
}

impl TrackingHalfLives {
    fn merge_into(self, half_lives: &mut BodyBoneHalfLives) {
        if let Some(value) = self.head {
            half_lives.head_seconds = value;
        }
        if let Some(value) = self.neck {
            half_lives.neck_seconds = value;
        }
        if let Some(value) = self.upper_chest {
            half_lives.upper_chest_seconds = value;
        }
        if let Some(value) = self.chest {
            half_lives.chest_seconds = value;
        }
        if let Some(value) = self.spine {
            half_lives.spine_seconds = value;
        }
    }

    fn from_half_lives(half_lives: BodyBoneHalfLives) -> Self {
        Self {
            head: Some(half_lives.head_seconds),
            neck: Some(half_lives.neck_seconds),
            upper_chest: Some(half_lives.upper_chest_seconds),
            chest: Some(half_lives.chest_seconds),
            spine: Some(half_lives.spine_seconds),
        }
    }
}

/// Per-axis rotation limits in degrees.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingLimit {
    /// Maximum absolute yaw.
    pub yaw: Option<f32>,
    /// Maximum absolute pitch.
    pub pitch: Option<f32>,
    /// Maximum absolute roll.
    pub roll: Option<f32>,
}

impl TrackingLimit {
    fn merge_into(self, limit: &mut BoneRotationLimit) {
        if let Some(value) = self.yaw {
            limit.yaw_radians = value.to_radians();
        }
        if let Some(value) = self.pitch {
            limit.pitch_radians = value.to_radians();
        }
        if let Some(value) = self.roll {
            limit.roll_radians = value.to_radians();
        }
    }

    fn from_limit(limit: BoneRotationLimit) -> Self {
        Self {
            yaw: Some(clean_degrees(limit.yaw_radians.to_degrees())),
            pitch: Some(clean_degrees(limit.pitch_radians.to_degrees())),
            roll: Some(clean_degrees(limit.roll_radians.to_degrees())),
        }
    }
}

/// Per-bone rotation limits in degrees.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingLimits {
    /// Head limits.
    pub head: Option<TrackingLimit>,
    /// Neck limits.
    pub neck: Option<TrackingLimit>,
    /// Upper-chest limits.
    pub upper_chest: Option<TrackingLimit>,
    /// Chest limits.
    pub chest: Option<TrackingLimit>,
    /// Spine limits.
    pub spine: Option<TrackingLimit>,
}

impl TrackingLimits {
    fn merge_into(self, limits: &mut BodyBoneRotationLimits) {
        if let Some(value) = self.head {
            value.merge_into(&mut limits.head);
        }
        if let Some(value) = self.neck {
            value.merge_into(&mut limits.neck);
        }
        if let Some(value) = self.upper_chest {
            value.merge_into(&mut limits.upper_chest);
        }
        if let Some(value) = self.chest {
            value.merge_into(&mut limits.chest);
        }
        if let Some(value) = self.spine {
            value.merge_into(&mut limits.spine);
        }
    }

    fn from_limits(limits: BodyBoneRotationLimits) -> Self {
        Self {
            head: Some(TrackingLimit::from_limit(limits.head)),
            neck: Some(TrackingLimit::from_limit(limits.neck)),
            upper_chest: Some(TrackingLimit::from_limit(limits.upper_chest)),
            chest: Some(TrackingLimit::from_limit(limits.chest)),
            spine: Some(TrackingLimit::from_limit(limits.spine)),
        }
    }
}

/// Rotation distribution section of the document.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BodyTrackingSection {
    /// Weight distribution for small yaw turns.
    pub small_yaw: Option<TrackingWeights>,
    /// Weight distribution for large yaw turns.
    pub large_yaw: Option<TrackingWeights>,
    /// Weight distribution for pitch.
    pub pitch: Option<TrackingWeights>,
    /// Weight distribution for roll.
    pub roll: Option<TrackingWeights>,
    /// Torso engagement boundaries in degrees.
    pub engagement: Option<TrackingEngagement>,
    /// Per-bone smoothing half-lives in seconds.
    pub half_life: Option<TrackingHalfLives>,
    /// Per-bone rotation limits in degrees.
    pub limits: Option<TrackingLimits>,
}

impl BodyTrackingSection {
    fn merge_into(self, profile: &mut BodyTrackingProfile) {
        if let Some(value) = self.small_yaw {
            value.merge_into(&mut profile.small_yaw_weights);
        }
        if let Some(value) = self.large_yaw {
            value.merge_into(&mut profile.large_yaw_weights);
        }
        if let Some(value) = self.pitch {
            value.merge_into(&mut profile.pitch_weights);
        }
        if let Some(value) = self.roll {
            value.merge_into(&mut profile.roll_weights);
        }
        if let Some(value) = self.engagement {
            if let Some(start) = value.start {
                profile.yaw_body_engagement_start_radians = start.to_radians();
            }
            if let Some(full) = value.full {
                profile.yaw_body_engagement_full_radians = full.to_radians();
            }
        }
        if let Some(value) = self.half_life {
            value.merge_into(&mut profile.bone_half_lives);
        }
        if let Some(value) = self.limits {
            value.merge_into(&mut profile.bone_rotation_limits);
        }
    }
}

/// Hand-target follow gains.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HandFollowGains {
    /// Lateral follow gain.
    pub x: Option<f32>,
    /// Vertical follow gain.
    pub y: Option<f32>,
    /// Depth follow gain.
    pub z: Option<f32>,
}

impl HandFollowGains {
    fn merge_into(self, gains: &mut Vec3) {
        if let Some(value) = self.x {
            gains.x = value;
        }
        if let Some(value) = self.y {
            gains.y = value;
        }
        if let Some(value) = self.z {
            gains.z = value;
        }
    }
}

/// Arm follow section of the document.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ArmTrackingSection {
    /// How much of the combined head/body translation each axis of the hand
    /// target follows.
    pub hand_follow: Option<HandFollowGains>,
}

/// User-tunable tracking distribution document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingProfileDocument {
    /// Document schema version.
    pub schema_version: Option<u32>,
    /// Rotation distribution over the upper-body chain.
    pub body: Option<BodyTrackingSection>,
    /// Arm hand-target follow gains.
    pub arm: Option<ArmTrackingSection>,
}

impl TrackingProfileDocument {
    /// Fully populated document carrying the current built-in defaults, used
    /// as the generated starting point for user tuning.
    #[must_use]
    pub fn template() -> Self {
        let body = crate::pose::natural_body_tracking_profile();
        let arm = DynamicArmProfile::default();
        Self {
            schema_version: Some(TRACKING_PROFILE_SCHEMA_VERSION),
            body: Some(BodyTrackingSection {
                small_yaw: Some(TrackingWeights::from_weights(body.small_yaw_weights)),
                large_yaw: Some(TrackingWeights::from_weights(body.large_yaw_weights)),
                pitch: Some(TrackingWeights::from_weights(body.pitch_weights)),
                roll: Some(TrackingWeights::from_weights(body.roll_weights)),
                engagement: Some(TrackingEngagement {
                    start: Some(clean_degrees(
                        body.yaw_body_engagement_start_radians.to_degrees(),
                    )),
                    full: Some(clean_degrees(
                        body.yaw_body_engagement_full_radians.to_degrees(),
                    )),
                }),
                half_life: Some(TrackingHalfLives::from_half_lives(body.bone_half_lives)),
                limits: Some(TrackingLimits::from_limits(body.bone_rotation_limits)),
            }),
            arm: Some(ArmTrackingSection {
                hand_follow: Some(HandFollowGains {
                    x: Some(arm.compensation_gains.x),
                    y: Some(arm.compensation_gains.y),
                    z: Some(arm.compensation_gains.z),
                }),
            }),
        }
    }

    /// Applies every present field onto the rotation profile.
    pub fn apply_body_to(&self, profile: &mut BodyTrackingProfile) {
        if let Some(section) = &self.body {
            section.merge_into(profile);
        }
    }

    /// Applies every present field onto the arm profile.
    pub fn apply_arm_to(&self, arm: &mut DynamicArmProfile) {
        if let Some(section) = &self.arm
            && let Some(gains) = section.hand_follow
        {
            gains.merge_into(&mut arm.compensation_gains);
        }
    }
}

/// Global body-tracking rotation profile applied to every bound avatar.
///
/// The application layer replaces the default with the value resolved from
/// the user tuning document; binding inserts the resource value on each
/// avatar root.
#[derive(Resource, Debug, Clone, PartialEq)]
pub struct GlobalBodyTrackingProfile(pub BodyTrackingProfile);

impl Default for GlobalBodyTrackingProfile {
    fn default() -> Self {
        Self(crate::pose::natural_body_tracking_profile())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-6;

    #[test]
    fn partial_document_overrides_only_present_fields() {
        let document: TrackingProfileDocument = toml::from_str(
            r#"
            [body.pitch]
            head = 0.7
            neck = 0.3

            [body.engagement]
            full = 40.0

            [arm.hand_follow]
            x = 0.1
            "#,
        )
        .unwrap();

        let mut body = natural_default();
        document.apply_body_to(&mut body);
        assert!((body.pitch_weights.head - 0.7).abs() < EPSILON);
        assert!((body.pitch_weights.neck - 0.3).abs() < EPSILON);
        // Untouched fields keep the default.
        assert!((body.pitch_weights.upper_chest - 0.14).abs() < EPSILON);
        assert!((body.roll_weights.head - 0.48).abs() < EPSILON);
        assert!((body.small_yaw_weights.head - 0.60).abs() < EPSILON);
        assert!((body.yaw_body_engagement_full_radians - 40.0_f32.to_radians()).abs() < EPSILON);
        assert!((body.yaw_body_engagement_start_radians - 8.0_f32.to_radians()).abs() < EPSILON);

        let mut arm = DynamicArmProfile::default();
        document.apply_arm_to(&mut arm);
        assert!((arm.compensation_gains.x - 0.1).abs() < EPSILON);
        assert!((arm.compensation_gains.z - 1.0).abs() < EPSILON);
    }

    #[test]
    fn template_serializes_and_reapplies_identically() {
        let template = TrackingProfileDocument::template();
        let text = toml::to_string_pretty(&template).unwrap();
        let parsed: TrackingProfileDocument = toml::from_str(&text).unwrap();
        assert_eq!(parsed, template);

        let mut body = natural_default();
        template.apply_body_to(&mut body);
        assert_eq!(body, natural_default());

        let mut arm = DynamicArmProfile::default();
        template.apply_arm_to(&mut arm);
        assert_eq!(arm, DynamicArmProfile::default());
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let result = toml::from_str::<TrackingProfileDocument>(
            r#"
            [body.small_yax]
            head = 0.5
            "#,
        );
        assert!(
            result.is_err(),
            "typos must fail loudly instead of being ignored"
        );
    }

    fn natural_default() -> BodyTrackingProfile {
        crate::pose::natural_body_tracking_profile()
    }
}
