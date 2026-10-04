// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Tests for per-model arm tuning and deterministic pose transitions.

use vtuber_avatar::{
    ARM_POSE_PROFILE_OVERRIDE_VERSION, ArmPoseOverrideStore, ArmPoseProfile,
    ArmPoseProfileOverride, ArmPoseProfileOverrideError, AvatarAssetId,
};

// Profile changes and return durations are exercised through JointPath and
// update_upper_limb_targets; the retired unconstrained slerp has no second writer.

#[test]
fn override_store_is_model_keyed_resettable_and_exportable() {
    let first = AvatarAssetId::new("sha256:first");
    let second = AvatarAssetId::new("sha256:second");
    let first_profile = ArmPoseProfile {
        arm_drop_radians: 0.55,
        ..Default::default()
    };
    let second_profile = ArmPoseProfile {
        arm_drop_radians: 0.85,
        ..Default::default()
    };

    let mut store = ArmPoseOverrideStore::default();
    store
        .set(
            first.0.clone(),
            ArmPoseProfileOverride::from_profile(first_profile),
        )
        .unwrap();
    store
        .set(
            second.0.clone(),
            ArmPoseProfileOverride::from_profile(second_profile),
        )
        .unwrap();

    assert_eq!(store.len(), 2);
    assert_eq!(store.profile_for(&first).unwrap(), first_profile);
    assert_eq!(store.profile_for(&second).unwrap(), second_profile);
    let exported: Vec<_> = store.entries().collect();
    assert_eq!(exported.len(), 2);
    assert!(exported.iter().any(|(id, _)| *id == first.0));

    // The resource remains usable across an avatar unload/reload boundary.
    assert!(store.reset(&first));
    assert!(store.profile_for(&first).is_none());
    assert_eq!(store.profile_for(&second).unwrap(), second_profile);
    assert!(!store.reset(&first));
}

#[test]
fn invalid_or_unknown_persisted_entries_are_rejected() {
    let mut store = ArmPoseOverrideStore::default();
    let valid = ArmPoseProfileOverride::from_profile(ArmPoseProfile::default());

    let mut unknown_version = valid;
    unknown_version.schema_version = ARM_POSE_PROFILE_OVERRIDE_VERSION + 1;
    assert_eq!(
        store.set("model", unknown_version),
        Err(vtuber_avatar::ArmPoseOverrideStoreError::InvalidProfile(
            ArmPoseProfileOverrideError::UnsupportedVersion {
                version: ARM_POSE_PROFILE_OVERRIDE_VERSION + 1
            }
        ))
    );

    let mut non_finite = valid;
    non_finite.finger_curl_radians = f32::NAN;
    assert!(matches!(
        store.set("model", non_finite),
        Err(vtuber_avatar::ArmPoseOverrideStoreError::InvalidProfile(
            ArmPoseProfileOverrideError::OutOfRangeOrNonFinite
        ))
    ));
    assert_eq!(
        store.set("", valid),
        Err(vtuber_avatar::ArmPoseOverrideStoreError::EmptyModelId)
    );

    let accepted = store.import_entries([
        ("valid".to_owned(), valid),
        ("bad-version".to_owned(), unknown_version),
        ("bad-number".to_owned(), non_finite),
    ]);
    assert_eq!(accepted, 1);
    assert_eq!(store.len(), 1);
    assert!(store.profile_for(&AvatarAssetId::new("valid")).is_some());
}
