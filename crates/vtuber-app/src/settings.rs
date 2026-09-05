//! Application-owned persistent settings.
//!
//! This module owns the user-config directory policy and serialization. The
//! avatar crate only receives validated `ArmPoseOverrideStore` values and
//! never needs to know where they came from on disk.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::prelude::{Res, Resource};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use vtuber_avatar::{ArmPoseOverrideStore, ArmPoseProfileOverride, DynamicArmProfileOverride};

/// Version of the application settings document.
pub const ARM_POSE_SETTINGS_SCHEMA_VERSION: u32 = 1;
/// File name used in the per-user application configuration directory.
pub const ARM_POSE_SETTINGS_FILE_NAME: &str = "settings.toml";

/// Application resource that owns the persistent arm-pose settings location.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct ArmPoseSettings {
    path: Option<PathBuf>,
    restored: ArmPoseOverrideStore,
}

impl Default for ArmPoseSettings {
    fn default() -> Self {
        Self {
            path: default_settings_path(),
            restored: ArmPoseOverrideStore::default(),
        }
    }
}

impl ArmPoseSettings {
    /// Loads settings from an explicit path. Missing or invalid data safely
    /// produces an empty store, which means automatic geometry-derived pose.
    #[must_use]
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let restored = match load_arm_pose_overrides(&path) {
            Ok(store) => store,
            Err(error) => {
                let backup = path.with_extension("toml.invalid");
                if path.is_file() && !backup.exists() {
                    let _ = fs::copy(&path, &backup);
                }
                bevy::log::warn!("arm-pose settings ignored: {error}");
                ArmPoseOverrideStore::default()
            }
        };
        Self {
            path: Some(path),
            restored,
        }
    }

    /// Loads settings from the platform user configuration directory.
    #[must_use]
    pub fn load_default() -> Self {
        match default_settings_path() {
            Some(path) => Self::load(path),
            None => Self::default(),
        }
    }

    /// Creates an empty settings resource targeting an explicit path.
    #[must_use]
    pub fn empty_at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            restored: ArmPoseOverrideStore::default(),
        }
    }

    /// Returns the configured settings path, if a platform config directory
    /// was available.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Returns the validated entries read before the Bevy app started.
    pub fn restored_entries(&self) -> impl Iterator<Item = (&str, &ArmPoseProfileOverride)> {
        self.restored.entries()
    }

    /// Saves the current validated avatar store.
    pub fn save(&self, store: &ArmPoseOverrideStore) -> Result<(), ArmPoseSettingsError> {
        let Some(path) = &self.path else {
            return Err(ArmPoseSettingsError::NoConfigDirectory);
        };
        save_arm_pose_overrides(path, store)
    }
}

/// Copies the validated settings loaded before startup into the avatar
/// resource. The optional resource keeps the app settings plugin harmless in
/// tests that do not install the avatar plugin.
pub fn restore_arm_pose_settings_system(
    settings: Res<ArmPoseSettings>,
    mut overrides: Option<bevy::prelude::ResMut<ArmPoseOverrideStore>>,
) {
    let Some(overrides) = overrides.as_deref_mut() else {
        return;
    };
    overrides.import_entries(
        settings
            .restored_entries()
            .map(|(model_id, profile)| (model_id.to_owned(), *profile)),
    );
}

/// Returns the application settings path for the current platform.
#[must_use]
pub fn default_settings_path() -> Option<PathBuf> {
    ProjectDirs::from("", "", "RusTuberV")
        .map(|dirs| dirs.config_dir().join(ARM_POSE_SETTINGS_FILE_NAME))
}

/// Loads and validates the arm-pose settings document.
pub fn load_arm_pose_overrides(path: &Path) -> Result<ArmPoseOverrideStore, ArmPoseSettingsError> {
    if !path.is_file() {
        return Ok(ArmPoseOverrideStore::default());
    }

    let text = fs::read_to_string(path)?;
    let document: ArmPoseSettingsDocument = toml::from_str(&text)?;
    if document.schema_version != ARM_POSE_SETTINGS_SCHEMA_VERSION {
        return Err(ArmPoseSettingsError::UnsupportedSchema {
            version: document.schema_version,
        });
    }

    let expected = document.arm_pose_overrides.len();
    let mut store = ArmPoseOverrideStore::default();
    let entries = document
        .arm_pose_overrides
        .into_iter()
        .map(|(model_id, profile)| (model_id, profile.into_runtime()));
    let accepted = store.import_entries(entries);
    if accepted != expected {
        return Err(ArmPoseSettingsError::InvalidEntry);
    }

    // Invalid/corrupt dynamic entries fall back to automatic defaults instead
    // of failing startup or panicking.
    let dynamic_entries = document
        .dynamic_arm_profiles
        .into_iter()
        .map(|(model_id, profile)| (model_id, profile.into_runtime()));
    store.import_dynamic_entries(dynamic_entries);
    Ok(store)
}

/// Saves only validated entries using a temporary sibling file before the
/// final rename, so a partial write does not become the active settings file.
pub fn save_arm_pose_overrides(
    path: &Path,
    store: &ArmPoseOverrideStore,
) -> Result<(), ArmPoseSettingsError> {
    let mut document = if path.is_file() {
        let text = fs::read_to_string(path)?;
        toml::from_str::<ArmPoseSettingsDocument>(&text)?
    } else {
        ArmPoseSettingsDocument {
            schema_version: ARM_POSE_SETTINGS_SCHEMA_VERSION,
            arm_pose_overrides: BTreeMap::new(),
            dynamic_arm_profiles: BTreeMap::new(),
        }
    };
    document.schema_version = ARM_POSE_SETTINGS_SCHEMA_VERSION;
    document.arm_pose_overrides = store
        .entries()
        .map(|(model_id, profile)| (model_id.to_owned(), PersistedArmPoseProfile::from(*profile)))
        .collect();
    document.dynamic_arm_profiles = store
        .dynamic_entries()
        .map(|(model_id, profile)| {
            (
                model_id.to_owned(),
                PersistedDynamicArmProfile::from(*profile),
            )
        })
        .collect();
    write_settings_atomically(path, &toml::to_string_pretty(&document)?)
}

fn write_settings_atomically(path: &Path, text: &str) -> Result<(), ArmPoseSettingsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("toml.tmp");
    fs::write(&temporary, text)?;
    if let Err(rename_error) = fs::rename(&temporary, path) {
        // Windows does not replace an existing file with rename. The target
        // is the explicitly selected settings file, so replacing it here is
        // within the persistence contract.
        if path.exists() {
            fs::remove_file(path)?;
            fs::rename(&temporary, path)?;
        } else {
            return Err(rename_error.into());
        }
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct ArmPoseSettingsDocument {
    schema_version: u32,
    #[serde(default)]
    arm_pose_overrides: BTreeMap<String, PersistedArmPoseProfile>,
    /// Per-model dynamic arm profiles; absent documents load as automatic
    /// defaults.
    #[serde(default)]
    dynamic_arm_profiles: BTreeMap<String, PersistedDynamicArmProfile>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedDynamicArmProfile {
    schema_version: u32,
    hand_anchor_ratio: [f32; 3],
    compensation_gains: [f32; 3],
    elbow_swivel_radians: f32,
    swivel_transition_width_ratio: f32,
    pole_influence: f32,
    twist_relax_weight: f32,
    twist_parent_child_crossfade: f32,
    shoulder_elevation_trim_radians: f32,
}

impl From<DynamicArmProfileOverride> for PersistedDynamicArmProfile {
    fn from(profile: DynamicArmProfileOverride) -> Self {
        Self {
            schema_version: profile.schema_version,
            hand_anchor_ratio: profile.hand_anchor_ratio,
            compensation_gains: profile.compensation_gains,
            elbow_swivel_radians: profile.elbow_swivel_radians,
            swivel_transition_width_ratio: profile.swivel_transition_width_ratio,
            pole_influence: profile.pole_influence,
            twist_relax_weight: profile.twist_relax_weight,
            twist_parent_child_crossfade: profile.twist_parent_child_crossfade,
            shoulder_elevation_trim_radians: profile.shoulder_elevation_trim_radians,
        }
    }
}

impl PersistedDynamicArmProfile {
    fn into_runtime(self) -> DynamicArmProfileOverride {
        DynamicArmProfileOverride {
            schema_version: self.schema_version,
            hand_anchor_ratio: self.hand_anchor_ratio,
            compensation_gains: self.compensation_gains,
            elbow_swivel_radians: self.elbow_swivel_radians,
            swivel_transition_width_ratio: self.swivel_transition_width_ratio,
            pole_influence: self.pole_influence,
            twist_relax_weight: self.twist_relax_weight,
            twist_parent_child_crossfade: self.twist_parent_child_crossfade,
            shoulder_elevation_trim_radians: self.shoulder_elevation_trim_radians,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedArmPoseProfile {
    schema_version: u32,
    arm_drop_radians: f32,
    reach_ratio: f32,
    forward_hand_offset_ratio: f32,
    elbow_pole_offset_ratio: f32,
    shoulder_follow_weight: f32,
    finger_curl_radians: f32,
}

impl From<ArmPoseProfileOverride> for PersistedArmPoseProfile {
    fn from(profile: ArmPoseProfileOverride) -> Self {
        Self {
            schema_version: profile.schema_version,
            arm_drop_radians: profile.arm_drop_radians,
            reach_ratio: profile.reach_ratio,
            forward_hand_offset_ratio: profile.forward_hand_offset_ratio,
            elbow_pole_offset_ratio: profile.elbow_pole_offset_ratio,
            shoulder_follow_weight: profile.shoulder_follow_weight,
            finger_curl_radians: profile.finger_curl_radians,
        }
    }
}

impl PersistedArmPoseProfile {
    fn into_runtime(self) -> ArmPoseProfileOverride {
        ArmPoseProfileOverride {
            schema_version: self.schema_version,
            arm_drop_radians: self.arm_drop_radians,
            reach_ratio: self.reach_ratio,
            forward_hand_offset_ratio: self.forward_hand_offset_ratio,
            elbow_pole_offset_ratio: self.elbow_pole_offset_ratio,
            shoulder_follow_weight: self.shoulder_follow_weight,
            finger_curl_radians: self.finger_curl_radians,
        }
    }
}

/// Errors returned by the application settings boundary.
#[derive(Debug, thiserror::Error)]
pub enum ArmPoseSettingsError {
    /// The platform did not provide a user config directory.
    #[error("no user configuration directory is available")]
    NoConfigDirectory,
    /// File-system read/write failure.
    #[error("settings I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// TOML decoding failure.
    #[error("settings TOML is malformed: {0}")]
    Decode(#[from] toml::de::Error),
    /// TOML encoding failure.
    #[error("settings TOML could not be encoded: {0}")]
    Encode(#[from] toml::ser::Error),
    /// A document version not understood by this build was encountered.
    #[error("unsupported settings schema version {version}")]
    UnsupportedSchema {
        /// Encountered settings schema version.
        version: u32,
    },
    /// At least one persisted profile was invalid.
    #[error("settings contain an invalid arm-pose profile")]
    InvalidEntry,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use vtuber_avatar::{ArmPoseProfile, AvatarAssetId};

    fn profile(drop: f32) -> ArmPoseProfileOverride {
        ArmPoseProfileOverride::from_profile(ArmPoseProfile {
            arm_drop_radians: drop,
            ..Default::default()
        })
    }

    #[test]
    fn round_trip_and_restart_equivalent_load_restore_model_local_overrides() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        let first = AvatarAssetId::new("sha256:first");
        let second = AvatarAssetId::new("sha256:second");
        let mut store = ArmPoseOverrideStore::default();
        store.set(first.0.clone(), profile(0.55)).unwrap();
        store.set(second.0.clone(), profile(0.85)).unwrap();

        save_arm_pose_overrides(&path, &store).expect("settings save");
        let restarted = load_arm_pose_overrides(&path).expect("settings reload");

        assert_eq!(
            restarted.profile_for(&first).unwrap().arm_drop_radians,
            0.55
        );
        assert_eq!(
            restarted.profile_for(&second).unwrap().arm_drop_radians,
            0.85
        );
    }

    #[test]
    fn reset_persists_only_the_selected_model() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        let first = AvatarAssetId::new("first");
        let second = AvatarAssetId::new("second");
        let mut store = ArmPoseOverrideStore::default();
        store.set(first.0.clone(), profile(0.55)).unwrap();
        store.set(second.0.clone(), profile(0.85)).unwrap();
        assert!(store.reset(&first));
        save_arm_pose_overrides(&path, &store).expect("settings save");

        let restored = load_arm_pose_overrides(&path).expect("settings reload");
        assert!(restored.profile_for(&first).is_none());
        assert!(restored.profile_for(&second).is_some());
    }

    #[test]
    fn unknown_malformed_and_invalid_values_fall_back_to_empty_defaults() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);

        fs::write(&path, "schema_version = 99\n").unwrap();
        assert!(load_arm_pose_overrides(&path).is_err());

        fs::write(&path, "this is not valid TOML = [").unwrap();
        assert!(load_arm_pose_overrides(&path).is_err());

        fs::write(
            &path,
            "schema_version = 1\n[arm_pose_overrides.bad]\nschema_version = 1\narm_drop_radians = 999\nreach_ratio = 0.99\nforward_hand_offset_ratio = 0.081\nelbow_pole_offset_ratio = 0.05\nshoulder_follow_weight = 0.18\nfinger_curl_radians = 0.17\n",
        )
        .unwrap();
        assert!(load_arm_pose_overrides(&path).is_err());

        fs::write(
            &path,
            "schema_version = 1\n[arm_pose_overrides.bad]\nschema_version = 1\narm_drop_radians = nan\nreach_ratio = 0.99\nforward_hand_offset_ratio = 0.081\nelbow_pole_offset_ratio = 0.05\nshoulder_follow_weight = 0.18\nfinger_curl_radians = 0.17\n",
        )
        .unwrap();
        assert!(load_arm_pose_overrides(&path).is_err());

        let loaded = ArmPoseSettings::load(&path);
        assert_eq!(loaded.restored_entries().count(), 0);
        assert!(path.with_extension("toml.invalid").is_file());
    }
}

#[cfg(test)]
mod dynamic_profile_tests {
    use super::*;
    use tempfile::tempdir;
    use vtuber_avatar::{
        ArmPoseProfile, ArmPoseProfileOverride, AvatarAssetId,
        DYNAMIC_ARM_PROFILE_OVERRIDE_VERSION, DynamicArmProfile, DynamicArmProfileOverride,
    };

    fn dynamic_override(trim: f32) -> DynamicArmProfileOverride {
        DynamicArmProfileOverride::from_profile(DynamicArmProfile {
            shoulder_elevation_trim_radians: trim,
            ..DynamicArmProfile::default()
        })
    }

    #[test]
    fn dynamic_profiles_persist_and_reload_per_model() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        let first = AvatarAssetId::new("sha256:first");
        let second = AvatarAssetId::new("sha256:second");
        let mut store = ArmPoseOverrideStore::default();
        store
            .set_dynamic_profile(first.0.clone(), dynamic_override(-0.13))
            .unwrap();
        store
            .set_dynamic_profile(second.0.clone(), dynamic_override(0.05))
            .unwrap();

        save_arm_pose_overrides(&path, &store).expect("settings save");
        let reloaded = load_arm_pose_overrides(&path).expect("settings reload");

        let first_trim = reloaded.dynamic_profile_for(&first).unwrap();
        let second_trim = reloaded.dynamic_profile_for(&second).unwrap();
        assert!((first_trim.shoulder_elevation_trim_radians - -0.13).abs() < 1e-6);
        assert!((second_trim.shoulder_elevation_trim_radians - 0.05).abs() < 1e-6);

        // Reset removes only the selected model's entry.
        assert!(store.reset_dynamic_profile(&first));
        assert!(store.dynamic_profile_for(&first).is_none());
        assert!(store.dynamic_profile_for(&second).is_some());
    }

    #[test]
    fn corrupt_or_old_version_dynamic_entries_fall_back_to_defaults_without_panicking() {
        let directory = tempfile::tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        std::fs::write(
            &path,
            format!(
                "schema_version = {ARM_POSE_SETTINGS_SCHEMA_VERSION}\n\
                 [dynamic_arm_profiles.\"sha256:bad\"]\n\
                 schema_version = 1\n\
                 hand_anchor_ratio = [0.0, 0.0, 0.0]\n\
                 compensation_gains = [2.5, 0.0, 0.0]\n\
                 elbow_swivel_radians = 99.0\n\
                 swivel_transition_width_ratio = 0.15\n\
                 pole_influence = 0.2\n\
                 twist_relax_weight = 0.7\n\
                 twist_parent_child_crossfade = 0.9\n\
                 shoulder_elevation_trim_radians = 9.9\n"
            ),
        )
        .expect("write corrupt settings");

        // Old version and out-of-range values are dropped; startup continues.
        let store = load_arm_pose_overrides(&path).expect("settings reload");
        assert!(
            store
                .dynamic_profile_for(&AvatarAssetId::new("sha256:bad"))
                .is_none()
        );
    }

    #[test]
    fn migration_from_legacy_v1_resets_to_automatic_defaults() {
        let legacy = ArmPoseProfileOverride::from_profile(ArmPoseProfile {
            arm_drop_radians: 0.4,
            reach_ratio: 0.8,
            ..ArmPoseProfile::default()
        });
        let migrated = DynamicArmProfileOverride::from_legacy_override(&legacy);
        assert_eq!(
            migrated.schema_version,
            DYNAMIC_ARM_PROFILE_OVERRIDE_VERSION
        );
        // No semantic reuse of legacy fields: the result is the automatic
        // deterministic default profile.
        let profile = migrated.into_profile().unwrap();
        assert_eq!(profile, DynamicArmProfile::default());
    }
}
