//! Application-owned persistent settings.
//!
//! This module owns the user-config directory policy and serialization. The
//! avatar crate only receives validated `ArmPoseOverrideStore` values and
//! never needs to know where they came from on disk.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::prelude::{Res, ResMut, Resource};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::expression_keys::{ExpressionBindingStore, ExpressionBindings};
use vtuber_avatar::{ArmPoseOverrideStore, ArmPoseProfileOverride, DynamicArmProfileOverride};

/// Version of the application settings document.
pub const ARM_POSE_SETTINGS_SCHEMA_VERSION: u32 = 1;
/// File name used in the per-user application configuration directory.
pub const ARM_POSE_SETTINGS_FILE_NAME: &str = "settings.toml";

/// UI language. Japanese is the initial choice, independent of the OS locale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UiLanguage {
    /// 日本語（既定）。
    #[default]
    Ja,
    /// English.
    En,
    /// 简体中文。
    #[serde(rename = "zh-Hans", alias = "zh")]
    Zh,
    /// 한국어.
    Ko,
}

impl UiLanguage {
    /// Select a translation. Every call supplies all four languages; no
    /// runtime translation lookup, missing-key fallback, or network is used.
    #[must_use]
    pub const fn pick<'a>(self, ja: &'a str, en: &'a str, zh: &'a str, ko: &'a str) -> &'a str {
        match self {
            Self::Ja => ja,
            Self::En => en,
            Self::Zh => zh,
            Self::Ko => ko,
        }
    }
}

/// Application resource that owns the persistent settings document location.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct ArmPoseSettings {
    path: Option<PathBuf>,
    restored: ArmPoseOverrideStore,
    restored_expression_bindings: ExpressionBindingStore,
    language: UiLanguage,
    arm_tracking_enabled: bool,
}

impl Default for ArmPoseSettings {
    fn default() -> Self {
        Self {
            path: default_settings_path(),
            restored: ArmPoseOverrideStore::default(),
            restored_expression_bindings: ExpressionBindingStore::default(),
            language: UiLanguage::default(),
            arm_tracking_enabled: false,
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
        let restored_expression_bindings = match load_expression_bindings(&path) {
            Ok(store) => store,
            Err(error) => {
                bevy::log::warn!("expression bindings ignored: {error}");
                ExpressionBindingStore::default()
            }
        };
        let language = load_language(&path);
        let arm_tracking_enabled = load_arm_tracking_enabled(&path);
        Self {
            path: Some(path),
            restored,
            restored_expression_bindings,
            language,
            arm_tracking_enabled,
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
            restored_expression_bindings: ExpressionBindingStore::default(),
            language: UiLanguage::default(),
            arm_tracking_enabled: false,
        }
    }

    /// Returns the configured settings path.
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

    /// Returns the expression bindings read before the Bevy app started.
    #[must_use]
    pub fn restored_expression_bindings(&self) -> &ExpressionBindingStore {
        &self.restored_expression_bindings
    }

    /// Saves expression bindings, preserving every other settings section.
    pub fn save_expression_bindings(
        &self,
        store: &ExpressionBindingStore,
    ) -> Result<(), ArmPoseSettingsError> {
        let Some(path) = &self.path else {
            return Err(ArmPoseSettingsError::NoConfigDirectory);
        };
        save_expression_bindings(path, store)
    }

    /// Returns the UI language loaded at startup.
    #[must_use]
    pub fn language(&self) -> UiLanguage {
        self.language
    }

    /// Persists the UI language and applies it to this resource.
    pub fn set_language(&mut self, language: UiLanguage) -> Result<(), ArmPoseSettingsError> {
        if let Some(path) = &self.path {
            save_language(path, language)?;
        }
        self.language = language;
        Ok(())
    }

    /// Whether observed webcam arm tracking was enabled at load time.
    #[must_use]
    pub fn arm_tracking_enabled(&self) -> bool {
        self.arm_tracking_enabled
    }

    /// Persists the observed arm-tracking switch.
    pub fn set_arm_tracking_enabled(&mut self, enabled: bool) -> Result<(), ArmPoseSettingsError> {
        if let Some(path) = &self.path {
            save_arm_tracking_enabled(path, enabled)?;
        }
        self.arm_tracking_enabled = enabled;
        Ok(())
    }
}

/// Copies validated startup settings into the avatar resource.
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

/// Copies startup expression bindings into the runtime store.
pub fn restore_expression_binding_settings_system(
    settings: Res<ArmPoseSettings>,
    mut store: ResMut<ExpressionBindingStore>,
) {
    store.replace_entries(
        settings
            .restored_expression_bindings()
            .entries()
            .map(|(model_id, bindings)| (model_id.to_owned(), bindings.clone())),
    );
}

/// Returns the application settings path for the current platform.
#[must_use]
pub fn default_settings_path() -> Option<PathBuf> {
    ProjectDirs::from("", "", "RusTuberV")
        .map(|dirs| dirs.config_dir().join(ARM_POSE_SETTINGS_FILE_NAME))
}

/// Loads UI language, defaulting to Japanese when the file is missing or unreadable.
#[must_use]
pub fn load_language(path: &Path) -> UiLanguage {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str::<ArmPoseSettingsDocument>(&text).ok())
        .map(|document| document.language)
        .unwrap_or_default()
}

/// Persists UI language, preserving the rest of the settings document.
pub fn save_language(path: &Path, language: UiLanguage) -> Result<(), ArmPoseSettingsError> {
    let mut document = if path.is_file() {
        let text = fs::read_to_string(path)?;
        toml::from_str::<ArmPoseSettingsDocument>(&text)?
    } else {
        ArmPoseSettingsDocument {
            schema_version: ARM_POSE_SETTINGS_SCHEMA_VERSION,
            language,
            arm_tracking_enabled: false,
            arm_pose_overrides: BTreeMap::new(),
            dynamic_arm_profiles: BTreeMap::new(),
            expression_bindings: BTreeMap::new(),
        }
    };
    document.language = language;
    write_settings_atomically(path, &toml::to_string_pretty(&document)?)
}

/// Loads the observed arm-tracking switch, defaulting to off.
#[must_use]
pub fn load_arm_tracking_enabled(path: &Path) -> bool {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str::<ArmPoseSettingsDocument>(&text).ok())
        .map(|document| document.arm_tracking_enabled)
        .unwrap_or(false)
}

/// Persists the observed arm-tracking switch, preserving other sections.
pub fn save_arm_tracking_enabled(path: &Path, enabled: bool) -> Result<(), ArmPoseSettingsError> {
    let mut document = if path.is_file() {
        let text = fs::read_to_string(path)?;
        toml::from_str::<ArmPoseSettingsDocument>(&text)?
    } else {
        ArmPoseSettingsDocument {
            schema_version: ARM_POSE_SETTINGS_SCHEMA_VERSION,
            language: load_language(path),
            arm_tracking_enabled: enabled,
            arm_pose_overrides: BTreeMap::new(),
            dynamic_arm_profiles: BTreeMap::new(),
            expression_bindings: BTreeMap::new(),
        }
    };
    document.arm_tracking_enabled = enabled;
    write_settings_atomically(path, &toml::to_string_pretty(&document)?)
}

/// Loads and validates model-specific expression bindings.
pub fn load_expression_bindings(
    path: &Path,
) -> Result<ExpressionBindingStore, ArmPoseSettingsError> {
    if !path.is_file() {
        return Ok(ExpressionBindingStore::default());
    }
    let text = fs::read_to_string(path)?;
    let document: ArmPoseSettingsDocument = toml::from_str(&text)?;
    if document.schema_version != ARM_POSE_SETTINGS_SCHEMA_VERSION {
        return Err(ArmPoseSettingsError::UnsupportedSchema {
            version: document.schema_version,
        });
    }
    for (model_id, bindings) in &document.expression_bindings {
        if bindings.has_duplicate_expressions() {
            return Err(ArmPoseSettingsError::InvalidExpressionBinding {
                model_id: model_id.clone(),
            });
        }
    }
    let mut store = ExpressionBindingStore::default();
    store.replace_entries(document.expression_bindings);
    Ok(store)
}

/// Saves expression bindings, preserving language, arm pose, and every other
/// model's assignments.
pub fn save_expression_bindings(
    path: &Path,
    store: &ExpressionBindingStore,
) -> Result<(), ArmPoseSettingsError> {
    let mut document = if path.is_file() {
        let text = fs::read_to_string(path)?;
        toml::from_str::<ArmPoseSettingsDocument>(&text)?
    } else {
        ArmPoseSettingsDocument {
            schema_version: ARM_POSE_SETTINGS_SCHEMA_VERSION,
            language: load_language(path),
            arm_tracking_enabled: load_arm_tracking_enabled(path),
            arm_pose_overrides: BTreeMap::new(),
            dynamic_arm_profiles: BTreeMap::new(),
            expression_bindings: BTreeMap::new(),
        }
    };
    document.schema_version = ARM_POSE_SETTINGS_SCHEMA_VERSION;
    document.expression_bindings = store
        .entries()
        .map(|(model_id, bindings)| (model_id.to_owned(), bindings.clone()))
        .collect();
    write_settings_atomically(path, &toml::to_string_pretty(&document)?)
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
    // Existing settings policy: invalid dynamic entries are ignored by the store.
    let dynamic_entries = document
        .dynamic_arm_profiles
        .into_iter()
        .map(|(model_id, profile)| (model_id, profile.into_runtime()));
    store.import_dynamic_entries(dynamic_entries);
    Ok(store)
}

/// Saves validated entries using the existing settings-file replacement policy.
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
            language: load_language(path),
            arm_tracking_enabled: load_arm_tracking_enabled(path),
            arm_pose_overrides: BTreeMap::new(),
            dynamic_arm_profiles: BTreeMap::new(),
            expression_bindings: BTreeMap::new(),
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
        // Existing Windows replacement behavior, unrelated to UI language.
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
    language: UiLanguage,
    /// Observed webcam arm tracking. Defaults off until the operator enables it.
    #[serde(default)]
    arm_tracking_enabled: bool,
    #[serde(default)]
    arm_pose_overrides: BTreeMap<String, PersistedArmPoseProfile>,
    #[serde(default)]
    dynamic_arm_profiles: BTreeMap<String, PersistedDynamicArmProfile>,
    /// Model-specific expression key assignments. A present entry with an
    /// empty `keys` map means the user removed every assignment; a missing
    /// entry means the model still uses the initial assignment.
    #[serde(default)]
    expression_bindings: BTreeMap<String, ExpressionBindings>,
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

/// Errors returned by the settings boundary.
#[derive(Debug, thiserror::Error)]
pub enum ArmPoseSettingsError {
    /// No user config directory.
    #[error("no user configuration directory is available")]
    NoConfigDirectory,
    /// Filesystem failure.
    #[error("settings I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid TOML.
    #[error("settings TOML is malformed: {0}")]
    Decode(#[from] toml::de::Error),
    /// Encoding failure.
    #[error("settings TOML could not be encoded: {0}")]
    Encode(#[from] toml::ser::Error),
    /// Unknown schema version.
    #[error("unsupported settings schema version {version}")]
    UnsupportedSchema {
        /// Encountered version.
        version: u32,
    },
    /// Invalid persisted profile.
    #[error("settings contain an invalid arm-pose profile")]
    InvalidEntry,
    /// Duplicate or unreadable expression binding for one model.
    #[error("settings contain an invalid expression binding for model {model_id}")]
    InvalidExpressionBinding {
        /// Model ID whose assignment is invalid.
        model_id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression_keys::ExpressionKey;
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
    fn ui_language_round_trips_and_defaults_to_japanese() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        assert_eq!(load_language(&path), UiLanguage::Ja);
        for language in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            save_language(&path, language).expect("language save");
            assert_eq!(load_language(&path), language);
            save_arm_pose_overrides(&path, &ArmPoseOverrideStore::default())
                .expect("settings save");
            assert_eq!(load_language(&path), language);
            assert_eq!(ArmPoseSettings::load(&path).language(), language);
        }
    }

    #[test]
    fn old_settings_without_a_language_start_in_japanese() {
        let document: ArmPoseSettingsDocument = toml::from_str("schema_version = 1\n").unwrap();
        assert_eq!(document.language, UiLanguage::Ja);
    }

    #[test]
    fn four_language_selection_is_exhaustive() {
        for (language, expected) in [
            (UiLanguage::Ja, "日本語"),
            (UiLanguage::En, "English"),
            (UiLanguage::Zh, "中文"),
            (UiLanguage::Ko, "한국어"),
        ] {
            assert_eq!(
                language.pick("日本語", "English", "中文", "한국어"),
                expected
            );
        }
    }

    fn bindings(pairs: &[(ExpressionKey, &str)]) -> ExpressionBindings {
        let mut bindings = ExpressionBindings::default();
        for (key, expression) in pairs {
            bindings.assign(*key, *expression);
        }
        bindings
    }

    #[test]
    fn expression_bindings_round_trip_and_distinguish_saved_empty() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        let mut store = ExpressionBindingStore::default();
        store.set(
            "sha256:first".into(),
            bindings(&[
                (ExpressionKey::Digit1, "happy"),
                (ExpressionKey::KeyQ, "笑顔"),
            ]),
        );
        store.set("sha256:empty".into(), ExpressionBindings::default());
        save_expression_bindings(&path, &store).expect("settings save");

        let restored = load_expression_bindings(&path).expect("settings reload");
        assert_eq!(
            restored
                .bindings_for("sha256:first")
                .expect("first model")
                .expression_for(ExpressionKey::Digit1),
            Some("happy")
        );
        assert_eq!(
            restored
                .bindings_for("sha256:first")
                .expect("first model")
                .expression_for(ExpressionKey::KeyQ),
            Some("笑顔")
        );
        assert!(
            restored
                .bindings_for("sha256:empty")
                .expect("saved empty entry")
                .is_empty()
        );
        assert!(restored.bindings_for("sha256:missing").is_none());
    }

    #[test]
    fn every_settings_writer_preserves_expression_bindings() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        let first = AvatarAssetId::new("sha256:first");

        save_language(&path, UiLanguage::En).expect("language save");
        let mut expression_store = ExpressionBindingStore::default();
        expression_store.set(first.0.clone(), bindings(&[(ExpressionKey::KeyA, "smile")]));
        save_expression_bindings(&path, &expression_store).expect("bindings save");
        let mut arm_store = ArmPoseOverrideStore::default();
        arm_store.set(first.0.clone(), profile(0.55)).unwrap();
        save_arm_pose_overrides(&path, &arm_store).expect("arm save");
        // Reverse order: language last.
        save_language(&path, UiLanguage::Ko).expect("language save");

        let restored = ArmPoseSettings::load(&path);
        assert_eq!(restored.language(), UiLanguage::Ko);
        let restored_arm = load_arm_pose_overrides(&path).expect("arm reload");
        assert_eq!(
            restored_arm.profile_for(&first).unwrap().arm_drop_radians,
            0.55
        );
        let restored_bindings = load_expression_bindings(&path).expect("bindings reload");
        assert_eq!(
            restored_bindings
                .bindings_for(&first.0)
                .expect("model entry")
                .expression_for(ExpressionKey::KeyA),
            Some("smile")
        );
    }

    #[test]
    fn expression_settings_keep_other_models_and_sections() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        save_language(&path, UiLanguage::Zh).expect("language save");
        let mut store = ExpressionBindingStore::default();
        store.set("a".into(), bindings(&[(ExpressionKey::Digit1, "happy")]));
        store.set("b".into(), bindings(&[(ExpressionKey::Digit1, "angry")]));
        save_expression_bindings(&path, &store).expect("save");

        let restored = load_expression_bindings(&path).expect("reload");
        assert_eq!(
            restored
                .bindings_for("a")
                .expect("model a")
                .expression_for(ExpressionKey::Digit1),
            Some("happy")
        );
        assert_eq!(
            restored
                .bindings_for("b")
                .expect("model b")
                .expression_for(ExpressionKey::Digit1),
            Some("angry")
        );
        assert_eq!(load_language(&path), UiLanguage::Zh);
    }

    #[test]
    fn old_settings_without_expression_section_load_defaults() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        fs::write(&path, "schema_version = 1\nlanguage = \"ja\"\n").unwrap();
        let restored = load_expression_bindings(&path).expect("old settings parse");
        assert!(restored.is_empty());
    }

    #[test]
    fn duplicate_expression_assignments_are_rejected() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        fs::write(
            &path,
            "schema_version = 1\n[expression_bindings.\"model\".keys]\nDigit1 = \"happy\"\nDigit2 = \"happy\"\n",
        )
        .unwrap();
        assert!(matches!(
            load_expression_bindings(&path),
            Err(ArmPoseSettingsError::InvalidExpressionBinding { .. })
        ));
    }

    #[test]
    fn unknown_expression_key_is_a_typed_error() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        fs::write(
            &path,
            "schema_version = 1\n[expression_bindings.\"model\".keys]\nNumpad1 = \"happy\"\n",
        )
        .unwrap();
        assert!(load_expression_bindings(&path).is_err());
    }

    #[test]
    fn unknown_malformed_and_invalid_values_fall_back_to_empty_defaults() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        fs::write(&path, "schema_version = 99\n").unwrap();
        assert!(load_arm_pose_overrides(&path).is_err());
        fs::write(&path, "this is not valid TOML = [").unwrap();
        assert!(load_arm_pose_overrides(&path).is_err());
        fs::write(&path, "schema_version = 1\n[arm_pose_overrides.bad]\nschema_version = 1\narm_drop_radians = 999\nreach_ratio = 0.99\nforward_hand_offset_ratio = 0.081\nelbow_pole_offset_ratio = 0.05\nshoulder_follow_weight = 0.18\nfinger_curl_radians = 0.17\n").unwrap();
        assert!(load_arm_pose_overrides(&path).is_err());
        fs::write(&path, "schema_version = 1\n[arm_pose_overrides.bad]\nschema_version = 1\narm_drop_radians = nan\nreach_ratio = 0.99\nforward_hand_offset_ratio = 0.081\nelbow_pole_offset_ratio = 0.05\nshoulder_follow_weight = 0.18\nfinger_curl_radians = 0.17\n").unwrap();
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
        assert!(
            (reloaded
                .dynamic_profile_for(&first)
                .unwrap()
                .shoulder_elevation_trim_radians
                - -0.13)
                .abs()
                < 1e-6
        );
        assert!(
            (reloaded
                .dynamic_profile_for(&second)
                .unwrap()
                .shoulder_elevation_trim_radians
                - 0.05)
                .abs()
                < 1e-6
        );
        assert!(store.reset_dynamic_profile(&first));
        assert!(store.dynamic_profile_for(&first).is_none());
        assert!(store.dynamic_profile_for(&second).is_some());
    }

    #[test]
    fn corrupt_or_old_version_dynamic_entries_fall_back_to_defaults_without_panicking() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        fs::write(&path, format!("schema_version = {ARM_POSE_SETTINGS_SCHEMA_VERSION}\n[dynamic_arm_profiles.\"sha256:bad\"]\nschema_version = 1\nhand_anchor_ratio = [0.0, 0.0, 0.0]\ncompensation_gains = [2.5, 0.0, 0.0]\nelbow_swivel_radians = 99.0\nswivel_transition_width_ratio = 0.15\npole_influence = 0.2\ntwist_relax_weight = 0.7\ntwist_parent_child_crossfade = 0.9\nshoulder_elevation_trim_radians = 9.9\n")).expect("write corrupt settings");
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
        assert_eq!(
            migrated.into_profile().unwrap(),
            DynamicArmProfile::default()
        );
    }
}
