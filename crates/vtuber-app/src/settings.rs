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
use vtuber_avatar::{
    ArmPoseOverrideStore, ArmPoseProfileOverride, DynamicArmProfileOverride, RichLookSettings,
};

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
    /// Model whose look was restored, including CLI startup loads.
    pub(crate) look_model_id: Option<String>,
    restored_expression_bindings: ExpressionBindingStore,
    language: UiLanguage,
    arm_tracking_enabled: bool,
}

impl Default for ArmPoseSettings {
    fn default() -> Self {
        Self {
            path: default_settings_path(),
            restored: ArmPoseOverrideStore::default(),
            look_model_id: None,
            restored_expression_bindings: ExpressionBindingStore::default(),
            language: UiLanguage::default(),
            arm_tracking_enabled: false,
        }
    }
}

impl ArmPoseSettings {
    /// Loads settings from an explicit path, reading the document once.
    ///
    /// A missing file yields the initial values. Every other I/O failure,
    /// malformed TOML, or unknown schema version is an error; the file is left
    /// untouched and the caller reports it instead of starting from replaced
    /// settings.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, ArmPoseSettingsError> {
        let path = path.into();
        let document = read_settings_document(&path)?;
        restore_settings_document(document, path)
    }

    /// Loads settings from the platform user configuration directory.
    pub fn load_default() -> Result<Self, ArmPoseSettingsError> {
        match default_settings_path() {
            Some(path) => Self::load(path),
            None => Ok(Self::default()),
        }
    }

    /// Creates an empty settings resource targeting an explicit path.
    #[must_use]
    pub fn empty_at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            restored: ArmPoseOverrideStore::default(),
            look_model_id: None,
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

    /// Returns the selected model's look; a model without an entry starts OFF.
    pub(crate) fn rich_look_for(
        &self,
        model_id: &str,
    ) -> Result<RichLookSettings, ArmPoseSettingsError> {
        let path = self
            .path
            .as_deref()
            .ok_or(ArmPoseSettingsError::NoConfigDirectory)?;
        Ok(load_rich_look_settings(path)?
            .get(model_id)
            .copied()
            .unwrap_or_default())
    }

    /// Saves the selected model's look, preserving every other settings
    /// section, including the model's arm-pose and tracking entries.
    pub(crate) fn save_rich_look(
        &self,
        model_id: String,
        settings: RichLookSettings,
    ) -> Result<(), ArmPoseSettingsError> {
        let path = self
            .path
            .as_deref()
            .ok_or(ArmPoseSettingsError::NoConfigDirectory)?;
        save_rich_look_settings(path, model_id, settings)
    }

    /// Returns the UI language loaded at startup.
    #[must_use]
    pub fn language(&self) -> UiLanguage {
        self.language
    }

    /// Persists the UI language and applies it to this resource.
    ///
    /// The in-memory value changes only after the file was written.
    pub fn set_language(&mut self, language: UiLanguage) -> Result<(), ArmPoseSettingsError> {
        let path = self
            .path
            .as_deref()
            .ok_or(ArmPoseSettingsError::NoConfigDirectory)?;
        save_language(path, language)?;
        self.language = language;
        Ok(())
    }

    /// Whether observed webcam arm tracking was enabled at load time.
    #[must_use]
    pub fn arm_tracking_enabled(&self) -> bool {
        self.arm_tracking_enabled
    }

    /// Persists the observed arm-tracking switch.
    ///
    /// The in-memory value changes only after the file was written.
    pub fn set_arm_tracking_enabled(&mut self, enabled: bool) -> Result<(), ArmPoseSettingsError> {
        let path = self
            .path
            .as_deref()
            .ok_or(ArmPoseSettingsError::NoConfigDirectory)?;
        save_arm_tracking_enabled(path, enabled)?;
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

/// File name of the optional validated eye-closure profile.
pub const EYE_CLOSURE_PROFILE_FILE_NAME: &str = "eye_closure_profile.json";

/// Returns the eye-closure profile path next to the application settings.
///
/// The profile is optional: a missing file means "no correction", never a
/// fabricated default threshold.
#[must_use]
pub fn default_eye_closure_profile_path() -> Option<PathBuf> {
    ProjectDirs::from("", "", "RusTuberV")
        .map(|dirs| dirs.config_dir().join(EYE_CLOSURE_PROFILE_FILE_NAME))
}

/// Loads and validates an eye-closure profile from an explicit path.
///
/// Only a missing file is "no correction": a directory, an unreadable file, or
/// invalid contents is an error.
///
/// # Errors
///
/// Returns a message for read/parse failures, a foreign feature identity, an
/// inference fingerprint that does not match the current runtime, or invalid
/// thresholds. The caller must not substitute a default profile on error.
pub fn load_eye_closure_thresholds(
    path: &Path,
) -> Result<Option<vtuber_tracking::EyeGeometryThresholds>, String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("failed to read {}: {error}", path.display())),
    };
    let document: vtuber_tracking::EyeClosureProfileDocument = serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    let current = vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256;
    if document.fingerprints.feature != vtuber_tracking::EYE_CLOSURE_FEATURE
        || !document
            .fingerprints
            .task_bundle_sha256
            .as_deref()
            .is_some_and(|hash| hash.eq_ignore_ascii_case(current))
    {
        return Err(format!(
            "{}: inference fingerprints do not match the current runtime",
            path.display()
        ));
    }
    document
        .validate()
        .map(Some)
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// Loads the stored UI language.
///
/// A missing file is the initial language; unreadable, malformed, and
/// unknown-version documents are errors.
///
/// # Errors
///
/// Propagates every settings read, parse, and schema failure.
pub fn load_language(path: &Path) -> Result<UiLanguage, ArmPoseSettingsError> {
    Ok(read_settings_document(path)?.language)
}

/// Persists UI language, preserving the rest of the settings document.
///
/// # Errors
///
/// Propagates every read, schema, encode, and write failure; the previous
/// bytes are kept.
pub fn save_language(path: &Path, language: UiLanguage) -> Result<(), ArmPoseSettingsError> {
    let mut document = read_settings_document(path)?;
    document.language = language;
    write_settings_atomically(path, &toml::to_string_pretty(&document)?)
}

/// Loads the stored arm-tracking switch.
///
/// # Errors
///
/// Propagates every settings read, parse, and schema failure.
pub fn load_arm_tracking_enabled(path: &Path) -> Result<bool, ArmPoseSettingsError> {
    Ok(read_settings_document(path)?.arm_tracking_enabled)
}

/// Persists the observed arm-tracking switch, preserving other sections.
///
/// # Errors
///
/// Propagates every read, schema, encode, and write failure; the previous
/// bytes are kept.
pub fn save_arm_tracking_enabled(path: &Path, enabled: bool) -> Result<(), ArmPoseSettingsError> {
    let mut document = read_settings_document(path)?;
    document.arm_tracking_enabled = enabled;
    write_settings_atomically(path, &toml::to_string_pretty(&document)?)
}

/// Loads and validates model-specific expression bindings.
///
/// # Errors
///
/// Propagates every read, parse, schema, and duplicate-assignment failure.
pub fn load_expression_bindings(
    path: &Path,
) -> Result<ExpressionBindingStore, ArmPoseSettingsError> {
    expression_store(read_settings_document(path)?.expression_bindings)
}

/// Saves expression bindings, preserving language, arm pose, and every other
/// model's assignments.
///
/// # Errors
///
/// Propagates every read, schema, encode, and write failure; the previous
/// bytes are kept.
pub fn save_expression_bindings(
    path: &Path,
    store: &ExpressionBindingStore,
) -> Result<(), ArmPoseSettingsError> {
    let mut document = read_settings_document(path)?;
    document.expression_bindings = store
        .entries()
        .map(|(model_id, bindings)| (model_id.to_owned(), bindings.clone()))
        .collect();
    write_settings_atomically(path, &toml::to_string_pretty(&document)?)
}

/// Loads and validates the arm-pose settings document.
///
/// # Errors
///
/// Propagates every read, parse, and schema failure, and rejects a document
/// whose arm-pose entries cannot all be validated.
pub fn load_arm_pose_overrides(path: &Path) -> Result<ArmPoseOverrideStore, ArmPoseSettingsError> {
    let document = read_settings_document(path)?;
    arm_store(document.arm_pose_overrides, document.dynamic_arm_profiles)
}

/// Saves validated entries using the existing settings-file replacement policy.
///
/// # Errors
///
/// Propagates every read, schema, encode, and write failure; the previous
/// bytes are kept.
pub fn save_arm_pose_overrides(
    path: &Path,
    store: &ArmPoseOverrideStore,
) -> Result<(), ArmPoseSettingsError> {
    let mut document = read_settings_document(path)?;
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

fn load_rich_look_settings(
    path: &Path,
) -> Result<BTreeMap<String, RichLookSettings>, ArmPoseSettingsError> {
    Ok(read_settings_document(path)?.rich_look)
}

fn save_rich_look_settings(
    path: &Path,
    model_id: String,
    settings: RichLookSettings,
) -> Result<(), ArmPoseSettingsError> {
    let mut document = read_settings_document(path)?;
    document.rich_look.insert(model_id, settings);
    write_settings_atomically(path, &toml::to_string_pretty(&document)?)
}

/// Builds the startup resource from one already-validated document.
fn restore_settings_document(
    document: ArmPoseSettingsDocument,
    path: PathBuf,
) -> Result<ArmPoseSettings, ArmPoseSettingsError> {
    let language = document.language;
    let arm_tracking_enabled = document.arm_tracking_enabled;
    let restored_expression_bindings = expression_store(document.expression_bindings)?;
    let restored = arm_store(document.arm_pose_overrides, document.dynamic_arm_profiles)?;
    Ok(ArmPoseSettings {
        path: Some(path),
        restored,
        look_model_id: None,
        restored_expression_bindings,
        language,
        arm_tracking_enabled,
    })
}

/// Returns the document that represents "nothing saved yet".
fn empty_settings_document() -> ArmPoseSettingsDocument {
    ArmPoseSettingsDocument {
        schema_version: ARM_POSE_SETTINGS_SCHEMA_VERSION,
        language: UiLanguage::default(),
        arm_tracking_enabled: false,
        arm_pose_overrides: BTreeMap::new(),
        dynamic_arm_profiles: BTreeMap::new(),
        expression_bindings: BTreeMap::new(),
        rich_look: BTreeMap::new(),
    }
}

/// Parses a settings document and accepts only the current schema version.
///
/// An unknown version is refused instead of being rewritten, so no writer can
/// save a newer document in the older format.
fn parse_settings_document(text: &str) -> Result<ArmPoseSettingsDocument, ArmPoseSettingsError> {
    let document: ArmPoseSettingsDocument = toml::from_str(text)?;
    if document.schema_version != ARM_POSE_SETTINGS_SCHEMA_VERSION {
        return Err(ArmPoseSettingsError::UnsupportedSchema {
            version: document.schema_version,
        });
    }
    Ok(document)
}

/// Reads and validates the settings document in a single read.
///
/// Only a missing file becomes the initial document; every other I/O failure
/// and every unknown schema version is an error.
fn read_settings_document(path: &Path) -> Result<ArmPoseSettingsDocument, ArmPoseSettingsError> {
    match fs::read_to_string(path) {
        Ok(text) => parse_settings_document(&text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(empty_settings_document()),
        Err(error) => Err(error.into()),
    }
}

/// Validates persisted expression assignments, rejecting duplicates.
fn expression_store(
    bindings: BTreeMap<String, ExpressionBindings>,
) -> Result<ExpressionBindingStore, ArmPoseSettingsError> {
    for (model_id, bindings) in &bindings {
        if bindings.has_duplicate_expressions() {
            return Err(ArmPoseSettingsError::InvalidExpressionBinding {
                model_id: model_id.clone(),
            });
        }
    }
    let mut store = ExpressionBindingStore::default();
    store.replace_entries(bindings);
    Ok(store)
}

/// Validates persisted arm-pose entries and their dynamic profiles.
fn arm_store(
    arm_pose_overrides: BTreeMap<String, PersistedArmPoseProfile>,
    dynamic_arm_profiles: BTreeMap<String, PersistedDynamicArmProfile>,
) -> Result<ArmPoseOverrideStore, ArmPoseSettingsError> {
    let expected = arm_pose_overrides.len();
    let mut store = ArmPoseOverrideStore::default();
    let accepted = store.import_entries(
        arm_pose_overrides
            .into_iter()
            .map(|(model_id, profile)| (model_id, profile.into_runtime())),
    );
    if accepted != expected {
        return Err(ArmPoseSettingsError::InvalidEntry);
    }
    // Existing settings policy: invalid dynamic entries are ignored by the store.
    store.import_dynamic_entries(
        dynamic_arm_profiles
            .into_iter()
            .map(|(model_id, profile)| (model_id, profile.into_runtime())),
    );
    Ok(store)
}

/// Writes the complete settings text by replacing the target with a unique
/// temporary file from the same directory.
///
/// The existing file is never deleted up front: a failed write or replacement
/// leaves the previous bytes in place, and unique temporary names keep
/// concurrent saves from clobbering each other's scratch file.
fn write_settings_atomically(path: &Path, text: &str) -> Result<(), ArmPoseSettingsError> {
    use std::io::Write as _;
    use tempfile::NamedTempFile;

    // A bare file name has no meaningful parent; the save directory is then
    // the current directory itself.
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    fs::create_dir_all(directory)?;
    let mut temporary = NamedTempFile::new_in(directory)?;
    temporary.write_all(text.as_bytes())?;
    temporary
        .persist(path)
        .map_err(|persist_error| ArmPoseSettingsError::Io(persist_error.error))?;
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
    /// Model-specific look switch and strength. A missing entry means the
    /// model starts OFF at full strength.
    #[serde(default)]
    rich_look: BTreeMap<String, RichLookSettings>,
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

    #[test]
    fn atomically_replaces_existing_file_with_the_second_content() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        write_settings_atomically(&path, "schema_version = 1\n").expect("first save");
        write_settings_atomically(&path, "schema_version = 1\nlanguage = \"en\"\n")
            .expect("second save");
        assert_eq!(
            fs::read_to_string(&path).expect("settings readable"),
            "schema_version = 1\nlanguage = \"en\"\n"
        );
    }

    #[test]
    fn failed_save_keeps_the_existing_bytes() {
        let directory = tempdir().expect("temporary settings directory");
        let original = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        fs::write(&original, "schema_version = 1\n").unwrap();
        // `original` is a file, so a path nested under it cannot resolve a
        // save directory and the write must fail without touching it.
        let nested = original.join(ARM_POSE_SETTINGS_FILE_NAME);
        assert!(write_settings_atomically(&nested, "schema_version = 2\n").is_err());
        assert_eq!(
            fs::read_to_string(&original).expect("settings readable"),
            "schema_version = 1\n"
        );
    }

    fn profile(drop: f32) -> ArmPoseProfileOverride {
        ArmPoseProfileOverride::from_profile(ArmPoseProfile {
            arm_drop_radians: drop,
            ..Default::default()
        })
    }

    #[test]
    fn rich_look_and_every_other_writer_preserve_each_other() {
        let directory = tempdir().unwrap();
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        let id = AvatarAssetId::new("sha256:first");
        let zero = RichLookSettings::try_new(true, 0.0).unwrap();
        let half = RichLookSettings::try_new(false, 0.5).unwrap();
        let mut arms = ArmPoseOverrideStore::default();
        arms.set(id.0.clone(), profile(0.55)).unwrap();
        arms.set_dynamic_profile(
            id.0.clone(),
            DynamicArmProfileOverride::from_profile(vtuber_avatar::DynamicArmProfile {
                shoulder_elevation_trim_radians: -0.1,
                ..Default::default()
            }),
        )
        .unwrap();
        let mut expressions = ExpressionBindingStore::default();
        expressions.set(id.0.clone(), bindings(&[(ExpressionKey::KeyA, "smile")]));
        save_language(&path, UiLanguage::Ko).unwrap();
        save_arm_pose_overrides(&path, &arms).unwrap();
        save_arm_tracking_enabled(&path, true).unwrap();
        save_expression_bindings(&path, &expressions).unwrap();
        let before: toml::Value = toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        save_rich_look_settings(&path, id.0.clone(), zero).unwrap();
        save_rich_look_settings(&path, "sha256:second".into(), half).unwrap();
        let mut after: toml::Value = toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        after.as_table_mut().unwrap().remove("rich_look");
        let mut before = before;
        before.as_table_mut().unwrap().remove("rich_look");
        assert_eq!(after, before);
        let expected = load_rich_look_settings(&path).unwrap();
        save_language(&path, UiLanguage::En).unwrap();
        assert_eq!(load_rich_look_settings(&path).unwrap(), expected);
        save_arm_pose_overrides(&path, &arms).unwrap();
        assert_eq!(load_rich_look_settings(&path).unwrap(), expected);
        save_arm_tracking_enabled(&path, false).unwrap();
        assert_eq!(load_rich_look_settings(&path).unwrap(), expected);
        save_expression_bindings(&path, &expressions).unwrap();
        assert_eq!(load_rich_look_settings(&path).unwrap(), expected);
        let restarted = ArmPoseSettings::load(&path).unwrap();
        assert_eq!(restarted.rich_look_for(&id.0).unwrap(), zero);
        assert_eq!(restarted.rich_look_for("sha256:second").unwrap(), half);
        assert_eq!(
            restarted.rich_look_for("sha256:new").unwrap(),
            RichLookSettings::default()
        );
    }

    #[test]
    fn rich_look_missing_section_defaults_but_invalid_files_return_errors() {
        let directory = tempdir().unwrap();
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        assert!(load_rich_look_settings(&path).unwrap().is_empty());
        fs::write(&path, "schema_version = 1\nlanguage = 'en'\n").unwrap();
        assert!(load_rich_look_settings(&path).unwrap().is_empty());
        for invalid in [
            "broken = [",
            "schema_version = 99",
            "schema_version = 1\n[rich_look.model]\nenabled = true\nstrength = 'bad'",
        ] {
            fs::write(&path, invalid).unwrap();
            assert!(load_rich_look_settings(&path).is_err());
            assert!(
                save_rich_look_settings(&path, "model".into(), RichLookSettings::default())
                    .is_err()
            );
            assert_eq!(fs::read_to_string(&path).unwrap(), invalid);
        }
        assert!(load_rich_look_settings(directory.path()).is_err());
    }

    #[test]
    fn invalid_rich_strength_prevents_load_and_save_without_replacing_bytes() {
        let directory = tempdir().unwrap();
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        for strength in ["-0.1", "1.1", "nan", "inf", "-inf"] {
            let text = format!(
                "schema_version = 1\n[rich_look.model]\nenabled = false\nstrength = {strength}\n"
            );
            fs::write(&path, &text).unwrap();
            assert!(ArmPoseSettings::load(&path).is_err());
            assert!(load_rich_look_settings(&path).is_err());
            assert!(save_language(&path, UiLanguage::En).is_err());
            assert_eq!(fs::read_to_string(&path).unwrap(), text);
        }
    }

    #[test]
    fn a_stale_material_roles_section_is_ignored() {
        let directory = tempdir().unwrap();
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        fs::write(
            &path,
            "schema_version = 1\n\
             [[material_roles.\"sha256:first\"]]\n\
             material_index = 0\n\
             selected = 'face'\n\
             [rich_look.\"sha256:first\"]\n\
             enabled = true\n\
             strength = 0.5\n",
        )
        .unwrap();
        let restarted = ArmPoseSettings::load(&path).unwrap();
        assert_eq!(
            restarted.rich_look_for("sha256:first").unwrap(),
            RichLookSettings::try_new(true, 0.5).unwrap()
        );
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
        assert_eq!(load_language(&path).unwrap(), UiLanguage::Ja);
        for language in [
            UiLanguage::Ja,
            UiLanguage::En,
            UiLanguage::Zh,
            UiLanguage::Ko,
        ] {
            save_language(&path, language).expect("language save");
            assert_eq!(load_language(&path).unwrap(), language);
            save_arm_pose_overrides(&path, &ArmPoseOverrideStore::default())
                .expect("settings save");
            assert_eq!(load_language(&path).unwrap(), language);
            assert_eq!(ArmPoseSettings::load(&path).unwrap().language(), language);
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

        let restored = ArmPoseSettings::load(&path).unwrap();
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
        assert_eq!(load_language(&path).unwrap(), UiLanguage::Zh);
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
    fn unknown_schema_makes_every_writer_refuse_and_keep_the_bytes() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        let foreign = "schema_version = 99\nlanguage = 'en'\n";
        fs::write(&path, foreign).unwrap();
        let mut arms = ArmPoseOverrideStore::default();
        arms.set(AvatarAssetId::new("sha256:first").0.clone(), profile(0.55))
            .unwrap();
        let mut expressions = ExpressionBindingStore::default();
        expressions.set(
            "sha256:first".into(),
            bindings(&[(ExpressionKey::KeyA, "smile")]),
        );
        for result in [
            save_language(&path, UiLanguage::Ko),
            save_arm_tracking_enabled(&path, true),
            save_arm_pose_overrides(&path, &arms),
            save_expression_bindings(&path, &expressions),
            save_rich_look_settings(&path, "sha256:first".into(), RichLookSettings::default()),
        ] {
            assert!(matches!(
                result,
                Err(ArmPoseSettingsError::UnsupportedSchema { version: 99 })
            ));
        }
        assert_eq!(fs::read_to_string(&path).unwrap(), foreign);
        assert!(ArmPoseSettings::load(&path).is_err());
        assert!(!path.with_extension("toml.invalid").exists());
    }

    #[test]
    fn unknown_malformed_and_invalid_values_are_errors_that_keep_the_file() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        for invalid in [
            "schema_version = 99\n",
            "this is not valid TOML = [",
            "schema_version = 1\n[arm_pose_overrides.bad]\nschema_version = 1\narm_drop_radians = 999\nreach_ratio = 0.99\nforward_hand_offset_ratio = 0.081\nelbow_pole_offset_ratio = 0.05\nshoulder_follow_weight = 0.18\nfinger_curl_radians = 0.17\n",
            "schema_version = 1\n[arm_pose_overrides.bad]\nschema_version = 1\narm_drop_radians = nan\nreach_ratio = 0.99\nforward_hand_offset_ratio = 0.081\nelbow_pole_offset_ratio = 0.05\nshoulder_follow_weight = 0.18\nfinger_curl_radians = 0.17\n",
        ] {
            fs::write(&path, invalid).unwrap();
            assert!(load_arm_pose_overrides(&path).is_err());
            assert!(ArmPoseSettings::load(&path).is_err());
            assert_eq!(fs::read_to_string(&path).unwrap(), invalid);
            assert!(!path.with_extension("toml.invalid").exists());
        }
        assert!(load_arm_pose_overrides(directory.path()).is_err());
    }

    #[test]
    fn a_missing_document_starts_from_the_initial_values() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        let loaded = ArmPoseSettings::load(&path).expect("missing settings are initial values");
        assert_eq!(loaded.language(), UiLanguage::default());
        assert!(!loaded.arm_tracking_enabled());
        assert_eq!(loaded.restored_entries().count(), 0);
        assert!(loaded.restored_expression_bindings().is_empty());
        assert_eq!(load_language(&path).unwrap(), UiLanguage::default());
        assert!(!load_arm_tracking_enabled(&path).unwrap());
    }

    #[test]
    fn a_set_without_a_writable_destination_keeps_the_memory_value() {
        let directory = tempdir().expect("temporary settings directory");
        let blocked = directory.path().join(ARM_POSE_SETTINGS_FILE_NAME);
        fs::write(&blocked, "schema_version = 1\n").unwrap();
        // A path nested under an existing file has no save directory, so the
        // read reports a missing document and the write then fails.
        let mut unwritable = ArmPoseSettings::empty_at(blocked.join(ARM_POSE_SETTINGS_FILE_NAME));
        assert!(unwritable.set_language(UiLanguage::Ko).is_err());
        assert!(unwritable.set_arm_tracking_enabled(true).is_err());
        assert_eq!(unwritable.language(), UiLanguage::Ja);
        assert!(!unwritable.arm_tracking_enabled());
        assert_eq!(
            fs::read_to_string(&blocked).unwrap(),
            "schema_version = 1\n"
        );

        let mut without_directory = ArmPoseSettings {
            path: None,
            ..ArmPoseSettings::default()
        };
        assert!(matches!(
            without_directory.set_language(UiLanguage::Ko),
            Err(ArmPoseSettingsError::NoConfigDirectory)
        ));
        assert!(matches!(
            without_directory.set_arm_tracking_enabled(true),
            Err(ArmPoseSettingsError::NoConfigDirectory)
        ));
        assert_eq!(without_directory.language(), UiLanguage::Ja);
        assert!(!without_directory.arm_tracking_enabled());
    }

    fn eye_closure_document() -> vtuber_tracking::EyeClosureProfileDocument {
        vtuber_tracking::EyeClosureProfileDocument {
            schema_version: vtuber_tracking::EYE_CLOSURE_PROFILE_SCHEMA_VERSION,
            algorithm_version: vtuber_tracking::EYE_CLOSURE_ALGORITHM_VERSION,
            feature: vtuber_tracking::EYE_CLOSURE_FEATURE.into(),
            status: vtuber_tracking::EyeClosureVerificationStatus::Verified,
            left: vtuber_tracking::EyeGeometryThresholdValues {
                close_gap: 0.2,
                reopen_gap: 0.5,
                min_blink: 0.0,
            },
            right: vtuber_tracking::EyeGeometryThresholdValues {
                close_gap: 0.2,
                reopen_gap: 0.5,
                min_blink: 0.0,
            },
            fingerprints: vtuber_tracking::EyeClosureFingerprints {
                task_bundle_sha256: Some(
                    vtuber_inference::backend::mediapipe::TASK_BUNDLE_SHA256.into(),
                ),
                feature: vtuber_tracking::EYE_CLOSURE_FEATURE.into(),
                preprocess: None,
            },
            applies_to: None,
        }
    }

    #[test]
    fn eye_closure_profile_loads_only_with_valid_matching_data() {
        let directory = tempdir().expect("temporary settings directory");
        let path = directory.path().join(EYE_CLOSURE_PROFILE_FILE_NAME);
        assert!(load_eye_closure_thresholds(&path).unwrap().is_none());
        assert!(load_eye_closure_thresholds(directory.path()).is_err());

        let document = eye_closure_document();
        fs::write(&path, serde_json::to_string(&document).unwrap()).unwrap();
        let thresholds = load_eye_closure_thresholds(&path)
            .expect("valid profile")
            .expect("some thresholds");
        assert_eq!(thresholds.left().close_gap(), 0.2);

        let mut foreign_fingerprint = eye_closure_document();
        foreign_fingerprint.fingerprints.task_bundle_sha256 = Some("0000".into());
        fs::write(&path, serde_json::to_string(&foreign_fingerprint).unwrap()).unwrap();
        assert!(load_eye_closure_thresholds(&path).is_err());

        let mut invalid = eye_closure_document();
        invalid.left.close_gap = 0.9;
        invalid.left.reopen_gap = 0.1;
        fs::write(&path, serde_json::to_string(&invalid).unwrap()).unwrap();
        assert!(load_eye_closure_thresholds(&path).is_err());
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
