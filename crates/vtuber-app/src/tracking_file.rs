//! File boundary for the user-tunable tracking profile.
//!
//! Loads `tracking_profile.toml`, applies it onto the built-in defaults, and
//! generates a fully populated starting template when the file does not
//! exist. Invalid or unsupported documents are returned to the startup
//! boundary as typed errors; the file itself is never rewritten, so user
//! edits are always preserved.

use std::fs;
use std::path::Path;

use vtuber_avatar::{
    DynamicArmProfile, GlobalBodyTrackingProfile, TRACKING_PROFILE_SCHEMA_VERSION,
    TrackingProfileDocument,
};

/// File name of the tracking profile document.
pub const TRACKING_PROFILE_FILE_NAME: &str = "tracking_profile.toml";

/// Human-readable header prepended to the generated template.
const TEMPLATE_HEADER: &str = r#"## 追従（トラッキング）配分チューニングファイル
## 反映にはアプリの再起動が必要です。
## - 角度の単位はすべて「度」、half_life（半減期）の単位は「秒」です。
## - 値を削除/省略した項目はビルトインの既定値になります。
## - weights（配分）は合計が1になるように自動的に正規化されます。
## - セクションや項目名のタイプミスがある場合、アプリは起動せずエラーを表示します。
"#;

/// Tracking profiles resolved from the document.
#[derive(Debug, Clone)]
pub struct TrackingProfileValues {
    /// Body rotation distribution applied to every bound avatar.
    pub body: GlobalBodyTrackingProfile,
    /// Arm hand-target follow profile.
    pub arm: DynamicArmProfile,
}

/// Loads the tracking profile document from `path`.
///
/// A missing file is seeded with the generated template before its values are
/// applied, so the next edit has every knob visible.
pub fn load_tracking_profile(
    path: &Path,
) -> Result<TrackingProfileValues, TrackingProfileFileError> {
    let document = if path.is_file() {
        load_document(path)?
    } else {
        let document = TrackingProfileDocument::template();
        write_template(path, &document)?;
        document
    };
    let mut values = TrackingProfileValues {
        body: GlobalBodyTrackingProfile::default(),
        arm: DynamicArmProfile::default(),
    };
    document.apply_body_to(&mut values.body.0);
    document.apply_arm_to(&mut values.arm);
    bevy::log::info!("tracking profile loaded from {}", path.display());
    Ok(values)
}

fn load_document(path: &Path) -> Result<TrackingProfileDocument, TrackingProfileFileError> {
    let text = fs::read_to_string(path)?;
    let document: TrackingProfileDocument = toml::from_str(&text)?;
    if let Some(version) = document.schema_version
        && version != TRACKING_PROFILE_SCHEMA_VERSION
    {
        return Err(TrackingProfileFileError::UnsupportedSchema { version });
    }
    Ok(document)
}

fn write_template(
    path: &Path,
    document: &TrackingProfileDocument,
) -> Result<(), TrackingProfileFileError> {
    let body = toml::to_string_pretty(document)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{TEMPLATE_HEADER}\n{body}"))?;
    bevy::log::info!("tracking profile template generated at {}", path.display());
    Ok(())
}

/// Errors surfaced while reading the tracking profile document.
#[derive(Debug, thiserror::Error)]
pub enum TrackingProfileFileError {
    /// File-system read failure.
    #[error("tracking profile I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// TOML decoding failure.
    #[error("tracking profile TOML is malformed: {0}")]
    Decode(#[from] toml::de::Error),
    /// TOML encoding failure while creating the initial template.
    #[error("tracking profile TOML could not be generated: {0}")]
    Encode(#[from] toml::ser::Error),
    /// The document schema version is not supported by this build.
    #[error("unsupported tracking profile schema version {version}")]
    UnsupportedSchema {
        /// Encountered schema version.
        version: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_profile_is_generated_and_loaded() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join(TRACKING_PROFILE_FILE_NAME);

        let values = load_tracking_profile(&path).unwrap();

        assert!(path.is_file());
        assert_eq!(values.body, GlobalBodyTrackingProfile::default());
        assert_eq!(values.arm, DynamicArmProfile::default());
    }

    #[test]
    fn malformed_profile_is_a_typed_startup_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join(TRACKING_PROFILE_FILE_NAME);
        fs::write(&path, "[body.small_yaw\nhead = 0.5").unwrap();

        assert!(matches!(
            load_tracking_profile(&path),
            Err(TrackingProfileFileError::Decode(_))
        ));
    }

    #[test]
    fn unsupported_schema_is_a_typed_startup_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join(TRACKING_PROFILE_FILE_NAME);
        fs::write(&path, "schema_version = 2\n").unwrap();

        assert!(matches!(
            load_tracking_profile(&path),
            Err(TrackingProfileFileError::UnsupportedSchema { version: 2 })
        ));
    }
}
