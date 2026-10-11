#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
use super::*;

#[test]
fn absent_calibration_is_a_file_error_and_unloaded_avatar_has_no_pose() {
    let directory = tempfile::tempdir().unwrap();
    assert!(matches!(
        read_calibration(&directory.path().join("absent.json")),
        Err(LookingGlassError::Read { .. })
    ));
    assert!(
        avatar_view_pose(&AvatarCameraControl::default(), &AvatarLifecycle::default()).is_none()
    );
}

#[test]
fn absent_option_does_not_request_output() {
    assert!(
        config_argument(["--model", "モデル.vrm"].map(OsString::from))
            .unwrap()
            .is_none()
    );
    let control = LookingGlassControl::default();
    assert!(control.pending.is_none());
    assert!(control.startup.is_none());
}

#[test]
fn explicit_option_requires_a_path_and_preserves_unicode() {
    assert!(matches!(
        config_argument([OsString::from("--looking-glass")]),
        Err(LookingGlassError::MissingPath)
    ));
    assert_eq!(
        config_argument(["--model", "a.vrm", "--looking-glass", "設定.json"].map(OsString::from))
            .unwrap(),
        Some(PathBuf::from("設定.json"))
    );
}

#[test]
fn output_selection_round_trips_without_overwriting_other_settings() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("settings.toml");
    let mut settings = crate::settings::AppSettings::empty_at(&path);
    settings
        .set_language(crate::settings::UiLanguage::En)
        .unwrap();
    let selection = OutputPreferences {
        device_id: "portrait".into(),
        calibration_path: directory.path().join("実機/visual.json"),
    };
    settings.save_looking_glass(selection.clone()).unwrap();
    let restored = crate::settings::AppSettings::load(path).unwrap();
    assert_eq!(restored.looking_glass(), &selection);
    assert_eq!(restored.language(), crate::settings::UiLanguage::En);
    // Only selection is persisted, never an enabled flag.
    assert!(LookingGlassControl::default().pending.is_none());
}
