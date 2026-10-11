#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
use super::*;

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
        device_id: "TEST-UNIT-SERIAL".into(),
    };
    settings.save_looking_glass(selection.clone()).unwrap();
    let restored = crate::settings::AppSettings::load(path).unwrap();
    assert_eq!(restored.looking_glass(), &selection);
    assert_eq!(restored.language(), crate::settings::UiLanguage::En);
    // Only selection is persisted, never an enabled flag.
    assert!(LookingGlassControl::default().pending.is_none());
}

fn bridge_display(serial: &str, width: u32, height: u32, mode: u32) -> Display {
    let raw = serde_json::json!({
        "serial": serial,
        "pitch": {"value": 52.0}, "slope": {"value": -7.0}, "center": {"value": 0.5},
        "DPI": {"value": 300.0}, "screenW": {"value": width}, "screenH": {"value": height},
        "viewCone": {"value": 40.0}, "invView": {"value": 1},
        "flipImageX": {"value": 0}, "flipImageY": {"value": 0}, "flipSubp": {"value": 0},
        "CellPatternMode": {"value": mode},
        "subpixelCells": [{"ROffsetX": 0.0, "ROffsetY": 0.0, "GOffsetX": 0.5, "GOffsetY": 0.0, "BOffsetX": 1.0, "BOffsetY": 0.0}]
    });
    Display {
        serial: serial.into(),
        name: "TEST DISPLAY".into(),
        hardware_version: "MODEL-NOT-IN-CATALOG".into(),
        calibration: raw.to_string(),
        quilt: Some(bridge::Quilt {
            columns: 8,
            rows: 9,
            width: 8192,
            height: 8192,
        }),
    }
}

#[test]
fn bridge_drives_uncatalogued_models_orientations_cells_and_quilt_edge_pixels() {
    for (width, height, mode) in [
        (7680, 4320, 1),
        (4320, 7680, 2),
        (3840, 2160, 3),
        (2160, 3840, 4),
    ] {
        let display = connected_display(bridge_display("TEST-ONLY", width, height, mode)).unwrap();
        assert_eq!(display.preset.screen, [width, height]);
        assert_eq!(display.preset.aspect, width as f32 / height as f32);
        let LookingGlassOutput::LightField {
            layout,
            calibration,
            ..
        } = selected_output(&display.preset, std::slice::from_ref(&display)).unwrap()
        else {
            panic!("expected light field")
        };
        assert_eq!(
            (
                layout.columns(),
                layout.rows(),
                layout.view_size().x,
                layout.view_size().y
            ),
            (8, 9, 1024, 910)
        );
        assert_eq!(calibration.screen_size(), UVec2::new(width, height));
    }
    let mut missing = bridge_display("TEST-ONLY", 1536, 2048, 0);
    missing.quilt = None;
    assert!(connected_display(missing).is_err());
}

#[test]
fn old_manual_path_is_ignored_when_loading_preferences() {
    let preferences: OutputPreferences =
        toml::from_str("device_id = 'portrait'\ncalibration_path = 'missing/device.json'\n")
            .unwrap();
    assert_eq!(preferences.device_id, "portrait");
    assert!(
        !toml::to_string(&preferences)
            .unwrap()
            .contains("calibration_path")
    );
}

#[test]
fn fresh_bridge_result_selects_the_unit_by_serial_and_rejects_a_disconnected_selection() {
    for selected in ["TEST-B", "DISCONNECTED"] {
        let (sender, receiver) = mpsc::sync_channel(1);
        sender
            .send(Ok(vec![
                bridge_display("TEST-A", 1536, 2048, 0),
                bridge_display("TEST-B", 3840, 2160, 1),
            ]))
            .unwrap();
        let mut app = App::new();
        app.insert_resource(LookingGlassControl {
            task: Some(BridgeTask {
                receiver: Mutex::new(receiver),
                start: Some(OutputPreferences {
                    device_id: selected.into(),
                }),
            }),
            ..default()
        })
        .init_resource::<PluginControl>()
        .init_resource::<crate::ui_model::UiViewModel>()
        .add_systems(Update, apply_request);
        app.update();
        let control = app.world().resource::<LookingGlassControl>();
        assert!(control.task.is_none());
        if selected == "TEST-B" {
            assert_eq!(control.saving.as_ref().unwrap().device_id, "TEST-B");
            assert!(control.error.is_none());
        } else {
            assert!(control.saving.is_none());
            assert!(
                control
                    .error
                    .as_deref()
                    .unwrap()
                    .contains("no longer connected")
            );
        }
    }
}

#[test]
fn stop_cancels_a_pending_bridge_start_before_accepting_its_result() {
    let (sender, receiver) = mpsc::sync_channel(1);
    sender
        .send(Ok(vec![bridge_display("TEST-A", 1536, 2048, 0)]))
        .unwrap();
    let mut control = LookingGlassControl {
        task: Some(BridgeTask {
            receiver: Mutex::new(receiver),
            start: Some(OutputPreferences {
                device_id: "TEST-A".into(),
            }),
        }),
        ..default()
    };
    control.stop();
    let mut app = App::new();
    app.insert_resource(control)
        .init_resource::<PluginControl>()
        .init_resource::<crate::ui_model::UiViewModel>()
        .add_systems(Update, apply_request);
    app.update();
    let control = app.world().resource::<LookingGlassControl>();
    assert!(control.task.is_none());
    assert!(control.saving.is_none());
    assert!(control.connected.is_empty());
}

#[test]
fn crate_unavailable_result_stops_discovery_and_reports_the_requested_ui_message() {
    let (sender, receiver) = mpsc::sync_channel(1);
    sender.send(Err(bridge::Error::Unavailable)).unwrap();
    let mut app = App::new();
    app.insert_resource(LookingGlassControl {
        task: Some(BridgeTask {
            receiver: Mutex::new(receiver),
            start: Some(OutputPreferences {
                device_id: "TEST-A".into(),
            }),
        }),
        ..default()
    })
    .init_resource::<PluginControl>()
    .insert_resource(crate::ui_model::UiViewModel {
        looking_glass: LookingGlassViewModel {
            loading: true,
            ..default()
        },
        ..default()
    })
    .add_systems(Update, apply_request);
    app.update();
    let control = app.world().resource::<LookingGlassControl>();
    assert!(control.task.is_none());
    assert!(control.saving.is_none());
    assert!(control.connected.is_empty());
    assert_eq!(control.error.as_deref(), Some("Bridgeが見つかりません"));
    assert!(
        !app.world()
            .resource::<crate::ui_model::UiViewModel>()
            .looking_glass
            .loading
    );
}
