//! RusTuberV integration for the independent bevy_looking_glass plugin.
//! Settings, files, avatar framing, shortcuts and executor policy belong here.
use bevy::camera::{ClearColorConfig, Exposure, visibility::RenderLayers};
use bevy::ecs::schedule::SingleThreadedExecutor;
use bevy::light::cluster::ClusterConfig;
use bevy::prelude::*;
use bevy::render::view::{DebandDither, Tonemapping};
use bevy::window::{MonitorSelection, PrimaryWindow, WindowMode, WindowResolution};
use bevy_looking_glass::LookingGlassControl as PluginControl;
use bevy_looking_glass::bridge::{self, Display};
pub use bevy_looking_glass::{DevicePreset, DisplayFormat, QuiltPreset};
use bevy_looking_glass::{
    LookingGlassConfig, LookingGlassOutput, LookingGlassPlugin, LookingGlassStatus,
    LookingGlassSystems, LookingGlassView, ViewLayout, ViewPose, ViewRenderSettings,
    device_presets, parse_calibration,
};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, mpsc};
use vtuber_avatar::{
    AVATAR_RENDER_LAYER, AvatarCameraControl, AvatarLifecycle, FIXED_VERTICAL_FOV,
};

/// Failure to load the explicitly requested Looking Glass configuration.
#[derive(Debug, thiserror::Error)]
pub enum LookingGlassError {
    /// Calibration or output rejected by the independent plugin.
    #[error(transparent)]
    Plugin(#[from] bevy_looking_glass::LookingGlassError),
    /// Factory device data could not be acquired through Bridge.
    #[error("{}", bridge_error_message(.0))]
    Bridge(#[from] bridge::Error),
    /// The command-line option needs a configuration path.
    #[error("--looking-glass requires a configuration file path")]
    MissingPath,
    /// A configuration or calibration file could not be read.
    #[error("could not read {path}: {source}")]
    Read {
        /// File being read.
        path: PathBuf,
        /// Original filesystem error.
        source: std::io::Error,
    },
    /// A document does not match the expected JSON schema.
    #[error("invalid JSON in {path}: {source}")]
    Json {
        /// File being parsed.
        path: PathBuf,
        /// Original JSON error.
        source: serde_json::Error,
    },
    /// Values cannot describe the requested projection or pixel layout.
    #[error("invalid Looking Glass configuration: {0}")]
    Invalid(&'static str),
}

fn bridge_error_message(error: &bridge::Error) -> String {
    match error {
        bridge::Error::Unavailable => "Bridgeが見つかりません".into(),
        bridge::Error::Response(message) => {
            format!("Bridgeから機器情報を取得できません: {message}")
        }
    }
}

/// Saved selection, without automatically starting an output at application launch.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputPreferences {
    /// Physical unit's serial, or the catalog identifier for an HLD.
    pub device_id: String,
}

/// Looking Glass settings and current output status presented by the UI.
#[derive(Clone, Debug, Default)]
pub struct LookingGlassViewModel {
    /// Connected Bridge displays, followed by the existing single-view HLD presets.
    pub devices: Vec<DevicePreset>,
    /// Last successfully saved selection.
    pub preferences: OutputPreferences,
    /// Whether an auxiliary output window exists.
    pub running: bool,
    /// A background Bridge acquisition is in progress.
    pub loading: bool,
    /// The settings page has already attempted device discovery this session.
    pub discovered: bool,
    /// Configuration or output error, if any.
    pub error: Option<String>,
}

#[derive(Deserialize)]
struct SettingsFile {
    calibration: PathBuf,
    columns: u32,
    rows: u32,
    view_width: u32,
    view_height: u32,
    depth_scale: f32,
}

/// Application requests use persisted selections; the plugin only sees parsed values.
#[derive(Resource, Default)]
pub struct LookingGlassControl {
    pending: Option<Option<OutputPreferences>>,
    startup: Option<LookingGlassConfig>,
    saving: Option<OutputPreferences>,
    error: Option<String>,
    refresh: bool,
    task: Option<BridgeTask>,
    connected: Vec<ConnectedDisplay>,
}

type BridgeResult = Result<Vec<Display>, bridge::Error>;

struct BridgeTask {
    receiver: Mutex<mpsc::Receiver<BridgeResult>>,
    start: Option<OutputPreferences>,
}

struct ConnectedDisplay {
    preset: DevicePreset,
    calibration: bevy_looking_glass::Calibration,
    bridge: Display,
}

impl LookingGlassControl {
    /// Acquire the selected unit's current calibration from Bridge and request output.
    pub fn start(&mut self, preferences: OutputPreferences) {
        self.pending = Some(Some(preferences));
    }
    /// Stop output.
    pub fn stop(&mut self) {
        self.pending = Some(None);
    }
    /// Refresh the connected-device list on a worker, leaving rendering responsive.
    pub fn refresh_devices(&mut self) {
        self.refresh = true;
    }
}

/// Install optional output after Bevy and avatar plugins; CLI errors fail startup.
pub fn configure(
    app: &mut App,
    args: impl IntoIterator<Item = OsString>,
) -> Result<(), LookingGlassError> {
    let startup = config_argument(args)?
        .map(|path| {
            let settings: SettingsFile = read_json(&path)?;
            let calibration_path = path
                .parent()
                .unwrap_or(Path::new(""))
                .join(&settings.calibration);
            if !settings.depth_scale.is_finite() || settings.depth_scale < 0.0 {
                return Err(LookingGlassError::Invalid(
                    "depth_scale must be finite and nonnegative",
                ));
            }
            Ok(application_config(LookingGlassOutput::LightField {
                layout: ViewLayout::new(
                    settings.columns,
                    settings.rows,
                    settings.view_width,
                    settings.view_height,
                )?,
                calibration: read_calibration(&calibration_path)?,
                depth_scale: settings.depth_scale,
            }))
        })
        .transpose()?;
    let preferences = app
        .world()
        .get_resource::<crate::settings::AppSettings>()
        .map(|settings| settings.looking_glass().clone())
        .unwrap_or_default();
    app.init_resource::<crate::ui_model::UiViewModel>();
    app.world_mut()
        .resource_mut::<crate::ui_model::UiViewModel>()
        .looking_glass = LookingGlassViewModel {
        devices: device_presets()?
            .into_iter()
            .filter(|preset| preset.format == DisplayFormat::Hld)
            .collect(),
        preferences,
        ..default()
    };
    app.insert_resource(LookingGlassControl {
        startup,
        ..default()
    })
    .add_plugins(LookingGlassPlugin)
    .add_systems(
        Update,
        (apply_request, window_keys)
            .chain()
            .after(crate::orchestrator::process_ui_actions_system)
            .before(LookingGlassSystems::Control),
    )
    .add_systems(Update, sync_status.after(LookingGlassSystems::Control))
    .add_systems(
        PostUpdate,
        sync_avatar_pose
            .after(TransformSystems::Propagate)
            .before(LookingGlassSystems::SyncViews),
    )
    .add_systems(Last, close_with_main);
    if let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) {
        render_app.add_systems(bevy::render::ExtractSchedule, sync_executor);
    }
    Ok(())
}

fn read_calibration(path: &Path) -> Result<bevy_looking_glass::Calibration, LookingGlassError> {
    let bytes = std::fs::read(path).map_err(|source| LookingGlassError::Read {
        path: path.to_owned(),
        source,
    })?;
    Ok(parse_calibration(&bytes)?)
}

fn selected_config(
    preset: &DevicePreset,
    connected: &[ConnectedDisplay],
) -> Result<LookingGlassConfig, LookingGlassError> {
    match preset.format {
        DisplayFormat::LightField => {
            let display = connected
                .iter()
                .find(|display| display.preset.id == preset.id)
                .ok_or(LookingGlassError::Invalid(
                    "selected Looking Glass is no longer connected",
                ))?;
            let mut config =
                application_config(preset.light_field_output(display.calibration.clone())?);
            config.display = Some(display.bridge.clone());
            Ok(config)
        }
        DisplayFormat::Hld => Ok(application_config(preset.hld_output()?)),
        DisplayFormat::Standalone => Err(LookingGlassError::Invalid(
            "musubi has no live desktop video input",
        )),
    }
}

fn connected_display(display: Display) -> Result<ConnectedDisplay, LookingGlassError> {
    let calibration = parse_calibration(display.calibration.as_bytes())?;
    let screen = calibration.screen_size();
    let quilt = display.quilt.as_ref().ok_or(LookingGlassError::Invalid(
        "Bridge returned no default quilt",
    ))?;
    let preset = DevicePreset {
        id: display.serial.clone(),
        name: format!(
            "{} · {} ({})",
            display.name, display.hardware_version, display.serial
        ),
        format: DisplayFormat::LightField,
        screen: screen.to_array(),
        aspect: screen.x as f32 / screen.y as f32,
        quilt: Some(QuiltPreset {
            columns: quilt.columns,
            rows: quilt.rows,
            width: quilt.width,
            height: quilt.height,
        }),
        safe_area: None,
        legacy: false,
        sources: Vec::new(),
    };
    // Validate Bridge's grid before publishing it as a startable selection.
    preset.light_field_output(calibration.clone())?;
    Ok(ConnectedDisplay {
        preset,
        calibration,
        bridge: display,
    })
}

fn application_config(output: LookingGlassOutput) -> LookingGlassConfig {
    let screen = output.screen_size();
    let mut config = LookingGlassConfig::new(output, RenderLayers::layer(AVATAR_RENDER_LAYER), 3);
    config.window.title =
        "RusTuberV — Looking Glass (F11: fullscreen / F10: quilt / Esc: close)".into();
    config.window.resolution = WindowResolution::new(screen.x.div_ceil(2), screen.y.div_ceil(2))
        .with_scale_factor_override(1.0);
    config.view_render = ViewRenderSettings {
        hdr: false,
        exposure: Exposure::BLENDER,
        tonemapping: Tonemapping::None,
        deband_dither: DebandDither::Disabled,
        cluster_config: ClusterConfig::Single,
        no_indirect_drawing: true,
        clear_color: ClearColorConfig::Custom(Color::BLACK),
    };
    config
}

fn apply_request(
    mut control: ResMut<LookingGlassControl>,
    mut plugin: ResMut<PluginControl>,
    mut vm: ResMut<crate::ui_model::UiViewModel>,
) {
    let pending = control.pending.take();
    let startup = control.startup.take();
    let refresh = std::mem::take(&mut control.refresh);
    if let Some(request) = pending {
        control.error = None;
        control.saving = None;
        control.task = None;
        vm.looking_glass.loading = false;
        if let Some(selection) = request {
            let preset = vm
                .looking_glass
                .devices
                .iter()
                .find(|p| p.id == selection.device_id);
            if preset.is_some_and(|preset| preset.format == DisplayFormat::Hld) {
                let result = preset
                    .ok_or(LookingGlassError::Invalid("unknown HLD model"))
                    .and_then(|preset| selected_config(preset, &control.connected));
                start_output(&mut control, &mut plugin, selection, result);
            } else {
                begin_discovery(&mut control, &mut vm.looking_glass, Some(selection));
            }
        } else {
            plugin.stop();
        }
    } else if refresh && control.task.is_none() {
        begin_discovery(&mut control, &mut vm.looking_glass, None);
    } else if let Some(config) = startup {
        plugin.start(config);
    }
    let result = control.task.as_mut().and_then(|task| {
        let receiver = task
            .receiver
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match receiver.try_recv() {
            Ok(result) => Some(result.map_err(LookingGlassError::from)),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(LookingGlassError::Invalid(
                "Bridge worker disconnected",
            ))),
        }
    });
    if let Some(result) = result {
        let selection = control.task.take().and_then(|task| task.start);
        vm.looking_glass.loading = false;
        vm.looking_glass
            .devices
            .retain(|preset| preset.format == DisplayFormat::Hld);
        control.connected.clear();
        let parsed = result.and_then(|displays| {
            displays
                .into_iter()
                .map(connected_display)
                .collect::<Result<Vec<_>, _>>()
        });
        match parsed {
            Ok(displays) => {
                let mut devices = displays
                    .iter()
                    .map(|display| display.preset.clone())
                    .collect::<Vec<_>>();
                devices.append(&mut vm.looking_glass.devices);
                vm.looking_glass.devices = devices;
                control.connected = displays;
                if let Some(selection) = selection {
                    let result = vm
                        .looking_glass
                        .devices
                        .iter()
                        .find(|preset| preset.id == selection.device_id)
                        .ok_or(LookingGlassError::Invalid(
                            "selected Looking Glass is no longer connected",
                        ))
                        .and_then(|preset| selected_config(preset, &control.connected));
                    start_output(&mut control, &mut plugin, selection, result);
                } else if control.connected.is_empty() {
                    control.error = Some("Looking Glassが見つかりません".into());
                }
            }
            Err(error) => control.error = Some(error.to_string()),
        }
    }
}

fn begin_discovery(
    control: &mut LookingGlassControl,
    vm: &mut LookingGlassViewModel,
    start: Option<OutputPreferences>,
) {
    control.error = None;
    vm.discovered = true;
    let (sender, receiver) = mpsc::sync_channel(1);
    match std::thread::Builder::new()
        .name("looking-glass-bridge".into())
        .spawn(move || {
            let _ = sender.send(bridge::discover());
        }) {
        Ok(_) => {
            control.task = Some(BridgeTask {
                receiver: Mutex::new(receiver),
                start,
            });
            vm.loading = true;
        }
        Err(error) => control.error = Some(format!("Bridgeの取得を開始できません: {error}")),
    }
}

fn start_output(
    control: &mut LookingGlassControl,
    plugin: &mut PluginControl,
    selection: OutputPreferences,
    result: Result<LookingGlassConfig, LookingGlassError>,
) {
    match result {
        Ok(config) => {
            plugin.set_quilt_preview(false);
            plugin.start(config);
            control.saving = Some(selection);
        }
        Err(error) => control.error = Some(error.to_string()),
    }
}

fn sync_status(
    mut control: ResMut<LookingGlassControl>,
    status: Res<LookingGlassStatus>,
    mut vm: ResMut<crate::ui_model::UiViewModel>,
    mut settings: ResMut<crate::settings::AppSettings>,
) {
    if let Some(selection) = control.saving.take()
        && status.error.is_none()
        && status.window.is_some()
    {
        match settings.save_looking_glass(selection.clone()) {
            Ok(()) => vm.looking_glass.preferences = selection,
            Err(error) => control.error = Some(error.to_string()),
        }
    }
    vm.looking_glass.running = status.window.is_some();
    vm.looking_glass.error = control.error.clone().or_else(|| status.error.clone());
}

fn avatar_view_pose(
    control: &AvatarCameraControl,
    lifecycle: &AvatarLifecycle,
) -> Option<ViewPose> {
    let pose = control.current_for(lifecycle.current_generation())?;
    let transform = pose.transform();
    let perspective = PerspectiveProjection::default();
    Some(ViewPose {
        transform,
        focus_distance: (pose.target() - transform.translation).dot(*transform.forward()),
        fov_y: FIXED_VERTICAL_FOV,
        near: perspective.near,
        far: perspective.far,
    })
}

fn sync_avatar_pose(
    control: Res<AvatarCameraControl>,
    lifecycle: Res<AvatarLifecycle>,
    mut view: ResMut<LookingGlassView>,
) {
    view.pose = avatar_view_pose(&control, &lifecycle);
}

fn window_keys(
    keys: Res<ButtonInput<KeyCode>>,
    status: Res<LookingGlassStatus>,
    mut control: ResMut<PluginControl>,
    mut windows: Query<&mut Window>,
) {
    let Some(entity) = status.window else {
        return;
    };
    let Ok(mut window) = windows.get_mut(entity) else {
        return;
    };
    if !window.focused {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        control.stop();
    }
    if keys.just_pressed(KeyCode::F11) {
        window.mode = match window.mode {
            WindowMode::Windowed => WindowMode::BorderlessFullscreen(MonitorSelection::Current),
            _ => WindowMode::Windowed,
        };
    }
    if keys.just_pressed(KeyCode::F10) {
        control.set_quilt_preview(!status.quilt_preview);
    }
}

fn close_with_main(
    mut commands: Commands,
    main: Query<(), With<PrimaryWindow>>,
    status: Res<LookingGlassStatus>,
) {
    if main.is_empty()
        && let Some(window) = status.window
        && let Ok(mut entity) = commands.get_entity(window)
    {
        entity.despawn();
    }
}

fn sync_executor(
    mut commands: Commands,
    status: bevy::render::Extract<Res<LookingGlassStatus>>,
    mut enabled: Local<bool>,
) {
    let multiview = status.light_field;
    if *enabled != multiview {
        *enabled = multiview;
        commands.queue(move |world: &mut World| {
            if let Some(mut schedules) = world.get_resource_mut::<bevy::ecs::schedule::Schedules>()
                && let Some(schedule) = schedules.get_mut(bevy::core_pipeline::Core3d)
            {
                if multiview {
                    schedule.set_executor(SingleThreadedExecutor::new());
                } else {
                    schedule.set_executor(bevy::ecs::schedule::MultiThreadedExecutor::new());
                }
            }
        });
    }
}

fn config_argument(
    args: impl IntoIterator<Item = OsString>,
) -> Result<Option<PathBuf>, LookingGlassError> {
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--looking-glass" {
            return args
                .next()
                .map(PathBuf::from)
                .map(Some)
                .ok_or(LookingGlassError::MissingPath);
        }
    }
    Ok(None)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LookingGlassError> {
    let bytes = std::fs::read(path).map_err(|source| LookingGlassError::Read {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| LookingGlassError::Json {
        path: path.to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests;
