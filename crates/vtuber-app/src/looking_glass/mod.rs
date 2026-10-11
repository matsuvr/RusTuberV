//! RusTuberV integration for the independent bevy_looking_glass plugin.
//! Settings, files, avatar framing, shortcuts and executor policy belong here.
use bevy::camera::{ClearColorConfig, Exposure, visibility::RenderLayers};
use bevy::ecs::schedule::SingleThreadedExecutor;
use bevy::light::cluster::ClusterConfig;
use bevy::prelude::*;
use bevy::render::view::{DebandDither, Tonemapping};
use bevy::window::{MonitorSelection, PrimaryWindow, WindowMode, WindowResolution};
use bevy_looking_glass::LookingGlassControl as PluginControl;
pub use bevy_looking_glass::{DevicePreset, DisplayFormat, QuiltPreset};
use bevy_looking_glass::{
    LookingGlassConfig, LookingGlassOutput, LookingGlassPlugin, LookingGlassStatus,
    LookingGlassSystems, LookingGlassView, ViewLayout, ViewPose, ViewRenderSettings,
    device_presets, parse_calibration,
};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use vtuber_avatar::{
    AVATAR_RENDER_LAYER, AvatarCameraControl, AvatarLifecycle, FIXED_VERTICAL_FOV,
};

/// Failure to load the explicitly requested Looking Glass configuration.
#[derive(Debug, thiserror::Error)]
pub enum LookingGlassError {
    /// Calibration or output rejected by the independent plugin.
    #[error(transparent)]
    Plugin(#[from] bevy_looking_glass::LookingGlassError),
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

/// Saved selection, without automatically starting an output at application launch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputPreferences {
    /// Stable identifier in the bundled device catalog.
    pub device_id: String,
    /// Raw calibration JSON belonging to this physical display.
    pub calibration_path: std::path::PathBuf,
}

impl Default for OutputPreferences {
    fn default() -> Self {
        Self {
            device_id: "portrait".into(),
            calibration_path: Default::default(),
        }
    }
}

/// Looking Glass settings and current output status presented by the UI.
#[derive(Clone, Debug, Default)]
pub struct LookingGlassViewModel {
    /// Presets loaded from the bundled JSON.
    pub devices: Vec<DevicePreset>,
    /// Last successfully saved selection.
    pub preferences: OutputPreferences,
    /// Whether an auxiliary output window exists.
    pub running: bool,
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
}

impl LookingGlassControl {
    /// Load the selected unit's calibration and request output.
    pub fn start(&mut self, preferences: OutputPreferences) {
        self.pending = Some(Some(preferences));
    }
    /// Stop output.
    pub fn stop(&mut self) {
        self.pending = Some(None);
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
        devices: device_presets()?,
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

fn selected_output(
    preset: &DevicePreset,
    preferences: &OutputPreferences,
) -> Result<LookingGlassOutput, LookingGlassError> {
    match preset.format {
        DisplayFormat::LightField => {
            Ok(preset.light_field_output(read_calibration(&preferences.calibration_path)?)?)
        }
        DisplayFormat::Hld => Ok(preset.hld_output()?),
        DisplayFormat::Standalone => Err(LookingGlassError::Invalid(
            "musubi has no live desktop video input",
        )),
    }
}

fn application_config(output: LookingGlassOutput) -> LookingGlassConfig {
    let screen = output.screen_size();
    let mut config = LookingGlassConfig::new(output, RenderLayers::layer(AVATAR_RENDER_LAYER), 3);
    config.window.title =
        "RusTuberV — Looking Glass (F11: fullscreen / F10: quilt / Esc: close)".into();
    config.window.resolution = WindowResolution::new(screen.x.div_ceil(2), screen.y.div_ceil(2))
        .with_scale_factor_override(1.0);
    config.view_render = ViewRenderSettings {
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
    vm: Res<crate::ui_model::UiViewModel>,
) {
    let pending = control.pending.take();
    let startup = control.startup.take();
    if let Some(request) = pending {
        control.error = None;
        control.saving = None;
        if let Some(selection) = request {
            let result = vm
                .looking_glass
                .devices
                .iter()
                .find(|p| p.id == selection.device_id)
                .ok_or(LookingGlassError::Invalid("unknown Looking Glass model"))
                .and_then(|preset| selected_output(preset, &selection));
            match result {
                Ok(output) => {
                    plugin.set_quilt_preview(false);
                    plugin.start(application_config(output));
                    control.saving = Some(selection);
                }
                Err(error) => control.error = Some(error.to_string()),
            }
        } else {
            plugin.stop();
        }
    } else if let Some(config) = startup {
        plugin.start(config);
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
