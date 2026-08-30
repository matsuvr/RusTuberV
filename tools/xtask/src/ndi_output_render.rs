//! GPU-backed validator for the Issue #46 transparent output contract.
//!
//! This is not a substitute for unit tests. It renders synthetic scenes through
//! the production offscreen camera/readback path and inspects CPU pixels.
//! When a GPU or readback completion is unavailable, the command exits 2
//! (`NOT RUN`) instead of reporting success.

use bevy::app::AppExit;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::winit::WinitPlugin;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use vtuber_avatar::{
    AVATAR_RENDER_LAYER, AvatarOutputCamera, AvatarOutputFrameSlot, AvatarOutputState,
    AvatarViewportCamera, VIEWPORT_ONLY_RENDER_LAYER, register_output_systems,
};
use vtuber_core::{VideoOutputFrame, VideoOutputProfile};

const VALIDATOR_WIDTH: u32 = 64;
const VALIDATOR_HEIGHT: u32 = 64;
const MAX_WAIT: Duration = Duration::from_secs(20);

/// Exit code used when the GPU/readback path cannot be exercised.
pub const EXIT_NOT_RUN: i32 = 2;

/// Run the GPU pixel-contract validator.
pub fn run(args: &[String]) -> Result<(), String> {
    let evidence = args
        .iter()
        .position(|argument| argument == "--evidence")
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from);

    let mut report = RenderValidationReport::default();
    match run_cases(&mut report) {
        Ok(()) => {
            report.result = "PASS".into();
            if let Some(path) = evidence {
                write_report(&path, &report)?;
            }
            println!("NDI/avatar GPU output pixel contracts: PASS");
            Ok(())
        }
        Err(RenderValidationError::NotRun(reason)) => {
            report.result = "NOT RUN".into();
            report.not_run_reason = Some(reason.clone());
            if let Some(path) = evidence {
                write_report(&path, &report)?;
            }
            Err(format!("NOT RUN: {reason}"))
        }
        Err(RenderValidationError::Failed(reason)) => {
            report.result = "FAIL".into();
            report.failure = Some(reason.clone());
            if let Some(path) = evidence {
                write_report(&path, &report)?;
            }
            Err(reason)
        }
    }
}

#[derive(Debug)]
enum RenderValidationError {
    NotRun(String),
    Failed(String),
}

#[derive(Default)]
struct RenderValidationReport {
    result: String,
    empty_alpha_zero: bool,
    opaque_alpha_255: bool,
    partial_alpha: bool,
    ground_excluded: bool,
    inactive_stops_readback: bool,
    not_run_reason: Option<String>,
    failure: Option<String>,
}

fn run_cases(report: &mut RenderValidationReport) -> Result<(), RenderValidationError> {
    let empty = capture_scene(SceneKind::Empty)?;
    report.empty_alpha_zero = empty.iter().all(|pixel| pixel[3] == 0);
    if !report.empty_alpha_zero {
        return Err(RenderValidationError::Failed(
            "empty/transparent scene did not produce A=0 for every pixel".into(),
        ));
    }

    let opaque = capture_scene(SceneKind::Opaque)?;
    let center = center_pixel(&opaque);
    report.opaque_alpha_255 = center[3] == 255;
    if !report.opaque_alpha_255 {
        return Err(RenderValidationError::Failed(format!(
            "opaque primitive center alpha was {}, expected 255",
            center[3]
        )));
    }

    let blended = capture_scene(SceneKind::Translucent)?;
    let blended_center = center_pixel(&blended);
    report.partial_alpha = (1..255).contains(&blended_center[3]);
    if !report.partial_alpha {
        return Err(RenderValidationError::Failed(format!(
            "translucent primitive center alpha was {}, expected 0 < A < 255",
            blended_center[3]
        )));
    }

    report.ground_excluded = empty.iter().all(|pixel| pixel[3] == 0);
    report.inactive_stops_readback = verify_inactive_stops_readback()?;
    Ok(())
}

#[derive(Clone, Copy)]
enum SceneKind {
    Empty,
    Opaque,
    Translucent,
}

fn capture_scene(kind: SceneKind) -> Result<Vec<[u8; 4]>, RenderValidationError> {
    let mut app = validator_app(kind, true)?;
    let deadline = Instant::now() + MAX_WAIT;
    let mut last = None;
    while Instant::now() < deadline {
        app.update();
        if let Some(frame) = app
            .world_mut()
            .resource_mut::<AvatarOutputFrameSlot>()
            .take_latest()
        {
            let sampled = pixels(&frame);
            if scene_matches(kind, &sampled) {
                return Ok(sampled);
            }
            last = Some(sampled);
        }
    }
    match last {
        Some(sampled) => Ok(sampled),
        None => Err(RenderValidationError::NotRun(
            "GPU readback did not complete; the local renderer/GPU path is unavailable".into(),
        )),
    }
}

fn scene_matches(kind: SceneKind, sampled: &[[u8; 4]]) -> bool {
    match kind {
        SceneKind::Empty => sampled.iter().all(|pixel| pixel[3] == 0),
        SceneKind::Opaque => center_pixel(sampled)[3] == 255,
        SceneKind::Translucent => (1..255).contains(&center_pixel(sampled)[3]),
    }
}

fn verify_inactive_stops_readback() -> Result<bool, RenderValidationError> {
    let mut app = validator_app(SceneKind::Opaque, true)?;
    let _ = wait_for_frame(&mut app)?;
    app.world_mut()
        .resource_mut::<AvatarOutputState>()
        .deactivate();
    for _ in 0..10 {
        app.update();
    }
    let _ = app
        .world_mut()
        .resource_mut::<AvatarOutputFrameSlot>()
        .take_latest();
    let received = app
        .world()
        .resource::<AvatarOutputFrameSlot>()
        .received_frames();
    for _ in 0..30 {
        app.update();
    }
    let later = app
        .world()
        .resource::<AvatarOutputFrameSlot>()
        .received_frames();
    if later > received {
        return Err(RenderValidationError::Failed(
            "deactivated output continued to publish readback frames".into(),
        ));
    }
    Ok(true)
}

fn validator_app(kind: SceneKind, active: bool) -> Result<App, RenderValidationError> {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin { ..default() })
            .disable::<PipelinedRenderingPlugin>()
            .disable::<WinitPlugin>()
            .disable::<bevy::log::LogPlugin>(),
    )
    .insert_resource(AvatarOutputState::with_profile(VideoOutputProfile {
        width: VALIDATOR_WIDTH,
        height: VALIDATOR_HEIGHT,
        fps: 60,
        pixel_format: vtuber_core::VideoOutputPixelFormat::Bgra8StraightAlpha,
    }))
    .insert_resource(ClearColor(Color::srgba(0.0, 0.0, 0.0, 0.0)))
    .insert_resource(vtuber_avatar::AvatarLifecycle::default())
    .insert_resource(SceneChoice(kind))
    .insert_resource(OutputArmed(false));
    register_output_systems(&mut app);
    app.add_systems(Startup, setup_validator_scene);
    app.add_systems(Update, activate_output_after_setup);
    app.finish();
    app.cleanup();
    if !active {
        app.world_mut()
            .resource_mut::<AvatarOutputState>()
            .deactivate();
    }
    Ok(app)
}

#[derive(Resource, Clone, Copy)]
struct SceneChoice(SceneKind);

#[derive(Resource, Clone, Copy)]
struct OutputArmed(bool);

fn setup_validator_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    choice: Res<SceneChoice>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(8.0, 8.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.0, 0.0),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, -1.0, 0.0),
        RenderLayers::layer(VIEWPORT_ONLY_RENDER_LAYER),
    ));

    let camera_transform =
        Transform::from_translation(Vec3::new(0.0, 0.0, 3.0)).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 1.0,
            ..default()
        }),
        AvatarViewportCamera::from_default_transform(camera_transform),
        camera_transform,
        RenderLayers::from_layers(&[AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER]),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 2_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.4, 0.0)),
        RenderLayers::from_layers(&[AVATAR_RENDER_LAYER, VIEWPORT_ONLY_RENDER_LAYER]),
    ));

    match choice.0 {
        SceneKind::Empty => {}
        SceneKind::Opaque => {
            commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(4.0, 4.0, 4.0))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.0, 1.0, 0.0),
                    unlit: true,
                    alpha_mode: AlphaMode::Opaque,
                    ..default()
                })),
                Transform::from_xyz(0.0, 0.0, 0.0),
                RenderLayers::layer(AVATAR_RENDER_LAYER),
            ));
        }
        SceneKind::Translucent => {
            commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(4.0, 4.0, 4.0))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgba(0.0, 0.0, 1.0, 0.4),
                    alpha_mode: AlphaMode::Blend,
                    unlit: true,
                    ..default()
                })),
                Transform::from_xyz(0.0, 0.0, 0.0),
                RenderLayers::layer(AVATAR_RENDER_LAYER),
            ));
        }
    }
}

fn activate_output_after_setup(
    mut state: ResMut<AvatarOutputState>,
    mut armed: ResMut<OutputArmed>,
    cameras: Query<Entity, With<AvatarOutputCamera>>,
) {
    if !armed.0 && cameras.iter().next().is_some() {
        state.activate();
        armed.0 = true;
    }
}

fn wait_for_frame(app: &mut App) -> Result<VideoOutputFrame, RenderValidationError> {
    let deadline = Instant::now() + MAX_WAIT;
    while Instant::now() < deadline {
        app.update();
        if let Some(frame) = app
            .world_mut()
            .resource_mut::<AvatarOutputFrameSlot>()
            .take_latest()
        {
            return Ok(frame);
        }
        if app
            .world()
            .get_resource::<Messages<AppExit>>()
            .is_some_and(|messages| !messages.is_empty())
        {
            break;
        }
    }
    Err(RenderValidationError::NotRun(
        "GPU readback did not complete; the local renderer/GPU path is unavailable".into(),
    ))
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn pixels(frame: &VideoOutputFrame) -> Vec<[u8; 4]> {
    frame
        .data
        .chunks_exact(4)
        .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
        .collect()
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn center_pixel(pixels: &[[u8; 4]]) -> [u8; 4] {
    let index = (VALIDATOR_HEIGHT / 2) * VALIDATOR_WIDTH + (VALIDATOR_WIDTH / 2);
    pixels[index as usize]
}

fn write_report(path: &PathBuf, report: &RenderValidationReport) -> Result<(), String> {
    let contents = format!(
        "result={}\n\
         empty_alpha_zero={}\n\
         opaque_alpha_255={}\n\
         partial_alpha={}\n\
         ground_excluded={}\n\
         inactive_stops_readback={}\n\
         not_run_reason={}\n\
         failure={}\n",
        report.result,
        report.empty_alpha_zero,
        report.opaque_alpha_255,
        report.partial_alpha,
        report.ground_excluded,
        report.inactive_stops_readback,
        report.not_run_reason.as_deref().unwrap_or(""),
        report.failure.as_deref().unwrap_or(""),
    );
    std::fs::write(path, contents)
        .map_err(|error| format!("cannot write GPU validator evidence: {error}"))?;
    Ok(())
}
