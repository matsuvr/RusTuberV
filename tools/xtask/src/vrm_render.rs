//! Renders each imported VRM through the avatar output camera to a PNG.

use crate::task_result::{TaskOutcome, TaskResult};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bevy::app::{App, AppExit};
use bevy::asset::io::{AssetSourceBuilder, AssetSourceBuilders};
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::time::TimeUpdateStrategy;
use bevy::window::ExitCondition;
use vtuber_avatar::{
    AvatarLifecycle, AvatarOutputFrameSlot, AvatarOutputState, AvatarViewportCamera,
    VtuberAvatarPlugin,
};
use vtuber_core::{VideoOutputFrame, VideoOutputProfile};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
/// Frames the renderer needs before the offscreen image contains the avatar.
const WARMUP_FRAMES: usize = 96;
const MAX_LOAD_FRAMES: usize = 1_200;
/// Exit code used when the GPU/readback path cannot be exercised.
pub const EXIT_NOT_RUN: i32 = 2;

/// Renders one VRM or every VRM in a directory.
pub fn run(args: &[String]) -> TaskResult {
    let input = args
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: cargo xtask vrm-render <vrm-or-dir> <out-dir>".to_owned())?;
    let out_dir = args
        .get(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: cargo xtask vrm-render <vrm-or-dir> <out-dir>".to_owned())?;

    let mut models = Vec::new();
    if input.is_dir() {
        let entries = std::fs::read_dir(&input)
            .map_err(|error| format!("cannot read {}: {error}", input.display()))?;
        for entry in entries {
            let path = entry
                .map_err(|error| format!("cannot read directory entry: {error}"))?
                .path();
            if path.extension().is_some_and(|extension| extension == "vrm") {
                models.push(path);
            }
        }
        models.sort();
    } else {
        models.push(input);
    }
    if models.is_empty() {
        return Err("no .vrm files found".to_owned().into());
    }
    std::fs::create_dir_all(&out_dir)
        .map_err(|error| format!("cannot create {}: {error}", out_dir.display()))?;

    let mut failed = Vec::new();
    for model in &models {
        match render_model(model, &out_dir) {
            Ok(summary) => print!("{summary}"),
            Err(RenderError::NotRun(reason)) => {
                return Ok(TaskOutcome::NotRun {
                    reason,
                    exit_code: EXIT_NOT_RUN,
                });
            }
            Err(RenderError::Failed(reason)) => {
                println!("{}: FAIL ({reason})", model.display());
                failed.push(model.clone());
            }
        }
    }
    if failed.is_empty() {
        Ok(TaskOutcome::Completed)
    } else {
        Err(format!(
            "{} model(s) differed from the requirement: {failed:?}",
            failed.len()
        )
        .into())
    }
}

enum RenderError {
    NotRun(String),
    Failed(String),
}

/// Builds the app, loads the model, and captures the avatar output.
fn render_model(path: &Path, out_dir: &Path) -> Result<String, RenderError> {
    let temporary = tempfile::tempdir()
        .map_err(|error| RenderError::Failed(format!("cannot create managed root: {error}")))?;
    let managed_root = temporary.path();

    let imported =
        vtuber_app::import::import_vrm(path, managed_root, vtuber_app::import::DEFAULT_SIZE_LIMIT)
            .map_err(|error| RenderError::Failed(format!("import failed: {error}")))?;
    let managed_root_string = managed_root
        .to_str()
        .ok_or_else(|| RenderError::Failed("managed root is not valid UTF-8".to_owned()))?;
    let mut sources = AssetSourceBuilders::default();
    sources.insert(
        "user",
        AssetSourceBuilder::platform_default(managed_root_string, None),
    );

    let mut orchestrator = vtuber_app::orchestrator::Orchestrator::new(managed_root.to_path_buf());
    orchestrator.queue_imported_model(imported);
    let mut app = App::new();
    app.insert_resource(sources)
        .insert_resource(orchestrator)
        .insert_resource(AvatarOutputState::with_profile(VideoOutputProfile {
            width: WIDTH,
            height: HEIGHT,
            fps: 60,
            pixel_format: vtuber_core::VideoOutputPixelFormat::Bgra8StraightAlpha,
        }))
        // Freeze the clock so idle motion and UV animation are deterministic.
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                // Pixel comparisons need every native material pipeline in
                // the initial frame, including the slower Standard variants.
                .set(RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                .disable::<PipelinedRenderingPlugin>()
                // Keep shader/pipeline diagnostics visible when a render fails.
                .disable::<bevy::winit::WinitPlugin>(),
        )
        .add_plugins(VtuberAvatarPlugin)
        .init_resource::<vtuber_app::AvatarIoRuntime>()
        .add_systems(
            Update,
            (
                vtuber_app::prepare_avatar_io_system,
                vtuber_app::orchestrator::sync_avatar_lifecycle_system
                    .after(vtuber_avatar::unload::despawn_unloading_avatar),
            )
                .chain(),
        )
        .add_systems(Update, (place_camera, activate_output));
    app.finish();
    app.cleanup();

    let name = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().replace(' ', "_"))
        .unwrap_or_else(|| "model".to_owned());

    let mut ready = false;
    for _ in 0..MAX_LOAD_FRAMES {
        app.update();
        if app.world().resource::<AvatarLifecycle>().state()
            == vtuber_avatar::AvatarLifecycleState::Ready
        {
            ready = true;
            break;
        }
        if exited(&mut app) {
            return Err(RenderError::NotRun("the renderer exited early".to_owned()));
        }
    }
    if !ready {
        return Err(RenderError::Failed(
            "the model never reached Ready".to_owned(),
        ));
    }

    let arm_deadline = Instant::now() + Duration::from_secs(60);
    loop {
        app.update();
        let mut query = app
            .world_mut()
            .query::<&vtuber_avatar::UpperLimbSolveStatus>();
        if query
            .iter(app.world())
            .any(|s| !matches!(s, vtuber_avatar::UpperLimbSolveStatus::Solving))
        {
            break;
        }
        if Instant::now() >= arm_deadline {
            return Err(RenderError::Failed(
                "upper limb initialization timed out".to_owned(),
            ));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut arm_status = app
        .world_mut()
        .query::<&vtuber_avatar::UpperLimbSolveStatus>();
    println!(
        "Upper limbs: {:?}",
        arm_status.iter(app.world()).collect::<Vec<_>>()
    );
    let arm_failure = arm_status
        .iter(app.world())
        .find(|s| !matches!(s, vtuber_avatar::UpperLimbSolveStatus::Feasible { .. }))
        .map(|s| format!("upper limb initialization: {s:?}"));
    for _ in 0..WARMUP_FRAMES {
        app.update();
    }
    let frame = take_frame(&mut app, deadline)?;
    if opaque_pixels(frame.data()) == 0 {
        return Err(RenderError::Failed("the avatar output is empty".into()));
    }
    let image = out_dir.join(format!("{name}.png"));
    write_png(&image, &frame)?;
    if let Some(error) = arm_failure {
        return Err(RenderError::Failed(error));
    }
    Ok(format!("{}\n", image.display()))
}

fn place_camera(mut cameras: Query<(&mut Transform, &mut Projection), With<AvatarViewportCamera>>) {
    let transform = Transform::from_translation(Vec3::new(0.0, 1.35, 1.45))
        .looking_at(Vec3::new(0.0, 1.25, 0.0), Vec3::Y);
    for (mut camera_transform, mut projection) in &mut cameras {
        if *camera_transform != transform {
            *camera_transform = transform;
        }
        if let Projection::Perspective(perspective) = &mut *projection {
            perspective.fov = std::f32::consts::FRAC_PI_4;
            perspective.aspect_ratio = 1.0;
        }
    }
}

fn activate_output(
    mut state: ResMut<AvatarOutputState>,
    cameras: Query<Entity, With<vtuber_avatar::AvatarOutputCamera>>,
) {
    if !state.is_rendering() && cameras.iter().next().is_some() {
        state.activate();
    }
}

fn exited(app: &mut App) -> bool {
    app.world()
        .get_resource::<Messages<AppExit>>()
        .is_some_and(|messages| !messages.is_empty())
}

fn take_frame(app: &mut App, deadline: Instant) -> Result<VideoOutputFrame, RenderError> {
    while Instant::now() < deadline {
        app.update();
        if let Some(frame) = app
            .world_mut()
            .resource_mut::<AvatarOutputFrameSlot>()
            .take_latest()
        {
            return Ok(frame);
        }
        if exited(app) {
            break;
        }
    }
    Err(RenderError::NotRun(
        "GPU readback did not complete; the local renderer/GPU path is unavailable".to_owned(),
    ))
}

/// Writes the frame as a PNG so the render can be inspected as an image.
fn write_png(path: &Path, frame: &VideoOutputFrame) -> Result<(), RenderError> {
    let mut rgba = Vec::with_capacity(frame.data().len());
    for pixel in frame.data().as_chunks::<4>().0 {
        rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
    }
    let buffer = image::RgbaImage::from_raw(WIDTH, HEIGHT, rgba)
        .ok_or_else(|| RenderError::Failed("frame size does not match the profile".to_owned()))?;
    buffer
        .save(path)
        .map_err(|error| RenderError::Failed(format!("cannot write {}: {error}", path.display())))
}

fn opaque_pixels(data: &[u8]) -> usize {
    data.as_chunks::<4>()
        .0
        .iter()
        .filter(|pixel| pixel[3] > 0)
        .count()
}
