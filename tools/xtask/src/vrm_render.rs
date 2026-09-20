//! Renders a VRM with the rich look off and on and records both frames.
//!
//! This is the mechanical check for the review requirement that switching the
//! look off reproduces the plain display and switching it on changes the
//! pixels: for every model it renders the same pose and camera through
//! OFF -> ON -> OFF -> ON -> strength 0, then reports the pixel differences.
//! When no GPU/readback is available the command exits 2 (`NOT RUN`) instead
//! of reporting success.

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
    ImportedAvatar, LoadImportedAvatarRequest, UserAssetPath, VtuberAvatarPlugin,
};
use vtuber_core::{VideoOutputFrame, VideoOutputProfile};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
/// Frames the renderer needs before the offscreen image contains the avatar.
const WARMUP_FRAMES: usize = 96;
/// Frames between the two look states so the readback reflects the change.
const SETTLE_FRAMES: usize = 120;
const MAX_LOAD_FRAMES: usize = 1_200;
/// Exit code used when the GPU/readback path cannot be exercised.
pub const EXIT_NOT_RUN: i32 = 2;

/// Renders one VRM or every VRM in a directory.
pub fn run(args: &[String]) -> Result<(), String> {
    let input = args
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: cargo xtask -- vrm-render <vrm-or-dir> <out-dir>".to_owned())?;
    let out_dir = args
        .get(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: cargo xtask -- vrm-render <vrm-or-dir> <out-dir>".to_owned())?;

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
        return Err("no .vrm files found".to_owned());
    }
    std::fs::create_dir_all(&out_dir)
        .map_err(|error| format!("cannot create {}: {error}", out_dir.display()))?;

    let mut failed = Vec::new();
    for model in &models {
        match render_model(model, &out_dir) {
            Ok(summary) => print!("{summary}"),
            Err(RenderError::NotRun(reason)) => return Err(format!("NOT RUN: {reason}")),
            Err(RenderError::Failed(reason)) => {
                println!("{}: FAIL ({reason})", model.display());
                failed.push(model.clone());
            }
        }
    }
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} model(s) differed from the requirement: {failed:?}",
            failed.len()
        ))
    }
}

enum RenderError {
    NotRun(String),
    Failed(String),
}

/// Builds the app, loads the model, and captures one frame per look state.
fn render_model(path: &Path, out_dir: &Path) -> Result<String, RenderError> {
    let managed_root =
        std::env::temp_dir().join(format!("rustuberv-vrm-render-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&managed_root);
    std::fs::create_dir_all(&managed_root)
        .map_err(|error| RenderError::Failed(format!("cannot create managed root: {error}")))?;

    let imported = vtuber_app::import::import_vrm(
        path,
        &managed_root,
        vtuber_app::import::DEFAULT_SIZE_LIMIT,
    )
    .map_err(|error| RenderError::Failed(format!("import failed: {error}")))?;
    let asset_id = vtuber_avatar::AvatarAssetId::new(&imported.id);
    let asset_path = UserAssetPath::avatar_model_path(&asset_id)
        .map_err(|error| RenderError::Failed(format!("asset path failed: {error}")))?;
    let expected = match imported.summary.generation {
        vtuber_app::import::VrmGeneration::Vrm0 => vtuber_avatar::ExpectedVrmGeneration::Vrm0,
        vtuber_app::import::VrmGeneration::Vrm1 => vtuber_avatar::ExpectedVrmGeneration::Vrm1,
    };
    let imported_avatar =
        ImportedAvatar::new(asset_id, asset_path, imported.name.clone(), expected);

    let managed_root_string = managed_root
        .to_str()
        .ok_or_else(|| RenderError::Failed("managed root is not valid UTF-8".to_owned()))?;
    let mut sources = AssetSourceBuilders::default();
    sources.insert(
        "user",
        AssetSourceBuilder::platform_default(managed_root_string, None),
    );

    let mut app = App::new();
    app.insert_resource(sources)
        .insert_resource(ManagedAvatar(imported_avatar))
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
                .set(RenderPlugin { ..default() })
                .disable::<PipelinedRenderingPlugin>()
                .disable::<bevy::log::LogPlugin>()
                .disable::<bevy::winit::WinitPlugin>(),
        )
        .add_plugins(VtuberAvatarPlugin)
        .add_systems(Startup, emit_load)
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

    let deadline = Instant::now() + Duration::from_secs(60);
    for _ in 0..WARMUP_FRAMES {
        app.update();
    }
    let off = take_frame(&mut app, deadline)?;

    set_look(&mut app, true, 1.0);
    for _ in 0..SETTLE_FRAMES {
        app.update();
    }
    let on = take_frame(&mut app, deadline)?;

    set_look(&mut app, false, 1.0);
    for _ in 0..SETTLE_FRAMES {
        app.update();
    }
    let off_restored = take_frame(&mut app, deadline)?;

    set_look(&mut app, true, 1.0);
    for _ in 0..SETTLE_FRAMES {
        app.update();
    }
    let on_again = take_frame(&mut app, deadline)?;

    set_look(&mut app, true, 0.0);
    for _ in 0..SETTLE_FRAMES {
        app.update();
    }
    let strength_zero = take_frame(&mut app, deadline)?;

    write_frame(&out_dir.join(format!("{name}.off.bgra")), &off)?;
    write_frame(&out_dir.join(format!("{name}.on.bgra")), &on)?;
    write_frame(
        &out_dir.join(format!("{name}.off-restored.bgra")),
        &off_restored,
    )?;
    write_frame(&out_dir.join(format!("{name}.on-again.bgra")), &on_again)?;
    write_frame(
        &out_dir.join(format!("{name}.strength-zero.bgra")),
        &strength_zero,
    )?;
    write_png(&out_dir.join(format!("{name}.off.png")), &off)?;
    write_png(&out_dir.join(format!("{name}.on.png")), &on)?;
    write_png(
        &out_dir.join(format!("{name}.off-restored.png")),
        &off_restored,
    )?;
    write_png(&out_dir.join(format!("{name}.on-again.png")), &on_again)?;
    write_png(
        &out_dir.join(format!("{name}.strength-zero.png")),
        &strength_zero,
    )?;

    let difference = mean_absolute_difference(&off.data, &on.data);
    let off_restore_difference = mean_absolute_difference(&off.data, &off_restored.data);
    let on_repeat_difference = mean_absolute_difference(&on.data, &on_again.data);
    let strength_zero_difference = mean_absolute_difference(&off.data, &strength_zero.data);
    let summary = format!(
        "{name}: off_mean={:.2} on_mean={:.2} mean_abs_diff={difference:.3} \
         off_restore_diff={off_restore_difference:.3} on_repeat_diff={on_repeat_difference:.3} \
         strength_zero_diff={strength_zero_difference:.3} off_opaque={} on_opaque={}\n  \
         probes off={:?} on={:?} off_restored={:?} strength_zero={:?}\n",
        mean(&off.data),
        mean(&on.data),
        opaque_pixels(&off.data),
        opaque_pixels(&on.data),
        probes(&off.data),
        probes(&on.data),
        probes(&off_restored.data),
        probes(&strength_zero.data)
    );
    let _ = std::fs::remove_dir_all(&managed_root);
    if difference < 0.5 {
        return Err(RenderError::Failed(format!(
            "switching the look on did not change the rendering (mean_abs_diff={difference:.3})"
        )));
    }
    if off_restore_difference >= 0.5 {
        return Err(RenderError::Failed(format!(
            "OFF after ON did not restore the original output (mean_abs_diff={off_restore_difference:.3})"
        )));
    }
    if strength_zero_difference >= 0.5 {
        return Err(RenderError::Failed(format!(
            "strength 0 did not restore the original output (mean_abs_diff={strength_zero_difference:.3})"
        )));
    }
    if on_repeat_difference >= 0.5 {
        return Err(RenderError::Failed(format!(
            "reapplying ON changed the output again (mean_abs_diff={on_repeat_difference:.3})"
        )));
    }
    Ok(summary)
}

#[derive(Resource)]
struct ManagedAvatar(ImportedAvatar);

fn emit_load(mut requests: MessageWriter<LoadImportedAvatarRequest>, avatar: Res<ManagedAvatar>) {
    requests.write(LoadImportedAvatarRequest {
        request_id: 1,
        imported: avatar.0.clone(),
    });
}

/// A fixed upper-body camera so every model is framed the same deterministic
/// way; the framing solve would otherwise depend on a window viewport.
///
/// The camera the plugin spawns is reused instead of adding a second one: the
/// output camera mirrors the single viewport camera, and two would break it.
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

fn set_look(app: &mut App, enabled: bool, strength: f32) {
    app.world_mut()
        .resource_mut::<vtuber_avatar::AvatarLookSettings>()
        .0 = vtuber_avatar::RichLookSettings { enabled, strength };
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

fn write_frame(path: &Path, frame: &VideoOutputFrame) -> Result<(), RenderError> {
    std::fs::write(path, &frame.data)
        .map_err(|error| RenderError::Failed(format!("cannot write {}: {error}", path.display())))
}

/// Writes the frame as a PNG so the render can be inspected as an image.
fn write_png(path: &Path, frame: &VideoOutputFrame) -> Result<(), RenderError> {
    let mut rgba = Vec::with_capacity(frame.data.len());
    for pixel in frame.data.as_chunks::<4>().0 {
        rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
    }
    let buffer = image::RgbaImage::from_raw(WIDTH, HEIGHT, rgba)
        .ok_or_else(|| RenderError::Failed("frame size does not match the profile".to_owned()))?;
    buffer.save(path).map_err(|error| {
        RenderError::Failed(format!("cannot write {}: {error}", path.display()))
    })
}

/// RGBA at the face, jacket and legs so a missing material is visible.
fn probes(data: &[u8]) -> Vec<[u8; 4]> {
    // `render_model` inserts the WIDTH×HEIGHT BGRA output profile, so the
    // frame buffer is exactly WIDTH×HEIGHT×4 bytes and every probe
    // coordinate below is inside it.
    #[allow(clippy::indexing_slicing)]
    let at = |x: u32, y: u32| {
        let index = ((y * WIDTH + x) * 4) as usize;
        [data[index], data[index + 1], data[index + 2], data[index + 3]]
    };
    vec![at(128, 55), at(128, 130), at(128, 215)]
}

fn opaque_pixels(data: &[u8]) -> usize {
    data.as_chunks::<4>().0.iter().filter(|pixel| pixel[3] > 0).count()
}

fn mean(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().map(|value| f64::from(*value)).sum::<f64>() / data.len() as f64
}

fn mean_absolute_difference(left: &[u8], right: &[u8]) -> f64 {
    if left.is_empty() || left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / left.len() as f64
}
