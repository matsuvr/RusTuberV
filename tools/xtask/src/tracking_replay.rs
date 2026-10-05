//! Bounded recorded-camera run through the desktop plugins, including inference.

use std::{collections::BTreeMap, fs, io::Write, path::PathBuf, time::Instant};

use bevy::{
    asset::io::{AssetSourceBuilder, AssetSourceBuilders},
    prelude::*,
};
use bevy_egui::EguiPlugin;
use serde::Serialize;
use serde_json::json;
use vtuber_app::{
    actions::UiAction,
    capture_runtime::{CameraBackendKind, CaptureRuntime},
    diagnostics::DiagnosticsSnapshot,
    inference_runtime::{InferenceProjectRoot, InferenceRuntime},
    orchestrator::Orchestrator,
    pose_runtime::PoseRuntime,
    settings::AppSettings,
    tracking_file::{TRACKING_PROFILE_FILE_NAME, load_tracking_profile},
    ui::UiShellPlugin,
};
use vtuber_avatar::{
    ArmPoseSourceKind, ArmSourceSelection, AvatarBinding, DynamicArmTargets, StartupModelPath,
    TrackedArmControl, UpperLimbSolveStatus, VtuberAvatarPlugin,
};
use vtuber_camera::{device::CameraBackend, replay::ReplaySource};
use vtuber_core::{MonoTimeNs, monotonic_now};

const WARMUP_SECONDS: f64 = 5.0;

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let [rgb, model, out, threads, presentation] = args else {
        return Err("usage: cargo xtask tracking-replay <640x360-30fps.rgb> <model.vrm> <new-output-directory> <logical-cpus> <vsync|uncapped>".into());
    };
    let present_mode = match presentation.as_str() {
        // Match the desktop's Window default: strict FIFO, not adaptive VSync.
        "vsync" => bevy::window::PresentMode::Fifo,
        "uncapped" => bevy::window::PresentMode::AutoNoVsync,
        _ => return Err("presentation must be vsync or uncapped".into()),
    };
    let threads = threads
        .parse::<std::num::NonZeroUsize>()
        .map_err(|e| e.to_string())?
        .get();
    let source = ReplaySource::load(std::path::Path::new(rgb)).map_err(|e| e.to_string())?;
    if source.duration().as_secs_f64() <= WARMUP_SECONDS {
        return Err("recording must exceed the five-second measurement warmup".into());
    }
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let out = fs::canonicalize(out).map_err(|e| e.to_string())?;
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let managed = out.join("managed");
    fs::create_dir(&managed).map_err(|e| e.to_string())?;
    let imported =
        vtuber_app::import::import_vrm(model, &managed, vtuber_app::import::DEFAULT_SIZE_LIMIT)
            .map_err(|e| e.to_string())?;
    let model_id = imported.id.clone();
    let mut sources = AssetSourceBuilders::default();
    sources.insert(
        "user",
        AssetSourceBuilder::platform_default(
            managed.to_str().ok_or("non-Unicode asset directory")?,
            None,
        ),
    );
    let tracking =
        load_tracking_profile(&root.join(TRACKING_PROFILE_FILE_NAME)).map_err(|e| e.to_string())?;
    let mut settings = AppSettings::empty_at(out.join("settings.toml"));
    settings
        .set_arm_tracking_enabled(true)
        .map_err(|e| e.to_string())?;
    let pose = PoseRuntime::new(root.clone());
    let capture = CaptureRuntime::with_backend_and_pose_output(
        CameraBackendKind::Replay(source.clone()),
        Some(pose.frame_slot()),
    );
    let mut orchestrator = Orchestrator::new(managed);
    orchestrator.queue_imported_model(imported);
    orchestrator.set_camera_list(source.enumerate().map_err(|e| e.to_string())?);
    // Windows can report all host CPUs even with a restricted process affinity.
    // Match the pool budget to the independently enforced process CPU budget.
    let pools = bevy::app::TaskPoolOptions::with_num_threads(threads);
    let mut app = App::new();
    app.insert_resource(sources)
        .insert_resource(bevy::winit::WinitSettings::continuous())
        .add_plugins(
            DefaultPlugins
                .set(bevy::app::TaskPoolPlugin {
                    task_pool_options: pools,
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "RusTuberV recorded-camera measurement".into(),
                        present_mode,
                        // OS display traces require a visible, unoccluded window.
                        // Keep this bounded measurement above other applications.
                        window_level: bevy::window::WindowLevel::AlwaysOnTop,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins((
            bevy::diagnostic::FrameTimeDiagnosticsPlugin::default(),
            bevy::diagnostic::SystemInformationDiagnosticsPlugin,
        ))
        .add_plugins(EguiPlugin::default())
        .add_plugins(VtuberAvatarPlugin)
        .insert_resource(tracking.body)
        .insert_resource(ArmSourceSelection {
            mode: ArmPoseSourceKind::VirtualHandAnchor,
            profile: tracking.arm,
        })
        .insert_resource(settings)
        .insert_resource(InferenceProjectRoot(root))
        .insert_resource(pose)
        .insert_resource(capture)
        .add_plugins(UiShellPlugin)
        .insert_resource(orchestrator)
        .insert_resource(StartupModelPath(Some(model_id)))
        .insert_resource(Measurement {
            source,
            out: out.clone(),
            launched: Instant::now(),
            frame_started: Instant::now(),
            selected: false,
            previous: None,
            rows: Vec::new(),
            timestamps: BTreeMap::new(),
            previous_arm: None,
            previous_face: 0,
            previous_pose: 0,
            next_picture: 10.0,
            finished: false,
            pictures: Vec::new(),
            threads,
            presentation: presentation.clone(),
            video_started_unix_seconds: None,
        })
        .add_systems(First, |mut m: ResMut<Measurement>| m.frame_started = Instant::now())
        // Exit is published before the shell's ordered Last-stage shutdown.
        .add_systems(Last, measure.before(vtuber_app::diagnostics::sync_engine_diagnostics));
    let exit = app.run();
    if !exit.is_success() || !out.join("summary.json").is_file() {
        return Err(format!(
            "replay did not complete; inspect {}",
            out.display()
        ));
    }
    println!("Replay results: {}", out.display());
    Ok(())
}

#[derive(Resource)]
struct Measurement {
    source: ReplaySource,
    out: PathBuf,
    launched: Instant,
    frame_started: Instant,
    selected: bool,
    previous: Option<MonoTimeNs>,
    rows: Vec<Row>,
    timestamps: BTreeMap<u64, MonoTimeNs>,
    previous_arm: Option<u64>,
    previous_face: u64,
    previous_pose: u64,
    next_picture: f64,
    finished: bool,
    pictures: Vec<(PathBuf, Image)>,
    threads: usize,
    presentation: String,
    video_started_unix_seconds: Option<f64>,
}

#[derive(Serialize)]
struct Row {
    video_seconds: f64,
    frame_ms: f64,
    main_interval_ms: f64,
    main_world_ms: f64,
    face_inference_ms: Option<f64>,
    pose_hand_inference_ms: Option<f64>,
    face_frames: u64,
    face_no_face_frames: u64,
    pose_hand_frames: u64,
    arm_source_seq: Option<u64>,
    arm_first_apply_ms: Option<f64>,
    arm_displayed_age_ms: Option<f64>,
    solver: String,
    // Shoulder, elbow, wrist for left then right; final global transforms.
    joints: Vec<([f32; 3], [f32; 4])>,
    process_cpu_percent: Option<f32>,
    process_memory_gib: Option<f32>,
    screenshot_requested: bool,
}

fn measure(world: &mut World) {
    world.resource_scope(|world, mut m: Mut<Measurement>| {
        if m.finished {
            return;
        }
        match sample(world, &mut m) {
            Ok(false) => {}
            Ok(true) => {
                m.finished = true;
                match finish(world, &m) {
                    Ok(()) => {
                        world.write_message(AppExit::Success);
                    }
                    Err(e) => {
                        eprintln!("Replay export failed: {e}");
                        world.write_message(AppExit::error());
                    }
                }
            }
            Err(e) => {
                m.finished = true;
                eprintln!("Replay failed: {e}");
                world.write_message(AppExit::error());
            }
        }
    });
}

fn sample(world: &mut World, m: &mut Measurement) -> Result<bool, String> {
    if m.launched.elapsed().as_secs_f64() > m.source.duration().as_secs_f64() + 120.0 {
        return Err("avatar/capture startup timed out".into());
    }
    if let Some(e) = world.resource::<Orchestrator>().last_error() {
        return Err(format!("{e:?}"));
    }
    let mut query = world.query::<(&AvatarBinding, &DynamicArmTargets, &UpperLimbSolveStatus)>();
    let current = query.iter(world).next().map(|(b, a, s)| (*b, *a, *s));
    if !m.selected
        && current.is_some_and(|(_, _, s)| matches!(s, UpperLimbSolveStatus::Feasible { .. }))
    {
        world
            .resource_mut::<Orchestrator>()
            .process_action(&UiAction::SelectCamera { index: 0 });
        m.selected = true;
    }
    let Some(start) = m.source.started_at() else {
        return Ok(false);
    };
    let now = monotonic_now();
    let seconds = now.0.saturating_sub(start.0) as f64 / 1e9;
    if m.video_started_unix_seconds.is_none() {
        // Correlate external presentation traces with the video. Capture this
        // once; wall-clock reads and adjustments must not drive animation.
        m.video_started_unix_seconds = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_secs_f64()
                - seconds,
        );
    }
    if seconds >= m.source.duration().as_secs_f64() {
        return Ok(true);
    }
    let Some(previous) = m.previous.replace(now) else {
        return Ok(false);
    };
    if let Some(frame) = world.resource::<TrackedArmControl>().frame.as_ref() {
        m.timestamps.insert(frame.source_seq.0, frame.captured_at);
    }
    let face = world.resource::<InferenceRuntime>().status();
    let pose = world.resource::<PoseRuntime>().status();
    if let Some(e) = face.last_failure.as_ref().or(pose.last_failure.as_ref()) {
        return Err(format!("{e:?}"));
    }
    let arm_seq = current.and_then(|(_, a, _)| a.source_seq.map(|s| s.0));
    let age = arm_seq
        .and_then(|s| m.timestamps.get(&s))
        .map(|t| now.0.saturating_sub(t.0) as f64 / 1e6);
    let new_arm = arm_seq != m.previous_arm;
    m.previous_arm = arm_seq;
    // A completed no-face result is still an inference, not a skipped image.
    let face_completed = face.frames_processed + face.no_face_frames;
    let face_ms = (face_completed != m.previous_face)
        .then_some(face.last_inference_duration)
        .flatten()
        .map(|d| d.as_secs_f64() * 1000.0);
    let pose_ms = (pose.frames_processed != m.previous_pose)
        .then_some(pose.last_inference_duration)
        .flatten()
        .map(|d| d.as_secs_f64() * 1000.0);
    m.previous_face = face_completed;
    m.previous_pose = pose.frames_processed;
    let mut joints = Vec::new();
    if let Some((binding, _, _)) = current {
        for arm in [binding.left_arm, binding.right_arm].into_iter().flatten() {
            for entity in [arm.upper_arm, arm.lower_arm, arm.hand] {
                if let Some(t) = world.get::<GlobalTransform>(entity) {
                    joints.push((t.translation().to_array(), t.rotation().to_array()));
                }
            }
        }
    }
    let picture = seconds >= m.next_picture && seconds + 1.0 < m.source.duration().as_secs_f64();
    if picture {
        let path = m
            .out
            .join(format!("avatar-{:02}.png", m.next_picture as u32));
        world
            .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
            .observe(
                move |capture: On<bevy::render::view::screenshot::ScreenshotCaptured>,
                      mut measurement: ResMut<Measurement>| {
                    // PNG encoding is CPU work and must not contaminate this run.
                    measurement
                        .pictures
                        .push((path.clone(), capture.image.clone()));
                },
            );
        m.next_picture += 10.0;
    }
    let diagnostics = world.resource::<DiagnosticsSnapshot>();
    m.rows.push(Row {
        video_seconds: seconds,
        // Match Bevy's production FPS diagnostic: pipelined rendering sends
        // its clock to Time<Real>. Last-to-Last also includes varying main
        // work within the frame and is retained as a separate observation.
        frame_ms: world.resource::<Time<Real>>().delta_secs_f64() * 1000.0,
        main_interval_ms: now.0.saturating_sub(previous.0) as f64 / 1e6,
        main_world_ms: m.frame_started.elapsed().as_secs_f64() * 1000.0,
        face_inference_ms: face_ms,
        pose_hand_inference_ms: pose_ms,
        face_frames: face.frames_processed,
        face_no_face_frames: face.no_face_frames,
        pose_hand_frames: pose.frames_processed,
        arm_source_seq: arm_seq,
        arm_first_apply_ms: new_arm.then_some(age).flatten(),
        arm_displayed_age_ms: age,
        solver: current
            .map(|(_, _, s)| format!("{s:?}"))
            .unwrap_or_else(|| "Unbound".into()),
        joints,
        process_cpu_percent: diagnostics.process_cpu_usage,
        process_memory_gib: diagnostics.process_memory_gib,
        screenshot_requested: picture,
    });
    Ok(false)
}

fn distribution(values: impl Iterator<Item = f64>) -> serde_json::Value {
    let mut values: Vec<_> = values.collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    let percentile = |fraction: f64| {
        values
            .get(((values.len() - 1) as f64 * fraction).round() as usize)
            .copied()
    };
    json!({ "samples": values.len(), "mean": values.iter().sum::<f64>() / values.len() as f64, "p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99), "max": values.last() })
}

fn finish(world: &mut World, m: &Measurement) -> Result<(), String> {
    for (path, image) in &m.pictures {
        image
            .clone()
            .try_into_dynamic()
            .map_err(|e| e.to_string())?
            .save(path)
            .map_err(|e| e.to_string())?;
    }
    let mut trace = std::io::BufWriter::new(
        fs::File::create(m.out.join("frames.jsonl")).map_err(|e| e.to_string())?,
    );
    for row in &m.rows {
        serde_json::to_writer(&mut trace, row).map_err(|e| e.to_string())?;
        writeln!(trace).map_err(|e| e.to_string())?;
    }
    trace.flush().map_err(|e| e.to_string())?;
    let rows: Vec<_> = m
        .rows
        .iter()
        .filter(|r| r.video_seconds >= WARMUP_SECONDS)
        .collect();
    let frames_ms: f64 = rows.iter().map(|r| r.frame_ms).sum();
    let first = rows.first().ok_or("no measured frames")?;
    let last = rows.last().ok_or("no measured frames")?;
    if last.face_frames == 0
        || last.pose_hand_frames == 0
        || !rows.iter().any(|r| r.arm_displayed_age_ms.is_some())
    {
        return Err("incomplete pipeline: face, Pose/Hand and admitted tracked arms are required; partial trace retained".into());
    }
    let duration = last.video_seconds - first.video_seconds;
    let face_detected = last.face_frames - first.face_frames;
    let face_no_face = last.face_no_face_frames - first.face_no_face_frames;
    let adapter = world
        .get_resource::<bevy::render::renderer::RenderAdapterInfo>()
        .map(|info| {
            json!({
                "name": info.name, "backend": format!("{:?}", info.backend),
                "device_type": format!("{:?}", info.device_type), "driver": info.driver,
                "driver_info": info.driver_info,
            })
        });
    let window = world
        .query_filtered::<&Window, With<bevy::window::PrimaryWindow>>()
        .iter(world)
        .next()
        .map(|window| json!({
            "width": window.physical_width(), "height": window.physical_height(),
            "present_mode": format!("{:?}", window.present_mode),
            "window_level": format!("{:?}", window.window_level),
            "desired_maximum_frame_latency": window.desired_maximum_frame_latency.map(|n| n.get()),
        }));
    let summary = json!({
        "adapter": adapter, "window": window,
        "warmup_seconds": WARMUP_SECONDS, "duration_seconds": duration,
        "available_parallelism": std::thread::available_parallelism().ok().map(|v| v.get()),
        "task_thread_budget": m.threads,
        "presentation": m.presentation,
        "video_started_unix_seconds": m.video_started_unix_seconds,
        "video_duration_seconds": m.source.duration().as_secs_f64(),
        "task_pool_threads": {
            "compute": bevy::tasks::ComputeTaskPool::get().thread_num(),
            "async_compute": bevy::tasks::AsyncComputeTaskPool::get().thread_num(),
            "io": bevy::tasks::IoTaskPool::get().thread_num(),
        },
        "input": "640x360 RGB8 30fps; paced file source; sensor/driver and NDI excluded",
        "latency_scope": "scheduled video capture to main-world admitted arm pose; not GPU presentation or photon latency",
        "fps": rows.len() as f64 * 1000.0 / frames_ms,
        "frame_ms": distribution(rows.iter().map(|r| r.frame_ms)),
        "main_interval_ms": distribution(rows.iter().map(|r| r.main_interval_ms)),
        "wall_fps": rows.len() as f64 * 1000.0 / rows.iter().map(|r| r.main_interval_ms).sum::<f64>(),
        "main_world_ms": distribution(rows.iter().map(|r| r.main_world_ms)),
        "face_hz": (face_detected + face_no_face) as f64 / duration,
        "face_detected_hz": face_detected as f64 / duration,
        "face_no_face_frames": face_no_face,
        "pose_hand_hz": (last.pose_hand_frames - first.pose_hand_frames) as f64 / duration,
        "face_inference_ms": distribution(rows.iter().filter_map(|r| r.face_inference_ms)),
        "pose_hand_inference_ms": distribution(rows.iter().filter_map(|r| r.pose_hand_inference_ms)),
        "arm_first_apply_ms": distribution(rows.iter().filter_map(|r| r.arm_first_apply_ms)),
        "arm_displayed_age_ms": distribution(rows.iter().filter_map(|r| r.arm_displayed_age_ms)),
        "process_cpu_percent": distribution(rows.iter().filter_map(|r| r.process_cpu_percent.map(f64::from))),
        "process_memory_gib": distribution(rows.iter().filter_map(|r| r.process_memory_gib.map(f64::from))),
        "screenshot_frames_included": true,
        "arm_missing_frames": rows.iter().filter(|r| r.arm_source_seq.is_none()).count(),
    });
    fs::write(
        m.out.join("summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{summary}");
    Ok(())
}
