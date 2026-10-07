//! Bounded recorded-camera run through the desktop plugins, including inference.

use std::{collections::BTreeMap, fs, io::Write, path::PathBuf, time::Instant};

use bevy::{
    asset::io::{AssetSourceBuilder, AssetSourceBuilders},
    prelude::*,
};
use bevy_egui::EguiPlugin;
use bevy_vrm1::prelude::{HipsBoneEntity, RestGlobalTransform, RestTransform};
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
    let Some([rgb, model, out, threads, presentation]) = args.get(..5) else {
        return Err("usage: cargo xtask tracking-replay <640x360-30fps.rgb> <model.vrm> <new-output-directory> <logical-cpus> <vsync|uncapped> [same-input-model.vrm]".into());
    };
    if args.len() > 6 {
        return Err("at most one same-input comparison model is supported".into());
    }
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
    let second_model = args
        .get(5)
        .map(|model| {
            vtuber_app::import::import_vrm(model, &managed, vtuber_app::import::DEFAULT_SIZE_LIMIT)
                .map_err(|e| e.to_string())
        })
        .transpose()?;
    let paired = second_model.is_some();
    let measurement_out = if paired {
        out.join("live")
    } else {
        out.clone()
    };
    let playback_out = out.join("same-input");
    if paired {
        fs::create_dir(&measurement_out).map_err(|e| e.to_string())?;
        fs::create_dir(&playback_out).map_err(|e| e.to_string())?;
    }
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
            out: measurement_out,
            launched: Instant::now(),
            frame_started: Instant::now(),
            selected: false,
            previous: None,
            rows: Vec::new(),
            timestamps: BTreeMap::new(),
            previous_arm: None,
            previous_face: 0,
            previous_pose: 0,
            next_picture: 2.0,
            finished: false,
            pictures: Vec::new(),
            threads,
            presentation: presentation.clone(),
            video_started_unix_seconds: None,
            initial_pose: None,
            initial_ready_frames: 0,
            second_model,
            playback_out: playback_out.clone(),
            controls: Vec::new(),
            playing: false,
            cursor: 0,
            playback_seconds: 0.0,
            playback_complete: false,
        })
        .add_systems(PostUpdate, paired_controls.before(vtuber_avatar::body_motion::update_body_tracking_position_input))
        .add_systems(First, |mut m: ResMut<Measurement>| m.frame_started = Instant::now())
        // Exit is published before the shell's ordered Last-stage shutdown.
        .add_systems(Last, measure.before(vtuber_app::diagnostics::sync_engine_diagnostics));
    let exit = app.run();
    let summary = if paired {
        playback_out.join("summary.json")
    } else {
        out.join("summary.json")
    };
    if !exit.is_success() || !summary.is_file() {
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
    initial_pose: Option<Vec<BonePose>>,
    initial_ready_frames: usize,
    second_model: Option<vtuber_app::import::ImportedModel>,
    playback_out: PathBuf,
    controls: Vec<RecordedControl>,
    playing: bool,
    cursor: usize,
    playback_seconds: f64,
    playback_complete: bool,
}

struct RecordedControl {
    seconds: f64,
    now: MonoTimeNs,
    delta: std::time::Duration,
    arm: TrackedArmControl,
    face: vtuber_avatar::ActiveControlFrame,
    mode: ArmPoseSourceKind,
}

// Store the exact face, thorax, shoulder, palm and finger input consumed by the
// first avatar. The comparison avatar receives those values in the same order,
// including the original per-frame time step, without another inference run.
fn paired_controls(world: &mut World) {
    world.resource_scope(|world, mut m: Mut<Measurement>| {
        // Idle sway has a generation-dependent seed. Freeze it for the static
        // initial photograph and for a comparison driven by identical input;
        // otherwise a reload introduces a different procedural body target.
        if !m.selected || m.second_model.is_some() || m.playing {
            world.resource_mut::<vtuber_avatar::LossIdleState>().reset();
        }
        if !m.selected {
            return;
        }
        let now = monotonic_now();
        if !m.playing {
            if m.second_model.is_some()
                && let Some(start) = m.source.started_at()
            {
                m.controls.push(RecordedControl {
                    seconds: now.0.saturating_sub(start.0) as f64 / 1e9,
                    now,
                    delta: world.resource::<Time<Real>>().delta(),
                    arm: *world.resource::<TrackedArmControl>(),
                    face: world
                        .resource::<vtuber_avatar::ActiveControlFrame>()
                        .clone(),
                    mode: world.resource::<ArmSourceSelection>().mode,
                });
            }
            return;
        }
        let Some(record) = m.controls.get(m.cursor) else {
            m.playback_complete = true;
            return;
        };
        let generation = world
            .resource::<vtuber_avatar::AvatarLifecycle>()
            .current_generation();
        let retime = |time: MonoTimeNs| {
            MonoTimeNs(now.0.saturating_sub(record.now.0.saturating_sub(time.0)))
        };
        let mut arm = record.arm;
        arm.generation = arm.generation.map(|_| generation);
        if let Some(frame) = &mut arm.frame {
            frame.captured_at = retime(frame.captured_at);
            frame.produced_at = retime(frame.produced_at);
        }
        let mut face = record.face.clone();
        face.generation = generation;
        if let Some(frame) = &mut face.frame {
            frame.captured_at = retime(frame.captured_at);
            frame.produced_at = retime(frame.produced_at);
        }
        *world.resource_mut::<TrackedArmControl>() = arm;
        *world.resource_mut::<vtuber_avatar::ActiveControlFrame>() = face;
        world.resource_mut::<ArmSourceSelection>().mode = record.mode;
        m.playback_seconds = record.seconds;
        m.cursor += 1;
        if let Some(next) = m.controls.get(m.cursor) {
            world.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(next.delta));
        }
    });
}

#[derive(Serialize)]
struct BonePose {
    name: String,
    position: [f32; 3],
    // Remove authored local/global rest axes using the VRM normalized-pose formula.
    normalized_rotation: [f32; 4],
    global_delta: [f32; 4],
}

fn bone_poses(world: &World, binding: AvatarBinding) -> Vec<BonePose> {
    let mut bones = vec![("head".to_owned(), binding.head)];
    for (name, entity) in [
        (
            "hips",
            world.get::<HipsBoneEntity>(binding.root).map(|b| b.0),
        ),
        ("spine", binding.spine),
        ("chest", binding.chest),
        ("upperChest", binding.upper_chest),
        ("neck", binding.neck),
    ] {
        if let Some(entity) = entity {
            bones.push((name.to_owned(), entity));
        }
    }
    for (side, arm) in [("left", binding.left_arm), ("right", binding.right_arm)] {
        let Some(arm) = arm else { continue };
        for (name, entity) in [
            ("Shoulder", arm.shoulder),
            ("UpperArm", Some(arm.upper_arm)),
            ("LowerArm", Some(arm.lower_arm)),
            ("Hand", Some(arm.hand)),
        ] {
            if let Some(entity) = entity {
                bones.push((format!("{side}{name}"), entity));
            }
        }
        for (name, finger) in [
            ("Thumb", arm.fingers.thumb),
            ("Index", arm.fingers.index),
            ("Middle", arm.fingers.middle),
            ("Ring", arm.fingers.ring),
            ("Little", arm.fingers.little),
        ] {
            for (joint, entity) in [
                ("Metacarpal", finger.metacarpal),
                ("Proximal", finger.proximal),
                ("Intermediate", finger.intermediate),
                ("Distal", finger.distal),
            ] {
                if let Some(entity) = entity {
                    bones.push((format!("{side}{name}{joint}"), entity));
                }
            }
        }
    }
    bones
        .into_iter()
        .filter_map(|(name, entity)| {
            let local = world.get::<Transform>(entity)?;
            let global = world.get::<GlobalTransform>(entity)?;
            let rest = world.get::<RestTransform>(entity)?;
            let rest_global = world.get::<RestGlobalTransform>(entity)?;
            let normalized = rest_global.rotation()
                * rest.rotation.inverse()
                * local.rotation
                * rest_global.rotation().inverse();
            Some(BonePose {
                name,
                position: global.translation().to_array(),
                normalized_rotation: normalized.normalize().to_array(),
                global_delta: (global.rotation() * rest_global.rotation().inverse())
                    .normalize()
                    .to_array(),
            })
        })
        .collect()
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
    bones: Vec<BonePose>,
    arm_observation: Option<serde_json::Value>,
    process_cpu_percent: Option<f32>,
    process_memory_gib: Option<f32>,
    screenshot_requested: bool,
}

fn measure(world: &mut World) {
    world.resource_scope(|world, mut m: Mut<Measurement>| {
        if m.finished {
            return;
        }
        match advance_replay_measurement(world, &mut m) {
            Ok(false) => {}
            Ok(true) => {
                if let Some(model) = m.second_model.take() {
                    if let Err(e) = finish(world, &m) {
                        eprintln!("Replay export failed: {e}");
                        world.write_message(AppExit::error());
                        return;
                    }
                    world
                        .resource_mut::<Orchestrator>()
                        .process_action(&UiAction::UnloadAvatar);
                    world
                        .resource_mut::<Orchestrator>()
                        .set_camera_list(Vec::new());
                    world
                        .resource_mut::<Orchestrator>()
                        .queue_imported_model(model);
                    m.out = m.playback_out.clone();
                    m.playing = true;
                    m.selected = false;
                    m.launched = Instant::now();
                    m.previous = None;
                    m.rows.clear();
                    m.timestamps.clear();
                    m.pictures.clear();
                    m.initial_pose = None;
                    m.initial_ready_frames = 0;
                    m.video_started_unix_seconds = None;
                    m.previous_arm = None;
                    m.previous_face = 0;
                    m.previous_pose = 0;
                    m.next_picture = 2.0;
                    return;
                }
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

/// Advance startup/capture selection, then collect timing, pose and image samples.
/// Return true once playback completes or reaches the recording duration;
/// return false while startup or measurement is still in progress.
fn advance_replay_measurement(world: &mut World, m: &mut Measurement) -> Result<bool, String> {
    if m.launched.elapsed().as_secs_f64() > m.source.duration().as_secs_f64() + 120.0 {
        return Err("avatar/capture startup timed out".into());
    }
    if let Some(e) = world.resource::<Orchestrator>().last_error() {
        return Err(format!("{e:?}"));
    }
    let generation = world
        .resource::<vtuber_avatar::AvatarLifecycle>()
        .current_generation();
    let ready = world.resource::<vtuber_avatar::AvatarLifecycle>().state()
        == vtuber_avatar::AvatarLifecycleState::Ready;
    let mut query = world.query_filtered::<(&AvatarBinding, &DynamicArmTargets, &UpperLimbSolveStatus), With<vtuber_avatar::ActiveAvatar>>();
    let current = query
        .iter(world)
        .find(|(b, _, _)| ready && b.generation == generation)
        .map(|(b, a, s)| (*b, *a, *s));
    if !m.selected
        && current.is_some_and(|(_, _, s)| matches!(s, UpperLimbSolveStatus::Feasible { .. }))
    {
        if m.initial_pose.is_none() {
            m.initial_ready_frames += 1;
            if m.initial_ready_frames < 96 {
                return Ok(false);
            }
            if let Some((binding, _, _)) = current {
                m.initial_pose = Some(bone_poses(world, binding));
                let path = m.out.join("avatar-initial.png");
                world
                    .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
                    .observe(
                        move |capture: On<bevy::render::view::screenshot::ScreenshotCaptured>,
                              mut measurement: ResMut<Measurement>| {
                            measurement
                                .pictures
                                .push((path.clone(), capture.image.clone()));
                        },
                    );
            }
            return Ok(false);
        }
        // Start the recording only after the initial rendered image has arrived.
        if m.pictures.is_empty() {
            return Ok(false);
        }
        if !m.playing {
            world
                .resource_mut::<Orchestrator>()
                .process_action(&UiAction::SelectCamera { index: 0 });
        } else if let Some(first) = m.controls.first() {
            world.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(first.delta));
        }
        m.selected = true;
    }
    if !m.selected {
        return Ok(false);
    }
    let now = monotonic_now();
    let seconds = if m.playing {
        if m.playback_complete {
            return Ok(true);
        }
        m.playback_seconds
    } else if let Some(start) = m.source.started_at() {
        now.0.saturating_sub(start.0) as f64 / 1e9
    } else {
        return Ok(false);
    };
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
        m.next_picture = if m.next_picture == 2.0 {
            5.0
        } else {
            m.next_picture + 5.0
        };
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
        bones: current
            .map(|(binding, _, _)| bone_poses(world, binding))
            .unwrap_or_default(),
        arm_observation: world.resource::<TrackedArmControl>().frame.map(|frame| {
            let sides: Vec<_> = [
                (frame.targets.left, frame.weights.left),
                (frame.targets.right, frame.weights.right),
            ]
            .into_iter()
            .map(|(target, weight)| {
                json!({
                    "wrist": target.map(|t| t.wrist),
                    "elbow": target.map(|t| t.elbow_pole),
                    "palm_normal": target.and_then(|t| t.palm_normal),
                    "palm_forward": target.and_then(|t| t.palm_forward),
                    "weight": [weight.wrist, weight.pole, weight.palm, weight.fingers],
                })
            })
            .collect();
            json!({"source_seq": frame.source_seq.0, "sides": sides})
        }),
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
    fs::write(
        m.out.join("initial-pose.json"),
        serde_json::to_vec_pretty(&m.initial_pose).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
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
    if (!m.playing && (last.face_frames == 0 || last.pose_hand_frames == 0))
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
        "input": if m.playing { "exact face/arm/thorax/finger controls and time steps from the live video run; no repeated inference" } else { "640x360 RGB8 30fps; paced file source; sensor/driver and NDI excluded" },
        "procedural_idle_frozen": !m.controls.is_empty() || m.playing,
        "latency_scope": "scheduled video capture to main-world admitted arm pose; not GPU presentation or photon latency",
        "fps": rows.len() as f64 * 1000.0 / if m.playing { rows.iter().map(|r| r.main_interval_ms).sum::<f64>() } else { frames_ms },
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
