//! Guided MediaPipe neutral-relative pose probe.
//!
//! Prompts are written to stderr so `--json` keeps stdout machine-readable.
//! The probe uses the same worker-owned `MediaPipeRuntime` as the production
//! inference boundary and records bounded per-phase transform samples.

#[cfg(target_os = "windows")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(target_os = "windows")]
use std::sync::{Arc, Mutex};
#[cfg(target_os = "windows")]
use std::time::Duration;

#[cfg(target_os = "windows")]
use vtuber_core::{
    CameraFaceTransform, FaceTrackingOutcome, HeadPose, LatestSlot, ReadResult, StopToken,
    VideoFrame,
};
#[cfg(target_os = "windows")]
use vtuber_inference::FaceTrackingInference;
#[cfg(target_os = "windows")]
use vtuber_inference::backend::mediapipe::{MediaPipeRuntime, TASK_BUNDLE_FILE};
#[cfg(target_os = "windows")]
use vtuber_tracking::relative_pose;

#[cfg(target_os = "windows")]
const PHASE_COUNT: usize = 7;
#[cfg(target_os = "windows")]
const MAX_PHASE_SAMPLES: usize = 96;
#[cfg(target_os = "windows")]
const COUNTDOWN: Duration = Duration::from_secs(3);
#[cfg(target_os = "windows")]
const COLLECTION: Duration = Duration::from_secs(3);
#[cfg(target_os = "windows")]
const FRAME_WAIT: Duration = Duration::from_millis(100);
#[cfg(target_os = "windows")]
const SIGN_THRESHOLD_RAD: f32 = 0.05;

#[cfg(target_os = "windows")]
const PHASES: [PhaseDefinition; PHASE_COUNT] = [
    PhaseDefinition {
        name: "neutral",
        instruction: "face the camera naturally and relax",
    },
    PhaseDefinition {
        name: "image_right",
        instruction: "turn your face toward the right side of the unmirrored camera image (your own left)",
    },
    PhaseDefinition {
        name: "image_left",
        instruction: "turn your face toward the left side of the unmirrored camera image (your own right)",
    },
    PhaseDefinition {
        name: "chin_up",
        instruction: "raise your chin",
    },
    PhaseDefinition {
        name: "chin_down",
        instruction: "nod your chin down only 3-5 degrees; keep your nose, both eyes, and mouth visible",
    },
    PhaseDefinition {
        name: "image_clockwise",
        instruction: "tilt so the top of your head moves toward the right side of the unmirrored camera image (image-clockwise)",
    },
    PhaseDefinition {
        name: "image_counter_clockwise",
        instruction: "tilt so the top of your head moves toward the left side of the unmirrored camera image (image-counter-clockwise)",
    },
];

/// Runs the guided pose probe.
pub fn run(args: &[String]) -> Result<(), String> {
    let options = Options::parse(args)?;
    if options.help {
        print_help();
        return Ok(());
    }
    if !options.guided {
        return Err("--guided is required for the sign proof".into());
    }

    #[cfg(target_os = "windows")]
    {
        run_windows(options)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = options;
        Err("mediapipe-pose-probe requires Windows MSMF".into())
    }
}

/// Prints command usage.
pub fn print_help() {
    println!("mediapipe-pose-probe - guided Windows MediaPipe pose sign proof");
    println!();
    println!("USAGE:");
    println!("  cargo run -p xtask -- mediapipe-pose-probe --camera 0 --guided --json");
    println!();
    println!("OPTIONS:");
    println!("  --camera <id-or-index>   MSMF camera descriptor id or enumeration index");
    println!("  --project-root <path>    Workspace root (default: current directory)");
    println!("  --guided                 Run the seven-phase sign protocol");
    println!("  --json                   Emit one bounded JSON summary");
    println!("  -h, --help               Show this help");
}

#[derive(Debug)]
struct Options {
    camera: Option<String>,
    project_root: PathBuf,
    guided: bool,
    json: bool,
    help: bool,
}

impl Options {
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Self {
            camera: None,
            project_root: std::env::current_dir()
                .map_err(|error| format!("cannot resolve project root: {error}"))?,
            guided: false,
            json: false,
            help: false,
        };
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "-h" | "--help" => options.help = true,
                "--guided" => options.guided = true,
                "--json" => options.json = true,
                "--camera" => {
                    index += 1;
                    options.camera = Some(required_value(args, index, "--camera")?);
                }
                "--project-root" => {
                    index += 1;
                    options.project_root =
                        PathBuf::from(required_value(args, index, "--project-root")?);
                }
                other => return Err(format!("unknown option `{other}`")),
            }
            index += 1;
        }
        Ok(options)
    }
}

fn required_value(args: &[String], index: usize, option: &str) -> Result<String, String> {
    args.get(index)
        .cloned()
        .filter(|value| !value.starts_with('-'))
        .ok_or_else(|| format!("{option} requires a value"))
}

#[cfg(target_os = "windows")]
#[derive(Clone, Copy, Debug)]
struct PhaseDefinition {
    name: &'static str,
    instruction: &'static str,
}

#[cfg(target_os = "windows")]
#[derive(Clone, Debug, Default)]
struct PhaseSamples {
    frames_seen: u64,
    transforms: Vec<CameraFaceTransform>,
    no_face_count: u64,
}

#[cfg(target_os = "windows")]
#[derive(Clone, Debug)]
struct ProbeData {
    phases: [PhaseSamples; PHASE_COUNT],
}

#[cfg(target_os = "windows")]
impl Default for ProbeData {
    fn default() -> Self {
        Self {
            phases: std::array::from_fn(|_| PhaseSamples::default()),
        }
    }
}

#[cfg(target_os = "windows")]
#[derive(Debug)]
struct ProbeWorkerOutput {
    data: ProbeData,
    library_source: String,
    failure: Option<String>,
}

#[cfg(target_os = "windows")]
fn run_windows(options: Options) -> Result<(), String> {
    use std::sync::atomic::{AtomicI8, Ordering};
    use std::thread;
    use std::time::Instant;

    use vtuber_camera::backend::msmf::MsmfBackend;
    use vtuber_camera::device::CameraBackend;
    use vtuber_camera::{CameraRequest, CaptureController};
    use vtuber_core::{WorkerHandle, WorkerResult};

    let task_path = options
        .project_root
        .join("assets")
        .join("models")
        .join(TASK_BUNDLE_FILE);
    let devices = MsmfBackend::new()
        .enumerate()
        .map_err(|error| format!("camera enumeration failed: {error}"))?;
    let device = choose_camera(&devices, options.camera.as_deref())?;

    let mut capture = CaptureController::new();
    capture
        .start_worker(MsmfBackend::new())
        .map_err(|error| format!("capture worker start failed: {error}"))?;
    if let Err(error) = capture.select_and_start(device.clone(), CameraRequest::default()) {
        let _ = capture.shutdown();
        return Err(format!("camera start failed: {error}"));
    }

    if !options.json {
        eprintln!("backend=mediapipe-face-landmarker");
        eprintln!("mediapipe_version={}", mediapipe::MEDIAPIPE_VERSION);
        eprintln!("camera={device}");
        eprintln!(
            "protocol=neutral,image-right,image-left,up,down,image-clockwise,image-counter-clockwise"
        );
    }

    // Arm neutral before the worker starts warming up.  Neutral is a baseline
    // capture phase, not a classifier, so discarding startup/countdown frames
    // can make a valid camera view look like a missing neutral sample.
    let phase = Arc::new(AtomicI8::new(0));
    let data = Arc::new(Mutex::new(ProbeData::default()));
    let frame_slot = capture.frame_slot();
    let frame_deadline = Instant::now() + Duration::from_secs(5);
    while frame_slot.generation() == 0 && Instant::now() < frame_deadline {
        if capture.worker_finished() {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }
    if frame_slot.generation() == 0 {
        let state = capture.state();
        let metrics = capture.metrics();
        let _ = capture.shutdown();
        return Err(format!(
            "camera produced no frames before pose probe start (state={state:?}, frames_captured={}, last_error={:?})",
            metrics.frames_captured, metrics.last_error
        ));
    }
    let worker = WorkerHandle::spawn("mediapipe-pose-probe", {
        let frame_slot = Arc::clone(&frame_slot);
        let phase = Arc::clone(&phase);
        let data = Arc::clone(&data);
        move |stop| run_worker(&task_path, frame_slot, phase, data, stop)
    });

    thread::sleep(Duration::from_secs(2));
    for (index, definition) in PHASES.iter().enumerate() {
        prompt_phase(*definition);
        phase.store(index as i8, Ordering::Release);
        thread::sleep(COLLECTION);
        phase.store(-1, Ordering::Release);
        eprintln!("  collected {}", definition.name);
    }

    worker.stop();
    let worker_result = worker.join();
    let capture_state = capture.state();
    let capture_metrics = capture.metrics();
    let frame_generation = frame_slot.generation();
    let _ = capture.shutdown();
    let output = match worker_result {
        WorkerResult::Completed(output) => output,
        WorkerResult::Panicked => return Err("pose probe worker panicked".into()),
        WorkerResult::SpawnFailed => return Err("pose probe worker failed to spawn".into()),
    };
    if let Some(failure) = output.failure {
        return Err(format!("MediaPipe pose probe failed: {failure}"));
    }
    let report = build_report(output.data, output.library_source).map_err(|error| {
        format!(
            "{error}; capture_state={capture_state:?}, capture_frames_captured={}, capture_last_error={:?}, frame_generation={frame_generation}",
            capture_metrics.frames_captured, capture_metrics.last_error
        )
    })?;
    print_report(&report, options.json);
    if !report.signs_pass {
        return Err("guided pose sign proof failed; inspect the phase medians before changing basis mapping".into());
    }
    Ok(())
}

#[cfg(target_os = "windows")]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn run_worker(
    task_path: &Path,
    frame_slot: Arc<LatestSlot<VideoFrame>>,
    phase: Arc<std::sync::atomic::AtomicI8>,
    data: Arc<Mutex<ProbeData>>,
    stop: StopToken,
) -> ProbeWorkerOutput {
    let library_source = mediapipe::loader::lib()
        .map(|library| library_source_name(&library.source))
        .unwrap_or_else(|_| "unknown".into());
    let mut runtime = match MediaPipeRuntime::from_task_path(task_path) {
        Ok(runtime) => runtime,
        Err(error) => {
            return ProbeWorkerOutput {
                data: data.lock().map(|guard| guard.clone()).unwrap_or_default(),
                library_source,
                failure: Some(error.to_string()),
            };
        }
    };
    let mut generation = 0;
    while !stop.is_stopped() {
        let Some(read) = frame_slot.wait_read_after(generation, FRAME_WAIT) else {
            continue;
        };
        let frame = match read {
            ReadResult::New(frame) => frame,
            ReadResult::Closed => break,
        };
        generation = frame_slot.generation();
        let phase_index = phase.load(std::sync::atomic::Ordering::Acquire);
        if phase_index < 0 {
            continue;
        }
        let index = phase_index as usize;
        if index >= PHASE_COUNT {
            continue;
        }
        match runtime.infer_face_tracking(&frame) {
            Ok(FaceTrackingOutcome::Face(sample)) => {
                if let Ok(mut guard) = data.lock() {
                    let phase = &mut guard.phases[index];
                    phase.frames_seen += 1;
                    let samples = &mut phase.transforms;
                    if samples.len() < MAX_PHASE_SAMPLES {
                        samples.push(sample.camera_to_face);
                    }
                }
            }
            Ok(FaceTrackingOutcome::NoFace { .. }) => {
                if let Ok(mut guard) = data.lock() {
                    let phase = &mut guard.phases[index];
                    phase.frames_seen += 1;
                    phase.no_face_count += 1;
                }
            }
            Err(error) => {
                return ProbeWorkerOutput {
                    data: data.lock().map(|guard| guard.clone()).unwrap_or_default(),
                    library_source,
                    failure: Some(error.to_string()),
                };
            }
        }
    }
    ProbeWorkerOutput {
        data: data.lock().map(|guard| guard.clone()).unwrap_or_default(),
        library_source,
        failure: None,
    }
}

#[cfg(target_os = "windows")]
#[derive(Clone, Debug)]
struct PhaseReport {
    name: &'static str,
    frames_seen: u64,
    samples: usize,
    no_face_count: u64,
    raw_transform: CameraFaceTransform,
    pose: HeadPose,
    sign_pass: bool,
}

#[cfg(target_os = "windows")]
#[derive(Clone, Debug)]
struct ProbeReport {
    library_source: String,
    phases: Vec<PhaseReport>,
    signs_pass: bool,
}

#[cfg(target_os = "windows")]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn build_report(data: ProbeData, library_source: String) -> Result<ProbeReport, String> {
    let neutral = median_transform(&data.phases[0]).ok_or_else(|| {
        format!(
            "neutral phase produced no valid face samples; {}",
            phase_diagnostics(&data)
        )
    })?;
    let mut phases = Vec::with_capacity(PHASE_COUNT);
    for (index, definition) in PHASES.iter().enumerate() {
        let phase = &data.phases[index];
        let raw_transform = median_transform(phase).unwrap_or(neutral);
        let pose = relative_pose(neutral, raw_transform)
            .map_err(|error| format!("{} relative pose failed: {error}", definition.name))?;
        let sign_pass = expected_sign(index, pose);
        phases.push(PhaseReport {
            name: definition.name,
            frames_seen: phase.frames_seen,
            samples: phase.transforms.len(),
            no_face_count: phase.no_face_count,
            raw_transform,
            pose,
            sign_pass,
        });
    }
    let signs_pass = phases.iter().all(|phase| phase.sign_pass);
    Ok(ProbeReport {
        library_source,
        phases,
        signs_pass,
    })
}

#[cfg(target_os = "windows")]
fn phase_diagnostics(data: &ProbeData) -> String {
    PHASES
        .iter()
        .zip(&data.phases)
        .map(|(definition, phase)| {
            format!(
                "{}:frames_seen={},face_samples={},no_face_count={}",
                definition.name,
                phase.frames_seen,
                phase.transforms.len(),
                phase.no_face_count
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(target_os = "windows")]
fn median_transform(phase: &PhaseSamples) -> Option<CameraFaceTransform> {
    let first = *phase.transforms.first()?;
    let mut quaternion = [0.0; 4];
    let mut translation = [0.0; 3];
    for transform in &phase.transforms {
        let sign = if dot4(transform.rotation_xyzw, first.rotation_xyzw) < 0.0 {
            -1.0
        } else {
            1.0
        };
        for (slot, value) in quaternion.iter_mut().zip(transform.rotation_xyzw) {
            *slot += value * sign;
        }
        for (slot, value) in translation.iter_mut().zip(transform.translation_xyz) {
            *slot += value;
        }
    }
    let count = phase.transforms.len() as f32;
    for value in &mut quaternion {
        *value /= count;
    }
    let norm = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return None;
    }
    for value in &mut quaternion {
        *value /= norm;
    }
    for value in &mut translation {
        *value /= count;
    }
    Some(CameraFaceTransform {
        rotation_xyzw: quaternion,
        translation_xyz: translation,
    })
}

#[cfg(target_os = "windows")]
fn dot4(left: [f32; 4], right: [f32; 4]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2] + left[3] * right[3]
}

#[cfg(target_os = "windows")]
fn expected_sign(index: usize, pose: HeadPose) -> bool {
    match index {
        0 => {
            pose.yaw_rad.abs() < SIGN_THRESHOLD_RAD
                && pose.pitch_rad.abs() < SIGN_THRESHOLD_RAD
                && pose.roll_rad.abs() < SIGN_THRESHOLD_RAD
        }
        1 => pose.yaw_rad > SIGN_THRESHOLD_RAD,
        2 => pose.yaw_rad < -SIGN_THRESHOLD_RAD,
        3 => pose.pitch_rad > SIGN_THRESHOLD_RAD,
        4 => pose.pitch_rad < -SIGN_THRESHOLD_RAD,
        5 => pose.roll_rad > SIGN_THRESHOLD_RAD,
        6 => pose.roll_rad < -SIGN_THRESHOLD_RAD,
        _ => false,
    }
}

#[cfg(target_os = "windows")]
fn prompt_phase(phase: PhaseDefinition) {
    eprintln!("\n[{}] {}", phase.name, phase.instruction);
    if phase.name != "neutral" {
        eprintln!("  first return to a neutral forward-facing pose and hold it");
        std::thread::sleep(Duration::from_secs(1));
    }
    for remaining in (1..=COUNTDOWN.as_secs()).rev() {
        eprint!("  starting in {remaining}...\r");
        std::thread::sleep(Duration::from_secs(1));
    }
    eprintln!("  recording for {} seconds", COLLECTION.as_secs());
}

#[cfg(target_os = "windows")]
fn print_report(report: &ProbeReport, json: bool) {
    if json {
        let phases = report
            .phases
            .iter()
            .map(|phase| {
                format!(
                    "{{\"name\":\"{}\",\"frames_seen\":{},\"samples\":{},\"no_face_count\":{},\"raw_rotation_xyzw\":[{:.6},{:.6},{:.6},{:.6}],\"raw_translation_xyz\":[{:.6},{:.6},{:.6}],\"yaw_rad\":{:.6},\"pitch_rad\":{:.6},\"roll_rad\":{:.6},\"sign_pass\":{}}}",
                    phase.name,
                    phase.frames_seen,
                    phase.samples,
                    phase.no_face_count,
                    phase.raw_transform.rotation_xyzw[0],
                    phase.raw_transform.rotation_xyzw[1],
                    phase.raw_transform.rotation_xyzw[2],
                    phase.raw_transform.rotation_xyzw[3],
                    phase.raw_transform.translation_xyz[0],
                    phase.raw_transform.translation_xyz[1],
                    phase.raw_transform.translation_xyz[2],
                    phase.pose.yaw_rad,
                    phase.pose.pitch_rad,
                    phase.pose.roll_rad,
                    phase.sign_pass
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{{\"backend\":\"mediapipe-face-landmarker\",\"mediapipe_version\":\"{}\",\"native_library_source\":\"{}\",\"signs_pass\":{},\"phases\":[{}]}}",
            mediapipe::MEDIAPIPE_VERSION,
            report.library_source,
            report.signs_pass,
            phases
        );
    } else {
        println!("backend=mediapipe-face-landmarker");
        println!("mediapipe_version={}", mediapipe::MEDIAPIPE_VERSION);
        println!("native_library_source={}", report.library_source);
        for phase in &report.phases {
            println!(
                "phase={} frames_seen={} samples={} no_face={} yaw_rad={:.4} pitch_rad={:.4} roll_rad={:.4} sign_pass={}",
                phase.name,
                phase.frames_seen,
                phase.samples,
                phase.no_face_count,
                phase.pose.yaw_rad,
                phase.pose.pitch_rad,
                phase.pose.roll_rad,
                phase.sign_pass
            );
        }
        println!("signs_pass={}", report.signs_pass);
    }
}

#[cfg(target_os = "windows")]
fn choose_camera(
    devices: &[vtuber_camera::CameraDescriptor],
    requested: Option<&str>,
) -> Result<vtuber_camera::CameraDescriptor, String> {
    let Some(requested) = requested else {
        return devices
            .first()
            .cloned()
            .ok_or_else(|| "no MSMF camera found".into());
    };
    if let Some(device) = devices.iter().find(|device| device.id == requested) {
        return Ok(device.clone());
    }
    let index = requested
        .parse::<usize>()
        .map_err(|_| format!("camera `{requested}` is not a descriptor id or numeric index"))?;
    devices
        .get(index)
        .cloned()
        .ok_or_else(|| format!("camera index {index} is not available"))
}

#[cfg(target_os = "windows")]
fn library_source_name(source: &mediapipe::loader::LibrarySource) -> String {
    match source {
        mediapipe::loader::LibrarySource::Env(_) => "environment override".into(),
        mediapipe::loader::LibrarySource::Cache(_) => "verified cache".into(),
        mediapipe::loader::LibrarySource::Downloaded(_) => "official PyPI wheel download".into(),
    }
}
