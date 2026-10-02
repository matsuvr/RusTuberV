//! Standalone Windows MediaPipe Face Landmarker gate.
//!
//! The smoke path uses the existing camera boundary, `vtuber-core`, and the
//! production `vtuber-inference` adapter, including task verification, pixel
//! conversion and validated output decoding. It does not construct Bevy or access a VRM.
//! The task is built and dropped in a supervised inference worker, while
//! camera frames cross only the capacity-one [`vtuber_core::LatestSlot`].

#[cfg(target_os = "windows")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(target_os = "windows")]
use std::sync::Arc;
use std::time::Duration;
#[cfg(target_os = "windows")]
use std::time::Instant;

#[cfg(target_os = "windows")]
use vtuber_core::{FaceTrackingOutcome, FrameSeq, LatestSlot, ReadResult, VideoFrame};
#[cfg(target_os = "windows")]
use vtuber_inference::backend::mediapipe::{
    MEDIAPIPE_VERSION, MediaPipeRuntime, native_library_source,
};
#[cfg(target_os = "windows")]
use vtuber_inference::{FaceTrackingInference, InferenceError, MediaPipeTask, MediaPipeTaskSource};

const MAX_TASK_DURATION: Duration = Duration::from_secs(24 * 60 * 60);
#[cfg(target_os = "windows")]
const FRAME_WAIT: Duration = Duration::from_millis(100);

/// Runs the standalone MediaPipe face gate.
pub fn run(args: &[String]) -> Result<(), String> {
    let options = Options::parse(args)?;
    if options.help {
        print_help();
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        run_windows(options)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = options;
        Err("mediapipe-face-smoke requires Windows MSMF".into())
    }
}

/// Prints the smoke command usage.
pub fn print_help() {
    println!("mediapipe-face-smoke - Windows MSMF MediaPipe Face Landmarker gate");
    println!();
    println!("USAGE:");
    println!("  cargo run -p xtask -- mediapipe-face-smoke [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --camera <id-or-index>   MSMF camera descriptor id or enumeration index");
    println!("  --duration <seconds>     Capture duration (default: 60)");
    println!("  --project-root <path>    Workspace root (default: current directory)");
    println!("  --json                   Emit one bounded JSON summary");
    println!("  -h, --help               Show this help");
    println!();
    println!(
        "Runs the production task/pixel/inference/output adapter; timings include that full call."
    );
}

#[derive(Debug)]
struct Options {
    camera: Option<String>,
    duration: Duration,
    project_root: PathBuf,
    json: bool,
    help: bool,
}

impl Options {
    #[expect(
        clippy::indexing_slicing,
        reason = "the loop condition bounds `index` by the argument length and `required_value` rejects a missing value argument"
    )]
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Self {
            camera: None,
            duration: Duration::from_secs(60),
            project_root: std::env::current_dir()
                .map_err(|error| format!("cannot resolve project root: {error}"))?,
            json: false,
            help: false,
        };
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "-h" | "--help" => options.help = true,
                "--json" => options.json = true,
                "--camera" => {
                    index += 1;
                    options.camera = Some(required_value(args, index, "--camera")?);
                }
                "--duration" => {
                    index += 1;
                    let value = required_value(args, index, "--duration")?;
                    let seconds = value
                        .parse::<u64>()
                        .map_err(|_| format!("invalid --duration value `{value}`"))?;
                    options.duration = Duration::from_secs(seconds);
                    if options.duration.is_zero() || options.duration > MAX_TASK_DURATION {
                        return Err("--duration must be between 1 second and 24 hours".into());
                    }
                }
                "--project-root" => {
                    index += 1;
                    options.project_root =
                        PathBuf::from(required_value(args, index, "--project-root")?);
                }
                other => return Err(format!("unknown mediapipe-face-smoke option `{other}`")),
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
#[derive(Clone, Debug, Default)]
struct SmokeStats {
    face_count: u64,
    no_face_count: u64,
    contract_failures: u64,
    inference_errors: u64,
    last_source_seq: Option<FrameSeq>,
    last_landmark_count: Option<usize>,
    last_blendshape_count: Option<usize>,
    last_matrix_count: Option<usize>,
    valid_matrix_count: u64,
    matrix_determinant: Option<f32>,
    matrix_orthogonality_error: Option<f32>,
    inference_durations: Vec<Duration>,
    first_result_at: Option<Instant>,
    last_result_at: Option<Instant>,
    backend_source: Option<String>,
    failure: Option<String>,
}

#[cfg(target_os = "windows")]
#[derive(Debug)]
struct WorkerOutput {
    stats: SmokeStats,
}

#[cfg(target_os = "windows")]
fn run_windows(options: Options) -> Result<(), String> {
    use std::thread;

    use vtuber_camera::backend::msmf::MsmfBackend;
    use vtuber_camera::device::CameraBackend;
    use vtuber_camera::{CameraRequest, CaptureController};
    use vtuber_core::{WorkerHandle, WorkerResult};

    let task_path = options
        .project_root
        .join("assets")
        .join("models")
        .join(MediaPipeTask::Face.file());

    let devices = MsmfBackend::new()
        .enumerate()
        .map_err(|error| format!("camera enumeration failed: {error}"))?;
    let device = choose_camera(&devices, options.camera.as_deref())?;

    let mut capture = CaptureController::new();
    capture
        .start_worker(MsmfBackend::new())
        .map_err(|error| format!("capture worker start failed: {error}"))?;
    if let Err(error) = capture.select_and_start(device.clone(), CameraRequest::default()) {
        if let Err(shutdown_error) = capture.shutdown() {
            eprintln!("capture shutdown failed: {shutdown_error}");
        }
        return Err(format!("camera start failed: {error}"));
    }

    if !options.json {
        println!("backend=mediapipe-face-landmarker");
        println!("mediapipe_version={MEDIAPIPE_VERSION}");
        println!("camera={device}");
        println!("task_bundle={}", MediaPipeTask::Face.file());
    }

    let frame_slot = capture.frame_slot();
    let worker_frame_slot = Arc::clone(&frame_slot);
    let worker_task_path = task_path.clone();
    let inference_worker = WorkerHandle::spawn("mediapipe-face-worker", move |stop| {
        run_worker(&worker_task_path, worker_frame_slot, stop)
    })
    .map_err(|error| format!("MediaPipe worker spawn failed: {error}"))?;

    let started = Instant::now();
    let mut next_restart = restart_interval(options.duration);
    let mut restart_count = 0u8;
    while started.elapsed() < options.duration {
        if restart_count < 3 && started.elapsed() >= next_restart {
            let restart = capture.stop().and_then(|()| {
                thread::sleep(Duration::from_millis(150));
                capture.select_and_start(device.clone(), CameraRequest::default())
            });
            if let Err(error) = restart {
                inference_worker.stop();
                let inference_result = inference_worker.join();
                if let Err(shutdown_error) = capture.shutdown() {
                    eprintln!("capture shutdown failed: {shutdown_error}");
                }
                return Err(format!(
                    "camera Stop/Start {} failed: {error}; inference={inference_result:?}",
                    restart_count + 1
                ));
            }
            restart_count += 1;
            next_restart += restart_interval(options.duration);
        }
        if inference_worker.is_finished() {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }

    // Stop inference before capture so no task reads a frame after camera
    // teardown begins. The slot remains open until the worker has joined.
    inference_worker.stop();
    let inference_result = inference_worker.join();
    let capture_metrics = capture.shutdown().map_err(|error| error.to_string())?;
    let stats = match inference_result {
        WorkerResult::Completed(WorkerOutput { stats }) => stats,
        WorkerResult::Panicked => return Err("MediaPipe worker panicked".into()),
    };
    if let Some(failure) = stats.failure.as_deref() {
        return Err(format!("MediaPipe worker failed: {failure}"));
    }
    if restart_count != 3 {
        return Err(format!(
            "Stop/Start gate incomplete: completed {restart_count}/3 cycles"
        ));
    }

    print_summary(&stats, &capture_metrics, options.json);
    validate_gate(&stats, &capture_metrics, restart_count)?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn run_worker(
    task_path: &Path,
    frame_slot: Arc<LatestSlot<VideoFrame>>,
    stop: vtuber_core::StopToken,
) -> WorkerOutput {
    let mut stats = SmokeStats::default();
    let mut runtime = match MediaPipeRuntime::from_task_source(&MediaPipeTaskSource::Path(
        task_path.to_path_buf(),
    )) {
        Ok(runtime) => runtime,
        Err(error) => {
            stats.failure = Some(error.to_string());
            return WorkerOutput { stats };
        }
    };
    stats.backend_source = match native_library_source() {
        Ok(source) => Some(source.into()),
        Err(error) => {
            stats.failure = Some(error.to_string());
            return WorkerOutput { stats };
        }
    };
    let mut last_generation = 0;
    while !stop.is_stopped() {
        let Some(read) = frame_slot.wait_read_after(last_generation, FRAME_WAIT) else {
            continue;
        };
        let (frame, read_generation) = match read {
            ReadResult::New {
                generation,
                value: frame,
            } => (frame, generation),
            ReadResult::Closed => break,
        };
        last_generation = read_generation;
        stats.last_source_seq = Some(frame.seq);
        let inference_started = Instant::now();
        let result = match runtime.infer_face_tracking(&frame) {
            Ok(result) => result,
            Err(error) => {
                if matches!(error, InferenceError::MediaPipeOutputContract(_)) {
                    stats.contract_failures += 1;
                } else {
                    stats.inference_errors += 1;
                }
                stats.failure = Some(error.to_string());
                break;
            }
        };
        let finished = Instant::now();
        stats
            .inference_durations
            .push(finished.duration_since(inference_started));
        stats.first_result_at.get_or_insert(inference_started);
        stats.last_result_at = Some(finished);
        record_result(&mut stats, result);
    }

    drop(runtime);
    WorkerOutput { stats }
}

#[cfg(target_os = "windows")]
fn record_result(stats: &mut SmokeStats, result: FaceTrackingOutcome) {
    match result {
        FaceTrackingOutcome::NoFace { .. } => stats.no_face_count += 1,
        FaceTrackingOutcome::Face(sample) => {
            stats.face_count += 1;
            stats.last_source_seq = Some(sample.source_seq);
            stats.last_landmark_count = Some(sample.landmarks.len());
            stats.last_blendshape_count = Some(sample.blendshapes.as_array().len());
            // The production adapter accepts exactly one validated source matrix.
            stats.last_matrix_count = Some(1);
            stats.valid_matrix_count += 1;
            stats.matrix_determinant = Some(sample.quality.matrix_determinant);
            stats.matrix_orthogonality_error = Some(sample.quality.matrix_orthogonality_error);
        }
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
fn restart_interval(duration: Duration) -> Duration {
    let quarter = duration / 4;
    quarter.max(Duration::from_millis(250))
}

#[cfg(target_os = "windows")]
fn result_rate_hz(stats: &SmokeStats) -> f64 {
    let result_count = stats.face_count + stats.no_face_count;
    match (stats.first_result_at, stats.last_result_at) {
        (Some(first), Some(last)) if last > first => {
            result_count as f64 / last.duration_since(first).as_secs_f64()
        }
        _ => 0.0,
    }
}

#[cfg(target_os = "windows")]
fn validate_gate(
    stats: &SmokeStats,
    capture: &vtuber_camera::CaptureMetrics,
    restart_count: u8,
) -> Result<(), String> {
    let mut failures = Vec::new();
    let result_hz = result_rate_hz(stats);
    if result_hz < 15.0 {
        failures.push(format!("result rate {result_hz:.3} Hz is below 15 Hz"));
    }
    if stats.face_count == 0 {
        failures.push("no face result was observed".into());
    }
    if stats.valid_matrix_count == 0 {
        failures.push("no valid 478-landmark/52-blendshape/one-matrix result was observed".into());
    }
    if stats.contract_failures != 0 {
        failures.push(format!(
            "{} output-contract failures were observed",
            stats.contract_failures
        ));
    }
    if stats.inference_errors != 0 {
        failures.push(format!(
            "{} inference errors were observed",
            stats.inference_errors
        ));
    }
    if restart_count != 3 {
        failures.push(format!("Stop/Start completed {restart_count}/3 cycles"));
    }
    if capture.publish_rejected_frames > capture.frames_captured {
        failures.push("capture publish rejection count exceeds capture count".into());
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("standalone gate failed: {}", failures.join("; ")))
    }
}

#[cfg(target_os = "windows")]
fn print_summary(stats: &SmokeStats, capture: &vtuber_camera::CaptureMetrics, json: bool) {
    let task_bundle_file = MediaPipeTask::Face.file();
    let task_bundle_sha256 = MediaPipeTask::Face.sha256();
    let durations = &stats.inference_durations;
    let p50_ms = percentile_ms(durations, 0.50);
    let p95_ms = percentile_ms(durations, 0.95);
    let result_hz = result_rate_hz(stats);
    let source = stats.backend_source.as_deref().unwrap_or("unknown");
    let landmarks = stats.last_landmark_count.unwrap_or(0);
    let blendshapes = stats.last_blendshape_count.unwrap_or(0);
    let matrices = stats.last_matrix_count.unwrap_or(0);
    let last_seq = stats.last_source_seq.map_or(0, |seq| seq.0);
    if json {
        let determinant = stats
            .matrix_determinant
            .map_or_else(|| "null".into(), |value| format!("{value:.6}"));
        let orthogonality_error = stats
            .matrix_orthogonality_error
            .map_or_else(|| "null".into(), |value| format!("{value:.6}"));
        println!(
            "{{\"backend\":\"mediapipe-face-landmarker\",\"mediapipe_version\":\"{MEDIAPIPE_VERSION}\",\"native_library_source\":\"{source}\",\"task_bundle\":\"{task_bundle_file}\",\"task_bundle_sha256\":\"{task_bundle_sha256}\",\"face_count\":{},\"no_face_count\":{},\"result_hz\":{result_hz:.3},\"p50_inference_ms\":{p50_ms:.3},\"p95_inference_ms\":{p95_ms:.3},\"landmarks\":{landmarks},\"blendshapes\":{blendshapes},\"matrices\":{matrices},\"determinant\":{determinant},\"orthogonality_error\":{orthogonality_error},\"contract_failures\":{},\"last_source_seq\":{last_seq},\"capture_frames\":{},\"capture_publish_rejected_frames\":{},\"latest_slot_capacity\":1}}",
            stats.face_count,
            stats.no_face_count,
            stats.contract_failures,
            capture.frames_captured,
            capture.publish_rejected_frames,
        );
    } else {
        let determinant = stats
            .matrix_determinant
            .map_or_else(|| "n/a".into(), |value| format!("{value:.6}"));
        let orthogonality_error = stats
            .matrix_orthogonality_error
            .map_or_else(|| "n/a".into(), |value| format!("{value:.6}"));
        println!("native_library_source={source}");
        println!("task_bundle_sha256={task_bundle_sha256}");
        println!("face_count={}", stats.face_count);
        println!("no_face_count={}", stats.no_face_count);
        println!("result_hz={result_hz:.3}");
        println!("p50_inference_ms={p50_ms:.3}");
        println!("p95_inference_ms={p95_ms:.3}");
        println!("landmarks={landmarks}");
        println!("blendshapes={blendshapes}");
        println!("matrices={matrices}");
        println!("determinant={determinant}");
        println!("orthogonality_error={orthogonality_error}");
        println!("contract_failures={}", stats.contract_failures);
        println!("last_source_seq={last_seq}");
        println!("capture_frames={}", capture.frames_captured);
        println!(
            "capture_publish_rejected_frames={}",
            capture.publish_rejected_frames
        );
        println!("latest_slot_capacity=1");
        println!("worker_shutdown=clean");
    }
}

#[cfg(any(target_os = "windows", test))]
#[expect(
    clippy::indexing_slicing,
    reason = "the empty case returns early and the clamp keeps the index below the vector length"
)]
fn percentile_ms(values: &[Duration], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index].as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::percentile_ms;
    use std::time::Duration;

    #[test]
    fn percentile_is_sorted() {
        let values = [
            Duration::from_millis(30),
            Duration::from_millis(10),
            Duration::from_millis(20),
        ];
        assert_eq!(percentile_ms(&values, 0.50), 20.0);
    }
}
