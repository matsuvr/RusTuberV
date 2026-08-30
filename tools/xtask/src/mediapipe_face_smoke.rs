//! Standalone Windows MediaPipe Face Landmarker gate.
//!
//! The smoke path uses the existing camera boundary, `vtuber-core`, and the
//! pinned `mediapipe-rs` binding. It does not construct Bevy or access a VRM.
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
use mediapipe::{
    Confidence, Delegate, FaceLandmarker, FaceLandmarkerResult, Image, IouThreshold,
    MEDIAPIPE_VERSION, ModelSource, Size, Timestamp,
};
#[cfg(target_os = "windows")]
use sha2::{Digest, Sha256};
#[cfg(target_os = "windows")]
use vtuber_core::FrameSeq;
#[cfg(target_os = "windows")]
use vtuber_core::{LatestSlot, ReadResult};
#[cfg(any(target_os = "windows", test))]
use vtuber_core::{MonoTimeNs, PixelFormat, VideoFrame};

#[cfg(target_os = "windows")]
const TASK_BUNDLE_FILE: &str = "face_landmarker.task";
#[cfg(target_os = "windows")]
const TASK_BUNDLE_SHA256: &str = "64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF";
const MAX_TASK_DURATION: Duration = Duration::from_secs(24 * 60 * 60);
#[cfg(target_os = "windows")]
const FRAME_WAIT: Duration = Duration::from_millis(100);
#[cfg(target_os = "windows")]
const MATRIX_AFFINE_EPSILON: f32 = 0.1;

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
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
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
        .join(TASK_BUNDLE_FILE);
    verify_task_bundle(&task_path)?;

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
        println!("backend=mediapipe-face-landmarker");
        println!("mediapipe_version={MEDIAPIPE_VERSION}");
        println!("camera={device}");
        println!("task_bundle={TASK_BUNDLE_FILE}");
    }

    let frame_slot = capture.frame_slot();
    let worker_frame_slot = Arc::clone(&frame_slot);
    let worker_task_path = task_path.clone();
    let inference_worker = WorkerHandle::spawn("mediapipe-face-worker", move |stop| {
        run_worker(&worker_task_path, worker_frame_slot, stop)
    });

    let started = Instant::now();
    let mut next_restart = restart_interval(options.duration);
    let mut restart_count = 0u8;
    while started.elapsed() < options.duration {
        if restart_count < 3 && started.elapsed() >= next_restart {
            capture.stop();
            thread::sleep(Duration::from_millis(150));
            if let Err(error) = capture.select_and_start(device.clone(), CameraRequest::default()) {
                inference_worker.stop();
                let inference_result = inference_worker.join();
                let _ = capture.shutdown();
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
    let capture_metrics = capture.shutdown();
    let stats = match inference_result {
        WorkerResult::Completed(WorkerOutput { stats }) => stats,
        WorkerResult::Panicked => return Err("MediaPipe worker panicked".into()),
        WorkerResult::SpawnFailed => return Err("MediaPipe worker failed to spawn".into()),
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
    let library_source = match mediapipe::loader::lib() {
        Ok(library) => library_source_name(&library.source),
        Err(error) => {
            stats.failure = Some(format!("native library load failed: {error}"));
            return WorkerOutput { stats };
        }
    };
    stats.backend_source = Some(library_source);

    let mut landmarker = match FaceLandmarker::builder(ModelSource::path(task_path))
        .delegate(Delegate::Cpu)
        .num_faces(std::num::NonZeroU32::MIN)
        .min_face_detection_confidence(Confidence::HALF)
        .min_face_presence_confidence(Confidence::HALF)
        .min_tracking_confidence(IouThreshold::HALF)
        .output_blendshapes(true)
        .output_transformation_matrixes(true)
        .build_for_video()
    {
        Ok(landmarker) => landmarker,
        Err(error) => {
            stats.failure = Some(format!("task construction failed: {error}"));
            return WorkerOutput { stats };
        }
    };

    let mut last_generation = 0;
    let mut last_timestamp_ms = None;
    let mut staging = Vec::new();
    while !stop.is_stopped() {
        let Some(read) = frame_slot.wait_read_after(last_generation, FRAME_WAIT) else {
            continue;
        };
        let frame = match read {
            ReadResult::New(frame) => frame,
            ReadResult::Closed => break,
        };
        last_generation = frame_slot.generation();
        stats.last_source_seq = Some(frame.seq);
        let timestamp_ms = match video_timestamp_ms(frame.captured_at, &mut last_timestamp_ms) {
            Ok(timestamp_ms) => timestamp_ms,
            Err(error) => {
                stats.failure = Some(error);
                break;
            }
        };
        let image_data = match frame_rgb(&frame, &mut staging) {
            Ok(image_data) => image_data,
            Err(error) => {
                stats.inference_errors += 1;
                stats.failure = Some(error);
                break;
            }
        };
        let image = match Image::from_rgb(
            Size {
                width: frame.width,
                height: frame.height,
            },
            image_data,
        ) {
            Ok(image) => image,
            Err(error) => {
                stats.inference_errors += 1;
                stats.failure = Some(format!("image construction failed: {error}"));
                break;
            }
        };
        let inference_started = Instant::now();
        let result = match landmarker.detect_for_video(&image, Timestamp::from_millis(timestamp_ms))
        {
            Ok(result) => result,
            Err(error) => {
                stats.inference_errors += 1;
                stats.failure = Some(format!("detect_for_video failed: {error}"));
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

    drop(landmarker);
    WorkerOutput { stats }
}

#[cfg(target_os = "windows")]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn record_result(stats: &mut SmokeStats, result: FaceLandmarkerResult) {
    let face_count = result.landmarks.len();
    if face_count == 0 {
        stats.no_face_count += 1;
        return;
    }
    if face_count != 1 || result.blendshapes.len() != 1 || result.transformation_matrixes.len() != 1
    {
        stats.contract_failures += 1;
        stats.face_count += 1;
        stats.last_landmark_count = result.landmarks.first().map(Vec::len);
        stats.last_blendshape_count = result.blendshapes.first().map(Vec::len);
        stats.last_matrix_count = Some(result.transformation_matrixes.len());
        return;
    }

    let landmarks = &result.landmarks[0];
    let blendshapes = &result.blendshapes[0];
    stats.last_landmark_count = Some(landmarks.len());
    stats.last_blendshape_count = Some(blendshapes.len());
    stats.last_matrix_count = Some(result.transformation_matrixes.len());
    let landmarks_valid = landmarks.len() == 478
        && landmarks.iter().all(|landmark| {
            landmark.point.x().is_finite()
                && landmark.point.y().is_finite()
                && landmark.point.z().is_finite()
                && landmark
                    .visibility
                    .is_none_or(|value| value.get().is_finite())
                && landmark
                    .presence
                    .is_none_or(|value| value.get().is_finite())
        });
    let blendshapes_valid = blendshapes.len() == 52
        && blendshapes.iter().all(|category| {
            category.score.get().is_finite() && (0.0..=1.0).contains(&category.score.get())
        });
    let matrix = matrix_quality(&result);
    let matrix_valid = matrix.is_some_and(|quality| quality.is_valid);
    if !(landmarks_valid && blendshapes_valid && matrix_valid) {
        stats.contract_failures += 1;
    } else {
        stats.valid_matrix_count += 1;
    }
    if let Some(quality) = matrix {
        stats.matrix_determinant = Some(quality.determinant);
        stats.matrix_orthogonality_error = Some(quality.orthogonality_error);
    }
    stats.face_count += 1;
}

#[cfg(target_os = "windows")]
#[derive(Clone, Copy, Debug)]
struct MatrixQuality {
    determinant: f32,
    orthogonality_error: f32,
    is_valid: bool,
}

#[cfg(target_os = "windows")]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn matrix_quality(result: &FaceLandmarkerResult) -> Option<MatrixQuality> {
    let matrix = result.transformation_matrixes.first()?;
    let mut values = [[0.0; 4]; 4];
    for (row, values_row) in values.iter_mut().enumerate() {
        for (col, value) in values_row.iter_mut().enumerate() {
            *value = matrix.get(row, col);
        }
    }
    if !values.iter().flatten().all(|value| value.is_finite()) {
        return None;
    }
    let determinant = determinant3(values);
    let mut error_squared = 0.0;
    for row in 0..3 {
        for col in 0..3 {
            let dot = (0..3)
                .map(|index| values[index][row] * values[index][col])
                .sum::<f32>();
            let expected = if row == col { 1.0 } else { 0.0 };
            error_squared += (dot - expected).powi(2);
        }
    }
    let orthogonality_error = error_squared.sqrt();
    let affine = values[3][0].abs() <= MATRIX_AFFINE_EPSILON
        && values[3][1].abs() <= MATRIX_AFFINE_EPSILON
        && values[3][2].abs() <= MATRIX_AFFINE_EPSILON
        && (values[3][3] - 1.0).abs() <= MATRIX_AFFINE_EPSILON;
    Some(MatrixQuality {
        determinant,
        orthogonality_error,
        is_valid: affine && determinant > 0.0,
    })
}

#[cfg(any(target_os = "windows", test))]
fn determinant3(matrix: [[f32; 4]; 4]) -> f32 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

#[cfg(any(target_os = "windows", test))]
fn video_timestamp_ms(
    captured_at: MonoTimeNs,
    last_timestamp_ms: &mut Option<i64>,
) -> Result<i64, String> {
    let candidate = i64::try_from(captured_at.0 / 1_000_000)
        .map_err(|_| "capture timestamp exceeds MediaPipe millisecond range".to_string())?;
    let timestamp_ms =
        last_timestamp_ms.map_or(candidate, |last| candidate.max(last.saturating_add(1)));
    *last_timestamp_ms = Some(timestamp_ms);
    Ok(timestamp_ms)
}

#[cfg(any(target_os = "windows", test))]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn frame_rgb<'a>(frame: &'a VideoFrame, staging: &'a mut Vec<u8>) -> Result<&'a [u8], String> {
    let width = usize::try_from(frame.width).map_err(|_| "frame width is too large".to_string())?;
    let height =
        usize::try_from(frame.height).map_err(|_| "frame height is too large".to_string())?;
    let channels = match frame.format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 => 4,
        PixelFormat::Gray8 => 1,
    };
    let row_bytes = width
        .checked_mul(channels)
        .ok_or_else(|| "frame row size overflow".to_string())?;
    if frame.stride_bytes < row_bytes {
        return Err(format!(
            "frame stride {} is smaller than row size {row_bytes}",
            frame.stride_bytes
        ));
    }
    let required = frame
        .stride_bytes
        .checked_mul(height)
        .ok_or_else(|| "frame buffer size overflow".to_string())?;
    if frame.data.len() < required {
        return Err(format!(
            "frame buffer has {} bytes but requires {required}",
            frame.data.len()
        ));
    }
    if frame.format == PixelFormat::Rgb8 && frame.stride_bytes == row_bytes {
        return Ok(frame.data.as_ref());
    }

    let rgb_row_bytes = width
        .checked_mul(3)
        .ok_or_else(|| "RGB row size overflow".to_string())?;
    let staging_len = rgb_row_bytes
        .checked_mul(height)
        .ok_or_else(|| "RGB frame size overflow".to_string())?;
    staging.resize(staging_len, 0);
    for row in 0..height {
        let source = &frame.data[row * frame.stride_bytes..row * frame.stride_bytes + row_bytes];
        let destination = &mut staging[row * rgb_row_bytes..(row + 1) * rgb_row_bytes];
        match frame.format {
            PixelFormat::Rgb8 => destination.copy_from_slice(source),
            PixelFormat::Bgr8 => {
                for (src, dst) in source.chunks_exact(3).zip(destination.chunks_exact_mut(3)) {
                    dst.copy_from_slice(&[src[2], src[1], src[0]]);
                }
            }
            PixelFormat::Rgba8 => {
                for (src, dst) in source.chunks_exact(4).zip(destination.chunks_exact_mut(3)) {
                    dst.copy_from_slice(&src[..3]);
                }
            }
            PixelFormat::Gray8 => {
                for (value, dst) in source.iter().zip(destination.chunks_exact_mut(3)) {
                    dst.fill(*value);
                }
            }
        }
    }
    Ok(staging)
}

#[cfg(target_os = "windows")]
fn verify_task_bundle(path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("task bundle read failed: {error}"))?;
    let digest = Sha256::digest(&bytes);
    let actual = digest
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect::<String>();
    if actual != TASK_BUNDLE_SHA256 {
        return Err(format!(
            "INFERENCE_MODEL_HASH_MISMATCH: expected {TASK_BUNDLE_SHA256}, got {actual}"
        ));
    }
    Ok(())
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
    if capture.frames_dropped > capture.frames_captured {
        failures.push("capture overwrite count exceeds capture count".into());
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("standalone gate failed: {}", failures.join("; ")))
    }
}

#[cfg(target_os = "windows")]
fn print_summary(stats: &SmokeStats, capture: &vtuber_camera::CaptureMetrics, json: bool) {
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
            "{{\"backend\":\"mediapipe-face-landmarker\",\"mediapipe_version\":\"{MEDIAPIPE_VERSION}\",\"native_library_source\":\"{source}\",\"task_bundle\":\"{TASK_BUNDLE_FILE}\",\"task_bundle_sha256\":\"{TASK_BUNDLE_SHA256}\",\"face_count\":{},\"no_face_count\":{},\"result_hz\":{result_hz:.3},\"p50_inference_ms\":{p50_ms:.3},\"p95_inference_ms\":{p95_ms:.3},\"landmarks\":{landmarks},\"blendshapes\":{blendshapes},\"matrices\":{matrices},\"determinant\":{determinant},\"orthogonality_error\":{orthogonality_error},\"contract_failures\":{},\"last_source_seq\":{last_seq},\"capture_frames\":{},\"capture_overwrites\":{},\"latest_slot_capacity\":1}}",
            stats.face_count,
            stats.no_face_count,
            stats.contract_failures,
            capture.frames_captured,
            capture.frames_dropped,
        );
    } else {
        let determinant = stats
            .matrix_determinant
            .map_or_else(|| "n/a".into(), |value| format!("{value:.6}"));
        let orthogonality_error = stats
            .matrix_orthogonality_error
            .map_or_else(|| "n/a".into(), |value| format!("{value:.6}"));
        println!("native_library_source={source}");
        println!("task_bundle_sha256={TASK_BUNDLE_SHA256}");
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
        println!("capture_overwrites={}", capture.frames_dropped);
        println!("latest_slot_capacity=1");
        println!("worker_shutdown=clean");
    }
}

#[cfg(any(target_os = "windows", test))]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
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
    use super::{determinant3, frame_rgb, percentile_ms, video_timestamp_ms};
    use std::time::Duration;
    use vtuber_core::{FrameSeq, MonoTimeNs, PixelFormat, VideoFrame};

    #[test]
    fn timestamp_adapter_is_strictly_increasing() {
        let mut last = None;
        assert_eq!(
            video_timestamp_ms(MonoTimeNs(10_000_000), &mut last).unwrap(),
            10
        );
        assert_eq!(
            video_timestamp_ms(MonoTimeNs(10_000_000), &mut last).unwrap(),
            11
        );
        assert_eq!(
            video_timestamp_ms(MonoTimeNs(9_000_000), &mut last).unwrap(),
            12
        );
    }

    #[test]
    fn rgb_repack_honors_stride_and_channel_order() {
        let frame = VideoFrame {
            seq: FrameSeq(1),
            captured_at: MonoTimeNs(0),
            width: 1,
            height: 2,
            stride_bytes: 4,
            format: PixelFormat::Bgr8,
            data: vec![3, 2, 1, 99, 6, 5, 4, 99].into(),
        };
        let mut staging = Vec::new();
        assert_eq!(
            frame_rgb(&frame, &mut staging).unwrap(),
            &[1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn identity_matrix_has_unit_determinant() {
        let matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(determinant3(matrix), 1.0);
    }

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
