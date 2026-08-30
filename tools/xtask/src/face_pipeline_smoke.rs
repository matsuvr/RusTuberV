//! Windows-only camera to composite-inference diagnostic command.

#[cfg(any(target_os = "windows", test))]
use std::io::Write;
#[cfg(target_os = "windows")]
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
#[cfg(target_os = "windows")]
use std::time::Instant;

#[cfg(target_os = "windows")]
use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, Rgb, Rgba};
#[cfg(target_os = "windows")]
use vtuber_core::types::{PixelFormat, VideoFrame};

#[cfg(target_os = "windows")]
use vtuber_inference::{InferenceMetrics, InferenceStage};
#[cfg(target_os = "windows")]
use vtuber_tracking::{
    CANONICAL_FACE_TEMPLATE, PlanarCorrespondence, PlanarLandmark, solve_planar_pose,
};

/// Runs the face-pipeline smoke command or prints its platform limitation.
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
        Err("face-pipeline-smoke requires Windows MSMF".into())
    }
}

/// Prints command usage without opening a camera.
pub fn print_help() {
    println!("face-pipeline-smoke - Windows MSMF composite inference probe");
    println!();
    println!("USAGE:");
    println!("  cargo run -p xtask -- face-pipeline-smoke [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --camera <id-or-index>   MSMF camera descriptor id or enumeration index");
    println!("  --duration <seconds>     Capture duration (default: 60)");
    println!("  --pipeline <id>          Legacy research pipeline id");
    println!("  --project-root <path>    Workspace/project root (default: current directory)");
    println!("  --snapshot <path>        Save one captured frame as a local JPEG");
    println!("  --guided-protocol        Print timed prompts for the full camera protocol");
    println!("  --json                   Emit one bounded JSON summary");
    println!("  -h, --help               Show this help");
    println!();
    println!("The command starts capture first, constructs the composite runtime in");
    println!("the inference worker, and joins inference before capture on shutdown.");
}

#[derive(Debug)]
struct Options {
    camera: Option<String>,
    duration: Duration,
    pipeline: Option<String>,
    project_root: PathBuf,
    snapshot: Option<PathBuf>,
    guided_protocol: bool,
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
            pipeline: None,
            project_root: std::env::current_dir()
                .map_err(|error| format!("cannot resolve project root: {error}"))?,
            snapshot: None,
            guided_protocol: false,
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
                    if seconds == 0 {
                        return Err("--duration must be greater than zero".into());
                    }
                    options.duration = Duration::from_secs(seconds);
                }
                "--pipeline" => {
                    index += 1;
                    options.pipeline = Some(required_value(args, index, "--pipeline")?);
                }
                "--project-root" => {
                    index += 1;
                    options.project_root =
                        PathBuf::from(required_value(args, index, "--project-root")?);
                }
                "--snapshot" => {
                    index += 1;
                    options.snapshot =
                        Some(PathBuf::from(required_value(args, index, "--snapshot")?));
                }
                "--guided-protocol" => options.guided_protocol = true,
                other => return Err(format!("unknown face-pipeline-smoke option `{other}`")),
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

#[cfg(any(target_os = "windows", test))]
#[derive(Clone, Copy, Debug)]
enum GuidedAction {
    None,
    StopCapture,
    StartCapture,
}

#[cfg(any(target_os = "windows", test))]
#[derive(Clone, Copy, Debug)]
struct GuidedPhase {
    label: &'static str,
    duration: Duration,
    action: GuidedAction,
}

#[cfg(any(target_os = "windows", test))]
struct GuidedProtocol {
    phases: Vec<GuidedPhase>,
    phase_index: usize,
    phase_deadline: Duration,
    announced: bool,
    last_countdown: Option<u64>,
    completed: bool,
}

#[cfg(any(target_os = "windows", test))]
impl GuidedProtocol {
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn new() -> Self {
        let mut phases = vec![GuidedPhase {
            label: "顔を中央に置き、neutralで正面を向く",
            duration: Duration::from_secs(60),
            action: GuidedAction::None,
        }];
        for cycle in 1..=5 {
            phases.push(GuidedPhase {
                label: match cycle {
                    1 => "顔を完全に画面外へ出す（1/5）",
                    2 => "顔を完全に画面外へ出す（2/5）",
                    3 => "顔を完全に画面外へ出す（3/5）",
                    4 => "顔を完全に画面外へ出す（4/5）",
                    _ => "顔を完全に画面外へ出す（5/5）",
                },
                duration: Duration::from_secs(5),
                action: GuidedAction::None,
            });
            phases.push(GuidedPhase {
                label: "顔を中央へ戻す",
                duration: Duration::from_secs(5),
                action: GuidedAction::None,
            });
        }
        phases.extend([
            GuidedPhase {
                label: "通常範囲で左を向く",
                duration: Duration::from_secs(6),
                action: GuidedAction::None,
            },
            GuidedPhase {
                label: "通常範囲で右を向く",
                duration: Duration::from_secs(6),
                action: GuidedAction::None,
            },
            GuidedPhase {
                label: "通常範囲で上を向く",
                duration: Duration::from_secs(6),
                action: GuidedAction::None,
            },
            GuidedPhase {
                label: "通常範囲で下を向く",
                duration: Duration::from_secs(6),
                action: GuidedAction::None,
            },
            GuidedPhase {
                label: "顔を画面端近くへ移動する",
                duration: Duration::from_secs(8),
                action: GuidedAction::None,
            },
        ]);
        for cycle in 1..=3 {
            phases.push(GuidedPhase {
                label: match cycle {
                    1 => "captureを停止中（1/3）",
                    2 => "captureを停止中（2/3）",
                    _ => "captureを停止中（3/3）",
                },
                duration: Duration::from_secs(3),
                action: GuidedAction::StopCapture,
            });
            phases.push(GuidedPhase {
                label: "captureを再開中。顔を中央へ戻す",
                duration: Duration::from_secs(8),
                action: GuidedAction::StartCapture,
            });
        }
        Self {
            phase_deadline: phases[0].duration,
            phases,
            phase_index: 0,
            announced: false,
            last_countdown: None,
            completed: false,
        }
    }

    fn total_duration(&self) -> Duration {
        self.phases.iter().map(|phase| phase.duration).sum()
    }

    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn tick(&mut self, elapsed: Duration) -> Option<GuidedAction> {
        if self.completed {
            return None;
        }
        while elapsed >= self.phase_deadline {
            self.phase_index += 1;
            if self.phase_index >= self.phases.len() {
                self.completed = true;
                println!("[guided] 全protocol完了");
                flush_stdout();
                return None;
            }
            self.phase_deadline += self.phases[self.phase_index].duration;
            self.announced = false;
            self.last_countdown = None;
        }

        let phase = self.phases[self.phase_index];
        let mut action = None;
        if !self.announced {
            println!(
                "[guided +{:03}s] {}（{}秒）",
                elapsed.as_secs(),
                phase.label,
                phase.duration.as_secs()
            );
            flush_stdout();
            self.announced = true;
            action = Some(phase.action);
        }
        let remaining = self.phase_deadline.saturating_sub(elapsed).as_secs();
        if remaining <= 3 && self.last_countdown != Some(remaining) {
            println!("[guided] 残り{remaining}秒");
            flush_stdout();
            self.last_countdown = Some(remaining);
        }
        action
    }

    fn is_complete(&self) -> bool {
        self.completed
    }
}

#[cfg(any(target_os = "windows", test))]
fn flush_stdout() {
    let _ = std::io::stdout().flush();
}

#[cfg(target_os = "windows")]
fn run_windows(options: Options) -> Result<(), String> {
    use std::sync::{Arc, Mutex};
    use vtuber_camera::backend::msmf::MsmfBackend;
    use vtuber_camera::device::CameraBackend;
    use vtuber_camera::{CameraRequest, CaptureController};
    use vtuber_core::{LatestSlot, ReadResult, WorkerHandle, WorkerResult};
    use vtuber_inference::state::SharedStatus;
    use vtuber_inference::{
        CompositeFrameInference, FailureStage, InferenceWorkerResult, InferenceWorkerState,
        InferenceWorkerStatus,
    };

    let manifest = options
        .project_root
        .join("assets")
        .join("models")
        .join("manifest.toml");
    let pipeline = vtuber_app::model_catalog::verify_research_pipeline_artifacts(&manifest)
        .map_err(|error| format!("model verification failed: {error}"))?;
    if let Some(requested) = options.pipeline.as_deref()
        && requested != pipeline.id
    {
        return Err(format!(
            "requested pipeline `{requested}` is not the manifest production pipeline `{}`",
            pipeline.id
        ));
    }

    let enumeration_backend = MsmfBackend::new();
    let devices = enumeration_backend
        .enumerate()
        .map_err(|error| format!("camera enumeration failed: {error}"))?;
    let device = choose_camera(&devices, options.camera.as_deref())?;
    println!("Selected camera: {device}");
    println!("Pipeline: {}", pipeline.id);

    let mut capture = CaptureController::new();
    capture
        .start_worker(MsmfBackend::new())
        .map_err(|error| format!("capture worker start failed: {error}"))?;
    if let Err(error) = capture.select_and_start(device.clone(), CameraRequest::default()) {
        let _ = capture.shutdown();
        return Err(format!("camera start failed: {error}"));
    }

    let frame_slot = capture.frame_slot();
    let output_slot = Arc::new(LatestSlot::new());
    let status: SharedStatus = Arc::new(Mutex::new(InferenceWorkerStatus::new()));
    let worker_status = Arc::clone(&status);
    let worker_frame_slot = Arc::clone(&frame_slot);
    let worker_output_slot = Arc::clone(&output_slot);
    let artifact_root = options.project_root.join("assets").join("models");
    let worker_descriptor = pipeline.clone();
    let inference_worker = WorkerHandle::spawn("face-pipeline-inference", move |stop| {
        match CompositeFrameInference::from_pipeline_descriptor(
            &worker_descriptor,
            &artifact_root,
            &vtuber_inference::RuntimeSettings::default(),
        ) {
            Ok(runtime) => vtuber_inference::worker::run_composite_inference_worker(
                Box::new(runtime),
                stop,
                worker_status,
                worker_frame_slot,
                worker_output_slot,
            ),
            Err(error) => {
                let final_metrics = {
                    let mut status = worker_status
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    status.record_failure(FailureStage::ModelLoad, error);
                    status.metrics()
                };
                InferenceWorkerResult { final_metrics }
            }
        }
    });

    let started = Instant::now();
    let mut guided = options.guided_protocol.then(GuidedProtocol::new);
    let deadline = started
        + guided
            .as_ref()
            .map(GuidedProtocol::total_duration)
            .unwrap_or(options.duration);
    let mut output_generation = 0u64;
    let mut frame_generation = 0u64;
    let mut snapshot_written = false;
    let mut latest_observation = None;
    let mut neutral_observation = None;
    let mut pose_stats = PoseStats::default();

    while Instant::now() < deadline {
        if let Some(protocol) = guided.as_mut()
            && let Some(action) = protocol.tick(started.elapsed())
        {
            match action {
                GuidedAction::None => {}
                GuidedAction::StopCapture => capture.stop(),
                GuidedAction::StartCapture => capture
                    .select_and_start(device.clone(), CameraRequest::default())
                    .map_err(|error| format!("guided capture restart failed: {error}"))?,
            }
        }

        if let Some(snapshot_path) = options.snapshot.as_deref()
            && !snapshot_written
        {
            match frame_slot.try_read_after(frame_generation) {
                Some(ReadResult::New(frame)) => {
                    frame_generation = frame_slot.generation();
                    save_snapshot(&frame, snapshot_path)?;
                    println!("Saved snapshot: {}", snapshot_path.display());
                    snapshot_written = true;
                }
                Some(ReadResult::Closed) | None => {}
            }
        }

        match output_slot.try_read_after(output_generation) {
            Some(ReadResult::New(observation)) => {
                output_generation = output_slot.generation();
                if neutral_observation.is_none() {
                    neutral_observation = Some(observation.clone());
                }
                if let Some(neutral) = neutral_observation.as_ref()
                    && let Some(pose) = solve_observation_pose(neutral, &observation)
                {
                    pose_stats.record(pose);
                }
                latest_observation = Some(observation);
            }
            Some(ReadResult::Closed) | None => {}
        }

        let capture_state = capture.state();
        if matches!(capture_state, vtuber_camera::CaptureServiceState::BackOff) {
            break;
        }
        if status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .state
            == InferenceWorkerState::Failed
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let snapshot_missing = options.snapshot.is_some() && !snapshot_written;
    let deadline_reached = Instant::now() >= deadline;
    if deadline_reached
        && let Some(protocol) = guided.as_mut()
        && !protocol.is_complete()
    {
        let total = protocol.total_duration();
        let _ = protocol.tick(total);
    }
    let guided_incomplete = guided
        .as_ref()
        .is_some_and(|protocol| !protocol.is_complete());

    let camera_state = capture.state();
    inference_worker.stop();
    let inference_result = inference_worker.join();
    output_slot.close();
    let capture_metrics = capture.shutdown();
    let status_snapshot = status
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
    let metrics = match inference_result {
        WorkerResult::Completed(result) => result.final_metrics,
        WorkerResult::Panicked => {
            return Err("face-pipeline inference worker panicked".into());
        }
        WorkerResult::SpawnFailed => {
            return Err("face-pipeline inference worker failed to spawn".into());
        }
    };

    let summary = SmokeSummary::from_run(
        &pipeline,
        &device,
        camera_state,
        &capture_metrics,
        &status_snapshot,
        &metrics,
        latest_observation.as_ref(),
        pose_stats,
        elapsed,
    );
    if options.json {
        println!("{}", summary.to_json());
    } else {
        summary.print_text();
    }
    if snapshot_missing {
        return Err("snapshot requested but no camera frame was published".into());
    }
    if guided_incomplete {
        return Err("guided protocol did not complete before capture stopped".into());
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn save_snapshot(frame: &VideoFrame, path: &Path) -> Result<(), String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| "snapshot path must use a .jpg or .jpeg extension".to_string())?;
    if extension != "jpg" && extension != "jpeg" {
        return Err("snapshot output must use a .jpg or .jpeg extension".into());
    }

    let image = match frame.format {
        PixelFormat::Rgb8 => {
            ImageBuffer::<Rgb<u8>, _>::from_raw(frame.width, frame.height, frame.data.to_vec())
                .map(DynamicImage::ImageRgb8)
        }
        PixelFormat::Bgr8 => {
            let mut rgb = frame.data.to_vec();
            for pixel in rgb.chunks_exact_mut(3) {
                pixel.swap(0, 2);
            }
            ImageBuffer::<Rgb<u8>, _>::from_raw(frame.width, frame.height, rgb)
                .map(DynamicImage::ImageRgb8)
        }
        PixelFormat::Rgba8 => {
            ImageBuffer::<Rgba<u8>, _>::from_raw(frame.width, frame.height, frame.data.to_vec())
                .map(DynamicImage::ImageRgba8)
        }
        PixelFormat::Gray8 => {
            ImageBuffer::<Luma<u8>, _>::from_raw(frame.width, frame.height, frame.data.to_vec())
                .map(DynamicImage::ImageLuma8)
        }
    }
    .ok_or_else(|| {
        format!(
            "camera frame buffer does not match {}x{} {:?}",
            frame.width, frame.height, frame.format
        )
    })?;

    image
        .save_with_format(path, ImageFormat::Jpeg)
        .map_err(|error| format!("snapshot write failed for {}: {error}", path.display()))
}

#[cfg(target_os = "windows")]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn choose_camera(
    devices: &[vtuber_camera::CameraDescriptor],
    requested: Option<&str>,
) -> Result<vtuber_camera::CameraDescriptor, String> {
    if devices.is_empty() {
        return Err("CAMERA_ENUM_FAILED: no MSMF camera devices found".into());
    }
    if let Some(requested) = requested {
        if let Ok(index) = requested.parse::<usize>() {
            return devices
                .get(index)
                .cloned()
                .ok_or_else(|| format!("camera enumeration index out of range: {index}"));
        }
        return devices
            .iter()
            .find(|device| device.id == requested)
            .cloned()
            .ok_or_else(|| format!("MSMF camera descriptor not found: {requested}"));
    }
    Ok(devices[0].clone())
}

#[cfg(target_os = "windows")]
#[derive(Clone, Copy, Debug, Default)]
struct PoseStats {
    finite_count: u64,
    yaw_min: f32,
    yaw_max: f32,
    pitch_min: f32,
    pitch_max: f32,
    roll_min: f32,
    roll_max: f32,
}

#[cfg(target_os = "windows")]
impl PoseStats {
    fn record(&mut self, pose: vtuber_core::HeadPose) {
        if self.finite_count == 0 {
            self.yaw_min = pose.yaw_rad;
            self.yaw_max = pose.yaw_rad;
            self.pitch_min = pose.pitch_rad;
            self.pitch_max = pose.pitch_rad;
            self.roll_min = pose.roll_rad;
            self.roll_max = pose.roll_rad;
        } else {
            self.yaw_min = self.yaw_min.min(pose.yaw_rad);
            self.yaw_max = self.yaw_max.max(pose.yaw_rad);
            self.pitch_min = self.pitch_min.min(pose.pitch_rad);
            self.pitch_max = self.pitch_max.max(pose.pitch_rad);
            self.roll_min = self.roll_min.min(pose.roll_rad);
            self.roll_max = self.roll_max.max(pose.roll_rad);
        }
        self.finite_count += 1;
    }
}

#[cfg(target_os = "windows")]
fn solve_observation_pose(
    neutral: &vtuber_core::RawFaceObservation,
    current: &vtuber_core::RawFaceObservation,
) -> Option<vtuber_core::HeadPose> {
    let correspondences = CANONICAL_FACE_TEMPLATE
        .iter()
        .filter_map(|canonical| {
            let reference = neutral.landmarks.get(canonical.index)?;
            let current = current.landmarks.get(canonical.index)?;
            Some(PlanarCorrespondence {
                canonical: *canonical,
                reference: PlanarLandmark {
                    x: reference.x,
                    y: reference.y,
                    confidence: reference.visibility,
                },
                current: PlanarLandmark {
                    x: current.x,
                    y: current.y,
                    confidence: current.visibility,
                },
            })
        })
        .collect::<Vec<_>>();
    solve_planar_pose(&correspondences)
        .ok()
        .map(|alignment| alignment.pose)
}

#[cfg(target_os = "windows")]
struct SmokeSummary {
    pipeline_id: String,
    camera: String,
    camera_state: String,
    capture_format: String,
    frames_captured: u64,
    face_count: u64,
    no_face_count: u64,
    detector_hz: f64,
    landmark_hz: f64,
    stage_error: String,
    detector_confidence: Option<f32>,
    roi: Option<vtuber_core::NormalizedRect>,
    crop_scale: f32,
    crop_y_offset: f32,
    finite_landmarks: usize,
    pose_stats: PoseStats,
}

#[cfg(target_os = "windows")]
impl SmokeSummary {
    #[allow(clippy::too_many_arguments)]
    fn from_run(
        pipeline: &vtuber_inference::FacePipelineDescriptor,
        camera: &vtuber_camera::CameraDescriptor,
        camera_state: vtuber_camera::CaptureServiceState,
        capture: &vtuber_camera::CaptureMetrics,
        status: &vtuber_inference::InferenceWorkerStatus,
        metrics: &InferenceMetrics,
        observation: Option<&vtuber_core::RawFaceObservation>,
        pose_stats: PoseStats,
        elapsed: f64,
    ) -> Self {
        let stage_error = status
            .last_failure
            .as_ref()
            .map(|failure| format!("{:?}: {}", failure.stage, failure.error))
            .unwrap_or_else(|| "none".into());
        let finite_landmarks = observation
            .map(|observation| {
                observation
                    .landmarks
                    .iter()
                    .filter(|landmark| {
                        landmark.x.is_finite()
                            && landmark.y.is_finite()
                            && landmark.z.is_finite()
                            && landmark.visibility.is_finite()
                    })
                    .count()
            })
            .unwrap_or(0);
        Self {
            pipeline_id: pipeline.id.clone(),
            camera: camera.to_string(),
            camera_state: format!("{camera_state:?}"),
            capture_format: capture
                .format
                .map(|format| format.to_string())
                .unwrap_or_else(|| "unknown".into()),
            frames_captured: capture.frames_captured,
            face_count: status.frames_processed,
            no_face_count: status.no_face_frames,
            detector_hz: metrics.stage(InferenceStage::Detector).count as f64 / elapsed,
            landmark_hz: metrics.stage(InferenceStage::Landmark).count as f64 / elapsed,
            stage_error,
            detector_confidence: observation.map(|observation| observation.face_confidence),
            roi: observation.map(|observation| observation.roi),
            crop_scale: pipeline.crop.square_scale,
            crop_y_offset: pipeline.crop.center_y_offset_fraction,
            finite_landmarks,
            pose_stats,
        }
    }

    fn print_text(&self) {
        println!("Face pipeline smoke summary");
        println!("  pipeline: {}", self.pipeline_id);
        println!("  camera: {}", self.camera);
        println!("  camera state: {}", self.camera_state);
        println!("  format: {}", self.capture_format);
        println!("  frames captured: {}", self.frames_captured);
        println!("  face/no-face: {}/{}", self.face_count, self.no_face_count);
        println!(
            "  detector/landmark Hz: {:.2}/{:.2}",
            self.detector_hz, self.landmark_hz
        );
        println!("  stage error: {}", self.stage_error);
        println!("  detector confidence: {:?}", self.detector_confidence);
        println!("  ROI: {:?}", self.roi);
        println!(
            "  crop scale/y offset: {:.3}/{:.3}",
            self.crop_scale, self.crop_y_offset
        );
        println!("  finite landmarks: {}", self.finite_landmarks);
        println!("  finite poses: {}", self.pose_stats.finite_count);
        if self.pose_stats.finite_count > 0 {
            println!(
                "  pose yaw/pitch/roll ranges: [{:.4}, {:.4}] [{:.4}, {:.4}] [{:.4}, {:.4}]",
                self.pose_stats.yaw_min,
                self.pose_stats.yaw_max,
                self.pose_stats.pitch_min,
                self.pose_stats.pitch_max,
                self.pose_stats.roll_min,
                self.pose_stats.roll_max
            );
        }
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"pipeline_id\":\"{}\",\"camera\":\"{}\",\"camera_state\":\"{}\",\"format\":\"{}\",\"frames_captured\":{},\"face_count\":{},\"no_face_count\":{},\"detector_hz\":{:.3},\"landmark_hz\":{:.3},\"stage_error\":\"{}\",\"detector_confidence\":{},\"roi\":{},\"crop_scale\":{:.4},\"crop_y_offset\":{:.4},\"finite_landmarks\":{},\"finite_pose_count\":{},\"pose_range\":{}}}",
            json_escape(&self.pipeline_id),
            json_escape(&self.camera),
            json_escape(&self.camera_state),
            json_escape(&self.capture_format),
            self.frames_captured,
            self.face_count,
            self.no_face_count,
            self.detector_hz,
            self.landmark_hz,
            json_escape(&self.stage_error),
            self.detector_confidence
                .map_or_else(|| "null".into(), |value| format!("{value:.6}")),
            self.roi.map_or_else(
                || "null".into(),
                |roi| {
                    format!(
                        "{{\"x\":{:.6},\"y\":{:.6},\"width\":{:.6},\"height\":{:.6}}}",
                        roi.x, roi.y, roi.width, roi.height
                    )
                }
            ),
            self.crop_scale,
            self.crop_y_offset,
            self.finite_landmarks,
            self.pose_stats.finite_count,
            if self.pose_stats.finite_count == 0 {
                "null".into()
            } else {
                format!(
                    "{{\"yaw\":[{:.6},{:.6}],\"pitch\":[{:.6},{:.6}],\"roll\":[{:.6},{:.6}]}}",
                    self.pose_stats.yaw_min,
                    self.pose_stats.yaw_max,
                    self.pose_stats.pitch_min,
                    self.pose_stats.pitch_max,
                    self.pose_stats.roll_min,
                    self.pose_stats.roll_max
                )
            }
        )
    }
}

#[cfg(target_os = "windows")]
fn json_escape(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| match character {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            other => vec![other],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::GuidedProtocol;
    use std::time::Duration;

    #[test]
    fn guided_protocol_marks_the_exact_deadline_complete() {
        let mut protocol = GuidedProtocol::new();
        let total = protocol.total_duration();

        protocol.tick(Duration::ZERO);
        assert!(!protocol.is_complete());
        protocol.tick(total);

        assert!(protocol.is_complete());
    }
}
