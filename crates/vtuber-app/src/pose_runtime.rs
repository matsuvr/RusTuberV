//! Application bridge for the observed-arm Pose worker.
//!
//! The capture worker fans the same camera frame out to the face and Pose
//! slots; this resource owns the Pose worker, advances the pure arm-tracking
//! state on new observations, and publishes the resulting control frame into
//! the avatar's [`vtuber_avatar::TrackedArmControl`] resource.

use std::sync::Arc;

use bevy::prelude::*;
use vtuber_avatar::{ArmSourceSelection, TrackedArmControl};
use vtuber_core::arm_tracking::{ArmControlFrame, PoseArmFrame};
use vtuber_core::{LatestSlot, ReadResult, VideoFrame, WorkerHandle, monotonic_now};
use vtuber_inference::{
    InferenceError, InferenceWorkerResult, MediaPipeTaskSource, SharedStatus, run_pose_worker,
};
use vtuber_tracking::arm_tracking::{ArmTrackingProfile, ArmTrackingState, step_arm_tracking};

use crate::capture_runtime::CaptureRuntime;
use crate::orchestrator::Orchestrator;

/// The packaged Pose task filename.
const POSE_TASK_FILE: &str = "pose_landmarker_full.task";

/// The packaged Hand Landmarker task filename.
const HAND_TASK_FILE: &str = "hand_landmarker.task";

/// Owns the Pose worker and the pure observed-arm tracking state.
#[derive(Resource)]
pub struct PoseRuntime {
    enabled: bool,
    frame_slot: Arc<LatestSlot<VideoFrame>>,
    output_slot: Arc<LatestSlot<PoseArmFrame>>,
    status: SharedStatus,
    worker: Option<WorkerHandle<InferenceWorkerResult>>,
    tracking: ArmTrackingState,
    profile: ArmTrackingProfile,
    task_path: std::path::PathBuf,
    hand_task_path: std::path::PathBuf,
    output_generation: u64,
    held_frame: Option<PoseArmFrame>,
    /// Debug-build raw observation log, opened lazily on the first frame.
    #[cfg(debug_assertions)]
    debug_log: Option<std::fs::File>,
}

impl PoseRuntime {
    /// Creates the runtime with the camera-facing frame slot.
    #[must_use]
    pub fn new(project_root: std::path::PathBuf) -> Self {
        Self {
            enabled: false,
            frame_slot: Arc::new(LatestSlot::new()),
            output_slot: Arc::new(LatestSlot::new()),
            status: Arc::new(std::sync::Mutex::new(
                vtuber_inference::InferenceWorkerStatus::new(),
            )),
            worker: None,
            tracking: ArmTrackingState::new(),
            profile: ArmTrackingProfile::default(),
            task_path: project_root
                .join("assets")
                .join("models")
                .join(POSE_TASK_FILE),
            hand_task_path: project_root
                .join("assets")
                .join("models")
                .join(HAND_TASK_FILE),
            output_generation: 0,
            held_frame: None,
            #[cfg(debug_assertions)]
            debug_log: None,
        }
    }

    /// Returns the capacity-one slot the camera writes Pose frames into.
    #[must_use]
    pub fn frame_slot(&self) -> Arc<LatestSlot<VideoFrame>> {
        Arc::clone(&self.frame_slot)
    }

    /// Whether observed arm tracking is selected.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Selects observed arm tracking on or off. The worker follows in the bridge.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Drops calibration and observation state for a fresh calibration.
    pub fn recalibrate(&mut self) {
        self.tracking.reset();
        self.held_frame = None;
    }

    /// Whether the Pose worker thread is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.worker.is_some()
    }

    /// Starts the Pose worker, loading the task bundle inside the worker thread.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::WorkerSpawnFailed`] if the OS refused to
    /// spawn the thread. No worker is retained on failure, so the caller may
    /// try again later.
    pub fn ensure_running(&mut self) -> Result<(), InferenceError> {
        if self.worker.is_some() {
            return Ok(());
        }
        let task = if self.task_path.is_file() {
            MediaPipeTaskSource::Path(self.task_path.clone())
        } else {
            MediaPipeTaskSource::Embedded
        };
        let hand_task = if self.hand_task_path.is_file() {
            MediaPipeTaskSource::Path(self.hand_task_path.clone())
        } else {
            MediaPipeTaskSource::Embedded
        };
        let status = Arc::clone(&self.status);
        let frame_slot = Arc::clone(&self.frame_slot);
        let output_slot = Arc::clone(&self.output_slot);
        let worker = WorkerHandle::spawn("pose-worker", move |stop| {
            run_pose_worker(stop, status, frame_slot, output_slot, &task, &hand_task)
        })
        .map_err(|error| InferenceError::WorkerSpawnFailed {
            kind: error.kind(),
            message: error.to_string(),
        })?;
        self.worker = Some(worker);
        Ok(())
    }

    /// Stops the Pose worker and drops its observation state.
    pub fn stop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.stop();
            let _ = worker.join();
        }
        self.tracking.reset();
        self.output_generation = 0;
        self.held_frame = None;
        self.output_slot.clear();
    }

    /// Advances the pure tracking state by one render tick.
    ///
    /// Like the face pipeline, the latest completed inference is retained and
    /// re-fed while no new result exists, so the arm smoother advances on the
    /// render clock and the hands move continuously between camera frames.
    fn read_latest(&mut self) -> Option<ArmControlFrame> {
        if let Some(ReadResult::New(frame)) =
            self.output_slot.try_read_after(self.output_generation)
        {
            self.output_generation = self.output_slot.generation();
            #[cfg(debug_assertions)]
            log_pose_frame(&frame, &mut self.debug_log);
            self.held_frame = Some(frame);
        }
        let profile = self.profile;
        let (next, control) = step_arm_tracking(
            &self.tracking,
            self.held_frame.as_ref(),
            monotonic_now(),
            &profile,
        );
        self.tracking = next;
        control
    }
}

impl Default for PoseRuntime {
    fn default() -> Self {
        Self::new(std::path::PathBuf::from("."))
    }
}

/// Debug-build trace of every decoded Pose frame.
///
/// Records at `info!` so it is visible without changing the log filter, and
/// appends the raw world landmarks to `mediapipe_pose_debug.csv` next to the
/// working directory. Comparing those rows with `propagation_debug.log` frame
/// by frame separates a bad MediaPipe observation from a tracking reaction:
/// each row is `seq,captured_ns,side,role,x,y,z,visibility,presence,score`
/// with one row per shoulder/elbow/wrist plus a hand-score row per side.
#[cfg(debug_assertions)]
fn log_pose_frame(frame: &PoseArmFrame, file: &mut Option<std::fs::File>) {
    use std::fmt::Write as _;
    use std::io::Write as _;
    if file.is_none()
        && let Ok(mut handle) = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("mediapipe_pose_debug.csv")
    {
        let _ = writeln!(
            handle,
            "seq,captured_ns,side,role,x,y,z,visibility,presence,score"
        );
        *file = Some(handle);
    }
    let score =
        |arm: &vtuber_core::arm_tracking::ArmLandmarks| arm.hand.and_then(|hand| hand.score);
    match &frame.observation {
        Some(observation) => bevy::log::info!(
            target: "palm_trace",
            "pose seq={} captured_ns={} left_hand_score={:?} right_hand_score={:?}",
            frame.source_seq.0,
            frame.captured_at.0,
            score(&observation.left),
            score(&observation.right),
        ),
        None => {
            bevy::log::info!(
                target: "palm_trace",
                "pose seq={} no_person",
                frame.source_seq.0,
            );
            if let Some(file) = file.as_mut() {
                let _ = writeln!(file, "{},{}", frame.source_seq.0, frame.captured_at.0);
            }
            return;
        }
    }
    let Some(file) = file.as_mut() else {
        return;
    };
    let Some(observation) = &frame.observation else {
        return;
    };
    let number =
        |value: Option<f32>| value.map_or_else(|| "-".to_string(), |value| format!("{value:.4}"));
    let mut rows = String::new();
    for (side, arm) in [("L", &observation.left), ("R", &observation.right)] {
        for (role, point) in [
            ("shoulder", &arm.shoulder),
            ("elbow", &arm.elbow),
            ("wrist", &arm.wrist),
        ] {
            let [x, y, z] = point.meters;
            let _ = writeln!(
                rows,
                "{},{},{side},{role},{x:.4},{y:.4},{z:.4},{},{},-",
                frame.source_seq.0,
                frame.captured_at.0,
                number(point.visibility),
                number(point.presence),
            );
        }
        let _ = writeln!(
            rows,
            "{},{},{side},hand,-,-,-,-,-,{}",
            frame.source_seq.0,
            frame.captured_at.0,
            number(arm.hand.and_then(|hand| hand.score)),
        );
        // The palm-plane source: wrist, index MCP, and pinky MCP world points
        // from the Hand Landmarker, so the observed normal can be recomputed
        // offline against the avatar trace.
        if let Some(hand) = arm.hand {
            for (role, landmark) in [
                ("hand_wrist", 0usize),
                ("hand_index", 5),
                ("hand_pinky", 17),
            ] {
                let Some(point) = hand.landmarks.get(landmark) else {
                    continue;
                };
                let [x, y, z] = point.meters;
                let _ = writeln!(
                    rows,
                    "{},{},{side},{role},{x:.4},{y:.4},{z:.4},{},{},-",
                    frame.source_seq.0,
                    frame.captured_at.0,
                    number(point.visibility),
                    number(point.presence),
                );
            }
        }
    }
    let _ = file.write_all(rows.as_bytes());
}

/// Applies the persisted arm-tracking switch at startup.
pub fn restore_pose_settings_system(
    settings: Res<crate::settings::ArmPoseSettings>,
    mut pose: ResMut<PoseRuntime>,
) {
    pose.set_enabled(settings.arm_tracking_enabled());
}

/// Starts or stops the Pose worker with the capture session.
///
/// The Pose worker never runs while capture is not active, so a stopped camera
/// cannot leave stale observations behind.
pub fn pose_worker_bridge_system(
    mut pose: ResMut<PoseRuntime>,
    capture: Res<CaptureRuntime>,
    _orchestrator: Res<Orchestrator>,
) {
    let capture_active = matches!(
        capture.state(),
        vtuber_camera::CaptureServiceState::Starting | vtuber_camera::CaptureServiceState::Running
    );
    if pose.enabled() && capture_active {
        if let Err(error) = pose.ensure_running() {
            error!("pose worker could not be started: {error}");
        }
    } else if pose.is_running() {
        pose.stop();
    }
}

/// Publishes the latest observed frame into the avatar's tracked control.
pub fn read_pose_output_system(
    lifecycle: Res<vtuber_avatar::AvatarLifecycle>,
    mut pose: ResMut<PoseRuntime>,
    mut tracked: ResMut<TrackedArmControl>,
) {
    if !pose.enabled() {
        tracked.frame = None;
        tracked.generation = None;
        return;
    }
    tracked.generation = Some(lifecycle.current_generation());
    if let Some(control) = pose.read_latest() {
        tracked.frame = Some(control);
    }
}

/// Selects tracked authority only while the Pose worker is actually running.
pub fn pose_source_selection_system(
    pose: Res<PoseRuntime>,
    mut selection: ResMut<ArmSourceSelection>,
) {
    let mode = if pose.enabled() && pose.is_running() {
        vtuber_avatar::ArmPoseSourceKind::TrackedPose
    } else {
        vtuber_avatar::ArmPoseSourceKind::VirtualHandAnchor
    };
    if selection.mode != mode {
        selection.mode = mode;
    }
}
