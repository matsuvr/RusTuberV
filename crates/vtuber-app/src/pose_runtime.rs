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
use vtuber_inference::{InferenceWorkerResult, MediaPipeTaskSource, SharedStatus, run_pose_worker};
use vtuber_tracking::arm_tracking::{ArmTrackingProfile, ArmTrackingState, step_arm_tracking};

use crate::capture_runtime::CaptureRuntime;
use crate::orchestrator::Orchestrator;

/// The packaged Pose task filename.
const POSE_TASK_FILE: &str = "pose_landmarker_full.task";

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
    output_generation: u64,
    held_frame: Option<PoseArmFrame>,
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
            output_generation: 0,
            held_frame: None,
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
    pub fn ensure_running(&mut self) {
        if self.worker.is_some() {
            return;
        }
        let task = if self.task_path.is_file() {
            MediaPipeTaskSource::Path(self.task_path.clone())
        } else {
            MediaPipeTaskSource::Embedded
        };
        let status = Arc::clone(&self.status);
        let frame_slot = Arc::clone(&self.frame_slot);
        let output_slot = Arc::clone(&self.output_slot);
        let worker = WorkerHandle::spawn("pose-worker", move |stop| {
            run_pose_worker(stop, status, frame_slot, output_slot, &task)
        });
        self.worker = Some(worker);
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
        pose.ensure_running();
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
