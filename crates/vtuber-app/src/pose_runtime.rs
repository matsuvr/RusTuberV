//! Application bridge for the observed-arm Pose worker.
//!
//! The capture worker fans the same camera frame out to the face and Pose
//! slots; this resource owns the Pose worker, advances the pure arm-tracking
//! state on new observations, and publishes the resulting control frame into
//! the avatar's [`vtuber_avatar::TrackedArmControl`] resource.

use std::sync::Arc;

use bevy::prelude::*;
use vtuber_avatar::{ArmSourceSelection, AvatarGeneration, TrackedArmControl};
use vtuber_core::arm_tracking::{ArmControlFrame, PoseArmFrame};
use vtuber_core::{LatestSlot, MonoTimeNs, ReadResult, VideoFrame, WorkerHandle, monotonic_now};
use vtuber_inference::{
    FailureStage, InferenceError, InferenceWorkerResult, InferenceWorkerState, MediaPipeTaskSource,
    SharedStatus, WorkerFailure, run_pose_worker,
};
use vtuber_tracking::arm_tracking::{ArmTrackingProfile, ArmTrackingState, step_arm_tracking};

use crate::capture_runtime::CaptureRuntime;
use crate::orchestrator::{Orchestrator, OrchestratorError};

/// The Hand Landmarker world landmarks written to the debug CSV.
///
/// The whole hand is logged rather than only the three points the palm plane
/// is built from, because the four finger chains are landmarks 5 through 20
/// and a middle-finger gesture cannot be diagnosed from the palm alone.
#[cfg(debug_assertions)]
const DEBUG_HAND_LANDMARKS: std::ops::RangeInclusive<usize> = 0..=20;

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
    last_avatar_generation: AvatarGeneration,
    observations_after: MonoTimeNs,
    power: vtuber_platform::PowerState,
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
                .join(vtuber_inference::MediaPipeTask::Pose.file()),
            hand_task_path: project_root
                .join("assets")
                .join("models")
                .join(vtuber_inference::MediaPipeTask::Hand.file()),
            output_generation: 0,
            held_frame: None,
            last_avatar_generation: AvatarGeneration::default(),
            observations_after: MonoTimeNs(0),
            power: vtuber_platform::power_state(),
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
    ///
    /// An explicit off-then-on reselection clears a retained worker failure so
    /// the next bridge tick starts a fresh worker. Re-assigning the
    /// same value does not, so a failing spawn is never retried implicitly.
    pub fn set_enabled(&mut self, enabled: bool) {
        let reselected = enabled && !self.enabled;
        self.enabled = enabled;
        if reselected {
            self.clear_worker_failure();
        }
    }

    /// Drops calibration and observation state for a fresh calibration.
    pub fn recalibrate(&mut self) {
        self.tracking.reset();
        self.invalidate_session(monotonic_now());
    }

    /// Whether the Pose worker thread is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(|worker| !worker.is_finished())
            && self.lock_status().state != InferenceWorkerState::Failed
    }

    /// Snapshot of the combined Pose/Hand worker timings and progress.
    #[must_use]
    pub fn status(&self) -> vtuber_inference::InferenceWorkerStatus {
        self.lock_status().clone()
    }

    /// Reaps an exited worker and clears its retained observations.
    fn poll_worker_exit(&mut self) -> Result<(), InferenceError> {
        let Some(result) =
            vtuber_inference::completion::reap_finished_worker(&mut self.worker, &self.status)
        else {
            return Ok(());
        };
        self.recalibrate();
        result.map(|_| ())
    }

    /// Starts the Pose worker, loading the task bundle inside the worker thread.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::WorkerSpawnFailed`] if the OS refused to spawn
    /// the thread. No worker is retained on failure, and the failure is recorded
    /// in the shared status, so the bridge does not retry the same request; an
    /// explicit arm-tracking off-then-on reselection clears it.
    pub fn ensure_running(&mut self) -> Result<(), InferenceError> {
        if self.worker.is_some() {
            return Ok(());
        }
        let task = MediaPipeTaskSource::from_packaged_path(self.task_path.clone());
        let hand_task = MediaPipeTaskSource::from_packaged_path(self.hand_task_path.clone());
        let status = Arc::clone(&self.status);
        let frame_slot = Arc::clone(&self.frame_slot);
        let output_slot = Arc::clone(&self.output_slot);
        let worker = WorkerHandle::spawn("pose-worker", move |stop| {
            run_pose_worker(stop, status, frame_slot, output_slot, &task, &hand_task)
        });
        // The status lock is not held across the spawn: the worker publishes its
        // own state as soon as it runs.
        self.finish_spawn(worker)
    }

    /// Retains a successfully spawned worker or records the spawn failure.
    ///
    /// The status is left untouched on success because the worker may already
    /// have published `LoadingModel` or `Running` before this runs.
    fn finish_spawn(
        &mut self,
        result: std::io::Result<WorkerHandle<InferenceWorkerResult>>,
    ) -> Result<(), InferenceError> {
        match result {
            Ok(worker) => {
                self.worker = Some(worker);
                Ok(())
            }
            Err(spawn_error) => {
                let error = InferenceError::WorkerSpawnFailed {
                    kind: spawn_error.kind(),
                    message: spawn_error.to_string(),
                };
                self.lock_status().record_failure(
                    vtuber_core::monotonic_now(),
                    FailureStage::WorkerSpawn,
                    error.clone(),
                );
                Err(error)
            }
        }
    }

    /// Clears a retained failure so an explicit reselection can retry.
    fn clear_worker_failure(&mut self) {
        let mut status = self.lock_status();
        if status.last_failure.is_none() {
            return;
        }
        status.last_failure = None;
        status.transition_to(InferenceWorkerState::Idle);
    }

    /// Locks the shared status, recovering from a poisoned worker panic.
    fn lock_status(&self) -> std::sync::MutexGuard<'_, vtuber_inference::InferenceWorkerStatus> {
        self.status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Stops the Pose worker and drops its observation state.
    pub fn stop(&mut self) -> Result<(), InferenceError> {
        let result = if let Some(worker) = self.worker.take() {
            worker.stop();
            worker.join()
        } else {
            vtuber_core::WorkerResult::Completed(InferenceWorkerResult::default())
        };
        self.recalibrate();
        vtuber_inference::completion::finish_worker_join(result, &self.status).map(|_| ())
    }

    fn invalidate_session(&mut self, now: MonoTimeNs) {
        self.tracking.reset_temporal();
        self.output_generation = 0;
        self.held_frame = None;
        self.output_slot.clear();
        // A worker can finish an in-flight camera frame after the slot is cleared.
        self.observations_after = now;
    }

    /// Advances the pure tracking state by one render tick.
    ///
    /// Like the face pipeline, the latest completed inference is retained and
    /// re-fed while no new result exists, so the arm smoother advances on the
    /// render clock and the hands move continuously between camera frames.
    fn read_latest(
        &mut self,
        generation: AvatarGeneration,
        now: MonoTimeNs,
    ) -> Option<ArmControlFrame> {
        let power = vtuber_platform::power_state();
        if power != self.power {
            self.power = power;
            self.invalidate_session(now);
        }
        if power.sleeping {
            return None;
        }
        if generation != self.last_avatar_generation {
            self.last_avatar_generation = generation;
            self.invalidate_session(now);
        }
        if let Some(ReadResult::New {
            generation,
            value: frame,
        }) = self.output_slot.try_read_after(self.output_generation)
        {
            self.output_generation = generation;
            if frame.captured_at >= self.observations_after {
                #[cfg(debug_assertions)]
                log_pose_frame(&frame, &mut self.debug_log);
                self.held_frame = Some(frame);
            }
        }
        if self.held_frame.as_ref().is_some_and(|frame| {
            std::time::Duration::from_nanos(now.0.saturating_sub(frame.captured_at.0))
                > crate::tracking_runtime::INFERENCE_SILENCE_TIMEOUT
        }) {
            self.held_frame = None;
        }
        let profile = self.profile;
        let (next, control) =
            step_arm_tracking(&self.tracking, self.held_frame.as_ref(), now, &profile);
        self.tracking = next;
        control
    }
}

impl Drop for PoseRuntime {
    fn drop(&mut self) {
        // Best-effort cleanup; explicit stop reports errors to its caller.
        let _ = self.stop();
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
/// each row is `seq,captured_ns,side,role,x,y,z,visibility,presence,handedness`
/// with one row per shoulder/elbow/wrist, one per Hand Landmarker landmark as
/// `hand_NN`, and a handedness row per side. The handedness column is the
/// left/right label confidence, not a detection quality, so it is logged for
/// diagnosis only and never gates tracking.
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
            "seq,captured_ns,side,role,x,y,z,visibility,presence,handedness"
        );
        *file = Some(handle);
    }
    let handedness = |arm: &vtuber_core::arm_tracking::ArmLandmarks| {
        arm.hand.and_then(|hand| hand.handedness_score)
    };
    match &frame.observation {
        Some(observation) => bevy::log::info!(
            target: "palm_trace",
            "pose seq={} captured_ns={} left_handedness={:?} right_handedness={:?}",
            frame.source_seq.0,
            frame.captured_at.0,
            handedness(&observation.left),
            handedness(&observation.right),
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
            number(arm.hand.and_then(|hand| hand.handedness_score)),
        );
        // The Hand Landmarker world points, so the observed palm normal and every
        // finger's flexion can be recomputed offline against the avatar trace.
        // Landmarks 0, 5, and 17 give the palm plane; 5 through 20 are the four
        // finger chains, so these same rows also settle whether a standalone
        // middle finger was ever observed.
        if let Some(hand) = arm.hand {
            for landmark in DEBUG_HAND_LANDMARKS {
                let Some(point) = hand.landmarks.get(landmark) else {
                    continue;
                };
                let [x, y, z] = point.meters;
                let _ = writeln!(
                    rows,
                    "{},{},{side},hand_{landmark:02},{x:.4},{y:.4},{z:.4},{},{},-",
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
    settings: Res<crate::settings::AppSettings>,
    mut pose: ResMut<PoseRuntime>,
) {
    pose.set_enabled(settings.arm_tracking_enabled());
}

/// Whether the bridge should ask for a Pose worker on this tick.
///
/// A retained failure blocks automatic restart until an explicit arm-tracking
/// off-then-on reselection starts another attempt.
fn should_start_pose(
    enabled: bool,
    capture_active: bool,
    has_worker: bool,
    last_failure: Option<&WorkerFailure>,
) -> bool {
    enabled && capture_active && !has_worker && last_failure.is_none()
}

/// Starts or stops the Pose worker with the capture session.
///
/// A temporarily suspended/reconnecting camera keeps its CPU worker waiting
/// on the bounded input slot, avoiding a join/model reload on the UI thread.
/// Sleep transitions separately invalidate observations and temporal state.
/// A worker failure is reported once
/// through the existing error presentation and then retained, leaving face
/// tracking and capture untouched.
pub fn pose_worker_bridge_system(
    mut pose: ResMut<PoseRuntime>,
    capture: Res<CaptureRuntime>,
    mut orchestrator: ResMut<Orchestrator>,
) {
    if let Err(error) = pose.poll_worker_exit() {
        orchestrator.set_last_error(Some(OrchestratorError::PoseInferenceFailed(
            error.to_string(),
        )));
    }
    let capture_active = matches!(
        capture.state(),
        vtuber_camera::CaptureServiceState::Starting
            | vtuber_camera::CaptureServiceState::Running
            | vtuber_camera::CaptureServiceState::Suspended
            | vtuber_camera::CaptureServiceState::Reconnecting
    );
    // Read the retained failure under a short status lock, then spawn outside it.
    let start = should_start_pose(
        pose.enabled(),
        capture_active,
        pose.is_running(),
        pose.lock_status().last_failure.as_ref(),
    );
    if start {
        if let Err(error) = pose.ensure_running() {
            error!("pose worker could not be started: {error}");
            orchestrator.set_last_error(Some(OrchestratorError::PoseInferenceFailed(
                error.to_string(),
            )));
        }
    } else if (!pose.enabled() || !capture_active) && pose.is_running() {
        // Only a deselected arm tracking or an inactive capture stops the
        // worker. Not starting this tick says nothing about a running worker.
        if let Err(error) = pose.stop() {
            error!("pose worker shutdown failed: {error}");
            orchestrator.set_last_error(Some(OrchestratorError::PoseInferenceFailed(
                error.to_string(),
            )));
        }
    }
}

/// Publishes the latest observed frame into the avatar's tracked control.
pub fn read_pose_output_system(
    lifecycle: Res<vtuber_avatar::AvatarLifecycle>,
    mut pose: ResMut<PoseRuntime>,
    mut tracked: ResMut<TrackedArmControl>,
) {
    if !pose.enabled() || !pose.is_running() {
        tracked.frame = None;
        tracked.generation = None;
        return;
    }
    let generation = lifecycle.current_generation();
    tracked.frame = pose
        .read_latest(generation, monotonic_now())
        .map(vtuber_tracking::gesture_safety::suppress_isolated_middle_extension);
    tracked.generation = tracked.frame.as_ref().map(|_| generation);
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

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use vtuber_camera::device::{CameraDescriptor, CameraRequest};

    /// A Pose runtime holding a recorded spawn failure, as if the OS had
    /// refused the thread. No OS thread is created.
    fn pose_with_spawn_failure() -> PoseRuntime {
        let mut pose = PoseRuntime::new(std::path::PathBuf::from("."));
        pose.set_enabled(true);
        let result: std::io::Result<WorkerHandle<InferenceWorkerResult>> =
            Err(std::io::Error::from(std::io::ErrorKind::WouldBlock));
        let error = pose
            .finish_spawn(result)
            .expect_err("a failed spawn reports the OS error");
        assert!(matches!(error, InferenceError::WorkerSpawnFailed { .. }));
        pose
    }

    /// A capture runtime whose mock device is open, so the bridge sees an
    /// active capture session.
    fn active_mock_capture() -> CaptureRuntime {
        let mut capture = CaptureRuntime::default();
        capture
            .controller_mut()
            .start_worker(vtuber_camera::mock::MockBackend::default())
            .expect("mock capture worker starts");
        capture
            .controller_mut()
            .select_and_start(
                CameraDescriptor {
                    id: "mock-0".into(),
                    label: "Mock Camera".into(),
                },
                CameraRequest::default(),
            )
            .expect("mock capture start");
        capture
    }

    /// A Pose worker that runs no model and only waits for its stop token.
    fn test_pose_worker() -> WorkerHandle<InferenceWorkerResult> {
        WorkerHandle::spawn("test-pose-worker", |stop| {
            while !stop.is_stopped() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            InferenceWorkerResult::default()
        })
        .expect("test pose worker spawns")
    }

    /// A tick with a retained spawn failure starts no worker and reports no
    /// error, so the notification cannot reappear every frame.
    fn assert_bridge_neither_retried_nor_reported(app: &App) {
        let pose = app.world().resource::<PoseRuntime>();
        assert!(!pose.is_running());
        assert!(pose.lock_status().last_failure.is_some());
        assert!(
            app.world()
                .resource::<Orchestrator>()
                .last_error()
                .is_none()
        );
    }

    #[test]
    fn missed_sleep_cycle_clears_held_and_in_flight_pose_results() {
        let mut pose = PoseRuntime::default();
        let frame = PoseArmFrame {
            source_seq: vtuber_core::FrameSeq(1),
            captured_at: MonoTimeNs(100),
            inference_finished_at: MonoTimeNs(110),
            observation: None,
        };
        pose.held_frame = Some(frame);
        pose.output_slot.publish(frame);
        pose.power.generation = pose.power.generation.wrapping_sub(1);
        assert!(
            pose.read_latest(AvatarGeneration::default(), MonoTimeNs(200))
                .is_none()
        );
        assert!(pose.held_frame.is_none());
        // A CPU inference started before suspend can complete after resume.
        pose.output_slot.publish(frame);
        assert!(
            pose.read_latest(AvatarGeneration::default(), MonoTimeNs(210))
                .is_none()
        );
        assert!(pose.held_frame.is_none());
    }

    #[test]
    fn pose_stop_reports_panic_and_clears_observation_state() {
        let mut pose = PoseRuntime::default();
        pose.output_generation = 123;
        pose.worker = Some(
            WorkerHandle::spawn("pose-panic-test", |_| {
                panic!("scripted worker panic");
            })
            .unwrap(),
        );
        assert_eq!(pose.stop(), Err(InferenceError::WorkerPanicked));
        assert!(!pose.is_running());
        assert_eq!(pose.output_generation, 0);
        assert!(pose.held_frame.is_none());
        assert!(matches!(
            pose.lock_status().last_failure.as_ref().map(|f| &f.error),
            Some(InferenceError::WorkerPanicked)
        ));
    }

    fn wait_for_pose_exit(pose: &PoseRuntime) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !pose.worker.as_ref().unwrap().is_finished() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    fn pose_session_reset_waits_for_new_capture_and_never_relabels_a_held_frame() {
        let mut pose = PoseRuntime::default();
        #[cfg(debug_assertions)]
        {
            pose.debug_log = Some(tempfile::tempfile().unwrap());
        }
        let first = AvatarGeneration(1);
        let second = AvatarGeneration(2);
        assert!(pose.read_latest(first, MonoTimeNs(10)).is_none());
        let frame = PoseArmFrame {
            source_seq: vtuber_core::FrameSeq(7),
            captured_at: MonoTimeNs(11),
            inference_finished_at: MonoTimeNs(12),
            observation: None,
        };
        assert!(pose.output_slot.publish(frame));
        let control = pose.read_latest(first, MonoTimeNs(13)).unwrap();
        assert_eq!(control.source_seq, frame.source_seq);
        assert_eq!(
            pose.read_latest(first, MonoTimeNs(14)).unwrap().captured_at,
            frame.captured_at
        );
        assert!(pose.read_latest(second, MonoTimeNs(15)).is_none());
        // An old camera frame finishes after the replacement reset.
        assert!(pose.output_slot.publish(PoseArmFrame {
            inference_finished_at: MonoTimeNs(16),
            ..frame
        }));
        assert!(pose.read_latest(second, MonoTimeNs(17)).is_none());
        assert!(pose.output_slot.publish(PoseArmFrame {
            captured_at: MonoTimeNs(18),
            ..frame
        }));
        assert!(pose.read_latest(second, MonoTimeNs(19)).is_some());

        pose.recalibrate();
        assert!(pose.read_latest(second, monotonic_now()).is_none());
        let captured_at = monotonic_now();
        assert!(pose.output_slot.publish(PoseArmFrame {
            captured_at,
            ..frame
        }));
        assert!(pose.read_latest(second, monotonic_now()).is_some());
        pose.stop().unwrap();
        assert!(pose.read_latest(second, monotonic_now()).is_none());
    }

    #[test]
    fn pose_poll_keeps_running_workers_and_reports_an_exited_panic_once() {
        let mut pose = PoseRuntime::default();
        pose.finish_spawn(Ok(test_pose_worker())).unwrap();
        assert_eq!(pose.poll_worker_exit(), Ok(()));
        assert!(pose.is_running());
        pose.stop().unwrap();

        pose.worker = Some(
            WorkerHandle::spawn("pose-poll-panic", |_| {
                panic!("scripted panic");
            })
            .unwrap(),
        );
        wait_for_pose_exit(&pose);
        assert!(!pose.is_running());
        assert_eq!(pose.poll_worker_exit(), Err(InferenceError::WorkerPanicked));
        assert!(pose.worker.is_none());
        assert_eq!(pose.poll_worker_exit(), Ok(()));
        assert_eq!(
            pose.lock_status().last_failure.as_ref().unwrap().stage,
            FailureStage::WorkerPanic
        );
    }

    #[test]
    fn completed_pose_load_failure_is_reported_once_and_releases_tracked_authority() {
        let mut pose = PoseRuntime::default();
        pose.set_enabled(true);
        let status = Arc::clone(&pose.status);
        let input = Arc::clone(&pose.frame_slot);
        let output = Arc::clone(&pose.output_slot);
        let missing = tempfile::tempdir()
            .unwrap()
            .path()
            .join("missing-pose.task");
        pose.worker = Some(
            WorkerHandle::spawn("pose-load-failure", move |stop| {
                let task = MediaPipeTaskSource::Path(missing);
                run_pose_worker(
                    stop,
                    status,
                    input,
                    output,
                    &task,
                    &MediaPipeTaskSource::Embedded,
                )
            })
            .unwrap(),
        );
        wait_for_pose_exit(&pose);
        assert_eq!(pose.lock_status().state, InferenceWorkerState::Failed);
        assert!(!pose.is_running());

        let mut app = App::new();
        app.insert_resource(pose)
            .insert_resource(active_mock_capture())
            .init_resource::<Orchestrator>()
            .insert_resource(ArmSourceSelection {
                mode: vtuber_avatar::ArmPoseSourceKind::TrackedPose,
                ..Default::default()
            })
            .add_systems(
                Update,
                (pose_worker_bridge_system, pose_source_selection_system).chain(),
            );
        app.update();
        assert!(matches!(
            app.world().resource::<Orchestrator>().last_error(),
            Some(OrchestratorError::PoseInferenceFailed(_))
        ));
        assert_eq!(
            app.world().resource::<ArmSourceSelection>().mode,
            vtuber_avatar::ArmPoseSourceKind::VirtualHandAnchor
        );
        assert!(app.world().resource::<PoseRuntime>().worker.is_none());
        assert_eq!(
            app.world()
                .resource::<PoseRuntime>()
                .lock_status()
                .last_failure
                .as_ref()
                .unwrap()
                .stage,
            FailureStage::ModelLoad
        );

        app.world_mut()
            .resource_mut::<Orchestrator>()
            .set_last_error(None);
        app.update();
        assert_bridge_neither_retried_nor_reported(&app);

        let mut pose = app.world_mut().resource_mut::<PoseRuntime>();
        pose.set_enabled(false);
        pose.set_enabled(true);
        assert!(pose.lock_status().last_failure.is_none());
        pose.finish_spawn(Ok(test_pose_worker())).unwrap();
        assert!(pose.is_running());
        pose.stop().unwrap();
        app.world_mut()
            .resource_mut::<CaptureRuntime>()
            .shutdown()
            .unwrap();
    }

    #[test]
    fn spawn_failure_is_recorded_and_reported_without_a_worker() {
        let mut pose = PoseRuntime::new(std::path::PathBuf::from("."));
        let result: std::io::Result<WorkerHandle<InferenceWorkerResult>> =
            Err(std::io::Error::other("simulated spawn failure"));
        let error = pose
            .finish_spawn(result)
            .expect_err("a failed spawn reports the OS error");

        assert!(!pose.is_running());
        let InferenceError::WorkerSpawnFailed { kind, message } = &error else {
            panic!("expected a spawn failure, got {error:?}");
        };
        assert_eq!(*kind, std::io::ErrorKind::Other);
        assert_eq!(message, "simulated spawn failure");

        let status = pose.lock_status();
        assert_eq!(status.state, InferenceWorkerState::Failed);
        let failure = status
            .last_failure
            .as_ref()
            .expect("the spawn failure is recorded");
        assert_eq!(failure.stage, FailureStage::WorkerSpawn);
        assert_eq!(failure.error, error);
    }

    #[test]
    fn should_start_pose_requires_enabled_capture_and_no_worker() {
        let failure = WorkerFailure {
            observed_at: vtuber_core::MonoTimeNs(1),
            stage: FailureStage::ModelLoad,
            error: InferenceError::LoadFailed("missing file".into()),
        };
        assert!(should_start_pose(true, true, false, None));
        assert!(!should_start_pose(false, true, false, None));
        assert!(!should_start_pose(true, false, false, None));
        assert!(!should_start_pose(true, true, true, None));
        assert!(!should_start_pose(true, true, false, Some(&failure)));
    }

    #[test]
    fn a_retained_spawn_failure_blocks_the_next_start() {
        let pose = pose_with_spawn_failure();
        let status = pose.lock_status();
        let failure = status
            .last_failure
            .as_ref()
            .expect("the spawn failure is recorded");
        assert!(!should_start_pose(true, true, false, Some(failure)));
    }

    #[test]
    fn only_an_off_then_on_reselection_clears_the_spawn_failure() {
        let mut pose = pose_with_spawn_failure();

        // Re-assigning the same value is not a retry request.
        pose.set_enabled(true);
        let status = pose.lock_status();
        let failure = status
            .last_failure
            .as_ref()
            .expect("the spawn failure is recorded");
        assert!(!should_start_pose(true, true, false, Some(failure)));
        drop(status);

        pose.set_enabled(false);
        assert!(pose.lock_status().last_failure.is_some());

        pose.set_enabled(true);
        let status = pose.lock_status();
        assert!(status.last_failure.is_none());
        assert_eq!(status.state, InferenceWorkerState::Idle);
        assert!(should_start_pose(
            true,
            true,
            false,
            status.last_failure.as_ref()
        ));
    }

    #[test]
    fn the_bridge_does_not_retry_or_re_report_a_retained_spawn_failure() {
        let mut app = App::new();
        app.insert_resource(pose_with_spawn_failure())
            .insert_resource(active_mock_capture())
            .init_resource::<Orchestrator>()
            .add_systems(Update, pose_worker_bridge_system);

        app.update();
        assert_bridge_neither_retried_nor_reported(&app);
        app.update();
        assert_bridge_neither_retried_nor_reported(&app);

        app.world_mut()
            .resource_mut::<CaptureRuntime>()
            .shutdown()
            .unwrap();
    }

    #[test]
    fn running_pose_worker_is_kept_across_active_bridge_ticks() {
        let worker = test_pose_worker();
        let stop_token = worker.stop_token();
        let mut pose = PoseRuntime::new(std::path::PathBuf::from("."));
        pose.set_enabled(true);
        // A worker that already reached Running: the successful spawn must not
        // overwrite the state it published.
        pose.lock_status()
            .transition_to(InferenceWorkerState::Running);
        pose.finish_spawn(Ok(worker))
            .expect("a successful spawn retains the worker");

        let mut app = App::new();
        app.insert_resource(pose)
            .insert_resource(active_mock_capture())
            .init_resource::<Orchestrator>()
            .add_systems(Update, pose_worker_bridge_system);

        for _ in 0..3 {
            app.update();
            let pose = app.world().resource::<PoseRuntime>();
            assert!(pose.is_running());
            assert!(!stop_token.is_stopped());
            assert_eq!(pose.lock_status().state, InferenceWorkerState::Running);
            assert!(
                app.world()
                    .resource::<Orchestrator>()
                    .last_error()
                    .is_none()
            );
        }

        app.world_mut()
            .resource_mut::<PoseRuntime>()
            .stop()
            .unwrap();
        assert!(stop_token.is_stopped());
        app.world_mut()
            .resource_mut::<CaptureRuntime>()
            .shutdown()
            .unwrap();
    }

    #[test]
    fn bridge_stops_pose_only_when_tracking_or_capture_is_disabled() {
        // Arm tracking off while the capture runs.
        let worker = test_pose_worker();
        let stop_token = worker.stop_token();
        let mut pose = PoseRuntime::new(std::path::PathBuf::from("."));
        pose.set_enabled(true);
        pose.finish_spawn(Ok(worker)).expect("spawn");
        pose.set_enabled(false);
        let mut app = App::new();
        app.insert_resource(pose)
            .insert_resource(active_mock_capture())
            .init_resource::<Orchestrator>()
            .add_systems(Update, pose_worker_bridge_system);

        app.update();
        let pose = app.world().resource::<PoseRuntime>();
        assert!(!pose.is_running());
        assert!(stop_token.is_stopped());
        // A later tick must not start it again while it stays deselected.
        app.update();
        assert!(!app.world().resource::<PoseRuntime>().is_running());
        app.world_mut()
            .resource_mut::<CaptureRuntime>()
            .shutdown()
            .unwrap();

        // Arm tracking on while the capture is idle.
        let worker = test_pose_worker();
        let stop_token = worker.stop_token();
        let mut pose = PoseRuntime::new(std::path::PathBuf::from("."));
        pose.set_enabled(true);
        pose.finish_spawn(Ok(worker)).expect("spawn");
        let mut app = App::new();
        app.insert_resource(pose)
            .insert_resource(CaptureRuntime::default())
            .init_resource::<Orchestrator>()
            .add_systems(Update, pose_worker_bridge_system);

        app.update();
        let pose = app.world().resource::<PoseRuntime>();
        assert!(!pose.is_running());
        assert!(stop_token.is_stopped());
        app.update();
        assert!(!app.world().resource::<PoseRuntime>().is_running());
    }
}
