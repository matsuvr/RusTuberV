//! Main-thread bridge from inference observations to the pure tracking core.

use std::time::Duration;

use bevy::prelude::*;
use vtuber_core::metrics::RateCounter;
use vtuber_core::{AvatarControlFrame, FaceTrackingOutcome, FrameSeq, MonoTimeNs, TrackingState};
use vtuber_tracking::{AutoNeutralCollector, AutoNeutralState, PipelineConfig, TrackingPipeline};

use crate::diagnostics::DiagnosticsSnapshot;
use crate::inference_runtime::InferenceRuntime;
use crate::orchestrator::{CalibrationRequest, Orchestrator};
use crate::ui_model::{CalibrationViewModel, TrackingState as UiTrackingState, UiViewModel};

/// Maximum capture age of either a face or a no-face inference result.
///
/// Distinct `NoFace` outcomes drive confidence at inference cadence. This
/// independent watchdog handles a stalled worker, including one whose last
/// result was `NoFace`, without inventing further inference outcomes.
pub(crate) const INFERENCE_SILENCE_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Clone, Debug, PartialEq)]
enum ObservationDispatch<'a> {
    /// No new inference result and the last result is still fresh.
    NoUpdate,
    /// A new, fresh face or no-face result is ready for tracking.
    Outcome(&'a FaceTrackingOutcome),
    /// No result is available for this session, or the watchdog expired.
    Unavailable,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ObservationGate {
    last_source_seq: Option<FrameSeq>,
    observations_after: Option<MonoTimeNs>,
}

impl ObservationGate {
    /// Clears the source-sequence boundary at a capture/session reset.
    ///
    /// Capture sequence numbers are owned by the capture session. A new
    /// session may legally start at the same sequence as the previous one,
    /// so retaining the old value would suppress its first face observation.
    fn reset(&mut self, now: MonoTimeNs) {
        self.last_source_seq = None;
        self.observations_after = Some(now);
    }

    fn dispatch<'a>(
        &mut self,
        latest: Option<&'a FaceTrackingOutcome>,
        now: MonoTimeNs,
    ) -> ObservationDispatch<'a> {
        let Some(outcome) = latest else {
            return ObservationDispatch::Unavailable;
        };
        let captured_at = match outcome {
            FaceTrackingOutcome::Face(sample) => sample.captured_at,
            FaceTrackingOutcome::NoFace { captured_at, .. } => *captured_at,
        };
        if self
            .observations_after
            .is_some_and(|boundary| captured_at < boundary)
        {
            return ObservationDispatch::Unavailable;
        }

        let age = Duration::from_nanos(now.0.saturating_sub(captured_at.0));
        if age > INFERENCE_SILENCE_TIMEOUT {
            return ObservationDispatch::Unavailable;
        }
        if self.last_source_seq == Some(outcome.source_seq()) {
            return ObservationDispatch::NoUpdate;
        }
        self.last_source_seq = Some(outcome.source_seq());
        ObservationDispatch::Outcome(outcome)
    }
}

/// Tracking domain state owned by the Bevy main thread.
#[derive(Resource)]
pub struct TrackingRuntime {
    pipeline: TrackingPipeline,
    auto_neutral: AutoNeutralCollector,
    recenter_requested: bool,
    last_update: Option<MonoTimeNs>,
    power: vtuber_platform::PowerState,
    last_avatar_generation: vtuber_avatar::AvatarGeneration,
    last_recenter_error: Option<String>,
    observation_gate: ObservationGate,
    /// Last numbered result, re-fed on render ticks without adding evidence.
    /// Cleared only when unavailable; explicit `NoFace` retains its identity.
    held_outcome: Option<FaceTrackingOutcome>,
    /// Whether the avatar bridge may retain the most recently published
    /// control frame for the current capture session.
    pub control_active: bool,
    /// Pending control paired with the avatar generation it was produced for.
    pub latest_control: Option<(vtuber_avatar::AvatarGeneration, AvatarControlFrame)>,
    /// Diagnostic string caches, rebuilt only when their source value changes.
    diagnostics_cache: TrackingDiagnosticsCache,
}

impl Default for TrackingRuntime {
    fn default() -> Self {
        let config = PipelineConfig::default();
        #[expect(
            clippy::expect_used,
            reason = "`PipelineConfig::default()` is a valid configuration by definition and the repository tests pin it"
        )]
        let pipeline = TrackingPipeline::try_new(config)
            .expect("default tracking configuration is an internal invariant");
        Self {
            pipeline,
            auto_neutral: AutoNeutralCollector::new(),
            recenter_requested: false,
            last_update: None,
            power: vtuber_platform::power_state(),
            last_avatar_generation: vtuber_avatar::AvatarGeneration::default(),
            last_recenter_error: None,
            observation_gate: ObservationGate::default(),
            held_outcome: None,
            control_active: false,
            latest_control: None,
            diagnostics_cache: TrackingDiagnosticsCache::default(),
        }
    }
}

impl TrackingRuntime {
    /// Invalidates observations, pending output and time state together.
    /// Camera calibration is retained unless the caller explicitly clears it.
    fn invalidate_session(&mut self, now: MonoTimeNs) {
        self.reset_control_state();
        self.discard_pending_observations(now);
    }

    /// An input fault invalidates observations, not the displayed pose or its
    /// spring state. A retry must supply a newly captured observation.
    fn discard_pending_observations(&mut self, now: MonoTimeNs) {
        self.observation_gate.reset(now);
        self.held_outcome = None;
    }

    // A changed neutral keeps the newly accepted observation, but old controls
    // and filter time must not outlive the reference that produced them.
    fn reset_control_state(&mut self) {
        self.pipeline.reset();
        self.latest_control = None;
        self.control_active = false;
        self.last_update = None;
    }
}

/// Loads the optional validated eye-closure profile at startup.
///
/// A missing or invalid profile leaves correction disabled. The runner never
/// substitutes a default threshold and never rewrites an incompatible
/// profile: it logs the reason and continues without correction.
pub fn load_eye_closure_profile_system(mut tracking: ResMut<TrackingRuntime>) {
    tracking.pipeline.clear_eye_closure();
    let Some(path) = crate::settings::default_eye_closure_profile_path() else {
        return;
    };
    if !path.is_file() {
        return;
    }
    match crate::settings::load_eye_closure_thresholds(&path) {
        Ok(Some(thresholds)) => tracking.pipeline.set_eye_closure_thresholds(thresholds),
        Ok(None) => {}
        Err(error) => bevy::log::warn!("eye-closure profile ignored: {error}"),
    }
}

/// Rebuilds a plain `String` diagnostic only when a `Copy` state value changes.
fn set_debug_cached<T: Copy + PartialEq + std::fmt::Debug>(
    cache: &mut Option<(T, String)>,
    value: T,
    target: &mut String,
) {
    if !matches!(cache, Some((cached, _)) if *cached == value) {
        *target = format!("{value:?}");
        if let Some((_, text)) = cache.as_mut() {
            *text = target.clone();
        } else {
            *cache = Some((value, target.clone()));
        }
    }
}

/// Rebuilds an `Option<String>` diagnostic only when a `Copy` state value changes.
fn set_debug_cached_opt<T: Copy + PartialEq + std::fmt::Debug>(
    cache: &mut Option<(T, String)>,
    value: T,
    target: &mut Option<String>,
) {
    if !matches!(cache, Some((cached, _)) if *cached == value) {
        let text = format!("{value:?}");
        *target = Some(text.clone());
        *cache = Some((value, text));
    }
}

/// Per-frame diagnostic strings rebuilt only when their source value changes.
#[derive(Default)]
struct TrackingDiagnosticsCache {
    tracking_state: Option<(TrackingState, String)>,
    auto_neutral_state: Option<(AutoNeutralState, String)>,
}

fn input_failure(
    state: crate::orchestrator::PipelineState,
    error: Option<&crate::orchestrator::OrchestratorError>,
) -> bool {
    use crate::orchestrator::{OrchestratorError, PipelineState};
    matches!(state, PipelineState::Stopping | PipelineState::Failed)
        && matches!(
            error,
            Some(OrchestratorError::CameraFailed(_) | OrchestratorError::InferenceFailed(_))
        )
}

/// Applies calibration intents and processes one latest inference result.
pub fn tracking_bridge_system(
    mut tracking: ResMut<TrackingRuntime>,
    inference: Res<InferenceRuntime>,
    lifecycle: Res<vtuber_avatar::AvatarLifecycle>,
    mut orchestrator: ResMut<Orchestrator>,
    mut view_model: ResMut<UiViewModel>,
    mut diagnostics: ResMut<DiagnosticsSnapshot>,
    mut tracking_rate: Local<Option<RateCounter>>,
) {
    let now = vtuber_core::monotonic_now();
    let power = vtuber_platform::power_state();
    if power != tracking.power {
        tracking.power = power;
        tracking.invalidate_session(now);
    }
    if lifecycle.current_generation() != tracking.last_avatar_generation {
        tracking.last_avatar_generation = lifecycle.current_generation();
        // The camera neutral is independent of the avatar model. Preserve it
        // across replacement, but always reset the smoothing and recovery
        // state so a new avatar cannot receive a stale frame.
        tracking.invalidate_session(now);
    }

    let pipeline_state = orchestrator.pipeline_state();
    let capture_inactive = !orchestrator.capture_desired()
        && matches!(
            pipeline_state,
            crate::orchestrator::PipelineState::Idle | crate::orchestrator::PipelineState::Stopping
        );
    let input_failed = input_failure(pipeline_state, orchestrator.last_error());
    if ((capture_inactive || pipeline_state == crate::orchestrator::PipelineState::Failed)
        && !input_failed)
        || power.sleeping
    {
        tracking.invalidate_session(now);
        tracking.recenter_requested = false;
        view_model.tracking = crate::ui_model::TrackingViewModel::default();
        view_model.calibration = calibration_view(&tracking);
        set_debug_cached(
            &mut tracking.diagnostics_cache.tracking_state,
            TrackingState::Starting,
            &mut diagnostics.tracking_state,
        );
        if diagnostics.tracking_backend.as_deref() != Some("mediapipe-face-landmarker") {
            diagnostics.tracking_backend = Some("mediapipe-face-landmarker".into());
        }
        if diagnostics.tracking_contract.as_deref()
            != Some("478 landmarks / 52 blendshapes / pose matrix")
        {
            diagnostics.tracking_contract =
                Some("478 landmarks / 52 blendshapes / pose matrix".into());
        }
        let auto_neutral_state = tracking.auto_neutral.state();
        set_debug_cached_opt(
            &mut tracking.diagnostics_cache.auto_neutral_state,
            auto_neutral_state,
            &mut diagnostics.auto_neutral_state,
        );
        diagnostics.face_tracking_calibration_ready = Some(matches!(
            tracking.auto_neutral.state(),
            vtuber_tracking::AutoNeutralState::Ready
        ));
        return;
    }

    if let Some(request) = orchestrator.take_calibration_request() {
        match request {
            CalibrationRequest::Begin | CalibrationRequest::Retry => {
                // Calibration is instant in the MediaPipe path: the next
                // valid face becomes the new neutral reference.
                tracking.auto_neutral.reset();
                tracking.invalidate_session(now);
                tracking.recenter_requested = true;
                tracking.last_recenter_error = None;
            }
            CalibrationRequest::Cancel => tracking.recenter_requested = false,
        }
    }

    let dt = tracking
        .last_update
        .map(|last| Duration::from_nanos(now.0.saturating_sub(last.0).min(250_000_000)))
        .unwrap_or_else(|| Duration::from_nanos(33_333_333));
    tracking.last_update = Some(now);

    if input_failed {
        tracking.discard_pending_observations(now);
    }
    let dispatch = tracking.observation_gate.dispatch(
        if input_failed {
            None
        } else {
            inference.latest_face_outcome.as_ref()
        },
        now,
    );
    let neutral_sample = match dispatch {
        ObservationDispatch::Outcome(FaceTrackingOutcome::Face(sample)) => Some(sample),
        _ => None,
    };
    if dispatch == ObservationDispatch::Unavailable {
        tracking.held_outcome = None;
    }

    if let Some(sample) = neutral_sample {
        let neutral_update = if tracking.recenter_requested {
            tracking.auto_neutral.recenter(sample)
        } else {
            tracking.auto_neutral.observe(sample)
        };
        match neutral_update {
            Ok(update) => {
                if update.pose_reference_changed {
                    tracking.reset_control_state();
                } else if update.gaze_baseline_changed {
                    tracking.pipeline.reset_gaze_filter();
                }
                tracking.recenter_requested = false;
                tracking.last_recenter_error = None;
            }
            Err(error) => {
                tracking.last_recenter_error = Some(error.to_string());
            }
        }
    }
    if let ObservationDispatch::Outcome(outcome) = dispatch {
        tracking.held_outcome = Some(outcome.clone());
    }

    let neutral = tracking.auto_neutral.reference();
    let gaze_baseline = tracking.auto_neutral.gaze_baseline();
    let held = tracking.held_outcome.take();
    let update =
        tracking
            .pipeline
            .update_mediapipe_outcome(held.as_ref(), neutral, gaze_baseline, now, dt);
    tracking.held_outcome = held;
    if let Some(frame) = update.frame {
        tracking.latest_control = Some((lifecycle.current_generation(), frame));
        tracking.control_active = true;
        let rate = tracking_rate.get_or_insert_with(|| RateCounter::new(1_000_000_000));
        rate.record(now.0);
    }
    if let Some(rate) = tracking_rate.as_mut() {
        diagnostics.tracking_rate = rate.prune_and_calculate_rate_hz(now.0) as f32;
    } else {
        diagnostics.tracking_rate = 0.0;
    }

    view_model.calibration = calibration_view(&tracking);
    view_model.tracking = tracking_view(
        if pipeline_state == crate::orchestrator::PipelineState::Starting {
            TrackingState::Starting
        } else {
            update.state
        },
        update.confidence.frame_confidence,
    );
    set_debug_cached(
        &mut tracking.diagnostics_cache.tracking_state,
        update.state,
        &mut diagnostics.tracking_state,
    );
    if diagnostics.tracking_backend.as_deref() != Some("mediapipe-face-landmarker") {
        diagnostics.tracking_backend = Some("mediapipe-face-landmarker".into());
    }
    if diagnostics.tracking_contract.as_deref()
        != Some("478 landmarks / 52 blendshapes / pose matrix")
    {
        diagnostics.tracking_contract = Some("478 landmarks / 52 blendshapes / pose matrix".into());
    }
    let auto_neutral_state = tracking.auto_neutral.state();
    set_debug_cached_opt(
        &mut tracking.diagnostics_cache.auto_neutral_state,
        auto_neutral_state,
        &mut diagnostics.auto_neutral_state,
    );
    diagnostics.face_tracking_calibration_ready = Some(matches!(
        tracking.auto_neutral.state(),
        vtuber_tracking::AutoNeutralState::Ready
    ));
}

fn calibration_view(runtime: &TrackingRuntime) -> CalibrationViewModel {
    let recent = runtime.auto_neutral.recent_sample_count();
    CalibrationViewModel {
        is_calibrating: runtime.recenter_requested,
        quality_score: (recent > 0).then(|| {
            (recent as f32 / vtuber_tracking::AUTO_NEUTRAL_MIN_SAMPLES as f32).clamp(0.0, 1.0)
        }),
        last_reject_reason: runtime.last_recenter_error.clone(),
        is_complete: runtime.auto_neutral.state() == AutoNeutralState::Ready,
    }
}

fn tracking_view(state: TrackingState, confidence: f32) -> crate::ui_model::TrackingViewModel {
    let state = match state {
        TrackingState::Tracking | TrackingState::Acquiring | TrackingState::Degraded => {
            UiTrackingState::Tracking
        }
        TrackingState::LostHold | TrackingState::ReturningNeutral => UiTrackingState::Lost,
        TrackingState::Starting => UiTrackingState::Initializing,
        TrackingState::Searching => UiTrackingState::WaitingForFace,
    };
    crate::ui_model::TrackingViewModel {
        is_tracking: matches!(state, UiTrackingState::Tracking),
        state,
        confidence: confidence.clamp(0.0, 1.0),
        // Searching/initializing is also a no-face state. Keeping this false
        // prevents the UI from claiming a face is present before the first
        // valid composite observation arrives.
        face_detected: matches!(state, UiTrackingState::Tracking),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use vtuber_core::FaceTrackingSample;
    use vtuber_core::face_tracking::{
        FaceBlendshapeSet, FaceLandmark, FaceTrackingQuality, MEDIAPIPE_FACE_LANDMARK_COUNT,
    };

    fn sample(seq: u64, finished_at: u64) -> FaceTrackingSample {
        FaceTrackingSample {
            source_seq: FrameSeq(seq),
            captured_at: MonoTimeNs(finished_at.saturating_sub(1_000_000)),
            inference_started_at: MonoTimeNs(finished_at.saturating_sub(500_000)),
            inference_finished_at: MonoTimeNs(finished_at),
            camera_to_face: vtuber_core::CameraFaceTransform::identity(),
            face_center: [0.5, 0.5],
            image_size: [640, 480],
            landmarks: vec![FaceLandmark::default(); MEDIAPIPE_FACE_LANDMARK_COUNT].into(),
            blendshapes: FaceBlendshapeSet::default(),
            quality: FaceTrackingQuality {
                landmark_presence_median: Some(1.0),
                matrix_orthogonality_error: 0.0,
                matrix_determinant: 1.0,
            },
        }
    }

    #[test]
    fn late_completion_does_not_make_an_old_camera_frame_fresh() {
        let mut gate = ObservationGate::default();
        let mut face = sample(7, 5_000_000_000);
        face.captured_at = MonoTimeNs(1_000_000_000);
        assert_eq!(
            gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(5_010_000_000),
            ),
            ObservationDispatch::Unavailable
        );
    }

    #[test]
    fn observation_gate_does_not_replay_a_fresh_face() {
        let mut gate = ObservationGate::default();
        let face = sample(7, 1_000_000_000);

        assert!(matches!(
            gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(1_050_000_000),
            ),
            ObservationDispatch::Outcome(FaceTrackingOutcome::Face(_))
        ));
        assert_eq!(
            gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(1_100_000_000),
            ),
            ObservationDispatch::NoUpdate
        );
    }

    #[test]
    fn observation_gate_turns_a_stale_face_into_face_loss() {
        let mut gate = ObservationGate::default();
        let face = sample(7, 1_000_000_000);

        let _ = gate.dispatch(
            Some(&FaceTrackingOutcome::Face(face.clone())),
            MonoTimeNs(1_050_000_000),
        );
        assert_eq!(
            gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(1_251_000_000),
            ),
            ObservationDispatch::Unavailable
        );
    }

    #[test]
    fn observation_gate_accepts_a_new_face_after_loss() {
        let mut gate = ObservationGate::default();
        let first = sample(7, 1_000_000_000);
        let second = sample(8, 1_300_000_000);

        let _ = gate.dispatch(
            Some(&FaceTrackingOutcome::Face(first.clone())),
            MonoTimeNs(1_050_000_000),
        );
        let _ = gate.dispatch(
            Some(&FaceTrackingOutcome::Face(first.clone())),
            MonoTimeNs(1_251_000_000),
        );
        assert!(matches!(
            gate.dispatch(
                Some(&FaceTrackingOutcome::Face(second.clone())),
                MonoTimeNs(1_350_000_000),
            ),
            ObservationDispatch::Outcome(FaceTrackingOutcome::Face(_))
        ));
    }

    #[test]
    fn observation_gate_reports_missing_output_as_face_loss() {
        let mut gate = ObservationGate::default();
        assert_eq!(
            gate.dispatch(None, MonoTimeNs(1)),
            ObservationDispatch::Unavailable
        );
    }

    #[test]
    fn observation_gate_reset_accepts_a_reused_capture_sequence() {
        let mut gate = ObservationGate::default();
        let face = sample(7, 1_000_000_000);

        assert!(matches!(
            gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(1_050_000_000),
            ),
            ObservationDispatch::Outcome(FaceTrackingOutcome::Face(_))
        ));
        gate.reset(MonoTimeNs(1_050_000_000));
        assert_eq!(
            gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(1_100_000_000),
            ),
            ObservationDispatch::Unavailable
        );
        let face = sample(7, 1_100_000_000);
        assert!(matches!(
            gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(1_150_000_000),
            ),
            ObservationDispatch::Outcome(FaceTrackingOutcome::Face(_))
        ));
    }

    #[test]
    fn session_reset_rejects_in_flight_observations_and_preserves_camera_neutral() {
        let mut runtime = TrackingRuntime::default();
        let mut face = sample(7, 1_000_000_000);
        assert!(runtime.auto_neutral.recenter(&face).is_ok());
        let neutral = runtime.auto_neutral.reference();
        runtime.held_outcome = Some(FaceTrackingOutcome::Face(face.clone()));
        runtime.control_active = true;
        runtime.last_update = Some(MonoTimeNs(1_000_000_000));

        runtime.invalidate_session(MonoTimeNs(1_050_000_000));
        assert_eq!(runtime.auto_neutral.reference(), neutral);
        assert!(runtime.held_outcome.is_none());
        assert!(runtime.latest_control.is_none());
        assert!(!runtime.control_active);
        assert!(runtime.last_update.is_none());
        face.inference_finished_at = MonoTimeNs(1_100_000_000);
        assert_eq!(
            runtime.observation_gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(1_150_000_000),
            ),
            ObservationDispatch::Unavailable
        );
    }

    #[test]
    fn input_fault_retains_the_head_response_but_rejects_in_flight_observations() {
        let mut runtime = TrackingRuntime::default();
        let mut face = sample(7, 1_000_000_000);
        face.camera_to_face.rotation_xyzw = [0.0, (0.3f32).sin(), 0.0, (0.3f32).cos()];
        let mut tracked = None;
        for tick in 1..=10 {
            tracked = runtime
                .pipeline
                .update_mediapipe(
                    Some(&face),
                    Some(vtuber_core::CameraFaceTransform::identity()),
                    None,
                    MonoTimeNs(1_000_000_000 + tick * 16_666_667),
                    Duration::from_millis(16),
                )
                .frame;
        }
        let tracked = tracked.expect("tracked head before input failure");
        runtime.latest_control = Some((vtuber_avatar::AvatarGeneration(1), tracked.clone()));
        runtime.held_outcome = Some(FaceTrackingOutcome::Face(face.clone()));
        runtime.control_active = true;
        runtime.discard_pending_observations(MonoTimeNs(1_200_000_000));
        assert!(runtime.control_active);
        assert_eq!(
            runtime.latest_control.as_ref().map(|(_, frame)| frame),
            Some(&tracked)
        );
        assert!(runtime.held_outcome.is_none());
        assert_eq!(
            runtime.observation_gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(1_210_000_000),
            ),
            ObservationDispatch::Unavailable
        );
        let held = runtime
            .pipeline
            .update_mediapipe(
                None,
                Some(vtuber_core::CameraFaceTransform::identity()),
                None,
                MonoTimeNs(1_220_000_000),
                Duration::from_millis(16),
            )
            .frame
            .expect("input failure must use the normal loss response");
        assert!((held.head.yaw_rad - tracked.head.yaw_rad).abs() < 1e-4);
        assert_eq!(held.source_seq, tracked.source_seq);
        assert_eq!(held.captured_at, tracked.captured_at);
        face.captured_at = MonoTimeNs(1_230_000_000);
        assert!(matches!(
            runtime.observation_gate.dispatch(
                Some(&FaceTrackingOutcome::Face(face.clone())),
                MonoTimeNs(1_240_000_000),
            ),
            ObservationDispatch::Outcome(FaceTrackingOutcome::Face(_))
        ));
    }

    #[test]
    fn input_failures_retain_pose_through_shutdown_but_explicit_stops_do_not() {
        use crate::orchestrator::{OrchestratorError, PipelineState};
        for error in [
            OrchestratorError::CameraFailed("disconnected".into()),
            OrchestratorError::InferenceFailed("worker exited".into()),
        ] {
            assert!(input_failure(PipelineState::Stopping, Some(&error)));
            assert!(input_failure(PipelineState::Failed, Some(&error)));
            assert!(!input_failure(PipelineState::Running, Some(&error)));
            assert!(!input_failure(PipelineState::Idle, Some(&error)));
        }
        assert!(!input_failure(PipelineState::Stopping, None));
        assert!(!input_failure(PipelineState::Failed, None));
    }

    #[test]
    fn no_face_is_not_reported_as_detected_while_searching() {
        let view = tracking_view(TrackingState::Searching, 0.0);
        assert_eq!(view.state, UiTrackingState::WaitingForFace);
        assert!(!view.face_detected);
        assert!(!calibration_view(&TrackingRuntime::default()).is_calibrating);
    }

    fn no_face(seq: u64, captured_at: u64) -> FaceTrackingOutcome {
        FaceTrackingOutcome::NoFace {
            source_seq: FrameSeq(seq),
            captured_at: MonoTimeNs(captured_at),
            inference_started_at: MonoTimeNs(captured_at),
            inference_finished_at: MonoTimeNs(captured_at),
        }
    }

    // A fixed synthetic outcome recording, replayed without rerunning inference. All
    // render rates consume the same source identities and capture timestamps.
    fn recorded_outcomes(faces: &[bool], inference_hz: u64) -> Vec<FaceTrackingOutcome> {
        faces
            .iter()
            .enumerate()
            .map(|(index, &has_face)| {
                let seq = index as u64 + 1;
                let captured_at = 1_000_000_000 + index as u64 * 1_000_000_000 / inference_hz;
                if has_face {
                    let mut face = sample(seq, captured_at);
                    face.captured_at = MonoTimeNs(captured_at);
                    face.inference_started_at = MonoTimeNs(captured_at);
                    FaceTrackingOutcome::Face(face)
                } else {
                    no_face(seq, captured_at)
                }
            })
            .collect()
    }

    fn replay_outcomes(
        outcomes: &[FaceTrackingOutcome],
        inference_hz: u64,
        render_hz: u64,
    ) -> Vec<vtuber_tracking::confidence::ConfidenceAssessment> {
        use vtuber_tracking::confidence::ConfidenceSignal;
        let mut gate = ObservationGate::default();
        let mut pipeline = TrackingPipeline::try_new(PipelineConfig::default()).expect("config");
        let mut held = None;
        let mut assessments = Vec::new();
        let mut previous: Option<vtuber_tracking::confidence::ConfidenceAssessment> = None;
        let mut last_now = 1_000_000_000;
        let ticks_per_outcome = render_hz / inference_hz;
        for tick in 0..outcomes.len() as u64 * ticks_per_outcome {
            let now = 1_000_000_000 + tick * 1_000_000_000 / render_hz;
            let latest = outcomes.get((tick / ticks_per_outcome) as usize);
            let dispatch = gate.dispatch(latest, MonoTimeNs(now));
            let fresh = matches!(dispatch, ObservationDispatch::Outcome(_));
            match dispatch {
                ObservationDispatch::Outcome(outcome) => held = Some(outcome.clone()),
                ObservationDispatch::Unavailable => panic!("live outcome stream timed out"),
                ObservationDispatch::NoUpdate => {}
            }
            let update = pipeline.update_mediapipe_outcome(
                held.as_ref(),
                Some(vtuber_core::CameraFaceTransform::identity()),
                None,
                MonoTimeNs(now),
                Duration::from_nanos(now - last_now),
            );
            if fresh {
                assessments.push(update.confidence);
            } else {
                let mut expected = previous.expect("prior tick");
                expected.signal = ConfidenceSignal::None;
                assert_eq!(
                    update.confidence, expected,
                    "held tick {tick} at {render_hz} Hz"
                );
            }
            previous = Some(update.confidence);
            last_now = now;
        }
        assessments
    }

    #[test]
    fn confidence_replay_is_independent_of_render_rate() {
        use vtuber_tracking::confidence::ConfidenceSignal;
        for misses in [1, 4, 5, 8] {
            let faces: Vec<_> = std::iter::repeat_n(true, 3)
                .chain(std::iter::repeat_n(false, misses))
                .chain(std::iter::repeat_n(true, 3))
                .collect();
            let mut reference = None;
            for inference_hz in [15, 30] {
                let outcomes = recorded_outcomes(&faces, inference_hz);
                for render_hz in [30, 60, 120] {
                    let assessments = replay_outcomes(&outcomes, inference_hz, render_hz);
                    if let Some(reference) = &reference {
                        assert_eq!(&assessments, reference);
                    } else {
                        reference = Some(assessments.clone());
                    }
                    let first_loss = assessments
                        .iter()
                        .position(|a| a.signal == ConfidenceSignal::Degrade);
                    assert_eq!(first_loss, (misses >= 5).then_some(7));
                    assert_eq!(
                        assessments.get(2).map(|a| a.signal),
                        Some(ConfidenceSignal::Acquire),
                    );
                    assert!(assessments.last().expect("recovered face").is_confident);
                    eprintln!(
                        "misses={misses} inference={inference_hz} render={render_hz} first_loss_seq={:?}",
                        first_loss.map(|index| index + 1),
                    );
                }
            }
        }
        let alternating = [true, true, true, false, true, false, true, false, true];
        for inference_hz in [15, 30] {
            for render_hz in [30, 60, 120] {
                assert!(
                    replay_outcomes(
                        &recorded_outcomes(&alternating, inference_hz),
                        inference_hz,
                        render_hz,
                    )
                    .iter()
                    .all(|a| a.signal != ConfidenceSignal::Degrade)
                );
            }
        }
    }

    #[test]
    fn no_face_identity_is_held_and_reused_sequences_work_after_reset() {
        let mut gate = ObservationGate::default();
        let missed = no_face(7, 1_000_000_000);
        assert_eq!(
            gate.dispatch(Some(&missed), MonoTimeNs(1_010_000_000)),
            ObservationDispatch::Outcome(&missed),
        );
        assert_eq!(
            gate.dispatch(Some(&missed), MonoTimeNs(1_020_000_000)),
            ObservationDispatch::NoUpdate,
        );
        gate.reset(MonoTimeNs(1_030_000_000));
        assert_eq!(
            gate.dispatch(Some(&missed), MonoTimeNs(1_040_000_000)),
            ObservationDispatch::Unavailable,
        );
        let fresh = no_face(7, 1_050_000_000);
        assert_eq!(
            gate.dispatch(Some(&fresh), MonoTimeNs(1_060_000_000)),
            ObservationDispatch::Outcome(&fresh),
        );
    }

    #[test]
    fn watchdog_expires_after_face_or_no_face_without_counting_render_ticks() {
        use vtuber_tracking::confidence::ConfidenceSignal;
        for inference_hz in [15, 30] {
            for render_hz in [30, 60, 120] {
                for last_has_face in [true, false] {
                    // One face can reconnect before the gate becomes confident.
                    // Silence must still leave Tracking/Acquiring in this case.
                    for warm_faces in [1, 3] {
                        let faces: Vec<_> = std::iter::repeat_n(true, warm_faces)
                            .chain((!last_has_face).then_some(false))
                            .collect();
                        let outcomes = recorded_outcomes(&faces, inference_hz);
                        let mut pipeline =
                            TrackingPipeline::try_new(PipelineConfig::default()).expect("config");
                        let mut gate = ObservationGate::default();
                        let last_capture = 1_000_000_000
                            + (outcomes.len() as u64 - 1) * 1_000_000_000 / inference_hz;
                        for (index, outcome) in outcomes.iter().enumerate() {
                            let now = 1_000_000_000 + index as u64 * 1_000_000_000 / inference_hz;
                            let _ = gate.dispatch(Some(outcome), MonoTimeNs(now));
                            let _ = pipeline.update_mediapipe_outcome(
                                Some(outcome),
                                Some(vtuber_core::CameraFaceTransform::identity()),
                                None,
                                MonoTimeNs(now),
                                Duration::from_secs_f64(1.0 / inference_hz as f64),
                            );
                        }
                        let last = outcomes.last().expect("last outcome");
                        assert_eq!(
                            gate.dispatch(Some(last), MonoTimeNs(last_capture + 250_000_000)),
                            ObservationDispatch::NoUpdate,
                        );
                        let late = last_capture + 250_000_000 + 1_000_000_000 / render_hz;
                        assert_eq!(
                            gate.dispatch(Some(last), MonoTimeNs(late)),
                            ObservationDispatch::Unavailable,
                        );
                        let update = pipeline.update_mediapipe_outcome(
                            None,
                            Some(vtuber_core::CameraFaceTransform::identity()),
                            None,
                            MonoTimeNs(late),
                            Duration::from_millis(250),
                        );
                        assert_eq!(update.state, TrackingState::LostHold);
                        assert_eq!(update.confidence.signal, ConfidenceSignal::Degrade);
                        assert_eq!(update.confidence.consecutive_bad, 0);
                        assert!(!update.confidence.is_confident);
                        let mut recovered = sample(100, late + 1);
                        recovered.captured_at = MonoTimeNs(late + 1);
                        recovered.inference_started_at = recovered.captured_at;
                        let recovered = FaceTrackingOutcome::Face(recovered);
                        assert!(matches!(
                            gate.dispatch(Some(&recovered), MonoTimeNs(late + 1)),
                            ObservationDispatch::Outcome(_)
                        ));
                        let update = pipeline.update_mediapipe_outcome(
                            Some(&recovered),
                            Some(vtuber_core::CameraFaceTransform::identity()),
                            None,
                            MonoTimeNs(late + 1),
                            Duration::from_nanos(1),
                        );
                        assert_eq!(update.state, TrackingState::Acquiring);
                        assert_eq!(update.confidence.consecutive_good, 1);
                    }
                }
            }
        }
    }
}
