//! Main-thread bridge from inference observations to the pure tracking core.

use std::time::Duration;

use bevy::prelude::*;
use vtuber_core::metrics::RateCounter;
use vtuber_core::{AvatarControlFrame, FaceTrackingSample, FrameSeq, MonoTimeNs, TrackingState};
use vtuber_tracking::{AutoNeutralCollector, AutoNeutralState, PipelineConfig, TrackingPipeline};

use crate::diagnostics::DiagnosticsSnapshot;
use crate::inference_runtime::InferenceRuntime;
use crate::orchestrator::{CalibrationRequest, Orchestrator};
use crate::ui_model::{CalibrationViewModel, TrackingState as UiTrackingState, UiViewModel};

/// Maximum age of an inference result before it is treated as face-lost.
///
/// The current inference slot carries successful face observations only. If a
/// detector reports no face without publishing an explicit `InferenceOutput`,
/// this watchdog converts the absence of a fresh result into the tracking
/// pipeline's normal `None` input. It prevents a last face observation from
/// being replayed forever while keeping normal 15 Hz inference output from
/// being mistaken for a loss.
const INFERENCE_SILENCE_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Clone, Debug, PartialEq)]
enum ObservationDispatch {
    /// No new inference result and the last result is still fresh.
    NoUpdate,
    /// A new, fresh face observation is ready for tracking.
    Face(Box<FaceTrackingSample>),
    /// The current inference result is absent or stale.
    NoFace,
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

    fn dispatch(
        &mut self,
        latest: Option<&FaceTrackingSample>,
        now: MonoTimeNs,
    ) -> ObservationDispatch {
        let Some(observation) = latest else {
            return ObservationDispatch::NoFace;
        };
        if self
            .observations_after
            .is_some_and(|boundary| observation.captured_at < boundary)
        {
            return ObservationDispatch::NoFace;
        }

        let age = Duration::from_nanos(now.0.saturating_sub(observation.inference_finished_at.0));
        let fresh = age <= INFERENCE_SILENCE_TIMEOUT;
        let is_new = self.last_source_seq != Some(observation.source_seq);
        if is_new {
            self.last_source_seq = Some(observation.source_seq);
            return if fresh {
                ObservationDispatch::Face(Box::new(observation.clone()))
            } else {
                ObservationDispatch::NoFace
            };
        }

        if fresh {
            ObservationDispatch::NoUpdate
        } else {
            ObservationDispatch::NoFace
        }
    }
}

/// Tracking domain state owned by the Bevy main thread.
#[derive(Resource)]
pub struct TrackingRuntime {
    pipeline: TrackingPipeline,
    auto_neutral: AutoNeutralCollector,
    recenter_requested: bool,
    last_update: Option<MonoTimeNs>,
    last_avatar_generation: vtuber_avatar::AvatarGeneration,
    last_recenter_error: Option<String>,
    observation_gate: ObservationGate,
    /// Last face observation dispatched into the pipeline.
    ///
    /// The pipeline is advanced on every main-thread frame with this sample
    /// so the emitted control frames form a continuous signal at the render
    /// rate. It is replaced when a new observation arrives and cleared on
    /// face loss.
    held_sample: Option<FaceTrackingSample>,
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
        let pipeline = TrackingPipeline::new(config)
            .expect("default tracking configuration is an internal invariant");
        Self {
            pipeline,
            auto_neutral: AutoNeutralCollector::new(),
            recenter_requested: false,
            last_update: None,
            last_avatar_generation: vtuber_avatar::AvatarGeneration::default(),
            last_recenter_error: None,
            observation_gate: ObservationGate::default(),
            held_sample: None,
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
        self.pipeline.reset();
        self.observation_gate.reset(now);
        self.held_sample = None;
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
    let pipeline_failed = pipeline_state == crate::orchestrator::PipelineState::Failed;
    if capture_inactive || pipeline_failed {
        tracking.invalidate_session(now);
        view_model.tracking = tracking_view(TrackingState::Starting, 0.0);
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

    let dispatch = tracking
        .observation_gate
        .dispatch(inference.latest_face_sample.as_ref(), now);
    let neutral_sample = match &dispatch {
        ObservationDispatch::Face(sample) => Some((**sample).clone()),
        // The held observation keeps feeding the pipeline between inference
        // results so the filters advance at the render rate. Re-running the
        // same sample through the filters is what reconstructs a continuous
        // motion signal; the pipeline memoizes the per-sample conversion.
        ObservationDispatch::NoUpdate | ObservationDispatch::NoFace => None,
    };
    if dispatch == ObservationDispatch::NoFace {
        tracking.held_sample = None;
    }

    if let Some(sample) = neutral_sample.as_ref() {
        let neutral_update = if tracking.recenter_requested {
            tracking.auto_neutral.recenter(sample)
        } else {
            tracking.auto_neutral.observe(sample)
        };
        match neutral_update {
            Ok(update) => {
                if update.pose_reference_changed {
                    tracking.pipeline.reset();
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
    if let Some(sample) = neutral_sample {
        tracking.held_sample = Some(sample);
    }

    let neutral = tracking.auto_neutral.reference();
    let gaze_baseline = tracking.auto_neutral.gaze_baseline();
    let held = tracking.held_sample.take();
    let update = tracking
        .pipeline
        .update_mediapipe(held.as_ref(), neutral, gaze_baseline, now, dt);
    tracking.held_sample = held;
    if let Some(frame) = update.frame {
        tracking.latest_control = Some((lifecycle.current_generation(), frame));
        tracking.control_active = true;
        let rate = tracking_rate.get_or_insert_with(|| RateCounter::new(1_000_000_000));
        rate.record(now.0);
    }
    if let Some(rate) = tracking_rate.as_mut() {
        diagnostics.tracking_rate = rate.rate_hz(now.0) as f32;
    } else {
        diagnostics.tracking_rate = 0.0;
    }

    view_model.calibration = calibration_view(&tracking);
    view_model.tracking = tracking_view(update.state, update.confidence.frame_confidence);
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
        is_calibrating: runtime.recenter_requested
            || runtime.auto_neutral.state() == AutoNeutralState::WaitingForFace,
        samples_collected: recent.min(u32::MAX as usize) as u32,
        samples_target: vtuber_tracking::AUTO_NEUTRAL_MIN_SAMPLES as u32,
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
        TrackingState::Starting | TrackingState::Searching => UiTrackingState::Initializing,
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
    use super::*;
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
    fn observation_gate_does_not_replay_a_fresh_face() {
        let mut gate = ObservationGate::default();
        let face = sample(7, 1_000_000_000);

        assert!(matches!(
            gate.dispatch(Some(&face), MonoTimeNs(1_050_000_000)),
            ObservationDispatch::Face(_)
        ));
        assert_eq!(
            gate.dispatch(Some(&face), MonoTimeNs(1_100_000_000)),
            ObservationDispatch::NoUpdate
        );
    }

    #[test]
    fn observation_gate_turns_a_stale_face_into_face_loss() {
        let mut gate = ObservationGate::default();
        let face = sample(7, 1_000_000_000);

        let _ = gate.dispatch(Some(&face), MonoTimeNs(1_050_000_000));
        assert_eq!(
            gate.dispatch(Some(&face), MonoTimeNs(1_251_000_000)),
            ObservationDispatch::NoFace
        );
    }

    #[test]
    fn observation_gate_accepts_a_new_face_after_loss() {
        let mut gate = ObservationGate::default();
        let first = sample(7, 1_000_000_000);
        let second = sample(8, 1_300_000_000);

        let _ = gate.dispatch(Some(&first), MonoTimeNs(1_050_000_000));
        let _ = gate.dispatch(Some(&first), MonoTimeNs(1_251_000_000));
        assert!(matches!(
            gate.dispatch(Some(&second), MonoTimeNs(1_350_000_000)),
            ObservationDispatch::Face(_)
        ));
    }

    #[test]
    fn observation_gate_reports_missing_output_as_face_loss() {
        let mut gate = ObservationGate::default();
        assert_eq!(
            gate.dispatch(None, MonoTimeNs(1)),
            ObservationDispatch::NoFace
        );
    }

    #[test]
    fn observation_gate_reset_accepts_a_reused_capture_sequence() {
        let mut gate = ObservationGate::default();
        let face = sample(7, 1_000_000_000);

        assert!(matches!(
            gate.dispatch(Some(&face), MonoTimeNs(1_050_000_000)),
            ObservationDispatch::Face(_)
        ));
        gate.reset(MonoTimeNs(1_050_000_000));
        assert_eq!(
            gate.dispatch(Some(&face), MonoTimeNs(1_100_000_000)),
            ObservationDispatch::NoFace
        );
        let face = sample(7, 1_100_000_000);
        assert!(matches!(
            gate.dispatch(Some(&face), MonoTimeNs(1_150_000_000)),
            ObservationDispatch::Face(_)
        ));
    }

    #[test]
    fn session_reset_rejects_in_flight_observations_and_preserves_camera_neutral() {
        let mut runtime = TrackingRuntime::default();
        let mut face = sample(7, 1_000_000_000);
        assert!(runtime.auto_neutral.recenter(&face).is_ok());
        let neutral = runtime.auto_neutral.reference();
        runtime.held_sample = Some(face.clone());
        runtime.control_active = true;
        runtime.last_update = Some(MonoTimeNs(1_000_000_000));

        runtime.invalidate_session(MonoTimeNs(1_050_000_000));
        assert_eq!(runtime.auto_neutral.reference(), neutral);
        assert!(runtime.held_sample.is_none());
        assert!(runtime.latest_control.is_none());
        assert!(!runtime.control_active);
        assert!(runtime.last_update.is_none());
        face.inference_finished_at = MonoTimeNs(1_100_000_000);
        assert_eq!(
            runtime
                .observation_gate
                .dispatch(Some(&face), MonoTimeNs(1_150_000_000)),
            ObservationDispatch::NoFace
        );
    }

    #[test]
    fn no_face_is_not_reported_as_detected_while_searching() {
        let view = tracking_view(TrackingState::Searching, 0.0);
        assert_eq!(view.state, UiTrackingState::Initializing);
        assert!(!view.face_detected);
    }
}
