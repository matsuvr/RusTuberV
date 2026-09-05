//! Main-thread bridge from inference observations to the pure tracking core.

use std::sync::Arc;
use std::time::Duration;

use bevy::prelude::*;
use vtuber_core::metrics::RateCounter;
use vtuber_core::{
    AvatarControlFrame, FaceTrackingSample, FrameSeq, LatestSlot, MonoTimeNs, TrackingState,
};
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
}

impl ObservationGate {
    /// Clears the source-sequence boundary at a capture/session reset.
    ///
    /// Capture sequence numbers are owned by the capture session. A new
    /// session may legally start at the same sequence as the previous one,
    /// so retaining the old value would suppress its first face observation.
    fn reset(&mut self) {
        self.last_source_seq = None;
    }

    fn dispatch(
        &mut self,
        latest: Option<&FaceTrackingSample>,
        now: MonoTimeNs,
    ) -> ObservationDispatch {
        let Some(observation) = latest else {
            return ObservationDispatch::NoFace;
        };

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
    /// Latest-only control frame for the avatar bridge.
    pub control_slot: Arc<LatestSlot<AvatarControlFrame>>,
    /// Most recent frame consumed by the avatar bridge.
    pub latest_control: Option<AvatarControlFrame>,
}

impl Default for TrackingRuntime {
    fn default() -> Self {
        let config = PipelineConfig::default();
        // Invariant: `PipelineConfig::default()` produces a valid config by
        // definition; repository tests pin this.
        #[allow(clippy::expect_used)]
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
            control_slot: Arc::new(LatestSlot::new()),
            latest_control: None,
        }
    }
}

impl TrackingRuntime {
    /// Returns the latest-only control frame slot.
    #[must_use]
    pub fn control_slot(&self) -> Arc<LatestSlot<AvatarControlFrame>> {
        Arc::clone(&self.control_slot)
    }
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
    if lifecycle.current_generation() != tracking.last_avatar_generation {
        tracking.last_avatar_generation = lifecycle.current_generation();
        // The camera neutral is independent of the avatar model. Preserve it
        // across replacement, but always reset the smoothing and recovery
        // state so a new avatar cannot receive a stale frame.
        tracking.pipeline.reset();
        tracking.observation_gate.reset();
        tracking.held_sample = None;
        tracking.control_slot.clear();
        tracking.latest_control = None;
        tracking.control_active = false;
        tracking.last_update = None;
    }

    let pipeline_state = orchestrator.pipeline_state();
    let capture_inactive = !orchestrator.capture_desired()
        && matches!(
            pipeline_state,
            crate::orchestrator::PipelineState::Idle | crate::orchestrator::PipelineState::Stopping
        );
    let pipeline_failed = pipeline_state == crate::orchestrator::PipelineState::Failed;
    if capture_inactive || pipeline_failed {
        tracking.pipeline.reset();
        tracking.observation_gate.reset();
        tracking.held_sample = None;
        tracking.control_slot.clear();
        tracking.latest_control = None;
        tracking.control_active = false;
        tracking.last_update = None;
        view_model.tracking = tracking_view(TrackingState::Starting, 0.0);
        view_model.calibration = calibration_view(&tracking);
        diagnostics.tracking_state = format!("{:?}", TrackingState::Starting);
        diagnostics.tracking_backend = Some("mediapipe-face-landmarker".into());
        diagnostics.tracking_contract = Some("478 landmarks / 52 blendshapes / pose matrix".into());
        diagnostics.auto_neutral_state = Some(format!("{:?}", tracking.auto_neutral.state()));
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
                tracking.recenter_requested = true;
                tracking.last_recenter_error = None;
            }
            CalibrationRequest::Cancel => tracking.recenter_requested = false,
        }
    }

    let now = vtuber_core::monotonic_now();
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
        let _ = tracking.control_slot.publish(frame.clone());
        tracking.latest_control = Some(frame);
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
    diagnostics.tracking_state = format!("{:?}", update.state);
    diagnostics.tracking_backend = Some("mediapipe-face-landmarker".into());
    diagnostics.tracking_contract = Some("478 landmarks / 52 blendshapes / pose matrix".into());
    diagnostics.auto_neutral_state = Some(format!("{:?}", tracking.auto_neutral.state()));
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
        gate.reset();
        assert!(matches!(
            gate.dispatch(Some(&face), MonoTimeNs(1_050_000_000)),
            ObservationDispatch::Face(_)
        ));
    }

    #[test]
    fn no_face_is_not_reported_as_detected_while_searching() {
        let view = tracking_view(TrackingState::Searching, 0.0);
        assert_eq!(view.state, UiTrackingState::Initializing);
        assert!(!view.face_detected);
    }
}
