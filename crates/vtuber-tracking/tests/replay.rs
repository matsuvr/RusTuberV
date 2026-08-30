// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Deterministic replay tests for the tracking pipeline.
//!
//! These tests use synthetic `RawFaceObservation` streams with explicit
//! monotonic timestamps. They verify that:
//!
//! * The same input stream + settings always produces the same output
//!   sequence within a fixed floating-point tolerance.
//! * Output sequence numbers and capture timestamps trace back to the input.
//! * Pipeline reset and calibration application produce well-defined
//!   boundaries without leaking state across runs.
//!
//! No recorded camera images or personal data are used.

use std::time::Duration;

use approx::assert_relative_eq;
use nalgebra::OVector;
use vtuber_core::control::{CalibrationSettings, TrackingPipelineSettings};
use vtuber_core::types::{
    AvatarControlFrame, FrameSeq, HeadPose, Landmark3, LandmarkSchemaId, MonoTimeNs,
    NamedCoefficient, RawExpressionObservation, RawFaceObservation, TrackingState,
};
use vtuber_tracking::{
    ConfidenceGateParams, ExpressionFilterParams, HeadFilterParams, LossRecoveryParams,
    NeutralProfile, NeutralValidationSettings, PipelineConfig, StateMachineParams,
    TrackingPipeline,
};

const SCHEMA: LandmarkSchemaId = LandmarkSchemaId("replay-test");
const FRAME_DT_NS: u64 = 33_333_333;

fn config() -> PipelineConfig {
    PipelineConfig {
        calibration: CalibrationSettings::try_new(5, 5.0, 0.5, 5.0f32.to_radians(), 0.15).unwrap(),
        validation: NeutralValidationSettings::try_new(5.0f32.to_radians(), 1.0).unwrap(),
        confidence_gate: ConfidenceGateParams {
            enter_threshold: 0.6,
            exit_threshold: 0.3,
            required_consecutive_good: 1,
            required_consecutive_bad: 1,
            max_count: 100,
        },
        state_machine: StateMachineParams {
            hold_duration: Duration::from_millis(100),
            return_duration: Duration::from_millis(200),
        },
        head_filter: HeadFilterParams::with_time_constant(0.05),
        gaze_filter: Default::default(),
        expression_filter: ExpressionFilterParams::with_time_constants(0.03, 0.10),
        loss_recovery: LossRecoveryParams {
            glide_duration: Duration::from_millis(100),
            decay_duration: Duration::from_millis(200),
            recovery_duration: Duration::from_millis(100),
            ..LossRecoveryParams::default()
        },
    }
}

fn synthetic_landmarks() -> Vec<Landmark3> {
    vtuber_tracking::pose::synthetic_face_points()
        .into_iter()
        .map(|p| Landmark3 {
            x: p[0],
            y: p[1],
            z: p[2],
            visibility: 1.0,
        })
        .collect()
}

fn rotate_landmarks(landmarks: &[Landmark3], pose: HeadPose) -> Vec<Landmark3> {
    let q = vtuber_tracking::pose::semantic_pose_to_quaternion(pose);
    landmarks
        .iter()
        .map(|lm| {
            let v = OVector::<f32, nalgebra::U3>::new(lm.x, lm.y, lm.z);
            let r = q * v;
            Landmark3 {
                x: r.x,
                y: r.y,
                z: r.z,
                visibility: lm.visibility,
            }
        })
        .collect()
}

fn neutral_profile() -> NeutralProfile {
    NeutralProfile {
        version: 2,
        schema: SCHEMA,
        landmarks: synthetic_landmarks(),
        head_pose: HeadPose::default(),
        gaze_baseline: vtuber_tracking::GazeNeutralBaseline::default(),
        blink_left_baseline: 0.05,
        blink_right_baseline: 0.05,
        mouth_open_baseline: 0.05,
        face_scale: 1.0,
        confidence_baseline: 0.9,
        collected_at: MonoTimeNs(0),
        model_hash: None,
        camera_fingerprint: None,
    }
}

fn relaxed_expression() -> RawExpressionObservation {
    RawExpressionObservation {
        blink_left: 0.05,
        blink_left_confidence: 0.9,
        blink_right: 0.05,
        blink_right_confidence: 0.9,
        mouth_open: 0.05,
        mouth_open_confidence: 0.9,
    }
}

fn blinking_expression() -> RawExpressionObservation {
    RawExpressionObservation {
        blink_left: 0.9,
        blink_left_confidence: 0.95,
        blink_right: 0.9,
        blink_right_confidence: 0.95,
        mouth_open: 0.05,
        mouth_open_confidence: 0.9,
    }
}

fn talking_expression() -> RawExpressionObservation {
    RawExpressionObservation {
        blink_left: 0.05,
        blink_left_confidence: 0.9,
        blink_right: 0.05,
        blink_right_confidence: 0.9,
        mouth_open: 0.8,
        mouth_open_confidence: 0.9,
    }
}

fn observation(
    seq: u64,
    landmarks: Vec<Landmark3>,
    expressions: RawExpressionObservation,
    blendshapes: Option<Vec<NamedCoefficient>>,
) -> RawFaceObservation {
    RawFaceObservation {
        source_seq: FrameSeq(seq),
        captured_at: MonoTimeNs(seq * FRAME_DT_NS),
        inference_started_at: MonoTimeNs(seq * FRAME_DT_NS + 5_000_000),
        inference_finished_at: MonoTimeNs(seq * FRAME_DT_NS + 25_000_000),
        face_confidence: 0.9,
        landmarks,
        blendshapes,
        expressions,
        roi: vtuber_core::types::NormalizedRect::default(),
        schema: SCHEMA,
    }
}

/// A small synthetic stream: neutral -> turn right -> blink -> mouth open ->
/// lost face -> reacquire with gaze.
fn synthetic_stream() -> Vec<Option<RawFaceObservation>> {
    let neutral = synthetic_landmarks();

    let turned = rotate_landmarks(
        &neutral,
        HeadPose {
            yaw_rad: 15.0f32.to_radians(),
            pitch_rad: 5.0f32.to_radians(),
            roll_rad: -5.0f32.to_radians(),
        },
    );

    vec![
        Some(observation(1, neutral.clone(), relaxed_expression(), None)),
        Some(observation(2, turned.clone(), relaxed_expression(), None)),
        Some(observation(3, neutral.clone(), blinking_expression(), None)),
        Some(observation(4, neutral.clone(), talking_expression(), None)),
        None, // lost
        None, // still lost
        Some(observation(
            7,
            turned.clone(),
            relaxed_expression(),
            Some(vec![
                NamedCoefficient {
                    name: "eyeLookLeft".into(),
                    value: 0.4,
                },
                NamedCoefficient {
                    name: "eyeLookUp".into(),
                    value: 0.2,
                },
            ]),
        )),
    ]
}

/// Runs a stream through a fresh pipeline and returns the emitted frames.
fn replay(
    stream: &[Option<RawFaceObservation>],
    pipeline: &mut TrackingPipeline,
) -> Vec<AvatarControlFrame> {
    let mut outputs = Vec::new();
    let mut now = MonoTimeNs(0);
    for item in stream {
        now = MonoTimeNs(now.0 + FRAME_DT_NS);
        let update = pipeline.update(item.as_ref(), now, Duration::from_nanos(FRAME_DT_NS));
        if let Some(frame) = update.frame {
            outputs.push(frame);
        }
    }
    outputs
}

fn assert_frames_eq(a: &AvatarControlFrame, b: &AvatarControlFrame) {
    assert_eq!(a.source_seq, b.source_seq);
    assert_eq!(a.captured_at, b.captured_at);
    assert_eq!(a.state, b.state);
    assert_relative_eq!(a.head.yaw_rad, b.head.yaw_rad, epsilon = 1e-5);
    assert_relative_eq!(a.head.pitch_rad, b.head.pitch_rad, epsilon = 1e-5);
    assert_relative_eq!(a.head.roll_rad, b.head.roll_rad, epsilon = 1e-5);
    assert_relative_eq!(
        a.expressions.blink_left,
        b.expressions.blink_left,
        epsilon = 1e-5
    );
    assert_relative_eq!(
        a.expressions.blink_right,
        b.expressions.blink_right,
        epsilon = 1e-5
    );
    assert_relative_eq!(a.expressions.aa, b.expressions.aa, epsilon = 1e-5);
    assert_eq!(a.gaze.state, b.gaze.state);
    assert_relative_eq!(a.gaze.horizontal, b.gaze.horizontal, epsilon = 1e-5);
    assert_relative_eq!(a.gaze.vertical, b.gaze.vertical, epsilon = 1e-5);
}

#[test]
fn replay_is_deterministic_for_same_stream_and_settings() {
    let mut pipeline_a = TrackingPipeline::new(config()).unwrap();
    pipeline_a.apply_calibration(neutral_profile()).unwrap();

    let mut pipeline_b = TrackingPipeline::new(config()).unwrap();
    pipeline_b.apply_calibration(neutral_profile()).unwrap();

    let stream = synthetic_stream();
    let outputs_a = replay(&stream, &mut pipeline_a);
    let outputs_b = replay(&stream, &mut pipeline_b);

    assert!(!outputs_a.is_empty(), "stream should produce frames");
    assert_eq!(outputs_a.len(), outputs_b.len());
    for (a, b) in outputs_a.iter().zip(outputs_b.iter()) {
        assert_frames_eq(a, b);
    }
}

#[test]
fn replay_output_timestamps_trace_back_to_input() {
    let mut pipeline = TrackingPipeline::new(config()).unwrap();
    pipeline.apply_calibration(neutral_profile()).unwrap();

    let stream = synthetic_stream();
    let outputs = replay(&stream, &mut pipeline);

    for frame in &outputs {
        // Every emitted frame must carry a source sequence and capture time
        // derived from one of the input observations.
        assert!(
            frame.source_seq.0 > 0 && frame.source_seq.0 <= 7,
            "unexpected source seq {:?}",
            frame.source_seq
        );
        assert_eq!(
            frame.captured_at,
            MonoTimeNs(frame.source_seq.0 * FRAME_DT_NS),
            "capture timestamp must match input sequence"
        );
    }
}

#[test]
fn replay_lost_face_returns_to_neutral() {
    let mut pipeline = TrackingPipeline::new(config()).unwrap();
    pipeline.apply_calibration(neutral_profile()).unwrap();

    let neutral = synthetic_landmarks();
    let turned = rotate_landmarks(
        &neutral,
        HeadPose {
            yaw_rad: 15.0f32.to_radians(),
            pitch_rad: 5.0f32.to_radians(),
            roll_rad: -5.0f32.to_radians(),
        },
    );

    // Build a stream that turns the head, then loses the face for long enough
    // that the loss-recovery decay fully completes before the stream ends.
    let mut stream: Vec<Option<RawFaceObservation>> = Vec::new();
    stream.push(Some(observation(1, neutral, relaxed_expression(), None)));
    stream.push(Some(observation(2, turned, relaxed_expression(), None)));
    for _seq in 3..=12 {
        stream.push(None);
    }

    let outputs = replay(&stream, &mut pipeline);
    let last = outputs.last().expect("stream should produce frames");
    assert_relative_eq!(last.head.yaw_rad, 0.0, epsilon = 1e-3);
    assert_relative_eq!(last.head.pitch_rad, 0.0, epsilon = 1e-3);
    assert_relative_eq!(last.head.roll_rad, 0.0, epsilon = 1e-3);
}

#[test]
fn dim_light_reacquire_reconnects_smoothly_without_snap() {
    let mut pipeline = TrackingPipeline::new(config()).unwrap();
    pipeline.apply_calibration(neutral_profile()).unwrap();

    let neutral = synthetic_landmarks();
    let turned = rotate_landmarks(
        &neutral,
        HeadPose {
            yaw_rad: 20.0f32.to_radians(),
            pitch_rad: 0.0,
            roll_rad: 0.0,
        },
    );

    // Dim-light landmarks: the face is detected, but landmark quality stays
    // between the gate's exit and enter thresholds (0.3 < confidence < 0.6
    // in this configuration).
    let mut dim_landmarks = turned.clone();
    for landmark in &mut dim_landmarks {
        landmark.visibility = 0.5;
    }
    let dim_observation = |seq: u64| -> RawFaceObservation {
        let mut obs = observation(seq, dim_landmarks.clone(), relaxed_expression(), None);
        obs.face_confidence = 0.9;
        obs
    };

    let mut stream: Vec<Option<RawFaceObservation>> = Vec::new();
    stream.push(Some(observation(1, neutral, relaxed_expression(), None)));
    stream.push(Some(observation(
        2,
        turned.clone(),
        relaxed_expression(),
        None,
    )));
    // Lose the face long enough for the glide and part of the decay.
    for _seq in 3..=12 {
        stream.push(None);
    }
    // Reacquire with a usable but sub-enter-threshold quality face.
    for seq in 13..=18 {
        stream.push(Some(dim_observation(seq)));
    }

    let mut outputs = Vec::new();
    let mut now = MonoTimeNs(0);
    let mut states = Vec::new();
    for item in &stream {
        now = MonoTimeNs(now.0 + FRAME_DT_NS);
        let update = pipeline.update(item.as_ref(), now, Duration::from_nanos(FRAME_DT_NS));
        states.push(update.state);
        if let Some(frame) = update.frame {
            outputs.push(frame);
        }
    }

    // The pipeline must leave the loss episode: a usable observation
    // reconnects tracking even though the enter threshold is never reached.
    assert!(
        states
            .iter()
            .any(|s| matches!(s, TrackingState::Tracking | TrackingState::Acquiring)),
        "dim-light reacquire should reconnect, states: {states:?}"
    );

    // Frames keep flowing through the whole episode (glide/decay never
    // starve the avatar bridge).
    assert_eq!(outputs.len(), stream.len());

    // The first reconnected output must blend toward the tracked pose, not
    // snap to it.
    let target_yaw = 20.0f32.to_radians();
    let reconnected = &outputs[12];
    assert!(
        reconnected.head.yaw_rad.abs() < target_yaw,
        "reacquire should not snap to the target, got {}",
        reconnected.head.yaw_rad
    );
    // And the pose must be moving toward the tracked target again.
    let last = outputs.last().expect("stream should produce frames");
    assert!(
        last.head.yaw_rad.abs() > reconnected.head.yaw_rad.abs(),
        "reconnected pose should approach the target, got {} from {}",
        last.head.yaw_rad,
        reconnected.head.yaw_rad
    );
}

#[test]
fn replay_reset_boundary_clears_state() {
    let stream = synthetic_stream();

    // Run the whole stream once without reset.
    let mut pipeline_once = TrackingPipeline::new(config()).unwrap();
    pipeline_once.apply_calibration(neutral_profile()).unwrap();
    let outputs_once = replay(&stream, &mut pipeline_once);

    // Run the stream, reset, then run the same stream again. The post-reset
    // outputs must match a fresh pipeline run of the same suffix.
    let mut pipeline_reset = TrackingPipeline::new(config()).unwrap();
    pipeline_reset.apply_calibration(neutral_profile()).unwrap();
    let first_half = &stream[..stream.len() / 2];
    let second_half = &stream[stream.len() / 2..];
    let _ = replay(first_half, &mut pipeline_reset);
    pipeline_reset.reset();
    let outputs_after_reset = replay(second_half, &mut pipeline_reset);

    let mut pipeline_fresh = TrackingPipeline::new(config()).unwrap();
    pipeline_fresh.apply_calibration(neutral_profile()).unwrap();
    let outputs_fresh = replay(second_half, &mut pipeline_fresh);

    assert_eq!(outputs_after_reset.len(), outputs_fresh.len());
    for (a, b) in outputs_after_reset.iter().zip(outputs_fresh.iter()) {
        assert_frames_eq(a, b);
    }

    // Sanity: the non-reset run produced different (stateful) output over the
    // full stream than the reset run over just the suffix.
    assert_eq!(
        outputs_once.len(),
        outputs_after_reset.len() + first_half.len()
    );
}

#[test]
fn replay_new_calibration_boundary_resets_filter_state() {
    // Run the same stream twice with recalibration applied at the same point.
    // The outputs must be identical, proving the boundary is deterministic.
    let stream = synthetic_stream();
    let split = stream.len() / 2;
    let first_half = &stream[..split];
    let second_half = &stream[split..];

    let mut pipeline_a = TrackingPipeline::new(config()).unwrap();
    pipeline_a.apply_calibration(neutral_profile()).unwrap();
    let _ = replay(first_half, &mut pipeline_a);
    pipeline_a.apply_calibration(neutral_profile()).unwrap();
    let outputs_a = replay(second_half, &mut pipeline_a);

    let mut pipeline_b = TrackingPipeline::new(config()).unwrap();
    pipeline_b.apply_calibration(neutral_profile()).unwrap();
    let _ = replay(first_half, &mut pipeline_b);
    pipeline_b.apply_calibration(neutral_profile()).unwrap();
    let outputs_b = replay(second_half, &mut pipeline_b);

    assert_eq!(outputs_a.len(), outputs_b.len());
    for (a, b) in outputs_a.iter().zip(outputs_b.iter()) {
        assert_frames_eq(a, b);
    }

    // Filter reset is observable: after recalibration, a neutral observation
    // yields near-identity head pose even if the preceding frames had the
    // head turned.
    let mut pipeline = TrackingPipeline::new(config()).unwrap();
    pipeline.apply_calibration(neutral_profile()).unwrap();
    let turned = rotate_landmarks(
        &synthetic_landmarks(),
        HeadPose {
            yaw_rad: 30.0f32.to_radians(),
            pitch_rad: 0.0,
            roll_rad: 0.0,
        },
    );
    let _ = replay(
        &[Some(observation(1, turned, relaxed_expression(), None))],
        &mut pipeline,
    );
    pipeline.apply_calibration(neutral_profile()).unwrap();
    let outputs = replay(
        &[Some(observation(
            2,
            synthetic_landmarks(),
            relaxed_expression(),
            None,
        ))],
        &mut pipeline,
    );
    let frame = outputs.first().expect("should emit frame");
    assert_relative_eq!(frame.head.yaw_rad, 0.0, epsilon = 1e-3);
}

#[test]
fn replay_uses_persisted_settings() {
    let persisted = TrackingPipelineSettings::new();
    let runtime_config = PipelineConfig::from_settings(&persisted);
    assert_eq!(runtime_config.calibration.required_sample_count(), 30);

    let mut pipeline = TrackingPipeline::new(runtime_config).unwrap();
    pipeline.apply_calibration(neutral_profile()).unwrap();

    let stream = synthetic_stream();
    let outputs = replay(&stream, &mut pipeline);
    assert!(!outputs.is_empty());
}

#[test]
fn replay_gaze_appears_when_blendshapes_present() {
    let mut pipeline = TrackingPipeline::new(config()).unwrap();
    pipeline.apply_calibration(neutral_profile()).unwrap();

    let stream = vec![Some(observation(
        1,
        synthetic_landmarks(),
        relaxed_expression(),
        Some(vec![
            NamedCoefficient {
                name: "eyeLookLeft".into(),
                value: 0.0,
            },
            NamedCoefficient {
                name: "eyeLookRight".into(),
                value: 0.6,
            },
            NamedCoefficient {
                name: "eyeLookUp".into(),
                value: 0.0,
            },
            NamedCoefficient {
                name: "eyeLookDown".into(),
                value: 0.3,
            },
        ]),
    ))];

    let outputs = replay(&stream, &mut pipeline);
    let frame = outputs.first().expect("should emit frame");
    assert!(frame.gaze.horizontal > 0.0, "right gaze must be positive");
    assert!(frame.gaze.vertical < 0.0, "down gaze must be negative");
}
