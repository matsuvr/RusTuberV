// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Table-driven integration tests for the tracking state machine.

use std::time::Duration;

use vtuber_core::{
    FrameSeq, Landmark3, LandmarkSchemaId, MonoTimeNs, NamedCoefficient, NormalizedRect,
    RawExpressionObservation, RawFaceObservation,
};
use vtuber_tracking::confidence::ConfidenceSignal;
use vtuber_tracking::state_machine::{
    StateMachineParams, TrackingAction, TrackingState, TrackingStateMachine, TransitionInput,
};

fn params() -> StateMachineParams {
    StateMachineParams {
        hold_duration: Duration::from_millis(100),
        return_duration: Duration::from_millis(200),
    }
}

fn dummy_observation() -> RawFaceObservation {
    RawFaceObservation {
        source_seq: FrameSeq(1),
        captured_at: MonoTimeNs(0),
        inference_started_at: MonoTimeNs(0),
        inference_finished_at: MonoTimeNs(0),
        face_confidence: 0.9,
        landmarks: vec![Landmark3 {
            x: 0.5,
            y: 0.5,
            z: 0.0,
            visibility: 1.0,
        }],
        blendshapes: Some(vec![NamedCoefficient {
            name: "blinkLeft".to_string(),
            value: 0.0,
        }]),
        expressions: RawExpressionObservation::default(),
        roi: NormalizedRect {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            rotation_rad: 0.0,
        },
        schema: LandmarkSchemaId("test"),
    }
}

fn input_with_observation(signal: ConfidenceSignal) -> TransitionInput<'static> {
    // Leak a stable observation so the borrowed reference lives for the
    // `'static` test lifetime. The state machine only inspects presence.
    let obs: &'static RawFaceObservation = Box::leak(Box::new(dummy_observation()));
    TransitionInput {
        signal,
        dt: Duration::from_millis(16),
        observation: Some(obs),
        observation_usable: true,
        calibration_available: true,
    }
}

fn input_without_observation(signal: ConfidenceSignal, dt: Duration) -> TransitionInput<'static> {
    TransitionInput {
        signal,
        dt,
        observation: None,
        observation_usable: false,
        calibration_available: true,
    }
}

fn input_with_unusable_observation(signal: ConfidenceSignal) -> TransitionInput<'static> {
    let obs: &'static RawFaceObservation = Box::leak(Box::new(dummy_observation()));
    TransitionInput {
        signal,
        dt: Duration::from_millis(16),
        observation: Some(obs),
        observation_usable: false,
        calibration_available: true,
    }
}

fn assert_transition(
    initial: TrackingState,
    input: TransitionInput<'_>,
    expected: TrackingState,
    expected_actions: &[TrackingAction],
) -> TrackingStateMachine {
    let mut sm = TrackingStateMachine::with_initial_state(params(), initial).unwrap();
    let result = sm.update(input);
    assert_eq!(
        result.previous, initial,
        "previous state mismatch for transition from {:?}",
        initial
    );
    assert_eq!(
        result.current,
        expected,
        "expected {:?} but got {:?} from {:?} with signal {:?} / observation {:?}",
        expected,
        result.current,
        initial,
        input.signal,
        input.observation.is_some()
    );
    assert_eq!(
        result.actions, expected_actions,
        "action mismatch for {:?} -> {:?}",
        initial, expected
    );
    sm
}

#[cfg(test)]
mod tracking_state_machine {
    use super::*;

    #[test]
    fn searching_stays_without_observation() {
        assert_transition(
            TrackingState::Searching,
            input_without_observation(ConfidenceSignal::Acquire, Duration::from_millis(16)),
            TrackingState::Searching,
            &[],
        );
    }

    #[test]
    fn searching_acquires_to_acquiring() {
        assert_transition(
            TrackingState::Searching,
            input_with_observation(ConfidenceSignal::Acquire),
            TrackingState::Acquiring,
            &[TrackingAction::ResetFilters],
        );
    }

    #[test]
    fn acquiring_enters_tracking_when_calibrated() {
        assert_transition(
            TrackingState::Acquiring,
            input_with_observation(ConfidenceSignal::Acquire),
            TrackingState::Tracking,
            &[TrackingAction::ResetFilters],
        );
    }

    #[test]
    fn acquiring_stays_without_calibration() {
        let obs = dummy_observation();
        let input = TransitionInput {
            signal: ConfidenceSignal::Acquire,
            dt: Duration::from_millis(16),
            observation: Some(&obs),
            observation_usable: true,
            calibration_available: false,
        };
        assert_transition(
            TrackingState::Acquiring,
            input,
            TrackingState::Acquiring,
            &[],
        );
    }

    #[test]
    fn acquiring_lost_holds_without_observation() {
        assert_transition(
            TrackingState::Acquiring,
            input_without_observation(ConfidenceSignal::Degrade, Duration::from_millis(16)),
            TrackingState::LostHold,
            &[TrackingAction::StartHold],
        );
    }

    #[test]
    fn tracking_degrades_with_observation() {
        assert_transition(
            TrackingState::Tracking,
            input_with_observation(ConfidenceSignal::Degrade),
            TrackingState::Degraded,
            &[],
        );
    }

    #[test]
    fn tracking_holds_without_observation() {
        assert_transition(
            TrackingState::Tracking,
            input_without_observation(ConfidenceSignal::Degrade, Duration::from_millis(16)),
            TrackingState::LostHold,
            &[TrackingAction::StartHold],
        );
    }

    #[test]
    fn degraded_recovers_with_observation() {
        assert_transition(
            TrackingState::Degraded,
            input_with_observation(ConfidenceSignal::Acquire),
            TrackingState::Tracking,
            &[],
        );
    }

    #[test]
    fn degraded_holds_without_observation() {
        assert_transition(
            TrackingState::Degraded,
            input_without_observation(ConfidenceSignal::Degrade, Duration::from_millis(16)),
            TrackingState::LostHold,
            &[TrackingAction::StartHold],
        );
    }

    #[test]
    fn lost_hold_advances_to_returning_neutral_on_timeout() {
        assert_transition(
            TrackingState::LostHold,
            input_without_observation(ConfidenceSignal::None, params().hold_duration),
            TrackingState::ReturningNeutral,
            &[TrackingAction::StartReturnToNeutral],
        );
    }

    #[test]
    fn returning_neutral_advances_to_searching_on_timeout() {
        assert_transition(
            TrackingState::ReturningNeutral,
            input_without_observation(ConfidenceSignal::None, params().return_duration),
            TrackingState::Searching,
            &[],
        );
    }

    #[test]
    fn lost_hold_is_not_permanent() {
        let mut sm =
            TrackingStateMachine::with_initial_state(params(), TrackingState::LostHold).unwrap();

        let result = sm.update(input_without_observation(
            ConfidenceSignal::None,
            params().hold_duration,
        ));
        assert_eq!(result.current, TrackingState::ReturningNeutral);

        let result = sm.update(input_without_observation(
            ConfidenceSignal::None,
            params().return_duration,
        ));
        assert_eq!(result.current, TrackingState::Searching);
    }

    #[test]
    fn searching_does_not_jump_directly_to_tracking() {
        let mut sm = TrackingStateMachine::new(params()).unwrap();
        assert_eq!(sm.state(), TrackingState::Searching);

        let result = sm.update(input_with_observation(ConfidenceSignal::Acquire));
        assert_eq!(result.current, TrackingState::Acquiring);
        assert_ne!(result.current, TrackingState::Tracking);

        let result = sm.update(input_with_observation(ConfidenceSignal::Acquire));
        assert_eq!(result.current, TrackingState::Tracking);
    }

    #[test]
    fn returning_neutral_reacquires_to_acquiring() {
        assert_transition(
            TrackingState::ReturningNeutral,
            input_with_observation(ConfidenceSignal::Acquire),
            TrackingState::Acquiring,
            &[TrackingAction::ResetFilters],
        );
    }

    #[test]
    fn lost_hold_reacquires_to_acquiring_before_timeout() {
        assert_transition(
            TrackingState::LostHold,
            input_with_observation(ConfidenceSignal::Acquire),
            TrackingState::Acquiring,
            &[TrackingAction::ResetFilters],
        );
    }

    #[test]
    fn searching_reacquires_on_usable_observation_without_signal() {
        // A dim-light face that never reaches the enter threshold must still
        // reconnect instead of leaving the avatar in neutral.
        assert_transition(
            TrackingState::Searching,
            input_with_observation(ConfidenceSignal::None),
            TrackingState::Acquiring,
            &[TrackingAction::ResetFilters],
        );
    }

    #[test]
    fn searching_stays_on_unusable_observation_without_signal() {
        assert_transition(
            TrackingState::Searching,
            input_with_unusable_observation(ConfidenceSignal::None),
            TrackingState::Searching,
            &[],
        );
    }

    #[test]
    fn lost_hold_reacquires_on_usable_observation_without_signal() {
        assert_transition(
            TrackingState::LostHold,
            input_with_observation(ConfidenceSignal::None),
            TrackingState::Acquiring,
            &[TrackingAction::ResetFilters],
        );
    }

    #[test]
    fn usable_observation_beats_lost_hold_timeout() {
        // Even past the hold timeout, a usable observation reconnects
        // instead of decaying to neutral.
        let obs = dummy_observation();
        let mut sm =
            TrackingStateMachine::with_initial_state(params(), TrackingState::LostHold).unwrap();
        let result = sm.update(TransitionInput {
            signal: ConfidenceSignal::None,
            dt: params().hold_duration,
            observation: Some(&obs),
            observation_usable: true,
            calibration_available: true,
        });
        assert_eq!(result.current, TrackingState::Acquiring);
        assert_eq!(result.actions, vec![TrackingAction::ResetFilters]);
    }

    #[test]
    fn lost_hold_times_out_on_unusable_observation() {
        // A below-exit-threshold observation must not prevent the timeout.
        let obs = dummy_observation();
        let mut sm =
            TrackingStateMachine::with_initial_state(params(), TrackingState::LostHold).unwrap();
        let result = sm.update(TransitionInput {
            signal: ConfidenceSignal::None,
            dt: params().hold_duration,
            observation: Some(&obs),
            observation_usable: false,
            calibration_available: true,
        });
        assert_eq!(result.current, TrackingState::ReturningNeutral);
    }

    #[test]
    fn returning_neutral_reacquires_on_usable_observation_without_signal() {
        assert_transition(
            TrackingState::ReturningNeutral,
            input_with_observation(ConfidenceSignal::None),
            TrackingState::Acquiring,
            &[TrackingAction::ResetFilters],
        );
    }
}
