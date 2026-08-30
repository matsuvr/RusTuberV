//! Pure lifecycle and warm-start ownership for a persistent GNM tracker.
//!
//! This module intentionally does not contain a solver or temporal penalty. It
//! decides which dynamic state may be reused as an optimizer initialization,
//! rejects duplicate/regressed source frames, and prevents invalid or stale
//! dynamic state from becoming the new valid authority. Fixed identity is not
//! stored here at all, so lifecycle transitions cannot mutate calibration.

/// Identity of one source frame on the monotonic capture timeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GnmFrameStamp {
    /// Monotonic source-frame sequence.
    pub source_seq: u64,
    /// Monotonic capture timestamp in microseconds.
    pub captured_at_micros: u64,
}

/// Internal lifecycle phase of the persistent GNM estimator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentGnmPhase {
    /// No fixed identity/calibration is available.
    Uncalibrated,
    /// Calibration exists but no valid dynamic fit has been published yet.
    ReadyForFirstFit,
    /// A recent valid dynamic state exists.
    Tracking,
    /// Recent tracking degraded, but a bounded-age valid state may still seed optimization.
    Degraded,
    /// Dynamic state is too stale/invalid to reuse.
    Lost,
    /// A fresh observation is being fitted after stale/lost dynamic state was cleared.
    Reacquiring,
}

/// Initialization selected for the next bounded per-frame solve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmFitInitialization {
    /// First fit after explicit calibration starts from neutral dynamic state.
    NeutralFirstFit,
    /// Reuse the previous valid dynamic state as optimizer initialization only.
    PreviousValid {
        /// Exact previous valid source frame being reused.
        source: GnmFrameStamp,
    },
    /// Reinitialize expression/joints safely and obtain pose from the current observation.
    ReinitializeDynamicState,
}

/// Outcome classification returned by the numerical fitter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmFitOutcome {
    /// Fit passed finite/bounds/residual validity checks and may become authority.
    Valid,
    /// Fit failed validation and must not replace the previous valid state.
    Invalid,
}

/// Pure lifecycle input event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentGnmEvent {
    /// Explicit calibration became ready. Dynamic state is reset.
    CalibrationReady,
    /// Calibration was invalidated/replaced. Dynamic state is reset and fitting stops.
    CalibrationInvalidated,
    /// One admitted source frame, optionally with a usable face observation.
    SourceFrame {
        /// Source identity/timestamp.
        stamp: GnmFrameStamp,
        /// Whether this frame has enough observation data to attempt a solve.
        observation_available: bool,
    },
    /// Result of the single currently pending bounded solve.
    FitResult {
        /// Source frame that was fitted.
        stamp: GnmFrameStamp,
        /// Validity classification of the result.
        outcome: GnmFitOutcome,
    },
}

/// Explicit action emitted by one lifecycle transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentGnmAction {
    /// Calibration event cleared all dynamic tracker state.
    ResetDynamicState,
    /// Source frame was observed while calibration was unavailable; no solve is allowed.
    SkipUncalibratedFrame,
    /// No usable observation was available for this frame.
    NoObservation {
        /// Whether the previous dynamic state crossed the configured stale-age limit and was cleared.
        dynamic_state_cleared: bool,
    },
    /// Start exactly one bounded solve with the selected initialization.
    StartFit {
        /// Initialization policy for this solve.
        initialization: GnmFitInitialization,
    },
    /// Current valid fit may become the new published/previous-valid dynamic state.
    PublishCurrentFit,
    /// Current invalid fit is rejected; a previous valid state, if any, remains stored internally.
    RejectInvalidFit,
    /// Repeated invalid fits crossed the configured bound; stale dynamic state was cleared.
    RejectInvalidFitAndLose,
}

/// Configuration governing only reuse/lifecycle bounds, not temporal smoothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentGnmLifecycleConfig {
    max_warm_start_gap_micros: u64,
    max_dynamic_reuse_gap_micros: u64,
    max_consecutive_invalid_fits: u32,
}

impl PersistentGnmLifecycleConfig {
    /// Creates a lifecycle configuration without implicit timing constants.
    pub fn new(
        max_warm_start_gap_micros: u64,
        max_dynamic_reuse_gap_micros: u64,
        max_consecutive_invalid_fits: u32,
    ) -> Result<Self, PersistentGnmLifecycleError> {
        if max_warm_start_gap_micros == 0 {
            return Err(PersistentGnmLifecycleError::InvalidConfig(
                "max_warm_start_gap_micros must be positive",
            ));
        }
        if max_dynamic_reuse_gap_micros < max_warm_start_gap_micros {
            return Err(PersistentGnmLifecycleError::InvalidConfig(
                "max_dynamic_reuse_gap_micros must be >= max_warm_start_gap_micros",
            ));
        }
        if max_consecutive_invalid_fits == 0 {
            return Err(PersistentGnmLifecycleError::InvalidConfig(
                "max_consecutive_invalid_fits must be positive",
            ));
        }
        Ok(Self {
            max_warm_start_gap_micros,
            max_dynamic_reuse_gap_micros,
            max_consecutive_invalid_fits,
        })
    }

    /// Returns the maximum age at which a previous valid state may warm-start a solve.
    pub fn max_warm_start_gap_micros(self) -> u64 {
        self.max_warm_start_gap_micros
    }

    /// Returns the maximum age at which dynamic state may remain reusable at all.
    pub fn max_dynamic_reuse_gap_micros(self) -> u64 {
        self.max_dynamic_reuse_gap_micros
    }

    /// Returns the consecutive-invalid-fit limit before dynamic state is discarded.
    pub fn max_consecutive_invalid_fits(self) -> u32 {
        self.max_consecutive_invalid_fits
    }
}

/// Explicit persistent lifecycle state carried between pure transitions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentGnmLifecycleState {
    /// Current lifecycle phase.
    pub phase: PersistentGnmPhase,
    /// Last admitted source frame, including no-face frames.
    pub last_input: Option<GnmFrameStamp>,
    /// Last fit that was validated and allowed to become dynamic-state authority.
    pub previous_valid: Option<GnmFrameStamp>,
    /// Source frame whose bounded solve is currently pending.
    pub pending_fit: Option<GnmFrameStamp>,
    /// Number of consecutive invalid fit results since the last valid fit/reset.
    pub consecutive_invalid_fits: u32,
}

impl Default for PersistentGnmLifecycleState {
    fn default() -> Self {
        Self {
            phase: PersistentGnmPhase::Uncalibrated,
            last_input: None,
            previous_valid: None,
            pending_fit: None,
            consecutive_invalid_fits: 0,
        }
    }
}

/// Result of one pure lifecycle transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentGnmLifecycleDecision {
    /// State to carry into the next transition.
    pub state: PersistentGnmLifecycleState,
    /// Explicit caller action for this event.
    pub action: PersistentGnmAction,
}

/// Advances persistent GNM state ownership without running numerical fitting.
///
/// The function contains no hold/prediction/smoothing policy. Missing or invalid
/// frames never authorize republishing the previous valid GNM state here.
pub fn advance_persistent_gnm_lifecycle(
    previous: PersistentGnmLifecycleState,
    event: PersistentGnmEvent,
    config: PersistentGnmLifecycleConfig,
) -> Result<PersistentGnmLifecycleDecision, PersistentGnmLifecycleError> {
    match event {
        PersistentGnmEvent::CalibrationReady => Ok(PersistentGnmLifecycleDecision {
            state: PersistentGnmLifecycleState {
                phase: PersistentGnmPhase::ReadyForFirstFit,
                ..PersistentGnmLifecycleState::default()
            },
            action: PersistentGnmAction::ResetDynamicState,
        }),
        PersistentGnmEvent::CalibrationInvalidated => Ok(PersistentGnmLifecycleDecision {
            state: PersistentGnmLifecycleState::default(),
            action: PersistentGnmAction::ResetDynamicState,
        }),
        PersistentGnmEvent::SourceFrame {
            stamp,
            observation_available,
        } => advance_source_frame(previous, stamp, observation_available, config),
        PersistentGnmEvent::FitResult { stamp, outcome } => {
            advance_fit_result(previous, stamp, outcome, config)
        }
    }
}

// Invariant: `previous_valid_is_warm` proves `previous_valid` is `Some` on
// the branch that reads it.
#[allow(clippy::expect_used)]
fn advance_source_frame(
    previous: PersistentGnmLifecycleState,
    stamp: GnmFrameStamp,
    observation_available: bool,
    config: PersistentGnmLifecycleConfig,
) -> Result<PersistentGnmLifecycleDecision, PersistentGnmLifecycleError> {
    if let Some(pending) = previous.pending_fit {
        return Err(PersistentGnmLifecycleError::FitStillPending {
            pending,
            incoming: stamp,
        });
    }
    validate_monotonic_source(previous.last_input, stamp)?;

    if previous.phase == PersistentGnmPhase::Uncalibrated {
        let mut state = previous;
        state.last_input = Some(stamp);
        return Ok(PersistentGnmLifecycleDecision {
            state,
            action: PersistentGnmAction::SkipUncalibratedFrame,
        });
    }

    if !observation_available {
        let mut state = previous;
        state.last_input = Some(stamp);
        let stale = state.previous_valid.is_none_or(|valid| {
            stamp
                .captured_at_micros
                .saturating_sub(valid.captured_at_micros)
                > config.max_dynamic_reuse_gap_micros
        });
        if stale {
            state.previous_valid = None;
            state.consecutive_invalid_fits = 0;
            state.phase = PersistentGnmPhase::Lost;
        } else {
            state.phase = PersistentGnmPhase::Degraded;
        }
        return Ok(PersistentGnmLifecycleDecision {
            state,
            action: PersistentGnmAction::NoObservation {
                dynamic_state_cleared: stale,
            },
        });
    }

    let gap_from_last_input = previous
        .last_input
        .map(|last| stamp.captured_at_micros - last.captured_at_micros);
    let long_input_gap =
        gap_from_last_input.is_some_and(|gap| gap > config.max_dynamic_reuse_gap_micros);

    let previous_valid_is_warm = previous.previous_valid.is_some_and(|valid| {
        stamp
            .captured_at_micros
            .saturating_sub(valid.captured_at_micros)
            <= config.max_warm_start_gap_micros
    });

    let initialization = if previous.phase == PersistentGnmPhase::ReadyForFirstFit
        && previous.previous_valid.is_none()
    {
        GnmFitInitialization::NeutralFirstFit
    } else if !long_input_gap
        && previous_valid_is_warm
        && matches!(
            previous.phase,
            PersistentGnmPhase::Tracking | PersistentGnmPhase::Degraded
        )
    {
        GnmFitInitialization::PreviousValid {
            source: previous.previous_valid.expect("checked Some above"),
        }
    } else {
        GnmFitInitialization::ReinitializeDynamicState
    };

    let reinitializing = matches!(
        initialization,
        GnmFitInitialization::ReinitializeDynamicState
    );
    let mut state = previous;
    state.last_input = Some(stamp);
    state.pending_fit = Some(stamp);
    if long_input_gap || reinitializing {
        state.previous_valid = None;
        state.consecutive_invalid_fits = 0;
        state.phase = PersistentGnmPhase::Reacquiring;
    }

    Ok(PersistentGnmLifecycleDecision {
        state,
        action: PersistentGnmAction::StartFit { initialization },
    })
}

fn advance_fit_result(
    previous: PersistentGnmLifecycleState,
    stamp: GnmFrameStamp,
    outcome: GnmFitOutcome,
    config: PersistentGnmLifecycleConfig,
) -> Result<PersistentGnmLifecycleDecision, PersistentGnmLifecycleError> {
    let Some(pending) = previous.pending_fit else {
        return Err(PersistentGnmLifecycleError::UnexpectedFitResult {
            expected: None,
            actual: stamp,
        });
    };
    if pending != stamp {
        return Err(PersistentGnmLifecycleError::UnexpectedFitResult {
            expected: Some(pending),
            actual: stamp,
        });
    }

    let mut state = previous;
    state.pending_fit = None;
    match outcome {
        GnmFitOutcome::Valid => {
            state.previous_valid = Some(stamp);
            state.consecutive_invalid_fits = 0;
            state.phase = PersistentGnmPhase::Tracking;
            Ok(PersistentGnmLifecycleDecision {
                state,
                action: PersistentGnmAction::PublishCurrentFit,
            })
        }
        GnmFitOutcome::Invalid => {
            state.consecutive_invalid_fits = state.consecutive_invalid_fits.saturating_add(1);
            if state.consecutive_invalid_fits >= config.max_consecutive_invalid_fits {
                state.previous_valid = None;
                state.phase = PersistentGnmPhase::Lost;
                Ok(PersistentGnmLifecycleDecision {
                    state,
                    action: PersistentGnmAction::RejectInvalidFitAndLose,
                })
            } else {
                state.phase = PersistentGnmPhase::Degraded;
                Ok(PersistentGnmLifecycleDecision {
                    state,
                    action: PersistentGnmAction::RejectInvalidFit,
                })
            }
        }
    }
}

fn validate_monotonic_source(
    previous: Option<GnmFrameStamp>,
    incoming: GnmFrameStamp,
) -> Result<(), PersistentGnmLifecycleError> {
    let Some(previous) = previous else {
        return Ok(());
    };
    if incoming.source_seq == previous.source_seq {
        return Err(PersistentGnmLifecycleError::DuplicateSourceSequence {
            source_seq: incoming.source_seq,
        });
    }
    if incoming.source_seq < previous.source_seq {
        return Err(PersistentGnmLifecycleError::RegressedSourceSequence {
            previous: previous.source_seq,
            incoming: incoming.source_seq,
        });
    }
    if incoming.captured_at_micros <= previous.captured_at_micros {
        return Err(PersistentGnmLifecycleError::RegressedTimestamp {
            previous: previous.captured_at_micros,
            incoming: incoming.captured_at_micros,
        });
    }
    Ok(())
}

/// Typed lifecycle validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PersistentGnmLifecycleError {
    /// Lifecycle timing/invalid-fit bounds are inconsistent.
    InvalidConfig(&'static str),
    /// Same source sequence was admitted twice.
    DuplicateSourceSequence {
        /// Duplicated source sequence.
        source_seq: u64,
    },
    /// Source sequence moved backwards.
    RegressedSourceSequence {
        /// Previous admitted sequence.
        previous: u64,
        /// Incoming sequence.
        incoming: u64,
    },
    /// Capture timestamp failed strict monotonicity.
    RegressedTimestamp {
        /// Previous admitted capture timestamp.
        previous: u64,
        /// Incoming capture timestamp.
        incoming: u64,
    },
    /// A new source frame arrived while a solve was still pending.
    FitStillPending {
        /// Pending solve source.
        pending: GnmFrameStamp,
        /// New incoming source that must be handled by the external latest-frame queue policy.
        incoming: GnmFrameStamp,
    },
    /// Fit result did not correspond to the one pending solve.
    UnexpectedFitResult {
        /// Pending source when one exists.
        expected: Option<GnmFrameStamp>,
        /// Actual fit-result source.
        actual: GnmFrameStamp,
    },
}

impl std::fmt::Display for PersistentGnmLifecycleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(reason) => {
                write!(formatter, "invalid GNM lifecycle config: {reason}")
            }
            Self::DuplicateSourceSequence { source_seq } => {
                write!(formatter, "duplicate GNM source sequence {source_seq}")
            }
            Self::RegressedSourceSequence { previous, incoming } => write!(
                formatter,
                "GNM source sequence regressed from {previous} to {incoming}"
            ),
            Self::RegressedTimestamp { previous, incoming } => write!(
                formatter,
                "GNM capture timestamp regressed from {previous} to {incoming}"
            ),
            Self::FitStillPending { pending, incoming } => write!(
                formatter,
                "GNM fit for source {} is still pending when source {} arrived",
                pending.source_seq, incoming.source_seq
            ),
            Self::UnexpectedFitResult { expected, actual } => match expected {
                Some(expected) => write!(
                    formatter,
                    "GNM fit result source {} does not match pending source {}",
                    actual.source_seq, expected.source_seq
                ),
                None => write!(
                    formatter,
                    "GNM fit result source {} arrived with no pending fit",
                    actual.source_seq
                ),
            },
        }
    }
}

impl std::error::Error for PersistentGnmLifecycleError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PersistentGnmLifecycleConfig {
        PersistentGnmLifecycleConfig::new(50_000, 250_000, 3).unwrap()
    }

    fn stamp(source_seq: u64, captured_at_micros: u64) -> GnmFrameStamp {
        GnmFrameStamp {
            source_seq,
            captured_at_micros,
        }
    }

    fn calibrated() -> PersistentGnmLifecycleState {
        advance_persistent_gnm_lifecycle(
            PersistentGnmLifecycleState::default(),
            PersistentGnmEvent::CalibrationReady,
            config(),
        )
        .unwrap()
        .state
    }

    fn fit_valid(
        state: PersistentGnmLifecycleState,
        frame: GnmFrameStamp,
    ) -> PersistentGnmLifecycleState {
        let started = advance_persistent_gnm_lifecycle(
            state,
            PersistentGnmEvent::SourceFrame {
                stamp: frame,
                observation_available: true,
            },
            config(),
        )
        .unwrap();
        advance_persistent_gnm_lifecycle(
            started.state,
            PersistentGnmEvent::FitResult {
                stamp: frame,
                outcome: GnmFitOutcome::Valid,
            },
            config(),
        )
        .unwrap()
        .state
    }

    #[test]
    fn first_fit_is_neutral_then_next_frame_warm_starts_previous_valid() {
        let state = calibrated();
        let first = stamp(1, 1_000_000);
        let first_start = advance_persistent_gnm_lifecycle(
            state,
            PersistentGnmEvent::SourceFrame {
                stamp: first,
                observation_available: true,
            },
            config(),
        )
        .unwrap();
        assert_eq!(
            first_start.action,
            PersistentGnmAction::StartFit {
                initialization: GnmFitInitialization::NeutralFirstFit
            }
        );
        let first_valid = advance_persistent_gnm_lifecycle(
            first_start.state,
            PersistentGnmEvent::FitResult {
                stamp: first,
                outcome: GnmFitOutcome::Valid,
            },
            config(),
        )
        .unwrap();
        assert_eq!(first_valid.action, PersistentGnmAction::PublishCurrentFit);

        let second = stamp(2, 1_016_000);
        let second_start = advance_persistent_gnm_lifecycle(
            first_valid.state,
            PersistentGnmEvent::SourceFrame {
                stamp: second,
                observation_available: true,
            },
            config(),
        )
        .unwrap();
        assert_eq!(
            second_start.action,
            PersistentGnmAction::StartFit {
                initialization: GnmFitInitialization::PreviousValid { source: first }
            }
        );
    }

    #[test]
    fn same_source_cannot_be_fitted_or_published_twice() {
        let first = stamp(1, 1_000_000);
        let tracked = fit_valid(calibrated(), first);
        assert!(matches!(
            advance_persistent_gnm_lifecycle(
                tracked,
                PersistentGnmEvent::SourceFrame {
                    stamp: first,
                    observation_available: true,
                },
                config(),
            ),
            Err(PersistentGnmLifecycleError::DuplicateSourceSequence { source_seq: 1 })
        ));
        assert!(matches!(
            advance_persistent_gnm_lifecycle(
                tracked,
                PersistentGnmEvent::FitResult {
                    stamp: first,
                    outcome: GnmFitOutcome::Valid,
                },
                config(),
            ),
            Err(PersistentGnmLifecycleError::UnexpectedFitResult { expected: None, .. })
        ));
    }

    #[test]
    fn source_sequence_and_timestamp_regression_are_fail_closed() {
        let first = stamp(10, 1_000_000);
        let tracked = fit_valid(calibrated(), first);
        assert!(matches!(
            advance_persistent_gnm_lifecycle(
                tracked,
                PersistentGnmEvent::SourceFrame {
                    stamp: stamp(9, 1_016_000),
                    observation_available: true,
                },
                config(),
            ),
            Err(PersistentGnmLifecycleError::RegressedSourceSequence { .. })
        ));
        assert!(matches!(
            advance_persistent_gnm_lifecycle(
                tracked,
                PersistentGnmEvent::SourceFrame {
                    stamp: stamp(11, 999_000),
                    observation_available: true,
                },
                config(),
            ),
            Err(PersistentGnmLifecycleError::RegressedTimestamp { .. })
        ));
    }

    #[test]
    fn invalid_fit_does_not_replace_previous_valid_state() {
        let first = stamp(1, 1_000_000);
        let tracked = fit_valid(calibrated(), first);
        let second = stamp(2, 1_016_000);
        let started = advance_persistent_gnm_lifecycle(
            tracked,
            PersistentGnmEvent::SourceFrame {
                stamp: second,
                observation_available: true,
            },
            config(),
        )
        .unwrap();
        let rejected = advance_persistent_gnm_lifecycle(
            started.state,
            PersistentGnmEvent::FitResult {
                stamp: second,
                outcome: GnmFitOutcome::Invalid,
            },
            config(),
        )
        .unwrap();
        assert_eq!(rejected.action, PersistentGnmAction::RejectInvalidFit);
        assert_eq!(rejected.state.previous_valid, Some(first));
        assert_eq!(rejected.state.phase, PersistentGnmPhase::Degraded);

        let third = stamp(3, 1_032_000);
        let retry = advance_persistent_gnm_lifecycle(
            rejected.state,
            PersistentGnmEvent::SourceFrame {
                stamp: third,
                observation_available: true,
            },
            config(),
        )
        .unwrap();
        assert_eq!(
            retry.action,
            PersistentGnmAction::StartFit {
                initialization: GnmFitInitialization::PreviousValid { source: first }
            }
        );
    }

    #[test]
    fn repeated_invalid_fits_clear_dynamic_state_at_configured_bound() {
        let first = stamp(1, 1_000_000);
        let mut state = fit_valid(calibrated(), first);
        for index in 0..2 {
            let frame = stamp(2 + index, 1_016_000 + index * 16_000);
            let started = advance_persistent_gnm_lifecycle(
                state,
                PersistentGnmEvent::SourceFrame {
                    stamp: frame,
                    observation_available: true,
                },
                config(),
            )
            .unwrap();
            let rejected = advance_persistent_gnm_lifecycle(
                started.state,
                PersistentGnmEvent::FitResult {
                    stamp: frame,
                    outcome: GnmFitOutcome::Invalid,
                },
                config(),
            )
            .unwrap();
            assert_eq!(rejected.action, PersistentGnmAction::RejectInvalidFit);
            state = rejected.state;
        }
        let frame = stamp(4, 1_048_000);
        let started = advance_persistent_gnm_lifecycle(
            state,
            PersistentGnmEvent::SourceFrame {
                stamp: frame,
                observation_available: true,
            },
            config(),
        )
        .unwrap();
        let lost = advance_persistent_gnm_lifecycle(
            started.state,
            PersistentGnmEvent::FitResult {
                stamp: frame,
                outcome: GnmFitOutcome::Invalid,
            },
            config(),
        )
        .unwrap();
        assert_eq!(lost.action, PersistentGnmAction::RejectInvalidFitAndLose);
        assert_eq!(lost.state.previous_valid, None);
        assert_eq!(lost.state.phase, PersistentGnmPhase::Lost);
    }

    #[test]
    fn short_no_face_gap_degrades_but_long_gap_clears_stale_expression_state() {
        let first = stamp(1, 1_000_000);
        let tracked = fit_valid(calibrated(), first);
        let short_gap = advance_persistent_gnm_lifecycle(
            tracked,
            PersistentGnmEvent::SourceFrame {
                stamp: stamp(2, 1_040_000),
                observation_available: false,
            },
            config(),
        )
        .unwrap();
        assert_eq!(short_gap.state.previous_valid, Some(first));
        assert_eq!(short_gap.state.phase, PersistentGnmPhase::Degraded);
        assert_eq!(
            short_gap.action,
            PersistentGnmAction::NoObservation {
                dynamic_state_cleared: false
            }
        );

        let long_gap = advance_persistent_gnm_lifecycle(
            short_gap.state,
            PersistentGnmEvent::SourceFrame {
                stamp: stamp(3, 1_300_001),
                observation_available: false,
            },
            config(),
        )
        .unwrap();
        assert_eq!(long_gap.state.previous_valid, None);
        assert_eq!(long_gap.state.phase, PersistentGnmPhase::Lost);
        assert_eq!(
            long_gap.action,
            PersistentGnmAction::NoObservation {
                dynamic_state_cleared: true
            }
        );
    }

    #[test]
    fn reacquire_after_long_gap_never_warm_starts_stale_dynamic_state() {
        let first = stamp(1, 1_000_000);
        let tracked = fit_valid(calibrated(), first);
        let reacquire = advance_persistent_gnm_lifecycle(
            tracked,
            PersistentGnmEvent::SourceFrame {
                stamp: stamp(2, 1_300_001),
                observation_available: true,
            },
            config(),
        )
        .unwrap();
        assert_eq!(reacquire.state.previous_valid, None);
        assert_eq!(reacquire.state.phase, PersistentGnmPhase::Reacquiring);
        assert_eq!(
            reacquire.action,
            PersistentGnmAction::StartFit {
                initialization: GnmFitInitialization::ReinitializeDynamicState
            }
        );
    }

    #[test]
    fn new_frame_while_fit_pending_is_owned_by_external_latest_frame_policy() {
        let first = stamp(1, 1_000_000);
        let started = advance_persistent_gnm_lifecycle(
            calibrated(),
            PersistentGnmEvent::SourceFrame {
                stamp: first,
                observation_available: true,
            },
            config(),
        )
        .unwrap();
        assert!(matches!(
            advance_persistent_gnm_lifecycle(
                started.state,
                PersistentGnmEvent::SourceFrame {
                    stamp: stamp(2, 1_016_000),
                    observation_available: true,
                },
                config(),
            ),
            Err(PersistentGnmLifecycleError::FitStillPending { .. })
        ));
    }

    #[test]
    fn explicit_calibration_events_are_the_only_identity_lifecycle_boundary() {
        let first = stamp(1, 1_000_000);
        let tracked = fit_valid(calibrated(), first);
        let invalidated = advance_persistent_gnm_lifecycle(
            tracked,
            PersistentGnmEvent::CalibrationInvalidated,
            config(),
        )
        .unwrap();
        assert_eq!(invalidated.state, PersistentGnmLifecycleState::default());
        assert_eq!(invalidated.action, PersistentGnmAction::ResetDynamicState);

        let recalibrated = advance_persistent_gnm_lifecycle(
            invalidated.state,
            PersistentGnmEvent::CalibrationReady,
            config(),
        )
        .unwrap();
        assert_eq!(
            recalibrated.state.phase,
            PersistentGnmPhase::ReadyForFirstFit
        );
        assert_eq!(recalibrated.state.previous_valid, None);
    }
}
