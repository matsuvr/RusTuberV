//! Source-aligned A/B backend contract and pure fallback arbitration.
//!
//! The types in this module make source-frame identity, latency timestamps, and
//! avatar output authority explicit. They intentionally do not run inference,
//! fitting, decoding, rendering, or persistence.

/// Backend that has authority to publish face-tracking output to the avatar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaceTrackingBackend {
    /// Existing stable MediaPipe path.
    DirectMediaPipe,
    /// Persistent temporal GNM path.
    GnmTemporal,
}

/// User/research runtime mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaceTrackingMode {
    /// Direct MediaPipe drives the avatar and GNM need not run.
    DirectMediaPipe,
    /// GNM is requested to drive the avatar when ready.
    GnmTemporal,
    /// Direct MediaPipe drives the avatar while GNM is evaluated for metrics.
    GnmTemporalShadow,
}

/// Immutable identity and shared inference timing for one camera source frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceFrameStamp {
    source_seq: u64,
    capture_micros: u64,
    inference_complete_micros: Option<u64>,
}

impl SourceFrameStamp {
    /// Creates a source-frame stamp.
    pub fn new(
        source_seq: u64,
        capture_micros: u64,
        inference_complete_micros: Option<u64>,
    ) -> Result<Self, AbBackendError> {
        if let Some(inference_complete) = inference_complete_micros
            && inference_complete < capture_micros
        {
            return Err(AbBackendError::InvalidTiming {
                field: "inference_complete_micros",
                reason: "inference completion cannot precede capture".to_owned(),
            });
        }
        Ok(Self {
            source_seq,
            capture_micros,
            inference_complete_micros,
        })
    }

    /// Returns the source frame sequence.
    pub fn source_seq(self) -> u64 {
        self.source_seq
    }

    /// Returns the source capture timestamp.
    pub fn capture_micros(self) -> u64 {
        self.capture_micros
    }

    /// Returns shared MediaPipe inference completion when instrumented.
    pub fn inference_complete_micros(self) -> Option<u64> {
        self.inference_complete_micros
    }
}

/// Timing attached to one backend output for a specific source frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendOutputTiming {
    source: SourceFrameStamp,
    backend: FaceTrackingBackend,
    fit_complete_micros: Option<u64>,
    decoder_complete_micros: Option<u64>,
    publish_micros: u64,
}

impl BackendOutputTiming {
    /// Creates a chronologically validated backend timing record.
    pub fn new(
        source: SourceFrameStamp,
        backend: FaceTrackingBackend,
        fit_complete_micros: Option<u64>,
        decoder_complete_micros: Option<u64>,
        publish_micros: u64,
    ) -> Result<Self, AbBackendError> {
        if backend == FaceTrackingBackend::DirectMediaPipe
            && (fit_complete_micros.is_some() || decoder_complete_micros.is_some())
        {
            return Err(AbBackendError::InvalidTiming {
                field: "direct backend stages",
                reason: "DirectMediaPipe must not report GNM fit/decoder stages".to_owned(),
            });
        }

        let inference_or_capture = source
            .inference_complete_micros()
            .unwrap_or(source.capture_micros());
        if let Some(fit_complete) = fit_complete_micros
            && fit_complete < inference_or_capture
        {
            return Err(AbBackendError::InvalidTiming {
                field: "fit_complete_micros",
                reason: "fit completion cannot precede shared inference completion".to_owned(),
            });
        }
        let fit_or_inference = fit_complete_micros.unwrap_or(inference_or_capture);
        if let Some(decoder_complete) = decoder_complete_micros
            && decoder_complete < fit_or_inference
        {
            return Err(AbBackendError::InvalidTiming {
                field: "decoder_complete_micros",
                reason: "decoder completion cannot precede fit/inference completion".to_owned(),
            });
        }
        let last_stage = decoder_complete_micros.unwrap_or(fit_or_inference);
        if publish_micros < last_stage {
            return Err(AbBackendError::InvalidTiming {
                field: "publish_micros",
                reason: "publish cannot precede the final instrumented stage".to_owned(),
            });
        }

        Ok(Self {
            source,
            backend,
            fit_complete_micros,
            decoder_complete_micros,
            publish_micros,
        })
    }

    /// Returns the source-frame stamp.
    pub fn source(self) -> SourceFrameStamp {
        self.source
    }

    /// Returns the backend that produced the output.
    pub fn backend(self) -> FaceTrackingBackend {
        self.backend
    }

    /// Returns optional GNM fit completion.
    pub fn fit_complete_micros(self) -> Option<u64> {
        self.fit_complete_micros
    }

    /// Returns optional GNM decoder completion.
    pub fn decoder_complete_micros(self) -> Option<u64> {
        self.decoder_complete_micros
    }

    /// Returns output publish timestamp.
    pub fn publish_micros(self) -> u64 {
        self.publish_micros
    }
}

/// One backend output carrying the common output type plus source timing.
#[derive(Clone, Debug, PartialEq)]
pub struct StampedBackendOutput<T> {
    /// Chronologically validated timing and source identity.
    pub timing: BackendOutputTiming,
    /// Common output payload, for example the canonical ARKit52 contract.
    pub output: T,
}

impl<T> StampedBackendOutput<T> {
    /// Wraps a backend output with validated timing.
    pub const fn new(timing: BackendOutputTiming, output: T) -> Self {
        Self { timing, output }
    }
}

/// Direct and GNM outputs proven to originate from the same shared source frame.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignedBackendOutputs<T> {
    /// Direct output.
    pub direct: StampedBackendOutput<T>,
    /// GNM output.
    pub gnm: StampedBackendOutput<T>,
}

impl<T> AlignedBackendOutputs<T> {
    /// Aligns outputs by exact shared source stamp, never by nearby publish time.
    pub fn new(
        direct: StampedBackendOutput<T>,
        gnm: StampedBackendOutput<T>,
    ) -> Result<Self, AbBackendError> {
        if direct.timing.backend() != FaceTrackingBackend::DirectMediaPipe {
            return Err(AbBackendError::WrongBackendRole {
                role: "direct",
                actual: direct.timing.backend(),
            });
        }
        if gnm.timing.backend() != FaceTrackingBackend::GnmTemporal {
            return Err(AbBackendError::WrongBackendRole {
                role: "gnm",
                actual: gnm.timing.backend(),
            });
        }
        if direct.timing.source() != gnm.timing.source() {
            return Err(AbBackendError::SourceMismatch {
                direct: direct.timing.source(),
                gnm: gnm.timing.source(),
            });
        }
        Ok(Self { direct, gnm })
    }

    /// Computes side-by-side latency using the same capture and inference stamps.
    pub fn latency_comparison(&self) -> AlignedLatencyComparison {
        let direct = backend_latency_metrics(self.direct.timing);
        let gnm = backend_latency_metrics(self.gnm.timing);
        AlignedLatencyComparison {
            direct,
            gnm,
            gnm_additional_end_to_end_ms: gnm.end_to_end_ms - direct.end_to_end_ms,
        }
    }
}

/// Latency breakdown for one backend output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackendLatencyMetrics {
    /// Shared capture-to-inference latency when inference completion is available.
    pub inference_ms: Option<f64>,
    /// Inference/capture to GNM fit completion when instrumented.
    pub fit_after_inference_ms: Option<f64>,
    /// Fit/inference to decoder completion when instrumented.
    pub decoder_after_fit_ms: Option<f64>,
    /// Final instrumented stage to output publish.
    pub publish_after_last_stage_ms: f64,
    /// Capture-to-output publish latency.
    pub end_to_end_ms: f64,
}

/// Computes a latency breakdown from validated timestamps.
pub fn backend_latency_metrics(timing: BackendOutputTiming) -> BackendLatencyMetrics {
    let source = timing.source();
    let inference = source.inference_complete_micros();
    let inference_or_capture = inference.unwrap_or(source.capture_micros());
    let fit = timing.fit_complete_micros();
    let fit_or_inference = fit.unwrap_or(inference_or_capture);
    let decoder = timing.decoder_complete_micros();
    let last_stage = decoder.unwrap_or(fit_or_inference);

    BackendLatencyMetrics {
        inference_ms: inference.map(|time| duration_ms(source.capture_micros(), time)),
        fit_after_inference_ms: fit.map(|time| duration_ms(inference_or_capture, time)),
        decoder_after_fit_ms: decoder.map(|time| duration_ms(fit_or_inference, time)),
        publish_after_last_stage_ms: duration_ms(last_stage, timing.publish_micros()),
        end_to_end_ms: duration_ms(source.capture_micros(), timing.publish_micros()),
    }
}

/// Latency comparison for an exact source-aligned A/B pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignedLatencyComparison {
    /// Direct latency breakdown.
    pub direct: BackendLatencyMetrics,
    /// GNM latency breakdown.
    pub gnm: BackendLatencyMetrics,
    /// Signed extra GNM capture-to-publish latency versus Direct.
    pub gnm_additional_end_to_end_ms: f64,
}

/// Reason that GNM cannot currently own avatar output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmUnavailableReason {
    /// Required GNM model is missing or invalid.
    ModelInvalid,
    /// Dense mapping is missing, incompatible, or invalid.
    MappingInvalid,
    /// Neutral calibration is unavailable or stale.
    CalibrationUnavailable,
    /// GNM-to-output projector/decoder is unavailable or invalid.
    DecoderUnavailable,
    /// Latest GNM state is too old for safe authority transfer.
    StaleOutput,
    /// GNM state/output became non-finite.
    NonFiniteState,
    /// Sustained latency or queue backlog exceeded the live budget.
    SustainedLatencyBacklog,
}

/// Transient GNM problem that should not force a one-frame backend flip.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmTransientIssue {
    /// One residual/outlier spike.
    ResidualSpike,
    /// One bounded solver failure.
    SolverFailure,
    /// One latency spike that has not become sustained backlog.
    LatencySpike,
}

/// Current GNM runtime health consumed by backend arbitration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmRuntimeHealth {
    /// GNM output is current and eligible for avatar authority.
    Ready,
    /// GNM is not eligible and should immediately fall back/stay Direct.
    Unavailable(GnmUnavailableReason),
    /// A transient issue occurred and hysteresis should apply.
    Transient(GnmTransientIssue),
}

/// Reason Direct currently owns output despite GNM mode being requested.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmFallbackReason {
    /// A hard/not-ready runtime condition prevents GNM authority.
    Unavailable(GnmUnavailableReason),
    /// A transient issue repeated enough times to cross the configured threshold.
    RepeatedTransient(GnmTransientIssue),
    /// GNM was requested while the first observed health sample was transient.
    InitialTransient(GnmTransientIssue),
}

/// Hysteresis parameters for backend authority changes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendSelectionConfig {
    transient_failures_before_fallback: u32,
    ready_frames_before_recover: u32,
}

impl BackendSelectionConfig {
    /// Creates non-zero fallback and recovery thresholds.
    pub fn new(
        transient_failures_before_fallback: u32,
        ready_frames_before_recover: u32,
    ) -> Result<Self, AbBackendError> {
        if transient_failures_before_fallback == 0 {
            return Err(AbBackendError::InvalidSelectionConfig(
                "transient_failures_before_fallback must be positive",
            ));
        }
        if ready_frames_before_recover == 0 {
            return Err(AbBackendError::InvalidSelectionConfig(
                "ready_frames_before_recover must be positive",
            ));
        }
        Ok(Self {
            transient_failures_before_fallback,
            ready_frames_before_recover,
        })
    }

    /// Returns the consecutive transient-failure threshold.
    pub fn transient_failures_before_fallback(self) -> u32 {
        self.transient_failures_before_fallback
    }

    /// Returns the consecutive-ready threshold after a fallback.
    pub fn ready_frames_before_recover(self) -> u32 {
        self.ready_frames_before_recover
    }
}

/// Explicit backend arbitration state carried between frames.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendSelectionState {
    /// Requested runtime mode.
    pub requested_mode: FaceTrackingMode,
    /// Backend with sole avatar-output authority for this frame.
    pub avatar_backend: FaceTrackingBackend,
    /// Whether the GNM path should be evaluated for output or shadow metrics.
    pub evaluate_gnm: bool,
    /// Current reason Direct is acting as GNM fallback, if any.
    pub fallback_reason: Option<GnmFallbackReason>,
    /// Consecutive transient failures while GNM had/was seeking authority.
    pub consecutive_transient_failures: u32,
    /// Consecutive ready frames observed while recovering from fallback.
    pub consecutive_ready_frames: u32,
}

/// One pure arbitration decision plus the state to carry forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendSelectionDecision {
    /// New explicit arbitration state.
    pub state: BackendSelectionState,
    /// Whether avatar-output authority changed this frame.
    pub authority_changed: bool,
    /// Whether callers must explicitly clear/coalesce previous detailed output.
    pub clear_previous_output: bool,
}

/// Advances backend selection with safe fallback and bounded hysteresis.
pub fn advance_backend_selection(
    previous: Option<BackendSelectionState>,
    requested_mode: FaceTrackingMode,
    gnm_health: GnmRuntimeHealth,
    config: BackendSelectionConfig,
) -> BackendSelectionDecision {
    let state = match requested_mode {
        FaceTrackingMode::DirectMediaPipe => direct_state(requested_mode, false, None),
        FaceTrackingMode::GnmTemporalShadow => {
            let diagnostic_reason = health_as_shadow_blocker(gnm_health);
            BackendSelectionState {
                requested_mode,
                avatar_backend: FaceTrackingBackend::DirectMediaPipe,
                evaluate_gnm: true,
                fallback_reason: diagnostic_reason,
                consecutive_transient_failures: 0,
                consecutive_ready_frames: 0,
            }
        }
        FaceTrackingMode::GnmTemporal => advance_requested_gnm(previous, gnm_health, config),
    };

    let authority_changed = previous
        .map(|previous| previous.avatar_backend != state.avatar_backend)
        .unwrap_or(false);
    BackendSelectionDecision {
        state,
        authority_changed,
        clear_previous_output: authority_changed,
    }
}

fn advance_requested_gnm(
    previous: Option<BackendSelectionState>,
    health: GnmRuntimeHealth,
    config: BackendSelectionConfig,
) -> BackendSelectionState {
    match health {
        GnmRuntimeHealth::Unavailable(reason) => BackendSelectionState {
            requested_mode: FaceTrackingMode::GnmTemporal,
            avatar_backend: FaceTrackingBackend::DirectMediaPipe,
            evaluate_gnm: true,
            fallback_reason: Some(GnmFallbackReason::Unavailable(reason)),
            consecutive_transient_failures: 0,
            consecutive_ready_frames: 0,
        },
        GnmRuntimeHealth::Transient(issue) => {
            let continuing_gnm = previous.is_some_and(|state| {
                state.requested_mode == FaceTrackingMode::GnmTemporal
                    && state.avatar_backend == FaceTrackingBackend::GnmTemporal
            });
            if continuing_gnm {
                let failures = previous
                    .map(|state| state.consecutive_transient_failures)
                    .unwrap_or(0)
                    .saturating_add(1);
                if failures < config.transient_failures_before_fallback {
                    BackendSelectionState {
                        requested_mode: FaceTrackingMode::GnmTemporal,
                        avatar_backend: FaceTrackingBackend::GnmTemporal,
                        evaluate_gnm: true,
                        fallback_reason: None,
                        consecutive_transient_failures: failures,
                        consecutive_ready_frames: 0,
                    }
                } else {
                    BackendSelectionState {
                        requested_mode: FaceTrackingMode::GnmTemporal,
                        avatar_backend: FaceTrackingBackend::DirectMediaPipe,
                        evaluate_gnm: true,
                        fallback_reason: Some(GnmFallbackReason::RepeatedTransient(issue)),
                        consecutive_transient_failures: failures,
                        consecutive_ready_frames: 0,
                    }
                }
            } else {
                BackendSelectionState {
                    requested_mode: FaceTrackingMode::GnmTemporal,
                    avatar_backend: FaceTrackingBackend::DirectMediaPipe,
                    evaluate_gnm: true,
                    fallback_reason: Some(GnmFallbackReason::InitialTransient(issue)),
                    consecutive_transient_failures: 1,
                    consecutive_ready_frames: 0,
                }
            }
        }
        GnmRuntimeHealth::Ready => ready_gnm_state(previous, config),
    }
}

fn ready_gnm_state(
    previous: Option<BackendSelectionState>,
    config: BackendSelectionConfig,
) -> BackendSelectionState {
    let recovering_from_fallback = previous.is_some_and(|state| {
        state.requested_mode == FaceTrackingMode::GnmTemporal
            && state.avatar_backend == FaceTrackingBackend::DirectMediaPipe
            && state.fallback_reason.is_some()
    });
    if recovering_from_fallback {
        let ready_frames = previous
            .map(|state| state.consecutive_ready_frames)
            .unwrap_or(0)
            .saturating_add(1);
        if ready_frames < config.ready_frames_before_recover {
            return BackendSelectionState {
                requested_mode: FaceTrackingMode::GnmTemporal,
                avatar_backend: FaceTrackingBackend::DirectMediaPipe,
                evaluate_gnm: true,
                fallback_reason: previous.and_then(|state| state.fallback_reason),
                consecutive_transient_failures: 0,
                consecutive_ready_frames: ready_frames,
            };
        }
    }

    BackendSelectionState {
        requested_mode: FaceTrackingMode::GnmTemporal,
        avatar_backend: FaceTrackingBackend::GnmTemporal,
        evaluate_gnm: true,
        fallback_reason: None,
        consecutive_transient_failures: 0,
        consecutive_ready_frames: 0,
    }
}

fn direct_state(
    requested_mode: FaceTrackingMode,
    evaluate_gnm: bool,
    fallback_reason: Option<GnmFallbackReason>,
) -> BackendSelectionState {
    BackendSelectionState {
        requested_mode,
        avatar_backend: FaceTrackingBackend::DirectMediaPipe,
        evaluate_gnm,
        fallback_reason,
        consecutive_transient_failures: 0,
        consecutive_ready_frames: 0,
    }
}

fn health_as_shadow_blocker(health: GnmRuntimeHealth) -> Option<GnmFallbackReason> {
    match health {
        GnmRuntimeHealth::Ready => None,
        GnmRuntimeHealth::Unavailable(reason) => Some(GnmFallbackReason::Unavailable(reason)),
        GnmRuntimeHealth::Transient(issue) => Some(GnmFallbackReason::InitialTransient(issue)),
    }
}

fn duration_ms(start_micros: u64, end_micros: u64) -> f64 {
    (end_micros - start_micros) as f64 / 1_000.0
}

/// Typed validation failure for A/B source alignment or backend configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AbBackendError {
    /// Timestamp ordering is invalid.
    InvalidTiming {
        /// Invalid timestamp/stage field.
        field: &'static str,
        /// Validation reason.
        reason: String,
    },
    /// An aligned A/B role carried the wrong backend tag.
    WrongBackendRole {
        /// Expected logical role (`direct` or `gnm`).
        role: &'static str,
        /// Actual backend tag.
        actual: FaceTrackingBackend,
    },
    /// Direct and GNM outputs did not originate from the exact same source stamp.
    SourceMismatch {
        /// Direct source stamp.
        direct: SourceFrameStamp,
        /// GNM source stamp.
        gnm: SourceFrameStamp,
    },
    /// Backend hysteresis configuration is invalid.
    InvalidSelectionConfig(&'static str),
}

impl std::fmt::Display for AbBackendError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTiming { field, reason } => {
                write!(formatter, "invalid A/B timing {field}: {reason}")
            }
            Self::WrongBackendRole { role, actual } => {
                write!(formatter, "A/B {role} role has backend {actual:?}")
            }
            Self::SourceMismatch { direct, gnm } => write!(
                formatter,
                "A/B source mismatch: Direct seq {} capture {}, GNM seq {} capture {}",
                direct.source_seq(),
                direct.capture_micros(),
                gnm.source_seq(),
                gnm.capture_micros()
            ),
            Self::InvalidSelectionConfig(reason) => {
                write!(formatter, "invalid backend selection config: {reason}")
            }
        }
    }
}

impl std::error::Error for AbBackendError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(seq: u64) -> SourceFrameStamp {
        SourceFrameStamp::new(seq, 1_000_000, Some(1_005_000)).unwrap()
    }

    fn direct_timing(seq: u64, publish_micros: u64) -> BackendOutputTiming {
        BackendOutputTiming::new(
            source(seq),
            FaceTrackingBackend::DirectMediaPipe,
            None,
            None,
            publish_micros,
        )
        .unwrap()
    }

    fn gnm_timing(seq: u64, publish_micros: u64) -> BackendOutputTiming {
        BackendOutputTiming::new(
            source(seq),
            FaceTrackingBackend::GnmTemporal,
            Some(1_012_000),
            Some(1_014_000),
            publish_micros,
        )
        .unwrap()
    }

    fn config() -> BackendSelectionConfig {
        BackendSelectionConfig::new(3, 2).unwrap()
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn exact_source_stamp_is_required_even_when_publish_times_are_near() {
        let direct = StampedBackendOutput::new(direct_timing(10, 1_016_000), [0.1_f32; 2]);
        let gnm = StampedBackendOutput::new(gnm_timing(11, 1_017_000), [0.2_f32; 2]);
        assert!(matches!(
            AlignedBackendOutputs::new(direct, gnm),
            Err(AbBackendError::SourceMismatch { .. })
        ));
    }

    #[test]
    fn aligned_outputs_share_one_inference_stamp_and_report_extra_latency() {
        let direct = StampedBackendOutput::new(direct_timing(10, 1_010_000), [0.1_f32; 2]);
        let gnm = StampedBackendOutput::new(gnm_timing(10, 1_016_000), [0.2_f32; 2]);
        let pair = AlignedBackendOutputs::new(direct, gnm).unwrap();
        let comparison = pair.latency_comparison();
        close(comparison.direct.inference_ms.unwrap(), 5.0);
        close(comparison.direct.end_to_end_ms, 10.0);
        close(comparison.gnm.fit_after_inference_ms.unwrap(), 7.0);
        close(comparison.gnm.decoder_after_fit_ms.unwrap(), 2.0);
        close(comparison.gnm.end_to_end_ms, 16.0);
        close(comparison.gnm_additional_end_to_end_ms, 6.0);
    }

    #[test]
    fn invalid_stage_order_is_rejected() {
        assert!(
            BackendOutputTiming::new(
                source(1),
                FaceTrackingBackend::GnmTemporal,
                Some(1_004_000),
                None,
                1_010_000,
            )
            .is_err()
        );
        assert!(
            BackendOutputTiming::new(
                source(1),
                FaceTrackingBackend::GnmTemporal,
                Some(1_012_000),
                Some(1_011_000),
                1_020_000,
            )
            .is_err()
        );
    }

    #[test]
    fn shadow_mode_never_changes_avatar_authority() {
        let decision = advance_backend_selection(
            None,
            FaceTrackingMode::GnmTemporalShadow,
            GnmRuntimeHealth::Ready,
            config(),
        );
        assert_eq!(
            decision.state.avatar_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        assert!(decision.state.evaluate_gnm);
        assert!(!decision.authority_changed);
    }

    #[test]
    fn gnm_not_ready_falls_back_to_direct_immediately() {
        let decision = advance_backend_selection(
            None,
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Unavailable(GnmUnavailableReason::CalibrationUnavailable),
            config(),
        );
        assert_eq!(
            decision.state.avatar_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        assert_eq!(
            decision.state.fallback_reason,
            Some(GnmFallbackReason::Unavailable(
                GnmUnavailableReason::CalibrationUnavailable
            ))
        );
    }

    #[test]
    fn one_transient_does_not_thrash_an_active_gnm_backend() {
        let ready = advance_backend_selection(
            None,
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Ready,
            config(),
        )
        .state;
        let one_spike = advance_backend_selection(
            Some(ready),
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Transient(GnmTransientIssue::ResidualSpike),
            config(),
        );
        assert_eq!(
            one_spike.state.avatar_backend,
            FaceTrackingBackend::GnmTemporal
        );
        assert!(!one_spike.authority_changed);
    }

    #[test]
    fn repeated_transients_fallback_then_require_ready_hysteresis_to_recover() {
        let mut state = advance_backend_selection(
            None,
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Ready,
            config(),
        )
        .state;
        for _ in 0..2 {
            state = advance_backend_selection(
                Some(state),
                FaceTrackingMode::GnmTemporal,
                GnmRuntimeHealth::Transient(GnmTransientIssue::SolverFailure),
                config(),
            )
            .state;
            assert_eq!(state.avatar_backend, FaceTrackingBackend::GnmTemporal);
        }
        let fallback = advance_backend_selection(
            Some(state),
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Transient(GnmTransientIssue::SolverFailure),
            config(),
        );
        assert_eq!(
            fallback.state.avatar_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        assert!(fallback.authority_changed);
        assert!(fallback.clear_previous_output);

        let first_ready = advance_backend_selection(
            Some(fallback.state),
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Ready,
            config(),
        );
        assert_eq!(
            first_ready.state.avatar_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        let second_ready = advance_backend_selection(
            Some(first_ready.state),
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Ready,
            config(),
        );
        assert_eq!(
            second_ready.state.avatar_backend,
            FaceTrackingBackend::GnmTemporal
        );
        assert!(second_ready.authority_changed);
        assert!(second_ready.clear_previous_output);
    }

    #[test]
    fn hard_failure_falls_back_without_waiting_for_transient_threshold() {
        let ready = advance_backend_selection(
            None,
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Ready,
            config(),
        )
        .state;
        let failed = advance_backend_selection(
            Some(ready),
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Unavailable(GnmUnavailableReason::NonFiniteState),
            config(),
        );
        assert_eq!(
            failed.state.avatar_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        assert!(failed.authority_changed);
        assert!(failed.clear_previous_output);
    }

    #[test]
    fn explicit_gnm_to_direct_switch_is_immediate_and_requests_output_clear() {
        let ready = advance_backend_selection(
            None,
            FaceTrackingMode::GnmTemporal,
            GnmRuntimeHealth::Ready,
            config(),
        )
        .state;
        let direct = advance_backend_selection(
            Some(ready),
            FaceTrackingMode::DirectMediaPipe,
            GnmRuntimeHealth::Ready,
            config(),
        );
        assert_eq!(
            direct.state.avatar_backend,
            FaceTrackingBackend::DirectMediaPipe
        );
        assert!(!direct.state.evaluate_gnm);
        assert!(direct.authority_changed);
        assert!(direct.clear_previous_output);
    }

    #[test]
    fn direct_timing_cannot_claim_gnm_fit_or_decoder_stages() {
        assert!(
            BackendOutputTiming::new(
                source(1),
                FaceTrackingBackend::DirectMediaPipe,
                Some(1_006_000),
                None,
                1_010_000,
            )
            .is_err()
        );
    }
}
