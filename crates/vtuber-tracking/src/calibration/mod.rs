//! Calibration: neutral reference collection and session state.
//!
//! This module is responsible for turning raw face observations into a
//! validated neutral profile, and for tracking the lifecycle of a calibration
//! session. It must not depend on Bevy or camera backends.

/// Immediate and robust auto-neutral selection for the MediaPipe path.
pub mod auto_neutral;
pub mod collector;
pub mod neutral;
pub mod types;

pub use auto_neutral::{
    AUTO_NEUTRAL_MIN_SAMPLES, AUTO_NEUTRAL_WINDOW, AutoNeutralCollector, AutoNeutralError,
    AutoNeutralState, AutoNeutralUpdate, GazeNeutralBaseline,
};
pub use collector::{CalibrationCollector, CollectorMetrics, RejectionReason, SampleDecision};
pub use neutral::{NeutralContext, NeutralReference, NeutralValidationSettings};
pub use types::{CalibrationInput, CalibrationSession, NeutralProfile};
