//! Idle motion contract.
//!
//! The current body-motion architecture keeps authored or animated pose as
//! the idle authority and does not add an independent procedural breathing
//! oscillator. This prevents duplicate writes to hips/body channels and keeps
//! tracking-loss micro-motion as the only additive low-frequency motion layer.
//!
//! The legacy #20 hips-only `sin²` breathing writer is retired. The
//! `hips.translation` channel has no runtime writer in `vtuber-avatar`; the
//! ADR-019 body/root writer owns avatar-root translation rather than the hips
//! bone. Tracking-loss micro-motion publishes through the
//! `BodyTrackingPositionInput` bridge and therefore composes without a second
//! idle writer.

use bevy::prelude::*;

/// Procedural idle amplitude in meters.
///
/// Zero is an architectural invariant: idle preserves authored/animated pose
/// while the tracking and recovery layers own motion.
pub const IDLE_PROCEDURAL_AMPLITUDE_METERS: f32 = 0.0;

/// Typed idle-motion policy attached to the active avatar root.
///
/// The component lets compatibility and trace validation verify the
/// zero-amplitude invariant. There is deliberately no runtime writer and no
/// user-facing toggle: the rest or animated pose remains the idle pose.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct IdleMotionProfile {
    /// Procedural idle amplitude in meters. Must be
    /// [`IDLE_PROCEDURAL_AMPLITUDE_METERS`] (zero).
    pub procedural_amplitude_meters: f32,
}

impl Default for IdleMotionProfile {
    fn default() -> Self {
        Self {
            procedural_amplitude_meters: IDLE_PROCEDURAL_AMPLITUDE_METERS,
        }
    }
}

impl IdleMotionProfile {
    /// Validates the contract invariant.
    ///
    /// # Errors
    ///
    /// Returns [`IdleMotionProfileError::NonZeroProceduralAmplitude`] when
    /// the amplitude is not exactly zero, or
    /// [`IdleMotionProfileError::NonFiniteAmplitude`] for non-finite values.
    pub fn validate(&self) -> Result<(), IdleMotionProfileError> {
        if !self.procedural_amplitude_meters.is_finite() {
            return Err(IdleMotionProfileError::NonFiniteAmplitude);
        }
        if self.procedural_amplitude_meters != IDLE_PROCEDURAL_AMPLITUDE_METERS {
            return Err(IdleMotionProfileError::NonZeroProceduralAmplitude);
        }
        Ok(())
    }
}

/// Errors produced by [`IdleMotionProfile::validate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdleMotionProfileError {
    /// The amplitude is not a finite value.
    NonFiniteAmplitude,
    /// The amplitude is not the zero-amplitude policy.
    NonZeroProceduralAmplitude,
}

impl std::fmt::Display for IdleMotionProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteAmplitude => f.write_str("idle amplitude must be finite"),
            Self::NonZeroProceduralAmplitude => {
                f.write_str("procedural idle amplitude must stay zero")
            }
        }
    }
}

impl std::error::Error for IdleMotionProfileError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_matches_the_zero_amplitude_contract() {
        let profile = IdleMotionProfile::default();
        assert_eq!(
            profile.procedural_amplitude_meters,
            IDLE_PROCEDURAL_AMPLITUDE_METERS
        );
        assert_eq!(profile.procedural_amplitude_meters, 0.0);
        assert_eq!(profile.validate(), Ok(()));
    }

    #[test]
    fn profile_rejects_nonzero_and_nonfinite_amplitudes() {
        let nonzero = IdleMotionProfile {
            procedural_amplitude_meters: 0.01,
        };
        assert_eq!(
            nonzero.validate(),
            Err(IdleMotionProfileError::NonZeroProceduralAmplitude)
        );

        let non_finite = IdleMotionProfile {
            procedural_amplitude_meters: f32::NAN,
        };
        assert_eq!(
            non_finite.validate(),
            Err(IdleMotionProfileError::NonFiniteAmplitude)
        );
    }
}
