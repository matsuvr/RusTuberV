//! Engine-independent domain types.

use std::sync::Arc;

/// Monotonic timestamp in nanoseconds from a process-local epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonoTimeNs(pub u64);

/// Monotonically increasing frame sequence number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameSeq(pub u64);

/// Pixel formats supported for decoded camera frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 24-bit RGB, 8 bits per channel.
    Rgb8,
    /// 24-bit BGR, 8 bits per channel.
    Bgr8,
    /// 32-bit RGBA, 8 bits per channel.
    Rgba8,
    /// 8-bit luminance.
    Gray8,
}

/// A decoded camera frame owned by the application.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoFrame {
    /// Sequence number assigned by the capture pipeline.
    pub seq: FrameSeq,
    /// When the frame was captured, in monotonic nanoseconds.
    pub captured_at: MonoTimeNs,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row stride in bytes.
    pub stride_bytes: usize,
    /// Pixel format of `data`.
    pub format: PixelFormat,
    /// Owned frame bytes.
    pub data: Arc<[u8]>,
}

/// 3D facial landmark with normalized image coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Landmark3 {
    /// Normalized image X, left = 0, right = 1.
    pub x: f32,
    /// Normalized image Y, top = 0, bottom = 1.
    pub y: f32,
    /// Model-defined relative depth.
    pub z: f32,
    /// Visibility or presence confidence in `[0, 1]`.
    pub visibility: f32,
}

/// Named blendshape or expression coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedCoefficient {
    /// Expression name, e.g. `blinkLeft` or `aa`.
    pub name: String,
    /// Coefficient in `[0, 1]`.
    pub value: f32,
}

/// Identifies which landmark schema an observation uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LandmarkSchemaId(pub &'static str);

/// Normalized region-of-interest rectangle.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NormalizedRect {
    /// Top-left X in `[0, 1]`.
    pub x: f32,
    /// Top-left Y in `[0, 1]`.
    pub y: f32,
    /// Width in `[0, 1]`.
    pub width: f32,
    /// Height in `[0, 1]`.
    pub height: f32,
    /// Rotation in radians.
    pub rotation_rad: f32,
}

/// Raw output from the inference worker.
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceOutput {
    /// Sequence of the source video frame.
    pub source_seq: FrameSeq,
    /// When the source frame was captured.
    pub captured_at: MonoTimeNs,
    /// When inference started.
    pub inference_started_at: MonoTimeNs,
    /// When inference finished.
    pub inference_finished_at: MonoTimeNs,
    /// Observed face, or `None` if no face was detected.
    pub observation: Option<RawFaceObservation>,
}

pub use crate::observation::RawExpressionObservation;

/// A single face observation produced by inference.
#[derive(Clone, Debug, PartialEq)]
pub struct RawFaceObservation {
    /// Sequence of the source video frame.
    pub source_seq: FrameSeq,
    /// When the source frame was captured.
    pub captured_at: MonoTimeNs,
    /// When inference started.
    pub inference_started_at: MonoTimeNs,
    /// When inference finished.
    pub inference_finished_at: MonoTimeNs,
    /// Overall face confidence in `[0, 1]`.
    pub face_confidence: f32,
    /// Facial landmarks.
    pub landmarks: Vec<Landmark3>,
    /// Optional blendshape coefficients.
    pub blendshapes: Option<Vec<NamedCoefficient>>,
    /// Raw expression coefficients before calibration.
    pub expressions: RawExpressionObservation,
    /// Face region of interest.
    pub roi: NormalizedRect,
    /// Landmark schema used by `landmarks`.
    pub schema: LandmarkSchemaId,
}

/// Semantic head pose in radians.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HeadPose {
    /// Positive when turning right in the unmirrored image.
    pub yaw_rad: f32,
    /// Positive when the chin goes up.
    pub pitch_rad: f32,
    /// Positive when the head tilts clockwise as viewed in the unmirrored image.
    pub roll_rad: f32,
}

/// Availability and reliability of a neutral-relative head translation observation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HeadTranslationState {
    /// A reliable neutral-relative translation estimate is available.
    Tracked,
    /// A usable estimate exists, but reliability is reduced.
    Degraded,
    /// No head translation observation is available.
    #[default]
    Unavailable,
}

/// Engine-neutral, neutral-relative head translation in physical units.
///
/// The translation is expressed in meters in the calibrated neutral face
/// basis aligned with the unmirrored camera/viewer frame:
///
/// * `x_meters`: positive toward the unmirrored image right.
/// * `y_meters`: positive up.
/// * `z_meters`: positive away from the camera (toward the scene).
///
/// Horizontal mirroring is a semantic transform applied at one place:
/// [`HeadTranslationSignal::mirrored`] flips only `x_meters`; `y_meters`,
/// `z_meters`, and all rotation semantics are unchanged. This type carries no
/// dependency on GNM, Bevy, VRM, or MediaPipe runtime types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadTranslationSignal {
    /// Neutral-relative X in meters; positive toward unmirrored image right.
    pub x_meters: f32,
    /// Neutral-relative Y in meters; positive up.
    pub y_meters: f32,
    /// Neutral-relative Z in meters; positive away from the camera.
    pub z_meters: f32,
    /// Whether this translation is tracked, degraded, or unavailable.
    pub state: HeadTranslationState,
}

impl Default for HeadTranslationSignal {
    fn default() -> Self {
        Self::UNAVAILABLE
    }
}

impl HeadTranslationSignal {
    /// Explicit unavailable signal. Unlike a zero tracked movement, it carries no observation.
    pub const UNAVAILABLE: Self = Self {
        x_meters: 0.0,
        y_meters: 0.0,
        z_meters: 0.0,
        state: HeadTranslationState::Unavailable,
    };

    /// Builds a tracked translation, safely degrading non-finite input to unavailable.
    #[must_use]
    pub fn tracked(x_meters: f32, y_meters: f32, z_meters: f32) -> Self {
        Self::available(x_meters, y_meters, z_meters, HeadTranslationState::Tracked)
    }

    /// Builds a degraded translation, safely degrading non-finite input to unavailable.
    #[must_use]
    pub fn degraded(x_meters: f32, y_meters: f32, z_meters: f32) -> Self {
        Self::available(x_meters, y_meters, z_meters, HeadTranslationState::Degraded)
    }

    fn available(x_meters: f32, y_meters: f32, z_meters: f32, state: HeadTranslationState) -> Self {
        if !x_meters.is_finite() || !y_meters.is_finite() || !z_meters.is_finite() {
            return Self::UNAVAILABLE;
        }
        Self {
            x_meters,
            y_meters,
            z_meters,
            state,
        }
    }

    /// Returns whether this signal contains a current or degraded observation.
    #[must_use]
    pub const fn is_available(self) -> bool {
        !matches!(self.state, HeadTranslationState::Unavailable)
    }

    /// Applies the horizontal-mirror semantic transform.
    ///
    /// Only `x_meters` is negated; `y_meters`, `z_meters`, and the state are
    /// preserved. Unavailable signals stay unavailable.
    #[must_use]
    pub const fn mirrored(self) -> Self {
        match self.state {
            HeadTranslationState::Unavailable => Self::UNAVAILABLE,
            _ => Self {
                x_meters: -self.x_meters,
                y_meters: self.y_meters,
                z_meters: self.z_meters,
                state: self.state,
            },
        }
    }

    /// Linearly blends two translation signals by factor `t` in `[0, 1]`.
    ///
    /// When both endpoints carry an observation, translations are lerped and
    /// the blended state is tracked only if both endpoints are tracked;
    /// otherwise it degrades. When either endpoint is unavailable the blend
    /// follows the same availability rules as gaze blending: past `t >= 1`
    /// the result equals `to`, before that it is unavailable unless both
    /// endpoints are available.
    #[must_use]
    pub fn blend(from: Self, to: Self, t: f32) -> Self {
        if !from.is_available() || !to.is_available() {
            if t >= 1.0 {
                return to;
            }
            return Self::UNAVAILABLE;
        }
        let t = t.clamp(0.0, 1.0);
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        let state = if matches!(from.state, HeadTranslationState::Tracked)
            && matches!(to.state, HeadTranslationState::Tracked)
        {
            HeadTranslationState::Tracked
        } else {
            HeadTranslationState::Degraded
        };
        Self {
            x_meters: lerp(from.x_meters, to.x_meters),
            y_meters: lerp(from.y_meters, to.y_meters),
            z_meters: lerp(from.z_meters, to.z_meters),
            state,
        }
    }
}

/// Availability and reliability of an eye-in-head gaze observation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GazeTrackingState {
    /// Both eyes provide a reliable common gaze estimate.
    Tracked,
    /// A usable estimate exists, but binocular agreement or visibility is reduced.
    Degraded,
    /// No new eye-in-head observation is available.
    #[default]
    Unavailable,
}

/// Engine-neutral, normalized eye-in-head gaze signal.
///
/// This is not a physical angle. Model-specific conversion to VRM LookAt
/// degrees belongs to the avatar adapter.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GazeSignal {
    /// Horizontal eye-in-head signal in `[-1, 1]`; image right is positive.
    pub horizontal: f32,
    /// Vertical eye-in-head signal in `[-1, 1]`; up is positive.
    pub vertical: f32,
    /// Reliability in `[0, 1]`.
    pub confidence: f32,
    /// Whether this value is tracked, degraded, or unavailable.
    pub state: GazeTrackingState,
}

impl GazeSignal {
    /// Explicit unavailable signal. Unlike centered tracked gaze, it carries no observation.
    pub const UNAVAILABLE: Self = Self {
        horizontal: 0.0,
        vertical: 0.0,
        confidence: 0.0,
        state: GazeTrackingState::Unavailable,
    };

    /// Builds a bounded tracked signal, safely degrading non-finite input to unavailable.
    #[must_use]
    pub fn tracked(horizontal: f32, vertical: f32, confidence: f32) -> Self {
        Self::available(horizontal, vertical, confidence, GazeTrackingState::Tracked)
    }

    /// Builds a bounded degraded signal, safely degrading non-finite input to unavailable.
    #[must_use]
    pub fn degraded(horizontal: f32, vertical: f32, confidence: f32) -> Self {
        Self::available(
            horizontal,
            vertical,
            confidence,
            GazeTrackingState::Degraded,
        )
    }

    fn available(
        horizontal: f32,
        vertical: f32,
        confidence: f32,
        state: GazeTrackingState,
    ) -> Self {
        if !horizontal.is_finite() || !vertical.is_finite() || !confidence.is_finite() {
            return Self::UNAVAILABLE;
        }
        Self {
            horizontal: horizontal.clamp(-1.0, 1.0),
            vertical: vertical.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            state,
        }
    }

    /// Returns whether this signal contains a current or degraded observation.
    #[must_use]
    pub const fn is_available(self) -> bool {
        !matches!(self.state, GazeTrackingState::Unavailable)
    }
}

/// Expression coefficients applied to the avatar.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExpressionCoefficients {
    /// Left eye blink.
    pub blink_left: f32,
    /// Right eye blink.
    pub blink_right: f32,
    /// `aa` mouth shape.
    pub aa: f32,
    /// `ih` mouth shape.
    pub ih: f32,
    /// `ou` mouth shape.
    pub ou: f32,
    /// `ee` mouth shape.
    pub ee: f32,
    /// `oh` mouth shape.
    pub oh: f32,
    /// Look left expression.
    pub look_left: f32,
    /// Look right expression.
    pub look_right: f32,
    /// Look up expression.
    pub look_up: f32,
    /// Look down expression.
    pub look_down: f32,
    /// Happy expression.
    pub happy: f32,
    /// Angry expression.
    pub angry: f32,
    /// Sad expression.
    pub sad: f32,
    /// Relaxed expression.
    pub relaxed: f32,
    /// Surprised expression.
    pub surprised: f32,
}

/// Tracking state machine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TrackingState {
    /// Pipeline is starting.
    #[default]
    Starting,
    /// Searching for a face.
    Searching,
    /// Face detected but not yet stable.
    Acquiring,
    /// Face is being tracked normally.
    Tracking,
    /// Tracking confidence is degraded.
    Degraded,
    /// Face was lost; holding last pose briefly.
    LostHold,
    /// Returning to neutral after lost hold expires.
    ReturningNeutral,
}

/// Control frame produced by the tracking filter and consumed by the avatar adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct AvatarControlFrame {
    /// Sequence of the source video frame.
    pub source_seq: FrameSeq,
    /// When the source frame was captured.
    pub captured_at: MonoTimeNs,
    /// When this control frame was produced.
    pub produced_at: MonoTimeNs,
    /// Aggregated tracking confidence in `[0, 1]`.
    pub confidence: f32,
    /// Current tracking state.
    pub state: TrackingState,
    /// Head pose relative to calibrated neutral.
    pub head: HeadPose,
    /// Neutral-relative head translation in meters relative to calibrated
    /// neutral, sharing the same source frame as `head`.
    ///
    /// Rotation-only producers may leave this at
    /// [`HeadTranslationSignal::UNAVAILABLE`]; consumers must treat an
    /// unavailable signal differently from zero tracked movement.
    pub head_translation: HeadTranslationSignal,
    /// Explicit normalized eye-in-head gaze signal.
    pub gaze: GazeSignal,
    /// Expression coefficients.
    pub expressions: ExpressionCoefficients,
    /// Optional validated detailed ARKit52 face state.
    ///
    /// The avatar adapter may use this only when the active model reports an
    /// effective Perfect Sync capability. `None` preserves the existing
    /// coarse-expression path.
    pub detailed_face: Option<crate::Arkit52Coefficients>,
}

#[cfg(test)]
mod gaze_contract_tests {
    use super::*;

    #[test]
    fn centered_tracked_gaze_is_distinct_from_unavailable() {
        let centered = GazeSignal::tracked(0.0, 0.0, 1.0);
        assert!(centered.is_available());
        assert_eq!(centered.horizontal, 0.0);
        assert_eq!(centered.vertical, 0.0);
        assert_ne!(centered, GazeSignal::UNAVAILABLE);
    }

    #[test]
    fn gaze_contract_clamps_ranges_and_rejects_non_finite_values() {
        let bounded = GazeSignal::degraded(2.0, -2.0, 4.0);
        assert_eq!(bounded.horizontal, 1.0);
        assert_eq!(bounded.vertical, -1.0);
        assert_eq!(bounded.confidence, 1.0);
        assert_eq!(
            GazeSignal::tracked(f32::NAN, 0.0, 1.0),
            GazeSignal::UNAVAILABLE
        );
    }
}

#[cfg(test)]
mod head_translation_contract_tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn zero_tracked_translation_is_distinct_from_unavailable() {
        let centered = HeadTranslationSignal::tracked(0.0, 0.0, 0.0);
        assert!(centered.is_available());
        assert_ne!(centered, HeadTranslationSignal::UNAVAILABLE);
        assert!(!HeadTranslationSignal::UNAVAILABLE.is_available());
        assert_eq!(
            HeadTranslationSignal::default(),
            HeadTranslationSignal::UNAVAILABLE
        );
    }

    #[test]
    fn translation_contract_rejects_non_finite_values() {
        assert_eq!(
            HeadTranslationSignal::tracked(f32::NAN, 0.0, 0.0),
            HeadTranslationSignal::UNAVAILABLE
        );
        assert_eq!(
            HeadTranslationSignal::tracked(0.0, f32::INFINITY, 0.0),
            HeadTranslationSignal::UNAVAILABLE
        );
        assert_eq!(
            HeadTranslationSignal::degraded(0.0, 0.0, f32::NEG_INFINITY),
            HeadTranslationSignal::UNAVAILABLE
        );
    }

    #[test]
    fn horizontal_mirror_flips_only_x_and_preserves_y_z_state() {
        let tracked = HeadTranslationSignal::tracked(0.05, -0.02, 0.10);
        let mirrored = tracked.mirrored();
        assert_close(mirrored.x_meters, -0.05);
        assert_close(mirrored.y_meters, -0.02);
        assert_close(mirrored.z_meters, 0.10);
        assert_eq!(mirrored.state, HeadTranslationState::Tracked);

        // Mirroring twice restores the original signal.
        assert_eq!(mirrored.mirrored(), tracked);

        // Degraded state survives mirroring.
        let degraded = HeadTranslationSignal::degraded(0.01, 0.02, 0.03).mirrored();
        assert_eq!(degraded.state, HeadTranslationState::Degraded);
        assert_close(degraded.x_meters, -0.01);

        // Unavailable stays unavailable and never manufactures an observation.
        assert_eq!(
            HeadTranslationSignal::UNAVAILABLE.mirrored(),
            HeadTranslationSignal::UNAVAILABLE
        );
    }

    #[test]
    fn blend_lerps_available_endpoints_and_degrades_mixed_states() {
        let a = HeadTranslationSignal::tracked(-0.04, 0.02, 0.0);
        let b = HeadTranslationSignal::tracked(0.06, -0.02, 0.10);
        let mid = HeadTranslationSignal::blend(a, b, 0.5);
        assert_close(mid.x_meters, 0.01);
        assert_close(mid.y_meters, 0.0);
        assert_close(mid.z_meters, 0.05);
        assert_eq!(mid.state, HeadTranslationState::Tracked);

        let degraded_target = HeadTranslationSignal {
            state: HeadTranslationState::Degraded,
            ..b
        };
        let degraded = HeadTranslationSignal::blend(a, degraded_target, 0.5);
        assert_eq!(degraded.state, HeadTranslationState::Degraded);
        assert_close(degraded.x_meters, 0.01);
    }

    #[test]
    fn blend_with_unavailable_follows_gaze_availability_rules() {
        let tracked = HeadTranslationSignal::tracked(0.04, 0.0, 0.0);
        // Before the end of the blend, no observation can be manufactured.
        assert_eq!(
            HeadTranslationSignal::blend(tracked, HeadTranslationSignal::UNAVAILABLE, 0.5),
            HeadTranslationSignal::UNAVAILABLE
        );
        // At t >= 1 the target wins even if it is unavailable.
        assert_eq!(
            HeadTranslationSignal::blend(tracked, HeadTranslationSignal::UNAVAILABLE, 1.0),
            HeadTranslationSignal::UNAVAILABLE
        );
        assert_eq!(
            HeadTranslationSignal::blend(HeadTranslationSignal::UNAVAILABLE, tracked, 1.0),
            tracked
        );
    }
}
