//! Region projectors from the engine-neutral [`vtuber_gnm::GnmFacialFeatures`]
//! snapshot onto canonical ARKit52 blendshape channels (GNM #67.2 and later).
//!
//! Every projector is a pure function of the validated feature snapshot: it
//! never reads MediaPipe blendshape scores, never observes a preview mirror,
//! and never depends on wall-clock time. Each projected channel carries an
//! explicit support classification so downstream consumers can distinguish
//! measured values from absent evidence; unobserved channels are reported as
//! [`ProjectedSupport::Unsupported`] with value `0.0` instead of being
//! fabricated.

use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, Arkit52ValueError, ArkitBlendshape};
use vtuber_gnm::GnmFacialFeatures;

/// Support classification of one projected ARKit channel.
///
/// The classification is deterministic and derived only from the availability
/// of the geometric evidence a channel is computed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectedSupport {
    /// The channel is computed from dedicated, well-conditioned geometry.
    Reliable,
    /// The channel is computed from geometry that is available but known to be
    /// shared with, or confounded by, another facial action.
    Experimental,
    /// The snapshot carries no usable observation for this channel; the value
    /// is always `0.0`.
    Unsupported,
}

/// One projected ARKit channel value plus its support classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedChannel {
    /// Projected coefficient in finite `[0, 1]`; `0.0` when unsupported.
    pub value: f32,
    /// Evidence classification of this value.
    pub support: ProjectedSupport,
}

impl ProjectedChannel {
    /// A `Reliable` channel with the given bounded value.
    #[must_use]
    pub fn reliable(value: f32) -> Self {
        Self {
            value,
            support: ProjectedSupport::Reliable,
        }
    }

    /// An `Experimental` channel with the given bounded value.
    #[must_use]
    pub fn experimental(value: f32) -> Self {
        Self {
            value,
            support: ProjectedSupport::Experimental,
        }
    }

    /// An `Unsupported` channel; always value `0.0`.
    pub const UNSUPPORTED: Self = Self {
        value: 0.0,
        support: ProjectedSupport::Unsupported,
    };
}

/// Aperture-reduction fraction (relative to the calibrated neutral aperture)
/// that corresponds to a full blink. Closure beyond this fraction saturates
/// `EyeBlink` at `1.0`.
const BLINK_FULL_CLOSURE_FRACTION: f32 = 0.85;

/// Aperture-reduction fraction that saturates the `EyeSquint` estimate. Squint
/// shares the aperture observation with blink, so its support is
/// [`ProjectedSupport::Experimental`].
const SQUINT_FULL_CLOSURE_FRACTION: f32 = 0.5;

/// Aperture-widening fraction (relative to the calibrated neutral aperture)
/// that saturates `EyeWide` at `1.0`.
const WIDE_FULL_WIDENING_FRACTION: f32 = 0.6;

/// Iris vertical displacement, in inter-ocular-scale units, that saturates the
/// `EyeLookUp`/`EyeLookDown` pair.
const GAZE_VERTICAL_SATURATION: f32 = 0.1;

/// Iris horizontal displacement, in inter-ocular-scale units, that saturates
/// the `EyeLookIn`/`EyeLookOut` pair.
const GAZE_HORIZONTAL_SATURATION: f32 = 0.08;

/// Eye and gaze ARKit channels projected from one [`GnmFacialFeatures`] snapshot.
///
/// Field names follow the canonical ARKit semantics; `left`/`right` are the
/// subject's anatomical sides, never image-space orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeGazeProjection {
    /// Left eye blink.
    pub eye_blink_left: ProjectedChannel,
    /// Right eye blink.
    pub eye_blink_right: ProjectedChannel,
    /// Left eye widening.
    pub eye_wide_left: ProjectedChannel,
    /// Right eye widening.
    pub eye_wide_right: ProjectedChannel,
    /// Left eye squint.
    pub eye_squint_left: ProjectedChannel,
    /// Right eye squint.
    pub eye_squint_right: ProjectedChannel,
    /// Left eye looks toward the nose.
    pub eye_look_in_left: ProjectedChannel,
    /// Right eye looks toward the nose.
    pub eye_look_in_right: ProjectedChannel,
    /// Left eye looks away from the nose.
    pub eye_look_out_left: ProjectedChannel,
    /// Right eye looks away from the nose.
    pub eye_look_out_right: ProjectedChannel,
    /// Left eye looks up.
    pub eye_look_up_left: ProjectedChannel,
    /// Right eye looks up.
    pub eye_look_up_right: ProjectedChannel,
    /// Left eye looks down.
    pub eye_look_down_left: ProjectedChannel,
    /// Right eye looks down.
    pub eye_look_down_right: ProjectedChannel,
}

/// Divides two values and clamps the quotient into `[0, 1]`.
///
/// Returns `None` when either input is non-finite or the denominator is not
/// strictly positive; callers treat `None` as missing evidence.
fn bounded_ratio(numerator: f32, denominator: f32) -> Option<f32> {
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    Some((numerator / denominator).clamp(0.0, 1.0))
}

/// Projects blink/wide/squint for one eye from its aperture feature.
///
/// Closure and widening are fractions of the calibrated neutral aperture, so
/// they are invariant under rigid head transforms and uniform scaling.
/// Negative closure (wider than neutral) contributes nothing to blink or
/// squint; negative widening (more closed than neutral) contributes nothing to
/// wide.
fn aperture_channels(
    eye: &vtuber_gnm::EyeApertureFeature,
) -> (ProjectedChannel, ProjectedChannel, ProjectedChannel) {
    let neutral = eye.neutral_aperture;
    let current = eye.current_aperture;
    if !neutral.is_finite() || !current.is_finite() || neutral <= 0.0 {
        return (
            ProjectedChannel::UNSUPPORTED,
            ProjectedChannel::UNSUPPORTED,
            ProjectedChannel::UNSUPPORTED,
        );
    }
    let closure = (neutral - current) / neutral;
    let widening = -closure;
    let blink = bounded_ratio(closure, BLINK_FULL_CLOSURE_FRACTION);
    let squint = bounded_ratio(closure, SQUINT_FULL_CLOSURE_FRACTION);
    let wide = bounded_ratio(widening, WIDE_FULL_WIDENING_FRACTION);
    (
        blink.map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable),
        wide.map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable),
        squint.map_or(
            ProjectedChannel::UNSUPPORTED,
            ProjectedChannel::experimental,
        ),
    )
}

/// The four gaze channels for one anatomical side.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SideGazeChannels {
    look_in: ProjectedChannel,
    look_out: ProjectedChannel,
    look_up: ProjectedChannel,
    look_down: ProjectedChannel,
}

/// Projects the gaze channels for one anatomical side from its iris feature.
///
/// `vertical_delta` is positive for gaze up; `horizontal_delta` is positive
/// toward the outer corner, which is `EyeLookOut` for both anatomical sides
/// because the feature is keyed on anatomical topology rather than image
/// orientation. A missing or non-finite iris feature yields four
/// [`ProjectedSupport::Unsupported`] channels.
fn side_gaze_channels(iris: Option<&vtuber_gnm::IrisSideAuxFeature>) -> SideGazeChannels {
    let Some(iris) = iris else {
        return SideGazeChannels {
            look_in: ProjectedChannel::UNSUPPORTED,
            look_out: ProjectedChannel::UNSUPPORTED,
            look_up: ProjectedChannel::UNSUPPORTED,
            look_down: ProjectedChannel::UNSUPPORTED,
        };
    };
    let up = match iris.vertical_delta {
        Some(vertical) => bounded_ratio(vertical, GAZE_VERTICAL_SATURATION),
        None => None,
    };
    let down = match iris.vertical_delta {
        Some(vertical) => bounded_ratio(-vertical, GAZE_VERTICAL_SATURATION),
        None => None,
    };
    let out = match iris.horizontal_delta {
        Some(horizontal) => bounded_ratio(horizontal, GAZE_HORIZONTAL_SATURATION),
        None => None,
    };
    let look_in = match iris.horizontal_delta {
        Some(horizontal) => bounded_ratio(-horizontal, GAZE_HORIZONTAL_SATURATION),
        None => None,
    };
    SideGazeChannels {
        look_in: look_in.map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable),
        look_out: out.map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable),
        look_up: up.map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable),
        look_down: down.map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable),
    }
}

/// Projects the eye-aperture and iris-gaze ARKit channels from one facial
/// feature snapshot.
///
/// The function is pure and total: it reads only the snapshot, reports
/// unsupported channels as value `0.0`, and clamps every produced value into
/// finite `[0, 1]`. Gaze is generated exclusively from the iris pairwise-
/// distance features; MediaPipe blendshape scores are never consulted.
#[must_use]
pub fn project_eye_gaze_channels(features: &vtuber_gnm::GnmFacialFeatures) -> EyeGazeProjection {
    let (blink_right, wide_right, squint_right) = aperture_channels(&features.eyes.right);
    let (blink_left, wide_left, squint_left) = aperture_channels(&features.eyes.left);

    let right_gaze = side_gaze_channels(features.irises.right.as_ref());
    let left_gaze = side_gaze_channels(features.irises.left.as_ref());

    EyeGazeProjection {
        eye_blink_left: blink_left,
        eye_blink_right: blink_right,
        eye_wide_left: wide_left,
        eye_wide_right: wide_right,
        eye_squint_left: squint_left,
        eye_squint_right: squint_right,
        eye_look_in_left: left_gaze.look_in,
        eye_look_in_right: right_gaze.look_in,
        eye_look_out_left: left_gaze.look_out,
        eye_look_out_right: right_gaze.look_out,
        eye_look_up_left: left_gaze.look_up,
        eye_look_up_right: right_gaze.look_up,
        eye_look_down_left: left_gaze.look_down,
        eye_look_down_right: right_gaze.look_down,
    }
}

/// Jaw-open lip-aperture delta, in mouth-width-scale units, that saturates
/// `JawOpen`.
const JAW_OPEN_SATURATION: f32 = 0.35;

/// Chin-to-nose-tip delta magnitude, in mouth-width-scale units, that
/// saturates `JawForward`.
const JAW_FORWARD_SATURATION: f32 = 0.15;

/// Chin lateral-shift delta, in mouth-width-scale units, that saturates the
/// `JawLeft`/`JawRight` pair.
const JAW_LATERAL_SATURATION: f32 = 0.15;

/// Lip-compression delta magnitude (negative jaw-open direction), in
/// mouth-width-scale units, that saturates `MouthClose`.
const MOUTH_CLOSE_SATURATION: f32 = 0.15;

/// Mouth-corner narrowing delta, in mouth-width-scale units, that saturates
/// the shared funnel/pucker constriction estimate.
const LIP_CONSTRICTION_SATURATION: f32 = 0.25;

/// Jaw and core-mouth ARKit channels projected from one [`GnmFacialFeatures`]
/// snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JawCoreMouthProjection {
    /// Jaw moves forward.
    pub jaw_forward: ProjectedChannel,
    /// Jaw moves toward the subject's anatomical left.
    pub jaw_left: ProjectedChannel,
    /// Jaw opens.
    pub jaw_open: ProjectedChannel,
    /// Jaw moves toward the subject's anatomical right.
    pub jaw_right: ProjectedChannel,
    /// Lips press together more than the calibrated neutral.
    pub mouth_close: ProjectedChannel,
    /// Lips round into a funnel shape.
    pub mouth_funnel: ProjectedChannel,
    /// Lips pucker forward.
    pub mouth_pucker: ProjectedChannel,
    /// Lips shift toward the subject's anatomical left without jaw motion.
    pub mouth_left: ProjectedChannel,
    /// Lips shift toward the subject's anatomical right without jaw motion.
    pub mouth_right: ProjectedChannel,
}

/// Projects the jaw and core-mouth ARKit channels from one facial feature
/// snapshot.
///
/// Sign conventions follow [`vtuber_gnm::MouthAuxFeatures`]:
///
/// - `jaw_open > 0` opens `JawOpen`; `jaw_open < 0` is lip compression and
///   drives a bounded `MouthClose` instead. The two channels are mutually
///   exclusive by construction, which resolves the obvious geometric
///   contradiction between them without letting either exceed `[0, 1]`.
/// - `jaw_forward < 0` (chin approaching the nose tip) opens `JawForward`.
/// - `jaw_lateral > 0` (chin toward the anatomical left) opens `JawLeft`
///   and leaves `JawRight` at zero; negative does the opposite.
/// - Funnel/pucker share the single corner-narrowing observation
///   (`width_delta < 0`) because the snapshot carries no protrusion or
///   rounding evidence to separate them; both are therefore
///   [`ProjectedSupport::Experimental`] with identical values.
/// - `MouthLeft`/`MouthRight` describe lip shift decoupled from the jaw; the
///   snapshot has no such observation, so they are permanently
///   [`ProjectedSupport::Unsupported`].
///
/// Missing features fail closed to [`ProjectedSupport::Unsupported`] with
/// value `0.0`; every produced value is finite in `[0, 1]`.
#[must_use]
pub fn project_jaw_core_mouth_channels(features: &GnmFacialFeatures) -> JawCoreMouthProjection {
    let mouth = features.mouth_jaw;

    let jaw_open = match mouth.jaw_open {
        Some(delta) if delta >= 0.0 => bounded_ratio(delta, JAW_OPEN_SATURATION)
            .map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable),
        _ => ProjectedChannel::UNSUPPORTED,
    };
    let mouth_close = match mouth.jaw_open {
        Some(delta) if delta < 0.0 => bounded_ratio(-delta, MOUTH_CLOSE_SATURATION).map_or(
            ProjectedChannel::UNSUPPORTED,
            ProjectedChannel::experimental,
        ),
        _ => ProjectedChannel::UNSUPPORTED,
    };
    let jaw_forward = mouth
        .jaw_forward
        .and_then(|delta| bounded_ratio(-delta, JAW_FORWARD_SATURATION))
        .map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable);
    let (jaw_left, jaw_right) = match mouth.jaw_lateral {
        Some(delta) => (
            bounded_ratio(delta, JAW_LATERAL_SATURATION),
            bounded_ratio(-delta, JAW_LATERAL_SATURATION),
        ),
        None => (None, None),
    };
    let constriction = mouth
        .width_delta
        .and_then(|delta| bounded_ratio(-delta, LIP_CONSTRICTION_SATURATION));
    let constriction_channel = constriction.map_or(ProjectedChannel::UNSUPPORTED, |value| {
        ProjectedChannel::experimental(value)
    });

    JawCoreMouthProjection {
        jaw_forward,
        jaw_left: jaw_left.map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable),
        jaw_open,
        jaw_right: jaw_right.map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable),
        mouth_close,
        mouth_funnel: constriction_channel,
        mouth_pucker: constriction_channel,
        // No dedicated lip-shift observation exists in the snapshot; these
        // channels stay unsupported rather than borrowing jaw-lateral motion.
        mouth_left: ProjectedChannel::UNSUPPORTED,
        mouth_right: ProjectedChannel::UNSUPPORTED,
    }
}

/// Brow raise/lower delta magnitude, in inter-ocular-scale units, that
/// saturates each brow channel.
const BROW_SATURATION: f32 = 0.08;

/// Brow and upper-face ARKit channels projected from one [`GnmFacialFeatures`]
/// snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrowCheekNoseProjection {
    /// Left brow moves down.
    pub brow_down_left: ProjectedChannel,
    /// Right brow moves down.
    pub brow_down_right: ProjectedChannel,
    /// Inner brows move up.
    pub brow_inner_up: ProjectedChannel,
    /// Left outer brow moves up.
    pub brow_outer_up_left: ProjectedChannel,
    /// Right outer brow moves up.
    pub brow_outer_up_right: ProjectedChannel,
    /// Cheeks puff.
    pub cheek_puff: ProjectedChannel,
    /// Left cheek squints.
    pub cheek_squint_left: ProjectedChannel,
    /// Right cheek squints.
    pub cheek_squint_right: ProjectedChannel,
    /// Left nostril sneers.
    pub nose_sneer_left: ProjectedChannel,
    /// Right nostril sneers.
    pub nose_sneer_right: ProjectedChannel,
}

fn mean_of_present(values: impl IntoIterator<Item = Option<f32>>) -> Option<f32> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        match value {
            Some(value) if value.is_finite() => {
                sum += value;
                count += 1;
            }
            // A non-finite measurement is not evidence.
            Some(_) => return None,
            None => {}
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f32)
    }
}

/// Contour-widening delta magnitude, in mouth-width-scale units, that
/// saturates the `CheekPuff` estimate.
const CHEEK_SATURATION: f32 = 0.08;

/// Provisional `CheekPuff` gate thresholds, in mouth-width-scale units.
///
/// The three-condition AND reproduces the lecture's geometric rule
/// (lower-jaw contour outward + mouth not widened + chin tip not out) in
/// neutral-relative, rigid-invariant deltas. Values are provisional until
/// the hold-based cheek evaluation calibrates the middle of the flat
/// interval; they are deliberately conservative to prefer misses over
/// false positives on talk/smile/purse.
const CHEEK_OUTWARD_MIN: f32 = 0.02;
const CHEEK_MOUTH_WIDTH_MAX: f32 = 0.02;
const CHEEK_JAW_FORWARD_MIN: f32 = -0.02;

/// Projects the brow, cheek, and nose ARKit channels from one facial feature
/// snapshot.
///
/// This function is the single deterministic home of the support
/// classification for the upper-face channels:
///
/// - Brow channels are [`ProjectedSupport::Reliable`] whenever their
///   dedicated neutral-relative brow-to-lid-apex deltas exist:
///   `brow_lower > 0` drives `BrowDownLeft`/`Right`, `outer_rise > 0`
///   drives `BrowOuterUpLeft`/`Right`, and the positive part of the mean
///   `inner_rise` drives the midline `BrowInnerUp` channel.
/// - `CheekPuff` is [`ProjectedSupport::Experimental`] only when all three
///   hold: the lower-cheek contour sits outward versus neutral, the mouth
///   is not widened, and the chin tip is not pushed forward. Any missing
///   evidence fails closed to [`ProjectedSupport::Unsupported`].
/// - `CheekSquintLeft`/`Right` and `NoseSneerLeft`/`Right` have no dedicated
///   observation in the current snapshot, so they stay permanently
///   [`ProjectedSupport::Unsupported`] with value `0.0`.
/// - No projector in this module ever produces a tongue coefficient;
///   `TongueOut` has no dedicated observation and stays fabricated-free.
///
/// Missing or non-finite brow deltas fail closed to
/// [`ProjectedSupport::Unsupported`] with value `0.0`; every produced value
/// is finite in `[0, 1]`.
#[must_use]
pub fn project_brow_cheek_nose_channels(features: &GnmFacialFeatures) -> BrowCheekNoseProjection {
    let brow_channel = |delta: Option<f32>| {
        delta
            .and_then(|delta| bounded_ratio(delta, BROW_SATURATION))
            .map_or(ProjectedChannel::UNSUPPORTED, ProjectedChannel::reliable)
    };
    let inner_up = mean_of_present([
        features.brows.right.inner_rise,
        features.brows.left.inner_rise,
    ])
    .map(|mean| mean.max(0.0));

    BrowCheekNoseProjection {
        brow_down_left: brow_channel(features.brows.left.brow_lower),
        brow_down_right: brow_channel(features.brows.right.brow_lower),
        brow_inner_up: brow_channel(inner_up),
        brow_outer_up_left: brow_channel(features.brows.left.outer_rise),
        brow_outer_up_right: brow_channel(features.brows.right.outer_rise),
        cheek_puff: cheek_puff_channel(features),
        cheek_squint_left: ProjectedChannel::UNSUPPORTED,
        cheek_squint_right: ProjectedChannel::UNSUPPORTED,
        nose_sneer_left: ProjectedChannel::UNSUPPORTED,
        nose_sneer_right: ProjectedChannel::UNSUPPORTED,
    }
}

/// Evaluates the three-condition `CheekPuff` gate.
///
/// All inputs are neutral-relative distance deltas, so a rigid head
/// transform alone cannot fire the channel. A `None` in any condition
/// means the mapping lacks the rows to judge it and fails closed.
fn cheek_puff_channel(features: &GnmFacialFeatures) -> ProjectedChannel {
    let (Some(outward), Some(width), Some(forward)) = (
        features.cheeks.contour_outward,
        features.mouth_jaw.width_delta,
        features.mouth_jaw.jaw_forward,
    ) else {
        return ProjectedChannel::UNSUPPORTED;
    };
    if !(outward.is_finite() && width.is_finite() && forward.is_finite()) {
        return ProjectedChannel::UNSUPPORTED;
    }
    if outward <= CHEEK_OUTWARD_MIN
        || width > CHEEK_MOUTH_WIDTH_MAX
        || forward < CHEEK_JAW_FORWARD_MIN
    {
        return ProjectedChannel::UNSUPPORTED;
    }
    bounded_ratio(outward - CHEEK_OUTWARD_MIN, CHEEK_SATURATION).map_or(
        ProjectedChannel::UNSUPPORTED,
        ProjectedChannel::experimental,
    )
}

/// Corner-lift delta magnitude, in mouth-width-scale units, that saturates
/// the `MouthSmile` pair.
const SMILE_SATURATION: f32 = 0.1;

/// Corner-lower delta magnitude, in mouth-width-scale units, that saturates
/// the `MouthFrown` pair.
const FROWN_SATURATION: f32 = 0.06;

/// Corner-widening delta magnitude, in mouth-width-scale units, that
/// saturates the `MouthStretch` pair.
const STRETCH_SATURATION: f32 = 0.15;

/// Lip-corner ARKit channels projected from one [`GnmFacialFeatures`]
/// snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipCornerProjection {
    /// Left mouth corner smiles.
    pub mouth_smile_left: ProjectedChannel,
    /// Right mouth corner smiles.
    pub mouth_smile_right: ProjectedChannel,
    /// Left mouth corner frowns.
    pub mouth_frown_left: ProjectedChannel,
    /// Right mouth corner frowns.
    pub mouth_frown_right: ProjectedChannel,
    /// Left mouth corner dimples.
    pub mouth_dimple_left: ProjectedChannel,
    /// Right mouth corner dimples.
    pub mouth_dimple_right: ProjectedChannel,
    /// Left mouth corner stretches.
    pub mouth_stretch_left: ProjectedChannel,
    /// Right mouth corner stretches.
    pub mouth_stretch_right: ProjectedChannel,
}

fn symmetric_pair(value: Option<ProjectedChannel>) -> (ProjectedChannel, ProjectedChannel) {
    let channel = value.unwrap_or(ProjectedChannel::UNSUPPORTED);
    (channel, channel)
}

/// Projects the lip-corner ARKit channels from one facial feature snapshot.
///
/// This function is the single deterministic definition of the input feature
/// and sign for each of the four semantics × two sides:
///
/// | Channels | Input feature | Sign |
/// |---|---|---|
/// | `MouthSmileLeft/Right` | `corner_lift > 0` | positive corner lift toward the upper-lip center |
/// | `MouthFrownLeft/Right` | `corner_lift < 0` | negative corner lift |
/// | `MouthDimpleLeft/Right` | none | permanently unsupported |
/// | `MouthStretchLeft/Right` | `width_delta > 0` | positive corner widening |
///
/// The snapshot carries only the mean of the two corners' lift and the total
/// width delta; it cannot separate the anatomical sides or distinguish a
/// dimple's retraction from other motions. Both members of each pair
/// therefore receive identical values classified
/// [`ProjectedSupport::Experimental`] whenever their evidence exists, except
/// `MouthDimple`, which stays permanently [`ProjectedSupport::Unsupported`].
/// Missing or non-finite deltas fail closed to value `0.0`; every produced
/// value is finite in `[0, 1]`.
#[must_use]
pub fn project_lip_corner_channels(features: &GnmFacialFeatures) -> LipCornerProjection {
    let lift = features.mouth_jaw.corner_lift;
    // Sign gating keeps each channel's support honest: a frown delta is not
    // zero-valued smile evidence, and constriction is not zero-valued
    // stretch evidence.
    let smile = match lift {
        Some(value) if value > 0.0 => bounded_ratio(value, SMILE_SATURATION),
        _ => None,
    };
    let frown = match lift {
        Some(value) if value < 0.0 => bounded_ratio(-value, FROWN_SATURATION),
        _ => None,
    };
    let stretch = match features.mouth_jaw.width_delta {
        Some(value) if value > 0.0 => bounded_ratio(value, STRETCH_SATURATION),
        _ => None,
    };
    let (smile_left, smile_right) = symmetric_pair(smile.map(ProjectedChannel::experimental));
    let (frown_left, frown_right) = symmetric_pair(frown.map(ProjectedChannel::experimental));
    let (stretch_left, stretch_right) = symmetric_pair(stretch.map(ProjectedChannel::experimental));

    LipCornerProjection {
        mouth_smile_left: smile_left,
        mouth_smile_right: smile_right,
        mouth_frown_left: frown_left,
        mouth_frown_right: frown_right,
        // No dedicated retraction/dimple observation exists in the snapshot;
        // dimple channels stay unsupported rather than borrowing corner lift.
        mouth_dimple_left: ProjectedChannel::UNSUPPORTED,
        mouth_dimple_right: ProjectedChannel::UNSUPPORTED,
        mouth_stretch_left: stretch_left,
        mouth_stretch_right: stretch_right,
    }
}

/// Lip roll/shrug/press ARKit channels projected from one
/// [`GnmFacialFeatures`] snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipRollShrugPressProjection {
    /// Lower lip rolls inward.
    pub mouth_roll_lower: ProjectedChannel,
    /// Upper lip rolls inward.
    pub mouth_roll_upper: ProjectedChannel,
    /// Lower lip shrugs.
    pub mouth_shrug_lower: ProjectedChannel,
    /// Upper lip shrugs.
    pub mouth_shrug_upper: ProjectedChannel,
    /// Left mouth presses.
    pub mouth_press_left: ProjectedChannel,
    /// Right mouth presses.
    pub mouth_press_right: ProjectedChannel,
}

/// Projects the lip roll, shrug, and press ARKit channels from one facial
/// feature snapshot.
///
/// This function is the single deterministic definition of the input feature
/// and sign for these channels:
///
/// | Channels | Input feature | Sign |
/// |---|---|---|
/// | `MouthRollLower/Upper` | none | permanently unsupported |
/// | `MouthShrugLower/Upper` | none | permanently unsupported |
/// | `MouthPressLeft/Right` | none | permanently unsupported |
///
/// The current snapshot contains no observation of lip rolling, shrugging,
/// or per-corner pressing: the aperture delta cannot separate an inward lip
/// roll from compression, no feature isolates vertical lip shrug, and corner
/// press has no per-side retraction measurement. Every channel is therefore
/// [`ProjectedSupport::Unsupported`] with value `0.0`. Roll and shrug are
/// never emitted as aliases of the same unfounded feature because nothing is
/// emitted at all; if a future snapshot adds dedicated observations, this
/// one site is where their signs get defined.
#[must_use]
pub fn project_lip_roll_shrug_press_channels(
    features: &GnmFacialFeatures,
) -> LipRollShrugPressProjection {
    let _ = features;
    LipRollShrugPressProjection {
        mouth_roll_lower: ProjectedChannel::UNSUPPORTED,
        mouth_roll_upper: ProjectedChannel::UNSUPPORTED,
        mouth_shrug_lower: ProjectedChannel::UNSUPPORTED,
        mouth_shrug_upper: ProjectedChannel::UNSUPPORTED,
        mouth_press_left: ProjectedChannel::UNSUPPORTED,
        mouth_press_right: ProjectedChannel::UNSUPPORTED,
    }
}

/// Lower/upper lip vertical ARKit channels projected from one
/// [`GnmFacialFeatures`] snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipLowerUpperProjection {
    /// Left lower lip moves down.
    pub mouth_lower_down_left: ProjectedChannel,
    /// Right lower lip moves down.
    pub mouth_lower_down_right: ProjectedChannel,
    /// Left upper lip moves up.
    pub mouth_upper_up_left: ProjectedChannel,
    /// Right upper lip moves up.
    pub mouth_upper_up_right: ProjectedChannel,
}

/// Projects the lower/upper lip vertical ARKit channels from one facial
/// feature snapshot.
///
/// Input feature and sign definition, in one deterministic place:
///
/// | Channels | Input feature | Sign |
/// |---|---|---|
/// | `MouthLowerDownLeft/Right` | none | permanently unsupported |
/// | `MouthUpperUpLeft/Right` | none | permanently unsupported |
///
/// The snapshot measures only the total lip-center aperture delta, which is
/// the sum of both lips' vertical motion; it cannot attribute opening to the
/// lower lip versus the upper lip, and there is no per-side lip measurement
/// at all. Attributing the shared aperture to either pair would fabricate a
/// separation the geometry does not observe, so every channel stays
/// [`ProjectedSupport::Unsupported`] with value `0.0`. If a future snapshot
/// revision adds per-lip rows, this one site is where their signs get
/// defined.
#[must_use]
pub fn project_lip_lower_upper_channels(features: &GnmFacialFeatures) -> LipLowerUpperProjection {
    let _ = features;
    LipLowerUpperProjection {
        mouth_lower_down_left: ProjectedChannel::UNSUPPORTED,
        mouth_lower_down_right: ProjectedChannel::UNSUPPORTED,
        mouth_upper_up_left: ProjectedChannel::UNSUPPORTED,
        mouth_upper_up_right: ProjectedChannel::UNSUPPORTED,
    }
}

/// Combined typed result of every lip/mouth-corner ARKit channel, merging
/// the corner ([`project_lip_corner_channels`]), roll/shrug/press
/// ([`project_lip_roll_shrug_press_channels`]), and lower/upper vertical
/// ([`project_lip_lower_upper_channels`]) projectors without duplicating any
/// channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipMouthCornerProjectorResult {
    /// Corner semantics (smile/frown/dimple/stretch, left/right).
    pub corner: LipCornerProjection,
    /// Roll/shrug/press semantics.
    pub roll_shrug_press: LipRollShrugPressProjection,
    /// Lower/upper vertical lip semantics.
    pub lower_upper: LipLowerUpperProjection,
}

/// Projects all lip/mouth-corner ARKit channels from one facial feature
/// snapshot into a single typed result.
///
/// Every ARKit channel owned by this region appears exactly once in the
/// result; the sub-projectors remain individually callable for diagnostics.
#[must_use]
pub fn project_lip_mouth_corner_channels(
    features: &GnmFacialFeatures,
) -> LipMouthCornerProjectorResult {
    LipMouthCornerProjectorResult {
        corner: project_lip_corner_channels(features),
        roll_shrug_press: project_lip_roll_shrug_press_channels(features),
        lower_upper: project_lip_lower_upper_channels(features),
    }
}

/// Canonical ARKit52 decode output: the validated coefficient vector plus the
/// per-channel evidence classification for all 52 channels.
#[derive(Clone, Debug, PartialEq)]
pub struct Arkit52DecodeResult {
    /// Validated finite `[0, 1]` coefficients in canonical channel order.
    pub coefficients: Arkit52Coefficients,
    /// Evidence classification per canonical channel index.
    pub supports: [ProjectedSupport; ARKIT52_CHANNEL_COUNT],
}

// Invariant: `ArkitBlendshape::index()` is always `< ARKIT52_CHANNEL_COUNT`,
// so every index below is in bounds by construction (same pattern as the
// vtuber-core contract).
#[allow(clippy::indexing_slicing)]
fn put_channel(
    values: &mut [f32; ARKIT52_CHANNEL_COUNT],
    supports: &mut [ProjectedSupport; ARKIT52_CHANNEL_COUNT],
    channel: ArkitBlendshape,
    projected: ProjectedChannel,
) {
    let index = channel.index();
    values[index] = projected.value;
    supports[index] = projected.support;
}

/// Decodes the temporal-coherent GNM facial feature snapshot into canonical
/// [`Arkit52Coefficients`] with per-channel support/reliability status.
///
/// The input is the neutral-normalized [`GnmFacialFeatures`] snapshot —
/// itself produced from GNM state through the identity calibration. Current
/// MediaPipe blendshape coefficients are neither a parameter nor a fallback
/// correction anywhere in this decoder. No post-decode smoothing is applied;
/// downstream consumers own any filtering policy.
///
/// Channel ownership: eye/gaze channels come from
/// [`project_eye_gaze_channels`], jaw/core-mouth from
/// [`project_jaw_core_mouth_channels`], lip/mouth-corner groups from
/// [`project_lip_mouth_corner_channels`], and brow/cheek/nose from
/// [`project_brow_cheek_nose_channels`]. Every one of the 52 channels
/// receives an explicit [`ProjectedSupport`]; `TongueOut` has no observation
/// anywhere in this module and is always [`ProjectedSupport::Unsupported`]
/// with value `0.0`.
///
/// # Errors
///
/// Returns [`Arkit52ValueError`] if the assembled vector fails the canonical
/// value validation. Every projector clamps its outputs into finite
/// `[0, 1]`, so this is a fail-closed guard, not an expected path.
pub fn decode_gnm_arkit52(
    features: &GnmFacialFeatures,
) -> Result<Arkit52DecodeResult, Arkit52ValueError> {
    let eyes = project_eye_gaze_channels(features);
    let jaw = project_jaw_core_mouth_channels(features);
    let lips = project_lip_mouth_corner_channels(features);
    let upper = project_brow_cheek_nose_channels(features);

    let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
    let mut supports = [ProjectedSupport::Unsupported; ARKIT52_CHANNEL_COUNT];

    // Eye aperture + gaze (#67.2).
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeBlinkLeft,
        eyes.eye_blink_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeBlinkRight,
        eyes.eye_blink_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeWideLeft,
        eyes.eye_wide_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeWideRight,
        eyes.eye_wide_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeSquintLeft,
        eyes.eye_squint_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeSquintRight,
        eyes.eye_squint_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeLookInLeft,
        eyes.eye_look_in_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeLookInRight,
        eyes.eye_look_in_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeLookOutLeft,
        eyes.eye_look_out_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeLookOutRight,
        eyes.eye_look_out_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeLookUpLeft,
        eyes.eye_look_up_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeLookUpRight,
        eyes.eye_look_up_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeLookDownLeft,
        eyes.eye_look_down_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::EyeLookDownRight,
        eyes.eye_look_down_right,
    );

    // Jaw + core mouth (#67.3).
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::JawForward,
        jaw.jaw_forward,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::JawLeft,
        jaw.jaw_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::JawOpen,
        jaw.jaw_open,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::JawRight,
        jaw.jaw_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthClose,
        jaw.mouth_close,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthFunnel,
        jaw.mouth_funnel,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthPucker,
        jaw.mouth_pucker,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthLeft,
        jaw.mouth_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthRight,
        jaw.mouth_right,
    );

    // Lip corners + roll/shrug/press + vertical lip motion (#67.4).
    let corner = lips.corner;
    let rsp = lips.roll_shrug_press;
    let vert = lips.lower_upper;
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthSmileLeft,
        corner.mouth_smile_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthSmileRight,
        corner.mouth_smile_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthFrownLeft,
        corner.mouth_frown_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthFrownRight,
        corner.mouth_frown_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthDimpleLeft,
        corner.mouth_dimple_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthDimpleRight,
        corner.mouth_dimple_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthStretchLeft,
        corner.mouth_stretch_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthStretchRight,
        corner.mouth_stretch_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthRollLower,
        rsp.mouth_roll_lower,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthRollUpper,
        rsp.mouth_roll_upper,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthShrugLower,
        rsp.mouth_shrug_lower,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthShrugUpper,
        rsp.mouth_shrug_upper,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthPressLeft,
        rsp.mouth_press_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthPressRight,
        rsp.mouth_press_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthLowerDownLeft,
        vert.mouth_lower_down_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthLowerDownRight,
        vert.mouth_lower_down_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthUpperUpLeft,
        vert.mouth_upper_up_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::MouthUpperUpRight,
        vert.mouth_upper_up_right,
    );

    // Brow/cheek/nose support table (#67.5).
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::BrowDownLeft,
        upper.brow_down_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::BrowDownRight,
        upper.brow_down_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::BrowInnerUp,
        upper.brow_inner_up,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::BrowOuterUpLeft,
        upper.brow_outer_up_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::BrowOuterUpRight,
        upper.brow_outer_up_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::CheekPuff,
        upper.cheek_puff,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::CheekSquintLeft,
        upper.cheek_squint_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::CheekSquintRight,
        upper.cheek_squint_right,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::NoseSneerLeft,
        upper.nose_sneer_left,
    );
    put_channel(
        &mut values,
        &mut supports,
        ArkitBlendshape::NoseSneerRight,
        upper.nose_sneer_right,
    );

    // TongueOut intentionally stays Unsupported/0: no projector produces a
    // tongue coefficient because no dedicated observation exists.

    Ok(Arkit52DecodeResult {
        coefficients: Arkit52Coefficients::try_from_array(values)?,
        supports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_gnm::{
        AnatomicalSide, BrowAuxFeatures, BrowSideAuxFeatures, EyeApertureFeature, EyeAuxFeatures,
        GnmFacialFeatures, IrisAuxFeatures, IrisSideAuxFeature, MouthAuxFeatures,
    };

    const NEUTRAL_APERTURE: f32 = 0.4;

    fn aperture(side: AnatomicalSide, current: f32) -> EyeApertureFeature {
        EyeApertureFeature {
            side,
            current_aperture: current,
            neutral_aperture: NEUTRAL_APERTURE,
            normalized_delta: (current - NEUTRAL_APERTURE) / 0.3,
        }
    }

    fn iris(
        side: AnatomicalSide,
        vertical: impl FnOnce() -> Option<f32>,
        horizontal: impl FnOnce() -> Option<f32>,
    ) -> IrisSideAuxFeature {
        IrisSideAuxFeature {
            side,
            vertical_delta: vertical(),
            horizontal_delta: horizontal(),
        }
    }

    fn snapshot(
        right_eye: EyeApertureFeature,
        left_eye: EyeApertureFeature,
        right_iris: Option<IrisSideAuxFeature>,
        left_iris: Option<IrisSideAuxFeature>,
    ) -> GnmFacialFeatures {
        let brow = |side| BrowSideAuxFeatures {
            side,
            inner_rise: None,
            brow_lower: None,
            outer_rise: None,
        };
        GnmFacialFeatures {
            eyes: EyeAuxFeatures {
                right: right_eye,
                left: left_eye,
            },
            irises: IrisAuxFeatures {
                right: right_iris,
                left: left_iris,
            },
            mouth_jaw: MouthAuxFeatures::default(),
            cheeks: vtuber_gnm::CheekAuxFeatures::default(),
            brows: BrowAuxFeatures {
                right: brow(AnatomicalSide::Right),
                left: brow(AnatomicalSide::Left),
            },
        }
    }

    fn neutral_snapshot_with_irises(irises: bool) -> GnmFacialFeatures {
        let (right_iris, left_iris) = if irises {
            (
                Some(iris(AnatomicalSide::Right, || Some(0.0), || Some(0.0))),
                Some(iris(AnatomicalSide::Left, || Some(0.0), || Some(0.0))),
            )
        } else {
            (None, None)
        };
        snapshot(
            aperture(AnatomicalSide::Right, NEUTRAL_APERTURE),
            aperture(AnatomicalSide::Left, NEUTRAL_APERTURE),
            right_iris,
            left_iris,
        )
    }

    #[test]
    fn neutral_snapshot_projects_near_zero_supported_channels() {
        let projection = project_eye_gaze_channels(&neutral_snapshot_with_irises(true));
        for channel in [
            projection.eye_blink_left,
            projection.eye_blink_right,
            projection.eye_wide_left,
            projection.eye_wide_right,
            projection.eye_squint_left,
            projection.eye_squint_right,
            projection.eye_look_in_left,
            projection.eye_look_in_right,
            projection.eye_look_out_left,
            projection.eye_look_out_right,
            projection.eye_look_up_left,
            projection.eye_look_up_right,
            projection.eye_look_down_left,
            projection.eye_look_down_right,
        ] {
            assert!(
                channel.value.abs() < 1e-6,
                "neutral must be ~0: {channel:?}"
            );
        }
        assert_eq!(
            projection.eye_blink_right.support,
            ProjectedSupport::Reliable
        );
        assert_eq!(
            projection.eye_squint_right.support,
            ProjectedSupport::Experimental
        );
        assert_eq!(
            projection.eye_look_up_right.support,
            ProjectedSupport::Reliable
        );
    }

    #[test]
    fn half_closure_pins_blink_and_squint_sign_and_range() {
        let closed_half = aperture(AnatomicalSide::Right, 0.5 * NEUTRAL_APERTURE);
        let open = aperture(AnatomicalSide::Left, NEUTRAL_APERTURE);
        let projection = project_eye_gaze_channels(&snapshot(closed_half, open, None, None));
        assert_eq!(
            projection.eye_blink_right.support,
            ProjectedSupport::Reliable
        );
        assert!(
            (projection.eye_blink_right.value - 0.5 / BLINK_FULL_CLOSURE_FRACTION).abs() < 1e-5
        );
        assert_eq!(
            projection.eye_squint_right.support,
            ProjectedSupport::Experimental
        );
        assert!((projection.eye_squint_right.value - 1.0).abs() < 1e-5);
        // The other side is untouched.
        assert!(projection.eye_blink_left.value.abs() < 1e-6);
        assert!(projection.eye_squint_left.value.abs() < 1e-6);
        // Widening is not triggered by closure.
        assert_eq!(projection.eye_wide_right.value, 0.0);
    }

    #[test]
    fn full_closure_saturates_blink_at_one() {
        let fully_closed = aperture(AnatomicalSide::Left, 0.0);
        let projection =
            project_eye_gaze_channels(&snapshot(fully_closed, fully_closed, None, None));
        assert_eq!(projection.eye_blink_left.value, 1.0);
        assert_eq!(projection.eye_blink_right.value, 1.0);
    }

    #[test]
    fn widening_pins_wide_sign_and_range() {
        let widened = aperture(AnatomicalSide::Right, NEUTRAL_APERTURE * 1.6);
        let projection = project_eye_gaze_channels(&snapshot(
            widened,
            aperture(AnatomicalSide::Left, NEUTRAL_APERTURE),
            None,
            None,
        ));
        assert_eq!(
            projection.eye_wide_right.support,
            ProjectedSupport::Reliable
        );
        assert!((projection.eye_wide_right.value - 1.0).abs() < 1e-5);
        assert_eq!(projection.eye_blink_right.value, 0.0);
        assert_eq!(projection.eye_squint_right.value, 0.0);
        assert_eq!(projection.eye_wide_left.value, 0.0);
    }

    #[test]
    fn degenerate_neutral_aperture_is_unsupported_not_zero_reliable() {
        let mut degenerate = aperture(AnatomicalSide::Right, 0.2);
        degenerate.neutral_aperture = 0.0;
        let projection = project_eye_gaze_channels(&snapshot(
            degenerate,
            aperture(AnatomicalSide::Left, NEUTRAL_APERTURE),
            None,
            None,
        ));
        for channel in [
            projection.eye_blink_right,
            projection.eye_wide_right,
            projection.eye_squint_right,
        ] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
    }

    #[test]
    fn gaze_directions_follow_anatomical_iris_deltas() {
        let right = iris(
            AnatomicalSide::Right,
            || Some(0.05), // up, half of vertical saturation
            || Some(0.04), // outward, half of horizontal saturation
        );
        let left = iris(
            AnatomicalSide::Left,
            || Some(-0.05), // down
            || Some(-0.04), // inward
        );
        let projection = project_eye_gaze_channels(&snapshot(
            aperture(AnatomicalSide::Right, NEUTRAL_APERTURE),
            aperture(AnatomicalSide::Left, NEUTRAL_APERTURE),
            Some(right),
            Some(left),
        ));
        assert!((projection.eye_look_up_right.value - 0.5).abs() < 1e-5);
        assert_eq!(projection.eye_look_down_right.value, 0.0);
        assert!((projection.eye_look_out_right.value - 0.5).abs() < 1e-5);
        assert_eq!(projection.eye_look_in_right.value, 0.0);

        assert!((projection.eye_look_down_left.value - 0.5).abs() < 1e-5);
        assert_eq!(projection.eye_look_up_left.value, 0.0);
        assert!((projection.eye_look_in_left.value - 0.5).abs() < 1e-5);
        assert_eq!(projection.eye_look_out_left.value, 0.0);
    }

    #[test]
    fn missing_or_non_finite_iris_features_are_explicitly_unsupported() {
        let no_iris = project_eye_gaze_channels(&neutral_snapshot_with_irises(false));
        for channel in [
            no_iris.eye_look_in_left,
            no_iris.eye_look_in_right,
            no_iris.eye_look_out_left,
            no_iris.eye_look_out_right,
            no_iris.eye_look_up_left,
            no_iris.eye_look_up_right,
            no_iris.eye_look_down_left,
            no_iris.eye_look_down_right,
        ] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }

        let nan_iris = iris(AnatomicalSide::Right, || Some(f32::NAN), || None);
        let projection = project_eye_gaze_channels(&snapshot(
            aperture(AnatomicalSide::Right, NEUTRAL_APERTURE),
            aperture(AnatomicalSide::Left, NEUTRAL_APERTURE),
            Some(nan_iris),
            None,
        ));
        assert_eq!(
            projection.eye_look_up_right.support,
            ProjectedSupport::Unsupported
        );
        assert_eq!(projection.eye_look_up_right.value, 0.0);
        assert_eq!(
            projection.eye_look_down_right.support,
            ProjectedSupport::Unsupported
        );
    }

    #[test]
    fn mirrored_sides_swap_projection_outputs_exactly() {
        let right_eye = aperture(AnatomicalSide::Right, 0.5 * NEUTRAL_APERTURE);
        let left_eye = aperture(AnatomicalSide::Left, NEUTRAL_APERTURE * 1.6);
        let right_iris = iris(AnatomicalSide::Right, || Some(0.05), || Some(0.02));
        let left_iris = iris(AnatomicalSide::Left, || Some(-0.03), || Some(-0.01));

        let direct = project_eye_gaze_channels(&snapshot(
            right_eye,
            left_eye,
            Some(right_iris),
            Some(left_iris),
        ));
        let mirrored = project_eye_gaze_channels(&snapshot(
            left_eye,
            right_eye,
            Some(left_iris),
            Some(right_iris),
        ));

        assert_eq!(direct.eye_blink_left, mirrored.eye_blink_right);
        assert_eq!(direct.eye_blink_right, mirrored.eye_blink_left);
        assert_eq!(direct.eye_wide_left, mirrored.eye_wide_right);
        assert_eq!(direct.eye_wide_right, mirrored.eye_wide_left);
        assert_eq!(direct.eye_squint_left, mirrored.eye_squint_right);
        assert_eq!(direct.eye_squint_right, mirrored.eye_squint_left);
        assert_eq!(direct.eye_look_in_left, mirrored.eye_look_in_right);
        assert_eq!(direct.eye_look_in_right, mirrored.eye_look_in_left);
        assert_eq!(direct.eye_look_out_left, mirrored.eye_look_out_right);
        assert_eq!(direct.eye_look_out_right, mirrored.eye_look_out_left);
        assert_eq!(direct.eye_look_up_left, mirrored.eye_look_up_right);
        assert_eq!(direct.eye_look_up_right, mirrored.eye_look_up_left);
        assert_eq!(direct.eye_look_down_left, mirrored.eye_look_down_right);
        assert_eq!(direct.eye_look_down_right, mirrored.eye_look_down_left);
    }

    fn mouth_features(
        jaw_open: Option<f32>,
        jaw_forward: Option<f32>,
        jaw_lateral: Option<f32>,
        width_delta: Option<f32>,
    ) -> vtuber_gnm::MouthAuxFeatures {
        vtuber_gnm::MouthAuxFeatures {
            jaw_open,
            jaw_forward,
            jaw_lateral,
            width_delta,
            corner_lift: None,
        }
    }

    fn snapshot_with_mouth(mouth: vtuber_gnm::MouthAuxFeatures) -> GnmFacialFeatures {
        let mut base = neutral_snapshot_with_irises(false);
        base.mouth_jaw = mouth;
        base
    }

    fn mouth_with_corner_lift(corner_lift: Option<f32>) -> vtuber_gnm::MouthAuxFeatures {
        vtuber_gnm::MouthAuxFeatures {
            corner_lift,
            ..vtuber_gnm::MouthAuxFeatures::default()
        }
    }

    #[test]
    fn smile_alone_pins_sign_and_range() {
        let projection =
            project_lip_corner_channels(&snapshot_with_mouth(mouth_with_corner_lift(Some(0.05))));
        assert!((projection.mouth_smile_left.value - 0.5).abs() < 1e-5);
        assert_eq!(projection.mouth_smile_right, projection.mouth_smile_left);
        assert_eq!(
            projection.mouth_smile_left.support,
            ProjectedSupport::Experimental
        );
        assert_eq!(projection.mouth_frown_left.value, 0.0);
        assert_eq!(projection.mouth_stretch_left.value, 0.0);
    }

    #[test]
    fn frown_alone_pins_sign_and_range() {
        let projection =
            project_lip_corner_channels(&snapshot_with_mouth(mouth_with_corner_lift(Some(-0.03))));
        assert!((projection.mouth_frown_left.value - 0.5).abs() < 1e-5);
        assert_eq!(projection.mouth_frown_right, projection.mouth_frown_left);
        assert_eq!(
            projection.mouth_frown_left.support,
            ProjectedSupport::Experimental
        );
        assert_eq!(projection.mouth_smile_left.value, 0.0);
    }

    #[test]
    fn stretch_alone_uses_positive_width_delta() {
        let mouth = vtuber_gnm::MouthAuxFeatures {
            width_delta: Some(0.075),
            ..vtuber_gnm::MouthAuxFeatures::default()
        };
        let projection = project_lip_corner_channels(&snapshot_with_mouth(mouth));
        assert!((projection.mouth_stretch_left.value - 0.5).abs() < 1e-5);
        assert_eq!(
            projection.mouth_stretch_right,
            projection.mouth_stretch_left
        );
        assert_eq!(
            projection.mouth_stretch_left.support,
            ProjectedSupport::Experimental
        );
        // Negative width (constriction) is not stretch evidence.
        let narrow = vtuber_gnm::MouthAuxFeatures {
            width_delta: Some(-0.15),
            ..vtuber_gnm::MouthAuxFeatures::default()
        };
        let narrowed = project_lip_corner_channels(&snapshot_with_mouth(narrow));
        assert_eq!(narrowed.mouth_stretch_left.value, 0.0);
        assert_eq!(
            narrowed.mouth_stretch_left.support,
            ProjectedSupport::Unsupported
        );
    }

    #[test]
    fn dimple_never_fires_even_with_full_other_evidence() {
        let full = vtuber_gnm::MouthAuxFeatures {
            corner_lift: Some(0.2),
            width_delta: Some(0.3),
            ..vtuber_gnm::MouthAuxFeatures::default()
        };
        let projection = project_lip_corner_channels(&snapshot_with_mouth(full));
        for channel in [projection.mouth_dimple_left, projection.mouth_dimple_right] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
    }

    #[test]
    fn lip_corner_mirror_fixture_is_symmetric_across_sides() {
        // The snapshot aggregates both corners; a mirrored face yields the
        // same aggregate evidence and therefore exactly symmetric outputs.
        let direct =
            project_lip_corner_channels(&snapshot_with_mouth(mouth_with_corner_lift(Some(0.04))));
        let mirrored =
            project_lip_corner_channels(&snapshot_with_mouth(mouth_with_corner_lift(Some(0.04))));
        assert_eq!(direct, mirrored);
        assert_eq!(direct.mouth_smile_left, direct.mouth_smile_right);
        assert_eq!(direct.mouth_frown_left, direct.mouth_frown_right);
        assert_eq!(direct.mouth_dimple_left, direct.mouth_dimple_right);
        assert_eq!(direct.mouth_stretch_left, direct.mouth_stretch_right);
    }

    #[test]
    fn neutral_lip_corners_project_near_zero_and_missing_evidence_is_unsupported() {
        let neutral =
            project_lip_corner_channels(&snapshot_with_mouth(mouth_with_corner_lift(Some(0.0))));
        assert!(neutral.mouth_smile_left.value.abs() < 1e-6);
        assert!(neutral.mouth_frown_right.value.abs() < 1e-6);
        assert_eq!(
            neutral.mouth_stretch_left.support,
            ProjectedSupport::Unsupported
        );

        let absent = project_lip_corner_channels(&snapshot_with_mouth(
            vtuber_gnm::MouthAuxFeatures::default(),
        ));
        for channel in [
            absent.mouth_smile_left,
            absent.mouth_smile_right,
            absent.mouth_frown_left,
            absent.mouth_frown_right,
            absent.mouth_dimple_left,
            absent.mouth_dimple_right,
            absent.mouth_stretch_left,
            absent.mouth_stretch_right,
        ] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
    }

    #[test]
    fn every_lip_corner_output_is_finite_and_within_unit_range() {
        let extreme = vtuber_gnm::MouthAuxFeatures {
            corner_lift: Some(-f32::INFINITY),
            width_delta: Some(f32::NAN),
            ..vtuber_gnm::MouthAuxFeatures::default()
        };
        let projection = project_lip_corner_channels(&snapshot_with_mouth(extreme));
        for channel in [
            projection.mouth_smile_left,
            projection.mouth_smile_right,
            projection.mouth_frown_left,
            projection.mouth_frown_right,
            projection.mouth_dimple_left,
            projection.mouth_dimple_right,
            projection.mouth_stretch_left,
            projection.mouth_stretch_right,
        ] {
            assert!(channel.value.is_finite());
            assert!((0.0..=1.0).contains(&channel.value));
        }
    }

    #[test]
    fn roll_shrug_press_stay_unsupported_and_never_alias_other_mouth_evidence() {
        // Even with every available mouth feature firing, roll/shrug/press
        // channels must not borrow their evidence.
        let full = vtuber_gnm::MouthAuxFeatures {
            jaw_open: Some(0.5),
            jaw_forward: Some(-0.2),
            jaw_lateral: Some(0.1),
            width_delta: Some(-0.3),
            corner_lift: Some(0.2),
        };
        let projection = project_lip_roll_shrug_press_channels(&snapshot_with_mouth(full));
        for channel in [
            projection.mouth_roll_lower,
            projection.mouth_roll_upper,
            projection.mouth_shrug_lower,
            projection.mouth_shrug_upper,
            projection.mouth_press_left,
            projection.mouth_press_right,
        ] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
    }

    #[test]
    fn roll_shrug_press_are_independent_classifications_not_shared_alias() {
        // Each channel family carries its own (absent-evidence)
        // classification; none of them is derived from another channel or a
        // shared fabricated value.
        let projection =
            project_lip_roll_shrug_press_channels(&neutral_snapshot_with_irises(false));
        assert_ne!(projection.mouth_roll_lower, ProjectedChannel::reliable(0.5));
        assert_eq!(projection.mouth_shrug_upper, ProjectedChannel::UNSUPPORTED);
        assert_eq!(projection.mouth_press_right, ProjectedChannel::UNSUPPORTED);
        assert_eq!(
            project_lip_roll_shrug_press_channels(&snapshot_with_mouth(
                vtuber_gnm::MouthAuxFeatures::default()
            )),
            projection
        );
    }

    #[test]
    fn lower_upper_lip_channels_stay_unsupported_without_per_lip_evidence() {
        // Even a large aperture delta is shared upper+lower evidence; it must
        // not be attributed to either lip pair.
        let opening = vtuber_gnm::MouthAuxFeatures {
            jaw_open: Some(0.5),
            ..vtuber_gnm::MouthAuxFeatures::default()
        };
        let projection = project_lip_lower_upper_channels(&snapshot_with_mouth(opening));
        for channel in [
            projection.mouth_lower_down_left,
            projection.mouth_lower_down_right,
            projection.mouth_upper_up_left,
            projection.mouth_upper_up_right,
        ] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
    }

    #[test]
    fn combined_lip_result_covers_each_channel_exactly_once() {
        use vtuber_core::ArkitBlendshape;

        let result = project_lip_mouth_corner_channels(&neutral_snapshot_with_irises(false));
        let channels = [
            (
                result.corner.mouth_smile_left,
                ArkitBlendshape::MouthSmileLeft,
            ),
            (
                result.corner.mouth_smile_right,
                ArkitBlendshape::MouthSmileRight,
            ),
            (
                result.corner.mouth_frown_left,
                ArkitBlendshape::MouthFrownLeft,
            ),
            (
                result.corner.mouth_frown_right,
                ArkitBlendshape::MouthFrownRight,
            ),
            (
                result.corner.mouth_dimple_left,
                ArkitBlendshape::MouthDimpleLeft,
            ),
            (
                result.corner.mouth_dimple_right,
                ArkitBlendshape::MouthDimpleRight,
            ),
            (
                result.corner.mouth_stretch_left,
                ArkitBlendshape::MouthStretchLeft,
            ),
            (
                result.corner.mouth_stretch_right,
                ArkitBlendshape::MouthStretchRight,
            ),
            (
                result.roll_shrug_press.mouth_roll_lower,
                ArkitBlendshape::MouthRollLower,
            ),
            (
                result.roll_shrug_press.mouth_roll_upper,
                ArkitBlendshape::MouthRollUpper,
            ),
            (
                result.roll_shrug_press.mouth_shrug_lower,
                ArkitBlendshape::MouthShrugLower,
            ),
            (
                result.roll_shrug_press.mouth_shrug_upper,
                ArkitBlendshape::MouthShrugUpper,
            ),
            (
                result.roll_shrug_press.mouth_press_left,
                ArkitBlendshape::MouthPressLeft,
            ),
            (
                result.roll_shrug_press.mouth_press_right,
                ArkitBlendshape::MouthPressRight,
            ),
            (
                result.lower_upper.mouth_lower_down_left,
                ArkitBlendshape::MouthLowerDownLeft,
            ),
            (
                result.lower_upper.mouth_lower_down_right,
                ArkitBlendshape::MouthLowerDownRight,
            ),
            (
                result.lower_upper.mouth_upper_up_left,
                ArkitBlendshape::MouthUpperUpLeft,
            ),
            (
                result.lower_upper.mouth_upper_up_right,
                ArkitBlendshape::MouthUpperUpRight,
            ),
        ];
        let mut indices = [0usize; 18];
        for (slot, (_, channel)) in indices.iter_mut().zip(channels.iter()) {
            *slot = channel.index();
        }
        indices.sort_unstable();
        for pair in indices.windows(2) {
            assert_ne!(pair[0], pair[1], "lip channels must appear exactly once");
        }
        assert_eq!(channels.len(), 18);
    }

    #[test]
    fn combined_result_matches_individual_projectors_on_mixed_motion() {
        let mixed = vtuber_gnm::MouthAuxFeatures {
            jaw_open: Some(0.1),
            width_delta: Some(-0.2),
            corner_lift: Some(0.06),
            ..vtuber_gnm::MouthAuxFeatures::default()
        };
        let snapshot = snapshot_with_mouth(mixed);
        let combined = project_lip_mouth_corner_channels(&snapshot);
        assert_eq!(combined.corner, project_lip_corner_channels(&snapshot));
        assert_eq!(
            combined.roll_shrug_press,
            project_lip_roll_shrug_press_channels(&snapshot)
        );
        assert_eq!(
            combined.lower_upper,
            project_lip_lower_upper_channels(&snapshot)
        );
        // Mixed motion sanity: smile fires from lift, constriction does not
        // leak into stretch.
        assert_eq!(
            combined.corner.mouth_smile_left.support,
            ProjectedSupport::Experimental
        );
        assert!(combined.corner.mouth_smile_left.value > 0.5);
        assert_eq!(combined.corner.mouth_stretch_left.value, 0.0);
    }

    #[test]
    fn decoder_covers_all_52_channels_with_explicit_support() {
        let result = decode_gnm_arkit52(&neutral_snapshot_with_irises(true))
            .expect("valid neutral snapshot must decode");
        // Every channel has finite [0,1] value and a non-default support.
        for (index, value) in result.coefficients.as_array().iter().enumerate() {
            assert!(value.is_finite());
            assert!((0.0..=1.0).contains(value), "channel {index} out of range");
        }
        // Spot-check ownership: tongue never fabricated.
        let tongue = ArkitBlendshape::TongueOut.index();
        assert_eq!(result.supports[tongue], ProjectedSupport::Unsupported);
        assert_eq!(result.coefficients.as_array()[tongue], 0.0);
    }

    #[test]
    fn decoder_matches_region_projectors_on_a_mixed_fixture() {
        let mut mixed = neutral_snapshot_with_irises(true);
        mixed.eyes.right.current_aperture = 0.5 * NEUTRAL_APERTURE;
        mixed.mouth_jaw.jaw_open = Some(0.2);
        mixed.mouth_jaw.corner_lift = Some(0.06);
        mixed.mouth_jaw.width_delta = Some(-0.15);
        mixed.brows.right.brow_lower = Some(0.08);
        mixed.irises.left = Some(iris(AnatomicalSide::Left, || Some(0.05), || Some(0.0)));

        let decoded = decode_gnm_arkit52(&mixed).expect("mixed fixture must decode");
        let eyes = project_eye_gaze_channels(&mixed);
        let jaw = project_jaw_core_mouth_channels(&mixed);
        let lips = project_lip_mouth_corner_channels(&mixed);
        let upper = project_brow_cheek_nose_channels(&mixed);

        let expect = |channel: ArkitBlendshape, projected: ProjectedChannel| {
            let index = channel.index();
            assert_eq!(decoded.coefficients.get(channel), projected.value);
            assert_eq!(decoded.supports[index], projected.support);
        };
        expect(ArkitBlendshape::EyeBlinkRight, eyes.eye_blink_right);
        expect(ArkitBlendshape::EyeLookUpLeft, eyes.eye_look_up_left);
        expect(ArkitBlendshape::JawOpen, jaw.jaw_open);
        expect(ArkitBlendshape::MouthClose, jaw.mouth_close);
        expect(
            ArkitBlendshape::MouthSmileLeft,
            lips.corner.mouth_smile_left,
        );
        expect(ArkitBlendshape::MouthFunnel, jaw.mouth_funnel);
        // Constriction fired on the fixture and lands on the funnel channel.
        assert!((decoded.coefficients.get(ArkitBlendshape::MouthFunnel) - 0.6).abs() < 1e-5);
        expect(
            ArkitBlendshape::MouthRollLower,
            lips.roll_shrug_press.mouth_roll_lower,
        );
        expect(ArkitBlendshape::BrowDownRight, upper.brow_down_right);
        // Smile actually fired on the fixture.
        assert!(decoded.coefficients.get(ArkitBlendshape::MouthSmileLeft) > 0.5);
    }

    #[test]
    fn neutral_decode_is_near_zero_on_supported_channels() {
        let decoded =
            decode_gnm_arkit52(&neutral_snapshot_with_irises(true)).expect("neutral must decode");
        for (channel, value) in ArkitBlendshape::ALL
            .into_iter()
            .zip(decoded.coefficients.as_array())
        {
            if decoded.supports[channel.index()] == ProjectedSupport::Unsupported {
                assert_eq!(*value, 0.0, "{channel:?} unsupported but nonzero");
            } else {
                assert!(value.abs() < 1e-6, "{channel:?} neutral must be ~0");
            }
        }
    }

    #[test]
    fn mirrored_snapshot_swaps_sides_in_canonical_output() {
        let right_blink = aperture(AnatomicalSide::Right, 0.5 * NEUTRAL_APERTURE);
        let left_wide = aperture(AnatomicalSide::Left, NEUTRAL_APERTURE * 1.6);
        let direct = decode_gnm_arkit52(&snapshot(right_blink, left_wide, None, None))
            .expect("direct decode failed");
        let mirrored = decode_gnm_arkit52(&snapshot(left_wide, right_blink, None, None))
            .expect("mirrored decode failed");
        assert_eq!(
            direct.coefficients.get(ArkitBlendshape::EyeBlinkRight),
            mirrored.coefficients.get(ArkitBlendshape::EyeBlinkLeft)
        );
        assert_eq!(
            direct.coefficients.get(ArkitBlendshape::EyeWideLeft),
            mirrored.coefficients.get(ArkitBlendshape::EyeWideRight)
        );
    }

    #[test]
    fn jaw_sign_and_range_pinned_by_synthetic_fixtures() {
        // Open: positive lip-aperture delta.
        let open = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            Some(0.175), // half of JAW_OPEN_SATURATION
            None,
            None,
            None,
        )));
        assert!((open.jaw_open.value - 0.5).abs() < 1e-5);
        assert_eq!(open.jaw_open.support, ProjectedSupport::Reliable);
        assert_eq!(open.mouth_close.value, 0.0);

        // Forward: negative chin-to-nose-tip delta.
        let forward = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            None,
            Some(-0.15),
            None,
            None,
        )));
        assert_eq!(forward.jaw_forward.value, 1.0);
        assert_eq!(forward.jaw_forward.support, ProjectedSupport::Reliable);

        // Lateral: positive delta is toward the anatomical left.
        let lateral = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            None,
            None,
            Some(0.075),
            None,
        )));
        assert!((lateral.jaw_left.value - 0.5).abs() < 1e-5);
        assert_eq!(lateral.jaw_right.value, 0.0);
        let mirrored = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            None,
            None,
            Some(-0.075),
            None,
        )));
        assert!((mirrored.jaw_right.value - 0.5).abs() < 1e-5);
        assert_eq!(mirrored.jaw_left.value, 0.0);
    }

    #[test]
    fn mouth_close_is_bounded_and_exclusive_with_jaw_open() {
        // Lip compression: negative lip-aperture delta drives MouthClose only.
        let closed = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            Some(-0.075),
            None,
            None,
            None,
        )));
        assert!((closed.mouth_close.value - 0.5).abs() < 1e-5);
        assert_eq!(closed.mouth_close.support, ProjectedSupport::Experimental);
        assert_eq!(closed.jaw_open.support, ProjectedSupport::Unsupported);
        assert_eq!(closed.jaw_open.value, 0.0);

        // Extreme contradiction: a huge positive jaw-open can never produce
        // MouthClose, and vice versa; both remain finite in [0, 1].
        let wide_open = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            Some(10.0),
            None,
            None,
            None,
        )));
        assert_eq!(wide_open.jaw_open.value, 1.0);
        assert_eq!(wide_open.mouth_close.value, 0.0);
        let pressed = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            Some(-10.0),
            None,
            None,
            None,
        )));
        assert_eq!(pressed.mouth_close.value, 1.0);
        assert_eq!(pressed.jaw_open.value, 0.0);
    }

    #[test]
    fn funnel_pucker_share_constriction_evidence_as_experimental() {
        let constricted = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            None,
            None,
            None,
            Some(-0.125), // corner narrowing, half saturation
        )));
        assert_eq!(
            constricted.mouth_funnel.support,
            ProjectedSupport::Experimental
        );
        assert_eq!(constricted.mouth_funnel, constricted.mouth_pucker);
        assert!((constricted.mouth_funnel.value - 0.5).abs() < 1e-5);
    }

    #[test]
    fn mouth_lateral_shift_stays_unsupported() {
        let projection = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            Some(0.1),
            Some(-0.05),
            Some(0.05),
            Some(0.1),
        )));
        for channel in [projection.mouth_left, projection.mouth_right] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
    }

    #[test]
    fn missing_mouth_features_fail_closed_to_unsupported_zero() {
        let projection = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            None, None, None, None,
        )));
        for channel in [
            projection.jaw_forward,
            projection.jaw_left,
            projection.jaw_open,
            projection.jaw_right,
            projection.mouth_close,
            projection.mouth_funnel,
            projection.mouth_pucker,
            projection.mouth_left,
            projection.mouth_right,
        ] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
    }

    #[test]
    fn non_finite_mouth_deltas_do_not_break_bounds() {
        let projection = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NAN),
            Some(-f32::INFINITY),
        )));
        for channel in [
            projection.jaw_forward,
            projection.jaw_left,
            projection.jaw_open,
            projection.jaw_right,
            projection.mouth_close,
            projection.mouth_funnel,
            projection.mouth_pucker,
        ] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
    }

    #[test]
    fn rigid_head_pose_alone_cannot_change_mouth_coefficients() {
        // The projector consumes only pairwise-distance deltas, which are
        // invariant under rigid head transforms; the same expression measured
        // under two different rigid poses yields byte-identical snapshots and
        // therefore identical projections. Unrelated feature families must
        // also not leak into the mouth channels.
        let mut with_irises = snapshot_with_mouth(mouth_features(Some(0.2), None, None, None));
        with_irises.irises.right = Some(IrisSideAuxFeature {
            side: AnatomicalSide::Right,
            vertical_delta: Some(0.05),
            horizontal_delta: Some(0.02),
        });
        let plain = snapshot_with_mouth(mouth_features(Some(0.2), None, None, None));
        assert_eq!(
            project_jaw_core_mouth_channels(&with_irises),
            project_jaw_core_mouth_channels(&plain)
        );
    }

    #[test]
    fn every_jaw_mouth_output_is_finite_and_within_unit_range() {
        let extreme = project_jaw_core_mouth_channels(&snapshot_with_mouth(mouth_features(
            Some(-30.0),
            Some(30.0),
            Some(-30.0),
            Some(30.0),
        )));
        for channel in [
            extreme.jaw_forward,
            extreme.jaw_left,
            extreme.jaw_open,
            extreme.jaw_right,
            extreme.mouth_close,
            extreme.mouth_funnel,
            extreme.mouth_pucker,
            extreme.mouth_left,
            extreme.mouth_right,
        ] {
            assert!(channel.value.is_finite());
            assert!((0.0..=1.0).contains(&channel.value));
        }
    }

    fn snapshot_with_brows(
        right: BrowSideAuxFeatures,
        left: BrowSideAuxFeatures,
    ) -> GnmFacialFeatures {
        let mut base = neutral_snapshot_with_irises(false);
        base.brows = BrowAuxFeatures { right, left };
        base
    }

    fn brow(
        side: AnatomicalSide,
        inner: Option<f32>,
        lower: Option<f32>,
        outer: Option<f32>,
    ) -> BrowSideAuxFeatures {
        BrowSideAuxFeatures {
            side,
            inner_rise: inner,
            brow_lower: lower,
            outer_rise: outer,
        }
    }

    #[test]
    fn brow_down_sign_side_and_range_pinned_by_fixture() {
        // Right brow lowered by half saturation; left brow absent.
        let projection = project_brow_cheek_nose_channels(&snapshot_with_brows(
            brow(AnatomicalSide::Right, None, Some(0.04), None),
            brow(AnatomicalSide::Left, None, None, None),
        ));
        assert!((projection.brow_down_right.value - 0.5).abs() < 1e-5);
        assert_eq!(
            projection.brow_down_right.support,
            ProjectedSupport::Reliable
        );
        assert_eq!(
            projection.brow_down_left.support,
            ProjectedSupport::Unsupported
        );
        assert_eq!(projection.brow_down_left.value, 0.0);
    }

    #[test]
    fn brow_inner_up_uses_mean_positive_part_of_both_sides() {
        let raised = project_brow_cheek_nose_channels(&snapshot_with_brows(
            brow(AnatomicalSide::Right, Some(0.08), None, None),
            brow(AnatomicalSide::Left, Some(0.08), None, None),
        ));
        assert_eq!(raised.brow_inner_up.value, 1.0);
        assert_eq!(raised.brow_inner_up.support, ProjectedSupport::Reliable);

        // Lowered inner brows do not fabricate inner-up.
        let lowered = project_brow_cheek_nose_channels(&snapshot_with_brows(
            brow(AnatomicalSide::Right, Some(-0.08), Some(0.04), None),
            brow(AnatomicalSide::Left, Some(-0.08), None, None),
        ));
        assert_eq!(lowered.brow_inner_up.value, 0.0);
        assert!((lowered.brow_down_right.value - 0.5).abs() < 1e-5);

        // One-sided raise uses the mean of the present measurements.
        let one_sided = project_brow_cheek_nose_channels(&snapshot_with_brows(
            brow(AnatomicalSide::Right, Some(0.04), None, None),
            brow(AnatomicalSide::Left, None, None, None),
        ));
        assert!((one_sided.brow_inner_up.value - 0.5).abs() < 1e-5);
    }

    #[test]
    fn brow_outer_up_is_side_keyed() {
        let projection = project_brow_cheek_nose_channels(&snapshot_with_brows(
            brow(AnatomicalSide::Right, None, None, Some(0.02)),
            brow(AnatomicalSide::Left, None, None, Some(0.06)),
        ));
        assert!((projection.brow_outer_up_right.value - 0.25).abs() < 1e-5);
        assert!((projection.brow_outer_up_left.value - 0.75).abs() < 1e-5);
    }

    #[test]
    fn cheek_squint_and_nose_are_deterministically_unsupported() {
        // Even with full brow evidence present, squint/nose channels never
        // fire: the snapshot carries no dedicated observation for them.
        // `CheekPuff` is excluded here; it has its own geometric gate below.
        let projection = project_brow_cheek_nose_channels(&snapshot_with_brows(
            brow(AnatomicalSide::Right, Some(0.1), Some(0.1), Some(0.1)),
            brow(AnatomicalSide::Left, Some(0.1), Some(0.1), Some(0.1)),
        ));
        for channel in [
            projection.cheek_squint_left,
            projection.cheek_squint_right,
            projection.nose_sneer_left,
            projection.nose_sneer_right,
        ] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
        // Without cheek/mouth evidence the puff gate also stays closed.
        assert_eq!(projection.cheek_puff.support, ProjectedSupport::Unsupported);
        assert_eq!(projection.cheek_puff.value, 0.0);
    }

    fn snapshot_with_cheek(
        outward: Option<f32>,
        width: Option<f32>,
        forward: Option<f32>,
    ) -> GnmFacialFeatures {
        let mut base = neutral_snapshot_with_irises(false);
        base.cheeks = vtuber_gnm::CheekAuxFeatures {
            contour_outward: outward,
        };
        base.mouth_jaw = vtuber_gnm::MouthAuxFeatures {
            width_delta: width,
            jaw_forward: forward,
            ..vtuber_gnm::MouthAuxFeatures::default()
        };
        base
    }

    #[test]
    fn cheek_puff_fires_only_on_three_condition_and() {
        let puffed = project_brow_cheek_nose_channels(&snapshot_with_cheek(
            Some(0.06),
            Some(0.0),
            Some(0.0),
        ));
        assert_eq!(puffed.cheek_puff.support, ProjectedSupport::Experimental);
        assert!(puffed.cheek_puff.value > 0.0);

        // Mouth widened (smile-like) blocks the gate.
        let smiled = project_brow_cheek_nose_channels(&snapshot_with_cheek(
            Some(0.06),
            Some(0.10),
            Some(0.0),
        ));
        assert_eq!(smiled.cheek_puff.support, ProjectedSupport::Unsupported);

        // Contour inward (purse/blow-like) blocks the gate.
        let pursed = project_brow_cheek_nose_channels(&snapshot_with_cheek(
            Some(-0.05),
            Some(0.0),
            Some(0.0),
        ));
        assert_eq!(pursed.cheek_puff.support, ProjectedSupport::Unsupported);

        // Chin pushed forward blocks the gate.
        let pushed = project_brow_cheek_nose_channels(&snapshot_with_cheek(
            Some(0.06),
            Some(0.0),
            Some(-0.10),
        ));
        assert_eq!(pushed.cheek_puff.support, ProjectedSupport::Unsupported);
    }

    #[test]
    fn cheek_puff_missing_or_non_finite_evidence_fails_closed() {
        for snapshot in [
            snapshot_with_cheek(None, Some(0.0), Some(0.0)),
            snapshot_with_cheek(Some(0.06), None, Some(0.0)),
            snapshot_with_cheek(Some(0.06), Some(0.0), None),
            snapshot_with_cheek(Some(f32::NAN), Some(0.0), Some(0.0)),
        ] {
            let projection = project_brow_cheek_nose_channels(&snapshot);
            assert_eq!(projection.cheek_puff.support, ProjectedSupport::Unsupported);
            assert_eq!(projection.cheek_puff.value, 0.0);
        }
    }

    #[test]
    fn non_finite_brow_delta_fails_closed_per_channel() {
        let projection = project_brow_cheek_nose_channels(&snapshot_with_brows(
            brow(AnatomicalSide::Right, Some(f32::NAN), Some(0.08), None),
            brow(AnatomicalSide::Left, Some(0.08), Some(f32::NAN), Some(0.08)),
        ));
        // NaN poisons the shared inner-up mean...
        assert_eq!(
            projection.brow_inner_up.support,
            ProjectedSupport::Unsupported
        );
        assert_eq!(projection.brow_inner_up.value, 0.0);
        // ...and its own side's down channel (left lower delta is NaN).
        assert_eq!(
            projection.brow_down_left.support,
            ProjectedSupport::Unsupported
        );
        // Independent channels stay reliable.
        assert_eq!(
            projection.brow_down_right.support,
            ProjectedSupport::Reliable
        );
        assert_eq!(projection.brow_down_right.value, 1.0);
        assert_eq!(projection.brow_outer_up_left.value, 1.0);
    }

    #[test]
    fn missing_brows_fail_closed_to_unsupported_zero() {
        let projection = project_brow_cheek_nose_channels(&neutral_snapshot_with_irises(false));
        for channel in [
            projection.brow_down_left,
            projection.brow_down_right,
            projection.brow_inner_up,
            projection.brow_outer_up_left,
            projection.brow_outer_up_right,
        ] {
            assert_eq!(channel.support, ProjectedSupport::Unsupported);
            assert_eq!(channel.value, 0.0);
        }
    }

    #[test]
    fn every_brow_output_is_finite_and_within_unit_range() {
        let extreme = project_brow_cheek_nose_channels(&snapshot_with_brows(
            brow(AnatomicalSide::Right, Some(-9.0), Some(9.0), Some(-9.0)),
            brow(AnatomicalSide::Left, Some(9.0), Some(-9.0), Some(9.0)),
        ));
        for channel in [
            extreme.brow_down_left,
            extreme.brow_down_right,
            extreme.brow_inner_up,
            extreme.brow_outer_up_left,
            extreme.brow_outer_up_right,
            extreme.cheek_puff,
            extreme.cheek_squint_left,
            extreme.cheek_squint_right,
            extreme.nose_sneer_left,
            extreme.nose_sneer_right,
        ] {
            assert!(channel.value.is_finite());
            assert!((0.0..=1.0).contains(&channel.value));
        }
    }

    #[test]
    fn every_output_is_finite_and_within_unit_range() {
        let extreme = snapshot(
            aperture(AnatomicalSide::Right, -3.0),
            aperture(AnatomicalSide::Left, 12.0),
            Some(iris(AnatomicalSide::Right, || Some(9.0), || Some(-7.0))),
            Some(iris(AnatomicalSide::Left, || Some(-9.0), || Some(0.003))),
        );
        let projection = project_eye_gaze_channels(&extreme);
        for channel in [
            projection.eye_blink_left,
            projection.eye_blink_right,
            projection.eye_wide_left,
            projection.eye_wide_right,
            projection.eye_squint_left,
            projection.eye_squint_right,
            projection.eye_look_in_left,
            projection.eye_look_in_right,
            projection.eye_look_out_left,
            projection.eye_look_out_right,
            projection.eye_look_up_left,
            projection.eye_look_up_right,
            projection.eye_look_down_left,
            projection.eye_look_down_right,
        ] {
            assert!(channel.value.is_finite());
            assert!((0.0..=1.0).contains(&channel.value));
        }
    }
}
