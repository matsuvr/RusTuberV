//! Engine-neutral Apple ARKit face-tracking contract.
//!
//! The contract deliberately contains the 52 ARKit semantic channels rather
//! than MediaPipe's `_neutral` channel.  Avatar adapters can map these stable
//! semantics to model-specific expression names without exposing VRM or Bevy
//! types to the tracking layers.

use std::fmt;

/// Number of ARKit face-tracking channels in the fixed v1 contract.
pub const ARKIT52_CHANNEL_COUNT: usize = 52;

/// Number of ARKit face-tracking channels excluding `TongueOut`.
pub const ARKIT_NON_TONGUE_CHANNEL_COUNT: usize = 51;

/// Real bilateral non-tongue semantic pairs used by asymmetry metrics.
///
/// Mutually exclusive directional channels such as `JawLeft`/`JawRight` and
/// `MouthLeft`/`MouthRight` are deliberately excluded.
pub const ARKIT_NON_TONGUE_LEFT_RIGHT_PAIRS: &[(ArkitBlendshape, ArkitBlendshape)] = &[
    (
        ArkitBlendshape::BrowDownLeft,
        ArkitBlendshape::BrowDownRight,
    ),
    (
        ArkitBlendshape::BrowOuterUpLeft,
        ArkitBlendshape::BrowOuterUpRight,
    ),
    (
        ArkitBlendshape::CheekSquintLeft,
        ArkitBlendshape::CheekSquintRight,
    ),
    (
        ArkitBlendshape::EyeBlinkLeft,
        ArkitBlendshape::EyeBlinkRight,
    ),
    (
        ArkitBlendshape::EyeLookDownLeft,
        ArkitBlendshape::EyeLookDownRight,
    ),
    (
        ArkitBlendshape::EyeLookInLeft,
        ArkitBlendshape::EyeLookInRight,
    ),
    (
        ArkitBlendshape::EyeLookOutLeft,
        ArkitBlendshape::EyeLookOutRight,
    ),
    (
        ArkitBlendshape::EyeLookUpLeft,
        ArkitBlendshape::EyeLookUpRight,
    ),
    (
        ArkitBlendshape::EyeSquintLeft,
        ArkitBlendshape::EyeSquintRight,
    ),
    (ArkitBlendshape::EyeWideLeft, ArkitBlendshape::EyeWideRight),
    (
        ArkitBlendshape::MouthDimpleLeft,
        ArkitBlendshape::MouthDimpleRight,
    ),
    (
        ArkitBlendshape::MouthFrownLeft,
        ArkitBlendshape::MouthFrownRight,
    ),
    (
        ArkitBlendshape::MouthLowerDownLeft,
        ArkitBlendshape::MouthLowerDownRight,
    ),
    (
        ArkitBlendshape::MouthPressLeft,
        ArkitBlendshape::MouthPressRight,
    ),
    (
        ArkitBlendshape::MouthSmileLeft,
        ArkitBlendshape::MouthSmileRight,
    ),
    (
        ArkitBlendshape::MouthStretchLeft,
        ArkitBlendshape::MouthStretchRight,
    ),
    (
        ArkitBlendshape::MouthUpperUpLeft,
        ArkitBlendshape::MouthUpperUpRight,
    ),
    (
        ArkitBlendshape::NoseSneerLeft,
        ArkitBlendshape::NoseSneerRight,
    ),
];

/// Stable semantic order for the ARKit 52 blendshape channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArkitBlendshape {
    /// Left brow moves down.
    BrowDownLeft,
    /// Right brow moves down.
    BrowDownRight,
    /// Inner brows move up.
    BrowInnerUp,
    /// Left outer brow moves up.
    BrowOuterUpLeft,
    /// Right outer brow moves up.
    BrowOuterUpRight,
    /// Cheeks puff.
    CheekPuff,
    /// Left cheek squints.
    CheekSquintLeft,
    /// Right cheek squints.
    CheekSquintRight,
    /// Left eye blinks.
    EyeBlinkLeft,
    /// Right eye blinks.
    EyeBlinkRight,
    /// Left eye looks down.
    EyeLookDownLeft,
    /// Right eye looks down.
    EyeLookDownRight,
    /// Left eye looks in.
    EyeLookInLeft,
    /// Right eye looks in.
    EyeLookInRight,
    /// Left eye looks out.
    EyeLookOutLeft,
    /// Right eye looks out.
    EyeLookOutRight,
    /// Left eye looks up.
    EyeLookUpLeft,
    /// Right eye looks up.
    EyeLookUpRight,
    /// Left eye squints.
    EyeSquintLeft,
    /// Right eye squints.
    EyeSquintRight,
    /// Left eye widens.
    EyeWideLeft,
    /// Right eye widens.
    EyeWideRight,
    /// Jaw moves forward.
    JawForward,
    /// Jaw moves left.
    JawLeft,
    /// Jaw opens.
    JawOpen,
    /// Jaw moves right.
    JawRight,
    /// Mouth closes.
    MouthClose,
    /// Left mouth corner dimples.
    MouthDimpleLeft,
    /// Right mouth corner dimples.
    MouthDimpleRight,
    /// Left mouth corner frowns.
    MouthFrownLeft,
    /// Right mouth corner frowns.
    MouthFrownRight,
    /// Mouth funnels.
    MouthFunnel,
    /// Mouth moves left.
    MouthLeft,
    /// Left lower lip moves down.
    MouthLowerDownLeft,
    /// Right lower lip moves down.
    MouthLowerDownRight,
    /// Left mouth presses.
    MouthPressLeft,
    /// Right mouth presses.
    MouthPressRight,
    /// Mouth puckers.
    MouthPucker,
    /// Mouth moves right.
    MouthRight,
    /// Lower lip rolls inward.
    MouthRollLower,
    /// Upper lip rolls inward.
    MouthRollUpper,
    /// Lower lip shrugs.
    MouthShrugLower,
    /// Upper lip shrugs.
    MouthShrugUpper,
    /// Left mouth corner smiles.
    MouthSmileLeft,
    /// Right mouth corner smiles.
    MouthSmileRight,
    /// Left mouth stretches.
    MouthStretchLeft,
    /// Right mouth stretches.
    MouthStretchRight,
    /// Left upper lip moves up.
    MouthUpperUpLeft,
    /// Right upper lip moves up.
    MouthUpperUpRight,
    /// Left nostril sneers.
    NoseSneerLeft,
    /// Right nostril sneers.
    NoseSneerRight,
    /// Tongue protrudes.
    TongueOut,
}

impl ArkitBlendshape {
    /// All channels in their stable serialized order.
    pub const ALL: [Self; ARKIT52_CHANNEL_COUNT] = [
        Self::BrowDownLeft,
        Self::BrowDownRight,
        Self::BrowInnerUp,
        Self::BrowOuterUpLeft,
        Self::BrowOuterUpRight,
        Self::CheekPuff,
        Self::CheekSquintLeft,
        Self::CheekSquintRight,
        Self::EyeBlinkLeft,
        Self::EyeBlinkRight,
        Self::EyeLookDownLeft,
        Self::EyeLookDownRight,
        Self::EyeLookInLeft,
        Self::EyeLookInRight,
        Self::EyeLookOutLeft,
        Self::EyeLookOutRight,
        Self::EyeLookUpLeft,
        Self::EyeLookUpRight,
        Self::EyeSquintLeft,
        Self::EyeSquintRight,
        Self::EyeWideLeft,
        Self::EyeWideRight,
        Self::JawForward,
        Self::JawLeft,
        Self::JawOpen,
        Self::JawRight,
        Self::MouthClose,
        Self::MouthDimpleLeft,
        Self::MouthDimpleRight,
        Self::MouthFrownLeft,
        Self::MouthFrownRight,
        Self::MouthFunnel,
        Self::MouthLeft,
        Self::MouthLowerDownLeft,
        Self::MouthLowerDownRight,
        Self::MouthPressLeft,
        Self::MouthPressRight,
        Self::MouthPucker,
        Self::MouthRight,
        Self::MouthRollLower,
        Self::MouthRollUpper,
        Self::MouthShrugLower,
        Self::MouthShrugUpper,
        Self::MouthSmileLeft,
        Self::MouthSmileRight,
        Self::MouthStretchLeft,
        Self::MouthStretchRight,
        Self::MouthUpperUpLeft,
        Self::MouthUpperUpRight,
        Self::NoseSneerLeft,
        Self::NoseSneerRight,
        Self::TongueOut,
    ];

    /// Returns the stable zero-based channel index.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Returns the canonical PascalCase name used by the avatar boundary.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::BrowDownLeft => "BrowDownLeft",
            Self::BrowDownRight => "BrowDownRight",
            Self::BrowInnerUp => "BrowInnerUp",
            Self::BrowOuterUpLeft => "BrowOuterUpLeft",
            Self::BrowOuterUpRight => "BrowOuterUpRight",
            Self::CheekPuff => "CheekPuff",
            Self::CheekSquintLeft => "CheekSquintLeft",
            Self::CheekSquintRight => "CheekSquintRight",
            Self::EyeBlinkLeft => "EyeBlinkLeft",
            Self::EyeBlinkRight => "EyeBlinkRight",
            Self::EyeLookDownLeft => "EyeLookDownLeft",
            Self::EyeLookDownRight => "EyeLookDownRight",
            Self::EyeLookInLeft => "EyeLookInLeft",
            Self::EyeLookInRight => "EyeLookInRight",
            Self::EyeLookOutLeft => "EyeLookOutLeft",
            Self::EyeLookOutRight => "EyeLookOutRight",
            Self::EyeLookUpLeft => "EyeLookUpLeft",
            Self::EyeLookUpRight => "EyeLookUpRight",
            Self::EyeSquintLeft => "EyeSquintLeft",
            Self::EyeSquintRight => "EyeSquintRight",
            Self::EyeWideLeft => "EyeWideLeft",
            Self::EyeWideRight => "EyeWideRight",
            Self::JawForward => "JawForward",
            Self::JawLeft => "JawLeft",
            Self::JawOpen => "JawOpen",
            Self::JawRight => "JawRight",
            Self::MouthClose => "MouthClose",
            Self::MouthDimpleLeft => "MouthDimpleLeft",
            Self::MouthDimpleRight => "MouthDimpleRight",
            Self::MouthFrownLeft => "MouthFrownLeft",
            Self::MouthFrownRight => "MouthFrownRight",
            Self::MouthFunnel => "MouthFunnel",
            Self::MouthLeft => "MouthLeft",
            Self::MouthLowerDownLeft => "MouthLowerDownLeft",
            Self::MouthLowerDownRight => "MouthLowerDownRight",
            Self::MouthPressLeft => "MouthPressLeft",
            Self::MouthPressRight => "MouthPressRight",
            Self::MouthPucker => "MouthPucker",
            Self::MouthRight => "MouthRight",
            Self::MouthRollLower => "MouthRollLower",
            Self::MouthRollUpper => "MouthRollUpper",
            Self::MouthShrugLower => "MouthShrugLower",
            Self::MouthShrugUpper => "MouthShrugUpper",
            Self::MouthSmileLeft => "MouthSmileLeft",
            Self::MouthSmileRight => "MouthSmileRight",
            Self::MouthStretchLeft => "MouthStretchLeft",
            Self::MouthStretchRight => "MouthStretchRight",
            Self::MouthUpperUpLeft => "MouthUpperUpLeft",
            Self::MouthUpperUpRight => "MouthUpperUpRight",
            Self::NoseSneerLeft => "NoseSneerLeft",
            Self::NoseSneerRight => "NoseSneerRight",
            Self::TongueOut => "TongueOut",
        }
    }

    /// Returns the explicitly supported lower-camel ecosystem alias.
    #[must_use]
    pub const fn lower_camel_alias(self) -> &'static str {
        mediapipe_alias(self)
    }

    /// Resolves the canonical name and the explicitly supported ecosystem aliases.
    ///
    /// Matching is intentionally exact.  In particular, arbitrary
    /// case-insensitive matching and MediaPipe's `_neutral` channel are not
    /// accepted as ARKit semantics.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|&channel| channel.canonical_name() == name || mediapipe_alias(channel) == name)
    }
}

const fn mediapipe_alias(channel: ArkitBlendshape) -> &'static str {
    // All canonical names are ASCII and begin with one uppercase letter.
    // Keep the alias table explicit in source rather than accepting arbitrary
    // case variants.
    match channel {
        ArkitBlendshape::BrowDownLeft => "browDownLeft",
        ArkitBlendshape::BrowDownRight => "browDownRight",
        ArkitBlendshape::BrowInnerUp => "browInnerUp",
        ArkitBlendshape::BrowOuterUpLeft => "browOuterUpLeft",
        ArkitBlendshape::BrowOuterUpRight => "browOuterUpRight",
        ArkitBlendshape::CheekPuff => "cheekPuff",
        ArkitBlendshape::CheekSquintLeft => "cheekSquintLeft",
        ArkitBlendshape::CheekSquintRight => "cheekSquintRight",
        ArkitBlendshape::EyeBlinkLeft => "eyeBlinkLeft",
        ArkitBlendshape::EyeBlinkRight => "eyeBlinkRight",
        ArkitBlendshape::EyeLookDownLeft => "eyeLookDownLeft",
        ArkitBlendshape::EyeLookDownRight => "eyeLookDownRight",
        ArkitBlendshape::EyeLookInLeft => "eyeLookInLeft",
        ArkitBlendshape::EyeLookInRight => "eyeLookInRight",
        ArkitBlendshape::EyeLookOutLeft => "eyeLookOutLeft",
        ArkitBlendshape::EyeLookOutRight => "eyeLookOutRight",
        ArkitBlendshape::EyeLookUpLeft => "eyeLookUpLeft",
        ArkitBlendshape::EyeLookUpRight => "eyeLookUpRight",
        ArkitBlendshape::EyeSquintLeft => "eyeSquintLeft",
        ArkitBlendshape::EyeSquintRight => "eyeSquintRight",
        ArkitBlendshape::EyeWideLeft => "eyeWideLeft",
        ArkitBlendshape::EyeWideRight => "eyeWideRight",
        ArkitBlendshape::JawForward => "jawForward",
        ArkitBlendshape::JawLeft => "jawLeft",
        ArkitBlendshape::JawOpen => "jawOpen",
        ArkitBlendshape::JawRight => "jawRight",
        ArkitBlendshape::MouthClose => "mouthClose",
        ArkitBlendshape::MouthDimpleLeft => "mouthDimpleLeft",
        ArkitBlendshape::MouthDimpleRight => "mouthDimpleRight",
        ArkitBlendshape::MouthFrownLeft => "mouthFrownLeft",
        ArkitBlendshape::MouthFrownRight => "mouthFrownRight",
        ArkitBlendshape::MouthFunnel => "mouthFunnel",
        ArkitBlendshape::MouthLeft => "mouthLeft",
        ArkitBlendshape::MouthLowerDownLeft => "mouthLowerDownLeft",
        ArkitBlendshape::MouthLowerDownRight => "mouthLowerDownRight",
        ArkitBlendshape::MouthPressLeft => "mouthPressLeft",
        ArkitBlendshape::MouthPressRight => "mouthPressRight",
        ArkitBlendshape::MouthPucker => "mouthPucker",
        ArkitBlendshape::MouthRight => "mouthRight",
        ArkitBlendshape::MouthRollLower => "mouthRollLower",
        ArkitBlendshape::MouthRollUpper => "mouthRollUpper",
        ArkitBlendshape::MouthShrugLower => "mouthShrugLower",
        ArkitBlendshape::MouthShrugUpper => "mouthShrugUpper",
        ArkitBlendshape::MouthSmileLeft => "mouthSmileLeft",
        ArkitBlendshape::MouthSmileRight => "mouthSmileRight",
        ArkitBlendshape::MouthStretchLeft => "mouthStretchLeft",
        ArkitBlendshape::MouthStretchRight => "mouthStretchRight",
        ArkitBlendshape::MouthUpperUpLeft => "mouthUpperUpLeft",
        ArkitBlendshape::MouthUpperUpRight => "mouthUpperUpRight",
        ArkitBlendshape::NoseSneerLeft => "noseSneerLeft",
        ArkitBlendshape::NoseSneerRight => "noseSneerRight",
        ArkitBlendshape::TongueOut => "tongueOut",
    }
}

/// A validated fixed-size ARKit52 coefficient vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arkit52Coefficients([f32; ARKIT52_CHANNEL_COUNT]);

impl Default for Arkit52Coefficients {
    fn default() -> Self {
        Self([0.0; ARKIT52_CHANNEL_COUNT])
    }
}

impl Arkit52Coefficients {
    /// Constructs coefficients after validating finiteness and `[0, 1]` bounds.
    pub fn try_from_array(values: [f32; ARKIT52_CHANNEL_COUNT]) -> Result<Self, Arkit52ValueError> {
        for (channel, value) in ArkitBlendshape::ALL.into_iter().zip(values) {
            validate_value(channel, value)?;
        }
        Ok(Self(values))
    }

    /// Constructs a complete vector from named canonical or known alias pairs.
    ///
    /// Every semantic must occur exactly once.  Unknown names, duplicate
    /// semantic aliases, missing semantics, and invalid values are rejected.
    // Invariant: `ArkitBlendshape::index()` is always `< ARKIT52_CHANNEL_COUNT`,
    // so every index below is in bounds by construction.
    #[allow(clippy::indexing_slicing)]
    pub fn try_from_named<I, S>(pairs: I) -> Result<Self, Arkit52NameError>
    where
        I: IntoIterator<Item = (S, f32)>,
        S: Into<String>,
    {
        let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
        let mut seen = [false; ARKIT52_CHANNEL_COUNT];
        for (name, value) in pairs {
            let name = name.into();
            let Some(channel) = ArkitBlendshape::from_name(&name) else {
                return Err(Arkit52NameError::UnknownName(name));
            };
            let index = channel.index();
            if seen[index] {
                return Err(Arkit52NameError::DuplicateName(name));
            }
            validate_value(channel, value).map_err(Arkit52NameError::InvalidValue)?;
            seen[index] = true;
            values[index] = value;
        }
        let missing = ArkitBlendshape::ALL
            .into_iter()
            .enumerate()
            .filter_map(|(index, channel)| (!seen[index]).then_some(channel))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(Arkit52NameError::MissingNames(missing));
        }
        Ok(Self(values))
    }

    /// Returns the validated value for one semantic.
    // Invariant: `channel.index() < ARKIT52_CHANNEL_COUNT` for every variant.
    #[allow(clippy::indexing_slicing)]
    #[must_use]
    pub const fn get(&self, channel: ArkitBlendshape) -> f32 {
        self.0[channel.index()]
    }

    /// Returns the fixed-size coefficient array.
    #[must_use]
    pub const fn as_array(&self) -> &[f32; ARKIT52_CHANNEL_COUNT] {
        &self.0
    }

    /// Consumes the vector and returns its fixed-size array.
    #[must_use]
    pub const fn into_array(self) -> [f32; ARKIT52_CHANNEL_COUNT] {
        self.0
    }
}

/// Extracts the stable ARKit channel order without `TongueOut`.
#[must_use]
pub fn arkit_non_tongue_values(
    coefficients: &Arkit52Coefficients,
) -> [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT] {
    let mut values = [0.0; ARKIT_NON_TONGUE_CHANNEL_COUNT];
    values.copy_from_slice(&coefficients.as_array()[..ARKIT_NON_TONGUE_CHANNEL_COUNT]);
    values
}

/// Restores the ARKit52 compatibility boundary with `TongueOut` fixed to zero.
///
/// # Errors
///
/// Returns [`Arkit52ValueError`] when a non-tongue value is not finite or is
/// outside `[0, 1]`.
pub fn arkit52_with_zero_tongue(
    values: [f32; ARKIT_NON_TONGUE_CHANNEL_COUNT],
) -> Result<Arkit52Coefficients, Arkit52ValueError> {
    let mut coefficients = [0.0; ARKIT52_CHANNEL_COUNT];
    coefficients[..ARKIT_NON_TONGUE_CHANNEL_COUNT].copy_from_slice(&values);
    Arkit52Coefficients::try_from_array(coefficients)
}

fn validate_value(channel: ArkitBlendshape, value: f32) -> Result<(), Arkit52ValueError> {
    if !value.is_finite() {
        return Err(Arkit52ValueError::NonFinite { channel, value });
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(Arkit52ValueError::OutOfRange { channel, value });
    }
    Ok(())
}

/// Value validation error for an ARKit52 coefficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arkit52ValueError {
    /// The coefficient is NaN or infinite.
    NonFinite {
        /// Semantic whose value was rejected.
        channel: ArkitBlendshape,
        /// Rejected value.
        value: f32,
    },
    /// The coefficient is outside `[0, 1]`.
    OutOfRange {
        /// Semantic whose value was rejected.
        channel: ArkitBlendshape,
        /// Rejected value.
        value: f32,
    },
}

impl fmt::Display for Arkit52ValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { channel, value } => {
                write!(
                    formatter,
                    "{} value {value} is not finite",
                    channel.canonical_name()
                )
            }
            Self::OutOfRange { channel, value } => write!(
                formatter,
                "{} value {value} is outside [0, 1]",
                channel.canonical_name()
            ),
        }
    }
}

impl std::error::Error for Arkit52ValueError {}

/// Named-vector construction error.
#[derive(Clone, Debug, PartialEq)]
pub enum Arkit52NameError {
    /// A name is not a canonical semantic or explicit alias.
    UnknownName(String),
    /// Two names resolve to the same semantic.
    DuplicateName(String),
    /// One or more stable semantics were omitted.
    MissingNames(Vec<ArkitBlendshape>),
    /// A named value failed numeric validation.
    InvalidValue(Arkit52ValueError),
}

impl fmt::Display for Arkit52NameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownName(name) => write!(formatter, "unknown ARKit52 name {name:?}"),
            Self::DuplicateName(name) => write!(formatter, "duplicate ARKit52 name {name:?}"),
            Self::MissingNames(names) => write!(formatter, "missing ARKit52 names: {names:?}"),
            Self::InvalidValue(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Arkit52NameError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_has_exactly_52_stable_entries() {
        assert_eq!(ArkitBlendshape::ALL.len(), ARKIT52_CHANNEL_COUNT);
        assert_eq!(ArkitBlendshape::TongueOut.index(), 51);
    }

    #[test]
    fn bilateral_pairs_are_real_non_tongue_semantics() {
        assert!(ARKIT_NON_TONGUE_LEFT_RIGHT_PAIRS.contains(&(
            ArkitBlendshape::EyeBlinkLeft,
            ArkitBlendshape::EyeBlinkRight
        )));
        assert!(
            !ARKIT_NON_TONGUE_LEFT_RIGHT_PAIRS
                .contains(&(ArkitBlendshape::JawLeft, ArkitBlendshape::JawRight))
        );
        assert!(
            ARKIT_NON_TONGUE_LEFT_RIGHT_PAIRS
                .iter()
                .all(|(left, right)| *left != ArkitBlendshape::TongueOut
                    && *right != ArkitBlendshape::TongueOut)
        );
    }

    #[test]
    fn non_tongue_boundary_excludes_and_zeroes_tongue_out() {
        let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = 0.4;
        values[ArkitBlendshape::TongueOut.index()] = 0.9;
        let coefficients = Arkit52Coefficients::try_from_array(values).expect("valid");

        let non_tongue = arkit_non_tongue_values(&coefficients);
        assert_eq!(non_tongue.len(), ARKIT_NON_TONGUE_CHANNEL_COUNT);
        assert_eq!(non_tongue[ArkitBlendshape::JawOpen.index()], 0.4);

        let restored = arkit52_with_zero_tongue(non_tongue).expect("valid");
        assert_eq!(restored.get(ArkitBlendshape::JawOpen), 0.4);
        assert_eq!(restored.get(ArkitBlendshape::TongueOut), 0.0);
    }

    #[test]
    fn canonical_names_round_trip() {
        for channel in ArkitBlendshape::ALL {
            assert_eq!(
                ArkitBlendshape::from_name(channel.canonical_name()),
                Some(channel)
            );
        }
    }

    #[test]
    fn known_pascal_and_mediapipe_aliases_resolve() {
        assert_eq!(
            ArkitBlendshape::from_name("JawOpen"),
            Some(ArkitBlendshape::JawOpen)
        );
        assert_eq!(
            ArkitBlendshape::from_name("jawOpen"),
            Some(ArkitBlendshape::JawOpen)
        );
        assert_eq!(
            ArkitBlendshape::from_name("TongueOut"),
            Some(ArkitBlendshape::TongueOut)
        );
        assert_eq!(
            ArkitBlendshape::from_name("tongueOut"),
            Some(ArkitBlendshape::TongueOut)
        );
        assert_eq!(ArkitBlendshape::from_name("_neutral"), None);
        assert_eq!(ArkitBlendshape::from_name("jawopen"), None);
    }

    #[test]
    fn named_constructor_reports_duplicate_missing_unknown_and_invalid_values() {
        let duplicate = Arkit52Coefficients::try_from_named([("JawOpen", 0.2), ("jawOpen", 0.3)])
            .expect_err("aliases must not silently overwrite");
        assert!(matches!(duplicate, Arkit52NameError::DuplicateName(_)));

        let missing = Arkit52Coefficients::try_from_named([("JawOpen", 0.2)])
            .expect_err("partial vectors must be rejected");
        assert!(matches!(missing, Arkit52NameError::MissingNames(_)));

        let unknown = Arkit52Coefficients::try_from_named(
            ArkitBlendshape::ALL
                .into_iter()
                .map(|channel| (channel.canonical_name(), 0.0))
                .chain([("_neutral", 0.0)]),
        )
        .expect_err("MediaPipe neutral is not ARKit52");
        assert!(matches!(unknown, Arkit52NameError::UnknownName(name) if name == "_neutral"));

        let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::JawOpen.index()] = f32::NAN;
        assert!(matches!(
            Arkit52Coefficients::try_from_array(values),
            Err(Arkit52ValueError::NonFinite { .. })
        ));
    }

    #[test]
    fn tongue_out_is_a_valid_bounded_channel() {
        let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
        values[ArkitBlendshape::TongueOut.index()] = 1.0;
        let coefficients = Arkit52Coefficients::try_from_array(values).expect("valid bounds");
        assert_eq!(coefficients.get(ArkitBlendshape::TongueOut), 1.0);
    }
}
