//! VRM expression and avatar capability discovery.
//!
//! This module inspects the expression map that `bevy_vrm1` builds during VRM
//! initialization and classifies the available presets into the categories
//! used by the VTuber adapter. It does not touch morph targets directly and
//! does not apply expression weights.
//!
//! The public [`AvatarCapabilities`] snapshot is engine-neutral: it contains
//! no Bevy `Entity` IDs and no `bevy_vrm1` component types. It is built once
//! when humanoid bone binding completes and is stored in the lifecycle for
//! UI consumption.

use bevy::prelude::*;
use bevy_vrm1::prelude::ExpressionEntityMap;
use std::collections::BTreeSet;
use vtuber_core::{ARKIT52_CHANNEL_COUNT, ArkitBlendshape};

const BLINK_LEFT: &str = "blinkLeft";
const BLINK_RIGHT: &str = "blinkRight";
const BLINK: &str = "blink";

const MOUTH_PRESETS: [&str; 5] = ["aa", "ih", "ou", "ee", "oh"];

/// How the model exposes eye-blink expressions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlinkMode {
    /// No blink expression is available.
    #[default]
    None,
    /// Both `blinkLeft` and `blinkRight` are available.
    PerEye,
    /// A single combined `blink` expression is available.
    Combined,
}

/// How the model exposes mouth-shape expressions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MouthMode {
    /// No mouth expression is available.
    #[default]
    None,
    /// Only the `aa` mouth expression is available.
    AaOnly,
    /// All five vowel expressions (`aa`, `ih`, `ou`, `ee`, `oh`) are available.
    Full,
}

/// Which look-direction expressions the model exposes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LookDirectionSet {
    /// `lookLeft` is present.
    pub left: bool,
    /// `lookRight` is present.
    pub right: bool,
    /// `lookUp` is present.
    pub up: bool,
    /// `lookDown` is present.
    pub down: bool,
}

impl LookDirectionSet {
    /// Returns `true` if at least one look-direction expression is available.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.left || self.right || self.up || self.down
    }

    /// Returns `true` when all four standard LookAt expressions are present.
    #[must_use]
    pub const fn complete(&self) -> bool {
        self.left && self.right && self.up && self.down
    }
}

/// Which emotion expressions the model exposes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EmotionSet {
    /// `happy` is present.
    pub happy: bool,
    /// `angry` is present.
    pub angry: bool,
    /// `sad` is present.
    pub sad: bool,
    /// `relaxed` is present.
    pub relaxed: bool,
    /// `surprised` is present.
    pub surprised: bool,
}

impl EmotionSet {
    /// Returns `true` if at least one emotion expression is available.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.happy || self.angry || self.sad || self.relaxed || self.surprised
    }
}

/// Engine-neutral presence of humanoid bones relevant to tracking output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BonePresence {
    /// `head` bone is present (always true for a bound avatar).
    pub head: bool,
    /// `neck` bone is present.
    pub neck: bool,
    /// `leftEye` bone is present.
    pub left_eye: bool,
    /// `rightEye` bone is present.
    pub right_eye: bool,
    /// `upperChest` bone is present.
    pub upper_chest: bool,
    /// `chest` bone is present.
    pub chest: bool,
    /// `spine` bone is present.
    pub spine: bool,
}

/// Model-authored VRM LookAt backend declaration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeclaredLookAtType {
    /// Bone LookAt was declared.
    Bone,
    /// Expression LookAt was declared.
    Expression,
    /// The model has no LookAt metadata.
    #[default]
    Missing,
}

/// Single gaze backend selected for runtime application.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SelectedGazeBackend {
    /// No gaze control is available.
    #[default]
    None,
    /// Apply VRM range maps to eye bones.
    Bone,
    /// Apply VRM range maps to look-direction expressions.
    Expression,
}

/// Why the selected backend differs from complete model metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GazeFallbackReason {
    /// The model did not declare LookAt metadata.
    MetadataMissing,
    /// The declared backend was unavailable and an alternate was selected.
    DeclaredBackendUnavailable,
    /// Only a partial set of look-direction expressions exists.
    PartialExpressions,
}

/// Discovered expression capabilities for a single VRM model.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExpressionCapabilities {
    /// Available blink mode, if any.
    pub blink: BlinkMode,
    /// Available mouth mode, if any.
    pub mouth: MouthMode,
    /// Available look-direction expressions.
    pub look: LookDirectionSet,
    /// Available emotion expressions.
    pub emotions: EmotionSet,
    /// Custom or unknown expression names that are not part of the MVP preset
    /// set. They are retained for diagnostics but are not auto-mapped.
    pub unknown: Vec<String>,
}

impl ExpressionCapabilities {
    /// Builds capabilities from the `bevy_vrm1` expression map.
    ///
    /// Passing `None` produces an empty capability set, which is the correct
    /// treatment for models that do not expose an expression map.
    #[must_use]
    pub fn from_map(map: Option<&ExpressionEntityMap>) -> Self {
        let names: BTreeSet<String> = map
            .map(|m| m.0.keys().map(|expr| expr.0.clone()).collect())
            .unwrap_or_default();
        Self::from_names(&names)
    }

    /// Builds capabilities from a set of expression names.
    ///
    /// The input order does not affect the result. Names are matched against
    /// the VRM 1.0 expression preset names exactly.
    #[must_use]
    pub fn from_names(names: &BTreeSet<String>) -> Self {
        let mut caps = Self::default();
        let mut unknown = Vec::new();

        for name in names {
            let known = classify_known(name, &mut caps);
            if !known {
                unknown.push(name.clone());
            }
        }

        // Blink priority: per-eye beats combined.
        if names.contains(BLINK_LEFT) && names.contains(BLINK_RIGHT) {
            caps.blink = BlinkMode::PerEye;
        } else if names.contains(BLINK) {
            caps.blink = BlinkMode::Combined;
        }

        // Mouth priority: full five-vowel set beats aa-only.
        if MOUTH_PRESETS.iter().all(|preset| names.contains(*preset)) {
            caps.mouth = MouthMode::Full;
        } else if names.contains(MOUTH_PRESETS[0]) {
            caps.mouth = MouthMode::AaOnly;
        }

        caps.unknown = unknown;
        caps
    }

    /// Returns `true` if any blink expression is available.
    #[must_use]
    pub const fn has_blink(&self) -> bool {
        !matches!(self.blink, BlinkMode::None)
    }

    /// Returns `true` if any mouth expression is available.
    #[must_use]
    pub const fn has_mouth(&self) -> bool {
        !matches!(self.mouth, MouthMode::None)
    }
}

/// Runtime capability snapshot for ARKit52/Perfect Sync-style custom
/// expressions.
///
/// `present_channels` records names found in VRM metadata.  A channel is
/// `effective` only when the runtime resolved at least one morph bind for its
/// expression entity; a metadata-only empty preset therefore remains present
/// but is not reported as usable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PerfectSyncCapabilities {
    /// ARKit52 channels whose known names are present in the model.
    pub present_channels: [bool; ARKIT52_CHANNEL_COUNT],
    /// ARKit52 channels with at least one resolved morph bind.
    pub effective_channels: [bool; ARKIT52_CHANNEL_COUNT],
    /// Unknown custom names retained for diagnostics.
    pub unknown_names: Vec<String>,
}

impl Default for PerfectSyncCapabilities {
    fn default() -> Self {
        Self {
            present_channels: [false; ARKIT52_CHANNEL_COUNT],
            effective_channels: [false; ARKIT52_CHANNEL_COUNT],
            unknown_names: Vec::new(),
        }
    }
}

impl PerfectSyncCapabilities {
    /// Builds a synthetic capability snapshot where every known name is
    /// treated as effective.  This is useful for pure contract tests.
    #[must_use]
    pub fn from_names<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::from_named_statuses(names.into_iter().map(|name| (name, true)))
    }

    /// Builds capabilities from known names and their resolved-bind status.
    ///
    /// Unknown extra expressions are retained but ignored by the 52-channel
    /// bitsets. Duplicate aliases are deterministic: presence is retained and
    /// effectiveness is true only when at least one occurrence is effective.
    #[must_use]
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn from_named_statuses<I, S>(statuses: I) -> Self
    where
        I: IntoIterator<Item = (S, bool)>,
        S: AsRef<str>,
    {
        let mut capabilities = Self::default();
        for (name, effective) in statuses {
            let name = name.as_ref();
            let Some(channel) = ArkitBlendshape::from_name(name) else {
                capabilities.unknown_names.push(name.to_owned());
                continue;
            };
            let index = channel.index();
            capabilities.present_channels[index] = true;
            capabilities.effective_channels[index] |= effective;
        }
        capabilities.unknown_names.sort();
        capabilities.unknown_names.dedup();
        capabilities
    }

    /// Inspects a runtime expression map and resolves effectiveness from the
    /// expression entity's binding status.
    #[must_use]
    pub fn from_map_with_effective<F>(
        map: Option<&ExpressionEntityMap>,
        mut is_effective: F,
    ) -> Self
    where
        F: FnMut(Entity) -> bool,
    {
        let Some(map) = map else {
            return Self::default();
        };
        Self::from_named_statuses(
            map.0
                .iter()
                .map(|(name, &entity)| (name.0.as_str(), is_effective(entity))),
        )
    }

    /// Returns whether a channel name is present.
    #[must_use]
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub const fn is_present(&self, channel: ArkitBlendshape) -> bool {
        self.present_channels[channel.index()]
    }

    /// Returns whether a channel has an effective morph binding.
    #[must_use]
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub const fn is_effective(&self, channel: ArkitBlendshape) -> bool {
        self.effective_channels[channel.index()]
    }

    /// Returns the number of effective ARKit52 channels.
    #[must_use]
    pub fn effective_count(&self) -> usize {
        self.effective_channels
            .iter()
            .filter(|&&value| value)
            .count()
    }

    /// Returns the number of present known ARKit52 names.
    #[must_use]
    pub fn present_count(&self) -> usize {
        self.present_channels.iter().filter(|&&value| value).count()
    }

    /// Returns `true` when every required Perfect Sync channel has a resolved
    /// morph bind.
    ///
    /// The 51 channels other than `TongueOut` are required. `TongueOut` is
    /// excluded because MediaPipe never tracks the tongue, so a model without
    /// a tongue expression still drives the detailed path. A channel that is
    /// merely named in metadata is not enough; it must resolve to at least
    /// one morph bind.
    #[must_use]
    pub fn supports_perfect_sync(&self) -> bool {
        ArkitBlendshape::ALL
            .into_iter()
            .filter(|&channel| channel != ArkitBlendshape::TongueOut)
            .all(|channel| self.is_effective(channel))
    }

    /// Returns true only when all 52 canonical semantics are present.
    #[must_use]
    pub fn complete_names(&self) -> bool {
        self.present_count() == ARKIT52_CHANNEL_COUNT
    }
}

/// Public, engine-neutral snapshot of avatar capabilities for UI consumption.
///
/// This struct contains no Bevy `Entity` IDs and no `bevy_vrm1` component
/// types. It is built once when binding completes and stored in the active
/// lifecycle. The UI can display it without querying Bevy internals.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AvatarCapabilities {
    /// Presence of tracked humanoid bones.
    pub bones: BonePresence,
    /// Available blink mode.
    pub blink: BlinkMode,
    /// Available mouth mode.
    pub mouth: MouthMode,
    /// Model-authored LookAt declaration, if any.
    pub declared_look_at: DeclaredLookAtType,
    /// Exclusively selected gaze backend.
    pub gaze_backend: SelectedGazeBackend,
    /// Diagnostic fallback reason, if fallback was required.
    pub gaze_fallback: Option<GazeFallbackReason>,
    /// Available look-direction expression candidates.
    pub look_directions: LookDirectionSet,
    /// Whether the model has SpringBone chains.
    pub spring_bone: bool,
    /// Custom or unknown expression names not part of the MVP preset set.
    pub unknown_expressions: Vec<String>,
    /// Perfect Sync/ARKit52 custom-expression capability snapshot.
    pub perfect_sync: PerfectSyncCapabilities,
}

impl AvatarCapabilities {
    /// Builds a capability snapshot from engine-neutral bone presence and
    /// already-classified expression capabilities.
    ///
    /// `has_spring_bone` is supplied by the caller because detecting it
    /// requires a scene traversal that is specific to the binding system.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn from_bones_and_expression_capabilities(
        bones: BonePresence,
        expressions: &ExpressionCapabilities,
        has_spring_bone: bool,
    ) -> Self {
        Self::from_model_capabilities_with_perfect_sync(
            bones,
            expressions,
            has_spring_bone,
            DeclaredLookAtType::Missing,
            PerfectSyncCapabilities::default(),
        )
    }

    pub(crate) fn from_model_capabilities_with_perfect_sync(
        bones: BonePresence,
        expressions: &ExpressionCapabilities,
        has_spring_bone: bool,
        declared_look_at: DeclaredLookAtType,
        perfect_sync: PerfectSyncCapabilities,
    ) -> Self {
        let (gaze_backend, gaze_fallback) =
            select_gaze_backend(declared_look_at, &expressions.look, &bones);
        Self {
            bones,
            blink: expressions.blink,
            mouth: expressions.mouth,
            declared_look_at,
            gaze_backend,
            gaze_fallback,
            look_directions: expressions.look,
            spring_bone: has_spring_bone,
            unknown_expressions: expressions.unknown.clone(),
            perfect_sync,
        }
    }

    /// Returns `true` if any unknown or unsupported expression preset was found.
    #[must_use]
    pub fn has_unknown(&self) -> bool {
        !self.unknown_expressions.is_empty()
    }

    /// Returns `true` if the model supports the full MVP tracking surface:
    /// head, blink, mouth `aa`, and at least one gaze method.
    #[must_use]
    pub fn is_fully_supported(&self) -> bool {
        self.bones.head
            && self.has_blink()
            && self.mouth != MouthMode::None
            && self.gaze_backend != SelectedGazeBackend::None
    }

    /// Human-readable summary for UI display.
    ///
    /// This is derived from the machine-readable fields and does not require
    /// a Bevy query.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::with_capacity(6);

        parts.push(format!("Bones: {}", self.bone_summary()));

        parts.push(format!(
            "Blink: {}",
            match self.blink {
                BlinkMode::None => "none",
                BlinkMode::PerEye => "per-eye",
                BlinkMode::Combined => "combined",
            }
        ));

        parts.push(format!(
            "Mouth: {}",
            match self.mouth {
                MouthMode::None => "none",
                MouthMode::AaOnly => "aa only",
                MouthMode::Full => "full vowels",
            }
        ));

        parts.push(format!(
            "Gaze: {}",
            match self.gaze_backend {
                SelectedGazeBackend::None => "none",
                SelectedGazeBackend::Expression => "expression",
                SelectedGazeBackend::Bone => "eye bones",
            }
        ));

        parts.push(format!(
            "SpringBone: {}",
            if self.spring_bone { "yes" } else { "no" }
        ));

        if self.has_unknown() {
            parts.push(format!(
                "{} unknown expression(s)",
                self.unknown_expressions.len()
            ));
        }

        parts.join("; ")
    }

    fn bone_summary(&self) -> String {
        let mut bones = Vec::new();
        if self.bones.head {
            bones.push("head");
        }
        if self.bones.neck {
            bones.push("neck");
        }
        if self.bones.left_eye && self.bones.right_eye {
            bones.push("eyes");
        } else {
            if self.bones.left_eye {
                bones.push("left eye");
            }
            if self.bones.right_eye {
                bones.push("right eye");
            }
        }
        if self.bones.upper_chest {
            bones.push("upper chest");
        }
        if self.bones.chest {
            bones.push("chest");
        }
        if self.bones.spine {
            bones.push("spine");
        }

        if bones.is_empty() {
            "none".to_string()
        } else {
            bones.join(", ")
        }
    }

    /// Returns `true` if any blink expression is available.
    #[must_use]
    pub const fn has_blink(&self) -> bool {
        !matches!(self.blink, BlinkMode::None)
    }
}

/// Selects one runtime backend from model metadata and available capabilities.
#[must_use]
pub fn select_gaze_backend(
    declared: DeclaredLookAtType,
    look: &LookDirectionSet,
    bones: &BonePresence,
) -> (SelectedGazeBackend, Option<GazeFallbackReason>) {
    let has_expression = look.any();
    let complete_expression = look.complete();
    let has_eyes = bones.left_eye && bones.right_eye;
    match declared {
        DeclaredLookAtType::Bone if has_eyes => (SelectedGazeBackend::Bone, None),
        DeclaredLookAtType::Expression if has_expression => {
            let fallback = (!complete_expression).then_some(GazeFallbackReason::PartialExpressions);
            (SelectedGazeBackend::Expression, fallback)
        }
        DeclaredLookAtType::Bone | DeclaredLookAtType::Expression => {
            let alternate = if complete_expression {
                SelectedGazeBackend::Expression
            } else if has_eyes {
                SelectedGazeBackend::Bone
            } else if has_expression {
                SelectedGazeBackend::Expression
            } else {
                SelectedGazeBackend::None
            };
            (
                alternate,
                Some(GazeFallbackReason::DeclaredBackendUnavailable),
            )
        }
        DeclaredLookAtType::Missing if complete_expression => (
            SelectedGazeBackend::Expression,
            Some(GazeFallbackReason::MetadataMissing),
        ),
        DeclaredLookAtType::Missing if has_eyes => (
            SelectedGazeBackend::Bone,
            Some(GazeFallbackReason::MetadataMissing),
        ),
        DeclaredLookAtType::Missing if has_expression => (
            SelectedGazeBackend::Expression,
            Some(GazeFallbackReason::PartialExpressions),
        ),
        DeclaredLookAtType::Missing => (
            SelectedGazeBackend::None,
            Some(GazeFallbackReason::MetadataMissing),
        ),
    }
}

fn classify_known(name: &str, caps: &mut ExpressionCapabilities) -> bool {
    match name {
        BLINK_LEFT | BLINK_RIGHT | BLINK => true,
        "lookLeft" => {
            caps.look.left = true;
            true
        }
        "lookRight" => {
            caps.look.right = true;
            true
        }
        "lookUp" => {
            caps.look.up = true;
            true
        }
        "lookDown" => {
            caps.look.down = true;
            true
        }
        "happy" => {
            caps.emotions.happy = true;
            true
        }
        "angry" => {
            caps.emotions.angry = true;
            true
        }
        "sad" => {
            caps.emotions.sad = true;
            true
        }
        "relaxed" => {
            caps.emotions.relaxed = true;
            true
        }
        "surprised" => {
            caps.emotions.surprised = true;
            true
        }
        _ => MOUTH_PRESETS.contains(&name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_vrm1::prelude::VrmExpression;

    fn make_map(names: &[&str]) -> ExpressionEntityMap {
        let mut world = World::new();
        let mut map = bevy::platform::collections::HashMap::default();
        for name in names {
            let entity = world.spawn_empty().id();
            map.insert(VrmExpression::from(*name), entity);
        }
        ExpressionEntityMap(map)
    }

    #[test]
    fn expression_capabilities_empty_when_map_missing() {
        let caps = ExpressionCapabilities::from_map(None);
        assert_eq!(caps.blink, BlinkMode::None);
        assert_eq!(caps.mouth, MouthMode::None);
        assert!(!caps.look.any());
        assert!(!caps.emotions.any());
        assert!(caps.unknown.is_empty());
    }

    #[test]
    fn expression_capabilities_empty_when_no_expressions() {
        let map = make_map(&[]);
        let caps = ExpressionCapabilities::from_map(Some(&map));
        assert_eq!(caps.blink, BlinkMode::None);
        assert_eq!(caps.mouth, MouthMode::None);
        assert!(caps.unknown.is_empty());
    }

    #[test]
    fn expression_capabilities_per_eye_blink() {
        let map = make_map(&["blinkLeft", "blinkRight"]);
        let caps = ExpressionCapabilities::from_map(Some(&map));
        assert_eq!(caps.blink, BlinkMode::PerEye);
        assert!(caps.has_blink());
    }

    #[test]
    fn expression_capabilities_combined_blink() {
        let map = make_map(&["blink"]);
        let caps = ExpressionCapabilities::from_map(Some(&map));
        assert_eq!(caps.blink, BlinkMode::Combined);
    }

    #[test]
    fn expression_capabilities_per_eye_beats_combined_blink() {
        let map = make_map(&["blink", "blinkLeft", "blinkRight"]);
        let caps = ExpressionCapabilities::from_map(Some(&map));
        assert_eq!(caps.blink, BlinkMode::PerEye);
    }

    #[test]
    fn expression_capabilities_full_mouth() {
        let map = make_map(&["aa", "ih", "ou", "ee", "oh"]);
        let caps = ExpressionCapabilities::from_map(Some(&map));
        assert_eq!(caps.mouth, MouthMode::Full);
    }

    #[test]
    fn expression_capabilities_aa_only_mouth() {
        let map = make_map(&["aa"]);
        let caps = ExpressionCapabilities::from_map(Some(&map));
        assert_eq!(caps.mouth, MouthMode::AaOnly);
    }

    #[test]
    fn expression_capabilities_order_independent() {
        let a = make_map(&["oh", "aa", "ee", "ih", "ou"]);
        let b = make_map(&["aa", "ih", "ou", "ee", "oh"]);
        assert_eq!(
            ExpressionCapabilities::from_map(Some(&a)),
            ExpressionCapabilities::from_map(Some(&b))
        );
    }

    #[test]
    fn expression_capabilities_custom_is_unknown() {
        let map = make_map(&["customExpression", "aa"]);
        let caps = ExpressionCapabilities::from_map(Some(&map));
        assert_eq!(caps.mouth, MouthMode::AaOnly);
        assert_eq!(caps.unknown, vec!["customExpression"]);
    }

    #[test]
    fn perfect_sync_full_52_names_are_complete() {
        let caps = PerfectSyncCapabilities::from_names(
            ArkitBlendshape::ALL
                .into_iter()
                .map(ArkitBlendshape::canonical_name),
        );
        assert_eq!(caps.present_count(), 52);
        assert_eq!(caps.effective_count(), 52);
        assert!(caps.complete_names());
        assert!(caps.is_effective(ArkitBlendshape::TongueOut));
        assert!(caps.unknown_names.is_empty());
    }

    #[test]
    fn perfect_sync_aliases_and_unknown_names_are_deterministic() {
        let caps = PerfectSyncCapabilities::from_named_statuses([
            ("JawOpen", true),
            ("jawOpen", false),
            ("TongueOut", false),
            ("customExtra", true),
        ]);
        assert_eq!(caps.present_count(), 2);
        assert_eq!(caps.effective_count(), 1);
        assert!(caps.is_effective(ArkitBlendshape::JawOpen));
        assert!(!caps.is_effective(ArkitBlendshape::TongueOut));
        assert_eq!(caps.unknown_names, vec!["customExtra"]);
    }

    #[test]
    fn perfect_sync_map_distinguishes_present_from_empty_binding() {
        let map = make_map(&["JawOpen", "TongueOut"]);
        let jaw = *map
            .0
            .get(&VrmExpression::from("JawOpen"))
            .expect("jaw expression should be present");
        let caps =
            PerfectSyncCapabilities::from_map_with_effective(Some(&map), |entity| entity == jaw);
        assert_eq!(caps.present_count(), 2);
        assert_eq!(caps.effective_count(), 1);
        assert!(caps.is_present(ArkitBlendshape::TongueOut));
        assert!(!caps.is_effective(ArkitBlendshape::TongueOut));
    }

    fn required_perfect_sync_names() -> impl Iterator<Item = &'static str> {
        ArkitBlendshape::ALL
            .into_iter()
            .filter(|channel| *channel != ArkitBlendshape::TongueOut)
            .map(ArkitBlendshape::canonical_name)
    }

    #[test]
    fn perfect_sync_requires_all_51_channels_except_tongue_out() {
        let all_52 = PerfectSyncCapabilities::from_names(
            ArkitBlendshape::ALL
                .into_iter()
                .map(ArkitBlendshape::canonical_name),
        );
        assert!(all_52.supports_perfect_sync());

        // TongueOut is not required: a model without a tongue expression is
        // still a complete Perfect Sync model.
        let without_tongue = PerfectSyncCapabilities::from_names(required_perfect_sync_names());
        assert!(without_tongue.supports_perfect_sync());

        let missing_one = PerfectSyncCapabilities::from_names(
            required_perfect_sync_names().filter(|name| *name != "JawOpen"),
        );
        assert!(!missing_one.supports_perfect_sync());
    }

    #[test]
    fn perfect_sync_needs_resolved_binds_not_just_names() {
        let statuses = required_perfect_sync_names()
            .map(|name| (name, name != "JawOpen"))
            .collect::<Vec<_>>();
        let caps = PerfectSyncCapabilities::from_named_statuses(statuses);

        assert_eq!(caps.present_count(), 51);
        assert_eq!(caps.effective_count(), 50);
        assert!(caps.is_present(ArkitBlendshape::JawOpen));
        assert!(!caps.is_effective(ArkitBlendshape::JawOpen));
        assert!(!caps.supports_perfect_sync());
    }

    #[test]
    fn perfect_sync_partial_channel_sets_are_not_supported() {
        let eye_look = PerfectSyncCapabilities::from_names([
            "EyeLookDownLeft",
            "EyeLookDownRight",
            "EyeLookInLeft",
            "EyeLookInRight",
            "EyeLookOutLeft",
            "EyeLookOutRight",
            "EyeLookUpLeft",
            "EyeLookUpRight",
        ]);
        assert!(!eye_look.supports_perfect_sync());

        let tongue_only = PerfectSyncCapabilities::from_names(["TongueOut"]);
        assert!(!tongue_only.supports_perfect_sync());
    }

    #[test]
    fn perfect_sync_metadata_without_resolved_bind_is_not_supported() {
        let map = make_map(&["TongueOut", "EyeLookUpLeft"]);
        let caps = PerfectSyncCapabilities::from_map_with_effective(Some(&map), |_| false);

        assert_eq!(caps.present_count(), 2);
        assert_eq!(caps.effective_count(), 0);
        assert!(!caps.supports_perfect_sync());
    }

    #[test]
    fn avatar_capability_snapshot_minimal() {
        let expressions = ExpressionCapabilities::default();
        let caps = AvatarCapabilities::from_bones_and_expression_capabilities(
            BonePresence {
                head: true,
                ..BonePresence::default()
            },
            &expressions,
            false,
        );

        assert!(caps.bones.head);
        assert!(!caps.bones.neck);
        assert!(!caps.bones.left_eye);
        assert!(!caps.bones.right_eye);
        assert_eq!(caps.blink, BlinkMode::None);
        assert_eq!(caps.mouth, MouthMode::None);
        assert_eq!(caps.gaze_backend, SelectedGazeBackend::None);
        assert!(!caps.spring_bone);
        assert!(caps.unknown_expressions.is_empty());
        assert!(!caps.is_fully_supported());
    }

    #[test]
    fn avatar_capability_snapshot_full() {
        let map = make_map(&[
            "blinkLeft",
            "blinkRight",
            "aa",
            "ih",
            "ou",
            "ee",
            "oh",
            "lookLeft",
            "lookRight",
            "lookUp",
            "lookDown",
        ]);
        let expressions = ExpressionCapabilities::from_map(Some(&map));
        let bones = BonePresence {
            head: true,
            neck: true,
            left_eye: true,
            right_eye: true,
            upper_chest: true,
            chest: false,
            spine: false,
        };
        let caps =
            AvatarCapabilities::from_bones_and_expression_capabilities(bones, &expressions, true);

        assert!(caps.bones.head);
        assert!(caps.bones.neck);
        assert!(caps.bones.left_eye);
        assert!(caps.bones.right_eye);
        assert!(!caps.bones.chest);
        assert_eq!(caps.blink, BlinkMode::PerEye);
        assert_eq!(caps.mouth, MouthMode::Full);
        assert_eq!(caps.gaze_backend, SelectedGazeBackend::Expression);
        assert!(caps.look_directions.left);
        assert!(caps.spring_bone);
        assert!(caps.is_fully_supported());
    }

    #[test]
    fn avatar_capability_snapshot_unknown() {
        let map = make_map(&["customA", "aa", "customB"]);
        let expressions = ExpressionCapabilities::from_map(Some(&map));
        let bones = BonePresence {
            head: true,
            ..BonePresence::default()
        };
        let caps =
            AvatarCapabilities::from_bones_and_expression_capabilities(bones, &expressions, false);

        assert_eq!(caps.mouth, MouthMode::AaOnly);
        assert!(caps.has_unknown());
        assert_eq!(caps.unknown_expressions, vec!["customA", "customB"]);
    }

    #[test]
    fn avatar_capability_snapshot_summary() {
        let map = make_map(&["blinkLeft", "blinkRight", "aa", "lookLeft"]);
        let expressions = ExpressionCapabilities::from_map(Some(&map));
        let bones = BonePresence {
            head: true,
            neck: true,
            left_eye: true,
            right_eye: true,
            upper_chest: false,
            chest: false,
            spine: false,
        };
        let caps =
            AvatarCapabilities::from_bones_and_expression_capabilities(bones, &expressions, true);

        let summary = caps.summary();
        assert!(summary.contains("Bones: head, neck, eyes"));
        assert!(summary.contains("Blink: per-eye"));
        assert!(summary.contains("Mouth: aa only"));
        assert!(summary.contains("Gaze: eye bones"));
        assert!(summary.contains("SpringBone: yes"));
    }

    #[test]
    fn avatar_capability_snapshot_gaze_modes() {
        let no_look = ExpressionCapabilities::default();
        assert_eq!(
            AvatarCapabilities::from_bones_and_expression_capabilities(
                BonePresence {
                    head: true,
                    ..BonePresence::default()
                },
                &no_look,
                false
            )
            .gaze_backend,
            SelectedGazeBackend::None
        );

        let look = LookDirectionSet {
            left: true,
            ..LookDirectionSet::default()
        };
        let expr_caps = ExpressionCapabilities {
            look,
            ..ExpressionCapabilities::default()
        };
        assert_eq!(
            AvatarCapabilities::from_bones_and_expression_capabilities(
                BonePresence {
                    head: true,
                    ..BonePresence::default()
                },
                &expr_caps,
                false
            )
            .gaze_backend,
            SelectedGazeBackend::Expression
        );

        let bones = BonePresence {
            head: true,
            left_eye: true,
            right_eye: true,
            ..BonePresence::default()
        };
        assert_eq!(
            AvatarCapabilities::from_bones_and_expression_capabilities(bones, &expr_caps, false)
                .gaze_backend,
            SelectedGazeBackend::Bone
        );

        let no_look = ExpressionCapabilities::default();
        assert_eq!(
            AvatarCapabilities::from_bones_and_expression_capabilities(bones, &no_look, false)
                .gaze_backend,
            SelectedGazeBackend::Bone
        );
    }

    #[test]
    fn declared_backend_is_exclusive_when_available() {
        let look = LookDirectionSet {
            left: true,
            right: true,
            up: true,
            down: true,
        };
        let bones = BonePresence {
            left_eye: true,
            right_eye: true,
            ..BonePresence::default()
        };
        assert_eq!(
            select_gaze_backend(DeclaredLookAtType::Bone, &look, &bones),
            (SelectedGazeBackend::Bone, None)
        );
        assert_eq!(
            select_gaze_backend(DeclaredLookAtType::Expression, &look, &bones),
            (SelectedGazeBackend::Expression, None)
        );
    }

    #[test]
    fn broken_declared_backend_uses_one_diagnosed_alternate() {
        let complete_look = LookDirectionSet {
            left: true,
            right: true,
            up: true,
            down: true,
        };
        let no_eyes = BonePresence::default();
        assert_eq!(
            select_gaze_backend(DeclaredLookAtType::Bone, &complete_look, &no_eyes),
            (
                SelectedGazeBackend::Expression,
                Some(GazeFallbackReason::DeclaredBackendUnavailable)
            )
        );

        let eyes = BonePresence {
            left_eye: true,
            right_eye: true,
            ..BonePresence::default()
        };
        assert_eq!(
            select_gaze_backend(
                DeclaredLookAtType::Expression,
                &LookDirectionSet::default(),
                &eyes
            ),
            (
                SelectedGazeBackend::Bone,
                Some(GazeFallbackReason::DeclaredBackendUnavailable)
            )
        );
    }

    #[test]
    fn one_eye_is_not_a_bone_backend() {
        let one_eye = BonePresence {
            left_eye: true,
            right_eye: false,
            ..BonePresence::default()
        };
        let selected = select_gaze_backend(
            DeclaredLookAtType::Bone,
            &LookDirectionSet::default(),
            &one_eye,
        );
        assert_eq!(selected.0, SelectedGazeBackend::None);
        assert_eq!(
            selected.1,
            Some(GazeFallbackReason::DeclaredBackendUnavailable)
        );
    }

    #[test]
    fn missing_metadata_uses_documented_preference_and_diagnostic() {
        let look = LookDirectionSet {
            left: true,
            right: true,
            up: true,
            down: true,
        };
        let eyes = BonePresence {
            left_eye: true,
            right_eye: true,
            ..BonePresence::default()
        };
        assert_eq!(
            select_gaze_backend(DeclaredLookAtType::Missing, &look, &eyes),
            (
                SelectedGazeBackend::Expression,
                Some(GazeFallbackReason::MetadataMissing)
            )
        );
    }
}
