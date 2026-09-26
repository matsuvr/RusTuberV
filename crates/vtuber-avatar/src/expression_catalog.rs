//! Engine-neutral expression catalog for the active avatar.
//!
//! The catalog is built once per avatar generation from the runtime
//! expression map and its bind status. It keeps every source-declared
//! expression, classifies it, records why it can or cannot be used, and
//! provides a deterministic auto-assignment order. It contains no Bevy
//! entities and no `bevy_vrm1` types.

use bevy_vrm1::prelude::ExpressionEntityMap;
use vtuber_core::ArkitBlendshape;

use crate::expression::status::ExpressionBindingStatus;

/// Standard emotion presets in the product priority order.
pub const EMOTIONAL_PRESETS: [&str; 5] = ["happy", "angry", "sad", "relaxed", "surprised"];
/// Standard `neutral` preset.
pub const NEUTRAL_PRESET: &str = "neutral";

/// Classification of a catalog entry for auto-assignment and display.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpressionKind {
    /// Standard emotion preset (`happy`, `angry`, `sad`, `relaxed`, `surprised`).
    EmotionalPreset,
    /// Standard `neutral`.
    Neutral,
    /// Author-defined custom expression.
    Custom,
    /// Tracking expression: standard blink/mouth/LookAt or an exact ARKit52
    /// canonical/alias name.
    Tracking,
}

/// Why an expression can or cannot be selected or auto-assigned.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExpressionAvailability {
    /// At least one declared bind resolved and is supported.
    Ready,
    /// The source declared the expression with no usable binds.
    Empty,
    /// The source declared morph binds that resolved to no scene node.
    Unresolved {
        /// Bounded diagnostic reason.
        reason: String,
    },
    /// The source declared a bind the runtime cannot represent faithfully.
    Unsupported {
        /// Bounded diagnostic reason.
        reason: String,
    },
}

impl ExpressionAvailability {
    /// Returns `true` when the expression may be selected or auto-assigned.
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// One catalog entry with a stable runtime ID and its source name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExpressionCatalogEntry {
    /// Exact runtime identifier used by the expression writer.
    pub id: String,
    /// Author string for custom expressions; runtime ID for standard ones.
    pub source_name: String,
    /// Classification for auto-assignment and display.
    pub kind: ExpressionKind,
    /// Availability with reason.
    pub availability: ExpressionAvailability,
}

/// Raw bind facts used to build one catalog entry.
///
/// This is the engine-neutral projection of the runtime
/// `ExpressionBindingStatus`. Keeping it separate lets the catalog and its
/// ordering be tested without a Bevy world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpressionCatalogInput<'a> {
    /// Exact runtime ID (the expression map key).
    pub id: &'a str,
    /// `true` when declared through the standard preset section/semantics.
    pub declared_as_preset: bool,
    /// Morph binds declared by the source.
    pub declared_morph_bind_count: usize,
    /// Morph binds that resolved to a scene node.
    pub resolved_morph_bind_count: usize,
    /// Material/texture binds declared by the source.
    pub declared_material_bind_count: usize,
    /// Material/texture binds whose index resolved and whose target property
    /// is representable by the resolved material.
    pub resolved_material_bind_count: usize,
    /// Declared binds whose glTF index did not resolve to a scene material.
    pub unresolved_material_bind_count: usize,
    /// Declared binds with an unknown or unrepresentable target property.
    pub unsupported_material_bind_count: usize,
}

impl ExpressionCatalogInput<'_> {
    fn availability(&self) -> ExpressionAvailability {
        if self.unsupported_material_bind_count > 0 {
            return ExpressionAvailability::Unsupported {
                reason: format!(
                    "{} material bind(s) use an unsupported property",
                    self.unsupported_material_bind_count
                ),
            };
        }
        if self.unresolved_material_bind_count > 0 {
            return ExpressionAvailability::Unresolved {
                reason: format!(
                    "{} material bind(s) did not resolve to a scene material",
                    self.unresolved_material_bind_count
                ),
            };
        }
        if self.resolved_morph_bind_count > 0 || self.resolved_material_bind_count > 0 {
            return ExpressionAvailability::Ready;
        }
        if self.declared_morph_bind_count > 0 {
            return ExpressionAvailability::Unresolved {
                reason: "morph binds did not resolve to any scene node".into(),
            };
        }
        ExpressionAvailability::Empty
    }
}

/// Catalog of all expressions declared by one avatar generation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AvatarExpressionCatalog {
    /// Stable imported-model ID this catalog belongs to.
    pub model_id: String,
    /// Lifecycle generation this catalog was built for.
    pub generation: u64,
    /// Entries sorted by exact runtime ID (`str::cmp`).
    pub entries: Vec<ExpressionCatalogEntry>,
}

impl AvatarExpressionCatalog {
    /// Builds a catalog from raw inputs, dropping the excluded `TongueOut`
    /// expression and sorting entries by runtime ID.
    #[must_use]
    pub fn build<'a>(
        model_id: String,
        generation: u64,
        inputs: impl IntoIterator<Item = ExpressionCatalogInput<'a>>,
    ) -> Self {
        let mut entries: Vec<ExpressionCatalogEntry> = inputs
            .into_iter()
            .filter(|input| !is_excluded_expression(input.id))
            .map(|input| ExpressionCatalogEntry {
                id: input.id.to_owned(),
                source_name: input.id.to_owned(),
                kind: classify_expression(input.id, input.declared_as_preset),
                availability: input.availability(),
            })
            .collect();
        entries.sort_by(|left, right| left.id.cmp(&right.id));
        Self {
            model_id,
            generation,
            entries,
        }
    }

    /// Returns the entry for an exact runtime ID.
    #[must_use]
    pub fn entry(&self, id: &str) -> Option<&ExpressionCatalogEntry> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    /// Returns the auto-assignment candidate IDs in stable product order:
    /// `happy, angry, sad, relaxed, surprised, neutral`, then remaining
    /// custom IDs in `str::cmp` order.
    ///
    /// Tracking expressions are excluded from auto-assignment but remain in
    /// the catalog for explicit user selection.
    #[must_use]
    pub fn auto_assignable_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = Vec::with_capacity(self.entries.len());
        for preset in EMOTIONAL_PRESETS {
            if let Some(entry) = self.entries.iter().find(|entry| {
                entry.id == preset
                    && entry.kind == ExpressionKind::EmotionalPreset
                    && entry.availability.is_ready()
            }) {
                ids.push(entry.id.as_str());
            }
        }
        if let Some(entry) = self
            .entries
            .iter()
            .find(|entry| entry.kind == ExpressionKind::Neutral && entry.availability.is_ready())
        {
            ids.push(entry.id.as_str());
        }
        for entry in &self.entries {
            if entry.kind == ExpressionKind::Custom && entry.availability.is_ready() {
                ids.push(entry.id.as_str());
            }
        }
        ids
    }

    /// Returns entries a user may explicitly select: every non-excluded
    /// entry, including tracking expressions and not-ready definitions.
    #[must_use]
    pub fn selectable_entries(&self) -> &[ExpressionCatalogEntry] {
        &self.entries
    }

    /// Returns `true` when this runtime ID is a tracking expression, which
    /// suppresses automatic face commands while manually selected.
    #[must_use]
    pub fn is_tracking(&self, id: &str) -> bool {
        self.entry(id)
            .is_some_and(|entry| entry.kind == ExpressionKind::Tracking)
    }
}

/// Builds the catalog from the runtime expression map and bind statuses.
///
/// `status` resolves one expression's bind facts by its exact runtime ID; a
/// missing entry means the fact could not be established and the expression
/// is treated as empty rather than assumed effective.
#[must_use]
pub fn build_catalog(
    model_id: String,
    generation: u64,
    map: &ExpressionEntityMap,
    status: impl Fn(&str) -> Option<ExpressionBindingStatus>,
) -> AvatarExpressionCatalog {
    AvatarExpressionCatalog::build(
        model_id,
        generation,
        map.0.iter().map(|(name, _)| {
            let facts = status(name.as_str()).unwrap_or_default();
            ExpressionCatalogInput {
                id: name.0.as_str(),
                declared_as_preset: facts.declared_as_preset,
                declared_morph_bind_count: facts.declared_morph_bind_count,
                resolved_morph_bind_count: facts.resolved_morph_bind_count,
                declared_material_bind_count: facts.declared_material_bind_count,
                resolved_material_bind_count: facts.resolved_material_bind_count,
                unresolved_material_bind_count: facts.unresolved_material_bind_count,
                unsupported_material_bind_count: facts.unsupported_material_bind_count,
            }
        }),
    )
}

/// Classifies one expression by exact name and source provenance.
///
/// Exact standard tracking names and exact ARKit52 canonical/alias names are
/// `Tracking` even when they were declared as VRM custom expressions (the
/// standard Perfect Sync authoring pattern). No fuzzy name matching is used.
#[must_use]
pub fn classify_expression(id: &str, declared_as_preset: bool) -> ExpressionKind {
    if is_tracking_expression(id) {
        return ExpressionKind::Tracking;
    }
    if declared_as_preset {
        if id == NEUTRAL_PRESET {
            return ExpressionKind::Neutral;
        }
        if EMOTIONAL_PRESETS.contains(&id) {
            return ExpressionKind::EmotionalPreset;
        }
    }
    ExpressionKind::Custom
}

/// Returns `true` for standard blink/mouth/LookAt names and exact ARKit52
/// canonical or lower-camel alias names.
#[must_use]
pub fn is_tracking_expression(id: &str) -> bool {
    matches!(
        id,
        "blink"
            | "blinkLeft"
            | "blinkRight"
            | "aa"
            | "ih"
            | "ou"
            | "ee"
            | "oh"
            | "lookUp"
            | "lookDown"
            | "lookLeft"
            | "lookRight"
    ) || ArkitBlendshape::from_name(id).is_some()
}

/// Expressions excluded from the catalog entirely.
///
/// `TongueOut`/`tongueOut` are excluded because MediaPipe never tracks the
/// tongue; they must not appear in settings candidates.
#[must_use]
pub fn is_excluded_expression(id: &str) -> bool {
    id == "TongueOut" || id == "tongueOut"
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;

    fn input<'a>(id: &'a str, preset: bool, resolved: usize) -> ExpressionCatalogInput<'a> {
        ExpressionCatalogInput {
            id,
            declared_as_preset: preset,
            declared_morph_bind_count: resolved,
            resolved_morph_bind_count: resolved,
            declared_material_bind_count: 0,
            resolved_material_bind_count: 0,
            unresolved_material_bind_count: 0,
            unsupported_material_bind_count: 0,
        }
    }

    fn catalog(ids: &[(&str, bool)]) -> AvatarExpressionCatalog {
        AvatarExpressionCatalog::build(
            "model".into(),
            1,
            ids.iter()
                .map(|(id, preset)| input(id, *preset, 1))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn emotional_priority_order_is_stable() {
        let catalog = catalog(&[
            ("zzz", false),
            ("neutral", true),
            ("relaxed", true),
            ("happy", true),
            ("aaa", false),
            ("surprised", true),
            ("angry", true),
            ("sad", true),
        ]);
        assert_eq!(
            catalog.auto_assignable_ids(),
            [
                "happy",
                "angry",
                "sad",
                "relaxed",
                "surprised",
                "neutral",
                "aaa",
                "zzz"
            ]
        );
    }

    #[test]
    fn missing_standard_presets_are_skipped_not_fabricated() {
        let catalog = catalog(&[("sad", true), ("customB", false), ("customA", false)]);
        assert_eq!(catalog.auto_assignable_ids(), ["sad", "customA", "customB"]);
        assert!(catalog.entry("happy").is_none());
    }

    #[test]
    fn input_insertion_order_does_not_change_output() {
        let first = catalog(&[("b", false), ("a", false), ("happy", true)]);
        let second = catalog(&[("happy", true), ("a", false), ("b", false)]);
        assert_eq!(first.entries, second.entries);
        assert_eq!(first.auto_assignable_ids(), second.auto_assignable_ids());
    }

    #[test]
    fn unicode_custom_names_are_preserved_exactly() {
        let catalog = catalog(&[("笑顔", false), ("smile", false)]);
        assert_eq!(
            catalog.auto_assignable_ids(),
            ["smile", "笑顔"],
            "ordering is exact Rust string order"
        );
        assert_eq!(catalog.entry("笑顔").unwrap().source_name, "笑顔");
    }

    #[test]
    fn fifty_customs_are_all_retained() {
        let ids: Vec<String> = (0..50).map(|index| format!("custom{index:02}")).collect();
        let catalog = AvatarExpressionCatalog::build(
            "model".into(),
            1,
            ids.iter().map(|id| input(id, false, 1)),
        );
        assert_eq!(catalog.entries.len(), 50);
        assert_eq!(catalog.auto_assignable_ids().len(), 50);
    }

    #[test]
    fn tracking_expressions_stay_in_catalog_but_not_auto_assignment() {
        let catalog = catalog(&[
            ("happy", true),
            ("blink", true),
            ("aa", true),
            ("lookLeft", true),
            ("JawOpen", false),
            ("jawOpen", false),
            ("custom", false),
        ]);
        assert_eq!(catalog.auto_assignable_ids(), ["happy", "custom"]);
        assert!(catalog.entry("blink").is_some());
        assert_eq!(
            catalog.entry("JawOpen").unwrap().kind,
            ExpressionKind::Tracking
        );
        assert_eq!(
            catalog.entry("jawOpen").unwrap().kind,
            ExpressionKind::Tracking
        );
    }

    #[test]
    fn tongue_out_is_excluded_from_the_catalog_entirely() {
        let catalog = catalog(&[("happy", true), ("TongueOut", false), ("tongueOut", false)]);
        assert!(catalog.entry("TongueOut").is_none());
        assert!(catalog.entry("tongueOut").is_none());
    }

    #[test]
    fn custom_never_promoted_to_standard_by_name() {
        let catalog = catalog(&[("happy", false), ("neutral", false)]);
        assert_eq!(catalog.entries[0].kind, ExpressionKind::Custom);
        assert_eq!(
            catalog.entry("neutral").unwrap().kind,
            ExpressionKind::Custom
        );
    }

    fn material_input<'a>(
        id: &'a str,
        resolved: usize,
        unresolved: usize,
        unsupported: usize,
    ) -> ExpressionCatalogInput<'a> {
        ExpressionCatalogInput {
            id,
            declared_as_preset: false,
            declared_morph_bind_count: 0,
            resolved_morph_bind_count: 0,
            declared_material_bind_count: resolved + unresolved + unsupported,
            resolved_material_bind_count: resolved,
            unresolved_material_bind_count: unresolved,
            unsupported_material_bind_count: unsupported,
        }
    }

    #[test]
    fn availability_distinguishes_empty_unresolved_and_unsupported() {
        assert_eq!(
            input("a", false, 0).availability(),
            ExpressionAvailability::Empty
        );
        let declared_unresolved = ExpressionCatalogInput {
            id: "a",
            declared_as_preset: false,
            declared_morph_bind_count: 2,
            resolved_morph_bind_count: 0,
            declared_material_bind_count: 0,
            resolved_material_bind_count: 0,
            unresolved_material_bind_count: 0,
            unsupported_material_bind_count: 0,
        };
        assert!(matches!(
            declared_unresolved.availability(),
            ExpressionAvailability::Unresolved { .. }
        ));
        assert!(matches!(
            material_input("a", 0, 1, 0).availability(),
            ExpressionAvailability::Unresolved { .. }
        ));
        assert!(matches!(
            material_input("a", 0, 0, 1).availability(),
            ExpressionAvailability::Unsupported { .. }
        ));
        assert_eq!(
            material_input("a", 1, 0, 0).availability(),
            ExpressionAvailability::Ready,
            "a color-only expression with a resolved bind is Ready"
        );
    }

    #[test]
    fn unresolved_or_unsupported_binds_override_resolved_morphs() {
        let with_unresolved = ExpressionCatalogInput {
            id: "a",
            declared_as_preset: false,
            declared_morph_bind_count: 1,
            resolved_morph_bind_count: 1,
            declared_material_bind_count: 1,
            resolved_material_bind_count: 0,
            unresolved_material_bind_count: 1,
            unsupported_material_bind_count: 0,
        };
        assert!(matches!(
            with_unresolved.availability(),
            ExpressionAvailability::Unresolved { .. }
        ));
        let with_unsupported = ExpressionCatalogInput {
            id: "a",
            declared_as_preset: false,
            declared_morph_bind_count: 0,
            resolved_morph_bind_count: 0,
            declared_material_bind_count: 1,
            resolved_material_bind_count: 1,
            unresolved_material_bind_count: 0,
            unsupported_material_bind_count: 1,
        };
        assert!(matches!(
            with_unsupported.availability(),
            ExpressionAvailability::Unsupported { .. }
        ));
    }

    #[test]
    fn not_ready_entries_are_never_auto_assigned() {
        let catalog = AvatarExpressionCatalog::build(
            "model".into(),
            1,
            [
                input("happy", true, 1),
                ExpressionCatalogInput {
                    id: "broken",
                    declared_as_preset: false,
                    declared_morph_bind_count: 1,
                    resolved_morph_bind_count: 0,
                    declared_material_bind_count: 0,
                    resolved_material_bind_count: 0,
                    unresolved_material_bind_count: 0,
                    unsupported_material_bind_count: 0,
                },
            ],
        );
        assert_eq!(catalog.auto_assignable_ids(), ["happy"]);
        assert!(catalog.entry("broken").is_some());
    }
}
