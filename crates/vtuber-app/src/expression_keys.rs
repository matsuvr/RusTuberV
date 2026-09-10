//! Fixed 36-key expression assignment logic.
//!
//! Keys are physical main-row digits and letters shared by JIS/US layouts.
//! The types here are pure data and pure transitions; Bevy key codes and egui
//! events are converted by the UI layer, and persistence lives in
//! [`crate::settings`].

use std::collections::BTreeMap;

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use vtuber_avatar::{AvatarExpressionCatalog, is_excluded_expression};

/// Number of assignable expression keys.
pub const EXPRESSION_KEY_COUNT: usize = 36;

/// The fixed expression key bank.
///
/// Declaration order is the assignment order: digits first, then QWERTY rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExpressionKey {
    /// `1`
    #[serde(rename = "Digit1")]
    Digit1,
    /// `2`
    #[serde(rename = "Digit2")]
    Digit2,
    /// `3`
    #[serde(rename = "Digit3")]
    Digit3,
    /// `4`
    #[serde(rename = "Digit4")]
    Digit4,
    /// `5`
    #[serde(rename = "Digit5")]
    Digit5,
    /// `6`
    #[serde(rename = "Digit6")]
    Digit6,
    /// `7`
    #[serde(rename = "Digit7")]
    Digit7,
    /// `8`
    #[serde(rename = "Digit8")]
    Digit8,
    /// `9`
    #[serde(rename = "Digit9")]
    Digit9,
    /// `0`
    #[serde(rename = "Digit0")]
    Digit0,
    /// `Q`
    #[serde(rename = "KeyQ")]
    KeyQ,
    /// `W`
    #[serde(rename = "KeyW")]
    KeyW,
    /// `E`
    #[serde(rename = "KeyE")]
    KeyE,
    /// `R`
    #[serde(rename = "KeyR")]
    KeyR,
    /// `T`
    #[serde(rename = "KeyT")]
    KeyT,
    /// `Y`
    #[serde(rename = "KeyY")]
    KeyY,
    /// `U`
    #[serde(rename = "KeyU")]
    KeyU,
    /// `I`
    #[serde(rename = "KeyI")]
    KeyI,
    /// `O`
    #[serde(rename = "KeyO")]
    KeyO,
    /// `P`
    #[serde(rename = "KeyP")]
    KeyP,
    /// `A`
    #[serde(rename = "KeyA")]
    KeyA,
    /// `S`
    #[serde(rename = "KeyS")]
    KeyS,
    /// `D`
    #[serde(rename = "KeyD")]
    KeyD,
    /// `F`
    #[serde(rename = "KeyF")]
    KeyF,
    /// `G`
    #[serde(rename = "KeyG")]
    KeyG,
    /// `H`
    #[serde(rename = "KeyH")]
    KeyH,
    /// `J`
    #[serde(rename = "KeyJ")]
    KeyJ,
    /// `K`
    #[serde(rename = "KeyK")]
    KeyK,
    /// `L`
    #[serde(rename = "KeyL")]
    KeyL,
    /// `Z`
    #[serde(rename = "KeyZ")]
    KeyZ,
    /// `X`
    #[serde(rename = "KeyX")]
    KeyX,
    /// `C`
    #[serde(rename = "KeyC")]
    KeyC,
    /// `V`
    #[serde(rename = "KeyV")]
    KeyV,
    /// `B`
    #[serde(rename = "KeyB")]
    KeyB,
    /// `N`
    #[serde(rename = "KeyN")]
    KeyN,
    /// `M`
    #[serde(rename = "KeyM")]
    KeyM,
}

impl ExpressionKey {
    /// All 36 keys in assignment order.
    pub const ALL: [ExpressionKey; EXPRESSION_KEY_COUNT] = [
        ExpressionKey::Digit1,
        ExpressionKey::Digit2,
        ExpressionKey::Digit3,
        ExpressionKey::Digit4,
        ExpressionKey::Digit5,
        ExpressionKey::Digit6,
        ExpressionKey::Digit7,
        ExpressionKey::Digit8,
        ExpressionKey::Digit9,
        ExpressionKey::Digit0,
        ExpressionKey::KeyQ,
        ExpressionKey::KeyW,
        ExpressionKey::KeyE,
        ExpressionKey::KeyR,
        ExpressionKey::KeyT,
        ExpressionKey::KeyY,
        ExpressionKey::KeyU,
        ExpressionKey::KeyI,
        ExpressionKey::KeyO,
        ExpressionKey::KeyP,
        ExpressionKey::KeyA,
        ExpressionKey::KeyS,
        ExpressionKey::KeyD,
        ExpressionKey::KeyF,
        ExpressionKey::KeyG,
        ExpressionKey::KeyH,
        ExpressionKey::KeyJ,
        ExpressionKey::KeyK,
        ExpressionKey::KeyL,
        ExpressionKey::KeyZ,
        ExpressionKey::KeyX,
        ExpressionKey::KeyC,
        ExpressionKey::KeyV,
        ExpressionKey::KeyB,
        ExpressionKey::KeyN,
        ExpressionKey::KeyM,
    ];

    /// Stable storage identifier (`Digit1`, `KeyQ`, ...).
    #[must_use]
    pub const fn storage_name(self) -> &'static str {
        match self {
            ExpressionKey::Digit1 => "Digit1",
            ExpressionKey::Digit2 => "Digit2",
            ExpressionKey::Digit3 => "Digit3",
            ExpressionKey::Digit4 => "Digit4",
            ExpressionKey::Digit5 => "Digit5",
            ExpressionKey::Digit6 => "Digit6",
            ExpressionKey::Digit7 => "Digit7",
            ExpressionKey::Digit8 => "Digit8",
            ExpressionKey::Digit9 => "Digit9",
            ExpressionKey::Digit0 => "Digit0",
            ExpressionKey::KeyQ => "KeyQ",
            ExpressionKey::KeyW => "KeyW",
            ExpressionKey::KeyE => "KeyE",
            ExpressionKey::KeyR => "KeyR",
            ExpressionKey::KeyT => "KeyT",
            ExpressionKey::KeyY => "KeyY",
            ExpressionKey::KeyU => "KeyU",
            ExpressionKey::KeyI => "KeyI",
            ExpressionKey::KeyO => "KeyO",
            ExpressionKey::KeyP => "KeyP",
            ExpressionKey::KeyA => "KeyA",
            ExpressionKey::KeyS => "KeyS",
            ExpressionKey::KeyD => "KeyD",
            ExpressionKey::KeyF => "KeyF",
            ExpressionKey::KeyG => "KeyG",
            ExpressionKey::KeyH => "KeyH",
            ExpressionKey::KeyJ => "KeyJ",
            ExpressionKey::KeyK => "KeyK",
            ExpressionKey::KeyL => "KeyL",
            ExpressionKey::KeyZ => "KeyZ",
            ExpressionKey::KeyX => "KeyX",
            ExpressionKey::KeyC => "KeyC",
            ExpressionKey::KeyV => "KeyV",
            ExpressionKey::KeyB => "KeyB",
            ExpressionKey::KeyN => "KeyN",
            ExpressionKey::KeyM => "KeyM",
        }
    }

    /// Display label (`1`, `Q`, ...). Uppercase letters do not require Shift.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            ExpressionKey::Digit1 => "1",
            ExpressionKey::Digit2 => "2",
            ExpressionKey::Digit3 => "3",
            ExpressionKey::Digit4 => "4",
            ExpressionKey::Digit5 => "5",
            ExpressionKey::Digit6 => "6",
            ExpressionKey::Digit7 => "7",
            ExpressionKey::Digit8 => "8",
            ExpressionKey::Digit9 => "9",
            ExpressionKey::Digit0 => "0",
            ExpressionKey::KeyQ => "Q",
            ExpressionKey::KeyW => "W",
            ExpressionKey::KeyE => "E",
            ExpressionKey::KeyR => "R",
            ExpressionKey::KeyT => "T",
            ExpressionKey::KeyY => "Y",
            ExpressionKey::KeyU => "U",
            ExpressionKey::KeyI => "I",
            ExpressionKey::KeyO => "O",
            ExpressionKey::KeyP => "P",
            ExpressionKey::KeyA => "A",
            ExpressionKey::KeyS => "S",
            ExpressionKey::KeyD => "D",
            ExpressionKey::KeyF => "F",
            ExpressionKey::KeyG => "G",
            ExpressionKey::KeyH => "H",
            ExpressionKey::KeyJ => "J",
            ExpressionKey::KeyK => "K",
            ExpressionKey::KeyL => "L",
            ExpressionKey::KeyZ => "Z",
            ExpressionKey::KeyX => "X",
            ExpressionKey::KeyC => "C",
            ExpressionKey::KeyV => "V",
            ExpressionKey::KeyB => "B",
            ExpressionKey::KeyN => "N",
            ExpressionKey::KeyM => "M",
        }
    }

    /// Parses a storage identifier.
    #[must_use]
    pub fn from_storage_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|key| key.storage_name() == name)
    }
}

/// One key-to-expression assignment map.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpressionBindings {
    #[serde(default)]
    keys: BTreeMap<ExpressionKey, String>,
}

impl ExpressionBindings {
    /// Builds the deterministic initial assignment from a model catalog.
    ///
    /// Uses the catalog's stable auto-assignment order and zips it onto the
    /// fixed key bank. Extra catalog candidates beyond 36 are retained by the
    /// catalog, not truncated here.
    #[must_use]
    pub fn default_for(catalog: &AvatarExpressionCatalog) -> Self {
        let mut bindings = Self::default();
        for (key, expression) in ExpressionKey::ALL
            .into_iter()
            .zip(catalog.auto_assignable_ids())
        {
            bindings.keys.insert(key, expression.to_owned());
        }
        bindings
    }

    /// Iterates assignments in key order.
    pub fn entries(&self) -> impl Iterator<Item = (ExpressionKey, &str)> {
        self.keys
            .iter()
            .map(|(key, expression)| (*key, expression.as_str()))
    }

    /// Returns the expression assigned to a key.
    #[must_use]
    pub fn expression_for(&self, key: ExpressionKey) -> Option<&str> {
        self.keys.get(&key).map(String::as_str)
    }

    /// Returns the key holding an expression, if any.
    #[must_use]
    pub fn key_for(&self, expression: &str) -> Option<ExpressionKey> {
        self.keys
            .iter()
            .find(|(_, value)| value.as_str() == expression)
            .map(|(key, _)| *key)
    }

    /// Returns `true` when nothing is assigned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Number of assigned keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Assigns one expression to one key.
    ///
    /// A single transition: the expression's old key is cleared, the key's
    /// old expression is cleared, then the new pair is inserted. The same pair
    /// is a no-op. Vacated keys are never auto-refilled.
    pub fn assign(&mut self, key: ExpressionKey, expression: impl Into<String>) {
        let expression = expression.into();
        if self.expression_for(key) == Some(expression.as_str()) {
            return;
        }
        if let Some(old_key) = self.key_for(&expression) {
            self.keys.remove(&old_key);
        }
        self.keys.remove(&key);
        self.keys.insert(key, expression);
    }

    /// Removes the expression assigned to a key.
    pub fn unassign(&mut self, key: ExpressionKey) {
        self.keys.remove(&key);
    }

    /// Removes an expression from whatever key holds it.
    pub fn unassign_expression(&mut self, expression: &str) {
        if let Some(key) = self.key_for(expression) {
            self.keys.remove(&key);
        }
    }

    /// Clears every assignment.
    pub fn clear(&mut self) {
        self.keys.clear();
    }

    /// Replaces the assignment with the catalog's deterministic default.
    pub fn reset_to_defaults(&mut self, catalog: &AvatarExpressionCatalog) {
        *self = Self::default_for(catalog);
    }

    /// Builds bindings from explicit pairs. Duplicate expressions keep the
    /// first key in iteration order; duplicates are rejected by the settings
    /// loader before this is used.
    #[must_use]
    pub fn from_pairs(pairs: impl IntoIterator<Item = (ExpressionKey, String)>) -> Self {
        let mut bindings = Self::default();
        for (key, expression) in pairs {
            if bindings.key_for(&expression).is_none() {
                bindings.keys.insert(key, expression);
            }
        }
        bindings
    }

    /// Returns `true` when the same expression appears on more than one key.
    #[must_use]
    pub fn has_duplicate_expressions(&self) -> bool {
        let mut seen = std::collections::BTreeSet::new();
        self.keys.values().any(|value| !seen.insert(value))
    }
}

/// Per-model expression bindings. A model with a present entry uses exactly
/// the user's assignment, including an explicitly empty map.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpressionBindingStore {
    models: BTreeMap<String, ExpressionBindings>,
    /// Monotonic change counter for cheap per-frame view-model gating.
    revision: u64,
}

impl ExpressionBindingStore {
    /// Returns the stored bindings for a model, if the user ever saved them.
    #[must_use]
    pub fn bindings_for(&self, model_id: &str) -> Option<&ExpressionBindings> {
        self.models.get(model_id)
    }

    /// Returns the change counter bumped by every mutation.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Stores the user's assignment for a model.
    pub fn set(&mut self, model_id: String, bindings: ExpressionBindings) {
        self.models.insert(model_id, bindings);
        self.revision = self.revision.wrapping_add(1);
    }

    /// Iterates stored models in ID order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &ExpressionBindings)> {
        self.models
            .iter()
            .map(|(model_id, bindings)| (model_id.as_str(), bindings))
    }

    /// Number of stored models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns `true` when no model has a stored entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Replaces all entries (settings restore).
    pub fn replace_entries(
        &mut self,
        entries: impl IntoIterator<Item = (String, ExpressionBindings)>,
    ) {
        self.models = entries.into_iter().collect();
        self.revision = self.revision.wrapping_add(1);
    }
}

/// Returns the effective bindings for a model: the saved entry when present
/// (including an explicitly empty one), otherwise the catalog default.
#[must_use]
pub fn effective_bindings(
    store: &ExpressionBindingStore,
    model_id: &str,
    catalog: Option<&AvatarExpressionCatalog>,
) -> ExpressionBindings {
    if let Some(bindings) = store.bindings_for(model_id) {
        return bindings.clone();
    }
    catalog.map_or_else(ExpressionBindings::default, ExpressionBindings::default_for)
}

/// Returns `true` when an expression may be newly assigned: it exists in the
/// catalog, is `Ready`, and is not the excluded tongue channel.
#[must_use]
pub fn can_assign_expression(catalog: Option<&AvatarExpressionCatalog>, expression: &str) -> bool {
    if is_excluded_expression(expression) {
        return false;
    }
    catalog.is_some_and(|catalog| {
        catalog
            .entry(expression)
            .is_some_and(|entry| entry.availability.is_ready())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_avatar::ExpressionCatalogInput;

    fn catalog_with(ids: &[&str]) -> AvatarExpressionCatalog {
        AvatarExpressionCatalog::build(
            "model".into(),
            1,
            ids.iter().map(|id| ExpressionCatalogInput {
                id,
                declared_as_preset: *id == "happy" || *id == "angry",
                declared_morph_bind_count: 1,
                resolved_morph_bind_count: 1,
                declared_material_bind_count: 0,
                unsupported_material_bind_count: 0,
            }),
        )
    }

    #[test]
    fn key_bank_is_36_unique_keys_in_the_documented_order() {
        assert_eq!(ExpressionKey::ALL.len(), EXPRESSION_KEY_COUNT);
        let unique: std::collections::BTreeSet<_> = ExpressionKey::ALL.into_iter().collect();
        assert_eq!(unique.len(), EXPRESSION_KEY_COUNT);
        assert_eq!(ExpressionKey::ALL[0].label(), "1");
        assert_eq!(ExpressionKey::ALL[9].label(), "0");
        assert_eq!(ExpressionKey::ALL[10].label(), "Q");
        assert_eq!(ExpressionKey::ALL[19].label(), "P");
        assert_eq!(ExpressionKey::ALL[20].label(), "A");
        assert_eq!(ExpressionKey::ALL[28].label(), "L");
        assert_eq!(ExpressionKey::ALL[29].label(), "Z");
        assert_eq!(ExpressionKey::ALL[35].label(), "M");
        for key in ExpressionKey::ALL {
            assert_eq!(
                ExpressionKey::from_storage_name(key.storage_name()),
                Some(key)
            );
        }
    }

    #[test]
    fn default_assignment_zips_candidates_onto_keys() {
        let catalog = catalog_with(&["happy", "zzz", "angry", "aaa"]);
        let bindings = ExpressionBindings::default_for(&catalog);
        assert_eq!(bindings.len(), 4);
        assert_eq!(
            bindings.expression_for(ExpressionKey::Digit1),
            Some("happy")
        );
        assert_eq!(
            bindings.expression_for(ExpressionKey::Digit2),
            Some("angry")
        );
        assert_eq!(bindings.expression_for(ExpressionKey::Digit3), Some("aaa"));
        assert_eq!(bindings.expression_for(ExpressionKey::Digit4), Some("zzz"));
        assert_eq!(bindings.expression_for(ExpressionKey::Digit5), None);
    }

    #[test]
    fn zero_one_four_and_overflow_candidates_are_clamped_to_36_keys() {
        assert_eq!(ExpressionBindings::default_for(&catalog_with(&[])).len(), 0);
        assert_eq!(
            ExpressionBindings::default_for(&catalog_with(&["happy"])).len(),
            1
        );
        let four = catalog_with(&["happy", "angry", "sad", "relaxed"]);
        assert_eq!(ExpressionBindings::default_for(&four).len(), 4);

        let overflow_ids: Vec<String> = (0..50).map(|index| format!("custom{index:02}")).collect();
        let overflow = AvatarExpressionCatalog::build(
            "model".into(),
            1,
            overflow_ids.iter().map(|id| ExpressionCatalogInput {
                id: id.as_str(),
                declared_as_preset: false,
                declared_morph_bind_count: 1,
                resolved_morph_bind_count: 1,
                declared_material_bind_count: 0,
                unsupported_material_bind_count: 0,
            }),
        );
        let bindings = ExpressionBindings::default_for(&overflow);
        assert_eq!(bindings.len(), 36);
        assert_eq!(
            bindings.expression_for(ExpressionKey::KeyM),
            Some("custom35")
        );
        assert_eq!(overflow.entries.len(), 50);
    }

    #[test]
    fn assign_moves_the_expression_and_vacates_the_old_key() {
        let catalog = catalog_with(&["happy", "angry"]);
        let mut bindings = ExpressionBindings::default_for(&catalog);
        assert_eq!(bindings.key_for("happy"), Some(ExpressionKey::Digit1));

        bindings.assign(ExpressionKey::KeyQ, "happy");
        assert_eq!(bindings.expression_for(ExpressionKey::Digit1), None);
        assert_eq!(bindings.key_for("happy"), Some(ExpressionKey::KeyQ));

        bindings.assign(ExpressionKey::Digit2, "happy");
        assert_eq!(
            bindings.expression_for(ExpressionKey::Digit2),
            Some("happy")
        );
        assert_eq!(bindings.expression_for(ExpressionKey::KeyQ), None);
        assert_eq!(
            bindings.expression_for(ExpressionKey::Digit1),
            None,
            "angry was unassigned, not shifted"
        );
    }

    #[test]
    fn assign_replaces_the_target_key_and_no_op_is_stable() {
        let catalog = catalog_with(&["happy", "angry"]);
        let mut bindings = ExpressionBindings::default_for(&catalog);
        let before = bindings.clone();
        bindings.assign(ExpressionKey::Digit1, "happy");
        assert_eq!(bindings, before);

        bindings.assign(ExpressionKey::Digit1, "angry");
        assert_eq!(
            bindings.expression_for(ExpressionKey::Digit1),
            Some("angry")
        );
        assert_eq!(bindings.expression_for(ExpressionKey::Digit2), None);
        assert_eq!(bindings.key_for("happy"), None);
    }

    #[test]
    fn unassign_and_reset_only_touch_the_requested_scope() {
        let catalog = catalog_with(&["happy", "angry", "sad"]);
        let mut bindings = ExpressionBindings::default_for(&catalog);
        bindings.unassign(ExpressionKey::Digit2);
        assert_eq!(bindings.expression_for(ExpressionKey::Digit2), None);
        assert_eq!(
            bindings.expression_for(ExpressionKey::Digit1),
            Some("happy")
        );

        bindings.clear();
        assert!(bindings.is_empty());
        bindings.reset_to_defaults(&catalog);
        assert_eq!(bindings.len(), 3);
        assert_eq!(
            bindings.expression_for(ExpressionKey::Digit1),
            Some("happy")
        );
    }

    #[test]
    fn vacated_keys_are_never_auto_refilled() {
        let catalog = catalog_with(&["happy", "angry", "sad", "relaxed"]);
        let mut bindings = ExpressionBindings::default_for(&catalog);
        // Simulate a reload: the default would fill Digit1..4, but the stored
        // empty first slot must stay empty.
        bindings.unassign(ExpressionKey::Digit1);
        assert_eq!(bindings.expression_for(ExpressionKey::Digit1), None);
        assert_eq!(
            bindings.expression_for(ExpressionKey::Digit2),
            Some("angry")
        );
    }

    #[test]
    fn store_distinguishes_missing_entry_from_saved_empty_entry() {
        let catalog = catalog_with(&["happy"]);
        let mut store = ExpressionBindingStore::default();
        assert!(store.bindings_for("model").is_none());
        assert_eq!(
            effective_bindings(&store, "model", Some(&catalog)).len(),
            1,
            "unsaved model uses the default"
        );

        store.set("model".into(), ExpressionBindings::default());
        assert_eq!(
            effective_bindings(&store, "model", Some(&catalog)).len(),
            0,
            "a saved empty map stays empty"
        );
    }

    #[test]
    fn model_a_and_b_bindings_do_not_share_assignments() {
        let catalog = catalog_with(&["happy"]);
        let mut store = ExpressionBindingStore::default();
        store.set("a".into(), ExpressionBindings::default_for(&catalog));
        assert!(store.bindings_for("b").is_none());
        assert_eq!(store.bindings_for("a").unwrap().len(), 1);
    }

    #[test]
    fn unicode_expression_ids_are_preserved_exactly() {
        let catalog = catalog_with(&["笑顔"]);
        let mut bindings = ExpressionBindings::default_for(&catalog);
        assert_eq!(bindings.expression_for(ExpressionKey::Digit1), Some("笑顔"));
        bindings.assign(ExpressionKey::KeyM, "笑顔");
        assert_eq!(bindings.key_for("笑顔"), Some(ExpressionKey::KeyM));
    }

    #[test]
    fn can_assign_requires_a_ready_catalog_entry_and_excludes_tongue() {
        let catalog = catalog_with(&["happy"]);
        assert!(can_assign_expression(Some(&catalog), "happy"));
        assert!(!can_assign_expression(Some(&catalog), "missing"));
        assert!(!can_assign_expression(None, "happy"));
        assert!(!can_assign_expression(Some(&catalog), "TongueOut"));
    }

    #[test]
    fn duplicate_expression_detection_is_exact() {
        let mut bindings = ExpressionBindings::default();
        bindings.assign(ExpressionKey::Digit1, "happy");
        bindings.assign(ExpressionKey::Digit2, "angry");
        assert!(!bindings.has_duplicate_expressions());
        // Directly constructing a duplicate is what the loader rejects.
        let duplicated = ExpressionBindings::from_pairs([
            (ExpressionKey::Digit1, "happy".to_string()),
            (ExpressionKey::Digit2, "happy".to_string()),
        ]);
        assert_eq!(duplicated.len(), 1, "from_pairs keeps the first key");
    }
}
