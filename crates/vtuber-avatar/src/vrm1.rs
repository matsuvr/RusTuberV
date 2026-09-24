//! VRM 1.0 managed-copy adaptation for the unmodified upstream runtime.
//!
//! The adopted upstream (`bevy_vrm1` v0.9.3) reads only
//! `VRMC_vrm.expressions.preset` and requires `isBinary`, `overrideBlink`,
//! `overrideLookAt`, and `overrideMouth` on every entry without serde
//! defaults, while the VRM 1.0 specification makes all of them optional
//! (defaults `false` / `"none"`) and keeps author-defined expressions in a
//! separate `custom` map. The first port of #91 passed VRM 1.0 sources
//! through unchanged, so spec-conforming files failed to load or lost their
//! custom expressions entirely.
//!
//! This module rewrites the managed copy (never the source file) into the
//! shape the upstream contract accepts, with the same semantics the vendored
//! runtime previously provided:
//!
//! - an empty `preset` object is created when only `custom` exists, so the
//!   upstream `Expressions` deserialization cannot fail on a missing field;
//! - author-defined `custom` expressions are merged into `preset` so the
//!   upstream registry registers them. A custom name that collides with a
//!   standard preset keeps the standard (matching the vendored behavior) and
//!   is removed from the retained `custom` section, so "in `custom`" remains
//!   an exact custom-origin record;
//! - a merged custom carries the managed-copy marker
//!   [`MERGED_CUSTOM_MARKER`] in the retained `custom` section. Re-running
//!   the adaptation therefore keeps that origin record instead of treating
//!   the entry as a source preset/custom collision, which makes the
//!   adaptation idempotent;
//! - omitted `isBinary` / `override*` fields are filled with their
//!   specification defaults;
//! - `materialColorBinds` / `textureTransformBinds` and the `custom` section
//!   are kept in the file (upstream ignores unknown fields) so the
//!   application's source-facts parser and the app-side material writer can
//!   still see them.

use serde_json::{Map, Value};

use crate::vrm0::convert::{Vrm0ConvertError, parse_glb, repack_glb};

/// Managed-copy marker written into `custom` entries this adaptation merged
/// into `preset`.
///
/// The upstream runtime ignores the `custom` section entirely; the marker is
/// only for this adaptation, so a second pass can tell "custom copied into
/// `preset` by the managed-copy adaptation" from a genuine source
/// preset/custom collision. It is deliberately not a versioned migration
/// format.
pub const MERGED_CUSTOM_MARKER: &str = "vtuberManagedCustomOrigin";

/// Adapts `VRMC_vrm.expressions` of a VRM 1.0 GLB to the upstream contract.
///
/// Returns `Ok(None)` when the input is not a VRM 1.0 GLB or already
/// satisfies the contract, so callers can pass the source through unchanged.
/// Returns `Err` when the input cannot be parsed as a GLB.
pub fn adapt_vrm1_expressions(bytes: &[u8]) -> Result<Option<Vec<u8>>, Vrm0ConvertError> {
    let (mut document, bin) = parse_glb(bytes)?;
    let Some(vrmc) = document
        .get_mut("extensions")
        .and_then(Value::as_object_mut)
        .and_then(|extensions| extensions.get_mut("VRMC_vrm"))
        .and_then(Value::as_object_mut)
    else {
        return Ok(None);
    };
    let Some(expressions) = vrmc.get_mut("expressions").and_then(Value::as_object_mut) else {
        return Ok(None);
    };

    let preset_present = expressions.contains_key("preset");
    let mut preset: Map<String, Value> = expressions
        .get("preset")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut custom: Map<String, Value> = expressions
        .get("custom")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut changed = false;
    if !preset_present {
        // Upstream's `Expressions.preset` has no serde default; a
        // custom-only source would fail to load entirely.
        changed = true;
    }
    // Merge custom expressions into `preset` (standards win on collision,
    // matching the vendored runtime); a merged custom stays in the `custom`
    // section as the provenance record, marked so a later pass recognizes
    // its own copy. A genuine source collision (an unmarked custom sharing a
    // standard preset name) is removed from the record so the origin rule
    // stays exact.
    custom.retain(|name, entry| {
        if preset.contains_key(name) {
            if entry.get(MERGED_CUSTOM_MARKER).and_then(Value::as_bool) == Some(true) {
                return true;
            }
            changed = true;
            return false;
        }
        preset.insert(name.clone(), entry.clone());
        if let Some(object) = entry.as_object_mut() {
            object.insert(MERGED_CUSTOM_MARKER.to_string(), Value::Bool(true));
        }
        changed = true;
        true
    });
    for entry in preset.values_mut() {
        let Some(object) = entry.as_object_mut() else {
            continue;
        };
        for (field, default) in [
            ("isBinary", Value::Bool(false)),
            ("overrideBlink", Value::String("none".into())),
            ("overrideLookAt", Value::String("none".into())),
            ("overrideMouth", Value::String("none".into())),
        ] {
            if !object.contains_key(field) {
                object.insert(field.to_string(), default);
                changed = true;
            }
        }
    }

    if !changed {
        return Ok(None);
    }
    expressions.insert("preset".to_string(), Value::Object(preset));
    if custom.is_empty() {
        expressions.remove("custom");
    } else {
        expressions.insert("custom".to_string(), Value::Object(custom));
    }

    let json = serde_json::to_vec(&document)
        .map_err(|error| Vrm0ConvertError::InvalidJson(error.to_string()))?;
    Ok(Some(repack_glb(&json, bin)))
}

#[cfg(test)]
mod tests {
    // Unit tests may use unwrap/expect (AGENTS.md: Production Rust panic policy).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn glb(vrmc: Value) -> Vec<u8> {
        let document = serde_json::json!({
            "asset": {"version": "2.0"},
            "extensions": {"VRMC_vrm": vrmc},
            "extensionsUsed": ["VRMC_vrm"]
        });
        repack_glb(&serde_json::to_vec(&document).unwrap(), None)
    }

    fn json_of(bytes: &[u8]) -> Value {
        let (document, _) = parse_glb(bytes).unwrap();
        document
    }

    fn expressions_of(bytes: &[u8]) -> Value {
        json_of(bytes)["extensions"]["VRMC_vrm"]["expressions"].clone()
    }

    #[test]
    fn custom_only_expressions_are_registered_and_kept() {
        let source = glb(serde_json::json!({
            "specVersion": "1.0",
            "expressions": {"custom": {
                "smile": {"isBinary": false, "morphTargetBinds": [{"node": 0, "index": 1, "weight": 1.0}]}
            }}
        }));
        let adapted = adapt_vrm1_expressions(&source).unwrap().expect("adapted");
        let expressions = expressions_of(&adapted);
        let preset = expressions["preset"].as_object().unwrap();
        assert!(preset.contains_key("smile"), "custom must reach the preset map upstream reads");
        let custom = expressions["custom"].as_object().unwrap();
        assert!(custom.contains_key("smile"), "custom origin record is kept");
        assert_eq!(preset["smile"]["morphTargetBinds"][0]["index"], 1);
    }

    #[test]
    fn missing_optional_fields_are_filled_with_spec_defaults() {
        let source = glb(serde_json::json!({
            "specVersion": "1.0",
            "expressions": {"preset": {"aa": {"morphTargetBinds": []}}}
        }));
        let adapted = adapt_vrm1_expressions(&source).unwrap().expect("adapted");
        let entry = expressions_of(&adapted)["preset"]["aa"].clone();
        assert_eq!(entry["isBinary"], false);
        assert_eq!(entry["overrideBlink"], "none");
        assert_eq!(entry["overrideLookAt"], "none");
        assert_eq!(entry["overrideMouth"], "none");
    }

    #[test]
    fn collisions_keep_the_standard_and_the_origin_rule_stays_exact() {
        let source = glb(serde_json::json!({
            "specVersion": "1.0",
            "expressions": {
                "preset": {"happy": {"isBinary": true, "overrideBlink": "block", "overrideLookAt": "none", "overrideMouth": "none"}},
                "custom": {"happy": {"isBinary": false}}
            }
        }));
        let adapted = adapt_vrm1_expressions(&source).unwrap().expect("adapted");
        let expressions = expressions_of(&adapted);
        assert_eq!(expressions["preset"]["happy"]["isBinary"], true);
        assert!(
            expressions["preset"]["happy"]["overrideBlink"] == "block",
            "the standard entry must win on collision"
        );
        assert!(
            expressions.get("custom").is_none(),
            "the colliding custom record must not survive and blur the origin"
        );
    }

    #[test]
    fn custom_only_adaptation_is_idempotent_and_keeps_the_custom_origin() {
        let source = glb(serde_json::json!({
            "specVersion": "1.0",
            "expressions": {"custom": {
                "smile": {"morphTargetBinds": [{"node": 0, "index": 1, "weight": 1.0}]}
            }}
        }));
        let first = adapt_vrm1_expressions(&source).unwrap().expect("adapted");
        let facts = crate::expression::source::parse_source_expressions(&json_of(&first));
        assert!(!facts.entry("smile").unwrap().declared_as_preset);

        // Re-running the adaptation on its own output changes nothing: the
        // retained custom origin record is not mistaken for a genuine
        // preset/custom collision.
        assert!(
            adapt_vrm1_expressions(&first).unwrap().is_none(),
            "a second adaptation pass must be a no-op"
        );
        let facts_again = crate::expression::source::parse_source_expressions(&json_of(&first));
        assert!(!facts_again.entry("smile").unwrap().declared_as_preset);
        assert_eq!(facts, facts_again, "expression facts must not change");
    }

    #[test]
    fn already_compliant_files_pass_through_unchanged() {
        let source = glb(serde_json::json!({
            "specVersion": "1.0",
            "expressions": {"preset": {
                "aa": {"isBinary": false, "overrideBlink": "none", "overrideLookAt": "none", "overrideMouth": "none"}
            }}
        }));
        assert!(adapt_vrm1_expressions(&source).unwrap().is_none());
    }

    #[test]
    fn empty_preset_object_is_created_for_custom_only_sources() {
        // `expressions` without any `preset` key fails upstream serde; the
        // adapted file must carry an explicit empty object.
        let source = glb(serde_json::json!({
            "specVersion": "1.0",
            "expressions": {"custom": {"wink": {}}}
        }));
        let adapted = adapt_vrm1_expressions(&source).unwrap().expect("adapted");
        let expressions = expressions_of(&adapted);
        assert!(expressions["preset"].is_object());
        assert!(expressions["preset"]["wink"].is_object());
        assert_eq!(expressions["preset"]["wink"]["isBinary"], false);
    }

    #[test]
    fn non_vrm_and_expressionless_inputs_pass_through() {
        assert!(adapt_vrm1_expressions(b"not a glb").is_err());
        let no_expressions = glb(serde_json::json!({"specVersion": "1.0"}));
        assert!(adapt_vrm1_expressions(&no_expressions).unwrap().is_none());
        let (document, _) = parse_glb(&no_expressions).unwrap();
        assert!(document.get("nodes").is_none());
    }
}
