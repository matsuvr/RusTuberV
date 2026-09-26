//! Source-level VRM expression facts shared by import and binding.
//!
//! The unmodified upstream runtime registers expressions from
//! `VRMC_vrm.expressions.preset` only and publishes no bind counts, so the
//! application derives its own facts from the managed copy's `VRMC_vrm`
//! extension. For VRM 1.0 sources the section is read as authored; for VRM
//! 0.x sources the import conversion writes the same shape (custom-origin
//! expressions are merged into `preset` so the upstream registry registers
//! them, and also recorded under `custom` so the origin stays recoverable).
//!
//! These facts feed [`crate::expression::status::ExpressionBindingStatus`],
//! the catalog, the Perfect Sync capability check, and the app-side material
//! bind writer. The pure builder is testable without a Bevy world.

use serde_json::Value;

/// Standard VRM material color bind targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialColorTarget {
    /// `color` — the base color.
    Color,
    /// `emissionColor` — the emissive color.
    EmissionColor,
    /// `shadeColor` — the MToon shade color.
    ShadeColor,
    /// `rimColor` — the MToon parametric rim color.
    RimColor,
    /// `outlineColor` — the MToon outline color.
    OutlineColor,
}

impl MaterialColorTarget {
    /// Parses one `materialColorBinds[].type` value. Unknown types stay
    /// `None` and count as unsupported instead of being silently applied.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "color" => Some(Self::Color),
            "emissionColor" => Some(Self::EmissionColor),
            "shadeColor" => Some(Self::ShadeColor),
            "rimColor" => Some(Self::RimColor),
            "outlineColor" => Some(Self::OutlineColor),
            _ => None,
        }
    }

    /// Returns `true` when a `StandardMaterial` (unlit/plain PBR fallback)
    /// has an equivalent property. MToon represents every target.
    #[must_use]
    pub const fn is_supported_by_standard(self) -> bool {
        matches!(self, Self::Color | Self::EmissionColor)
    }
}

/// How a resolved glTF material represents color targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialKind {
    /// MToon material: every standard color target is representable.
    MToon,
    /// Plain PBR/unlit `StandardMaterial`: only base color and emission.
    Standard,
}

impl MaterialKind {
    /// Whether the target property can be applied to this material kind.
    #[must_use]
    pub const fn supports(self, target: MaterialColorTarget) -> bool {
        match self {
            Self::MToon => true,
            Self::Standard => target.is_supported_by_standard(),
        }
    }
}

/// One declared morph bind. The node name is the Bevy scene name upstream
/// resolves binds by (`gltf.json` node name, or the computed `GltfNode{i}`).
#[derive(Clone, Debug, PartialEq)]
pub struct SourceMorphBind {
    /// Resolved scene name of the bound node.
    pub node_name: String,
    /// Morph target index within the node's mesh.
    pub morph_index: usize,
    /// Bind weight in `0..=1`.
    pub weight: f32,
}

/// One declared `materialColorBinds` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceMaterialColorBind {
    /// glTF material index the bind targets.
    pub material: usize,
    /// Parsed target property; `None` for unknown (unsupported) types.
    pub target: Option<MaterialColorTarget>,
    /// Linear RGBA target value from the source.
    pub target_value: [f32; 4],
}

/// One declared `textureTransformBinds` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceTextureTransformBind {
    /// glTF material index the bind targets.
    pub material: usize,
    /// Optional scale; VRM default is `[1, 1]`.
    pub scale: Option<[f32; 2]>,
    /// Optional offset; VRM default is `[0, 0]`.
    pub offset: Option<[f32; 2]>,
}

/// Facts for one source expression definition.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceExpressionEntry {
    /// Exact runtime ID (the expression map key).
    pub name: String,
    /// `false` when the source declared the expression under the author
    /// defined `custom` map; `true` for standard preset semantics.
    pub declared_as_preset: bool,
    /// Whether weights above 0.5 snap to 1.0.
    pub is_binary: bool,
    /// Raw `overrideMouth` value (`"none"` / `"block"` / `"blend"`).
    pub override_mouth: String,
    /// Raw `overrideBlink` value.
    pub override_blink: String,
    /// Raw `overrideLookAt` value.
    pub override_look_at: String,
    /// Declared morph target binds.
    pub morph_binds: Vec<SourceMorphBind>,
    /// Declared material color binds.
    pub material_color_binds: Vec<SourceMaterialColorBind>,
    /// Declared texture transform binds.
    pub texture_transform_binds: Vec<SourceTextureTransformBind>,
}

/// Parsed `VRMC_vrm.expressions` facts for one managed model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceExpressions {
    /// One entry per declared expression (preset and custom sections).
    pub entries: Vec<SourceExpressionEntry>,
}

impl SourceExpressions {
    /// Returns the entry for an exact runtime ID.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&SourceExpressionEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }
}

fn finite_f32(value: &Value) -> Option<f32> {
    value
        .as_f64()
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
}

/// Reads one expression object (a `preset`/`custom` map entry) into facts.
///
/// `nodes` is the glTF node array of the same document, used to resolve morph
/// bind node indices to the scene names the runtime looks up.
fn parse_expression_entry(
    name: &str,
    declared_as_preset: bool,
    entry: &Value,
    nodes: &[Value],
) -> SourceExpressionEntry {
    let node_name = |node: usize| {
        nodes
            .get(node)
            .and_then(|node| node.get("name").and_then(Value::as_str))
            .map(str::to_owned)
            .unwrap_or_else(|| format!("GltfNode{node}"))
    };
    let morph_binds = entry
        .get("morphTargetBinds")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|bind| {
            let node = bind.get("node").and_then(Value::as_u64)?;
            let morph_index = bind.get("index").and_then(Value::as_u64)?;
            let weight = bind.get("weight").and_then(finite_f32)?;
            let node = usize::try_from(node).ok()?;
            let morph_index = usize::try_from(morph_index).ok()?;
            Some(SourceMorphBind {
                node_name: node_name(node),
                morph_index,
                weight,
            })
        })
        .collect();
    let material_color_binds = entry
        .get("materialColorBinds")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|bind| {
            let material = usize::try_from(bind.get("material").and_then(Value::as_u64)?).ok()?;
            let target = bind
                .get("type")
                .and_then(Value::as_str)
                .and_then(MaterialColorTarget::parse);
            let target_value = bind.get("targetValue")?;
            let red = target_value.get(0).and_then(finite_f32)?;
            let green = target_value.get(1).and_then(finite_f32)?;
            let blue = target_value.get(2).and_then(finite_f32)?;
            let alpha = target_value.get(3).and_then(finite_f32)?;
            Some(SourceMaterialColorBind {
                material,
                target,
                target_value: [red, green, blue, alpha],
            })
        })
        .collect();
    let texture_transform_binds = entry
        .get("textureTransformBinds")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|bind| {
            let material = usize::try_from(bind.get("material").and_then(Value::as_u64)?).ok()?;
            let scale = bind.get("scale").and_then(parse_vec2);
            let offset = bind.get("offset").and_then(parse_vec2);
            Some(SourceTextureTransformBind {
                material,
                scale,
                offset,
            })
        })
        .collect();
    SourceExpressionEntry {
        name: name.to_owned(),
        declared_as_preset,
        is_binary: entry
            .get("isBinary")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        override_mouth: entry
            .get("overrideMouth")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_owned(),
        override_blink: entry
            .get("overrideBlink")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_owned(),
        override_look_at: entry
            .get("overrideLookAt")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_owned(),
        morph_binds,
        material_color_binds,
        texture_transform_binds,
    }
}

fn parse_vec2(value: &Value) -> Option<[f32; 2]> {
    let x = value.get(0).and_then(finite_f32)?;
    let y = value.get(1).and_then(finite_f32)?;
    Some([x, y])
}

/// Parses `VRMC_vrm.expressions` facts out of a full glTF JSON document.
///
/// Author-defined expressions keep the exact source name and custom origin;
/// no name-based classification happens here.
#[must_use]
pub fn parse_source_expressions(document: &Value) -> SourceExpressions {
    let nodes: &[Value] = document
        .get("nodes")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let expressions = document
        .get("extensions")
        .and_then(|extensions| extensions.get("VRMC_vrm"))
        .and_then(|vrm| vrm.get("expressions"));
    let custom: std::collections::BTreeMap<String, &Value> = expressions
        .and_then(|value| value.get("custom"))
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(name, entry)| entry.as_object().map(|_| (name.clone(), entry)))
        .collect();
    let mut entries: Vec<SourceExpressionEntry> = expressions
        .and_then(|value| value.get("preset"))
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter(|(_name, entry)| entry.as_object().is_some())
        .map(|(name, entry)| {
            // The import conversion merges custom-origin expressions into
            // `preset` (the only map the upstream runtime reads) and keeps
            // the `custom` section as the provenance record. Presence there
            // decides the origin; colliding names were removed at conversion
            // time, so the rule is exact.
            parse_expression_entry(name, !custom.contains_key(name), entry, nodes)
        })
        .collect();
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    SourceExpressions { entries }
}

/// Computes per-expression bind facts from the source definitions and what
/// actually resolved into the scene.
///
/// `is_node_resolved` decides whether a morph bind's node name exists in the
/// loaded scene (upstream resolves binds by node name). `material_kind`
/// reports the resolved material kind for a glTF material index, or `None`
/// when no scene mesh uses it.
#[must_use]
pub fn build_binding_statuses(
    facts: &SourceExpressions,
    is_node_resolved: impl Fn(&str) -> bool,
    material_kind: impl Fn(usize) -> Option<MaterialKind>,
) -> std::collections::BTreeMap<String, crate::expression::status::ExpressionBindingStatus> {
    use crate::expression::status::ExpressionBindingStatus;

    facts
        .entries
        .iter()
        .map(|entry| {
            let declared_morph_bind_count = entry.morph_binds.len();
            let resolved_morph_bind_count = entry
                .morph_binds
                .iter()
                .filter(|bind| is_node_resolved(&bind.node_name))
                .count();

            let mut resolved_material_bind_count = 0;
            let mut unresolved_material_bind_count = 0;
            let mut unsupported_material_bind_count = 0;
            let mut classify_color = |material: usize, target: Option<MaterialColorTarget>| {
                match material_kind(material) {
                    None => unresolved_material_bind_count += 1,
                    Some(kind) => match target.filter(|target| kind.supports(*target)) {
                        Some(_) => resolved_material_bind_count += 1,
                        None => unsupported_material_bind_count += 1,
                    },
                }
            };
            for bind in &entry.material_color_binds {
                classify_color(bind.material, bind.target);
            }
            for bind in &entry.texture_transform_binds {
                match material_kind(bind.material) {
                    None => unresolved_material_bind_count += 1,
                    // Both material kinds carry a UV transform.
                    Some(_) => resolved_material_bind_count += 1,
                }
            }

            (
                entry.name.clone(),
                ExpressionBindingStatus {
                    resolved_morph_bind_count,
                    declared_morph_bind_count,
                    resolved_material_bind_count,
                    declared_material_bind_count: entry.material_color_binds.len()
                        + entry.texture_transform_binds.len(),
                    unresolved_material_bind_count,
                    unsupported_material_bind_count,
                    declared_as_preset: entry.declared_as_preset,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]
    use super::*;
    use crate::expression::status::ExpressionBindingStatus;

    fn document(expressions: Value) -> Value {
        serde_json::json!({
            "nodes": [
                {"name": "Body"},
                {"name": "Face"},
                {"mesh": 0}
            ],
            "extensions": {"VRMC_vrm": {"expressions": expressions}}
        })
    }

    #[test]
    fn parses_preset_and_custom_provenance() {
        // The conversion shape: custom-origin entries appear in `preset` (the
        // map the runtime reads) and under `custom` as the origin record.
        let facts = parse_source_expressions(&document(serde_json::json!({
            "preset": {
                "happy": {"morphTargetBinds": [{"node": 1, "index": 0, "weight": 1.0}]},
                "smile": {"morphTargetBinds": [{"node": 1, "index": 1, "weight": 0.5}]}
            },
            "custom": {"smile": {"morphTargetBinds": [{"node": 1, "index": 1, "weight": 0.5}]}}
        })));
        assert_eq!(facts.entries.len(), 2);
        let happy = facts.entry("happy").unwrap();
        assert!(happy.declared_as_preset);
        let smile = facts.entry("smile").unwrap();
        assert!(!smile.declared_as_preset);
        assert_eq!(smile.morph_binds[0].node_name, "Face");
    }

    #[test]
    fn custom_origin_wins_when_recorded() {
        let facts = parse_source_expressions(&document(serde_json::json!({
            "preset": {"joy": {"isBinary": true}},
            "custom": {"joy": {"isBinary": false}}
        })));
        assert!(!facts.entry("joy").unwrap().declared_as_preset);
    }

    #[test]
    fn missing_fields_use_spec_defaults() {
        let facts = parse_source_expressions(&document(serde_json::json!({
            "preset": {"aa": {}}
        })));
        let entry = facts.entry("aa").unwrap();
        assert!(!entry.is_binary);
        assert_eq!(entry.override_mouth, "none");
        assert!(entry.morph_binds.is_empty());
    }

    #[test]
    fn unnamed_nodes_use_computed_scene_names() {
        let document = serde_json::json!({
            "nodes": [{"mesh": 0}],
            "extensions": {"VRMC_vrm": {"expressions": {"preset": {"blink": {
                "morphTargetBinds": [{"node": 0, "index": 0, "weight": 1.0}]
            }}}}}
        });
        let facts = parse_source_expressions(&document);
        assert_eq!(
            facts.entry("blink").unwrap().morph_binds[0].node_name,
            "GltfNode0"
        );
    }

    #[test]
    fn unknown_material_target_is_recorded_but_unsupported() {
        let facts = parse_source_expressions(&document(serde_json::json!({
            "preset": {"weird": {"materialColorBinds": [
                {"material": 0, "type": "unknownProp", "targetValue": [1.0, 0.0, 0.0, 1.0]},
                {"material": 0, "type": "color", "targetValue": [1.0, 0.0, 0.0, 1.0]}
            ]}}
        })));
        let entry = facts.entry("weird").unwrap();
        assert_eq!(entry.material_color_binds.len(), 2);
        assert_eq!(entry.material_color_binds[0].target, None);
        assert_eq!(
            entry.material_color_binds[1].target,
            Some(MaterialColorTarget::Color)
        );
    }

    fn status(facts: &SourceExpressions) -> ExpressionBindingStatus {
        let statuses = build_binding_statuses(
            facts,
            |name| name == "Face",
            |material| (material == 0).then_some(MaterialKind::MToon),
        );
        statuses[facts.entries[0].name.as_str()]
    }

    #[test]
    fn statuses_distinguish_resolved_unresolved_and_empty() {
        let facts = parse_source_expressions(&document(serde_json::json!({
            "preset": {
                "ready": {"morphTargetBinds": [{"node": 1, "index": 0, "weight": 1.0}]},
                "unresolved": {"morphTargetBinds": [{"node": 2, "index": 0, "weight": 1.0}]}
            }
        })));
        assert_eq!(status(&facts).resolved_morph_bind_count, 1);
        assert!(facts.entry("unresolved").is_some());
        let statuses =
            build_binding_statuses(&facts, |name| name == "Face", |_| Some(MaterialKind::MToon));
        assert_eq!(statuses["unresolved"].resolved_morph_bind_count, 0);
        assert_eq!(statuses["unresolved"].declared_morph_bind_count, 1);

        let empty = parse_source_expressions(&document(serde_json::json!({
            "preset": {"hollow": {}}
        })));
        let empty_status = status(&empty);
        assert_eq!(empty_status.declared_morph_bind_count, 0);
        assert_eq!(empty_status.resolved_morph_bind_count, 0);
    }

    #[test]
    fn statuses_distinguish_material_kinds_and_unresolved_materials() {
        let facts = parse_source_expressions(&document(serde_json::json!({
            "preset": {"m": {"materialColorBinds": [
                {"material": 0, "type": "shadeColor", "targetValue": [1.0, 0.0, 0.0, 1.0]},
                {"material": 1, "type": "color", "targetValue": [1.0, 0.0, 0.0, 1.0]},
                {"material": 0, "type": "color", "targetValue": [1.0, 0.0, 0.0, 1.0]}
            ]}}
        })));
        let statuses = build_binding_statuses(
            &facts,
            |_| true,
            |material| (material == 0).then_some(MaterialKind::Standard),
        );
        let m = &statuses["m"];
        assert_eq!(m.declared_material_bind_count, 3);
        assert_eq!(m.resolved_material_bind_count, 1);
        assert_eq!(m.unresolved_material_bind_count, 1);
        assert_eq!(m.unsupported_material_bind_count, 1);
    }

    #[test]
    fn texture_transform_binds_resolve_like_material_binds() {
        let facts = parse_source_expressions(&document(serde_json::json!({
            "preset": {"uv": {"textureTransformBinds": [
                {"material": 0, "scale": [2.0, 2.0], "offset": [0.1, 0.1]},
                {"material": 7}
            ]}}
        })));
        let statuses = build_binding_statuses(
            &facts,
            |_| true,
            |material| (material == 0).then_some(MaterialKind::MToon),
        );
        let uv = &statuses["uv"];
        assert_eq!(uv.resolved_material_bind_count, 1);
        assert_eq!(uv.unresolved_material_bind_count, 1);
        assert!(uv.declared_as_preset);
    }
}
