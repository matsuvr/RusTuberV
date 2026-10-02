//! VRM 0 expression identity used by inspection, diagnostics and conversion.

use serde_json::Value;

/// The runtime ID and its source meaning.
pub struct LegacyExpressionId {
    /// VRM 1 preset ID or the author's custom name.
    pub name: String,
    /// A known source preset selected a standard semantic.
    pub is_standard: bool,
    /// An unnamed custom group received its index-based ID.
    pub is_unnamed: bool,
}

/// Resolves a known preset, otherwise preserves the custom author name.
/// Unknown, absent and empty presets do not infer semantics from author names.
pub fn resolve_legacy_expression_id(group: &Value, index: usize) -> LegacyExpressionId {
    let preset = group
        .get("presetName")
        .and_then(Value::as_str)
        .map(str::trim);
    let standard = match preset {
        Some("A" | "a") => Some("aa"),
        Some("I" | "i") => Some("ih"),
        Some("U" | "u") => Some("ou"),
        Some("E" | "e") => Some("ee"),
        Some("O" | "o") => Some("oh"),
        Some("Blink" | "blink") => Some("blink"),
        Some("Blink_L" | "blink_l") => Some("blinkLeft"),
        Some("Blink_R" | "blink_r") => Some("blinkRight"),
        Some("LookUp" | "lookup") => Some("lookUp"),
        Some("LookDown" | "lookdown") => Some("lookDown"),
        Some("LookLeft" | "lookleft") => Some("lookLeft"),
        Some("LookRight" | "lookright") => Some("lookRight"),
        Some("Joy" | "joy") => Some("happy"),
        Some("Angry" | "angry") => Some("angry"),
        Some("Sorrow" | "sorrow") => Some("sad"),
        Some("Fun" | "fun") => Some("relaxed"),
        Some("Neutral" | "neutral") => Some("neutral"),
        _ => None,
    };
    if let Some(name) = standard {
        return LegacyExpressionId {
            name: name.into(),
            is_standard: true,
            is_unnamed: false,
        };
    }
    let name = group
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty());
    LegacyExpressionId {
        name: name
            .map(str::to_owned)
            .unwrap_or_else(|| format!("custom_{index}")),
        is_standard: false,
        is_unnamed: name.is_none(),
    }
}
