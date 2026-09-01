//! Per-frame expression command builder.
//!
//! Converts tracking data (blink, mouth, gaze weights) into a name→weight
//! mapping suitable for `ModifyExpressions`. The builder is pure and
//! separated from Bevy event sending.

use std::collections::HashMap;

use vtuber_core::types::AvatarControlFrame;
use vtuber_core::{ARKIT52_CHANNEL_COUNT, Arkit52Coefficients, ArkitBlendshape};

use crate::capabilities::{AvatarCapabilities, PerfectSyncCapabilities};

/// A single expression command: name → weight.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpressionCommand {
    /// Expression name (e.g. "blink", "aa", "lookLeft").
    pub name: String,
    /// Weight in [0, 1].
    pub weight: f32,
}

/// Metrics for the expression command builder.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExpressionCommandMetrics {
    /// Number of non-finite weights dropped.
    pub non_finite_dropped: u64,
    /// Number of out-of-range weights clamped.
    pub out_of_range_clamped: u64,
    /// Number of duplicate names suppressed.
    pub duplicates_suppressed: u64,
    /// Number of empty commands (no expressions to send).
    pub empty_commands: u64,
}

/// Pure builder that converts tracking data into expression commands.
///
/// # Invariants
///
/// - Weights are clamped to [0, 1].
/// - Non-finite weights are dropped.
/// - Duplicate names within a frame are suppressed (first wins).
/// - Empty output produces no command.
#[derive(Clone, Debug, Default)]
pub struct ExpressionCommandBuilder {
    metrics: ExpressionCommandMetrics,
}

impl ExpressionCommandBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get current metrics.
    #[must_use]
    pub fn metrics(&self) -> &ExpressionCommandMetrics {
        &self.metrics
    }

    /// Reset metrics.
    pub fn reset_metrics(&mut self) {
        self.metrics = ExpressionCommandMetrics::default();
    }

    /// Build expression commands from raw weight inputs.
    ///
    /// Accepts an iterator of (name, weight) pairs. Applies:
    /// - Non-finite rejection
    /// - Range clamping to [0, 1]
    /// - Duplicate suppression (first wins)
    ///
    /// Returns an empty Vec if no valid expressions remain.
    #[must_use]
    pub fn build<'a>(
        &mut self,
        expressions: impl IntoIterator<Item = (&'a str, f32)>,
    ) -> Vec<ExpressionCommand> {
        let mut seen = HashMap::new();
        let mut commands = Vec::new();

        for (name, weight) in expressions {
            // Non-finite rejection.
            if !weight.is_finite() {
                self.metrics.non_finite_dropped += 1;
                continue;
            }

            // Range clamping.
            let clamped = if !(0.0..=1.0).contains(&weight) {
                self.metrics.out_of_range_clamped += 1;
                weight.clamp(0.0, 1.0)
            } else {
                weight
            };

            // Duplicate suppression.
            if seen.contains_key(name) {
                self.metrics.duplicates_suppressed += 1;
                continue;
            }
            seen.insert(name.to_string(), ());

            commands.push(ExpressionCommand {
                name: name.to_string(),
                weight: clamped,
            });
        }

        if commands.is_empty() {
            self.metrics.empty_commands += 1;
        }

        commands
    }

    /// Build a "reset to zero" command for all previously active expressions.
    ///
    /// This is used when the avatar changes or tracking is lost, to ensure
    /// all expressions return to neutral.
    #[must_use]
    pub fn build_reset<'a>(
        &self,
        active_expressions: impl IntoIterator<Item = &'a str>,
    ) -> Vec<ExpressionCommand> {
        active_expressions
            .into_iter()
            .map(|name| ExpressionCommand {
                name: name.to_string(),
                weight: 0.0,
            })
            .collect()
    }
}

/// Build expression commands from an AvatarControlFrame and capabilities.
///
/// This is a convenience function that extracts blink, mouth, and gaze
/// weights from the control frame and builds commands. The actual mapping
/// from raw tracking values to expression names is done by the blink/mouth/gaze
/// modules; this function just assembles the final command list.
///
/// Returns an empty Vec if no expressions should be sent.
#[must_use]
pub fn build_frame_commands(
    _frame: &AvatarControlFrame,
    _capabilities: &AvatarCapabilities,
    blink_weights: &[(String, f32)],
    mouth_weights: &[(String, f32)],
    gaze_weights: &[(String, f32)],
) -> Vec<ExpressionCommand> {
    let mut builder = ExpressionCommandBuilder::new();

    // Combine all weight sources.
    let all_weights: Vec<(&str, f32)> = blink_weights
        .iter()
        .chain(mouth_weights.iter())
        .chain(gaze_weights.iter())
        .map(|(name, weight)| (name.as_str(), *weight))
        .collect();

    builder.build(all_weights)
}

/// Builds the 51 Perfect Sync commands for effective channels.
///
/// `TongueOut` is never sent because MediaPipe does not track the tongue and
/// the coefficient is always zero. The resolver is supplied by the avatar
/// adapter so VRM-specific expression map lookup remains outside the
/// engine-neutral ARKit52 contract.
#[must_use]
pub fn build_detailed_face_commands(
    coefficients: &Arkit52Coefficients,
    capabilities: &PerfectSyncCapabilities,
    mut resolve_name: impl FnMut(ArkitBlendshape) -> Option<String>,
) -> Vec<ExpressionCommand> {
    let mut builder = ExpressionCommandBuilder::new();
    let mut named = Vec::with_capacity(ARKIT52_CHANNEL_COUNT - 1);
    for channel in ArkitBlendshape::ALL {
        if channel == ArkitBlendshape::TongueOut || !capabilities.is_effective(channel) {
            continue;
        }
        if let Some(name) = resolve_name(channel) {
            named.push((name, coefficients.get(channel)));
        }
    }
    builder.build(named.iter().map(|(name, weight)| (name.as_str(), *weight)))
}

/// Builds one pure face command list from exactly one of two paths.
///
/// The detailed path is chosen only when `detailed_face` is present and all
/// 51 required Perfect Sync channels are effective; it sends the 51 custom
/// Expression commands and no coarse blink, mouth, or LookAt command. Every
/// other case uses the standard path, which sends the supplied coarse weights
/// and no custom detailed command. The two paths are never mixed in a frame.
#[must_use]
pub fn build_face_commands(
    detailed_face: Option<&Arkit52Coefficients>,
    capabilities: &AvatarCapabilities,
    blink_weights: &[(String, f32)],
    mouth_weights: &[(String, f32)],
    gaze_weights: &[(String, f32)],
    resolve_name: impl FnMut(ArkitBlendshape) -> Option<String>,
) -> Vec<ExpressionCommand> {
    let supports_detailed = capabilities.perfect_sync.supports_perfect_sync();
    if let Some(coefficients) = detailed_face.filter(|_| supports_detailed) {
        return build_detailed_face_commands(
            coefficients,
            &capabilities.perfect_sync,
            resolve_name,
        );
    }

    let mut builder = ExpressionCommandBuilder::new();
    builder.build(
        blink_weights
            .iter()
            .chain(mouth_weights.iter())
            .chain(gaze_weights.iter())
            .map(|(name, weight)| (name.as_str(), *weight)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_command_stable_input_stable_output() {
        let mut builder = ExpressionCommandBuilder::new();
        let input = [("blink", 0.5), ("aa", 0.3)];

        let result1 = builder.build(input.iter().map(|(n, w)| (*n, *w)));
        let result2 = builder.build(input.iter().map(|(n, w)| (*n, *w)));

        assert_eq!(
            result1, result2,
            "stable input should produce stable output"
        );
    }

    #[test]
    fn expression_command_nan_dropped() {
        let mut builder = ExpressionCommandBuilder::new();
        let input = [("blink", f32::NAN), ("aa", 0.5)];

        let result = builder.build(input.iter().map(|(n, w)| (*n, *w)));

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "aa");
        assert_eq!(builder.metrics().non_finite_dropped, 1);
    }

    #[test]
    fn expression_command_infinity_dropped() {
        let mut builder = ExpressionCommandBuilder::new();
        let input = [("blink", f32::INFINITY), ("aa", f32::NEG_INFINITY)];

        let result = builder.build(input.iter().map(|(n, w)| (*n, *w)));

        assert!(result.is_empty());
        assert_eq!(builder.metrics().non_finite_dropped, 2);
    }

    #[test]
    fn expression_command_out_of_range_clamped() {
        let mut builder = ExpressionCommandBuilder::new();
        let input = [("blink", 1.5), ("aa", -0.3)];

        let result = builder.build(input.iter().map(|(n, w)| (*n, *w)));

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].weight, 1.0);
        assert_eq!(result[1].weight, 0.0);
        assert_eq!(builder.metrics().out_of_range_clamped, 2);
    }

    #[test]
    fn expression_command_duplicate_suppressed() {
        let mut builder = ExpressionCommandBuilder::new();
        let input = [("blink", 0.5), ("blink", 0.8), ("aa", 0.3)];

        let result = builder.build(input.iter().map(|(n, w)| (*n, *w)));

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "blink");
        assert_eq!(result[0].weight, 0.5, "first occurrence wins");
        assert_eq!(builder.metrics().duplicates_suppressed, 1);
    }

    #[test]
    fn expression_command_empty_input() {
        let mut builder = ExpressionCommandBuilder::new();
        let input: [(&str, f32); 0] = [];

        let result = builder.build(input.iter().copied());

        assert!(result.is_empty());
        assert_eq!(builder.metrics().empty_commands, 1);
    }

    #[test]
    fn expression_command_all_invalid_empty() {
        let mut builder = ExpressionCommandBuilder::new();
        let input = [("blink", f32::NAN), ("aa", f32::INFINITY)];

        let result = builder.build(input.iter().map(|(n, w)| (*n, *w)));

        assert!(result.is_empty());
        assert_eq!(builder.metrics().empty_commands, 1);
    }

    #[test]
    fn expression_command_reset_produces_zeros() {
        let builder = ExpressionCommandBuilder::new();
        let active = ["blink", "aa", "lookLeft"];

        let result = builder.build_reset(active.iter().copied());

        assert_eq!(result.len(), 3);
        for cmd in &result {
            assert_eq!(cmd.weight, 0.0);
        }
    }

    #[test]
    fn expression_command_bevy_event_separation() {
        // Verify that the builder produces pure data, not Bevy events.
        let mut builder = ExpressionCommandBuilder::new();
        let result = builder.build([("blink", 0.5)].iter().map(|(n, w)| (*n, *w)));

        // The result is Vec<ExpressionCommand>, not a Bevy event type.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "blink");
        assert_eq!(result[0].weight, 0.5);
    }

    #[test]
    fn detailed_face_commands_skip_tongue_out_and_ineffective_channels() {
        let coefficients = Arkit52Coefficients::try_from_array({
            let mut values = [0.0; ARKIT52_CHANNEL_COUNT];
            values[ArkitBlendshape::TongueOut.index()] = 0.75;
            values[ArkitBlendshape::JawOpen.index()] = 0.4;
            values[ArkitBlendshape::BrowInnerUp.index()] = 0.6;
            values
        })
        .unwrap();
        let capabilities = PerfectSyncCapabilities::from_named_statuses([
            ("TongueOut", true),
            ("JawOpen", true),
            ("BrowInnerUp", false),
        ]);
        let commands = build_detailed_face_commands(&coefficients, &capabilities, |channel| {
            Some(channel.canonical_name().to_owned())
        });
        assert_eq!(
            commands,
            [ExpressionCommand {
                name: "JawOpen".to_owned(),
                weight: 0.4,
            }]
        );
    }
}
