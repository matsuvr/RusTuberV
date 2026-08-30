//! Expression event coalescing and change epsilon.
//!
//! Ensures at most one `ModifyExpressions` event per avatar per frame,
//! skips sending when the change is below epsilon, and resets state
//! on avatar generation change.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_vrm1::prelude::{
    ExpressionBindingStatus, ExpressionEntityMap, LookAtExpressionWeights, ModifyExpressions,
    VrmExpression,
};
use vtuber_core::ArkitBlendshape;

use crate::capabilities::SelectedGazeBackend;
use crate::expression::blink::{RawBlinkInput, map_blink_with_fallback};
use crate::expression::command::{ExpressionCommand, build_face_commands};
use crate::expression::mouth::{RawMouthInput, map_mouth_with_fallback};
use crate::lifecycle::{AvatarLifecycle, AvatarLifecycleState};
use crate::mirror::AvatarMotionMirror;
use crate::unload::ActiveControlFrame;

/// Default epsilon for change detection.
pub const DEFAULT_CHANGE_EPSILON: f32 = 0.01;

/// Tracks the previous expression state for coalescing.
#[derive(Clone, Debug, Default)]
pub struct ExpressionStateTracker {
    /// Previous frame's expression weights.
    previous: HashMap<String, f32>,
    /// Current avatar generation.
    generation: u64,
    /// Change epsilon threshold.
    epsilon: f32,
}

impl ExpressionStateTracker {
    /// Create a new tracker with default epsilon.
    #[must_use]
    pub fn new() -> Self {
        Self {
            previous: HashMap::new(),
            generation: 0,
            epsilon: DEFAULT_CHANGE_EPSILON,
        }
    }

    /// Create a new tracker with custom epsilon.
    #[must_use]
    pub fn with_epsilon(epsilon: f32) -> Self {
        Self {
            previous: HashMap::new(),
            generation: 0,
            epsilon,
        }
    }

    /// Reset the tracker for a new avatar generation.
    pub fn reset_for_generation(&mut self, generation: u64) {
        self.previous.clear();
        self.generation = generation;
    }

    /// Get the current generation.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Compute the commands to send, applying coalescing and epsilon.
    ///
    /// Returns `None` if no changes exceed epsilon (skip sending).
    /// Returns `Some(commands)` with the full set of commands to send,
    /// including explicit zeros for expressions that were previously
    /// non-zero but are now zero.
    #[must_use]
    pub fn compute_commands(
        &mut self,
        new_commands: &[ExpressionCommand],
        current_generation: u64,
    ) -> Option<Vec<ExpressionCommand>> {
        // Reset if generation changed.
        if current_generation != self.generation {
            self.reset_for_generation(current_generation);
        }

        // Build the new weight map.
        let mut new_weights: HashMap<String, f32> = HashMap::new();
        for cmd in new_commands {
            new_weights.insert(cmd.name.clone(), cmd.weight);
        }

        // Check if any change exceeds epsilon.
        let mut has_significant_change = false;

        // Check new/changed values.
        for (name, &weight) in &new_weights {
            let prev = self.previous.get(name).copied().unwrap_or(0.0);
            if (weight - prev).abs() > self.epsilon {
                has_significant_change = true;
                break;
            }
        }

        // Check for values that disappeared (need explicit zero).
        if !has_significant_change {
            for (name, &prev_weight) in &self.previous {
                if prev_weight.abs() > self.epsilon && !new_weights.contains_key(name) {
                    has_significant_change = true;
                    break;
                }
            }
        }

        if !has_significant_change {
            return None;
        }

        // Build the output commands, including explicit zeros for disappeared expressions.
        let mut output: Vec<ExpressionCommand> = new_commands.to_vec();

        // Add explicit zeros for previously non-zero expressions that are now gone.
        for (name, &prev_weight) in &self.previous {
            if prev_weight.abs() > self.epsilon && !new_weights.contains_key(name) {
                output.push(ExpressionCommand {
                    name: name.clone(),
                    weight: 0.0,
                });
            }
        }

        // Update previous state.
        self.previous = new_weights;

        Some(output)
    }

    /// Force a reset of the previous state (e.g., on avatar unload).
    pub fn force_reset(&mut self) {
        self.previous.clear();
    }
}

/// Coalesce multiple expression command lists into a single list.
///
/// Later entries override earlier ones for the same expression name.
#[must_use]
pub fn coalesce_commands(command_lists: &[Vec<ExpressionCommand>]) -> Vec<ExpressionCommand> {
    let mut merged: HashMap<String, f32> = HashMap::new();

    for list in command_lists {
        for cmd in list {
            merged.insert(cmd.name.clone(), cmd.weight);
        }
    }

    merged
        .into_iter()
        .map(|(name, weight)| ExpressionCommand { name, weight })
        .collect()
}

/// Applies one coalesced expression update to the active VRM root.
///
/// The system only emits [`ModifyExpressions`] for names present in the
/// runtime-generated [`ExpressionEntityMap`]. Unsupported capabilities are
/// therefore a normal no-op rather than a runtime error.
pub fn apply_tracked_expressions(
    mut commands: Commands,
    lifecycle: Res<AvatarLifecycle>,
    control_frame: Res<ActiveControlFrame>,
    mirror: Option<Res<AvatarMotionMirror>>,
    expression_maps: Query<(&ExpressionEntityMap, Option<&LookAtExpressionWeights>)>,
    expression_statuses: Query<&ExpressionBindingStatus>,
    mut tracker: Local<ExpressionStateTracker>,
) {
    if lifecycle.state() != AvatarLifecycleState::Ready {
        tracker.force_reset();
        return;
    }
    let Some(root) = lifecycle.active_root() else {
        tracker.force_reset();
        return;
    };
    let Some(frame) = control_frame.frame.as_ref() else {
        return;
    };
    let Ok((expression_map, look_at_weights)) = expression_maps.get(root) else {
        tracker.force_reset();
        return;
    };
    let Some(capabilities) = lifecycle.capabilities() else {
        return;
    };

    let blink = map_blink_with_fallback(
        &blink_input(frame, mirror.is_none_or(|mirror| mirror.is_enabled())),
        capabilities.blink,
    );
    let mouth = map_mouth_with_fallback(
        &RawMouthInput {
            openness: frame.expressions.aa,
            aa: frame.expressions.aa,
            ih: frame.expressions.ih,
            ou: frame.expressions.ou,
            ee: frame.expressions.ee,
            oh: frame.expressions.oh,
        },
        capabilities.mouth,
    );
    let gaze = look_at_expression_commands(capabilities.gaze_backend, look_at_weights.copied());
    let built = build_face_commands(
        frame.detailed_face.as_ref(),
        capabilities,
        &blink,
        &mouth,
        &gaze,
        |channel| {
            resolve_detailed_expression_name(expression_map, channel, |entity| {
                expression_statuses
                    .get(entity)
                    .is_ok_and(|status| status.resolved_morph_bind_count > 0)
            })
        },
    );
    let available = built.into_iter().filter(|command| {
        expression_map
            .0
            .contains_key(&VrmExpression::from(command.name.as_str()))
    });
    let available: Vec<ExpressionCommand> = available.collect();
    let Some(changed) = tracker.compute_commands(&available, control_frame.generation.0) else {
        return;
    };

    commands.trigger(ModifyExpressions::from_iter(
        root,
        changed
            .into_iter()
            .map(|command| (VrmExpression::from(command.name.as_str()), command.weight)),
    ));
}

fn resolve_detailed_expression_name(
    expression_map: &ExpressionEntityMap,
    channel: ArkitBlendshape,
    mut is_effective: impl FnMut(Entity) -> bool,
) -> Option<String> {
    [channel.canonical_name(), channel.lower_camel_alias()]
        .into_iter()
        .find_map(|name| {
            let expression = VrmExpression::from(name);
            let entity = expression_map.0.get(&expression)?;
            is_effective(*entity).then(|| name.to_owned())
        })
}

fn blink_input(frame: &vtuber_core::AvatarControlFrame, mirrored: bool) -> RawBlinkInput {
    let (left, right) = if mirrored {
        // VRM's left/right names are anatomical. Swapping them preserves the
        // image-space side when the avatar is presented as a mirror.
        (frame.expressions.blink_right, frame.expressions.blink_left)
    } else {
        (frame.expressions.blink_left, frame.expressions.blink_right)
    };
    RawBlinkInput {
        left,
        right,
        combined: left.max(right),
    }
}

fn look_at_expression_commands(
    backend: SelectedGazeBackend,
    weights: Option<LookAtExpressionWeights>,
) -> Vec<(String, f32)> {
    if backend != SelectedGazeBackend::Expression {
        return Vec::new();
    }
    let weights = weights.unwrap_or_default();
    vec![
        ("lookLeft".to_owned(), weights.look_left),
        ("lookRight".to_owned(), weights.look_right),
        ("lookUp".to_owned(), weights.look_up),
        ("lookDown".to_owned(), weights.look_down),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_vrm1::prelude::{ExpressionBindingStatus, VrmExpression};
    use vtuber_core::{
        AvatarControlFrame, ExpressionCoefficients, FrameSeq, GazeSignal, HeadPose,
        HeadTranslationSignal, MonoTimeNs, TrackingState,
    };

    fn cmd(name: &str, weight: f32) -> ExpressionCommand {
        ExpressionCommand {
            name: name.to_string(),
            weight,
        }
    }

    fn resolve_jaw_open_alias(
        canonical_bind_count: usize,
        lower_camel_bind_count: usize,
    ) -> Option<String> {
        let mut world = World::new();
        let canonical_entity = world
            .spawn(ExpressionBindingStatus {
                resolved_morph_bind_count: canonical_bind_count,
            })
            .id();
        let lower_camel_entity = world
            .spawn(ExpressionBindingStatus {
                resolved_morph_bind_count: lower_camel_bind_count,
            })
            .id();
        let expression_map = ExpressionEntityMap(
            [
                (VrmExpression::from("JawOpen"), canonical_entity),
                (VrmExpression::from("jawOpen"), lower_camel_entity),
            ]
            .into_iter()
            .collect(),
        );

        resolve_detailed_expression_name(&expression_map, ArkitBlendshape::JawOpen, |entity| {
            world
                .get::<ExpressionBindingStatus>(entity)
                .is_some_and(|status| status.resolved_morph_bind_count > 0)
        })
    }

    #[test]
    fn detailed_name_resolution_skips_noop_canonical_alias() {
        assert_eq!(
            resolve_jaw_open_alias(0, 1).as_deref(),
            Some("jawOpen"),
            "an effective lower-camel alias must win over a no-op canonical alias"
        );
    }

    #[test]
    fn detailed_name_resolution_skips_noop_lower_camel_alias() {
        assert_eq!(
            resolve_jaw_open_alias(1, 0).as_deref(),
            Some("JawOpen"),
            "an effective canonical alias must remain eligible"
        );
    }

    #[test]
    fn detailed_name_resolution_prefers_canonical_when_both_aliases_are_effective() {
        assert_eq!(resolve_jaw_open_alias(1, 1).as_deref(), Some("JawOpen"));
    }

    #[test]
    fn detailed_name_resolution_emits_no_command_when_both_aliases_are_noop() {
        assert!(resolve_jaw_open_alias(0, 0).is_none());
    }

    #[test]
    fn mirrored_blink_input_swaps_only_side_specific_channels() {
        let frame = AvatarControlFrame {
            source_seq: FrameSeq(1),
            captured_at: MonoTimeNs(1),
            produced_at: MonoTimeNs(1),
            confidence: 1.0,
            state: TrackingState::Tracking,
            head: HeadPose::default(),
            head_translation: HeadTranslationSignal::UNAVAILABLE,
            gaze: GazeSignal::UNAVAILABLE,
            expressions: ExpressionCoefficients {
                blink_left: 0.2,
                blink_right: 0.8,
                ..Default::default()
            },
            detailed_face: None,
        };

        assert_eq!(
            blink_input(&frame, true),
            RawBlinkInput {
                left: 0.8,
                right: 0.2,
                combined: 0.8,
            }
        );
        assert_eq!(
            blink_input(&frame, false),
            RawBlinkInput {
                left: 0.2,
                right: 0.8,
                combined: 0.8,
            }
        );
    }

    #[test]
    fn expression_coalescing_first_frame_always_sends() {
        let mut tracker = ExpressionStateTracker::new();
        let commands = vec![cmd("blink", 0.5)];

        let result = tracker.compute_commands(&commands, 1);
        assert!(result.is_some(), "first frame should always send");
    }

    #[test]
    fn expression_coalescing_steady_state_skips() {
        let mut tracker = ExpressionStateTracker::new();
        let commands = vec![cmd("blink", 0.5)];

        // First frame sends.
        let _ = tracker.compute_commands(&commands, 1);

        // Same commands next frame should skip.
        let result = tracker.compute_commands(&commands, 1);
        assert!(result.is_none(), "unchanged commands should be skipped");
    }

    #[test]
    fn expression_coalescing_small_change_skips() {
        let mut tracker = ExpressionStateTracker::new();
        let commands1 = vec![cmd("blink", 0.5)];
        let commands2 = vec![cmd("blink", 0.505)]; // within epsilon

        let _ = tracker.compute_commands(&commands1, 1);
        let result = tracker.compute_commands(&commands2, 1);
        assert!(result.is_none(), "change within epsilon should be skipped");
    }

    #[test]
    fn expression_coalescing_large_change_sends() {
        let mut tracker = ExpressionStateTracker::new();
        let commands1 = vec![cmd("blink", 0.5)];
        let commands2 = vec![cmd("blink", 0.7)]; // exceeds epsilon

        let _ = tracker.compute_commands(&commands1, 1);
        let result = tracker.compute_commands(&commands2, 1);
        assert!(result.is_some(), "change exceeding epsilon should send");
    }

    #[test]
    fn expression_coalescing_disappeared_expression_gets_zero() {
        let mut tracker = ExpressionStateTracker::new();
        let commands1 = vec![cmd("blink", 0.5), cmd("aa", 0.3)];
        let commands2 = vec![cmd("blink", 0.5)]; // aa disappeared

        let _ = tracker.compute_commands(&commands1, 1);
        let result = tracker.compute_commands(&commands2, 1).unwrap();

        // Should include explicit zero for aa.
        assert!(result.iter().any(|c| c.name == "aa" && c.weight == 0.0));
    }

    #[test]
    fn expression_coalescing_generation_change_resets() {
        let mut tracker = ExpressionStateTracker::new();
        let commands = vec![cmd("blink", 0.5)];

        let _ = tracker.compute_commands(&commands, 1);

        // Generation change should reset and send again.
        let result = tracker.compute_commands(&commands, 2);
        assert!(result.is_some(), "generation change should force send");
        assert_eq!(tracker.generation(), 2);
    }

    #[test]
    fn expression_coalescing_empty_commands_steady() {
        let mut tracker = ExpressionStateTracker::new();
        let empty: Vec<ExpressionCommand> = vec![];

        // First frame with empty commands.
        let result1 = tracker.compute_commands(&empty, 1);
        // Empty first frame should still be None (nothing to send).
        assert!(result1.is_none());

        // Second frame also empty.
        let result2 = tracker.compute_commands(&empty, 1);
        assert!(result2.is_none());
    }

    #[test]
    fn expression_coalescing_merge_lists() {
        let list1 = vec![cmd("blink", 0.5), cmd("aa", 0.3)];
        let list2 = vec![cmd("blink", 0.7), cmd("lookLeft", 0.4)];

        let merged = coalesce_commands(&[list1, list2]);

        // Later list should override.
        assert!(merged.iter().any(|c| c.name == "blink" && c.weight == 0.7));
        assert!(merged.iter().any(|c| c.name == "aa" && c.weight == 0.3));
        assert!(
            merged
                .iter()
                .any(|c| c.name == "lookLeft" && c.weight == 0.4)
        );
    }

    #[test]
    fn expression_coalescing_one_event_per_frame() {
        // The compute_commands function returns a single Vec,
        // ensuring only one event is sent per frame.
        let mut tracker = ExpressionStateTracker::new();
        let commands = vec![cmd("blink", 0.5), cmd("aa", 0.3)];

        let result = tracker.compute_commands(&commands, 1);
        assert!(result.is_some());
        // The result is a single Vec, not multiple.
        let output = result.unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn expression_backend_forwards_only_vrm_look_at_weights() {
        let commands = look_at_expression_commands(
            SelectedGazeBackend::Expression,
            Some(LookAtExpressionWeights {
                look_left: 0.7,
                look_right: 0.0,
                look_up: 0.2,
                look_down: 0.0,
            }),
        );
        assert_eq!(commands.len(), 4);
        assert!(
            commands
                .iter()
                .any(|(name, weight)| name == "lookLeft" && *weight == 0.7)
        );
        assert!(
            commands
                .iter()
                .any(|(name, weight)| name == "lookUp" && *weight == 0.2)
        );
    }

    #[test]
    fn bone_backend_never_emits_gaze_expressions() {
        let commands = look_at_expression_commands(
            SelectedGazeBackend::Bone,
            Some(LookAtExpressionWeights {
                look_left: 1.0,
                ..LookAtExpressionWeights::default()
            }),
        );
        assert!(commands.is_empty());
    }
}
