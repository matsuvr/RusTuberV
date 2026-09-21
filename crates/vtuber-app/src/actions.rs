//! UI actions — commands emitted by the UI layer.
//!
//! The UI reads [`crate::ui_model::UiViewModel`] snapshots and emits these
//! actions. The orchestrator processes them and updates domain state.
//! The UI never calls camera, filesystem, or VRM APIs directly.

use std::path::PathBuf;

use vtuber_avatar::ArmPoseProfileOverride;

/// An independent edit to the look switch or strength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RichLookChange {
    /// Keep the configured strength while switching the effect.
    Enabled(bool),
    /// Keep the switch, including when the strength becomes zero.
    Strength(f32),
}

pub(crate) fn reduce_rich_look(
    mut current: vtuber_avatar::RichLookSettings,
    change: RichLookChange,
) -> vtuber_avatar::RichLookSettings {
    match change {
        RichLookChange::Enabled(enabled) => current.enabled = enabled,
        RichLookChange::Strength(strength) => current.strength = strength,
    }
    current
}

/// Actions that the UI can emit.
///
/// These are processed by the orchestrator, which translates them into
/// domain service calls. The UI layer should only construct these values
/// and send them — it should not perform the actual operations.
#[derive(Clone, Debug, PartialEq)]
pub enum UiAction {
    // --- Pane navigation ---
    /// Switch the detail pane to a different category.
    SwitchPane(crate::ui_model::Pane),

    // --- Camera actions ---
    /// Refresh the list of available cameras.
    RefreshCameras,
    /// Select a camera by index.
    SelectCamera {
        /// Camera index from the available list.
        index: usize,
    },
    /// Restore the current avatar's last successful auto-framed camera pose
    /// and recenter the tracking neutral so the currently observed facing
    /// direction becomes the new front.
    ResetAvatarCamera,

    // --- Avatar actions ---
    /// Import a VRM model from the given path.
    ImportAvatar {
        /// Path to the VRM file.
        path: PathBuf,
    },
    /// Read a selected VRM and open the license review sheet without
    /// importing it.
    RequestAvatarImportReview {
        /// Path to the VRM file.
        path: PathBuf,
    },
    /// Update the license review acceptance checkbox.
    SetAvatarImportReviewAccepted {
        /// Whether the reviewer checked the acceptance box.
        accepted: bool,
    },
    /// Import the reviewed VRM. Ignored unless the reviewer accepted.
    AcceptAvatarImportReview,
    /// Dismiss the license review sheet without importing.
    CancelAvatarImportReview,
    /// Unload the current avatar.
    UnloadAvatar,

    // --- Lifecycle actions ---
    /// Start all workers (capture → inference → tracking).
    Start,
    /// Stop all workers in reverse order.
    Stop,

    // --- NDI output actions ---
    /// Start the optional transparent avatar NDI output.
    StartNdiOutput,
    /// Stop the optional transparent avatar NDI output.
    StopNdiOutput,

    // --- Calibration actions ---
    /// Begin calibration sequence.
    BeginCalibration,
    /// Cancel in-progress calibration.
    CancelCalibration,
    /// Retry calibration after failure.
    RetryCalibration,

    // --- Preview actions ---
    /// Toggle preview mirroring.
    ToggleMirror,
    /// Toggle preview visibility.
    TogglePreview,
    /// Toggle mirror-style avatar motion.
    ToggleAvatarMotionMirror,

    // --- Observed arm tracking ---
    /// Enable or disable webcam shoulder/elbow/wrist tracking.
    SetArmTrackingEnabled {
        /// Whether observed arms should replace the virtual hand anchors.
        enabled: bool,
    },
    /// Drop the current subject calibration and start a fresh one.
    RecalibrateArms,

    // --- Avatar pose settings ---
    /// Store a bounded per-model default-arm profile and re-resolve it.
    SetArmPoseProfile {
        /// The six validated profile parameters edited by the settings UI.
        profile: ArmPoseProfileOverride,
    },
    /// Remove the active model's override and return to geometry-derived pose.
    ResetArmPoseProfile,

    // --- Expression key bindings ---
    /// Assign or unassign a fixed expression key for a specific model
    /// generation.
    ///
    /// `expression: None` means "unassigned". The target model and generation
    /// are captured when the UI issues the action, so a swap that happens
    /// before processing cannot apply the operation to the new model.
    AssignExpressionKey {
        /// Model ID the UI displayed when the action was issued.
        model_id: String,
        /// Avatar generation the UI displayed when the action was issued.
        generation: vtuber_avatar::AvatarGeneration,
        /// Fixed physical key.
        key: crate::expression_keys::ExpressionKey,
        /// Exact runtime expression ID, or `None` to unassign.
        expression: Option<String>,
    },
    /// Restore the initial assignment for a specific model generation.
    ResetExpressionBindings {
        /// Model ID the UI displayed when the action was issued.
        model_id: String,
        /// Avatar generation the UI displayed when the action was issued.
        generation: vtuber_avatar::AvatarGeneration,
    },
    /// Toggle the expression currently assigned to a key.
    ToggleExpressionKey {
        /// Avatar generation the input was collected against.
        generation: vtuber_avatar::AvatarGeneration,
        /// Fixed physical key.
        key: crate::expression_keys::ExpressionKey,
    },
    /// Remove the manual expression layer without selecting another.
    ClearManualExpression {
        /// Avatar generation the UI displayed when the action was issued.
        generation: vtuber_avatar::AvatarGeneration,
    },

    // --- Error actions ---
    /// Dismiss the current error (does not clear domain failure state).
    DismissError,
    /// Retry after a recoverable error.
    RetryAfterError,

    // --- Appearance ---
    /// Switch the UI language and persist the choice.
    SetLanguage(crate::settings::UiLanguage),

    // --- Rich look ---
    /// Change the live look without writing the settings file.
    ChangeRichLook(RichLookChange),
    /// Persist the look when a checkbox or slider edit is complete.
    SaveRichLook,
}

impl UiAction {
    /// Check if this action requires a running pipeline.
    #[must_use]
    pub fn requires_running_pipeline(&self) -> bool {
        matches!(
            self,
            UiAction::BeginCalibration | UiAction::CancelCalibration | UiAction::RetryCalibration
        )
    }

    /// Check if this action is a pane navigation action.
    #[must_use]
    pub fn is_navigation(&self) -> bool {
        matches!(self, UiAction::SwitchPane(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui_model::Pane;

    #[test]
    fn actions_start_does_not_require_pipeline() {
        assert!(!UiAction::Start.requires_running_pipeline());
    }

    #[test]
    fn actions_calibration_requires_pipeline() {
        assert!(UiAction::BeginCalibration.requires_running_pipeline());
        assert!(UiAction::CancelCalibration.requires_running_pipeline());
        assert!(UiAction::RetryCalibration.requires_running_pipeline());
    }

    #[test]
    fn actions_stop_does_not_require_pipeline() {
        assert!(!UiAction::Stop.requires_running_pipeline());
    }

    #[test]
    fn ndi_output_actions_are_independent_of_tracking_pipeline() {
        assert!(!UiAction::StartNdiOutput.requires_running_pipeline());
        assert!(!UiAction::StopNdiOutput.requires_running_pipeline());
        assert!(!UiAction::StartNdiOutput.is_navigation());
        assert!(!UiAction::StopNdiOutput.is_navigation());
    }

    #[test]
    fn actions_navigation_is_navigation() {
        assert!(UiAction::SwitchPane(Pane::Camera).is_navigation());
        assert!(UiAction::SwitchPane(Pane::Diagnostics).is_navigation());
    }

    #[test]
    fn actions_non_navigation_is_not_navigation() {
        assert!(!UiAction::Start.is_navigation());
        assert!(!UiAction::Stop.is_navigation());
        assert!(!UiAction::RefreshCameras.is_navigation());
    }

    #[test]
    fn actions_import_avatar_carries_path() {
        let path = PathBuf::from("/tmp/model.vrm");
        let action = UiAction::ImportAvatar { path: path.clone() };
        match action {
            UiAction::ImportAvatar { path: p } => assert_eq!(p, path),
            _ => panic!("expected ImportAvatar"),
        }
    }

    #[test]
    fn actions_request_avatar_import_review_carries_path() {
        let path = PathBuf::from("/tmp/model.vrm");
        let action = UiAction::RequestAvatarImportReview { path: path.clone() };
        match action {
            UiAction::RequestAvatarImportReview { path: p } => assert_eq!(p, path),
            _ => panic!("expected RequestAvatarImportReview"),
        }
    }

    #[test]
    fn actions_review_decisions_are_distinct_one_shot_actions() {
        assert!(!UiAction::AcceptAvatarImportReview.is_navigation());
        assert!(!UiAction::CancelAvatarImportReview.is_navigation());
        assert!(!UiAction::AcceptAvatarImportReview.requires_running_pipeline());
        assert_ne!(
            UiAction::SetAvatarImportReviewAccepted { accepted: true },
            UiAction::SetAvatarImportReviewAccepted { accepted: false }
        );
    }

    #[test]
    fn actions_select_camera_carries_index() {
        let action = UiAction::SelectCamera { index: 3 };
        match action {
            UiAction::SelectCamera { index } => assert_eq!(index, 3),
            _ => panic!("expected SelectCamera"),
        }
    }

    #[test]
    fn reset_camera_is_a_distinct_one_shot_action() {
        assert!(!UiAction::ResetAvatarCamera.is_navigation());
        assert!(!UiAction::ResetAvatarCamera.requires_running_pipeline());
    }

    #[test]
    fn actions_are_clone_and_eq() {
        let a = UiAction::Start;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn expression_key_actions_carry_their_key_and_expression() {
        let key = crate::expression_keys::ExpressionKey::Digit1;
        let generation = vtuber_avatar::AvatarGeneration(3);
        let assign = UiAction::AssignExpressionKey {
            model_id: "model-a".into(),
            generation,
            key,
            expression: Some("happy".into()),
        };
        match assign {
            UiAction::AssignExpressionKey {
                model_id,
                generation: g,
                key: k,
                expression,
            } => {
                assert_eq!(model_id, "model-a");
                assert_eq!(g, generation);
                assert_eq!(k, key);
                assert_eq!(expression.as_deref(), Some("happy"));
            }
            _ => panic!("expected AssignExpressionKey"),
        }
        assert_ne!(
            UiAction::ToggleExpressionKey { generation, key },
            UiAction::ClearManualExpression { generation }
        );
    }

    #[test]
    fn rich_reducer_keeps_switch_and_strength_independent() {
        use vtuber_avatar::RichLookSettings;
        let initial = RichLookSettings::default();
        assert!(!initial.enabled);
        assert_eq!(initial.strength, 1.0);
        for strength in [0.0, 0.5, 1.0] {
            let tuned = reduce_rich_look(initial, RichLookChange::Strength(strength));
            let on = reduce_rich_look(tuned, RichLookChange::Enabled(true));
            let off = reduce_rich_look(on, RichLookChange::Enabled(false));
            assert_eq!(
                off,
                RichLookSettings {
                    enabled: false,
                    strength
                }
            );
            assert_eq!(reduce_rich_look(off, RichLookChange::Enabled(true)), on);
        }
    }

    #[test]
    fn rich_look_actions_carry_their_value() {
        assert_eq!(
            UiAction::ChangeRichLook(RichLookChange::Enabled(true)),
            UiAction::ChangeRichLook(RichLookChange::Enabled(true))
        );
        assert_ne!(
            UiAction::ChangeRichLook(RichLookChange::Enabled(true)),
            UiAction::ChangeRichLook(RichLookChange::Enabled(false))
        );
        assert_eq!(
            UiAction::ChangeRichLook(RichLookChange::Strength(0.25)),
            UiAction::ChangeRichLook(RichLookChange::Strength(0.25))
        );
        assert!(!UiAction::ChangeRichLook(RichLookChange::Enabled(true)).is_navigation());
        assert!(
            !UiAction::ChangeRichLook(RichLookChange::Strength(1.0)).requires_running_pipeline()
        );
    }

    #[test]
    fn actions_are_debug() {
        let action = UiAction::ImportAvatar {
            path: PathBuf::from("/tmp/model.vrm"),
        };
        let debug = format!("{:?}", action);
        assert!(debug.contains("ImportAvatar"));
        assert!(debug.contains("model.vrm"));
    }
}
