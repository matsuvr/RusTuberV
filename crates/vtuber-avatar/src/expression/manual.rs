//! Manual expression selection state.
//!
//! The UI and raw keyboard produce [`ManualExpressionRequest`] values bound to
//! an avatar generation. This module owns the single-selection state machine
//! and validates requests against the active lifecycle generation. It is pure
//! state plus one small Bevy system; the actual expression writing stays in
//! [`crate::expression::system`].

use bevy::prelude::*;
use vtuber_core::ArkitBlendshape;

use crate::expression_catalog::{AvatarExpressionCatalog, ExpressionKind};
use crate::lifecycle::{AvatarGeneration, AvatarLifecycle, AvatarLifecycleState};

/// The one manually selected expression for the active avatar generation.
///
/// `selected == None` means the manual layer is off and the existing tracking
/// contract applies. `neutral` can be selected as a normal expression; it is
/// still distinct from clearing the manual layer.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct ManualExpressionSelection {
    /// Generation this selection belongs to.
    pub generation: AvatarGeneration,
    /// Selected runtime expression ID, if any.
    pub selected: Option<String>,
}

impl ManualExpressionSelection {
    /// Applies a toggle intent from the UI or keyboard.
    ///
    /// A request from a different generation replaces the stale selection with
    /// a fresh one; the same expression toggles off.
    pub fn toggle(&mut self, generation: AvatarGeneration, expression: &str) {
        if self.generation != generation {
            self.generation = generation;
            self.selected = None;
        }
        if self.selected.as_deref() == Some(expression) {
            self.selected = None;
        } else {
            self.selected = Some(expression.to_owned());
        }
    }

    /// Clears the manual layer. Requests from another generation are ignored
    /// so an old model's UI cannot clear a newer model's selection.
    pub fn clear(&mut self, generation: AvatarGeneration) {
        if self.generation == generation {
            self.selected = None;
        }
    }

    /// Drops any selection state (unload, model change, failure).
    pub fn reset(&mut self) {
        self.selected = None;
    }

    /// Returns the selected ID only when it still belongs to the given
    /// generation and exists in the catalog.
    #[must_use]
    pub fn selected_in<'a>(
        &'a self,
        generation: AvatarGeneration,
        catalog: Option<&'a AvatarExpressionCatalog>,
    ) -> Option<&'a str> {
        if self.generation != generation {
            return None;
        }
        let selected = self.selected.as_deref()?;
        let catalog = catalog?;
        // Not-ready definitions are never substituted with another
        // expression; they simply do not reach the writer.
        catalog
            .entry(selected)
            .filter(|entry| entry.availability.is_ready())
            .map(|entry| entry.id.as_str())
    }
}

/// System set for the manual request handler.
///
/// The application orders its UI-action system before this set so a toggle
/// recorded in the same frame is applied by the writer later that frame.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManualExpressionSet;

/// Intent emitted by the UI layer into the avatar runtime.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub enum ManualExpressionRequest {
    /// Toggle one expression for the given generation.
    Toggle {
        /// Avatar generation the UI acted on.
        generation: AvatarGeneration,
        /// Exact runtime expression ID.
        expression: String,
    },
    /// Remove the manual layer for the given generation.
    Clear {
        /// Avatar generation the UI acted on.
        generation: AvatarGeneration,
    },
}

/// Applies manual requests and keeps the selection valid for the lifecycle.
///
/// The system processes requests in event order (last toggle wins) and clears
/// the selection whenever the avatar is not ready.
pub fn apply_manual_expression_requests(
    lifecycle: Res<AvatarLifecycle>,
    mut selection: ResMut<ManualExpressionSelection>,
    mut requests: MessageReader<ManualExpressionRequest>,
) {
    if lifecycle.state() != AvatarLifecycleState::Ready {
        selection.reset();
        return;
    }
    let generation = lifecycle.current_generation();
    for request in requests.read() {
        match request {
            ManualExpressionRequest::Toggle {
                generation: request_generation,
                expression,
            } if *request_generation == generation => {
                selection.toggle(generation, expression);
            }
            ManualExpressionRequest::Clear {
                generation: request_generation,
            } if *request_generation == generation => {
                selection.clear(generation);
            }
            _ => {
                // Requests from an older generation never reach the new model.
            }
        }
    }
}

/// Classifies a manually selected expression for the composer.
///
/// Returns `None` when the catalog is unavailable, which keeps the existing
/// standard face path (never mixing an unknown manual ID into the writer).
#[must_use]
pub fn manual_tracking_kind(
    catalog: Option<&AvatarExpressionCatalog>,
    selected: Option<&str>,
) -> Option<ExpressionKind> {
    let selected = selected?;
    Some(catalog?.entry(selected)?.kind)
}

/// Returns `true` when a tracking expression should be excluded from manual
/// composition's automatic face commands.
#[must_use]
pub fn is_tracking_selection(
    catalog: Option<&AvatarExpressionCatalog>,
    selected: Option<&str>,
) -> bool {
    matches!(
        manual_tracking_kind(catalog, selected),
        Some(ExpressionKind::Tracking)
    )
}

/// Returns `true` when the expression is the excluded tongue channel.
#[must_use]
pub fn is_excluded_expression_id(id: &str) -> bool {
    id == "TongueOut"
        || id == "tongueOut"
        || ArkitBlendshape::from_name(id) == Some(ArkitBlendshape::TongueOut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression_catalog::ExpressionCatalogInput;

    fn generation(value: u64) -> AvatarGeneration {
        AvatarGeneration(value)
    }

    fn catalog() -> AvatarExpressionCatalog {
        AvatarExpressionCatalog::build(
            "model".into(),
            4,
            [
                ExpressionCatalogInput {
                    id: "happy",
                    declared_as_preset: true,
                    declared_morph_bind_count: 1,
                    resolved_morph_bind_count: 1,
                    declared_material_bind_count: 0,
                    unsupported_material_bind_count: 0,
                },
                ExpressionCatalogInput {
                    id: "JawOpen",
                    declared_as_preset: false,
                    declared_morph_bind_count: 1,
                    resolved_morph_bind_count: 1,
                    declared_material_bind_count: 0,
                    unsupported_material_bind_count: 0,
                },
            ],
        )
    }

    #[test]
    fn toggle_transitions_match_the_contract() {
        let mut selection = ManualExpressionSelection::default();
        selection.toggle(generation(3), "happy");
        assert_eq!(selection.selected.as_deref(), Some("happy"));

        selection.toggle(generation(3), "happy");
        assert_eq!(selection.selected, None);

        selection.toggle(generation(3), "happy");
        selection.toggle(generation(3), "angry");
        assert_eq!(selection.selected.as_deref(), Some("angry"));

        selection.clear(generation(3));
        assert_eq!(selection.selected, None);
    }

    #[test]
    fn toggle_for_a_new_generation_replaces_the_stale_selection() {
        let mut selection = ManualExpressionSelection {
            generation: generation(1),
            selected: Some("happy".into()),
        };
        selection.toggle(generation(2), "angry");
        assert_eq!(selection.generation, generation(2));
        assert_eq!(selection.selected.as_deref(), Some("angry"));
    }

    #[test]
    fn request_system_filters_stale_generations_and_keeps_event_order() {
        let mut app = App::new();
        app.init_resource::<ManualExpressionSelection>()
            .init_resource::<AvatarLifecycle>()
            .add_message::<ManualExpressionRequest>()
            .add_systems(Update, apply_manual_expression_requests);
        let root = app.world_mut().spawn_empty().id();
        {
            let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
            lifecycle.request_load(root).unwrap();
            lifecycle.start_binding(root);
            lifecycle.finish_ready();
        }
        let current = app
            .world()
            .resource::<AvatarLifecycle>()
            .current_generation();
        let stale = AvatarGeneration(current.0 + 100);

        app.world_mut()
            .resource_mut::<Messages<ManualExpressionRequest>>()
            .write(ManualExpressionRequest::Toggle {
                generation: current,
                expression: "happy".into(),
            });
        app.world_mut()
            .resource_mut::<Messages<ManualExpressionRequest>>()
            .write(ManualExpressionRequest::Toggle {
                generation: current,
                expression: "happy".into(),
            });
        app.update();
        assert_eq!(
            app.world().resource::<ManualExpressionSelection>().selected,
            None,
            "two toggles in one frame collapse in event order"
        );

        app.world_mut()
            .resource_mut::<Messages<ManualExpressionRequest>>()
            .write(ManualExpressionRequest::Toggle {
                generation: current,
                expression: "happy".into(),
            });
        app.world_mut()
            .resource_mut::<Messages<ManualExpressionRequest>>()
            .write(ManualExpressionRequest::Toggle {
                generation: current,
                expression: "angry".into(),
            });
        app.world_mut()
            .resource_mut::<Messages<ManualExpressionRequest>>()
            .write(ManualExpressionRequest::Clear { generation: stale });
        app.update();
        assert_eq!(
            app.world()
                .resource::<ManualExpressionSelection>()
                .selected
                .as_deref(),
            Some("angry"),
            "stale clear must not touch the current model"
        );
    }

    #[test]
    fn clear_from_another_generation_is_ignored() {
        let mut selection = ManualExpressionSelection {
            generation: generation(2),
            selected: Some("happy".into()),
        };
        selection.clear(generation(1));
        assert_eq!(selection.selected.as_deref(), Some("happy"));
        selection.clear(generation(2));
        assert_eq!(selection.selected, None);
    }

    #[test]
    fn selected_in_requires_generation_and_catalog_membership() {
        let catalog = catalog();
        let selection = ManualExpressionSelection {
            generation: generation(4),
            selected: Some("happy".into()),
        };
        assert_eq!(
            selection.selected_in(generation(4), Some(&catalog)),
            Some("happy")
        );
        assert_eq!(selection.selected_in(generation(5), Some(&catalog)), None);
        assert_eq!(selection.selected_in(generation(4), None), None);

        let unknown = ManualExpressionSelection {
            generation: generation(4),
            selected: Some("missing".into()),
        };
        assert_eq!(unknown.selected_in(generation(4), Some(&catalog)), None);
    }

    #[test]
    fn tracking_classification_uses_exact_catalog_kind() {
        let catalog = catalog();
        assert!(is_tracking_selection(Some(&catalog), Some("JawOpen")));
        assert!(!is_tracking_selection(Some(&catalog), Some("happy")));
        assert!(!is_tracking_selection(Some(&catalog), Some("missing")));
        assert!(!is_tracking_selection(None, Some("JawOpen")));
    }

    #[test]
    fn tongue_out_is_never_a_valid_manual_id() {
        assert!(is_excluded_expression_id("TongueOut"));
        assert!(is_excluded_expression_id("tongueOut"));
        assert!(!is_excluded_expression_id("JawOpen"));
    }
}
