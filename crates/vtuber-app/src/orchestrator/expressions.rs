//! Expression binding mutations, manual intents, and UI snapshots.

use bevy::prelude::*;
use vtuber_avatar::{
    AvatarExpressionCatalog, AvatarGeneration, AvatarLifecycle, ManualExpressionRequest,
    ManualExpressionSelection,
};

use super::{Orchestrator, OrchestratorError};
use crate::expression_keys::{
    ExpressionBindingStore, ExpressionBindings, ExpressionKey, can_assign_expression,
    effective_bindings, manual_request_for_key,
};
use crate::settings::AppSettings;
use crate::ui_model::{
    ExpressionEntryViewModel, ExpressionKeyBindingViewModel, ExpressionViewModel, UiViewModel,
};

/// Returns only a catalog consistent with the current model target.
fn current_expression_catalog<'a>(
    orchestrator: &Orchestrator,
    lifecycle: &'a AvatarLifecycle,
) -> Option<&'a AvatarExpressionCatalog> {
    if lifecycle.state() != vtuber_avatar::AvatarLifecycleState::Ready {
        return None;
    }
    let model_id = orchestrator.active_model_id()?;
    let catalog = lifecycle.expression_catalog()?;
    (catalog.generation == lifecycle.current_generation().0 && catalog.model_id == model_id)
        .then_some(catalog)
}

/// Expression key mutation requested by the UI.
pub(super) enum ExpressionBindingAction<'a> {
    /// Assign an expression (or unassign with `None`) to one key.
    Assign {
        target: &'a crate::actions::ModelActionTarget,
        key: ExpressionKey,
        expression: Option<&'a str>,
    },
    /// Restore the target model's deterministic default assignment.
    Reset {
        target: &'a crate::actions::ModelActionTarget,
    },
}

impl ExpressionBindingAction<'_> {
    fn target(&self) -> &crate::actions::ModelActionTarget {
        match self {
            Self::Assign { target, .. } | Self::Reset { target } => target,
        }
    }
}

/// Applies one expression binding mutation through copy-save-commit.
///
/// The target model and generation must still be current; an action collected
/// before a model swap never changes the new model's settings. The candidate
/// is written to `settings.toml` first; only a successful save commits it to
/// the runtime store and clears the manual selection. A failed save keeps the
/// previous assignment and reports the existing typed error.
pub(super) fn apply_expression_binding_action(
    orchestrator: &mut Orchestrator,
    action: ExpressionBindingAction<'_>,
    store: &mut Option<ResMut<ExpressionBindingStore>>,
    settings: Option<&AppSettings>,
    lifecycle: Option<&AvatarLifecycle>,
    manual_requests: &mut Option<MessageWriter<ManualExpressionRequest>>,
) {
    let Some(lifecycle) = lifecycle else {
        return;
    };
    let target = action.target();
    let Some(catalog) = current_expression_catalog(orchestrator, lifecycle) else {
        return;
    };
    let model_id = &target.model_id;
    let Some(store) = store.as_deref_mut() else {
        return;
    };
    let current = effective_bindings(store, model_id, Some(catalog));
    let mut candidate = current.clone();
    match action {
        ExpressionBindingAction::Assign {
            expression, key, ..
        } => match expression {
            Some(expression) => {
                if !can_assign_expression(Some(catalog), expression) {
                    return;
                }
                candidate.assign(key, expression.to_owned());
            }
            None => candidate.unassign(key),
        },
        ExpressionBindingAction::Reset { .. } => {
            candidate.reset_to_defaults(catalog);
        }
    }
    if candidate == current {
        return;
    }
    let mut next = store.clone();
    next.set(model_id.clone(), candidate);
    if let Some(settings) = settings
        && let Err(error) = settings.save_expression_bindings(&next)
    {
        orchestrator.set_last_error(Some(OrchestratorError::ExpressionSettingsFailed(
            error.to_string(),
        )));
        return;
    }
    *store = next;
    if let Some(requests) = manual_requests.as_mut() {
        requests.write(ManualExpressionRequest::Clear {
            generation: target.generation,
        });
    }
}

/// Resolves the key's binding snapshot and emits a toggle intent for the
/// generation the input was collected against.
///
/// Missing or not-ready expressions are dropped rather than substituted, and
/// the issued generation is passed through unchanged.
pub(super) fn toggle_expression_key_action(
    orchestrator: &Orchestrator,
    generation: AvatarGeneration,
    key: ExpressionKey,
    store: Option<&ExpressionBindingStore>,
    lifecycle: Option<&AvatarLifecycle>,
    manual_requests: &mut Option<MessageWriter<ManualExpressionRequest>>,
) {
    let Some(lifecycle) = lifecycle else {
        return;
    };
    // The catalog must be the one consistent with the live model; a pending
    // replacement must not contribute its saved bindings to an old-generation
    // toggle.
    let Some(catalog) = current_expression_catalog(orchestrator, lifecycle) else {
        return;
    };
    if catalog.generation != generation.0 {
        return;
    }
    let Some(store) = store else {
        return;
    };
    let bindings = effective_bindings(store, &catalog.model_id, Some(catalog));
    let Some(expression) = bindings.expression_for(key) else {
        return;
    };
    if !can_assign_expression(Some(catalog), expression) {
        return;
    }
    let Some(request) = manual_request_for_key(generation, key, &bindings) else {
        return;
    };
    if let Some(requests) = manual_requests.as_mut() {
        requests.write(request);
    }
}

/// Emits a manual clear for exactly the generation the UI acted on.
pub(super) fn clear_manual_expression_action(
    generation: AvatarGeneration,
    manual_requests: &mut Option<MessageWriter<ManualExpressionRequest>>,
) {
    let Some(requests) = manual_requests.as_mut() else {
        return;
    };
    requests.write(ManualExpressionRequest::Clear { generation });
}

/// Cheap view-model rebuild key: model, catalog generation, store revision,
/// manual generation, and the selected ID.
type ExpressionViewModelSignature = (String, u64, u64, u64, String);

/// Rebuilds the expression catalog/binding view model.
///
/// Registered by the shell after manual request processing so the selected
/// marker is consistent with the same frame that handled the toggle.
pub fn sync_expression_view_model(
    orchestrator: Res<Orchestrator>,
    lifecycle: Option<Res<AvatarLifecycle>>,
    store: Option<Res<ExpressionBindingStore>>,
    manual: Option<Res<ManualExpressionSelection>>,
    mut view_model: ResMut<UiViewModel>,
    mut last_signature: Local<Option<ExpressionViewModelSignature>>,
) {
    // A pending import may have updated the orchestrator model while the
    // render-side catalog is still the old one. Only the consistent catalog
    // may produce an operable snapshot; the avatar itself keeps rendering.
    let catalog = match lifecycle.as_deref() {
        Some(lifecycle) => current_expression_catalog(&orchestrator, lifecycle),
        None => None,
    };
    let signature = (
        catalog.map_or_else(String::new, |catalog| catalog.model_id.clone()),
        catalog.map_or(0, |catalog| catalog.generation),
        store.as_deref().map_or(0, ExpressionBindingStore::revision),
        manual.as_deref().map_or(0, |manual| manual.generation.0),
        manual
            .as_deref()
            .and_then(|manual| manual.selected.clone())
            .unwrap_or_default(),
    );
    if last_signature.as_ref() == Some(&signature) {
        return;
    }
    *last_signature = Some(signature);

    let selected = manual.as_deref().and_then(|manual| manual.selected.clone());
    // Actions emitted from this snapshot carry exactly the model/generation
    // shown here so a later swap cannot apply them to the new avatar.
    let target = catalog.map(|catalog| crate::actions::ModelActionTarget {
        model_id: catalog.model_id.clone(),
        generation: AvatarGeneration(catalog.generation),
    });
    let bindings = match (catalog, store.as_deref()) {
        (Some(catalog), Some(store)) => effective_bindings(store, &catalog.model_id, Some(catalog)),
        (Some(catalog), None) => ExpressionBindings::default_for(catalog),
        (None, _) => ExpressionBindings::default(),
    };
    let entries = catalog
        .map(|catalog| {
            catalog
                .selectable_entries()
                .iter()
                .map(|entry| ExpressionEntryViewModel {
                    id: entry.id.clone(),
                    source_name: entry.source_name.clone(),
                    kind: entry.kind,
                    availability: entry.availability.clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    let binding_rows = ExpressionKey::ALL
        .into_iter()
        .map(|key| {
            let expression = bindings.expression_for(key).map(str::to_owned);
            ExpressionKeyBindingViewModel {
                key,
                selected: expression
                    .as_deref()
                    .is_some_and(|id| Some(id) == selected.as_deref()),
                expression,
            }
        })
        .collect();
    view_model.expression = ExpressionViewModel {
        target,
        has_catalog: catalog.is_some(),
        entries,
        bindings: binding_rows,
        selected,
    };
}
