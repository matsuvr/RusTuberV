//! Shared native/Rich material ownership and update order.
use super::{AvatarLookSettings, RichLookSettings};
use crate::lifecycle::{AvatarLifecycle, AvatarLifecycleState};
use bevy::asset::{AssetEventSystems, AssetId};
use bevy::pbr::{ExtendedMaterial, Material, MaterialExtension};
use bevy::prelude::*;
use std::collections::HashMap;

/// Material-specific fields used by the two Rich material variants.
#[doc(hidden)]
pub trait RichMaterialExtension: MaterialExtension + Clone {
    /// Native material owning expression values.
    type Native: Material + Clone;
    /// Builds the extension, excluding unsupported native materials.
    fn create(native: &Self::Native, settings: RichLookSettings) -> Option<Self>;
    /// Returns whether the expression fields and extension are current.
    fn matches(
        current: &ExtendedMaterial<Self::Native, Self>,
        native: &Self::Native,
        settings: RichLookSettings,
    ) -> bool;
    /// Copies changed expression fields and extension settings.
    fn sync(
        current: &mut ExtendedMaterial<Self::Native, Self>,
        native: &Self::Native,
        settings: RichLookSettings,
    );
}

/// Keeps the native material handle while a mesh renders its Rich variant.
#[derive(Component, Debug, Clone)]
pub struct RichMaterialSwap<E: RichMaterialExtension> {
    /// Native asset owning current expression and UV values.
    pub(crate) native: Handle<E::Native>,
    rich: Handle<ExtendedMaterial<E::Native, E>>,
}

pub(super) fn register_rich_material_systems<E: RichMaterialExtension>(app: &mut App) {
    app.add_systems(
        Update,
        switch_rich_materials::<E>.after(super::apply_look_settings_changes),
    );
    app.add_systems(
        PostUpdate,
        sync_rich_materials::<E>
            .after(crate::expression::material::apply_expression_materials)
            .before(AssetEventSystems),
    );
}

#[expect(
    clippy::type_complexity,
    reason = "Bevy query selects the native handle and its concrete Rich swap component"
)]
fn switch_rich_materials<E: RichMaterialExtension>(
    mut commands: Commands,
    lifecycle: Res<AvatarLifecycle>,
    settings: Res<AvatarLookSettings>,
    parents: Query<&ChildOf>,
    native_assets: Res<Assets<E::Native>>,
    mut rich_assets: ResMut<Assets<ExtendedMaterial<E::Native, E>>>,
    meshes: Query<(
        Entity,
        Option<&MeshMaterial3d<E::Native>>,
        Option<&RichMaterialSwap<E>>,
    )>,
) {
    if lifecycle.state() != AvatarLifecycleState::Ready {
        return;
    }
    let Some(root) = lifecycle.active_root() else {
        return;
    };
    let enabled = settings.0.enabled();
    let mut rich_by_native: HashMap<AssetId<E::Native>, Handle<ExtendedMaterial<E::Native, E>>> =
        meshes
            .iter()
            .filter_map(|(_, _, swap)| swap.map(|swap| (swap.native.id(), swap.rich.clone())))
            .collect();
    for (entity, native, swap) in &meshes {
        if !crate::binding::is_descendant(entity, root, &parents) {
            continue;
        }
        match (enabled, native, swap) {
            (true, Some(native), None) => {
                let Some(source) = native_assets.get(native.id()) else {
                    continue;
                };
                let Some(extension) = E::create(source, settings.0) else {
                    continue;
                };
                let rich = if let Some(existing) = rich_by_native.get(&native.id()) {
                    existing.clone()
                } else {
                    let created = rich_assets.add(ExtendedMaterial {
                        base: source.clone(),
                        extension,
                    });
                    rich_by_native.insert(native.id(), created.clone());
                    created
                };
                commands
                    .entity(entity)
                    .remove::<MeshMaterial3d<E::Native>>()
                    .insert((
                        MeshMaterial3d(rich.clone()),
                        RichMaterialSwap::<E> {
                            native: native.0.clone(),
                            rich,
                        },
                    ));
            }
            (false, None, Some(swap)) => {
                commands
                    .entity(entity)
                    .remove::<MeshMaterial3d<ExtendedMaterial<E::Native, E>>>()
                    .remove::<RichMaterialSwap<E>>()
                    .insert(MeshMaterial3d(swap.native.clone()));
            }
            _ => {}
        }
    }
}

fn sync_rich_materials<E: RichMaterialExtension>(
    settings: Res<AvatarLookSettings>,
    native_assets: Res<Assets<E::Native>>,
    mut rich_assets: ResMut<Assets<ExtendedMaterial<E::Native, E>>>,
    swaps: Query<&RichMaterialSwap<E>>,
) {
    for swap in &swaps {
        let Some(native) = native_assets.get(swap.native.id()) else {
            continue;
        };
        let Some(current) = rich_assets.get(swap.rich.id()) else {
            continue;
        };
        if E::matches(current, native, settings.0) {
            continue;
        }
        if let Some(mut material) = rich_assets.get_mut(swap.rich.id()) {
            E::sync(&mut material, native, settings.0);
        }
    }
}
