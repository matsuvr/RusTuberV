//! Per-material role state for the look settings UI.
//!
//! `#91` carries no rich rendering: this module keeps only the role enum, the
//! persisted per-material selections, and the resource/message plumbing the
//! settings UI edits. Role inference and every system that writes materials,
//! lights, or the finish were removed with the old rich implementation; they
//! return in the follow-up issues (`#93`–`#96`).

use std::collections::{BTreeMap, BTreeSet};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// The material's display role: which role preset a future look would tune.
///
/// A role never converts the material's shader kind: an unlit material stays
/// unlit and an MToon material stays MToon. Unknown materials resolve to
/// [`MaterialRole::General`], the regular shared preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaterialRole {
    /// The first version's modest common enhancement.
    General,
    /// The author's colors, expression shading and shade colors dominate, with
    /// a weak gloss.
    Face,
    /// A broad, weak gloss that never applies the head's face normal treatment.
    Skin,
    /// A clearer highlight than the face, without directional flow data.
    Hair,
    /// Restrained gloss that keeps the author's roughness variation.
    Fabric,
    /// The authored metallic/roughness surface and IBL stay in charge.
    Metal,
    /// The drawn eyes, MatCap and alpha/UV expression stay in charge.
    Eye,
}

/// Resolves the effective role: a user selection wins, otherwise the inferred
/// role applies. `selected == None` is the UI's "Auto".
#[must_use]
pub fn resolve_material_role(
    inferred: MaterialRole,
    selected: Option<MaterialRole>,
) -> MaterialRole {
    selected.unwrap_or(inferred)
}

/// One persisted per-material role selection. `selected == None` is "Auto".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialRoleOverride {
    /// The glTF material index the selection applies to.
    pub material_index: usize,
    /// The user's role selection; `None` is the UI's "Auto".
    pub selected: Option<MaterialRole>,
}

/// Per-material roles and names for the active avatar.
///
/// User overrides are replaced from the persisted settings. Shared materials
/// share one index and therefore one role, so a shared-material edit acts on
/// the whole material.
#[derive(Resource, Default, Debug)]
pub struct AvatarMaterialRoles {
    inferred: BTreeMap<usize, MaterialRole>,
    selected: BTreeMap<usize, MaterialRole>,
    names: BTreeMap<usize, String>,
}

impl AvatarMaterialRoles {
    /// Replaces every user selection. `None` selections remove the override
    /// and return the material to "Auto".
    pub fn replace_overrides(&mut self, overrides: impl Iterator<Item = MaterialRoleOverride>) {
        self.selected.clear();
        for MaterialRoleOverride {
            material_index,
            selected,
        } in overrides
        {
            match selected {
                Some(role) => {
                    self.selected.insert(material_index, role);
                }
                None => {
                    self.selected.remove(&material_index);
                }
            }
        }
    }

    /// The effective role of one material: the user selection, else the
    /// inferred role, else `General`.
    #[must_use]
    pub fn role(&self, index: usize) -> MaterialRole {
        let inferred = self
            .inferred
            .get(&index)
            .copied()
            .unwrap_or(MaterialRole::General);
        resolve_material_role(inferred, self.selected.get(&index).copied())
    }

    /// The material name recorded for one index, if the glTF source named it.
    #[must_use]
    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(&index).map(String::as_str)
    }

    /// The recorded materials as `(index, name, user selection)` triples in
    /// ascending index order.
    #[must_use]
    pub fn entries(&self) -> Vec<(usize, &str, Option<MaterialRole>)> {
        self.inferred
            .keys()
            .chain(self.selected.keys())
            .copied()
            .collect::<BTreeSet<usize>>()
            .into_iter()
            .map(|index| {
                (
                    index,
                    self.name(index).unwrap_or(""),
                    self.selected.get(&index).copied(),
                )
            })
            .collect()
    }

    /// Drops every recorded material and selection.
    pub fn clear(&mut self) {
        self.inferred.clear();
        self.selected.clear();
        self.names.clear();
    }
}

/// Requests that [`AvatarMaterialRoles`] replace its user selections.
#[derive(Message, Clone, Debug, PartialEq)]
pub struct MaterialRoleOverridesChanged(pub Vec<MaterialRoleOverride>);

/// Copies queued role-override changes into [`AvatarMaterialRoles`].
///
/// This system only updates the resource: it never touches materials, files or
/// the inference.
pub fn apply_material_role_overrides(
    mut changes: MessageReader<MaterialRoleOverridesChanged>,
    mut roles: ResMut<AvatarMaterialRoles>,
) {
    for change in changes.read() {
        roles.replace_overrides(change.0.iter().copied());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_overrides_replace_and_clear_by_index() {
        let mut roles = AvatarMaterialRoles::default();
        assert_eq!(roles.role(0), MaterialRole::General);
        roles.replace_overrides(
            [
                MaterialRoleOverride {
                    material_index: 0,
                    selected: Some(MaterialRole::Hair),
                },
                MaterialRoleOverride {
                    material_index: 1,
                    selected: None,
                },
            ]
            .into_iter(),
        );
        assert_eq!(roles.role(0), MaterialRole::Hair);
        assert_eq!(roles.role(1), MaterialRole::General);
        assert_eq!(roles.entries(), vec![(0, "", Some(MaterialRole::Hair)),]);
        roles.clear();
        assert_eq!(roles.role(0), MaterialRole::General);
        assert!(roles.entries().is_empty());
    }

    #[test]
    fn override_messages_replace_the_resource_selections() {
        let mut app = App::new();
        app.init_resource::<AvatarMaterialRoles>()
            .add_message::<MaterialRoleOverridesChanged>()
            .add_systems(Update, apply_material_role_overrides);
        app.world_mut()
            .resource_mut::<Messages<MaterialRoleOverridesChanged>>()
            .write(MaterialRoleOverridesChanged(vec![MaterialRoleOverride {
                material_index: 2,
                selected: Some(MaterialRole::Eye),
            }]));
        app.update();
        assert_eq!(
            app.world().resource::<AvatarMaterialRoles>().role(2),
            MaterialRole::Eye
        );
    }
}
