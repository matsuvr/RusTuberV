//! Per-expression bind status owned by the application.
//!
//! A VRM expression preset can be present in metadata while resolving to no
//! scene node; capability inspection uses this component to distinguish that
//! present-but-no-op case from an effective expression. Material and texture
//! binds are counted separately so a color-only or UV-only expression is not
//! treated as empty just because it has no morph binds.
//!
//! The unmodified upstream runtime publishes no bind counts, so the facts are
//! computed once at bind time from the source expression definitions
//! (`VrmSourceExpressions`) resolved against the live scene; see
//! `expression::source::build_binding_statuses`. No presence-based
//! approximation is applied anywhere else.

use bevy::prelude::*;

/// Bind facts for one expression entity.
#[derive(Component, Reflect, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[reflect(Component)]
pub struct ExpressionBindingStatus {
    /// Number of morph binds that resolved to a scene node.
    pub resolved_morph_bind_count: usize,
    /// Number of morph binds declared by the source.
    pub declared_morph_bind_count: usize,
    /// Material/texture binds whose glTF index resolved to a scene material
    /// and whose target property is representable by that material.
    pub resolved_material_bind_count: usize,
    /// Number of material/texture binds declared by the source.
    pub declared_material_bind_count: usize,
    /// Declared binds whose glTF index did not resolve to any scene material.
    pub unresolved_material_bind_count: usize,
    /// Declared binds whose target property is not representable (unknown
    /// target, or a standard-material target such as `shadeColor`).
    pub unsupported_material_bind_count: usize,
    /// `true` when the source declared this expression as a standard preset;
    /// `false` for author-defined custom expressions.
    pub declared_as_preset: bool,
}
