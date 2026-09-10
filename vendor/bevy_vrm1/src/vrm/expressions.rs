use crate::prelude::ChildSearcher;
use crate::system_set::VrmSystemSets;
use crate::vrm::gltf::extensions::VrmExtensions;
use crate::vrm::gltf::extensions::vrmc_vrm::{
    MaterialColorBind, MorphTargetBind, TextureTransformBind,
};
use crate::vrm::mtoon::prelude::MToonMaterial;
use crate::vrm::{Vrm, VrmExpression};
use crate::vrma::RetargetSource;
use bevy::animation::{AnimatedBy, AnimationTargetId};
use bevy::app::Plugin;
use bevy::asset::{Assets, Handle};
use bevy::math::Affine2;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpressionCategory {
    Mouth,
    Blink,
    LookAt,
    Other,
}

impl ExpressionCategory {
    pub fn from_preset_name(name: &str) -> Self {
        match name {
            "aa" | "ih" | "ou" | "ee" | "oh" => Self::Mouth,
            "blink" | "blinkLeft" | "blinkRight" => Self::Blink,
            "lookUp" | "lookDown" | "lookLeft" | "lookRight" => Self::LookAt,
            _ => Self::Other,
        }
    }
}

#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpressionOverrideType {
    None,
    Block,
    Blend,
}

impl ExpressionOverrideType {
    pub fn rate(
        &self,
        weight: f32,
    ) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Block => {
                if weight > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Blend => weight,
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "block" => Self::Block,
            "blend" => Self::Blend,
            _ => Self::None,
        }
    }
}

#[derive(Component, Reflect, Debug, Clone)]
#[reflect(Component)]
pub struct ExpressionOverrideSettings {
    pub override_mouth: ExpressionOverrideType,
    pub override_blink: ExpressionOverrideType,
    pub override_look_at: ExpressionOverrideType,
}

#[derive(Component, Reflect, Debug, Clone, Copy, PartialEq, Eq)]
#[reflect(Component)]
pub(crate) struct ExpressionCategoryTag(pub ExpressionCategory);

#[derive(Component, Reflect, Debug, Clone, Copy)]
#[reflect(Component)]
pub struct BinaryExpression;

#[derive(Reflect, Debug, Clone)]
pub(crate) struct ExpressionMetadata {
    pub nodes: Vec<ExpressionNode>,
    pub category: ExpressionCategory,
    pub override_settings: ExpressionOverrideSettings,
    pub is_binary: bool,
    /// `true` when the source placed this expression in the standard preset
    /// map (VRM 1.0) or a known semantic (VRM 0.x). Custom expressions keep
    /// `false` so catalogs never promote them to standard presets.
    pub declared_as_preset: bool,
    pub material_color_binds: Vec<ExpressionMaterialColorBind>,
    pub texture_transform_binds: Vec<ExpressionTextureTransformBind>,
    pub unsupported_material_bind_count: usize,
}

#[derive(Reflect, Debug, Clone)]
pub(crate) struct ExpressionNode {
    pub node_index: usize,
    pub morph_target_index: usize,
    pub weight: f32,
}

/// Standard VRM 1.0 material color bind targets.
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MaterialColorTarget {
    BaseColor,
    EmissionColor,
    ShadeColor,
    RimColor,
    OutlineColor,
}

impl MaterialColorTarget {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "color" => Some(Self::BaseColor),
            "emissionColor" => Some(Self::EmissionColor),
            "shadeColor" => Some(Self::ShadeColor),
            "rimColor" => Some(Self::RimColor),
            "outlineColor" => Some(Self::OutlineColor),
            _ => None,
        }
    }

    /// `true` when a `StandardMaterial` has an equivalent property. MToon
    /// supports every standard target; the unlit/Standard fallback only has
    /// base color and emission.
    fn is_supported_by_standard_material(self) -> bool {
        matches!(self, Self::BaseColor | Self::EmissionColor)
    }
}

/// Resolved material color bind stored on an expression entity.
#[derive(Reflect, Debug, Clone, Copy)]
pub(crate) struct ExpressionMaterialColorBind {
    pub material_index: usize,
    pub target: MaterialColorTarget,
    /// Linear RGBA target value from the source.
    pub target_value: LinearRgba,
}

/// Resolved texture transform bind stored on an expression entity.
#[derive(Reflect, Debug, Clone, Copy)]
pub(crate) struct ExpressionTextureTransformBind {
    pub material_index: usize,
    pub scale: Vec2,
    pub offset: Vec2,
}

/// Material binds declared by one expression, kept on its expression entity.
#[derive(Component, Reflect, Debug, Clone, Default)]
#[reflect(Component)]
pub struct ExpressionMaterialBinds {
    pub colors: Vec<ExpressionMaterialColorBind>,
    pub transforms: Vec<ExpressionTextureTransformBind>,
}

/// glTF material index attached to a mesh entity once its material is
/// resolved by the MToon setup pass. Expression material binds reference
/// materials by this stable glTF index.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct VrmMaterialIndex(pub usize);

/// Expression-free base values captured once when a mesh material is
/// finalized. Every expression application restores these exact values first,
/// so a weight of zero can never accumulate drift.
#[derive(Component, Debug, Clone, Copy)]
pub struct VrmMaterialBaseValues {
    pub base_color: LinearRgba,
    pub emissive: LinearRgba,
    pub shade_color: LinearRgba,
    pub rim_color: LinearRgba,
    pub outline_color: LinearRgba,
    /// Base UV transform. Expression texture binds are applied on top of it.
    pub uv_transform: Affine2,
}

impl VrmMaterialBaseValues {
    pub fn from_mtoon(material: &MToonMaterial) -> Self {
        let base_color = material.base_color.to_linear();
        Self {
            base_color: LinearRgba::new(
                base_color.red,
                base_color.green,
                base_color.blue,
                base_color.alpha,
            ),
            emissive: material.emissive,
            shade_color: material.shade.color,
            rim_color: material.rim_lighting.color,
            outline_color: material.outline.color,
            uv_transform: material.uv_transform,
        }
    }

    pub fn from_standard(material: &StandardMaterial) -> Self {
        let base_color = material.base_color.to_linear();
        Self {
            base_color: LinearRgba::new(
                base_color.red,
                base_color.green,
                base_color.blue,
                base_color.alpha,
            ),
            emissive: material.emissive,
            shade_color: LinearRgba::BLACK,
            rim_color: LinearRgba::BLACK,
            outline_color: LinearRgba::BLACK,
            uv_transform: material.uv_transform,
        }
    }
}

/// Cached mapping from expression name to expression entity.
/// Built during VRM initialization. Use this to query available expressions.
#[derive(Component, Deref, Reflect)]
pub struct ExpressionEntityMap(pub HashMap<VrmExpression, Entity>);

/// Declared and resolved bind status for one expression entity.
///
/// A VRM expression preset can be present in metadata while resolving to no
/// scene node. Avatar capability inspection uses this component to distinguish
/// that present-but-no-op case from an effective expression. Material and
/// texture binds are counted separately so a color-only or UV-only expression
/// is not treated as empty just because it has no morph binds.
#[derive(Component, Reflect, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[reflect(Component)]
pub struct ExpressionBindingStatus {
    /// Number of morph binds that resolved to a scene node.
    pub resolved_morph_bind_count: usize,
    /// Number of morph binds declared by the source.
    pub declared_morph_bind_count: usize,
    /// Number of material/texture binds declared by the source.
    pub declared_material_bind_count: usize,
    /// Declared material binds whose target property is not representable.
    pub unsupported_material_bind_count: usize,
    /// `true` when the source declared this expression as a standard preset.
    pub declared_as_preset: bool,
}

/// Override weight for a single expression entity.
/// Inserted by [`SetExpressions`] or [`ModifyExpressions`], removed by [`ClearExpressions`].
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct ExpressionOverride(pub f32);

/// Sets expression weights on a VRM model, **replacing all previous overrides**.
///
/// Trigger this event to directly control facial expressions.
/// Expression weights are clamped to `0.0..=1.0`.
/// Expressions not included in this call will return to VRMA animation control.
///
/// For partial updates that preserve existing overrides, see [`ModifyExpressions`].
///
/// **Note**: Triggering both `SetExpressions` and [`ModifyExpressions`]
/// on the same entity in the same frame produces undefined results.
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_vrm1::prelude::*;
///
/// fn set_happy(mut commands: Commands, vrms: Query<Entity, With<Vrm>>) {
///     for vrm in vrms.iter() {
///         commands.trigger(SetExpressions::single(vrm, "happy", 1.0));
///     }
/// }
/// ```
#[derive(EntityEvent, Debug)]
pub struct SetExpressions {
    #[event_target]
    pub entity: Entity,
    pub weights: HashMap<VrmExpression, f32>,
}

impl SetExpressions {
    /// Creates a [`SetExpressions`] event for a single expression.
    pub fn single(
        entity: Entity,
        expression: impl Into<VrmExpression>,
        weight: f32,
    ) -> Self {
        Self {
            entity,
            weights: [(expression.into(), weight)].into_iter().collect(),
        }
    }

    /// Creates a [`SetExpressions`] event from an iterator of expression-weight pairs.
    pub fn from_iter(
        entity: Entity,
        iter: impl IntoIterator<Item = (impl Into<VrmExpression>, f32)>,
    ) -> Self {
        Self {
            entity,
            weights: iter.into_iter().map(|(e, w)| (e.into(), w)).collect(),
        }
    }
}

/// Modifies specific expression weights without affecting others (partial update).
///
/// Unlike [`SetExpressions`] which replaces all overrides,
/// this only inserts/updates the specified expressions.
/// Existing overrides not mentioned in this call remain unchanged.
///
/// This is the equivalent of `UniVRM`'s `SetWeight()` and three-vrm's `setValue()`.
/// Ideal for lip-sync where mouth expressions are updated every frame
/// while other expression overrides (e.g. emotions) remain active.
///
/// **Note**: Triggering both [`SetExpressions`] and `ModifyExpressions`
/// on the same entity in the same frame produces undefined results.
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_vrm1::prelude::*;
///
/// fn add_blink(mut commands: Commands, vrms: Query<Entity, With<Vrm>>) {
///     for vrm in vrms.iter() {
///         // Only modifies "blink", leaves other overrides (e.g. "happy") intact
///         commands.trigger(ModifyExpressions::single(vrm, "blink", 1.0));
///     }
/// }
/// ```
#[derive(EntityEvent, Debug)]
pub struct ModifyExpressions {
    #[event_target]
    pub entity: Entity,
    pub weights: HashMap<VrmExpression, f32>,
}

/// The five VRM preset mouth expressions used for lip-sync.
const MOUTH_EXPRESSIONS: [&str; 5] = ["aa", "ih", "ou", "ee", "oh"];

impl ModifyExpressions {
    /// Creates a [`ModifyExpressions`] event for a single expression.
    pub fn single(
        entity: Entity,
        expression: impl Into<VrmExpression>,
        weight: f32,
    ) -> Self {
        Self {
            entity,
            weights: [(expression.into(), weight)].into_iter().collect(),
        }
    }

    /// Creates a [`ModifyExpressions`] event from an iterator of expression-weight pairs.
    pub fn from_iter(
        entity: Entity,
        iter: impl IntoIterator<Item = (impl Into<VrmExpression>, f32)>,
    ) -> Self {
        Self {
            entity,
            weights: iter.into_iter().map(|(e, w)| (e.into(), w)).collect(),
        }
    }

    /// Sets a single mouth expression for lip-sync, resetting all other mouth
    /// expressions to 0.0.
    ///
    /// This is a convenience method that sets all five VRM preset mouth
    /// expressions (aa, ih, ou, ee, oh) with the specified one active and
    /// the rest at 0.0. Non-mouth expression overrides are preserved.
    ///
    /// Inserts `ExpressionOverride(0.0)` for inactive mouth expressions,
    /// which overrides any VRMA animation value. Use [`ClearExpressions`]
    /// to return all expressions to VRMA control.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use bevy_vrm1::prelude::*;
    ///
    /// fn lip_sync(mut commands: Commands, vrms: Query<Entity, With<Vrm>>) {
    ///     for vrm in vrms.iter() {
    ///         commands.trigger(ModifyExpressions::mouth(vrm, "aa", 0.8));
    ///     }
    /// }
    /// ```
    pub fn mouth(
        entity: Entity,
        expression: impl Into<VrmExpression>,
        weight: f32,
    ) -> Self {
        let active = expression.into();
        let mut weights: HashMap<VrmExpression, f32> = MOUTH_EXPRESSIONS
            .iter()
            .map(|&name| (VrmExpression::from(name), 0.0))
            .collect();
        weights.insert(active, weight);
        Self { entity, weights }
    }

    /// Sets multiple mouth expressions for blended lip-sync, resetting
    /// unspecified mouth expressions to 0.0.
    ///
    /// Useful for blend-based lip-sync where multiple vowels are active
    /// simultaneously (e.g. aa=0.3, ih=0.5). Non-mouth expression overrides
    /// are preserved.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use bevy_vrm1::prelude::*;
    ///
    /// fn blended_lip_sync(mut commands: Commands, vrms: Query<Entity, With<Vrm>>) {
    ///     for vrm in vrms.iter() {
    ///         commands.trigger(ModifyExpressions::mouth_weights(
    ///             vrm,
    ///             [("aa", 0.3), ("ih", 0.5)],
    ///         ));
    ///     }
    /// }
    /// ```
    pub fn mouth_weights(
        entity: Entity,
        iter: impl IntoIterator<Item = (impl Into<VrmExpression>, f32)>,
    ) -> Self {
        let mut weights: HashMap<VrmExpression, f32> = MOUTH_EXPRESSIONS
            .iter()
            .map(|&name| (VrmExpression::from(name), 0.0))
            .collect();
        for (expr, weight) in iter {
            weights.insert(expr.into(), weight);
        }
        Self { entity, weights }
    }
}

/// Clears all expression overrides, returning control to VRMA animation.
///
/// After triggering this event, expressions previously set by [`SetExpressions`]
/// or [`ModifyExpressions`] will be controlled by VRMA animation again.
#[derive(EntityEvent, Debug)]
pub struct ClearExpressions {
    #[event_target]
    pub entity: Entity,
}

#[derive(EntityEvent)]
pub(crate) struct RequestInitializeExpressions(pub(crate) Entity);

#[derive(Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub(crate) struct BindExpressionNode {
    pub expression_entity: Entity,
    pub index: usize,
    pub weight: f32,
}

#[derive(Component, Reflect)]
#[reflect(Component)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", reflect(Serialize, Deserialize))]
pub(crate) struct RetargetExpressionNodes(pub(crate) Vec<BindExpressionNode>);

#[derive(Component, Deref, Reflect)]
pub(crate) struct VrmExpressionRegistry(pub(crate) HashMap<VrmExpression, ExpressionMetadata>);

impl VrmExpressionRegistry {
    pub fn new(extensions: &VrmExtensions) -> Self {
        let Some(expressions) = extensions.vrmc_vrm.expressions.as_ref() else {
            return Self(HashMap::default());
        };
        let mut registry = HashMap::default();
        for (name, preset) in &expressions.preset {
            registry.insert(
                VrmExpression(name.clone()),
                expression_metadata(name, preset, true),
            );
        }
        // Preset keys win on malformed collisions; custom names must not
        // replace standard semantics.
        for (name, preset) in &expressions.custom {
            registry
                .entry(VrmExpression(name.clone()))
                .or_insert_with(|| expression_metadata(name, preset, false));
        }
        Self(registry)
    }
}

fn expression_metadata(
    name: &str,
    preset: &crate::vrm::gltf::extensions::vrmc_vrm::VrmPreset,
    declared_as_preset: bool,
) -> ExpressionMetadata {
    let expression_nodes = preset
        .morph_target_binds
        .as_ref()
        .map(|binds| binds.iter().map(convert_to_node).collect::<Vec<_>>())
        .unwrap_or_default();
    let (material_color_binds, unsupported_colors) =
        convert_material_color_binds(&preset.material_color_binds);
    let texture_transform_binds = preset
        .texture_transform_binds
        .iter()
        .map(convert_texture_transform_bind)
        .collect::<Vec<_>>();
    ExpressionMetadata {
        nodes: expression_nodes,
        category: ExpressionCategory::from_preset_name(name),
        override_settings: ExpressionOverrideSettings {
            override_mouth: ExpressionOverrideType::parse(&preset.override_mouth),
            override_blink: ExpressionOverrideType::parse(&preset.override_blink),
            override_look_at: ExpressionOverrideType::parse(&preset.override_look_at),
        },
        is_binary: preset.is_binary,
        declared_as_preset,
        material_color_binds,
        texture_transform_binds,
        unsupported_material_bind_count: unsupported_colors,
    }
}

fn convert_material_color_binds(
    binds: &[MaterialColorBind]
) -> (Vec<ExpressionMaterialColorBind>, usize) {
    let mut converted = Vec::with_capacity(binds.len());
    let mut unsupported = 0;
    for bind in binds {
        let Some(target) = MaterialColorTarget::parse(&bind.bind_type) else {
            unsupported += 1;
            continue;
        };
        converted.push(ExpressionMaterialColorBind {
            material_index: bind.material,
            target,
            target_value: LinearRgba::new(
                bind.target_value[0],
                bind.target_value[1],
                bind.target_value[2],
                bind.target_value[3],
            ),
        });
    }
    (converted, unsupported)
}

fn convert_texture_transform_bind(bind: &TextureTransformBind) -> ExpressionTextureTransformBind {
    ExpressionTextureTransformBind {
        material_index: bind.material,
        // VRM defaults: scale = [1, 1], offset = [0, 0].
        scale: bind.scale.map_or(Vec2::ONE, Vec2::from),
        offset: bind.offset.map_or(Vec2::ZERO, Vec2::from),
    }
}

pub(crate) struct VrmExpressionPlugin;

impl Plugin for VrmExpressionPlugin {
    fn build(
        &self,
        app: &mut App,
    ) {
        app.register_type::<BindExpressionNode>()
            .register_type::<RetargetExpressionNodes>()
            .register_type::<VrmExpressionRegistry>()
            .register_type::<ExpressionEntityMap>()
            .register_type::<ExpressionBindingStatus>()
            .register_type::<ExpressionOverride>()
            .register_type::<ExpressionOverrideSettings>()
            .register_type::<ExpressionCategoryTag>()
            .register_type::<BinaryExpression>()
            .add_observer(apply_initialize_expressions)
            .add_observer(apply_set_expressions)
            .add_observer(apply_modify_expressions)
            .add_observer(apply_clear_expressions)
            .add_systems(
                PostUpdate,
                (
                    bind_expressions,
                    bind_expression_materials.after(bind_expressions),
                )
                    .in_set(VrmSystemSets::Expressions)
                    .after(VrmSystemSets::GazeControl),
            );
    }
}

fn convert_to_node(bind: &MorphTargetBind) -> ExpressionNode {
    ExpressionNode {
        node_index: bind.node,
        morph_target_index: bind.index,
        weight: bind.weight,
    }
}

fn apply_initialize_expressions(
    trigger: On<RequestInitializeExpressions>,
    mut commands: Commands,
    expressions: Query<&VrmExpressionRegistry>,
    material_registries: Query<&crate::vrm::mtoon::VrmcMaterialRegistry>,
    searcher: ChildSearcher,
) {
    let vrm_entity = trigger.event_target();
    let expressions_root = commands.spawn(Name::new(Vrm::EXPRESSIONS_ROOT)).id();
    commands.entity(vrm_entity).add_child(expressions_root);

    let Ok(registry) = expressions.get(vrm_entity) else {
        commands
            .entity(vrm_entity)
            .insert(ExpressionEntityMap(HashMap::default()));
        return;
    };
    let material_registry = material_registries.get(vrm_entity).ok();
    let mut entity_map = HashMap::default();
    for (expression, metadata) in registry.iter() {
        let resolved_nodes = obtain_expression_nodes(vrm_entity, &searcher, &metadata.nodes);
        let resolved_morph_bind_count = resolved_nodes.len();
        let unresolved_material_bind_count =
            unresolved_material_bind_count(metadata, material_registry);
        let mut entity_commands = commands.spawn((
            Name::new(expression.to_string()),
            RetargetSource,
            Transform::default(),
            AnimationPlayer::default(),
            RetargetExpressionNodes(resolved_nodes),
            ExpressionBindingStatus {
                resolved_morph_bind_count,
                declared_morph_bind_count: metadata.nodes.len(),
                declared_material_bind_count: metadata.material_color_binds.len()
                    + metadata.texture_transform_binds.len(),
                unsupported_material_bind_count: metadata.unsupported_material_bind_count
                    + unresolved_material_bind_count,
                declared_as_preset: metadata.declared_as_preset,
            },
            ExpressionMaterialBinds {
                colors: metadata.material_color_binds.clone(),
                transforms: metadata.texture_transform_binds.clone(),
            },
            ExpressionCategoryTag(metadata.category),
            metadata.override_settings.clone(),
        ));
        if metadata.is_binary {
            entity_commands.insert(BinaryExpression);
        }
        let expression_entity = entity_commands.id();
        commands.entity(expression_entity).insert((
            AnimationTargetId::from_name(&Name::new(expression.to_string())),
            AnimatedBy(expression_entity),
        ));
        commands
            .entity(expressions_root)
            .add_child(expression_entity);
        entity_map.insert(expression.clone(), expression_entity);
    }
    commands
        .entity(vrm_entity)
        .insert(ExpressionEntityMap(entity_map));
}

fn bind_expressions(
    mut morph_query: Query<&mut MorphWeights>,
    rig_expressions: Query<(
        &Transform,
        &RetargetExpressionNodes,
        &ExpressionCategoryTag,
        &ExpressionOverrideSettings,
        Option<&ExpressionOverride>,
        Option<&BinaryExpression>,
    )>,
    mut last_signature: Local<Vec<u64>>,
    mut signature_scratch: Local<Vec<u64>>,
) {
    // Signature of every expression's resolved input (raw weight bits + binary
    // flag, in iteration order). All outputs of this system are a pure function
    // of these values, so an unchanged signature means the morph weights are
    // already up to date and the zero+accumulate passes can be skipped.
    signature_scratch.clear();
    for (tf, _retarget, _category_tag, _override_settings, maybe_override, maybe_binary) in
        rig_expressions.iter()
    {
        let raw_weight = match maybe_override {
            Some(ExpressionOverride(w)) => *w,
            None => tf.translation.x,
        };
        let is_binary = maybe_binary.is_some();
        signature_scratch.push((u64::from(raw_weight.to_bits()) << 1) | u64::from(is_binary));
    }
    if !signature_scratch.is_empty() && *last_signature == *signature_scratch {
        return;
    }
    last_signature.clear();
    last_signature.extend_from_slice(&signature_scratch);

    // Pass 1: Collect output weights and accumulate override rates.
    // Also collect all mesh entities that need resetting.
    let mut mouth_rate: f32 = 0.0;
    let mut blink_rate: f32 = 0.0;
    let mut look_at_rate: f32 = 0.0;

    struct ExpressionEntry {
        output_weight: f32,
        category: ExpressionCategory,
        is_binary: bool,
        binds: Vec<(Entity, usize, f32)>,
    }

    let mut entries: Vec<ExpressionEntry> = Vec::new();
    let mut mesh_entities: Vec<Entity> = Vec::new();

    for (tf, retarget, category_tag, override_settings, maybe_override, maybe_binary) in
        rig_expressions.iter()
    {
        let raw_weight = match maybe_override {
            Some(ExpressionOverride(w)) => *w,
            None => tf.translation.x,
        };
        let is_binary = maybe_binary.is_some();
        let output_weight = if is_binary {
            if raw_weight > 0.5 { 1.0 } else { 0.0 }
        } else {
            raw_weight.clamp(0.0, 1.0)
        };

        mouth_rate += override_settings.override_mouth.rate(output_weight);
        blink_rate += override_settings.override_blink.rate(output_weight);
        look_at_rate += override_settings.override_look_at.rate(output_weight);

        let binds: Vec<(Entity, usize, f32)> = retarget
            .0
            .iter()
            .map(|b| (b.expression_entity, b.index, b.weight))
            .collect();
        for &(entity, _, _) in &binds {
            mesh_entities.push(entity);
        }

        entries.push(ExpressionEntry {
            output_weight,
            category: category_tag.0,
            is_binary,
            binds,
        });
    }

    // Pass 2: Compute per-category multipliers.
    let mouth_mul = 1.0 - mouth_rate.clamp(0.0, 1.0);
    let blink_mul = 1.0 - blink_rate.clamp(0.0, 1.0);
    let look_at_mul = 1.0 - look_at_rate.clamp(0.0, 1.0);

    // Pass 3: Reset morph weights, then accumulate.
    mesh_entities.sort_unstable();
    mesh_entities.dedup();
    for &entity in &mesh_entities {
        if let Ok(mut morph_weights) = morph_query.get_mut(entity) {
            for w in morph_weights.weights_mut().iter_mut() {
                *w = 0.0;
            }
        }
    }

    for entry in &entries {
        let multiplier = match entry.category {
            ExpressionCategory::Mouth => mouth_mul,
            ExpressionCategory::Blink => blink_mul,
            ExpressionCategory::LookAt => look_at_mul,
            ExpressionCategory::Other => 1.0,
        };
        let final_weight = if entry.is_binary && multiplier < 1.0 {
            0.0
        } else {
            entry.output_weight * multiplier
        };
        for &(entity, index, bind_weight) in &entry.binds {
            if let Ok(mut morph_weights) = morph_query.get_mut(entity) {
                morph_weights.weights_mut()[index] += final_weight * bind_weight;
            }
        }
    }
}

/// Applies the same final expression weight used for morphs to material color
/// and texture-transform binds.
///
/// Every signature change first restores every material to its captured base
/// values, then accumulates `base + (target - base) * weight`. A zero weight
/// therefore restores the exact base instead of leaving stale color/UV state.
fn bind_expression_materials(
    mtoon_materials: Option<ResMut<Assets<MToonMaterial>>>,
    standard_materials: Option<ResMut<Assets<StandardMaterial>>>,
    rig_expressions: Query<(
        Entity,
        &Transform,
        &ExpressionOverrideSettings,
        Option<&ExpressionOverride>,
        Option<&BinaryExpression>,
    )>,
    bound_expressions: Query<(
        Entity,
        &ExpressionMaterialBinds,
        &ExpressionCategoryTag,
        &Transform,
        Option<&ExpressionOverride>,
        Option<&BinaryExpression>,
    )>,
    materials: Query<(
        Entity,
        &VrmMaterialIndex,
        Option<&MeshMaterial3d<MToonMaterial>>,
        Option<&MeshMaterial3d<StandardMaterial>>,
        Option<&VrmMaterialBaseValues>,
    )>,
    mut last_signature: Local<Vec<u64>>,
    mut signature_scratch: Local<Vec<u64>>,
) {
    // Material assets are optional for headless tests and for models without
    // the material plugins; morph binding still runs independently.
    let (Some(mut mtoon_materials), Some(mut standard_materials)) =
        (mtoon_materials, standard_materials)
    else {
        return;
    };
    // The signature includes the expression entity identity so a replacement
    // model with an identical weight column still performs its first apply.
    signature_scratch.clear();
    for (entity, tf, _settings, maybe_override, maybe_binary) in rig_expressions.iter() {
        let raw_weight = match maybe_override {
            Some(ExpressionOverride(w)) => *w,
            None => tf.translation.x,
        };
        signature_scratch.push(entity.to_bits());
        signature_scratch
            .push((u64::from(raw_weight.to_bits()) << 1) | u64::from(maybe_binary.is_some()));
    }
    if !signature_scratch.is_empty() && *last_signature == *signature_scratch {
        return;
    }
    last_signature.clear();
    last_signature.extend_from_slice(&signature_scratch);

    let mut mouth_rate: f32 = 0.0;
    let mut blink_rate: f32 = 0.0;
    let mut look_at_rate: f32 = 0.0;
    for (_entity, tf, override_settings, maybe_override, maybe_binary) in rig_expressions.iter() {
        let raw_weight = match maybe_override {
            Some(ExpressionOverride(w)) => *w,
            None => tf.translation.x,
        };
        let output_weight = output_weight(raw_weight, maybe_binary.is_some());
        mouth_rate += override_settings.override_mouth.rate(output_weight);
        blink_rate += override_settings.override_blink.rate(output_weight);
        look_at_rate += override_settings.override_look_at.rate(output_weight);
    }
    let mouth_mul = 1.0 - mouth_rate.clamp(0.0, 1.0);
    let blink_mul = 1.0 - blink_rate.clamp(0.0, 1.0);
    let look_at_mul = 1.0 - look_at_rate.clamp(0.0, 1.0);

    // Restore every material to its expression-free base first.
    for (_entity, _index, mtoon_handle, standard_handle, base) in materials.iter() {
        let Some(base) = base else { continue };
        if let Some(handle) = mtoon_handle {
            if let Some(mut material) = mtoon_materials.get_mut(handle.id()) {
                material.base_color = Color::LinearRgba(base.base_color);
                material.emissive = base.emissive;
                material.shade.color = base.shade_color;
                material.rim_lighting.color = base.rim_color;
                material.outline.color = base.outline_color;
                material.uv_transform = base.uv_transform;
            }
        } else if let Some(handle) = standard_handle
            && let Some(mut material) = standard_materials.get_mut(handle.id())
        {
            material.base_color = Color::LinearRgba(base.base_color);
            material.emissive = base.emissive;
            material.uv_transform = base.uv_transform;
        }
    }

    // Collect material entities per glTF index.
    let mut by_index: HashMap<usize, Vec<Entity>> = HashMap::default();
    for (entity, index, _mtoon, _standard, base) in materials.iter() {
        if base.is_none() {
            continue;
        }
        by_index.entry(index.0).or_default().push(entity);
    }

    // Deterministic expression order for multiplicative UV composition.
    let mut ordered: Vec<(Entity, f32, &ExpressionMaterialBinds)> = bound_expressions
        .iter()
        .map(
            |(entity, binds, category_tag, tf, maybe_override, maybe_binary)| {
                let raw_weight = match maybe_override {
                    Some(ExpressionOverride(w)) => *w,
                    None => tf.translation.x,
                };
                let output = output_weight(raw_weight, maybe_binary.is_some());
                let multiplier = match category_tag.0 {
                    ExpressionCategory::Mouth => mouth_mul,
                    ExpressionCategory::Blink => blink_mul,
                    ExpressionCategory::LookAt => look_at_mul,
                    ExpressionCategory::Other => 1.0,
                };
                let final_weight = if maybe_binary.is_some() && multiplier < 1.0 {
                    0.0
                } else {
                    output * multiplier
                };
                (entity, final_weight, binds)
            },
        )
        .collect();
    ordered.sort_unstable_by_key(|(entity, _, _)| entity.to_bits());

    let mut color_accum: HashMap<(Entity, MaterialColorTarget), LinearRgba> = HashMap::default();
    let mut uv_accum: HashMap<Entity, Affine2> = HashMap::default();
    for (_expression, final_weight, binds) in &ordered {
        if *final_weight <= 0.0 {
            continue;
        }
        for bind in &binds.colors {
            let Some(entities) = by_index.get(&bind.material_index) else {
                continue;
            };
            for &entity in entities {
                let Ok((_entity, _index, _mtoon, _standard, Some(base))) = materials.get(entity)
                else {
                    continue;
                };
                let Some(base_color) = base_color_for_target(base, bind.target) else {
                    continue;
                };
                let entry = color_accum
                    .entry((entity, bind.target))
                    .or_insert(base_color);
                *entry = LinearRgba::new(
                    entry.red + (bind.target_value.red - base_color.red) * final_weight,
                    entry.green + (bind.target_value.green - base_color.green) * final_weight,
                    entry.blue + (bind.target_value.blue - base_color.blue) * final_weight,
                    entry.alpha + (bind.target_value.alpha - base_color.alpha) * final_weight,
                );
            }
        }
        for bind in &binds.transforms {
            let Some(entities) = by_index.get(&bind.material_index) else {
                continue;
            };
            for &entity in entities {
                let Ok((_entity, _index, _mtoon, _standard, Some(base))) = materials.get(entity)
                else {
                    continue;
                };
                let entry = uv_accum.entry(entity).or_insert(base.uv_transform);
                *entry = Affine2::from_translation(bind.offset)
                    * Affine2::from_scale(bind.scale)
                    * *entry;
            }
        }
    }

    for ((entity, target), value) in color_accum {
        let Ok((_entity, _index, mtoon_handle, standard_handle, _base)) = materials.get(entity)
        else {
            continue;
        };
        if let Some(handle) = mtoon_handle {
            if let Some(mut material) = mtoon_materials.get_mut(handle.id()) {
                set_mtoon_color_target(&mut material, target, value);
            }
        } else if let Some(handle) = standard_handle {
            if let Some(mut material) = standard_materials.get_mut(handle.id()) {
                set_standard_color_target(&mut material, target, value);
            }
        }
    }
    for (entity, transform) in uv_accum {
        let Ok((_entity, _index, mtoon_handle, standard_handle, _base)) = materials.get(entity)
        else {
            continue;
        };
        if let Some(handle) = mtoon_handle {
            if let Some(mut material) = mtoon_materials.get_mut(handle.id()) {
                material.uv_transform = transform;
            }
        } else if let Some(handle) = standard_handle
            && let Some(mut material) = standard_materials.get_mut(handle.id())
        {
            material.uv_transform = transform;
        }
    }
}

fn output_weight(
    raw_weight: f32,
    is_binary: bool,
) -> f32 {
    if is_binary {
        if raw_weight > 0.5 { 1.0 } else { 0.0 }
    } else {
        raw_weight.clamp(0.0, 1.0)
    }
}

fn base_color_for_target(
    base: &VrmMaterialBaseValues,
    target: MaterialColorTarget,
) -> Option<LinearRgba> {
    match target {
        MaterialColorTarget::BaseColor => Some(base.base_color),
        MaterialColorTarget::EmissionColor => Some(base.emissive),
        MaterialColorTarget::ShadeColor => Some(base.shade_color),
        MaterialColorTarget::RimColor => Some(base.rim_color),
        MaterialColorTarget::OutlineColor => Some(base.outline_color),
    }
}

fn set_mtoon_color_target(
    material: &mut MToonMaterial,
    target: MaterialColorTarget,
    value: LinearRgba,
) {
    match target {
        MaterialColorTarget::BaseColor => material.base_color = Color::LinearRgba(value),
        MaterialColorTarget::EmissionColor => material.emissive = value,
        MaterialColorTarget::ShadeColor => material.shade.color = value,
        MaterialColorTarget::RimColor => material.rim_lighting.color = value,
        MaterialColorTarget::OutlineColor => material.outline.color = value,
    }
}

fn set_standard_color_target(
    material: &mut StandardMaterial,
    target: MaterialColorTarget,
    value: LinearRgba,
) {
    match target {
        MaterialColorTarget::BaseColor => material.base_color = Color::LinearRgba(value),
        MaterialColorTarget::EmissionColor => material.emissive = value,
        MaterialColorTarget::ShadeColor
        | MaterialColorTarget::RimColor
        | MaterialColorTarget::OutlineColor => {
            debug_assert!(!target.is_supported_by_standard_material());
        }
    }
}

fn apply_set_expressions(
    trigger: On<SetExpressions>,
    cache: Query<&ExpressionEntityMap>,
    mut commands: Commands,
) {
    let vrm_entity = trigger.event_target();
    let Ok(map) = cache.get(vrm_entity) else {
        #[cfg(feature = "log")]
        warn!(
            "SetExpressions: ExpressionEntityMap not found for entity {:?}. VRM may not be initialized yet.",
            vrm_entity
        );
        return;
    };
    // Remove overrides not present in the new weights so that
    // each SetExpressions call fully replaces the previous state.
    for (&expr_entity, expression) in map.0.iter().map(|(e, id)| (id, e)) {
        if !trigger.weights.contains_key(expression) {
            commands.entity(expr_entity).remove::<ExpressionOverride>();
        }
    }
    for (expression, weight) in trigger.weights.iter() {
        let Some(&expr_entity) = map.0.get(expression) else {
            #[cfg(feature = "log")]
            warn!("SetExpressions: expression '{}' not found", expression);
            continue;
        };
        commands
            .entity(expr_entity)
            .insert(ExpressionOverride(weight.clamp(0.0, 1.0)));
    }
}

fn apply_modify_expressions(
    trigger: On<ModifyExpressions>,
    cache: Query<&ExpressionEntityMap>,
    mut commands: Commands,
) {
    let vrm_entity = trigger.event_target();
    let Ok(map) = cache.get(vrm_entity) else {
        #[cfg(feature = "log")]
        warn!(
            "ModifyExpressions: ExpressionEntityMap not found for entity {:?}. VRM may not be initialized yet.",
            vrm_entity
        );
        return;
    };
    for (expression, weight) in trigger.weights.iter() {
        let Some(&expr_entity) = map.0.get(expression) else {
            #[cfg(feature = "log")]
            warn!("ModifyExpressions: expression '{}' not found", expression);
            continue;
        };
        commands
            .entity(expr_entity)
            .insert(ExpressionOverride(weight.clamp(0.0, 1.0)));
    }
}

fn apply_clear_expressions(
    trigger: On<ClearExpressions>,
    cache: Query<&ExpressionEntityMap>,
    mut commands: Commands,
) {
    let vrm_entity = trigger.event_target();
    let Ok(map) = cache.get(vrm_entity) else {
        return;
    };
    for &expr_entity in map.0.values() {
        commands.entity(expr_entity).remove::<ExpressionOverride>();
    }
}

/// Counts material/texture binds whose glTF material index does not resolve
/// to a loaded material. These cannot be rendered faithfully and are reported
/// through the expression binding status instead of being silently dropped.
fn unresolved_material_bind_count(
    metadata: &ExpressionMetadata,
    registry: Option<&crate::vrm::mtoon::VrmcMaterialRegistry>,
) -> usize {
    let Some(registry) = registry else {
        return 0;
    };
    let valid = |index: usize| registry.indices.values().any(|value| *value == index);
    metadata
        .material_color_binds
        .iter()
        .filter(|bind| !valid(bind.material_index))
        .count()
        + metadata
            .texture_transform_binds
            .iter()
            .filter(|bind| !valid(bind.material_index))
            .count()
}

fn obtain_expression_nodes(
    vrm_entity: Entity,
    searcher: &ChildSearcher,
    nodes: &[ExpressionNode],
) -> Vec<BindExpressionNode> {
    nodes
        .iter()
        .flat_map(|node| {
            Some(BindExpressionNode {
                expression_entity: searcher.find_from_node_index(vrm_entity, node.node_index)?,
                index: node.morph_target_index,
                weight: node.weight,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use crate::tests::{TestResult, test_app};
    use crate::vrm::expressions::{
        BinaryExpression, BindExpressionNode, ClearExpressions, ExpressionBindingStatus,
        ExpressionCategory, ExpressionCategoryTag, ExpressionEntityMap, ExpressionMaterialBinds,
        ExpressionMaterialColorBind, ExpressionMetadata, ExpressionNode, ExpressionOverride,
        ExpressionOverrideSettings, ExpressionOverrideType, ExpressionTextureTransformBind,
        MaterialColorTarget, ModifyExpressions, RequestInitializeExpressions,
        RetargetExpressionNodes, SetExpressions, VrmExpressionPlugin, VrmExpressionRegistry,
        VrmMaterialBaseValues, VrmMaterialIndex, expression_metadata,
    };
    use bevy::ecs::system::RunSystemOnce;
    use bevy::math::Affine2;
    use bevy::prelude::*;

    fn default_override_settings() -> ExpressionOverrideSettings {
        ExpressionOverrideSettings {
            override_mouth: ExpressionOverrideType::None,
            override_blink: ExpressionOverrideType::None,
            override_look_at: ExpressionOverrideType::None,
        }
    }

    fn simple_metadata(
        _name: &str,
        index: usize,
    ) -> ExpressionMetadata {
        ExpressionMetadata {
            nodes: vec![ExpressionNode {
                node_index: 0,
                morph_target_index: index,
                weight: 1.0,
            }],
            category: ExpressionCategory::Other,
            override_settings: default_override_settings(),
            is_binary: false,
            declared_as_preset: true,
            material_color_binds: Vec::new(),
            texture_transform_binds: Vec::new(),
            unsupported_material_bind_count: 0,
        }
    }

    #[test]
    fn test_obtain_expression_nodes() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let vrm_entity = app
            .world_mut()
            .spawn((VrmExpressionRegistry(
                [(VrmExpression::from("happy"), simple_metadata("Test", 0))]
                    .into_iter()
                    .collect(),
            ),))
            .with_children(|c| {
                c.spawn((Name::new("Test"), VrmNodeIndex(0)));
            })
            .id();

        app.world_mut()
            .commands()
            .entity(vrm_entity)
            .trigger(RequestInitializeExpressions);
        app.update();

        app.world_mut()
            .run_system_once(move |s: ChildSearcher| s.find_expressions_root(vrm_entity))
            .expect("Failed to run system")
            .expect("Expression root not found");

        app.world_mut()
            .run_system_once(move |s: ChildSearcher| s.find_from_name(vrm_entity, "happy"))
            .expect("Failed to run system")
            .expect("Expression node not found");
        Ok(())
    }

    #[test]
    fn expression_binding_status_distinguishes_empty_and_resolved_binds() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let vrm_entity = app
            .world_mut()
            .spawn((VrmExpressionRegistry(
                [
                    (VrmExpression::from("effective"), simple_metadata("Test", 0)),
                    (
                        VrmExpression::from("empty"),
                        ExpressionMetadata {
                            nodes: Vec::new(),
                            ..simple_metadata("Test", 0)
                        },
                    ),
                ]
                .into_iter()
                .collect(),
            ),))
            .with_children(|c| {
                c.spawn((Name::new("Test"), VrmNodeIndex(0)));
            })
            .id();

        app.world_mut()
            .commands()
            .entity(vrm_entity)
            .trigger(RequestInitializeExpressions);
        app.update();

        let map = app
            .world()
            .get::<ExpressionEntityMap>(vrm_entity)
            .expect("expression map should be initialized");
        let effective = *map
            .0
            .get(&VrmExpression::from("effective"))
            .expect("effective expression should be present");
        let empty = *map
            .0
            .get(&VrmExpression::from("empty"))
            .expect("empty expression should be present");
        assert_eq!(
            app.world()
                .get::<ExpressionBindingStatus>(effective)
                .expect("status should be attached")
                .resolved_morph_bind_count,
            1
        );
        assert_eq!(
            app.world()
                .get::<ExpressionBindingStatus>(empty)
                .expect("status should be attached")
                .resolved_morph_bind_count,
            0
        );
        Ok(())
    }

    #[test]
    fn test_set_expressions() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let vrm_entity = app
            .world_mut()
            .spawn((VrmExpressionRegistry(
                [(VrmExpression::from("happy"), simple_metadata("Test", 0))]
                    .into_iter()
                    .collect(),
            ),))
            .with_children(|c| {
                c.spawn((Name::new("Test"), VrmNodeIndex(0)));
            })
            .id();

        app.world_mut()
            .commands()
            .entity(vrm_entity)
            .trigger(RequestInitializeExpressions);
        app.update();

        app.world_mut()
            .commands()
            .trigger(SetExpressions::single(vrm_entity, "happy", 0.8));
        app.update();

        let map = app.world().get::<ExpressionEntityMap>(vrm_entity).unwrap();
        let expr_entity = *map.0.get(&VrmExpression::from("happy")).unwrap();

        let override_val = app
            .world()
            .get::<ExpressionOverride>(expr_entity)
            .expect("ExpressionOverride not found");
        assert!((override_val.0 - 0.8).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_expression_entity_map_built_on_init() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let vrm_entity = app
            .world_mut()
            .spawn((VrmExpressionRegistry(
                [(VrmExpression::from("happy"), simple_metadata("Test", 0))]
                    .into_iter()
                    .collect(),
            ),))
            .with_children(|c| {
                c.spawn((Name::new("Test"), VrmNodeIndex(0)));
            })
            .id();

        app.world_mut()
            .commands()
            .entity(vrm_entity)
            .trigger(RequestInitializeExpressions);
        app.update();

        let map = app
            .world()
            .get::<ExpressionEntityMap>(vrm_entity)
            .expect("ExpressionEntityMap not found");

        assert!(map.0.contains_key(&VrmExpression::from("happy")));
        Ok(())
    }

    #[test]
    fn test_clear_expressions() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let vrm_entity = app
            .world_mut()
            .spawn((VrmExpressionRegistry(
                [(VrmExpression::from("happy"), simple_metadata("Test", 0))]
                    .into_iter()
                    .collect(),
            ),))
            .with_children(|c| {
                c.spawn((Name::new("Test"), VrmNodeIndex(0)));
            })
            .id();

        app.world_mut()
            .commands()
            .entity(vrm_entity)
            .trigger(RequestInitializeExpressions);
        app.update();

        app.world_mut()
            .commands()
            .trigger(SetExpressions::single(vrm_entity, "happy", 0.8));
        app.update();

        let map = app.world().get::<ExpressionEntityMap>(vrm_entity).unwrap();
        let expr_entity = *map.0.get(&VrmExpression::from("happy")).unwrap();
        assert!(app.world().get::<ExpressionOverride>(expr_entity).is_some());

        app.world_mut()
            .commands()
            .trigger(ClearExpressions { entity: vrm_entity });
        app.update();

        assert!(app.world().get::<ExpressionOverride>(expr_entity).is_none());
        Ok(())
    }

    #[test]
    fn test_set_expressions_replaces_previous() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let vrm_entity = app
            .world_mut()
            .spawn((VrmExpressionRegistry(
                [
                    (VrmExpression::from("happy"), simple_metadata("MeshA", 0)),
                    (VrmExpression::from("angry"), simple_metadata("MeshB", 0)),
                ]
                .into_iter()
                .collect(),
            ),))
            .with_children(|c| {
                c.spawn(Name::new("MeshA"));
                c.spawn(Name::new("MeshB"));
            })
            .id();

        app.world_mut()
            .commands()
            .entity(vrm_entity)
            .trigger(RequestInitializeExpressions);
        app.update();

        let map = app.world().get::<ExpressionEntityMap>(vrm_entity).unwrap();
        let happy_entity = *map.0.get(&VrmExpression::from("happy")).unwrap();
        let angry_entity = *map.0.get(&VrmExpression::from("angry")).unwrap();

        app.world_mut()
            .commands()
            .trigger(SetExpressions::single(vrm_entity, "happy", 1.0));
        app.update();

        assert!(
            app.world()
                .get::<ExpressionOverride>(happy_entity)
                .is_some()
        );
        assert!(
            app.world()
                .get::<ExpressionOverride>(angry_entity)
                .is_none()
        );

        app.world_mut()
            .commands()
            .trigger(SetExpressions::single(vrm_entity, "angry", 0.7));
        app.update();

        assert!(
            app.world()
                .get::<ExpressionOverride>(happy_entity)
                .is_none(),
            "Previous expression override should be removed"
        );
        let angry_override = app
            .world()
            .get::<ExpressionOverride>(angry_entity)
            .expect("New expression override not found");
        assert!((angry_override.0 - 0.7).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_bind_weight_applied() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let mesh_entity = app
            .world_mut()
            .spawn(MorphWeights::new(vec![0.0], None)?)
            .id();

        // bind.weight = 0.5, expression weight via transform = 0.8
        // expected: 0.8 * 0.5 = 0.4
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.8, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 0,
                weight: 0.5,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Other),
            default_override_settings(),
        ));
        app.update();

        let morph = app.world().get::<MorphWeights>(mesh_entity).unwrap();
        assert!(
            (morph.weights()[0] - 0.4).abs() < f32::EPSILON,
            "Expected 0.4, got {}",
            morph.weights()[0]
        );
        Ok(())
    }

    #[test]
    fn test_additive_accumulation() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let mesh_entity = app
            .world_mut()
            .spawn(MorphWeights::new(vec![0.0], None)?)
            .id();

        // Two expressions targeting the same morph index on the same mesh
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.3, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 0,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Other),
            default_override_settings(),
        ));
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 0,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Other),
            default_override_settings(),
        ));
        app.update();

        let morph = app.world().get::<MorphWeights>(mesh_entity).unwrap();
        assert!(
            (morph.weights()[0] - 0.8).abs() < f32::EPSILON,
            "Expected additive 0.3 + 0.5 = 0.8, got {}",
            morph.weights()[0]
        );
        Ok(())
    }

    #[test]
    fn test_override_block() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let mesh_entity = app
            .world_mut()
            .spawn(MorphWeights::new(vec![0.0, 0.0], None)?)
            .id();

        // "happy" expression with overrideMouth=block, weight=1.0
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 0,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Other),
            ExpressionOverrideSettings {
                override_mouth: ExpressionOverrideType::Block,
                override_blink: ExpressionOverrideType::None,
                override_look_at: ExpressionOverrideType::None,
            },
        ));
        // "aa" mouth expression, weight=0.7
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.7, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 1,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Mouth),
            default_override_settings(),
        ));
        app.update();

        let morph = app.world().get::<MorphWeights>(mesh_entity).unwrap();
        // "happy" at index 0: 1.0 (Other, no suppression)
        assert!(
            (morph.weights()[0] - 1.0).abs() < f32::EPSILON,
            "Expected happy=1.0, got {}",
            morph.weights()[0]
        );
        // "aa" at index 1: 0.0 (Mouth suppressed by block, multiplier=0.0)
        assert!(
            (morph.weights()[1] - 0.0).abs() < f32::EPSILON,
            "Expected mouth suppressed to 0.0, got {}",
            morph.weights()[1]
        );
        Ok(())
    }

    #[test]
    fn test_override_blend() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let mesh_entity = app
            .world_mut()
            .spawn(MorphWeights::new(vec![0.0, 0.0], None)?)
            .id();

        // Expression with overrideMouth=blend, weight=0.6
        // mouthRate += 0.6, mouthMul = 1.0 - 0.6 = 0.4
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.6, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 0,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Other),
            ExpressionOverrideSettings {
                override_mouth: ExpressionOverrideType::Blend,
                override_blink: ExpressionOverrideType::None,
                override_look_at: ExpressionOverrideType::None,
            },
        ));
        // Mouth expression, weight=1.0
        // finalWeight = 1.0 * 0.4 = 0.4
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 1,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Mouth),
            default_override_settings(),
        ));
        app.update();

        let morph = app.world().get::<MorphWeights>(mesh_entity).unwrap();
        assert!(
            (morph.weights()[0] - 0.6).abs() < f32::EPSILON,
            "Expected 0.6, got {}",
            morph.weights()[0]
        );
        assert!(
            (morph.weights()[1] - 0.4).abs() < f32::EPSILON,
            "Expected mouth attenuated to 0.4, got {}",
            morph.weights()[1]
        );
        Ok(())
    }

    #[test]
    fn test_is_binary() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let mesh_entity = app
            .world_mut()
            .spawn(MorphWeights::new(vec![0.0, 0.0], None)?)
            .id();

        // Binary expression with raw weight 0.3 → output 0.0
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.3, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 0,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Other),
            default_override_settings(),
            BinaryExpression,
        ));
        // Binary expression with raw weight 0.7 → output 1.0
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.7, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 1,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Other),
            default_override_settings(),
            BinaryExpression,
        ));
        app.update();

        let morph = app.world().get::<MorphWeights>(mesh_entity).unwrap();
        assert!(
            (morph.weights()[0] - 0.0).abs() < f32::EPSILON,
            "Expected binary threshold: 0.3 → 0.0, got {}",
            morph.weights()[0]
        );
        assert!(
            (morph.weights()[1] - 1.0).abs() < f32::EPSILON,
            "Expected binary threshold: 0.7 → 1.0, got {}",
            morph.weights()[1]
        );
        Ok(())
    }

    #[test]
    fn test_modify_expressions_preserves_existing() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let vrm_entity = app
            .world_mut()
            .spawn((VrmExpressionRegistry(
                [
                    (VrmExpression::from("happy"), simple_metadata("MeshA", 0)),
                    (VrmExpression::from("angry"), simple_metadata("MeshB", 0)),
                ]
                .into_iter()
                .collect(),
            ),))
            .with_children(|c| {
                c.spawn(Name::new("MeshA"));
                c.spawn(Name::new("MeshB"));
            })
            .id();

        app.world_mut()
            .commands()
            .entity(vrm_entity)
            .trigger(RequestInitializeExpressions);
        app.update();

        let map = app.world().get::<ExpressionEntityMap>(vrm_entity).unwrap();
        let happy_entity = *map.0.get(&VrmExpression::from("happy")).unwrap();
        let angry_entity = *map.0.get(&VrmExpression::from("angry")).unwrap();

        // Set happy via SetExpressions
        app.world_mut()
            .commands()
            .trigger(SetExpressions::single(vrm_entity, "happy", 1.0));
        app.update();

        assert!(
            app.world()
                .get::<ExpressionOverride>(happy_entity)
                .is_some()
        );

        // Modify angry — happy override should be preserved
        app.world_mut()
            .commands()
            .trigger(ModifyExpressions::single(vrm_entity, "angry", 0.7));
        app.update();

        // happy override is still present
        let happy_override = app
            .world()
            .get::<ExpressionOverride>(happy_entity)
            .expect("Existing override should be preserved by ModifyExpressions");
        assert!((happy_override.0 - 1.0).abs() < f32::EPSILON);

        // angry override was added
        let angry_override = app
            .world()
            .get::<ExpressionOverride>(angry_entity)
            .expect("ModifyExpressions should add new override");
        assert!((angry_override.0 - 0.7).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_is_binary_override_suppression() -> TestResult {
        let mut app = test_app();
        app.add_plugins(VrmExpressionPlugin);

        let mesh_entity = app
            .world_mut()
            .spawn(MorphWeights::new(vec![0.0, 0.0], None)?)
            .id();

        // Expression with overrideBlink=blend, weight=0.3
        // blinkRate += 0.3, blinkMul = 0.7
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(0.3, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 0,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Other),
            ExpressionOverrideSettings {
                override_mouth: ExpressionOverrideType::None,
                override_blink: ExpressionOverrideType::Blend,
                override_look_at: ExpressionOverrideType::None,
            },
        ));
        // Binary blink expression, weight=1.0
        // multiplier=0.7 < 1.0, binary → finalWeight = 0.0
        app.world_mut().spawn((
            Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            RetargetExpressionNodes(vec![BindExpressionNode {
                expression_entity: mesh_entity,
                index: 1,
                weight: 1.0,
            }]),
            ExpressionCategoryTag(ExpressionCategory::Blink),
            default_override_settings(),
            BinaryExpression,
        ));
        app.update();

        let morph = app.world().get::<MorphWeights>(mesh_entity).unwrap();
        assert!(
            (morph.weights()[1] - 0.0).abs() < f32::EPSILON,
            "Expected binary blink fully suppressed to 0.0, got {}",
            morph.weights()[1]
        );
        Ok(())
    }

    fn spawn_material_expression(app: &mut App) -> (Entity, bevy::asset::AssetId<MToonMaterial>) {
        app.add_plugins(VrmExpressionPlugin);
        app.init_asset::<StandardMaterial>();
        app.init_asset::<MToonMaterial>();
        let mut material = MToonMaterial::default();
        material.base_color = Color::linear_rgba(0.2, 0.4, 0.6, 1.0);
        material.emissive = LinearRgba::new(0.05, 0.1, 0.15, 1.0);
        material.shade.color = LinearRgba::new(0.3, 0.2, 0.1, 1.0);
        material.uv_transform =
            Affine2::from_scale_angle_translation(Vec2::new(1.5, 2.0), 0.0, Vec2::new(0.1, 0.2));
        let base = VrmMaterialBaseValues::from_mtoon(&material);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<MToonMaterial>>()
            .add(material);
        app.world_mut()
            .spawn((MeshMaterial3d(handle.clone()), VrmMaterialIndex(0), base));
        let expression = app
            .world_mut()
            .spawn((
                ExpressionMaterialBinds {
                    colors: vec![
                        ExpressionMaterialColorBind {
                            material_index: 0,
                            target: MaterialColorTarget::BaseColor,
                            target_value: LinearRgba::new(0.8, 0.1, 0.2, 1.0),
                        },
                        ExpressionMaterialColorBind {
                            material_index: 0,
                            target: MaterialColorTarget::ShadeColor,
                            target_value: LinearRgba::new(0.9, 0.8, 0.7, 1.0),
                        },
                    ],
                    transforms: vec![ExpressionTextureTransformBind {
                        material_index: 0,
                        scale: Vec2::new(2.0, 3.0),
                        offset: Vec2::new(0.25, -0.5),
                    }],
                },
                ExpressionCategoryTag(ExpressionCategory::Other),
                Transform::default(),
                default_override_settings(),
                ExpressionOverride(1.0),
            ))
            .id();
        (expression, handle.id())
    }

    fn assert_close(
        actual: f32,
        expected: f32,
        context: &str,
    ) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "{context}: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn material_color_and_uv_follow_weight_and_restore_base() -> TestResult {
        let mut app = test_app();
        let (expression, material_id) = spawn_material_expression(&mut app);

        for _ in 0..3 {
            app.world_mut()
                .entity_mut(expression)
                .insert(ExpressionOverride(1.0));
            app.update();
            {
                let materials = app.world().resource::<Assets<MToonMaterial>>();
                let material = materials.get(material_id).unwrap();
                let color = material.base_color.to_linear();
                assert_close(color.red, 0.8, "base color red at weight 1");
                assert_close(color.green, 0.1, "base color green at weight 1");
                assert_close(material.shade.color.red, 0.9, "shade color red at weight 1");
                let sample = material.uv_transform.transform_point2(Vec2::ONE);
                assert_close(sample.x, 3.45, "uv x at weight 1");
                assert_close(sample.y, 6.1, "uv y at weight 1");
            }

            app.world_mut()
                .entity_mut(expression)
                .insert(ExpressionOverride(0.0));
            app.update();
            {
                let materials = app.world().resource::<Assets<MToonMaterial>>();
                let material = materials.get(material_id).unwrap();
                let color = material.base_color.to_linear();
                assert_close(color.red, 0.2, "base color red at weight 0");
                assert_close(color.green, 0.4, "base color green at weight 0");
                assert_close(color.blue, 0.6, "base color blue at weight 0");
                assert_close(material.shade.color.red, 0.3, "shade color restored");
                assert_close(material.emissive.red, 0.05, "emissive restored");
                let sample = material.uv_transform.transform_point2(Vec2::ONE);
                assert_close(sample.x, 1.6, "uv x restored");
                assert_close(sample.y, 2.2, "uv y restored");
            }
        }
        Ok(())
    }

    #[test]
    fn material_color_bind_accumulates_across_two_expressions() -> TestResult {
        let mut app = test_app();
        let (_expression, material_id) = spawn_material_expression(&mut app);
        // A second expression doubles the delta on base color.
        app.world_mut().spawn((
            ExpressionMaterialBinds {
                colors: vec![ExpressionMaterialColorBind {
                    material_index: 0,
                    target: MaterialColorTarget::BaseColor,
                    target_value: LinearRgba::new(0.8, 0.1, 0.2, 1.0),
                }],
                transforms: Vec::new(),
            },
            ExpressionCategoryTag(ExpressionCategory::Other),
            Transform::default(),
            default_override_settings(),
            ExpressionOverride(1.0),
        ));
        app.update();

        let materials = app.world().resource::<Assets<MToonMaterial>>();
        let color = materials.get(material_id).unwrap().base_color.to_linear();
        // base 0.2 + (0.8 - 0.2) + (0.8 - 0.2) = 1.4 (added twice by design).
        assert_close(color.red, 1.4, "accumulated base color");
        Ok(())
    }

    #[test]
    fn expression_metadata_counts_material_binds_and_unknown_targets() {
        use crate::vrm::gltf::extensions::vrmc_vrm::{
            MaterialColorBind, TextureTransformBind, VrmPreset,
        };
        let preset = VrmPreset {
            is_binary: false,
            morph_target_binds: None,
            material_color_binds: vec![
                MaterialColorBind {
                    material: 2,
                    bind_type: "shadeColor".into(),
                    target_value: [0.1, 0.2, 0.3, 1.0],
                },
                MaterialColorBind {
                    material: 2,
                    bind_type: "notAStandardTarget".into(),
                    target_value: [0.0; 4],
                },
            ],
            texture_transform_binds: vec![TextureTransformBind {
                material: 2,
                scale: None,
                offset: None,
            }],
            override_blink: "none".into(),
            override_look_at: "none".into(),
            override_mouth: "none".into(),
        };
        let metadata = expression_metadata("customThing", &preset, false);
        assert!(!metadata.declared_as_preset);
        assert_eq!(metadata.material_color_binds.len(), 1);
        assert_eq!(metadata.texture_transform_binds.len(), 1);
        assert_eq!(metadata.unsupported_material_bind_count, 1);
        // VRM defaults: scale = [1, 1], offset = [0, 0].
        assert_eq!(metadata.texture_transform_binds[0].scale, Vec2::ONE);
        assert_eq!(metadata.texture_transform_binds[0].offset, Vec2::ZERO);
    }
}
