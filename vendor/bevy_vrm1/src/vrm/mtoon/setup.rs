use crate::prelude::*;
use crate::vrm::expressions::{
    VrmMaterialAppliedValues, VrmMaterialBaseValues, VrmMaterialIndex,
};
use crate::vrm::gltf::materials::LegacyAlphaMode;
use bevy::app::{App, Plugin};
use bevy::asset::Assets;
use bevy::math::{Affine2, Vec2};
use bevy::prelude::*;
use bevy::render::render_resource::Face;

pub struct MToonMaterialSetupPlugin;

impl Plugin for MToonMaterialSetupPlugin {
    fn build(
        &self,
        app: &mut App,
    ) {
        app.add_systems(Update, turn_to_mtoon_material);
    }
}

fn turn_to_mtoon_material(
    mut commands: Commands,
    mut mtoon_materials: ResMut<Assets<MToonMaterial>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    registries: Query<&VrmcMaterialRegistry>,
    parents: Query<&ChildOf>,
    added_materials: Query<
        (Entity, &MeshMaterial3d<StandardMaterial>),
        Added<MeshMaterial3d<StandardMaterial>>,
    >,
) {
    added_materials.iter().for_each(|(entity, handle)| {
        let root = parents.root_ancestor(entity);
        let Ok(registry) = registries.get(root) else {
            return;
        };
        let index = registry.indices.get(&handle.id()).copied();
        let Some(extension) = registry.materials.get(&handle.id()) else {
            // Plain PBR/unlit/unknown materials are neither converted nor
            // modified, but they still register their glTF index and base
            // values so expression binds can resolve and restore faithfully.
            if let (Some(index), Some(material)) = (index, standard_materials.get(handle.id())) {
                let base = VrmMaterialBaseValues::from_standard(material);
                commands.entity(entity).insert((
                    VrmMaterialIndex(index),
                    base,
                    VrmMaterialAppliedValues(base),
                ));
            }
            return;
        };
        // The unlit/Standard fallback keeps the `StandardMaterial` asset; it
        // still needs the glTF material index and captured base values so
        // expression color/UV binds can restore and apply.
        if extension.legacy_standard_fallback {
            if let Some(mut material) = standard_materials.get_mut(handle.id()) {
                material.unlit = true;
                material.base_color_texture = extension
                    .legacy_base_texture
                    .and_then(|index| registry.images.get(index))
                    .cloned()
                    .or_else(|| material.base_color_texture.clone());
                if let Some(color) = extension.legacy_base_color {
                    material.base_color =
                        Color::linear_rgba(color[0], color[1], color[2], color[3]);
                }
                if let Some(alpha_mode) = extension.legacy_alpha_mode {
                    material.alpha_mode = match alpha_mode {
                        LegacyAlphaMode::Opaque => AlphaMode::Opaque,
                        LegacyAlphaMode::Mask(cutoff) => AlphaMode::Mask(cutoff),
                        LegacyAlphaMode::Blend => AlphaMode::Blend,
                    };
                }
                if let Some(double_sided) = extension.legacy_double_sided {
                    material.double_sided = double_sided;
                    material.cull_mode = if double_sided { None } else { Some(Face::Back) };
                }
                if let Some(transform) = extension.legacy_uv_transform {
                    material.uv_transform = Affine2::from_scale_angle_translation(
                        Vec2::new(transform[0], transform[1]),
                        0.0,
                        Vec2::new(transform[2], transform[3]),
                    );
                }
                if let Some(index) = index {
                    let base = VrmMaterialBaseValues::from_standard(&material);
                    commands.entity(entity).insert((
                        VrmMaterialIndex(index),
                        base,
                        VrmMaterialAppliedValues(base),
                    ));
                }
            }
            return;
        }
        let Some(base) = standard_materials.get(handle.id()).cloned() else {
            return;
        };
        let legacy_double_sided = extension.legacy_double_sided;
        let material = MToonMaterial {
            base_color_texture: extension
                .legacy_base_texture
                .and_then(|index| registry.images.get(index))
                .cloned()
                .or_else(|| base.base_color_texture.clone()),
            uv_animation_mask_texture: extension
                .uv_animation_mask_texture
                .and_then(|tex| registry.images.get(tex.index))
                .cloned(),
            shade_multiply_texture: extension
                .shade_multiply_texture
                .and_then(|tex| registry.images.get(tex.index))
                .cloned(),
            shading_shift_texture: extension
                .shading_shift_texture
                .and_then(|tex| registry.images.get(tex.index))
                .cloned(),
            matcap_texture: extension
                .matcap_texture
                .and_then(|tex| registry.images.get(tex.index))
                .cloned(),
            rim_multiply_texture: extension
                .rim_multiply_texture
                .and_then(|tex| registry.images.get(tex.index))
                .cloned(),
            outline_width_multiply_texture: extension
                .outline_width_multiply_texture
                .and_then(|tex| registry.images.get(tex.index))
                .cloned(),
            normal_texture: extension
                .legacy_normal_texture
                .and_then(|index| registry.images.get(index))
                .cloned()
                .or_else(|| base.normal_map_texture.clone()),
            normal_texture_scale: extension.legacy_normal_scale.unwrap_or(1.0),
            shade: Shade::from(extension),
            outline: MToonOutline::from(extension),
            rim_lighting: RimLighting::from(extension),
            uv_animation: UVAnimation::from(extension),
            gi_equalization_factor: extension.gi_equalization_factor,
            double_sided: legacy_double_sided.unwrap_or(base.double_sided),
            alpha_mode: extension
                .legacy_alpha_mode
                .map(|mode| match mode {
                    LegacyAlphaMode::Opaque => AlphaMode::Opaque,
                    LegacyAlphaMode::Mask(cutoff) => AlphaMode::Mask(cutoff),
                    LegacyAlphaMode::Blend => AlphaMode::Blend,
                })
                .unwrap_or(base.alpha_mode),
            depth_bias: base.depth_bias,
            render_queue_offset: extension.render_queue_offset_number,
            transparent_with_z_write: extension.transparent_with_z_write,
            opaque_renderer_method: base.opaque_render_method,
            base_color: extension
                .legacy_base_color
                .map(|color| Color::linear_rgba(color[0], color[1], color[2], color[3]))
                .unwrap_or(base.base_color),
            cull_mode: if legacy_double_sided == Some(true) {
                None
            } else {
                base.cull_mode
            },
            emissive: extension
                .legacy_emissive
                .map(|color| LinearRgba::rgb(color[0], color[1], color[2]))
                .unwrap_or(base.emissive),
            emissive_texture: extension
                .legacy_emissive_texture
                .and_then(|index| registry.images.get(index))
                .cloned()
                .or_else(|| base.emissive_texture.clone()),
            uv_transform: extension
                .legacy_uv_transform
                .map(|transform| {
                    Affine2::from_scale_angle_translation(
                        Vec2::new(transform[0], transform[1]),
                        0.0,
                        Vec2::new(transform[2], transform[3]),
                    )
                })
                .unwrap_or(base.uv_transform),
        };
        let base_values = VrmMaterialBaseValues::from_mtoon(&material);
        let mut cmd = commands.entity(entity);
        cmd.remove::<MeshMaterial3d<StandardMaterial>>()
            .insert(MeshMaterial3d(mtoon_materials.add(material)));
        if let Some(index) = index {
            cmd.insert((
                VrmMaterialIndex(index),
                base_values,
                VrmMaterialAppliedValues(base_values),
            ));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_app;
    use bevy::app::Update;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn main_texture_fallback_reaches_real_mtoon_material_setup() {
        let source = json!({
            "shader": "VRM/MToon",
            "textureProperties": {"_MainTex": 0}
        });
        let extension =
            crate::vrm::gltf::materials::convert_legacy_material_properties_with_texture_count(
                &source,
                Some(1),
            )
            .expect("legacy material should convert");

        let mut app = test_app();
        app.init_asset::<StandardMaterial>();
        app.init_asset::<MToonMaterial>();
        let standard_handle = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let image_handle = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(Image::default());
        let root = app
            .world_mut()
            .spawn(VrmcMaterialRegistry {
                images: vec![image_handle.clone()],
                materials: HashMap::from([(standard_handle.id(), extension)]),
                indices: HashMap::from([(standard_handle.id(), 0)]),
            })
            .id();
        let material_entity = app
            .world_mut()
            .spawn((MeshMaterial3d(standard_handle), ChildOf(root)))
            .id();
        app.add_systems(Update, turn_to_mtoon_material);

        app.update();

        let material_id = app
            .world()
            .get::<MeshMaterial3d<MToonMaterial>>(material_entity)
            .expect("setup must replace StandardMaterial with MToonMaterial")
            .0
            .id();
        let material = app
            .world()
            .resource::<Assets<MToonMaterial>>()
            .get(material_id)
            .expect("setup must create the MToon asset");
        assert_eq!(material.shade_multiply_texture, Some(image_handle));
        assert_eq!(
            app.world()
                .get::<VrmMaterialIndex>(material_entity)
                .map(|index| index.0),
            Some(0)
        );
        assert!(
            app.world()
                .get::<VrmMaterialBaseValues>(material_entity)
                .is_some()
        );
    }

    #[test]
    fn legacy_bump_map_reaches_real_mtoon_material_setup() {
        let source = json!({
            "shader": "VRM/MToon",
            "floatProperties": {"_BumpScale": 0.5},
            "textureProperties": {"_BumpMap": 1}
        });
        let extension =
            crate::vrm::gltf::materials::convert_legacy_material_properties_with_texture_count(
                &source,
                Some(2),
            )
            .expect("legacy material should convert");

        let mut app = test_app();
        app.init_asset::<StandardMaterial>();
        app.init_asset::<MToonMaterial>();
        let standard_handle = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let normal_handle = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(Image::default());
        let other_handle = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(Image::default());
        let root = app
            .world_mut()
            .spawn(VrmcMaterialRegistry {
                images: vec![other_handle, normal_handle.clone()],
                materials: HashMap::from([(standard_handle.id(), extension)]),
                indices: HashMap::from([(standard_handle.id(), 0)]),
            })
            .id();
        let material_entity = app
            .world_mut()
            .spawn((MeshMaterial3d(standard_handle), ChildOf(root)))
            .id();
        app.add_systems(Update, turn_to_mtoon_material);

        app.update();

        let material_id = app
            .world()
            .get::<MeshMaterial3d<MToonMaterial>>(material_entity)
            .expect("setup must replace StandardMaterial with MToonMaterial")
            .0
            .id();
        let material = app
            .world()
            .resource::<Assets<MToonMaterial>>()
            .get(material_id)
            .expect("setup must create the MToon asset");
        assert_eq!(material.normal_texture, Some(normal_handle));
        assert_eq!(material.normal_texture_scale, 0.5);
    }

    fn material_metadata(
        colors: Vec<crate::vrm::expressions::ExpressionMaterialColorBind>,
        transforms: Vec<crate::vrm::expressions::ExpressionTextureTransformBind>,
    ) -> crate::vrm::expressions::ExpressionMetadata {
        use crate::vrm::expressions::{
            ExpressionCategory, ExpressionMetadata, ExpressionOverrideSettings,
            ExpressionOverrideType,
        };
        ExpressionMetadata {
            nodes: Vec::new(),
            category: ExpressionCategory::Other,
            override_settings: ExpressionOverrideSettings {
                override_mouth: ExpressionOverrideType::None,
                override_blink: ExpressionOverrideType::None,
                override_look_at: ExpressionOverrideType::None,
            },
            is_binary: false,
            declared_as_preset: true,
            material_color_binds: colors,
            texture_transform_binds: transforms,
            unsupported_material_bind_count: 0,
        }
    }

    fn color_bind(
        material_index: usize,
        target: crate::vrm::expressions::MaterialColorTarget,
    ) -> crate::vrm::expressions::ExpressionMaterialColorBind {
        crate::vrm::expressions::ExpressionMaterialColorBind {
            material_index,
            target,
            target_value: LinearRgba::new(0.8, 0.2, 0.2, 1.0),
        }
    }

    fn transform_bind(material_index: usize) -> crate::vrm::expressions::ExpressionTextureTransformBind {
        crate::vrm::expressions::ExpressionTextureTransformBind {
            material_index,
            scale: Vec2::new(2.0, 2.0),
            offset: Vec2::new(0.1, 0.1),
        }
    }

    fn spawn_material_root(
        app: &mut App,
        index: usize,
        extension: Option<crate::vrm::gltf::materials::VrmcMaterialsExtensitions>,
        entries: Vec<(&str, crate::vrm::expressions::ExpressionMetadata)>,
    ) -> Entity {
        use crate::vrm::expressions::{
            RequestInitializeExpressions, VrmExpressionRegistry,
        };
        let handle = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let mut materials = HashMap::new();
        if let Some(extension) = extension {
            materials.insert(handle.id(), extension);
        }
        let root = app
            .world_mut()
            .spawn(VrmcMaterialRegistry {
                images: Vec::new(),
                materials,
                indices: HashMap::from([(handle.id(), index)]),
            })
            .id();
        app.world_mut()
            .spawn((MeshMaterial3d(handle), ChildOf(root)));
        let registry = entries
            .into_iter()
            .map(|(name, metadata)| (VrmExpression::from(name), metadata))
            .collect();
        app.world_mut()
            .entity_mut(root)
            .insert(VrmExpressionRegistry(registry));
        app.world_mut()
            .commands()
            .entity(root)
            .trigger(RequestInitializeExpressions);
        root
    }

    fn expression_status(
        app: &App,
        root: Entity,
        name: &str,
    ) -> crate::vrm::expressions::ExpressionBindingStatus {
        use crate::vrm::expressions::{ExpressionBindingStatus, ExpressionEntityMap};
        let map = app
            .world()
            .get::<ExpressionEntityMap>(root)
            .expect("expression map");
        let entity = *map
            .0
            .get(&VrmExpression::from(name))
            .expect("expression entity");
        *app.world()
            .get::<ExpressionBindingStatus>(entity)
            .expect("binding status")
    }

    #[test]
    fn expression_material_resolution_uses_real_material_kinds() {
        use crate::vrm::expressions::MaterialColorTarget;
        let mtoon_extension =
            crate::vrm::gltf::materials::convert_legacy_material_properties_with_texture_count(
                &json!({"shader": "VRM/MToon"}),
                Some(0),
            )
            .expect("mtoon extension");
        let mut unlit_extension =
            crate::vrm::gltf::materials::convert_legacy_material_properties_with_texture_count(
                &json!({"shader": "VRM/UnlitTexture"}),
                Some(0),
            )
            .expect("unlit extension");
        unlit_extension.legacy_standard_fallback = true;

        let mut app = test_app();
        app.add_plugins(crate::vrm::expressions::VrmExpressionPlugin);
        app.init_asset::<StandardMaterial>();
        app.init_asset::<MToonMaterial>();
        app.add_systems(Update, turn_to_mtoon_material);

        // Plain VRM 1.0 PBR material: base/emission and UV are representable,
        // shade/rim/outline are not.
        let pbr_root = spawn_material_root(
            &mut app,
            0,
            None,
            vec![
                (
                    "colorOnly",
                    material_metadata(vec![color_bind(0, MaterialColorTarget::BaseColor)], vec![]),
                ),
                ("uvOnly", material_metadata(vec![], vec![transform_bind(0)])),
                (
                    "shadeOnly",
                    material_metadata(vec![color_bind(0, MaterialColorTarget::ShadeColor)], vec![]),
                ),
                (
                    "missing",
                    material_metadata(vec![color_bind(9, MaterialColorTarget::BaseColor)], vec![]),
                ),
            ],
        );
        // VRM 0.x unlit fallback also stays a `StandardMaterial`.
        let unlit_root = spawn_material_root(
            &mut app,
            0,
            Some(unlit_extension),
            vec![
                (
                    "colorOnly",
                    material_metadata(vec![color_bind(0, MaterialColorTarget::BaseColor)], vec![]),
                ),
                (
                    "shadeOnly",
                    material_metadata(vec![color_bind(0, MaterialColorTarget::ShadeColor)], vec![]),
                ),
            ],
        );
        // MToon represents every standard color target.
        let mtoon_root = spawn_material_root(
            &mut app,
            0,
            Some(mtoon_extension),
            vec![(
                "shadeOnly",
                material_metadata(vec![color_bind(0, MaterialColorTarget::ShadeColor)], vec![]),
            )],
        );

        app.update();

        let pbr_color = expression_status(&app, pbr_root, "colorOnly");
        assert_eq!(pbr_color.resolved_material_bind_count, 1);
        assert_eq!(pbr_color.unresolved_material_bind_count, 0);
        assert_eq!(pbr_color.unsupported_material_bind_count, 0);
        assert_eq!(pbr_color.declared_material_bind_count, 1);

        let pbr_uv = expression_status(&app, pbr_root, "uvOnly");
        assert_eq!(pbr_uv.resolved_material_bind_count, 1);

        let pbr_shade = expression_status(&app, pbr_root, "shadeOnly");
        assert_eq!(pbr_shade.resolved_material_bind_count, 0);
        assert_eq!(pbr_shade.unsupported_material_bind_count, 1);

        let pbr_missing = expression_status(&app, pbr_root, "missing");
        assert_eq!(pbr_missing.resolved_material_bind_count, 0);
        assert_eq!(pbr_missing.unresolved_material_bind_count, 1);

        assert_eq!(
            expression_status(&app, unlit_root, "colorOnly").resolved_material_bind_count,
            1
        );
        assert_eq!(
            expression_status(&app, unlit_root, "shadeOnly").unsupported_material_bind_count,
            1
        );

        let mtoon_shade = expression_status(&app, mtoon_root, "shadeOnly");
        assert_eq!(mtoon_shade.resolved_material_bind_count, 1);
        assert_eq!(mtoon_shade.unsupported_material_bind_count, 0);

        // The real setup system registers index/base for every scene material,
        // including plain Standard PBR and the unlit fallback.
        let mut meshes = app.world_mut().query::<(
            &VrmMaterialIndex,
            &VrmMaterialBaseValues,
            Option<&MeshMaterial3d<MToonMaterial>>,
        )>();
        let registered: Vec<_> = meshes.iter(app.world()).collect();
        assert_eq!(registered.len(), 3, "all three material meshes registered");
        assert_eq!(
            registered
                .iter()
                .filter(|(_, _, mtoon)| mtoon.is_some())
                .count(),
            1,
            "only the MToon material converts"
        );
    }
}
