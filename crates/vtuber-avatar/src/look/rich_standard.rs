use bevy::asset::{Handle, load_internal_asset, uuid_handle};
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::{Shader, ShaderRef};

#[cfg(test)]
use super::AvatarLookSettings;
use super::RichLookSettings;
#[cfg(test)]
use super::rich_material::RichMaterialSwap;
use super::rich_material::{RichMaterialExtension, register_rich_material_systems};
#[cfg(test)]
use crate::lifecycle::AvatarLifecycle;
const RICH_STANDARD_FRAGMENT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("64a8f4b1-7a13-4f72-8ac3-0d5bba3de8a2");

#[derive(Asset, AsBindGroup, Clone, Debug, Reflect)]
#[doc(hidden)]
pub struct RichStandardExtension {
    #[uniform(100)]
    strength: f32,
}

impl MaterialExtension for RichStandardExtension {
    fn fragment_shader() -> ShaderRef {
        RICH_STANDARD_FRAGMENT_SHADER_HANDLE.into()
    }
}

type RichStandardMaterial = ExtendedMaterial<StandardMaterial, RichStandardExtension>;

pub(crate) fn register_rich_standard(app: &mut App) {
    app.add_plugins(MaterialPlugin::<RichStandardMaterial>::default());
    load_internal_asset!(
        app,
        RICH_STANDARD_FRAGMENT_SHADER_HANDLE,
        "standard_rich.wgsl",
        Shader::from_wgsl
    );
    register_rich_material_systems::<RichStandardExtension>(app);
}

impl RichMaterialExtension for RichStandardExtension {
    type Native = StandardMaterial;
    fn create(native: &StandardMaterial, settings: RichLookSettings) -> Option<Self> {
        (!native.unlit).then_some(Self {
            strength: settings.strength(),
        })
    }
    fn matches(
        current: &RichStandardMaterial,
        native: &StandardMaterial,
        settings: RichLookSettings,
    ) -> bool {
        current.base.base_color == native.base_color
            && current.base.emissive == native.emissive
            && current.base.uv_transform == native.uv_transform
            && current.extension.strength
                == if settings.enabled() {
                    settings.strength()
                } else {
                    0.0
                }
    }
    fn sync(
        current: &mut RichStandardMaterial,
        native: &StandardMaterial,
        settings: RichLookSettings,
    ) {
        current.base.base_color = native.base_color;
        current.base.emissive = native.emissive;
        current.base.uv_transform = native.uv_transform;
        current.extension.strength = if settings.enabled() {
            settings.strength()
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )] // tests may panic (AGENTS.md)
    use super::*;
    use bevy::asset::AssetPlugin;
    use bevy::color::LinearRgba;
    use bevy::math::{Affine2, Vec2};
    use bevy::pbr::MeshMaterial3d;

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<StandardMaterial>()
            .init_asset::<RichStandardMaterial>()
            .init_resource::<AvatarLifecycle>()
            .init_resource::<AvatarLookSettings>();
        register_rich_material_systems::<RichStandardExtension>(&mut app);
        app
    }

    fn enter_ready(app: &mut App, root: Entity) {
        let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
    }

    fn native_material(app: &mut App, unlit: bool) -> Handle<StandardMaterial> {
        let mut materials = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
        materials.add(StandardMaterial {
            base_color: Color::LinearRgba(LinearRgba::new(0.2, 0.3, 0.4, 0.8)),
            emissive: LinearRgba::new(0.05, 0.06, 0.07, 1.0),
            perceptual_roughness: 0.23,
            metallic: 0.4,
            reflectance: 0.6,
            clearcoat: 0.7,
            uv_transform: Affine2::from_scale_angle_translation(
                Vec2::new(2.0, 3.0),
                0.2,
                Vec2::new(0.1, 0.2),
            ),
            unlit,
            ..default()
        })
    }

    fn spawn_avatar_mesh(app: &mut App, handle: &Handle<StandardMaterial>) -> (Entity, Entity) {
        let root = app.world_mut().spawn_empty().id();
        let mesh = app
            .world_mut()
            .spawn((ChildOf(root), MeshMaterial3d(handle.clone())))
            .id();
        enter_ready(app, root);
        (root, mesh)
    }

    fn set_look(app: &mut App, enabled: bool, strength: f32) {
        app.world_mut().resource_mut::<AvatarLookSettings>().0 =
            crate::look::RichLookSettings::try_new(enabled, strength).unwrap();
    }

    fn rich_handle(app: &mut App, mesh: Entity) -> Option<Handle<RichStandardMaterial>> {
        app.world()
            .get::<MeshMaterial3d<RichStandardMaterial>>(mesh)
            .map(|material| material.0.clone())
    }

    fn native_handle(app: &mut App, mesh: Entity) -> Option<Handle<StandardMaterial>> {
        app.world()
            .get::<MeshMaterial3d<StandardMaterial>>(mesh)
            .map(|material| material.0.clone())
    }

    #[test]
    fn enabling_swaps_lit_standard_and_preserves_author_material() {
        let mut app = test_app();
        let native = native_material(&mut app, false);
        let (root, mesh) = spawn_avatar_mesh(&mut app, &native);
        let shared_mesh = app
            .world_mut()
            .spawn((ChildOf(root), MeshMaterial3d(native.clone())))
            .id();
        set_look(&mut app, true, 0.5);
        app.update();

        let rich = rich_handle(&mut app, mesh).expect("mesh renders the rich material");
        assert_eq!(
            rich_handle(&mut app, shared_mesh).map(|handle| handle.id()),
            Some(rich.id())
        );
        assert!(native_handle(&mut app, mesh).is_none());
        let swap = app
            .world()
            .get::<RichMaterialSwap<RichStandardExtension>>(mesh)
            .expect("swap component keeps the native handle");
        assert_eq!(swap.native.id(), native.id());

        let material = app
            .world()
            .resource::<Assets<RichStandardMaterial>>()
            .get(rich.id())
            .expect("rich asset exists");
        let expected = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(native.id())
            .expect("native asset exists");
        assert_eq!(material.base.base_color, expected.base_color);
        assert_eq!(material.base.emissive, expected.emissive);
        assert_eq!(
            material.base.perceptual_roughness,
            expected.perceptual_roughness
        );
        assert_eq!(material.base.metallic, expected.metallic);
        assert_eq!(material.base.reflectance, expected.reflectance);
        assert_eq!(material.base.clearcoat, expected.clearcoat);
        assert_eq!(material.base.uv_transform, expected.uv_transform);
    }

    #[test]
    fn unlit_standard_stays_native() {
        let mut app = test_app();
        let native = native_material(&mut app, true);
        let (_, mesh) = spawn_avatar_mesh(&mut app, &native);
        set_look(&mut app, true, 1.0);
        app.update();

        assert_eq!(
            native_handle(&mut app, mesh).map(|handle| handle.id()),
            Some(native.id())
        );
        assert!(rich_handle(&mut app, mesh).is_none());
        assert_eq!(
            app.world()
                .resource::<Assets<RichStandardMaterial>>()
                .iter()
                .count(),
            0
        );
    }

    #[test]
    fn zero_strength_keeps_rich_selected_and_updates_without_new_assets() {
        let mut app = test_app();
        let native = native_material(&mut app, false);
        let (_, mesh) = spawn_avatar_mesh(&mut app, &native);
        set_look(&mut app, true, 0.0);
        app.update();

        let before = rich_handle(&mut app, mesh).expect("rich material");
        assert_eq!(
            app.world()
                .resource::<Assets<RichStandardMaterial>>()
                .get(before.id())
                .expect("rich asset exists")
                .extension
                .strength,
            0.0
        );

        set_look(&mut app, true, 0.5);
        app.update();
        let after = rich_handle(&mut app, mesh).expect("rich material");
        assert_eq!(after.id(), before.id());
        assert_eq!(
            app.world()
                .resource::<Assets<RichStandardMaterial>>()
                .get(after.id())
                .expect("rich asset exists")
                .extension
                .strength,
            0.5
        );
        assert_eq!(
            app.world()
                .resource::<Assets<RichStandardMaterial>>()
                .iter()
                .count(),
            1
        );
    }

    #[test]
    fn current_native_values_sync_to_rich_and_survive_disabling() {
        let mut app = test_app();
        let native = native_material(&mut app, false);
        let (_, mesh) = spawn_avatar_mesh(&mut app, &native);
        set_look(&mut app, true, 1.0);
        app.update();
        let rich = rich_handle(&mut app, mesh).expect("rich material");
        let uv_transform =
            Affine2::from_scale_angle_translation(Vec2::new(4.0, 5.0), 0.4, Vec2::splat(0.3));

        {
            let mut materials = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
            let mut material = materials.get_mut(native.id()).expect("native asset");
            material.base_color = Color::LinearRgba(LinearRgba::new(0.9, 0.1, 0.1, 0.7));
            material.emissive = LinearRgba::new(0.2, 0.1, 0.05, 1.0);
            material.uv_transform = uv_transform;
        }
        app.update();

        let material = app
            .world()
            .resource::<Assets<RichStandardMaterial>>()
            .get(rich.id())
            .expect("rich asset exists");
        assert_eq!(
            material.base.base_color,
            Color::LinearRgba(LinearRgba::new(0.9, 0.1, 0.1, 0.7))
        );
        assert_eq!(material.base.emissive, LinearRgba::new(0.2, 0.1, 0.05, 1.0));
        assert_eq!(material.base.uv_transform, uv_transform);

        set_look(&mut app, false, 1.0);
        app.update();
        assert_eq!(
            native_handle(&mut app, mesh).map(|handle| handle.id()),
            Some(native.id())
        );
        let native = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(native.id())
            .expect("restored native asset");
        assert_eq!(
            native.base_color,
            Color::LinearRgba(LinearRgba::new(0.9, 0.1, 0.1, 0.7))
        );
        assert_eq!(native.uv_transform, uv_transform);
    }
}
