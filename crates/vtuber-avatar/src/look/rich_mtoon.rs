//! The app-side Rich MToon material and Native/Rich switch (#94).
//!
//! OFF uses the transparency-fixed upstream `MToonMaterial`. ON wraps the
//! current native material in Bevy's `ExtendedMaterial`, which keeps the
//! upstream bindings, alpha/depth behavior, and specialization while replacing
//! the fragment lighting with bounded cel shading. WESL imports the Native
//! helpers for material inputs and author effects. The native
//! asset remains the owner of current expression and UV values.

use bevy::asset::{Handle, load_internal_asset, uuid_handle};
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::{Shader, ShaderRef};
use bevy_vrm1::prelude::MToonMaterial;

#[cfg(test)]
use super::AvatarLookSettings;
use super::RichLookSettings;
#[cfg(test)]
use super::rich_material::RichMaterialSwap;
use super::rich_material::{RichMaterialExtension, register_rich_material_systems};
#[cfg(test)]
use crate::lifecycle::AvatarLifecycle;

pub(crate) const RICH_MTOON_FRAGMENT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("7c2f9a41-6d3b-4e58-9a17-0f2c8d5b1e63");
pub(crate) const RICH_MTOON_VERTEX_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("b48d1e07-3c95-4a26-8f71-2d6b9c4a7e05");

const MTOON_UV_SHADER_HANDLE: Handle<Shader> = uuid_handle!("1f6f4388-e204-4b0a-a6c6-157e045a6ee1");

/// Cel shading blend, separate from upstream MToon's bindings 100..116.
#[derive(Asset, AsBindGroup, Clone, Debug, Reflect)]
pub struct RichMtoonExtension {
    #[uniform(200)]
    strength: f32,
}

impl MaterialExtension for RichMtoonExtension {
    fn vertex_shader() -> ShaderRef {
        RICH_MTOON_VERTEX_SHADER_HANDLE.into()
    }

    fn fragment_shader() -> ShaderRef {
        RICH_MTOON_FRAGMENT_SHADER_HANDLE.into()
    }
}

/// Upstream MToon data rendered by the app-side cel shader.
pub type RichMtoonMaterial = ExtendedMaterial<MToonMaterial, RichMtoonExtension>;
/// Registers the Rich material, its shaders, the switch and the sync.
pub(crate) fn register_rich_mtoon(app: &mut App) {
    app.add_plugins(MaterialPlugin::<RichMtoonMaterial>::default());
    load_internal_asset!(app, MTOON_UV_SHADER_HANDLE, "mtoon_uv.wesl", |source, _| {
        Shader::from_wesl(source, "embedded://vtuber_avatar/look/mtoon_uv.wesl")
    });
    load_internal_asset!(
        app,
        RICH_MTOON_FRAGMENT_SHADER_HANDLE,
        "mtoon_rich.wesl",
        |source, _| Shader::from_wesl(source, "embedded://vtuber_avatar/look/mtoon_rich.wesl")
    );
    load_internal_asset!(
        app,
        RICH_MTOON_VERTEX_SHADER_HANDLE,
        "mtoon_rich_vertex.wesl",
        |source, _| Shader::from_wesl(
            source,
            "embedded://vtuber_avatar/look/mtoon_rich_vertex.wesl"
        )
    );
    register_rich_material_systems::<RichMtoonExtension>(app);
    crate::look::rich_outline::register_rich_outline(app);
}

impl RichMaterialExtension for RichMtoonExtension {
    type Native = MToonMaterial;
    fn create(_native: &MToonMaterial, settings: RichLookSettings) -> Option<Self> {
        Some(Self {
            strength: settings.strength(),
        })
    }
    fn matches(
        current: &RichMtoonMaterial,
        native: &MToonMaterial,
        settings: RichLookSettings,
    ) -> bool {
        current.base == *native && current.extension.strength == settings.strength()
    }
    fn sync(current: &mut RichMtoonMaterial, native: &MToonMaterial, settings: RichLookSettings) {
        current.base.clone_from(native);
        current.extension.strength = settings.strength();
    }
}

#[cfg(test)]
mod tests {
    // Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use bevy::asset::AssetPlugin;
    use bevy::color::LinearRgba;
    use bevy::pbr::MeshMaterial3d;
    use bevy_vrm1::prelude::Shade;

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<MToonMaterial>()
            .init_asset::<RichMtoonMaterial>()
            .init_resource::<AvatarLifecycle>()
            .init_resource::<AvatarLookSettings>();
        register_rich_material_systems::<RichMtoonExtension>(&mut app);
        app
    }

    fn enter_ready(app: &mut App, root: Entity) {
        let mut lifecycle = app.world_mut().resource_mut::<AvatarLifecycle>();
        lifecycle.request_load(root).unwrap();
        lifecycle.start_binding(root);
        lifecycle.finish_ready();
    }

    fn native_material(app: &mut App) -> Handle<MToonMaterial> {
        let mut materials = app.world_mut().resource_mut::<Assets<MToonMaterial>>();
        materials.add(MToonMaterial {
            base_color: Color::LinearRgba(LinearRgba::new(0.2, 0.3, 0.4, 1.0)),
            shade: Shade {
                color: LinearRgba::new(0.5, 0.5, 0.5, 1.0),
                ..Shade::default()
            },
            ..MToonMaterial::default()
        })
    }

    fn spawn_avatar_mesh(app: &mut App, handle: &Handle<MToonMaterial>) -> (Entity, Entity) {
        let root = app.world_mut().spawn_empty().id();
        let mesh = app
            .world_mut()
            .spawn((
                crate::expression::material::VrmMaterialIndex(0),
                ChildOf(root),
                MeshMaterial3d(handle.clone()),
            ))
            .id();
        enter_ready(app, root);
        (root, mesh)
    }

    fn set_look(app: &mut App, enabled: bool, strength: f32) {
        app.world_mut().resource_mut::<AvatarLookSettings>().0 =
            crate::look::RichLookSettings::try_new(enabled, strength).unwrap();
    }

    fn rich_handle(app: &mut App, mesh: Entity) -> Option<Handle<RichMtoonMaterial>> {
        app.world()
            .get::<MeshMaterial3d<RichMtoonMaterial>>(mesh)
            .map(|material| material.0.clone())
    }

    fn native_handle(app: &mut App, mesh: Entity) -> Option<Handle<MToonMaterial>> {
        app.world()
            .get::<MeshMaterial3d<MToonMaterial>>(mesh)
            .map(|material| material.0.clone())
    }

    #[test]
    fn enabling_the_look_swaps_the_mesh_and_keeps_the_native_handle() {
        let mut app = test_app();
        let native = native_material(&mut app);
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
            .get::<RichMaterialSwap<RichMtoonExtension>>(mesh)
            .expect("swap component keeps the native handle");
        assert_eq!(swap.native.id(), native.id());

        let material = app
            .world()
            .resource::<Assets<RichMtoonMaterial>>()
            .get(rich.id())
            .expect("rich asset exists");
        assert_eq!(
            material.base.base_color,
            Color::LinearRgba(LinearRgba::new(0.2, 0.3, 0.4, 1.0))
        );
        assert_eq!(
            material.base.shade.color,
            LinearRgba::new(0.5, 0.5, 0.5, 1.0)
        );
    }

    #[test]
    fn disabling_the_look_restores_the_native_handle() {
        let mut app = test_app();
        let native = native_material(&mut app);
        let (_, mesh) = spawn_avatar_mesh(&mut app, &native);
        set_look(&mut app, true, 1.0);
        app.update();
        assert!(rich_handle(&mut app, mesh).is_some());

        set_look(&mut app, false, 1.0);
        app.update();
        assert!(rich_handle(&mut app, mesh).is_none());
        assert!(
            app.world()
                .get::<RichMaterialSwap<RichMtoonExtension>>(mesh)
                .is_none()
        );
        let restored = native_handle(&mut app, mesh).expect("native material restored");
        assert_eq!(restored.id(), native.id());
    }

    #[test]
    fn zero_strength_keeps_the_rich_material_selected() {
        let mut app = test_app();
        let native = native_material(&mut app);
        let (_, mesh) = spawn_avatar_mesh(&mut app, &native);
        set_look(&mut app, true, 0.0);
        app.update();

        // The material stays selected, but the uniform selects the exact
        // native fragment result and the preset despawns its spot lights.
        assert!(rich_handle(&mut app, mesh).is_some());
        let rich = rich_handle(&mut app, mesh).unwrap();
        assert_eq!(
            app.world()
                .resource::<Assets<RichMtoonMaterial>>()
                .get(rich.id())
                .unwrap()
                .extension
                .strength,
            0.0
        );
        assert!(
            app.world()
                .get::<RichMaterialSwap<RichMtoonExtension>>(mesh)
                .is_some()
        );
    }

    #[test]
    fn strength_changes_do_not_recreate_the_rich_asset() {
        let mut app = test_app();
        let native = native_material(&mut app);
        let (_, mesh) = spawn_avatar_mesh(&mut app, &native);
        set_look(&mut app, true, 1.0);
        app.update();
        let before = rich_handle(&mut app, mesh).expect("rich material");

        set_look(&mut app, true, 0.25);
        app.update();
        let after = rich_handle(&mut app, mesh).expect("rich material");
        assert_eq!(after.id(), before.id());
        assert_eq!(
            app.world()
                .resource::<Assets<RichMtoonMaterial>>()
                .get(after.id())
                .unwrap()
                .extension
                .strength,
            0.25
        );
        assert_eq!(
            app.world()
                .resource::<Assets<RichMtoonMaterial>>()
                .iter()
                .count(),
            1,
            "no new material asset per strength change"
        );
    }

    #[test]
    fn expression_writes_to_the_native_material_reach_the_rich_asset() {
        let mut app = test_app();
        let native = native_material(&mut app);
        let (_, mesh) = spawn_avatar_mesh(&mut app, &native);
        set_look(&mut app, true, 1.0);
        app.update();
        let rich = rich_handle(&mut app, mesh).expect("rich material");

        app.world_mut()
            .resource_mut::<Assets<MToonMaterial>>()
            .get_mut(native.id())
            .expect("native asset")
            .base_color = Color::LinearRgba(LinearRgba::new(0.9, 0.1, 0.1, 1.0));
        app.update();

        let material = app
            .world()
            .resource::<Assets<RichMtoonMaterial>>()
            .get(rich.id())
            .expect("rich asset exists");
        assert_eq!(
            material.base.base_color,
            Color::LinearRgba(LinearRgba::new(0.9, 0.1, 0.1, 1.0))
        );

        set_look(&mut app, false, 1.0);
        app.update();
        assert_eq!(
            native_handle(&mut app, mesh).map(|handle| handle.id()),
            Some(native.id())
        );
        assert_eq!(
            app.world()
                .resource::<Assets<MToonMaterial>>()
                .get(native.id())
                .expect("restored native asset")
                .base_color,
            Color::LinearRgba(LinearRgba::new(0.9, 0.1, 0.1, 1.0))
        );
    }

    #[test]
    fn meshes_outside_the_active_root_stay_native() {
        let mut app = test_app();
        let native = native_material(&mut app);
        let (_, _) = spawn_avatar_mesh(&mut app, &native);
        let other = app.world_mut().spawn(MeshMaterial3d(native.clone())).id();
        set_look(&mut app, true, 1.0);
        app.update();

        assert!(native_handle(&mut app, other).is_some());
        assert!(rich_handle(&mut app, other).is_none());
    }

    #[test]
    fn author_settings_and_textures_sync_and_survive_round_trip() {
        let mut app = test_app();
        let native = native_material(&mut app);
        let (_, mesh) = spawn_avatar_mesh(&mut app, &native);
        set_look(&mut app, true, 0.5);
        app.update();
        let rich = rich_handle(&mut app, mesh).unwrap();

        // Exercise material edits while Rich is active, including alpha and
        // shade settings beyond the expression colors synchronized before.
        for alpha in [AlphaMode::Opaque, AlphaMode::Mask(0.4), AlphaMode::Blend] {
            let expected = {
                let mut assets = app.world_mut().resource_mut::<Assets<MToonMaterial>>();
                let mut material = assets.get_mut(native.id()).unwrap();
                material.alpha_mode = alpha;
                material.double_sided = true;
                material.transparent_with_z_write = true;
                material.base_color_texture = Some(Handle::default());
                material.shade_multiply_texture = Some(Handle::default());
                material.shading_shift_texture = Some(Handle::default());
                material.shade.shading_shift_factor = 0.3;
                material.shade.toony_factor = 1.0;
                material.emissive = LinearRgba::new(2.0, 0.1, 0.2, 1.0);
                material.uv_transform = bevy::math::Affine2::from_translation(Vec2::splat(0.2));
                material.clone()
            };
            app.update();
            assert_eq!(
                app.world()
                    .resource::<Assets<RichMtoonMaterial>>()
                    .get(rich.id())
                    .unwrap()
                    .base,
                expected
            );
        }

        set_look(&mut app, false, 0.5);
        app.update();
        assert_eq!(native_handle(&mut app, mesh).unwrap().id(), native.id());
        let source = app
            .world()
            .resource::<Assets<MToonMaterial>>()
            .get(native.id())
            .unwrap();
        assert_eq!(source.alpha_mode, AlphaMode::Blend);
        assert_eq!(source.shade.toony_factor, 1.0);
        assert_eq!(source.emissive.red, 2.0);
    }
}
