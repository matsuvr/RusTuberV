//! The app-side Rich MToon material and Native/Rich switch (#94).
//!
//! OFF uses the transparency-fixed upstream `MToonMaterial`. ON wraps the
//! current native material in Bevy's `ExtendedMaterial`, which keeps the
//! upstream bindings, alpha/depth behavior, and specialization while replacing
//! only the fragment shader with the additive Rich shader. The native asset
//! remains the owner of current expression and UV values.

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

/// Shader override with no additional material bindings.
#[derive(Asset, AsBindGroup, Clone, Debug, Reflect)]
pub struct RichMtoonExtension {}

impl MaterialExtension for RichMtoonExtension {
    fn vertex_shader() -> ShaderRef {
        RICH_MTOON_VERTEX_SHADER_HANDLE.into()
    }

    fn fragment_shader() -> ShaderRef {
        RICH_MTOON_FRAGMENT_SHADER_HANDLE.into()
    }
}

/// Upstream MToon data rendered by the app-side additive shader.
pub type RichMtoonMaterial = ExtendedMaterial<MToonMaterial, RichMtoonExtension>;
/// Registers the Rich material, its shaders, the switch and the sync.
pub(crate) fn register_rich_mtoon(app: &mut App) {
    app.add_plugins(MaterialPlugin::<RichMtoonMaterial>::default());
    load_internal_asset!(
        app,
        MTOON_UV_SHADER_HANDLE,
        "mtoon_uv.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        RICH_MTOON_FRAGMENT_SHADER_HANDLE,
        "mtoon_rich.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        RICH_MTOON_VERTEX_SHADER_HANDLE,
        "mtoon_rich_vertex.wgsl",
        Shader::from_wgsl
    );
    register_rich_material_systems::<RichMtoonExtension>(app);
    crate::look::rich_outline::register_rich_outline(app);
}

impl RichMaterialExtension for RichMtoonExtension {
    type Native = MToonMaterial;
    fn create(_native: &MToonMaterial, _settings: RichLookSettings) -> Option<Self> {
        Some(Self {})
    }
    fn matches(
        current: &RichMtoonMaterial,
        native: &MToonMaterial,
        _settings: RichLookSettings,
    ) -> bool {
        current.base.base_color == native.base_color
            && current.base.emissive == native.emissive
            && current.base.shade.color == native.shade.color
            && current.base.rim_lighting.color == native.rim_lighting.color
            && current.base.outline.color == native.outline.color
            && current.base.uv_transform == native.uv_transform
    }
    fn sync(current: &mut RichMtoonMaterial, native: &MToonMaterial, _settings: RichLookSettings) {
        current.base.base_color = native.base_color;
        current.base.emissive = native.emissive;
        current.base.shade.color = native.shade.color;
        current.base.rim_lighting.color = native.rim_lighting.color;
        current.base.outline.color = native.outline.color;
        current.base.uv_transform = native.uv_transform;
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

        // The Rich shader stays selected at 0%; the added terms are zero
        // because #93 despawns the spot lights, so the shader's only added
        // input is gone.
        assert!(rich_handle(&mut app, mesh).is_some());
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
}
