mod material;
mod outline_pass;
mod setup;

use crate::error::vrm_error;
use crate::prelude::*;
use crate::vrm::gltf::extensions::{LegacyShaderKind, classify_legacy_shader};
use crate::vrm::gltf::materials::{
    VrmcMaterialsExtensitions, convert_legacy_material_properties_with_render_queue_offset,
    plan_legacy_render_queue_offsets,
};
use crate::vrm::mtoon::material::{MToonMaterialKey, MToonShadingMode};
use crate::vrm::mtoon::outline_pass::MToonOutlinePlugin;
use crate::vrm::mtoon::setup::MToonMaterialSetupPlugin;
use bevy::asset::{AssetId, load_internal_asset, uuid_handle};
use bevy::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

pub mod prelude {
    pub use crate::vrm::mtoon::{MtoonMaterialPlugin, VrmcMaterialRegistry, material::prelude::*};
}

const MTOON_FRAGMENT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("9a96eff2-1676-1dc0-9abc-2fd5e7134443");
const MTOON_RICH_FRAGMENT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("d3c8a1f4-7b52-4e09-a6d1-9f0c2b4e8a73");
const MTOON_NATIVE_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("b7e4c2a9-5d18-4f36-8c07-1e6a9d3b5f24");
const MTOON_VERTEX_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("f4041db8-c464-b84c-e3c9-e618527945a1");
const MTOON_TYPES_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("5d9302a3-6498-9d2a-fadb-842d01c87697");
const MTOON_LIGHTING_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("0a2f6d02-4c31-4c1f-9f0e-6b3b2ae9f1c8");
const MTOON_ALPHA_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("6f1c5b47-2f0a-4f4e-8c2d-9d4a7e5b3c11");
const MTOON_UV_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("e2a7c4d1-6b93-4f58-8a02-3d5e9c1f7b64");
const MTOON_PORTRAIT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("c4d5e6f7-1a2b-4c3d-9e8f-7a6b5c4d3e21");
const MTOON_PREPASS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("3b7e9a51-8d24-4a6b-9f13-5c2e8f7a0d42");

/// The fragment shader for one material's shading mode.
///
/// The Native and Rich displays are two small paths over the same fixed
/// reference: this selects which one a pipeline compiles against.
#[must_use]
pub(crate) fn mtoon_fragment_shader(key: &MToonMaterialKey) -> Handle<Shader> {
    if key.contains(MToonMaterialKey::RICH_SHADING) {
        MTOON_RICH_FRAGMENT_SHADER_HANDLE
    } else {
        MTOON_FRAGMENT_SHADER_HANDLE
    }
}

pub struct MtoonMaterialPlugin;

impl Plugin for MtoonMaterialPlugin {
    fn build(
        &self,
        app: &mut App,
    ) {
        app.register_type::<MToonMaterial>()
            .register_type::<MToonShadingMode>()
            .register_type::<MToonOutline>()
            .register_type::<VrmcMaterialRegistry>()
            .register_type::<RimLighting>()
            .register_type::<UVAnimation>()
            .register_type::<Shade>()
            .add_plugins(MaterialPlugin::<MToonMaterial>::default())
            .add_plugins((MToonMaterialSetupPlugin, MToonOutlinePlugin));
        load_internal_asset!(
            app,
            MTOON_FRAGMENT_SHADER_HANDLE,
            "mtoon_fragment.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MTOON_RICH_FRAGMENT_SHADER_HANDLE,
            "mtoon_rich_fragment.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MTOON_NATIVE_SHADER_HANDLE,
            "mtoon_native.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MTOON_TYPES_SHADER_HANDLE,
            "mtoon_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MTOON_LIGHTING_SHADER_HANDLE,
            "mtoon_lighting.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MTOON_ALPHA_SHADER_HANDLE,
            "mtoon_alpha.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MTOON_UV_SHADER_HANDLE,
            "mtoon_uv.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MTOON_PORTRAIT_SHADER_HANDLE,
            "mtoon_portrait.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MTOON_PREPASS_SHADER_HANDLE,
            "mtoon_prepass.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MTOON_VERTEX_SHADER_HANDLE,
            "mtoon_vertex.wgsl",
            Shader::from_wgsl
        );
    }
}

#[derive(Component, Default, Debug, Reflect)]
#[reflect(Component)]
pub struct VrmcMaterialRegistry {
    pub images: Vec<Handle<Image>>,
    pub materials: HashMap<AssetId<StandardMaterial>, VrmcMaterialsExtensitions>,
    /// glTF material index for each loaded `StandardMaterial` asset id.
    /// Expression material binds reference materials by this stable index.
    pub indices: HashMap<AssetId<StandardMaterial>, usize>,
    /// glTF material name for each glTF material index. Names are not unique
    /// across materials, so they are resolved through the index only.
    pub names: HashMap<usize, String>,
}

impl VrmcMaterialRegistry {
    pub fn new(
        gltf: &Gltf,
        images: Vec<Handle<Image>>,
        asset_server: &AssetServer,
    ) -> Self {
        Self::try_new(gltf, images, asset_server).unwrap_or_default()
    }

    fn try_new(
        gltf: &Gltf,
        images: Vec<Handle<Image>>,
        asset_server: &AssetServer,
    ) -> Option<Self> {
        // Match glTF materials to Bevy `StandardMaterial` handles by index,
        // not by name. The glTF spec does not require material names to be
        // unique, and some exporters (e.g. VRoid) produce multiple materials
        // that share a name. `Gltf::named_materials` is a `HashMap` keyed by
        // name, so duplicates collapse to a single entry and any meshes bound
        // to the overwritten materials skip the MToon conversion entirely,
        // rendering with the default `StandardMaterial` instead.
        let source = gltf.source.as_ref()?;
        let legacy_properties = source
            .extensions()
            .and_then(|extensions| extensions.get("VRM"))
            .and_then(|vrm| vrm.get("materialProperties"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let render_queue_offsets =
            plan_legacy_render_queue_offsets(&legacy_properties, source.materials().count());
        let mut materials = HashMap::new();
        let mut indices = HashMap::new();
        let mut names = HashMap::new();
        for material in source.materials() {
            let Some(index) = material.index() else {
                continue;
            };
            if let Some(name) = material.name() {
                names.insert(index, name.to_string());
            }
            let Some(gltf_material_path) = gltf.materials.get(index).and_then(|m| m.path()) else {
                continue;
            };
            let Some(label) = gltf_material_path.label() else {
                continue;
            };
            let std_path = gltf_material_path
                .clone()
                .with_label(format!("{label}/std"));
            let asset_id = asset_server.load::<StandardMaterial>(std_path).id();
            indices.insert(asset_id, index);
            let modern = material
                .extensions()
                .and_then(|extensions| extensions.get("VRMC_materials_mtoon"))
                .cloned();
            // VRM 0.x materialProperties is parallel to glTF materials. The
            // glTF material index is the only stable identity; names and
            // occurrence order are not.
            let legacy = legacy_properties.get(index).cloned();
            let render_queue_offset = render_queue_offsets.get(index).copied().flatten();
            if let Some(shader) = legacy
                .as_ref()
                .and_then(|value| value.get("shader"))
                .and_then(Value::as_str)
            {
                match classify_legacy_shader(shader) {
                    LegacyShaderKind::SupportedUnlit => {
                        if let Some(mut properties) =
                            convert_legacy_material_properties_with_render_queue_offset(
                                legacy.as_ref().unwrap_or(&Value::Null),
                                Some(source.textures().count()),
                                render_queue_offset,
                            )
                        {
                            properties.legacy_standard_fallback = true;
                            properties.legacy_z_write_requested =
                                shader == "VRM/UnlitTransparentZWrite";
                            materials.insert(asset_id, properties);
                        }
                        continue;
                    }
                    LegacyShaderKind::Passthrough => continue,
                    LegacyShaderKind::Unknown => {
                        #[cfg(feature = "log")]
                        bevy::log::warn!(
                            "VRM 0.x material {index} uses unsupported shader '{shader}'; keeping glTF StandardMaterial fallback"
                        );
                        continue;
                    }
                    LegacyShaderKind::MToon => {}
                }
            } else if legacy.is_some() {
                continue;
            }
            let Some(properties) = modern
                .and_then(|value| match serde_json::from_value(value) {
                    Ok(properties) => Some(properties),
                    Err(error) => {
                        vrm_error!("Failed to parse VRMC_materials_mtoon", error);
                        None
                    }
                })
                .or_else(|| {
                    legacy.and_then(|value| {
                        convert_legacy_material_properties_with_render_queue_offset(
                            &value,
                            Some(source.textures().count()),
                            render_queue_offset,
                        )
                    })
                })
            else {
                continue;
            };
            materials.insert(asset_id, properties);
        }
        Some(Self {
            materials,
            images,
            indices,
            names,
        })
    }
}
