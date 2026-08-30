mod material;
mod outline_pass;
mod setup;

use crate::error::vrm_error;
use crate::prelude::*;
use crate::vrm::gltf::extensions::{classify_legacy_shader, LegacyShaderKind};
use crate::vrm::gltf::materials::{
    convert_legacy_material_properties_with_render_queue_offset,
    plan_legacy_render_queue_offsets, VrmcMaterialsExtensitions,
};
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
const MTOON_VERTEX_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("f4041db8-c464-b84c-e3c9-e618527945a1");
const MTOON_TYPES_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("5d9302a3-6498-9d2a-fadb-842d01c87697");

pub struct MtoonMaterialPlugin;

impl Plugin for MtoonMaterialPlugin {
    fn build(
        &self,
        app: &mut App,
    ) {
        app.register_type::<MToonMaterial>()
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
            MTOON_TYPES_SHADER_HANDLE,
            "mtoon_types.wgsl",
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
        for material in source.materials() {
            let Some(index) = material.index() else {
                continue;
            };
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
                        if let Some(mut properties) = convert_legacy_material_properties_with_render_queue_offset(
                            legacy.as_ref().unwrap_or(&Value::Null),
                            Some(source.textures().count()),
                            render_queue_offset,
                        ) {
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
        Some(Self { materials, images })
    }
}
