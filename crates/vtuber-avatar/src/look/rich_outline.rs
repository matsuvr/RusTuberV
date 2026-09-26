//! The app-side outline connection for the Rich MToon material (#94).
//!
//! The upstream MToon outline pass is bound to the `MToonMaterial` asset type,
//! so a mesh swapped to `RichMtoonMaterial` would lose its outline. This
//! module is the thin app-side equivalent for the Rich material only: it
//! reuses the Rich material's shaders (with the upstream `OUTLINE_PASS`
//! branches), material layout and bind group data, and its own extraction of
//! the Rich mesh instances. The phase item, pipeline, draw command and pass
//! structure mirror the upstream outline pass. It is not a generic outline
//! framework and does not add a second renderer.
//!
//! The upstream pass stores material instances in its own private map, so the
//! Rich connection has its own extraction and queue. The app-side path does
//! not depend on upstream retaining stale native entries after a swap.

use std::ops::Range;

use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::ecs::entity::EntityHash;
use bevy::math::FloatOrd;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    DrawMesh, MaterialExtensionBindGroupData, MaterialPipeline, MaterialPipelineKey, MeshPipeline,
    MeshPipelineKey, PreparedMaterial, RenderMeshInstances, SetMaterialBindGroup, SetMeshBindGroup,
    SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup, ViewKeyCache, alpha_mode_pipeline_key,
    init_material_pipeline, queue_material_meshes,
};
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::render::erased_render_asset::ErasedRenderAssets;
use bevy::render::mesh::RenderMesh;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, PhaseItem,
    PhaseItemExtraIndex, SetItemPipeline, SortedPhaseItem, SortedRenderPhasePlugin,
    ViewSortedRenderPhases, sort_phase_system,
};
use bevy::render::render_resource::{
    AsBindGroup, BindGroupLayoutDescriptor, CachedRenderPipelineId, CompareFunction, Face,
    PipelineCache, RenderPipelineDescriptor, SpecializedMeshPipeline, SpecializedMeshPipelineError,
    SpecializedMeshPipelines,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::sync_world::{MainEntity, MainEntityHashMap};
use bevy::render::view::{
    ExtractedView, Msaa, RenderVisibleEntities, RetainedViewEntity, ViewDepthTexture, ViewTarget,
};
use bevy::render::{
    Extract, GpuResourceAppExt, Render, RenderApp, RenderDebugFlags, RenderStartup, RenderSystems,
};
use bevy::shader::ShaderDefVal;
use bevy_vrm1::prelude::MToonMaterialKey;
use indexmap::IndexMap;

use super::rich_mtoon::{
    RICH_MTOON_FRAGMENT_SHADER_HANDLE, RICH_MTOON_VERTEX_SHADER_HANDLE, RichMtoonMaterial,
};

const RICH_OUTLINE_MATERIAL_BIND_GROUP: usize = 3;

/// The app-side outline phase item for the Rich material.
struct RichOutlinePhaseItem {
    sort_key: FloatOrd,
    entity: (Entity, MainEntity),
    pipeline: CachedRenderPipelineId,
    draw_function: DrawFunctionId,
    batch_range: Range<u32>,
    extra_index: PhaseItemExtraIndex,
    indexed: bool,
}

impl PhaseItem for RichOutlinePhaseItem {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity.0
    }

    #[inline]
    fn main_entity(&self) -> MainEntity {
        self.entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for RichOutlinePhaseItem {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.sort_key
    }

    fn recalculate_sort_keys(
        _items: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        _view: &ExtractedView,
    ) {
    }

    #[inline]
    fn indexed(&self) -> bool {
        self.indexed
    }
}

impl CachedRenderPipelinePhaseItem for RichOutlinePhaseItem {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

#[derive(Resource)]
struct RichOutlinePipeline {
    base: MaterialPipeline,
    material_layout: BindGroupLayoutDescriptor,
}

/// The outline pipeline key: the mesh key plus the extended material's bind
/// group data, including upstream `MToonMaterialKey` in the base half.
#[derive(Clone, Hash, PartialEq, Eq)]
struct RichOutlineKey {
    mesh_key: MeshPipelineKey,
    bind_group_data: MaterialExtensionBindGroupData<MToonMaterialKey, ()>,
}

impl SpecializedMeshPipeline for RichOutlinePipeline {
    type Key = RichOutlineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        const PASS_NAME: &str = "OUTLINE_PASS";
        let mut descriptor = self.base.mesh_pipeline.specialize(key.mesh_key, layout)?;
        let material_key = MaterialPipelineKey {
            mesh_key: key.mesh_key,
            bind_group_data: key.bind_group_data,
        };
        RichMtoonMaterial::specialize(&self.base, &mut descriptor, layout, material_key)?;

        descriptor.vertex.shader = RICH_MTOON_VERTEX_SHADER_HANDLE;
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader = RICH_MTOON_FRAGMENT_SHADER_HANDLE;
        }

        if descriptor.layout.len() <= RICH_OUTLINE_MATERIAL_BIND_GROUP {
            descriptor.layout.resize(
                RICH_OUTLINE_MATERIAL_BIND_GROUP + 1,
                BindGroupLayoutDescriptor::default(),
            );
        }
        if let Some(slot) = descriptor.layout.get_mut(RICH_OUTLINE_MATERIAL_BIND_GROUP) {
            *slot = self.material_layout.clone();
        }

        descriptor
            .label
            .replace("rich_mtoon_outline_pipeline".into());

        let material_bind_group_def = ShaderDefVal::Int(
            "MATERIAL_BIND_GROUP".into(),
            RICH_OUTLINE_MATERIAL_BIND_GROUP as i32,
        );
        descriptor
            .vertex
            .shader_defs
            .push(material_bind_group_def.clone());
        descriptor.vertex.shader_defs.push(PASS_NAME.into());
        if let Some(depth_stencil) = descriptor.depth_stencil.as_mut() {
            depth_stencil.depth_compare = Some(CompareFunction::GreaterEqual);
        }
        descriptor.primitive.cull_mode = Some(Face::Front);
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.push(material_bind_group_def);
            fragment.shader_defs.push(PASS_NAME.into());
        }
        Ok(descriptor)
    }
}

type DrawRichOutline = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    SetMaterialBindGroup<3>,
    DrawMesh,
);

/// Registers the Rich outline phase, pipeline and pass.
pub(crate) fn register_rich_outline(app: &mut App) {
    app.add_plugins(
        SortedRenderPhasePlugin::<RichOutlinePhaseItem, MeshPipeline>::new(
            RenderDebugFlags::default(),
        ),
    );
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_gpu_resource::<SpecializedMeshPipelines<RichOutlinePipeline>>()
        .init_resource::<DrawFunctions<RichOutlinePhaseItem>>()
        .add_render_command::<RichOutlinePhaseItem, DrawRichOutline>()
        .init_resource::<ViewSortedRenderPhases<RichOutlinePhaseItem>>()
        .init_resource::<RichOutlineMaterialInstances>()
        .add_systems(
            RenderStartup,
            init_rich_outline_pipeline.after(init_material_pipeline),
        )
        .add_systems(
            ExtractSchedule,
            (
                extract_rich_outline_camera_phases,
                extract_rich_outline_materials,
            ),
        )
        .add_systems(
            Render,
            (
                queue_rich_outlines
                    .in_set(RenderSystems::QueueMeshes)
                    .after(queue_material_meshes),
                sort_phase_system::<RichOutlinePhaseItem>.in_set(RenderSystems::PhaseSort),
            ),
        )
        .add_systems(
            Core3d,
            rich_outline_draw_pass
                .in_set(Core3dSystems::MainPass)
                .after(main_transparent_pass_3d),
        );
}

fn init_rich_outline_pipeline(
    mut commands: Commands,
    material_pipeline: Res<MaterialPipeline>,
    render_device: Res<RenderDevice>,
) {
    let material_layout = RichMtoonMaterial::bind_group_layout_descriptor(&render_device);
    commands.insert_resource(RichOutlinePipeline {
        base: material_pipeline.clone(),
        material_layout,
    });
}

fn extract_rich_outline_camera_phases(
    mut outline_phases: ResMut<ViewSortedRenderPhases<RichOutlinePhaseItem>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
    cameras: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
) {
    live_entities.clear();
    for (main_entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }
        let retained_view_entity = RetainedViewEntity::new(main_entity.into(), None, 0);
        outline_phases.prepare_for_new_frame(retained_view_entity);
        live_entities.insert(retained_view_entity);
    }
    outline_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
}

#[derive(Resource, Default, Deref, DerefMut)]
struct RichOutlineMaterialInstances(MainEntityHashMap<AssetId<RichMtoonMaterial>>);

fn extract_rich_outline_materials(
    mut instances: ResMut<RichOutlineMaterialInstances>,
    materials: Extract<Query<(Entity, &MeshMaterial3d<RichMtoonMaterial>)>>,
) {
    instances.0.clear();
    for (entity, material) in &materials {
        instances.0.insert(entity.into(), material.id());
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Bevy injects this system's resources and query filters, so the parameter list is the declared ECS contract and has no call site to restructure"
)]
fn queue_rich_outlines(
    mut pipelines: ResMut<SpecializedMeshPipelines<RichOutlinePipeline>>,
    mut outline_phases: ResMut<ViewSortedRenderPhases<RichOutlinePhaseItem>>,
    mut views: Query<(&ExtractedView, &RenderVisibleEntities)>,
    view_key_cache: Res<ViewKeyCache>,
    render_visibility_ranges: Res<bevy::render::view::RenderVisibilityRanges>,
    instances: Res<RichOutlineMaterialInstances>,
    render_materials: Res<ErasedRenderAssets<PreparedMaterial>>,
    draw_functions: Res<DrawFunctions<RichOutlinePhaseItem>>,
    pipeline_cache: Res<PipelineCache>,
    outline_pipeline: Res<RichOutlinePipeline>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
) {
    for (view, visible_entities) in &mut views {
        let Some(view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };
        let Some(outline_phase) = outline_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let draw_function_id = draw_functions.read().id::<DrawRichOutline>();
        let Some(visible_mesh_entities) = visible_entities.get::<Mesh3d>() else {
            continue;
        };
        for (render_entity, visible_entity) in &visible_mesh_entities.entities_cpu_culling {
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id()) else {
                continue;
            };
            let Some(asset_id) = instances.get(visible_entity) else {
                continue;
            };
            let Some(material) = render_materials.get(*asset_id) else {
                continue;
            };

            let mut mesh_pipeline_key_bits: MeshPipelineKey =
                material.properties.mesh_pipeline_key_bits.downcast();
            mesh_pipeline_key_bits.insert(alpha_mode_pipeline_key(
                material.properties.alpha_mode,
                &Msaa::from_samples(view_key.msaa_samples()),
            ));
            let mut mesh_key = *view_key
                | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits())
                | mesh_pipeline_key_bits;

            if render_visibility_ranges.entity_has_crossfading_visibility_ranges(*visible_entity) {
                mesh_key |= MeshPipelineKey::VISIBILITY_RANGE_DITHER;
            }

            if view_key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
                if mesh_instance
                    .flags()
                    .contains(bevy::pbr::RenderMeshInstanceFlags::HAS_PREVIOUS_SKIN)
                {
                    mesh_key |= MeshPipelineKey::HAS_PREVIOUS_SKIN;
                }
                if mesh_instance
                    .flags()
                    .contains(bevy::pbr::RenderMeshInstanceFlags::HAS_PREVIOUS_MORPH)
                {
                    mesh_key |= MeshPipelineKey::HAS_PREVIOUS_MORPH;
                }
            }

            let rich_key = material
                .properties
                .material_key
                .to_key::<MaterialExtensionBindGroupData<MToonMaterialKey, ()>>();

            // Skip the outline for double-sided meshes (cull_mode: None): the
            // inverted-hull outline assumes back faces are invisible in the
            // main pass.
            let base_key = rich_key.base;
            if !base_key.intersects(MToonMaterialKey::CULL_FRONT | MToonMaterialKey::CULL_BACK) {
                continue;
            }

            let outline_key = RichOutlineKey {
                mesh_key,
                bind_group_data: rich_key,
            };

            let pipeline_id = match pipelines.specialize(
                &pipeline_cache,
                &outline_pipeline,
                outline_key,
                &mesh.layout,
            ) {
                Ok(id) => id,
                Err(err) => {
                    error!("Failed to specialize the rich MToon outline pipeline: {err}");
                    continue;
                }
            };
            let distance = material.properties.depth_bias;
            outline_phase.add_transient(RichOutlinePhaseItem {
                sort_key: FloatOrd(distance),
                entity: (*render_entity, *visible_entity),
                pipeline: pipeline_id,
                draw_function: draw_function_id,
                batch_range: 0..0,
                extra_index: PhaseItemExtraIndex::None,
                indexed: mesh.indexed(),
            });
        }
    }
}

fn rich_outline_draw_pass(
    world: &World,
    view: bevy::render::renderer::ViewQuery<(
        &bevy::render::camera::ExtractedCamera,
        &ExtractedView,
        &ViewTarget,
        &ViewDepthTexture,
    )>,
    outline_phases: Res<ViewSortedRenderPhases<RichOutlinePhaseItem>>,
    mut render_context: bevy::render::renderer::RenderContext,
) {
    let view_entity = view.entity();
    let (camera, extracted_view, target, depth_texture) = view.into_inner();

    let Some(outline_pass) = outline_phases.get(&extracted_view.retained_view_entity) else {
        return;
    };
    if outline_pass.items.is_empty() {
        return;
    }

    let mut render_pass = render_context.begin_tracked_render_pass(
        bevy::render::render_resource::RenderPassDescriptor {
            label: Some("rich mtoon outline pass"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: Some(
                depth_texture.get_attachment(bevy::render::render_resource::StoreOp::Store),
            ),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        },
    );
    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }
    if let Err(err) = outline_pass.render(&mut render_pass, world, view_entity) {
        error!("Error encountered while rendering the rich MToon outline phase: {err}");
    }
}
