//! Avatar-only egui paint callback.
//!
//! The shared avatar image stores gamma-premultiplied sRGB RGB bytes. The
//! normal egui image shader is intentionally left unchanged because it is
//! also used for text and unrelated UI images. This callback converts the
//! avatar texels to linear-premultiplied RGB immediately before the egui
//! target blends them.

use bevy::asset::{Handle, load_internal_asset, uuid_handle};
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendState,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, FilterMode, FragmentState,
    MipmapFilterMode, MultisampleState, PipelineCache, PrimitiveState, RenderPipelineDescriptor,
    Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType,
    SpecializedRenderPipeline, SpecializedRenderPipelines, TextureSampleType, UniformBuffer,
    VertexState, binding_types::sampler, binding_types::texture_2d, binding_types::uniform_buffer,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::renderer::RenderQueue;
use bevy::render::sync_world::RenderEntity;
use bevy::render::texture::GpuImage;
use bevy::render::{GpuResourceAppExt, RenderApp, RenderStartup};
use bevy::shader::Shader;
use bevy_egui::egui::{self, Painter, Rect, Response, Sense, Ui, Vec2};
use bevy_egui::render::{EguiBevyPaintCallback, EguiBevyPaintCallbackImpl, EguiPipelineKey};
use vtuber_core::VideoOutputProfile;

const AVATAR_PREVIEW_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("2e4c7dd4-1b15-4f1f-b5db-7ed4d0fd04c4");

/// The shared avatar image and its fixed display profile.
#[derive(Clone)]
pub struct AvatarPreviewTexture {
    image: Handle<Image>,
    profile: VideoOutputProfile,
}

impl AvatarPreviewTexture {
    /// Creates a preview texture descriptor for the shared avatar image.
    #[must_use]
    pub const fn new(image: Handle<Image>, profile: VideoOutputProfile) -> Self {
        Self { image, profile }
    }

    /// Returns the shared avatar image handle.
    #[must_use]
    pub fn image(&self) -> &Handle<Image> {
        &self.image
    }

    /// Returns the fixed avatar output profile.
    #[must_use]
    pub const fn profile(&self) -> VideoOutputProfile {
        self.profile
    }
}

/// Paints one avatar image rectangle through the avatar-only GPU callback.
pub fn paint_avatar_preview(
    ui: &mut Ui,
    image: Handle<Image>,
    size: Vec2,
    corner_radius: f32,
) -> Response {
    let (response, painter) = ui.allocate_painter(size, Sense::hover());
    paint_avatar_preview_at(&painter, response.rect, image, corner_radius);
    response
}

/// Paints the shared avatar image into an already laid-out rectangle.
pub fn paint_avatar_preview_at(
    painter: &Painter,
    rect: Rect,
    image: Handle<Image>,
    corner_radius: f32,
) {
    painter.add(EguiBevyPaintCallback::new_paint_callback(
        rect,
        AvatarPreviewPaintCallback {
            image,
            corner_radius,
        },
    ));
}

/// Registers the avatar preview callback's render pipeline.
pub struct AvatarPreviewPlugin;

impl Plugin for AvatarPreviewPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            AVATAR_PREVIEW_SHADER_HANDLE,
            "avatar_preview.wgsl",
            Shader::from_wgsl
        );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<AvatarPreviewPipeline>>()
            .add_systems(RenderStartup, init_avatar_preview_pipeline);
    }
}

#[derive(Resource)]
struct AvatarPreviewPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
}

#[derive(Clone, Copy, Debug, ShaderType)]
struct AvatarPreviewUniform {
    corner_radius: f32,
    width: f32,
    height: f32,
    unused: f32,
}

impl SpecializedRenderPipeline for AvatarPreviewPipeline {
    type Key = EguiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("avatar_preview".into()),
            layout: vec![self.bind_group_layout.clone()],
            vertex: VertexState {
                shader: AVATAR_PREVIEW_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: Some("vertex".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: AVATAR_PREVIEW_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        }
    }
}

fn init_avatar_preview_pipeline(mut commands: Commands, render_device: Res<RenderDevice>) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "avatar_preview_texture",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<AvatarPreviewUniform>(false),
            ),
        ),
    );
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("avatar_preview_nearest_texels"),
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: MipmapFilterMode::Nearest,
        ..default()
    });
    commands.insert_resource(AvatarPreviewPipeline {
        bind_group_layout,
        sampler,
    });
}

struct AvatarPreviewPaintCallback {
    image: Handle<Image>,
    corner_radius: f32,
}

#[derive(Component)]
struct AvatarPreviewPipelineId(CachedRenderPipelineId);

#[derive(Component)]
struct AvatarPreviewBindGroup(BindGroup);

#[derive(Component)]
struct AvatarPreviewUniformBuffer {
    _buffer: UniformBuffer<AvatarPreviewUniform>,
}

impl EguiBevyPaintCallbackImpl for AvatarPreviewPaintCallback {
    fn update(
        &self,
        info: egui::PaintCallbackInfo,
        render_entity: RenderEntity,
        pipeline_key: EguiPipelineKey,
        world: &mut World,
    ) {
        let pipeline_id = world.resource_scope(
            |world, mut pipelines: Mut<SpecializedRenderPipelines<AvatarPreviewPipeline>>| {
                let pipeline = world.resource::<AvatarPreviewPipeline>();
                let pipeline_cache = world.resource::<PipelineCache>();
                pipelines.specialize(pipeline_cache, pipeline, pipeline_key)
            },
        );
        let Some((bind_group, uniform)) = create_avatar_preview_bind_group(self, info, world)
        else {
            return;
        };
        world.entity_mut(render_entity.id()).insert((
            AvatarPreviewPipelineId(pipeline_id),
            AvatarPreviewBindGroup(bind_group),
            AvatarPreviewUniformBuffer { _buffer: uniform },
        ));
        world
            .resource_mut::<PipelineCache>()
            .block_on_render_pipeline(pipeline_id);
    }

    fn render<'pass>(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut TrackedRenderPass<'pass>,
        render_entity: RenderEntity,
        _pipeline_key: EguiPipelineKey,
        world: &'pass World,
    ) {
        let Some(entity) = world.get_entity(render_entity.id()).ok() else {
            return;
        };
        let Some(pipeline_id) = entity.get::<AvatarPreviewPipelineId>() else {
            return;
        };
        let Some(bind_group) = entity.get::<AvatarPreviewBindGroup>() else {
            return;
        };
        let Some(pipeline) = world
            .resource::<PipelineCache>()
            .get_render_pipeline(pipeline_id.0)
        else {
            return;
        };
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group.0, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

fn create_avatar_preview_bind_group(
    callback: &AvatarPreviewPaintCallback,
    info: egui::PaintCallbackInfo,
    world: &World,
) -> Option<(BindGroup, UniformBuffer<AvatarPreviewUniform>)> {
    let pipeline = world.resource::<AvatarPreviewPipeline>();
    let pipeline_cache = world.resource::<PipelineCache>();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let gpu_image = gpu_images.get(callback.image.id())?;
    let render_device = world.resource::<RenderDevice>();
    let render_queue = world.resource::<RenderQueue>();
    let size = info.viewport.size() * info.pixels_per_point;
    let mut uniform = UniformBuffer::from(AvatarPreviewUniform {
        corner_radius: callback.corner_radius * info.pixels_per_point,
        width: size.x,
        height: size.y,
        unused: 0.0,
    });
    uniform.write_buffer(render_device, render_queue);
    let binding = uniform.binding()?;
    Some((
        render_device.create_bind_group(
            Some("avatar_preview_bind_group"),
            &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
            &BindGroupEntries::sequential((&gpu_image.texture_view, &pipeline.sampler, binding)),
        ),
        uniform,
    ))
}
