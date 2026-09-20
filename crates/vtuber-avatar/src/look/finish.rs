//! The portrait finish: the look's HDR display transform shared by the
//! window viewport and the avatar output (Issue #74).
//!
//! The rich look turns both avatar cameras HDR (`bevy::camera::Hdr`) and
//! applies one preset exposure and display transform. The window keeps Bevy's
//! own tonemapping pass; the transparent output image needs the alpha-aware
//! fullscreen finish pass instead, because non-linear tone mapping must act
//! on the unassociated color `C = Cp / a` of the premultiplied main-pass
//! result and never on the coverage alone.
//!
//! Alpha/color-space contract across the boundaries:
//!
//! | Boundary | Color | Alpha |
//! |---|---|---|
//! | Main pass (look on, `Hdr`) | linear HDR internal texture | premultiplied |
//! | Main pass (look off) | gamma-composited internal texture (today's standard path) | premultiplied |
//! | MSAA resolve | linear, box filtered | premultiplied |
//! | Bevy tonemapping pass (viewport) | linear in, linear out | passthrough |
//! | Finish pass (output view, look on) | tone on unassociated color | passthrough (`a = 0` → `0`) |
//! | Upscaling blit | sRGB encode at the write | passthrough |
//! | Output image `Bgra8UnormSrgb` | sRGB-encoded | premultiplied |
//! | UI preview sampling | the same image, no copy | premultiplied |
//! | `VideoOutputFrame::from_padded_bgra8` | sRGB bytes | unpremultiplied once (existing) |
//! | NDI | BGRA8 sRGB | straight (existing) |

use bevy::asset::{Handle, load_internal_asset, uuid_handle};
use bevy::camera::{Exposure, Hdr};
use bevy::core_pipeline::{
    FullscreenShader,
    schedule::Core3d,
    tonemapping::{
        Tonemapping, TonemappingLuts, get_lut_bind_group_layout_entries, get_lut_bindings,
        tonemapping,
    },
    upscaling::upscaling,
};
use bevy::prelude::*;
use bevy::render::{
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    texture::{FallbackImage, GpuImage},
    view::{ExtractedView, ViewTarget},
};
use bevy::shader::{Shader, ShaderDefVal};

use crate::framing::AvatarViewportCamera;
use crate::look::preset::{RichLookSettings, blend_look_scalar, effective_look_strength};
use crate::render_output::AvatarOutputCamera;

const FINISH_SHADER_HANDLE: Handle<Shader> = uuid_handle!("7c2d6a51-4e10-4f8a-9b3d-25e91c4a7f62");

/// The one rich-look finish preset: HDR compositing with the app's default
/// camera exposure and Bevy's default neutral display transform.
///
/// The exposure equals the app's original camera default (`Exposure::BLENDER`),
/// so switching the look on keeps the portrait's brightness and adds the
/// highlight roll-off; a future preset can retune `exposure_ev100` and every
/// camera follows the resolved value.
pub const PORTRAIT_FINISH: PortraitFinish = PortraitFinish {
    hdr: true,
    exposure_ev100: Exposure::EV100_BLENDER,
    tonemapping: Tonemapping::TonyMcMapface,
};

/// The camera finish the look shares between the viewport and the output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortraitFinish {
    /// Whether the camera composites into an internal HDR texture.
    pub hdr: bool,
    /// The camera exposure the material light is scaled by.
    pub exposure_ev100: f32,
    /// The display transform.
    pub tonemapping: Tonemapping,
}

/// Resolves the finish for the current look settings.
///
/// The look off, or a strength of zero, returns `original` unchanged. At full
/// strength the preset's fixed exposure and tone apply; an intermediate
/// strength interpolates the exposure between the original and the preset and
/// switches the discrete parts (`hdr`, `tonemapping`) as soon as the look is
/// on.
#[must_use]
pub fn resolve_portrait_finish(
    original: &PortraitFinish,
    settings: RichLookSettings,
) -> PortraitFinish {
    let strength = effective_look_strength(settings);
    if strength == 0.0 {
        return *original;
    }
    PortraitFinish {
        hdr: PORTRAIT_FINISH.hdr,
        exposure_ev100: blend_look_scalar(
            original.exposure_ev100,
            PORTRAIT_FINISH.exposure_ev100,
            strength,
        ),
        tonemapping: PORTRAIT_FINISH.tonemapping,
    }
}

/// The viewport camera's finish state before the look took it over, restored
/// while the look is off.
///
/// `blend_look_scalar` interpolates exposure between the endpoints, so the
/// captured ev100 and the preset's value are both kept: the capture also
/// remembers whether the camera carried an `Exposure` component, so the
/// restore can put the camera back exactly as it was.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OriginalFinish {
    finish: PortraitFinish,
    /// Whether the original camera actually carried an `Exposure` component.
    exposure_component: bool,
}

impl OriginalFinish {
    /// The exposure to restore: the captured component, or nothing when the
    /// component was absent and the camera used Bevy's extract-time default.
    fn exposure(&self) -> Option<Exposure> {
        self.exposure_component.then_some(Exposure {
            ev100: self.finish.exposure_ev100,
        })
    }
}

/// The finish state the look restores when it turns off.
#[derive(Resource, Debug, Default)]
pub struct PortraitFinishState {
    original: Option<OriginalFinish>,
    last_applied: Option<PortraitFinish>,
}

impl PortraitFinishState {
    /// The captured original finish, once it exists.
    #[must_use]
    pub fn original(&self) -> Option<PortraitFinish> {
        self.original.map(|original| original.finish)
    }

    /// The last finish written to the cameras.
    #[must_use]
    pub fn last_applied(&self) -> Option<PortraitFinish> {
        self.last_applied
    }
}

/// Writes the resolved finish onto the avatar's cameras and marks the output
/// view for the finish pass.
///
/// The capture happens once, from the components the cameras actually carry,
/// so the plain display can be restored exactly. Writes happen only when the
/// resolved finish differs from the last applied one.
#[allow(clippy::type_complexity)]
pub fn sync_portrait_finish(
    settings: Res<crate::look::AvatarLookSettings>,
    mut state: ResMut<PortraitFinishState>,
    mut commands: Commands,
    mut cameras: Query<
        (Entity, Has<Hdr>, Option<&Exposure>, &mut Tonemapping),
        Or<(With<AvatarViewportCamera>, With<AvatarOutputCamera>)>,
    >,
    output_markers: Query<(), With<AvatarOutputCamera>>,
) {
    if state.original.is_none() {
        for (entity, hdr, exposure, tonemapping) in cameras.iter_mut() {
            if output_markers.contains(entity) {
                continue;
            }
            state.original = Some(OriginalFinish {
                finish: PortraitFinish {
                    hdr,
                    exposure_ev100: exposure
                        .map_or(Exposure::default().ev100, |exposure| exposure.ev100),
                    tonemapping: *tonemapping,
                },
                exposure_component: exposure.is_some(),
            });
            break;
        }
    }
    let Some(original) = state.original else {
        return;
    };
    let resolved = resolve_portrait_finish(&original.finish, settings.0);
    if state.last_applied == Some(resolved) {
        return;
    }
    // The finish applies only what changes the image: with the preset equal
    // to the original there is no pass, no HDR switch and no component write.
    let active = resolved != original.finish;

    for (entity, hdr, exposure, mut tonemapping) in &mut cameras {
        if hdr != resolved.hdr {
            if resolved.hdr {
                commands.entity(entity).insert(Hdr);
            } else {
                commands.entity(entity).remove::<Hdr>();
            }
        }
        let is_output = output_markers.contains(entity);
        let target = if is_output && active {
            Tonemapping::None
        } else if is_output {
            original.finish.tonemapping
        } else {
            resolved.tonemapping
        };
        if *tonemapping != target {
            *tonemapping = target;
        }
        if is_output {
            if active {
                commands.entity(entity).insert(PortraitFinishPass {
                    tonemapping: resolved.tonemapping,
                });
            } else {
                commands.entity(entity).remove::<PortraitFinishPass>();
            }
        }
        sync_camera_exposure(
            &mut commands,
            entity,
            if active {
                Some(resolved.exposure_ev100)
            } else {
                original.exposure().map(|exposure| exposure.ev100)
            },
            exposure,
        );
    }

    state.last_applied = Some(resolved);
}

/// Inserts, updates or removes the camera's `Exposure` component so its
/// effective ev100 matches the target. A camera without the component uses
/// Bevy's default exposure, so a matching target writes nothing.
fn sync_camera_exposure(
    commands: &mut Commands,
    entity: Entity,
    target_ev100: Option<f32>,
    current: Option<&Exposure>,
) {
    let current_ev100 = current.map_or(Exposure::default().ev100, |exposure| exposure.ev100);
    let target = target_ev100.unwrap_or(Exposure::default().ev100);
    if current_ev100 == target {
        return;
    }
    match target_ev100 {
        Some(ev100) => {
            commands.entity(entity).insert(Exposure { ev100 });
        }
        None => {
            commands.entity(entity).remove::<Exposure>();
        }
    }
}

/// Marks the avatar output view for the alpha-aware finish pass and carries
/// the resolved display transform into the render world.
#[derive(Component, Clone, Copy, Debug, PartialEq, ExtractComponent)]
pub struct PortraitFinishPass {
    /// The display transform the finish pass applies.
    pub tonemapping: Tonemapping,
}

/// The finish pass's per-view uniform.
#[derive(Clone, Copy, Debug, ShaderType)]
struct FinishUniform {
    /// The resolved display transform, as the [`Tonemapping`] discriminant.
    tonemapping: u32,
    unused_1: f32,
    unused_2: f32,
    unused_3: f32,
}

#[derive(Component, Deref, DerefMut)]
struct ViewPortraitFinishPipeline(#[deref] CachedRenderPipelineId, Tonemapping);

#[derive(Component)]
struct FinishUniformBufferOffset(u32);

#[derive(Resource, Default)]
struct FinishUniforms(DynamicUniformBuffer<FinishUniform>);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PortraitFinishPipelineKey {
    tonemapping: Tonemapping,
}

#[derive(Resource)]
struct PortraitFinishPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    fullscreen_shader: FullscreenShader,
}

impl SpecializedRenderPipeline for PortraitFinishPipeline {
    type Key = PortraitFinishPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();
        shader_defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_TEXTURE_BINDING_INDEX".into(),
            2,
        ));
        shader_defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_SAMPLER_BINDING_INDEX".into(),
            3,
        ));
        match key.tonemapping {
            Tonemapping::None => shader_defs.push("TONEMAP_METHOD_NONE".into()),
            Tonemapping::Reinhard => shader_defs.push("TONEMAP_METHOD_REINHARD".into()),
            Tonemapping::ReinhardLuminance => {
                shader_defs.push("TONEMAP_METHOD_REINHARD_LUMINANCE".into());
            }
            Tonemapping::AcesFitted => shader_defs.push("TONEMAP_METHOD_ACES_FITTED".into()),
            Tonemapping::AgX => shader_defs.push("TONEMAP_METHOD_AGX".into()),
            Tonemapping::SomewhatBoringDisplayTransform => {
                shader_defs.push("TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into());
            }
            Tonemapping::TonyMcMapface => {
                shader_defs.push("TONEMAP_METHOD_TONY_MC_MAPFACE".into());
            }
            Tonemapping::BlenderFilmic => shader_defs.push("TONEMAP_METHOD_BLENDER_FILMIC".into()),
            Tonemapping::KhronosPbrNeutral => shader_defs.push("TONEMAP_METHOD_PBR_NEUTRAL".into()),
        }
        RenderPipelineDescriptor {
            label: Some("portrait_finish".into()),
            layout: vec![self.bind_group_layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: FINISH_SHADER_HANDLE,
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

fn init_portrait_finish_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let lut_entries = get_lut_bind_group_layout_entries();
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "portrait_finish_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                lut_entries[0],
                lut_entries[1],
                uniform_buffer::<FinishUniform>(true),
            ),
        ),
    );
    let sampler = render_device.create_sampler(&SamplerDescriptor::default());
    commands.insert_resource(PortraitFinishPipeline {
        bind_group_layout,
        sampler,
        fullscreen_shader: fullscreen_shader.clone(),
    });
}

#[allow(clippy::type_complexity)]
fn prepare_portrait_finish_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<PortraitFinishPipeline>>,
    pipeline: Res<PortraitFinishPipeline>,
    views: Query<
        (
            Entity,
            Option<&PortraitFinishPass>,
            Option<&ViewPortraitFinishPipeline>,
        ),
        With<ExtractedView>,
    >,
) {
    for (entity, marker, existing) in &views {
        let Some(marker) = marker else {
            // The look turned off: drop the view's pipeline so the finish
            // pass cannot run on the plain display's LDR main texture.
            if existing.is_some() {
                commands
                    .entity(entity)
                    .remove::<ViewPortraitFinishPipeline>()
                    .remove::<FinishUniformBufferOffset>();
            }
            continue;
        };
        if existing.is_some_and(|view| view.1 == marker.tonemapping) {
            continue;
        }
        let key = PortraitFinishPipelineKey {
            tonemapping: marker.tonemapping,
        };
        let pipeline = pipelines.specialize(&pipeline_cache, &pipeline, key);
        commands
            .entity(entity)
            .insert(ViewPortraitFinishPipeline(pipeline, key.tonemapping));
    }
}

fn prepare_portrait_finish_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut uniforms: ResMut<FinishUniforms>,
    views: Query<(Entity, &PortraitFinishPass)>,
) {
    uniforms.0.clear();
    for (entity, marker) in &views {
        let offset = uniforms.0.push(&FinishUniform {
            tonemapping: marker.tonemapping as u32,
            unused_1: 0.0,
            unused_2: 0.0,
            unused_3: 0.0,
        });
        commands
            .entity(entity)
            .insert(FinishUniformBufferOffset(offset));
    }
    uniforms.0.write_buffer(&render_device, &render_queue);
}

// The draw-side view params mirror Bevy's own tonemapping system shape.
#[allow(clippy::too_many_arguments)]
fn portrait_finish_pass(
    view: ViewQuery<(
        &ViewTarget,
        &ViewPortraitFinishPipeline,
        &FinishUniformBufferOffset,
    )>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<PortraitFinishPipeline>,
    finish_uniforms: Res<FinishUniforms>,
    lut_images: Res<RenderAssets<GpuImage>>,
    tonemapping_luts: Res<TonemappingLuts>,
    fallback_image: Res<FallbackImage>,
    mut ctx: RenderContext,
) {
    let (target, view_pipeline, finish_offset) = view.into_inner();
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(**view_pipeline) else {
        return;
    };
    let Some(finish_binding) = finish_uniforms.0.binding() else {
        return;
    };
    let lut = get_lut_bindings(
        &lut_images,
        &tonemapping_luts,
        &view_pipeline.1,
        &fallback_image,
    );

    let post_process = target.post_process_write();
    let bind_group = ctx.render_device().create_bind_group(
        Some("portrait_finish_bind_group"),
        &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
        &BindGroupEntries::sequential((
            post_process.source,
            &pipeline.sampler,
            lut.0,
            lut.1,
            finish_binding,
        )),
    );

    let mut render_pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("portrait_finish"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    render_pass.set_pipeline(render_pipeline);
    render_pass.set_bind_group(0, &bind_group, &[finish_offset.0]);
    render_pass.draw(0..3, 0..1);
}

/// Registers the finish state, the camera sync system and the GPU pass.
pub fn register_portrait_finish(app: &mut App) {
    load_internal_asset!(app, FINISH_SHADER_HANDLE, "finish.wgsl", Shader::from_wgsl);
    app.add_plugins(ExtractComponentPlugin::<PortraitFinishPass>::default());
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_gpu_resource::<SpecializedRenderPipelines<PortraitFinishPipeline>>()
        .init_gpu_resource::<FinishUniforms>()
        .add_systems(RenderStartup, init_portrait_finish_pipeline)
        .add_systems(
            Render,
            (
                prepare_portrait_finish_pipelines,
                prepare_portrait_finish_uniforms,
            )
                .in_set(RenderSystems::Prepare),
        )
        .add_systems(
            Core3d,
            portrait_finish_pass.after(tonemapping).before(upscaling),
        );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::AvatarViewportCamera;

    type Pixel = [f32; 4];

    fn original_finish() -> PortraitFinish {
        PortraitFinish {
            hdr: false,
            exposure_ev100: Exposure::default().ev100,
            tonemapping: Tonemapping::TonyMcMapface,
        }
    }

    #[test]
    fn off_and_zero_strength_return_the_original_finish() {
        let original = original_finish();
        for settings in [
            RichLookSettings {
                enabled: false,
                strength: 1.0,
            },
            RichLookSettings {
                enabled: true,
                strength: 0.0,
            },
        ] {
            assert_eq!(resolve_portrait_finish(&original, settings), original);
        }
    }

    #[test]
    fn full_strength_applies_the_preset() {
        let resolved = resolve_portrait_finish(
            &original_finish(),
            RichLookSettings {
                enabled: true,
                strength: 1.0,
            },
        );
        assert_eq!(resolved, PORTRAIT_FINISH);
        assert!(resolved.hdr);
        assert_eq!(resolved.tonemapping, Tonemapping::TonyMcMapface);
    }

    #[test]
    fn an_intermediate_strength_blends_the_exposure_only() {
        let resolved = resolve_portrait_finish(
            &original_finish(),
            RichLookSettings {
                enabled: true,
                strength: 0.5,
            },
        );
        assert!(resolved.hdr);
        assert_eq!(resolved.tonemapping, PORTRAIT_FINISH.tonemapping);
        assert_eq!(
            resolved.exposure_ev100,
            (original_finish().exposure_ev100 + PORTRAIT_FINISH.exposure_ev100) / 2.0
        );
    }

    #[test]
    fn the_preset_exposure_matches_the_default_camera() {
        assert_eq!(PORTRAIT_FINISH.exposure_ev100, Exposure::default().ev100);
    }

    fn look_sync_app() -> (App, Entity, Entity) {
        let mut app = App::new();
        app.init_resource::<crate::look::AvatarLookSettings>()
            .init_resource::<PortraitFinishState>()
            .add_systems(Update, sync_portrait_finish);
        let viewport = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                Tonemapping::TonyMcMapface,
                AvatarViewportCamera::from_default_transform(Transform::default()),
            ))
            .id();
        let output = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                Tonemapping::TonyMcMapface,
                AvatarOutputCamera,
            ))
            .id();
        app.update();
        (app, viewport, output)
    }

    fn set_settings(app: &mut App, settings: RichLookSettings) {
        app.world_mut()
            .resource_mut::<crate::look::AvatarLookSettings>()
            .0 = settings;
        app.update();
    }

    #[test]
    fn the_look_off_leaves_the_cameras_untouched() {
        let (app, viewport, output) = look_sync_app();
        assert!(
            app.world().get::<Hdr>(viewport).is_none() && app.world().get::<Hdr>(output).is_none()
        );
        assert!(app.world().get::<Exposure>(viewport).is_none());
        assert!(app.world().get::<Exposure>(output).is_none());
        assert!(app.world().get::<PortraitFinishPass>(output).is_none());
    }

    #[test]
    fn the_look_on_turns_the_cameras_hdr_and_marks_the_output_view() {
        let (mut app, viewport, output) = look_sync_app();
        let on = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        set_settings(&mut app, on);
        let resolved = resolve_portrait_finish(&original_finish(), on);
        assert!(app.world().get::<Hdr>(viewport).is_some());
        assert!(app.world().get::<Hdr>(output).is_some());
        assert_eq!(
            app.world().get::<PortraitFinishPass>(output),
            Some(&PortraitFinishPass {
                tonemapping: resolved.tonemapping,
            })
        );
        // The output camera routes the curve through the finish pass instead.
        assert_eq!(
            app.world().get::<Tonemapping>(output),
            Some(&Tonemapping::None)
        );
        assert_eq!(
            app.world().get::<Tonemapping>(viewport),
            Some(&resolved.tonemapping)
        );
    }

    #[test]
    fn the_look_off_again_restores_the_original_finish() {
        let (mut app, viewport, output) = look_sync_app();
        set_settings(
            &mut app,
            RichLookSettings {
                enabled: true,
                strength: 1.0,
            },
        );
        set_settings(
            &mut app,
            RichLookSettings {
                enabled: false,
                strength: 0.4,
            },
        );
        assert!(app.world().get::<Hdr>(viewport).is_none());
        assert!(app.world().get::<Hdr>(output).is_none());
        assert!(app.world().get::<PortraitFinishPass>(output).is_none());
        assert_eq!(
            app.world().get::<Tonemapping>(output),
            Some(&Tonemapping::TonyMcMapface)
        );
        assert_eq!(
            app.world().get::<Tonemapping>(viewport),
            Some(&Tonemapping::TonyMcMapface)
        );
        assert!(app.world().get::<Exposure>(viewport).is_none());
        assert!(app.world().get::<Exposure>(output).is_none());
        // The capture never promoted the applied values to the original.
        assert_eq!(
            app.world().resource::<PortraitFinishState>().original(),
            Some(original_finish())
        );
    }

    #[test]
    fn fixed_frames_do_not_write_again() {
        let (mut app, viewport, _) = look_sync_app();
        let on = RichLookSettings {
            enabled: true,
            strength: 1.0,
        };
        set_settings(&mut app, on);
        let first = app
            .world()
            .get::<Exposure>(viewport)
            .map(|exposure| exposure.ev100);
        let state = app.world().resource::<PortraitFinishState>().last_applied();
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(
            app.world().resource::<PortraitFinishState>().last_applied(),
            state
        );
        assert_eq!(
            app.world()
                .get::<Exposure>(viewport)
                .map(|exposure| exposure.ev100),
            first
        );
    }

    #[test]
    fn a_captured_exposure_component_is_restored() {
        let mut app = App::new();
        app.init_resource::<crate::look::AvatarLookSettings>()
            .init_resource::<PortraitFinishState>()
            .add_systems(Update, sync_portrait_finish);
        let viewport = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                Tonemapping::TonyMcMapface,
                Exposure {
                    ev100: Exposure::EV100_OVERCAST,
                },
                AvatarViewportCamera::from_default_transform(Transform::default()),
            ))
            .id();
        app.world_mut().spawn((
            Camera3d::default(),
            Tonemapping::TonyMcMapface,
            AvatarOutputCamera,
        ));
        app.update();

        set_settings(
            &mut app,
            RichLookSettings {
                enabled: true,
                strength: 1.0,
            },
        );
        assert_eq!(
            app.world()
                .get::<Exposure>(viewport)
                .map(|exposure| exposure.ev100),
            Some(Exposure::EV100_BLENDER)
        );
        set_settings(
            &mut app,
            RichLookSettings {
                enabled: false,
                strength: 1.0,
            },
        );
        assert_eq!(
            app.world()
                .get::<Exposure>(viewport)
                .map(|exposure| exposure.ev100),
            Some(Exposure::EV100_OVERCAST)
        );
    }

    /// CPU mirror of the pass's `finish_premultiplied_linear`.
    fn finish_premultiplied_fixture(rgba: Pixel, curve: fn([f32; 3]) -> [f32; 3]) -> Pixel {
        let coverage = rgba[3];
        if coverage <= 0.0 {
            return [0.0; 4];
        }
        let color = curve([rgba[0] / coverage, rgba[1] / coverage, rgba[2] / coverage]);
        [
            color[0] * coverage,
            color[1] * coverage,
            color[2] * coverage,
            coverage,
        ]
    }

    /// Composites a premultiplied source over a premultiplied background.
    fn premultiplied_over(source: Pixel, background: Pixel) -> Pixel {
        let a = source[3];
        [
            source[0] + background[0] * (1.0 - a),
            source[1] + background[1] * (1.0 - a),
            source[2] + background[2] * (1.0 - a),
            source[3] + background[3] * (1.0 - a),
        ]
    }

    /// Composites a straight-alpha source over a straight background.
    fn straight_over(source: Pixel, background: Pixel) -> Pixel {
        let (a, b) = (source[3], background[3]);
        let out_a = a + b * (1.0 - a);
        [
            (source[0] * a + background[0] * b * (1.0 - a)) / out_a,
            (source[1] * a + background[1] * b * (1.0 - a)) / out_a,
            (source[2] * a + background[2] * b * (1.0 - a)) / out_a,
            out_a,
        ]
    }

    fn reinhard(rgb: [f32; 3]) -> [f32; 3] {
        [
            rgb[0] / (1.0 + rgb[0]),
            rgb[1] / (1.0 + rgb[1]),
            rgb[2] / (1.0 + rgb[2]),
        ]
    }

    fn identical(rgb: [f32; 3]) -> [f32; 3] {
        rgb
    }

    fn close_enough(left: f32, right: f32) -> bool {
        (left - right).abs() < 1e-5
    }

    fn assert_pixel_close_enough(left: Pixel, right: Pixel) {
        for (index, (l, r)) in left.into_iter().zip(right).enumerate() {
            assert!(
                close_enough(l, r),
                "channel {index}: {l} != {r} (left {left:?}, right {right:?})"
            );
        }
    }

    #[test]
    fn alpha_zero_stays_transparent_over_any_curve() {
        for curve in [identical, reinhard] {
            assert_eq!(
                finish_premultiplied_fixture([5.0, 2.0, 1.0, 0.0], curve),
                [0.0; 4]
            );
        }
    }

    #[test]
    fn every_coverage_keeps_its_alpha_exactly() {
        for coverage in [0.25, 0.5, 1.0] {
            let out = finish_premultiplied_fixture([4.0, 3.0, 2.0, coverage], reinhard);
            assert!(close_enough(out[3], coverage));
        }
    }

    #[test]
    fn partial_coverage_composites_like_the_unassociated_color() {
        // The main pass stores the premultiplied color `Cp = C * a`; the
        // fixture unprepares to `C`, applies the curve and re-associates.
        // The finished premultiplied pixel must composite over any opaque
        // background exactly like the straight finished color does.
        let color = [4.0, 3.0, 2.0];
        for coverage in [0.25, 0.5, 1.0] {
            let source = [
                color[0] * coverage,
                color[1] * coverage,
                color[2] * coverage,
                coverage,
            ];
            let finished = finish_premultiplied_fixture(source, reinhard);
            let toned = reinhard(color);
            for background in [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.6, 0.2, 0.9, 1.0],
            ] {
                let composited = premultiplied_over(finished, background);
                let expected = straight_over([toned[0], toned[1], toned[2], coverage], background);
                assert_pixel_close_enough(composited, expected);
            }
        }
    }

    #[test]
    fn full_coverage_and_identity_curve_pass_through() {
        let out = finish_premultiplied_fixture([0.4, 0.5, 0.6, 1.0], identical);
        assert_pixel_close_enough(out, [0.4, 0.5, 0.6, 1.0]);
    }

    #[test]
    fn the_finish_rolls_highlights_off_without_white_edges() {
        // A bright highlight keeps headroom instead of clipping to white.
        let out = finish_premultiplied_fixture([20.0, 20.0, 20.0, 1.0], reinhard);
        assert!(out[..3].iter().all(|channel| *channel < 1.0));
        assert!(out[..3].iter().all(|channel| *channel > 0.7));

        // A partial-coverage edge keeps the premultiplied bound `rgb <= a`,
        // so no background channel can pick up a white fringe.
        for coverage in [0.25, 0.5] {
            let out = finish_premultiplied_fixture([20.0, 20.0, 20.0, coverage], reinhard);
            assert!(
                out[..3].iter().all(|channel| *channel <= out[3] + 1e-5),
                "coverage {coverage}: {out:?}"
            );
        }
    }

    #[test]
    fn the_finish_preserves_channel_order() {
        let blue_dominant = finish_premultiplied_fixture([1.0, 2.0, 8.0, 1.0], reinhard);
        assert!(blue_dominant[2] > blue_dominant[1] && blue_dominant[1] > blue_dominant[0]);
    }
}
