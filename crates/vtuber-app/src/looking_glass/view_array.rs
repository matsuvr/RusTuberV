//! Capture completed views with one output blit and no quilt render pass.
//!
//! All view cameras share one small logical render target. Bevy consequently
//! shares their main/depth allocations, but gives each camera its own clear
//! operation. The camera driver runs each view's complete schedule in order:
//! clear, draw (including both MToon outline paths), then copy one array layer.
//! Do not defer the copies until after the camera loop: the scratch image has
//! already been overwritten by the next view at that point.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::{Core3d, Core3dSystems, upscaling::upscaling};
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    Extent3d, Origin3d, TexelCopyTextureInfo, TextureAspect, TextureDimension, TextureFormat,
    TextureUsages, TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::{RenderContext, ViewQuery};
use bevy::render::texture::GpuImage;
use bevy::render::view::ViewTarget;

use super::optics::Layout;

// Use the same explicit SDR format for the scratch target and array. An image
// target (unlike a window) is not subject to Bevy's BGRA swapchain conversion.
// No Hdr or CompositingSpace component is installed on these cameras.
pub(super) const VIEW_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

#[derive(Component, Clone, ExtractComponent)]
#[extract_app(RenderApp)]
pub(super) struct ViewCamera {
    pub index: u32,
    pub array: Handle<Image>,
}

pub(super) fn install(app: &mut App) {
    app.add_plugins(ExtractComponentPlugin::<ViewCamera>::default());
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            Core3d,
            copy_view_to_array
                .after(Core3dSystems::PostProcess)
                .after(upscaling),
        );
    }
}

pub(super) fn image(layout: Layout) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: layout.view_width,
            height: layout.view_height,
            depth_or_array_layers: layout.count(),
        },
        TextureDimension::D2,
        VIEW_FORMAT,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.label = Some("looking_glass_view_array");
    image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image
}

fn copy_view_to_array(
    view: ViewQuery<(&ViewCamera, &ViewTarget)>,
    images: Res<RenderAssets<GpuImage>>,
    mut context: RenderContext,
) {
    let (camera, target) = view.into_inner();
    let Some(array) = images.get(&camera.array) else {
        // Assets can still be preparing on the first frame.
        return;
    };
    let source = target.main_texture();
    context.command_encoder().copy_texture_to_texture(
        source.as_image_copy(),
        TexelCopyTextureInfo {
            texture: &array.texture,
            mip_level: 0,
            origin: Origin3d {
                x: 0,
                y: 0,
                z: camera.index,
            },
            aspect: TextureAspect::All,
        },
        source.size(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_uses_one_sdr_layer_per_view_without_cpu_pixel_storage() {
        let layout = Layout {
            columns: 11,
            rows: 6,
            view_width: 186,
            view_height: 341,
        };
        let image = image(layout);
        assert_eq!(
            image.texture_descriptor.size,
            Extent3d {
                width: 186,
                height: 341,
                depth_or_array_layers: 66,
            }
        );
        assert_eq!(image.texture_descriptor.format, VIEW_FORMAT);
        assert_eq!(image.texture_descriptor.sample_count, 1);
        assert_eq!(image.texture_descriptor.mip_level_count, 1);
        assert_eq!(
            image.texture_descriptor.usage,
            TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING
        );
        assert_eq!(
            image
                .texture_view_descriptor
                .and_then(|view| view.dimension),
            Some(TextureViewDimension::D2Array)
        );
        assert!(image.data.is_none());
    }
}
