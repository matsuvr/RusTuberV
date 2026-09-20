#define_import_path mtoon::alpha

// The static base-color sampling used by the MToon shadow/prepass alpha test,
// so the cutout coverage follows the same source alpha as the lit image.
//
// The Native and Rich fragment shaders sample the base color through
// `mtoon::native`; this module is only for the shadow/prepass pipeline, which
// has no access to the view time globals.

#import bevy_pbr::mesh_view_bindings::view

#import mtoon::types::{
    material,
    base_color_texture,
    base_color_sampler,
    BASE_COLOR_TEXTURE,
}

// The material's UV transform.
fn mtoon_transformed_uv(uv: vec2<f32>) -> vec2<f32> {
    return (material.uv_transform * vec3(uv, 1.0)).xy;
}

// The base color texel at `uv`, or opaque white when the material has no base
// color texture.
fn mtoon_base_color_at_uv(uv: vec2<f32>) -> vec4<f32> {
    if ((material.flags & BASE_COLOR_TEXTURE) == 0u) {
        return vec4<f32>(1.0);
    }
    return textureSampleBias(base_color_texture, base_color_sampler, uv, view.mip_bias);
}

// The base color alpha at `uv`.
fn mtoon_alpha_at_uv(uv: vec2<f32>) -> f32 {
    return mtoon_base_color_at_uv(uv).a;
}
