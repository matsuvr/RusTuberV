#define_import_path mtoon::alpha

// Shared base-color sampling and UV transform used by the main MToon pass and
// the shadow/prepass alpha test, so the cutout coverage follows the same source
// alpha as the lit image.

#import bevy_pbr::mesh_view_bindings::{view, globals}

#import mtoon::types::{
    material,
    base_color_texture,
    base_color_sampler,
    uv_animation_mask_texture,
    uv_animation_mask_sampler,
    BASE_COLOR_TEXTURE,
    UV_ANIMATION_MASK_TEXTURE,
}

// The material's UV transform. Available in the shadow/prepass pipeline, which
// has no access to the view time globals.
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

// The transformed UV followed by the perpetual UV animation. This reads the
// view time globals, so it is only usable from the main pass; the shadow pass
// uses `mtoon_transformed_uv`.
fn mtoon_animated_uv(uv: vec2<f32>) -> vec2<f32> {
    let transformed = mtoon_transformed_uv(uv);
    let time = mtoon_uv_time(transformed);
    let translate = time * vec2(material.uv_animation_scroll_speed_x, material.uv_animation_rotation_speed_y);
    let rotate_rad = fract(time * material.uv_animation_rotation_speed);
    let cos_rotate = cos(rotate_rad);
    let sin_rotate = sin(rotate_rad);
    let pivot = vec2<f32>(0.5, 0.5);
    return mat2x2(cos_rotate, -sin_rotate, sin_rotate, cos_rotate) * (transformed - pivot) + pivot + translate;
}

fn mtoon_uv_time(uv: vec2<f32>) -> f32 {
    if ((material.flags & UV_ANIMATION_MASK_TEXTURE) != 0u) {
        let mask = textureSampleBias(uv_animation_mask_texture, uv_animation_mask_sampler, uv, view.mip_bias).b;
        return mask * globals.time;
    }
    return globals.time;
}
