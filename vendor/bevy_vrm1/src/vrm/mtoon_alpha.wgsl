#define_import_path mtoon::alpha

// The authored base color alpha, shared by the Rich transparent discard and
// the MToon shadow/prepass alpha test, so the cutout coverage follows the same
// source alpha as the lit image.
//
// The Native and Rich fragment shaders sample the base color through
// `mtoon::native`; this module is only for the look-side alpha checks. The UV
// animation is shared with the lit pass through `mtoon::uv`; the prepass entry
// point supplies the frame clock from the prepass view bind group's globals.

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

// The authored base color alpha at `uv`: `material.base_color.a` times the base
// color texel alpha. The lit pass's Mask test uses this same product, so the
// shadow/prepass test must include the authored base alpha as well; the
// texture alpha alone would leave a shadow where the lit pass discards.
fn mtoon_alpha_at_uv(uv: vec2<f32>) -> f32 {
    var base_color = material.base_color;
    if ((material.flags & BASE_COLOR_TEXTURE) != 0u) {
        base_color *= textureSampleBias(base_color_texture, base_color_sampler, uv, view.mip_bias);
    }
    return base_color.a;
}
