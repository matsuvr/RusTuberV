#define_import_path mtoon::uv

// The MToon UV animation, shared by the lit fragment stage and the
// shadow/prepass alpha test.
//
// The frame clock is a parameter, never a global read: the lit passes read the
// forward view globals (`mtoon::native`), the shadow/prepass alpha test reads
// the prepass view bind group's globals. Both passes of one frame therefore
// evaluate the same clock, so a UV-animated cutout keeps the silhouette it
// shows in the lit image when its shadow is generated.
//
// `mtoon_uv_animation` is the upstream `calc_animated_uv` expression; do not
// change it. `mtoon::native` must keep rendering the upstream `bevy_vrm1`
// result at revision `f9593fd78136fb9e0507bcae111e09291ec9b82a` (0.9.1).

#import bevy_pbr::mesh_view_bindings::view

#import mtoon::types::{
    material,
    uv_animation_mask_texture,
    uv_animation_mask_sampler,
    UV_ANIMATION_MASK_TEXTURE,
}

// The animation clock at `uv`: the caller's view time, scaled per texel by the
// material's UV animation mask when it has one.
fn mtoon_uv_time(uv: vec2<f32>, view_time: f32) -> f32 {
    if((material.flags & UV_ANIMATION_MASK_TEXTURE) != 0u) {
        let mask = textureSampleBias(uv_animation_mask_texture, uv_animation_mask_sampler, uv, view.mip_bias).b;
        return mask * view_time;
    }else{
        return view_time;
    }
}

// The pure UV animation expression at a given animation clock.
fn mtoon_uv_animation(uv: vec2<f32>, time: f32) -> vec2<f32> {
    let translate = time * vec2(material.uv_animation_scroll_speed_x, material.uv_animation_rotation_speed_y);
    let rotate_rad = fract(time * material.uv_animation_rotation_speed);
    let cos_rotate = cos(rotate_rad);
    let sin_rotate = sin(rotate_rad);
    let pivot = vec2<f32>(0.5, 0.5);
    return mat2x2(cos_rotate, -sin_rotate, sin_rotate, cos_rotate) * (uv - pivot) + pivot + translate;
}

// The material's animated UV for one frame clock.
fn mtoon_animated_uv(uv: vec2<f32>, view_time: f32) -> vec2<f32> {
    return mtoon_uv_animation(uv, mtoon_uv_time(uv, view_time));
}
