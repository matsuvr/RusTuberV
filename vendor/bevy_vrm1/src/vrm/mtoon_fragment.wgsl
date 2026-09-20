// The MToon Native fragment stage.
//
// This entry point is the upstream `bevy_vrm1` fragment stage at revision
// `f9593fd78136fb9e0507bcae111e09291ec9b82a` (0.9.1). The computation itself
// lives in `mtoon_native.wgsl` so the Rich fragment shader can evaluate the
// same reference; this file only binds the vertex output to it.
//
// Keep this file free of look-specific work. The Rich display is
// `mtoon_rich_fragment.wgsl` and is selected by `MToonMaterial::shading_mode`.

#import bevy_pbr::forward_io::{
    VertexOutput,
    FragmentOutput,
}
#import mtoon::types::{
    material,
    OUTLINE_WORLD_COORDINATES,
}
#import mtoon::native::{
    calc_animated_uv,
    make_pbr_input,
    make_mtoon_input,
    apply_mtoon_lighting,
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
#ifdef OUTLINE_PASS
    // Currently, txhe outline only supports world coordinates.
    if((material.outline_flags & OUTLINE_WORLD_COORDINATES) == 0u) {
        discard;
    }
#endif

    var vertex_input = in;
    vertex_input.uv = calc_animated_uv((material.uv_transform * vec3(in.uv, 1.0)).xy);

    var out: FragmentOutput;
    var pbr_input = make_pbr_input(vertex_input, is_front);
    let mtoon_input = make_mtoon_input(vertex_input, pbr_input);
    out.color = apply_mtoon_lighting(mtoon_input);

#ifdef OUTLINE_PASS
    let outline_color = material.outline_color.rgb * mix(vec3(1.), out.color.rgb, material.outline_lighting_mix_factor);
    out.color = vec4(outline_color, mtoon_input.lit_color.a);
#endif

    return out;
}
