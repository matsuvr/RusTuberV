#import bevy_pbr::{
    prepass_io,
    mesh_view_bindings::view,
}
#import bevy_render::globals::Globals

#import mtoon::types::{
    material,
    ALPHA_MODE_MASK,
    ALPHA_MODE_ALPHA_TO_COVERAGE,
    RICH_SHADING,
}

#import mtoon::alpha::{
    mtoon_transformed_uv,
    mtoon_alpha_at_uv,
}

#import mtoon::uv::mtoon_animated_uv

// Shadow/prepass fragment shader. The MToon bind group has no `pbr_bindings`
// material, so the mesh's default prepass fragment cannot test alpha; without
// this shader a cutout hair card casts a solid quad shadow.
//
// A depth-only shadow key does not define `PREPASS_FRAGMENT`, so
// `prepass_io::FragmentOutput` is unavailable here: the shader only discards,
// it has no color targets.
//
// The prepass view bind group stores the frame globals at binding 1 (the
// forward view bind group has them at binding 11). Reading the same frame
// clock here lets the shadow use the material's animated UV at the same time
// as the lit pass, so a UV-animated cutout casts the silhouette it shows.
//
// The main pass uses a derivative-based cutoff for antialiasing, while the
// shadow map uses the plain cutoff: this keeps the coverage a superset of the
// lit pixels without copying the main pass's AA.
//
// The tested alpha is the shared authored base alpha (`material.base_color.a`
// times the base color texel): the same value the lit Mask test uses, so an
// authored base alpha of 0 discards the shadow exactly as it discards the lit
// fragment.
//
// The cutout shadow is a Rich effect: it is active only while the material is
// on the Rich display path and the added effect amount is positive, so the
// Native display keeps the upstream depth-only shadow even if a saved portrait
// strength is positive.
@group(0) @binding(1) var<uniform> prepass_globals: Globals;

@fragment
fn fragment(in: prepass_io::VertexOutput) {
#ifdef VERTEX_UVS_A
    if (material.portrait_strength > 0.0
        && (material.flags & RICH_SHADING) != 0u
        && ((material.flags & ALPHA_MODE_MASK) != 0u
            || (material.flags & ALPHA_MODE_ALPHA_TO_COVERAGE) != 0u))
    {
        let uv = mtoon_animated_uv(mtoon_transformed_uv(in.uv), prepass_globals.time);
        if (mtoon_alpha_at_uv(uv) < material.alpha_cutoff) {
            discard;
        }
    }
#endif
}
