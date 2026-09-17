#import bevy_pbr::{
    prepass_io,
    mesh_view_bindings::view,
}

#import mtoon::types::{
    material,
    ALPHA_MODE_MASK,
    ALPHA_MODE_ALPHA_TO_COVERAGE,
}

#import mtoon::alpha::{
    mtoon_transformed_uv,
    mtoon_alpha_at_uv,
}

// Shadow/prepass fragment shader. The MToon bind group has no `pbr_bindings`
// material, so the mesh's default prepass fragment cannot test alpha; without
// this shader a cutout hair card casts a solid quad shadow.
//
// A depth-only shadow key does not define `PREPASS_FRAGMENT`, so
// `prepass_io::FragmentOutput` is unavailable here: the shader only discards,
// it has no color targets.
//
// The main pass uses a derivative-based cutoff for antialiasing, while the
// shadow map uses the plain cutoff: this keeps the coverage a superset of the
// lit pixels without copying the main pass's AA.
@fragment
fn fragment(in: prepass_io::VertexOutput) {
#ifdef VERTEX_UVS_A
    if ((material.flags & ALPHA_MODE_MASK) != 0u
        || (material.flags & ALPHA_MODE_ALPHA_TO_COVERAGE) != 0u)
    {
        let uv = mtoon_transformed_uv(in.uv);
        if (mtoon_alpha_at_uv(uv) < material.alpha_cutoff) {
            discard;
        }
    }
#endif
}
