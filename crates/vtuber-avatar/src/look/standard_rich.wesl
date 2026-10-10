// Bevy v0.20.0 pbr.wesl (MIT OR Apache-2.0), with only the existing additive
// Rich rim between lighting and the standard post-lighting processing.
import bevy_pbr::{
    render::{
        pbr_types,
        pbr_functions::alpha_discard,
        pbr_fragment::pbr_input_from_standard_material,
    },
    decal::clustered::apply_decals,
};
@if(PREPASS_PIPELINE)
import bevy_pbr::{
    prepass::io::{VertexOutput, FragmentOutput},
    deferred::functions::deferred_output,
};
@else
import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};
@if(VISIBILITY_RANGE_DITHER)
import bevy_pbr::render::pbr_functions::visibility_range_dither;
@if(MESHLET_MESH_MATERIAL_PASS)
import bevy_pbr::meshlet::visibility_buffer_resolve::resolve_vertex_output;
@if(MATERIAL_OIT_ENABLED)
import bevy_core_pipeline::oit::draw::oit_draw;
@if(FORWARD_DECAL)
import bevy_pbr::decal::forward::get_forward_decal_info;

struct RichStandardExtension {
    strength: f32,
}
@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> rich_standard: RichStandardExtension;
const RICH_RIM_STRENGTH: f32 = 0.04;
const RICH_RIM_POWER: f32 = 3.0;

fn apply_rich_rim(input: pbr_types::PbrInput) -> vec3<f32> {
    let fresnel = pow(max(1.0 - dot(input.N, input.V), 0.0), RICH_RIM_POWER);
    return rich_standard.strength * RICH_RIM_STRENGTH * fresnel * input.material.base_color.rgb;
}

@fragment
fn fragment(
    @if(MESHLET_MESH_MATERIAL_PASS)
    @builtin(position) frag_coord: vec4<f32>,
    @if(!MESHLET_MESH_MATERIAL_PASS)
    vertex_output: VertexOutput,
    @if(!MESHLET_MESH_MATERIAL_PASS)
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    @if(MESHLET_MESH_MATERIAL_PASS)
    let vertex_output = resolve_vertex_output(frag_coord);
    @if(MESHLET_MESH_MATERIAL_PASS)
    let is_front = true;
    var input = vertex_output;
    @if(VISIBILITY_RANGE_DITHER)
    visibility_range_dither(input.position, input.visibility_range_dither);
    @if(FORWARD_DECAL)
    let forward_decal_info = get_forward_decal_info(input);
    @if(FORWARD_DECAL)
    input.world_position = forward_decal_info.world_position;
    @if(FORWARD_DECAL)
    input.uv = forward_decal_info.uv;
    var pbr_input = pbr_input_from_standard_material(input, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );
    apply_decals(&pbr_input);
    @if(PREPASS_PIPELINE)
    let out = deferred_output(input, pbr_input);
    @if(!PREPASS_PIPELINE)
    var out: FragmentOutput;
    @if(!PREPASS_PIPELINE)
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
        out.color = vec4<f32>(out.color.rgb + apply_rich_rim(pbr_input), out.color.a);
    } else {
        out.color = pbr_input.material.base_color;
    }
    @if(!PREPASS_PIPELINE)
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    @if(MATERIAL_OIT_ENABLED) {
        let alpha_mode = pbr_input.material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS;
        if alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND {
            oit_draw(input.position, vec4(out.color.rgb * out.color.a, out.color.a));
            discard;
        }
        if alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_PREMULTIPLIED
            || alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_ADD {
            oit_draw(input.position, out.color);
            discard;
        }
    }
    @if(FORWARD_DECAL)
    out.color.a = min(forward_decal_info.alpha, out.color.a);
    return out;
}
