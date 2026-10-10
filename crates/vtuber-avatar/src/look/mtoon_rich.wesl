// App-side additive Rich MToon fragment shader.
// WESL lets us import the Native base instead of keeping a second copy of its
// alpha, material and lighting functions. Only the two added terms are local.
import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    mesh_view_bindings::view,
    clustered_forward::{
        get_clusterable_object_id,
        unpack_clusterable_object_index_ranges,
        view_fragment_cluster_index,
    },
    pbr_lighting::{F_AB, LightingInput, perceptualRoughnessToRoughness, spot_light},
};
import bevy_vrm1::vrm::{
    mtoon_types::{MToonInput, material, OUTLINE_WORLD_COORDINATES},
    mtoon_fragment::{make_pbr_input, make_mtoon_input, apply_mtoon_lighting, calc_uv_time},
};
import vtuber_avatar::look::mtoon_uv::animate_uv;

// These artistic constants and the light-side strength remain unchanged.
const RICH_GLOSS_ROUGHNESS: f32 = 0.35;
const RICH_RIM_STRENGTH: f32 = 0.75;
const RICH_RIM_POWER: f32 = 3.0;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    @if(OUTLINE_PASS) {
        if ((material.outline_flags & OUTLINE_WORLD_COORDINATES) == 0u) {
            discard;
        }
    }
    var vertex_input = in;
    let uv = (material.uv_transform * vec3(in.uv, 1.0)).xy;
    vertex_input.uv = animate_uv(uv, calc_uv_time(uv), vec2(material.uv_animation_scroll_speed_x, material.uv_animation_rotation_speed_y), material.uv_animation_rotation_speed);
    var out: FragmentOutput;
    var pbr_input = make_pbr_input(vertex_input, is_front);
    let mtoon_input = make_mtoon_input(vertex_input, pbr_input);
    out.color = apply_mtoon_lighting(mtoon_input);
    @if(!OUTLINE_PASS) {
        let added_light = apply_added_lighting(mtoon_input);
        let added_rim = apply_added_rim(mtoon_input, added_light);
        out.color = vec4<f32>(out.color.rgb + added_light + added_rim, out.color.a);
    }
    @if(OUTLINE_PASS) {
        let outline_color = material.outline_color.rgb * mix(vec3(1.), out.color.rgb, material.outline_lighting_mix_factor);
        out.color = vec4(outline_color, mtoon_input.lit_color.a);
    }
    return out;
}

// Sum only the preset's spot lights. The light intensity already contains
// strength; the exposure conversion also happens exactly once.
fn apply_added_lighting(in: MToonInput) -> vec3<f32> {
    let N = in.pbr.N;
    let V = in.pbr.V;
    let NdotV = max(dot(N, V), 0.0001);
    let perceptual_roughness = RICH_GLOSS_ROUGHNESS;
    let roughness = perceptualRoughnessToRoughness(perceptual_roughness);
    var lighting_input: LightingInput;
    lighting_input.layers[0].N = N;
    lighting_input.layers[0].R = reflect(-V, N);
    lighting_input.layers[0].NdotV = NdotV;
    lighting_input.layers[0].perceptual_roughness = perceptual_roughness;
    lighting_input.layers[0].roughness = roughness;
    lighting_input.P = in.pbr.world_position.xyz;
    lighting_input.V = V;
    lighting_input.diffuse_color = in.lit_color.rgb;
    lighting_input.metallic = 0.0;
    lighting_input.F0_dielectric = vec3<f32>(0.04);
    lighting_input.F0_metallic = in.lit_color.rgb;
    lighting_input.F_ab = F_AB(perceptual_roughness, NdotV);
    let view_z = dot(vec4<f32>(
        view.view_from_world[0].z,
        view.view_from_world[1].z,
        view.view_from_world[2].z,
        view.view_from_world[3].z
    ), in.pbr.world_position);
    let cluster_index = view_fragment_cluster_index(in.pbr.frag_coord.xy, view_z, in.pbr.is_orthographic);
    let ranges = unpack_clusterable_object_index_ranges(cluster_index);
    var added = vec3<f32>(0.0);
    // In Bevy 0.20 rect lights follow spot lights in the cluster index list.
    for (var i: u32 = ranges.first_spot_light_index_offset; i < ranges.first_rect_light_index_offset; i = i + 1u) {
        let light_id = get_clusterable_object_id(i);
        added += spot_light(light_id, &lighting_input, true);
    }
    return view.exposure * added;
}

// A view Fresnel scaled by the added light: no added light means no added rim.
fn apply_added_rim(in: MToonInput, added_light: vec3<f32>) -> vec3<f32> {
    let fresnel = pow(saturate(1.0 - dot(in.pbr.N, in.pbr.V)), RICH_RIM_POWER);
    return RICH_RIM_STRENGTH * fresnel * added_light;
}
