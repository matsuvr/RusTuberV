// MToon-compatible cel lighting, inspired by NiloCat's public toon example.
// Copyright (c) 2020 ColinLeung-NiloCat (MIT); see THIRD_PARTY_NOTICES.md.
// Bevy light decoding follows Bevy 0.20 (MIT OR Apache-2.0).
// MToon owns alpha, textures, shade colors, emission, matcap and authored rim.
import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    mesh_view_bindings::{view, lights, clustered_lights},
    mesh_view_types::{
        DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
        POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
        POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
    },
    clustered_forward::{
        get_clusterable_object_id,
        unpack_clusterable_object_index_ranges,
        view_fragment_cluster_index,
    },
    pbr_lighting::getDistanceAttenuation,
    shadows::{fetch_directional_shadow, fetch_spot_shadow},
};
import bevy_vrm1::vrm::{
    mtoon_types::{MToonInput, material, OUTLINE_WORLD_COORDINATES},
    mtoon_fragment::{
        make_pbr_input, make_mtoon_input, apply_mtoon_lighting, calc_uv_time,
        calc_shade_color, calc_mtoon_lighting_reflectance_shading_shift,
        apply_global_illumination, apply_emissive_light, apply_rim_lighting,
    },
};
import vtuber_avatar::look::mtoon_uv::animate_uv;

struct RichMtoonExtension {
    strength: f32,
}
// Upstream MToon occupies bindings 100..116.
@group(constants::MATERIAL_BIND_GROUP) @binding(200) var<uniform> rich_mtoon: RichMtoonExtension;

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
    let pbr_input = make_pbr_input(vertex_input, is_front);
    let input = make_mtoon_input(vertex_input, pbr_input);
    var out: FragmentOutput;
    let native = apply_mtoon_lighting(input);
    out.color = native;
    if rich_mtoon.strength > 0.0 {
        out.color = vec4(mix(native.rgb, apply_cel_lighting(input), rich_mtoon.strength), native.a);
    }
    @if(OUTLINE_PASS) {
        let outline_color = material.outline_color.rgb * mix(vec3(1.0), out.color.rgb, material.outline_lighting_mix_factor);
        out.color = vec4(outline_color, input.lit_color.a);
    }
    return out;
}

fn cel_surface(input: MToonInput, shade: vec3<f32>, shift: f32, softness: f32, direction: vec3<f32>, shadow: f32) -> vec3<f32> {
    // Keep the signed dot product: back-facing surfaces belong to the shade
    // side. MToon's own shift/texture and toony factor define the boundary.
    let area = smoothstep(-softness, softness, dot(input.world_normal, direction) + shift);
    return mix(shade, input.lit_color.rgb, area * shadow);
}

fn apply_cel_lighting(input: MToonInput) -> vec3<f32> {
    let shade = calc_shade_color(input);
    let shift = calc_mtoon_lighting_reflectance_shading_shift(input);
    // Derivatives are evaluated before the per-cluster light loop. They keep
    // fully hard author settings finite and soften subpixel cel boundaries.
    let pixel_width = length(fwidth(input.world_normal)) + fwidth(shift);
    let softness = max(1.0 - material.shading_toony_factor, max(pixel_width, 0.0001));
    let view_z = dot(vec4(
        view.view_from_world[0].z,
        view.view_from_world[1].z,
        view.view_from_world[2].z,
        view.view_from_world[3].z
    ), input.world_position);
    var direct = vec3(0.0);
    var light_sum = vec3(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i += 1u) {
        let light = lights.directional_lights[i];
        var shadow = 1.0;
        if (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_directional_shadow(i, input.world_position, input.world_normal, view_z, input.pbr.frag_coord.xy);
        }
        // Bevy stores physical illuminance. Convert to display exposure once,
        // then bound the light, not the author's base/shade/emission colors.
        let energy = saturate(view.exposure * light.color.rgb);
        direct += cel_surface(input, shade, shift, softness, light.direction_to_light, shadow) * energy;
        light_sum += energy;
    }
    let cluster = view_fragment_cluster_index(input.pbr.frag_coord.xy, view_z, input.pbr.is_orthographic);
    let ranges = unpack_clusterable_object_index_ranges(cluster);
    // The existing look rig supplies two spots; rect lights follow this range.
    for (var i = ranges.first_spot_light_index_offset; i < ranges.first_rect_light_index_offset; i += 1u) {
        let id = get_clusterable_object_id(i);
        let light = clustered_lights.data[id];
        let offset = light.position_radius.xyz - input.world_position.xyz;
        let distance_squared = max(dot(offset, offset), 0.0001);
        let direction = offset * inverseSqrt(distance_squared);
        var spot_direction = vec3(light.light_custom_data.x, 0.0, light.light_custom_data.y);
        spot_direction.y = sqrt(max(0.0, 1.0 - dot(spot_direction, spot_direction)));
        if (light.flags & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE) != 0u {
            spot_direction.y = -spot_direction.y;
        }
        let cone = saturate(dot(-spot_direction, direction) * light.light_custom_data.z + light.light_custom_data.w);
        let attenuation = min(4.0, getDistanceAttenuation(distance_squared, light.color_inverse_square_range.w)) * cone * cone;
        // The shared rig scales lumens for Standard materials. Recover its
        // full-strength value here so the final MToon mix applies strength once.
        let energy = 0.25 * attenuation * saturate(view.exposure * light.color_inverse_square_range.rgb / rich_mtoon.strength);
        var shadow = 1.0;
        if (light.flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_spot_shadow(id, input.world_position, input.world_normal, light.shadow_map_near_z, input.pbr.frag_coord.xy);
        }
        direct += cel_surface(input, shade, shift, softness, direction, shadow) * energy;
        light_sum += energy;
    }
    // Normalize only excess light energy, preserving light hue and authored
    // HDR colors. Nearby spots cannot wash the albedo out to white.
    direct /= max(1.0, max(light_sum.r, max(light_sum.g, light_sum.b)));
    let indirect = apply_global_illumination(input);
    // Nilo's max composition keeps ambient and direct from bleaching textures.
    let diffuse = max(indirect, direct);
    let rim = apply_rim_lighting(input.pbr, input.uv, diffuse, vec3(0.0));
    return diffuse + rim + apply_emissive_light(input);
}
