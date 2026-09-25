// App-side Rich MToon fragment shader (RusTuberV #94).
//
// The native base pass below is copied from the transparency-fixed upstream
// bevy_vrm1 v0.9.3 `src/vrm/mtoon_fragment.wgsl` (fork commit
// 6ee2f610d6c95b542a3317028da10aba26c9e8de, MIT OR Apache-2.0) because the
// upstream MToon fragment functions are not an importable module. The app-side
// additions are `apply_added_lighting` / `apply_added_rim` only: the native
// result is kept and those terms are summed on top. The added light carries
// the look strength on the light side (#93), and the rim is a function of that
// added light, so with no spot lights (OFF or 0%) both added terms are exactly
// zero and the native display is kept.
//
// The material bindings come from the upstream `mtoon::types` module, and the
// Rich material keeps the upstream bind group layout unchanged. The app-side
// outline connection selects Rich meshes explicitly.

#import bevy_pbr::{
    forward_io::{
        VertexOutput,
        FragmentOutput,
    },
    pbr_fragment::pbr_input_from_vertex_output,
    pbr_types::PbrInput,
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    shadows::fetch_directional_shadow,
    ambient::ambient_light,
    mesh_view_bindings::{
        view,
        lights,
        globals,
    },
    clustered_forward::{
        get_clusterable_object_id,
        unpack_clusterable_object_index_ranges,
        view_fragment_cluster_index,
    },
    lighting::{
        F_AB,
        LightingInput,
        perceptualRoughnessToRoughness,
        spot_light,
    },
}
#import mtoon::types::{
    MToonInput,
    MToonMaterialUniform,
    material,
    base_color_texture,
    base_color_sampler,
    shading_shift_texture,
    shading_shift_texture_sampler,
    shade_multiply_texture,
    shade_multiply_texture_sampler,
    rim_multiply_texture,
    rim_multiply_sampler,
    uv_animation_mask_texture,
    uv_animation_mask_sampler,
    matcap_texture,
    matcap_sampler,
    emissive_texture,
    emissive_sampler,
    BASE_COLOR_TEXTURE,
    SHADING_SHIFT_TEXTURE,
    SHADE_MULTIPLY_TEXTURE,
    RIM_MAP_TEXTURE,
    UV_ANIMATION_MASK_TEXTURE,
    MATCAP_TEXTURE,
    EMISSIVE_TEXTURE,
    DOUBLE_SIDED,
    ALPHA_MODE_MASK,
    ALPHA_MODE_BLEND,
    ALPHA_MODE_ALPHA_TO_COVERAGE,
    OUTLINE_WORLD_COORDINATES,
}

// MToon has no roughness input; the app-side gloss is an artistic layer, so it
// uses one fixed dielectric roughness.
const RICH_GLOSS_ROUGHNESS: f32 = 0.35;
// The added rim is a weak view Fresnel scaled by the added spot-light
// contribution, so it fades with the look strength and reaches zero with it.
const RICH_RIM_STRENGTH: f32 = 0.75;
const RICH_RIM_POWER: f32 = 3.0;
// UniVRM MToon10 alpha blending threshold (EPSILON_FP16), also used by Native.
const MTOON_ALPHA_EPSILON: f32 = 0.0009765625;

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

#ifndef OUTLINE_PASS
    let added_light = apply_added_lighting(mtoon_input);
    let added_rim = apply_added_rim(mtoon_input, added_light);
    out.color = vec4<f32>(out.color.rgb + added_light + added_rim, out.color.a);
#endif

#ifdef OUTLINE_PASS
    let outline_color = material.outline_color.rgb * mix(vec3(1.), out.color.rgb, material.outline_lighting_mix_factor);
    out.color = vec4(outline_color, mtoon_input.lit_color.a);
#endif

    return out;
}

fn make_pbr_input(
    vertex_input: VertexOutput,
    is_front: bool,
) -> PbrInput{
    let double_sided = (material.flags & DOUBLE_SIDED) != 0;
    var pbr_input = pbr_input_from_vertex_output(vertex_input, is_front, double_sided);
    pbr_input.material.base_color = lit_color(vertex_input.uv);
    pbr_input.material.metallic = 0.0;
    pbr_input.material.emissive = material.emissive_color;
    return pbr_input;
}

fn lit_color(uv: vec2<f32>) -> vec4<f32> {
    var base_color = material.base_color;
    if((material.flags & BASE_COLOR_TEXTURE) != 0u) {
        base_color *= textureSampleBias(base_color_texture, base_color_sampler, uv, view.mip_bias);
    }
    if ((material.flags & ALPHA_MODE_BLEND) != 0u) {
        if (base_color.a < MTOON_ALPHA_EPSILON) {
            discard;
        }
    }
    if((material.flags & ALPHA_MODE_MASK) != 0u || (material.flags & ALPHA_MODE_ALPHA_TO_COVERAGE) != 0u) {
        let raw = base_color.a;
        let tmpAlpha = (raw - material.alpha_cutoff) / max(fwidth(raw), 0.00001) + 0.5;
        if(tmpAlpha < material.alpha_cutoff) {
            discard;
        }else{
            base_color.a = 1.0;
        }
    }
#ifdef OUTLINE_PASS
    if((material.flags & ALPHA_MODE_BLEND) != 0u) {
        base_color.a = 1.0;
    }
#endif
    return base_color;
}

fn make_mtoon_input(in: VertexOutput, pbr_input: PbrInput) -> MToonInput{
    let uv = in.uv;
    return MToonInput(
        pbr_input,
        uv,
        pbr_input.V,
        in.world_position,
        pbr_input.N,
        pbr_input.material.base_color,
    );
}

fn calc_animated_uv(uv: vec2<f32>) -> vec2<f32>{
    let time = calc_uv_time(uv);
    let translate = time * vec2(material.uv_animation_scroll_speed_x, material.uv_animation_rotation_speed_y);
    let rotate_rad = fract(time * material.uv_animation_rotation_speed);
    let cos_rotate = cos(rotate_rad);
    let sin_rotate = sin(rotate_rad);
    let pivot = vec2<f32>(0.5, 0.5);
    return mat2x2(cos_rotate, -sin_rotate, sin_rotate, cos_rotate) * (uv - pivot) + pivot + translate;
}

fn calc_uv_time(uv: vec2<f32>) -> f32{
    if((material.flags & UV_ANIMATION_MASK_TEXTURE) != 0u) {
        let mask = textureSampleBias(uv_animation_mask_texture, uv_animation_mask_sampler, uv, view.mip_bias).b;
        return mask * globals.time;
    }else{
        return globals.time;
    }
}

fn apply_mtoon_lighting(in: MToonInput) -> vec4<f32> {
    let direct = apply_directional_lights(in);
    let in_direct = apply_global_illumination(in);
    let emissive = apply_emissive_light(in);
    let rim = apply_rim_lighting(in.pbr, in.uv, direct, in_direct);
    return vec4<f32>(direct + in_direct + emissive + rim, in.lit_color.a);
}

fn apply_directional_lights(in: MToonInput) -> vec3<f32>{
    var direct: vec3<f32> = vec3(0.);
    var shade_color: vec3<f32> = calc_shade_color(in);
    var shading: f32 = 0.0;
    for (var i: u32 = 0u; i < lights.n_directional_lights; i = i + 1u) {
        // Keep the light's direct contribution regardless of whether shadow
        // maps are enabled for it. `shadows_enabled` gates shadow-map
        // sampling only; Bevy PBR treats it the same way. Previously this
        // loop skipped the entire light when shadows were disabled, which
        // made shadow-less directional lights fail to illuminate MToon
        // characters at all.
        shading += calc_mtoon_lighting_shading(in, i);
    }
    return mix(shade_color, in.lit_color.rgb, shading);
}

fn calc_mtoon_lighting_shading(
    input: MToonInput,
    light_id: u32,
) -> f32 {
    let light = &lights.directional_lights[light_id];
    let NdotL = saturate(dot(input.world_normal, (*light).direction_to_light));
    let shade_shift = calc_mtoon_lighting_reflectance_shading_shift(input);
    let shade_input = mix(-1., 1., mtoon_linearstep(-1., 1., NdotL));
    let view_z = dot(vec4<f32>(
        view.view_from_world[0].z,
        view.view_from_world[1].z,
        view.view_from_world[2].z,
        view.view_from_world[3].z
    ), input.world_position);
    // Only sample the shadow map when this light actually casts shadows;
    // default to full illumination otherwise so the light still contributes
    // to MToon shading.
    var shadow = 1.0;
    if ((*light).flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
        shadow = fetch_directional_shadow(
            light_id,
            input.world_position,
            input.world_normal,
            view_z,
            input.pbr.frag_coord.xy,
        );
    }
    let shading =  mtoon_linearstep(-1.0 + material.shading_toony_factor, 1.0 - material.shading_toony_factor, shade_input + shade_shift) * shadow;
   return shading;
}

fn calc_mtoon_lighting_reflectance_shading_shift(
    input: MToonInput,
) -> f32 {
    if((material.flags & SHADING_SHIFT_TEXTURE) != 0u) {
        return textureSampleBias(shading_shift_texture, shading_shift_texture_sampler, input.uv, view.mip_bias).r * material.shading_shift_texture_scale + material.shading_shift_factor;
    } else {
        return material.shading_shift_factor;
    }
}

//FIXME: This code is likely an incomplete implementation.
// https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_materials_mtoon-1.0/README.md#lighting
fn apply_global_illumination(
    in: MToonInput,
) -> vec3<f32> {
    let base_color = in.lit_color.rgb;
    let diffuse_color = calc_diffuse_color(
        base_color,
        in.pbr.material.diffuse_transmission,
    );
    let in_direct_light = ambient_light(
        in.world_position,
        in.world_normal,
        in.world_view_dir,
        dot(in.world_normal, in.world_view_dir),
        diffuse_color,
        // Is the reflection color unnecessary?
        vec3(0.),
        in.pbr.material.perceptual_roughness,
        in.pbr.diffuse_occlusion,
    );
    return view.exposure * in_direct_light;
}

fn calc_shade_color(in: MToonInput) -> vec3<f32>{
   let base_color = material.shade_color.rgb;
   if((material.flags & SHADE_MULTIPLY_TEXTURE) != 0u) {
       return base_color * textureSampleBias(shade_multiply_texture, shade_multiply_texture_sampler, in.uv, view.mip_bias).rgb;
   }else{
      return base_color;
   }
}

fn apply_emissive_light(in: MToonInput) -> vec3<f32> {
    let emissive = in.pbr.material.emissive.rgb;
    if ((material.flags & EMISSIVE_TEXTURE) != 0u) {
        return emissive * textureSampleBias(emissive_texture, emissive_sampler, in.uv, view.mip_bias).rgb;
    } else {
        return emissive;
    }
}

fn apply_rim_lighting(in: PbrInput, uv: vec2<f32>, direct_light: vec3<f32>, in_direct: vec3<f32>) -> vec3<f32>{
    var rim = vec3(0.);
    let world_view_x = normalize(vec3<f32>(in.V.z, 0.0, -in.V.x));
    let world_view_y = cross(in.V, world_view_x);
    let matcap_uv = vec2<f32>(dot(world_view_x, in.N), dot(world_view_y, in.N)) * 0.495 + 0.5;
    let epsilon = 0.0001;
    if((material.flags & MATCAP_TEXTURE) != 0u) {
        rim = material.mat_cap_color.rgb * textureSampleBias(matcap_texture, matcap_sampler, matcap_uv, view.mip_bias).rgb;
    }

    let parametric_rim = saturate(1.0 - dot(in.N, in.V) + material.parametric_rim_lift_factor);
    rim += pow(parametric_rim, max(material.parametric_rim_fresnel_power, epsilon)) * material.parametric_rim_color.rgb;
    if((material.flags & RIM_MAP_TEXTURE) != 0u) {
        rim *= textureSampleBias(rim_multiply_texture, rim_multiply_sampler, uv, view.mip_bias).rgb;
    }
    rim *= mix(vec3(1.0), direct_light + in_direct, material.rim_lighting_mix_factor);
    return rim;
}

fn mtoon_linearstep(a: f32, b: f32, t: f32) -> f32 {
    return saturate((t - a) / (b - a));
}

fn calc_diffuse_color(
    base_color: vec3<f32>,
    diffuse_transmission: f32
) -> vec3<f32> {
    return base_color * (1.0 - diffuse_transmission);
}

// The additional spot lights (#93's two fixed lights).
//
// The native base already carries the single front directional light, and the
// preset adds no point lights, so only the spot list is summed. The spot
// intensity already contains the look strength on the light side (#93), so it
// is used as is: the strength is never applied twice. The exposure conversion
// happens once, here, and the term is added to the native result.
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
    for (var i: u32 = ranges.first_spot_light_index_offset; i < ranges.first_reflection_probe_index_offset; i = i + 1u) {
        let light_id = get_clusterable_object_id(i);
        added += spot_light(light_id, &lighting_input, true);
    }
    return view.exposure * added;
}

// The weak added rim: a view Fresnel scaled by the added spot-light
// contribution, so it is zero whenever the added light is zero (OFF, 0%, or
// no light reaching the fragment) and never forces a white edge on a dark
// material.
fn apply_added_rim(in: MToonInput, added_light: vec3<f32>) -> vec3<f32> {
    let fresnel = pow(saturate(1.0 - dot(in.pbr.N, in.pbr.V)), RICH_RIM_POWER);
    return RICH_RIM_STRENGTH * fresnel * added_light;
}
