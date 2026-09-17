#import bevy_pbr::{
    forward_io::{
        VertexOutput,
        FragmentOutput,
    },
    pbr_fragment::pbr_input_from_vertex_output,
    pbr_functions::calculate_tbn_mikktspace,
    pbr_types::PbrInput,
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    shadows::fetch_directional_shadow,
    ambient::ambient_light,
    mesh_view_bindings::{
        view,
        lights,
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
    matcap_texture,
    matcap_sampler,
    emissive_texture,
    emissive_sampler,
    normal_texture,
    normal_texture_sampler,
    BASE_COLOR_TEXTURE,
    SHADING_SHIFT_TEXTURE,
    SHADE_MULTIPLY_TEXTURE,
    RIM_MAP_TEXTURE,
    MATCAP_TEXTURE,
    EMISSIVE_TEXTURE,
    NORMAL_TEXTURE,
    DOUBLE_SIDED,
    ALPHA_MODE_MASK,
    ALPHA_MODE_BLEND,
    ALPHA_MODE_ALPHA_TO_COVERAGE,
    OUTLINE_WORLD_COORDINATES,
}
#import mtoon::alpha::{
    mtoon_animated_uv,
    mtoon_base_color_at_uv,
}
#import mtoon::lighting::{
    mtoon_shading_weight,
    mtoon_direct_term,
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
    vertex_input.uv = mtoon_animated_uv(in.uv);

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

fn make_pbr_input(
    vertex_input: VertexOutput,
    is_front: bool,
) -> PbrInput{
    let double_sided = (material.flags & DOUBLE_SIDED) != 0;
    var pbr_input = pbr_input_from_vertex_output(vertex_input, is_front, double_sided);
    pbr_input.material.base_color = lit_color(vertex_input.uv);
    pbr_input.material.metallic = 0.0;
    pbr_input.material.emissive = material.emissive_color;
    pbr_input.N = mtoon_world_normal(vertex_input, pbr_input, is_front, double_sided);
    return pbr_input;
}

// Resolves the lighting normal. Without a normal texture the material uses the
// geometric (vertex/skinning/morph) normal, per the MToon surface-normal
// specification. The outline extrusion keeps using the geometric normal.
fn mtoon_world_normal(
    vertex_input: VertexOutput,
    pbr_input: PbrInput,
    is_front: bool,
    double_sided: bool,
) -> vec3<f32> {
    var normal = pbr_input.N;
#ifdef VERTEX_TANGENTS
    if ((material.flags & NORMAL_TEXTURE) != 0u) {
        let TBN = calculate_tbn_mikktspace(pbr_input.world_normal, vertex_input.world_tangent);
        var tangent_normal = textureSampleBias(
            normal_texture,
            normal_texture_sampler,
            vertex_input.uv,
            view.mip_bias,
        ).rgb * 2.0 - 1.0;
        // glTF `normalTexture.scale` scales the X and Y components.
        tangent_normal = vec3<f32>(
            tangent_normal.xy * material.normal_texture_scale,
            tangent_normal.z,
        );
        if double_sided && !is_front {
            tangent_normal = -tangent_normal;
        }
        normal = normalize(tangent_normal.x * TBN[0] + tangent_normal.y * TBN[1] + tangent_normal.z * TBN[2]);
    }
#endif
    return normal;
}

fn lit_color(uv: vec2<f32>) -> vec4<f32> {
    var base_color = material.base_color;
    if((material.flags & BASE_COLOR_TEXTURE) != 0u) {
        base_color *= mtoon_base_color_at_uv(uv);
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
    // Fully transparent fragments must not write depth. A blend material with
    // `transparentWithZWrite` would otherwise let an overlay quad's transparent
    // background occlude coplanar transparent layers behind it (for example a
    // shape-key symbol quad drawn over a speech-bubble quad).
    if((material.flags & ALPHA_MODE_BLEND) != 0u && base_color.a <= 0.0) {
        discard;
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

fn apply_mtoon_lighting(in: MToonInput) -> vec4<f32> {
    let direct = apply_directional_lights(in);
    let indirect = apply_global_illumination(in);
    let emissive = apply_emissive_light(in);
    let rim = apply_rim_lighting(in.pbr, in.uv, direct, indirect);
    // Bevy applies the camera exposure once to direct and indirect light and
    // leaves emissive absolute (`emissive_exposure_weight` defaults to 0).
    // MToon follows the same split so exposure is not applied twice.
    return vec4<f32>(view.exposure * (direct + indirect + rim) + emissive, in.lit_color.a);
}

fn apply_directional_lights(in: MToonInput) -> vec3<f32>{
    let shade_color: vec3<f32> = calc_shade_color(in);
    let shade_shift: f32 = calc_mtoon_lighting_reflectance_shading_shift(in);
    var direct: vec3<f32> = vec3(0.);
    for (var i: u32 = 0u; i < lights.n_directional_lights; i = i + 1u) {
        // The light's radiance is premultiplied by its illuminance in the
        // Bevy uniform, so a weaker secondary light contributes less. Shadow
        // maps only gate this light's own visibility; a light without shadows
        // still illuminates.
        let light = &lights.directional_lights[i];
        let shading = calc_mtoon_lighting_shading(in, i, shade_shift);
        direct += mtoon_direct_term(in.lit_color.rgb, shade_color, shading, (*light).color.rgb);
    }
    return direct;
}

fn calc_mtoon_lighting_shading(
    input: MToonInput,
    light_id: u32,
    shade_shift: f32,
) -> f32 {
    let light = &lights.directional_lights[light_id];
    let ndotl = dot(input.world_normal, (*light).direction_to_light);
    let view_z = dot(vec4<f32>(
        view.view_from_world[0].z,
        view.view_from_world[1].z,
        view.view_from_world[2].z,
        view.view_from_world[3].z
    ), input.world_position);
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
    return mtoon_shading_weight(ndotl, shade_shift, material.shading_toony_factor) * shadow;
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

// MToon's global illumination is roughly direction independent: the
// `giEqualizationFactor` interpolates between the ambient light evaluated for
// the surface normal and an average of the up/down samples, as described by
// the specification's two-point approximation.
fn apply_global_illumination(
    in: MToonInput,
) -> vec3<f32> {
    let diffuse_color = calc_diffuse_color(
        in.lit_color.rgb,
        in.pbr.material.diffuse_transmission,
    );
    let passthrough = mtoon_ambient(in, in.world_normal, diffuse_color);
    let uniformed = 0.5 * (mtoon_ambient(in, vec3<f32>(0.0, 1.0, 0.0), diffuse_color)
        + mtoon_ambient(in, vec3<f32>(0.0, -1.0, 0.0), diffuse_color));
    return mix(passthrough, uniformed, material.gi_equalization_factor);
}

fn mtoon_ambient(
    in: MToonInput,
    normal: vec3<f32>,
    diffuse_color: vec3<f32>,
) -> vec3<f32> {
    return ambient_light(
        in.world_position,
        normal,
        in.world_view_dir,
        dot(normal, in.world_view_dir),
        diffuse_color,
        vec3(0.),
        in.pbr.material.perceptual_roughness,
        in.pbr.diffuse_occlusion,
    );
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

fn apply_rim_lighting(in: PbrInput, uv: vec2<f32>, direct_light: vec3<f32>, indirect_light: vec3<f32>) -> vec3<f32>{
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
    rim *= mix(vec3(1.0), direct_light + indirect_light, material.rim_lighting_mix_factor);
    return rim;
}

fn calc_diffuse_color(
    base_color: vec3<f32>,
    diffuse_transmission: f32
) -> vec3<f32> {
    return base_color * (1.0 - diffuse_transmission);
}
