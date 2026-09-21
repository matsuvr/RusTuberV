// The MToon Rich fragment stage, selected by `MToonMaterial::shading_mode`.
//
// The entry point always evaluates the fixed Native result first
// (`mtoon::native`), then the Rich terms, and composes them with
// `compose_rich_mtoon`. Because the composition is the identity at
// `strength == 0`, choosing Rich with no added effect renders exactly the
// Native display. The Rich inputs are individually gated on `strength > 0` as
// well, so no look-specific normal, alpha or shadow input reaches the Native
// computation.

#import bevy_pbr::{
    forward_io::{
        VertexOutput,
        FragmentOutput,
    },
    pbr_functions::calculate_tbn_mikktspace,
    pbr_types::PbrInput,
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    shadows::fetch_directional_shadow,
    ambient::ambient_light,
    lighting::perceptualRoughnessToRoughness,
    mesh_view_bindings::{
        view,
        lights,
    },
}
#import mtoon::types::{
    MToonInput,
    MToonPortraitUniform,
    material,
    normal_texture,
    normal_texture_sampler,
    DOUBLE_SIDED,
    ALPHA_MODE_BLEND,
    NORMAL_TEXTURE,
    OUTLINE_WORLD_COORDINATES,
}
#import mtoon::native::{
    calc_animated_uv,
    make_pbr_input,
    make_mtoon_input,
    apply_mtoon_lighting,
    apply_emissive_light,
    apply_rim_lighting,
    calc_diffuse_color,
    calc_mtoon_lighting_reflectance_shading_shift,
    calc_shade_color,
}
#import mtoon::lighting::{
    mtoon_direct_term,
    mtoon_shading_weight,
}
#import mtoon::alpha::mtoon_alpha_at_uv
#import mtoon::portrait::{
    compose_rich_mtoon,
    portrait_direct_specular,
    portrait_environment_specular,
    portrait_rim,
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

    // The fixed Native result: untouched normal, alpha, shadow and UV inputs.
    let native_pbr_input = make_pbr_input(vertex_input, is_front);
    let native_input = make_mtoon_input(vertex_input, native_pbr_input);
    let native = apply_mtoon_lighting(native_input);

    // The Rich display never writes depth for a fully transparent blend
    // fragment, which would otherwise let an overlay quad's transparent
    // background occlude coplanar transparent layers behind it (for example a
    // shape-key symbol quad drawn over a speech-bubble quad). This is a Rich
    // term: at strength 0 the Native alpha/depth behavior is kept exactly.
    //
    // The check uses the authored base alpha directly: the Native `lit_color`
    // forces blend alpha to 1 in the outline pass, so it cannot be used to
    // detect the transparent region there.
    if (material.portrait_strength > 0.0
        && (material.flags & ALPHA_MODE_BLEND) != 0u
        && mtoon_alpha_at_uv(vertex_input.uv) <= 0.0)
    {
        discard;
    }

    // The Rich terms are a lit-pass layer: the outline pass keeps the author's
    // own outline result, mixed from the fixed Native lit color, so the added
    // portrait specular/IBL/rim never flows into the line color and the line
    // is the same as the Native display's at any strength.
#ifdef OUTLINE_PASS
    let outline_color = material.outline_color.rgb * mix(vec3(1.), native.rgb, material.outline_lighting_mix_factor);
    var color = vec4(outline_color, native_input.lit_color.a);
#else
    let rich_pbr_input = make_rich_pbr_input(vertex_input, is_front);
    let rich_input = make_mtoon_input(vertex_input, rich_pbr_input);
    let rich_rgb = apply_rich_mtoon_lighting(rich_input, make_portrait_params());

    var color = compose_rich_mtoon(native, rich_rgb, material.portrait_strength);
#endif

    var out: FragmentOutput;
    out.color = color;
    return out;
}

// The Rich lighting normal: the geometric normal at strength 0, and the
// material's normal texture while the look adds its effects.
fn make_rich_pbr_input(
    vertex_input: VertexOutput,
    is_front: bool,
) -> PbrInput {
    var pbr_input = make_pbr_input(vertex_input, is_front);
    let double_sided = (material.flags & DOUBLE_SIDED) != 0;
    pbr_input.N = rich_world_normal(vertex_input, pbr_input, is_front, double_sided);
    return pbr_input;
}

fn rich_world_normal(
    vertex_input: VertexOutput,
    pbr_input: PbrInput,
    is_front: bool,
    double_sided: bool,
) -> vec3<f32> {
    var normal = pbr_input.N;
#ifdef VERTEX_TANGENTS
    if (material.portrait_strength > 0.0 && (material.flags & NORMAL_TEXTURE) != 0u) {
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

fn make_portrait_params() -> MToonPortraitUniform {
    return MToonPortraitUniform(
        material.portrait_strength,
        material.portrait_specular_gain,
        material.portrait_perceptual_roughness,
        material.portrait_environment_gain,
        material.portrait_rim_gain,
        material.portrait_rim_power,
    );
}

// The Rich lighting result for the same lights as the Native path: each light
// contributes its own color and radiance, and the added specular and rim are
// collected alongside.
struct MToonRichDirectResult {
    direct: vec3<f32>,
    specular: vec3<f32>,
    rim: vec3<f32>,
}

fn apply_rich_mtoon_lighting(in: MToonInput, params: MToonPortraitUniform) -> vec3<f32> {
    let lights_result = apply_directional_lights(in, params);
    let indirect = apply_rich_global_illumination(in);
    let emissive = apply_emissive_light(in);
    // The author's own MatCap/parametric rim stays exactly as authored.
    let author_rim = apply_rim_lighting(in.pbr, in.uv, lights_result.direct, indirect);
    // Bevy applies the camera exposure once to direct and indirect light and
    // leaves emissive absolute (`emissive_exposure_weight` defaults to 0).
    // The added portrait terms carry their own exposure so the terms and the
    // scene light stay on the same scale.
    let base = view.exposure * (lights_result.direct + indirect + author_rim) + emissive;
    let environment_specular = portrait_environment_specular(
        in.world_normal,
        in.world_view_dir,
        perceptualRoughnessToRoughness(params.perceptual_roughness),
        in.world_position.xyz,
    );
    let extra = params.specular_gain * lights_result.specular
        + params.environment_gain * environment_specular
        + params.rim_gain * lights_result.rim;
    return base + view.exposure * extra;
}

fn apply_directional_lights(in: MToonInput, params: MToonPortraitUniform) -> MToonRichDirectResult {
    let shade_color: vec3<f32> = calc_shade_color(in);
    let shade_shift: f32 = calc_mtoon_lighting_reflectance_shading_shift(in);
    let roughness = perceptualRoughnessToRoughness(params.perceptual_roughness);
    var result = MToonRichDirectResult(vec3(0.), vec3(0.), vec3(0.));
    for (var i: u32 = 0u; i < lights.n_directional_lights; i = i + 1u) {
        // The light's radiance is premultiplied by its illuminance in the
        // Bevy uniform, so a weaker secondary light contributes less. Shadow
        // maps only gate this light's own visibility; a light without shadows
        // still illuminates.
        let light = &lights.directional_lights[i];
        let visibility = calc_rich_light_visibility(in, i);
        let shading = mtoon_shading_weight(
            dot(in.world_normal, (*light).direction_to_light),
            shade_shift,
            material.shading_toony_factor,
        ) * visibility;
        result.direct += mtoon_direct_term(
            in.lit_color.rgb,
            shade_color,
            shading,
            (*light).color.rgb,
        );
        // The shadow visibility also gates the added specular, so a highlight
        // cannot shine through the key light's shadow.
        result.specular += portrait_direct_specular(
            in.world_normal,
            in.world_view_dir,
            (*light).direction_to_light,
            (*light).color.rgb * visibility,
            roughness,
        );
        result.rim += portrait_rim(
            in.world_normal,
            in.world_view_dir,
            (*light).direction_to_light,
            (*light).color.rgb,
            params.rim_power,
        );
    }
    return result;
}

/// The shadow visibility of one directional light (1.0 when it casts none).
fn calc_rich_light_visibility(
    input: MToonInput,
    light_id: u32,
) -> f32 {
    let light = &lights.directional_lights[light_id];
    if ((*light).flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) == 0u {
        return 1.0;
    }
    let view_z = dot(vec4<f32>(
        view.view_from_world[0].z,
        view.view_from_world[1].z,
        view.view_from_world[2].z,
        view.view_from_world[3].z
    ), input.world_position);
    return fetch_directional_shadow(
        light_id,
        input.world_position,
        input.world_normal,
        view_z,
        input.pbr.frag_coord.xy,
    );
}

// The Rich global illumination: the same ambient read as the Native path with
// the specification's two-point `giEqualizationFactor` approximation.
fn apply_rich_global_illumination(in: MToonInput) -> vec3<f32> {
    let diffuse_color = calc_diffuse_color(
        in.lit_color.rgb,
        in.pbr.material.diffuse_transmission,
    );
    let passthrough = rich_ambient(in, in.world_normal, diffuse_color);
    let uniformed = 0.5 * (rich_ambient(in, vec3<f32>(0.0, 1.0, 0.0), diffuse_color)
        + rich_ambient(in, vec3<f32>(0.0, -1.0, 0.0), diffuse_color));
    return mix(passthrough, uniformed, material.gi_equalization_factor);
}

fn rich_ambient(
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
