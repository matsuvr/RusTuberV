#define_import_path mtoon::portrait

// The Rich display's extra material terms: a modest non-metal specular from
// each light, a view-reflected environment term, and a lighting-side rim.
//
// Everything here is an artistic layer on top of the authored MToon result:
// it is evaluated in linear space, adds no PBR diffuse, and never changes the
// material's alpha. The composition with the Native result is
// `compose_rich_mtoon`, the single place where `strength` is applied, so the
// CPU gains and the shader gains are not multiplied twice.

#import bevy_pbr::{
    lighting::{
        D_GGX,
        F_AB,
        F_Schlick_vec,
        V_SmithGGXCorrelated,
    },
    mesh_view_bindings::light_probes,
}
// The view environment cubemaps are only bound while the view has an
// environment map (`ENVIRONMENT_MAP`), so the sampling below is guarded the
// same way and returns zero without one. The whole-module import keeps the
// conditional bindings reachable without importing names that a define set
// without `ENVIRONMENT_MAP` does not export.
#import bevy_pbr::mesh_view_bindings as mtoon_env_bindings

// The added direct-light specular for one light.
//
// `light_rgb` already carries the light's radiance and, for shadow-casting
// lights, the shadow visibility, so a highlight cannot shine through a shadow.
// The material is treated as a non-metal dielectric.
fn portrait_direct_specular(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    light_rgb: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let n_dot_l = saturate(dot(n, l));
    let n_dot_v = saturate(dot(n, v));
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(l + v);
    let f0 = vec3<f32>(0.04);
    let d = D_GGX(roughness, saturate(dot(n, h)));
    let visibility = V_SmithGGXCorrelated(roughness, n_dot_v, n_dot_l);
    let f = F_Schlick_vec(f0, 1.0, saturate(dot(l, h)));
    return d * visibility * f * light_rgb * n_dot_l;
}

// The added environment reflection, using the view's specular cubemap and its
// roughness mips. The environment diffuse is intentionally not added: the
// standard MToon GI already carries it.
//
// Bevy's environment prefilter keys each output mip by a *perceptual*
// roughness (`environment_filter.wgsl` treats `constants.roughness` as
// perceptual and converts internally), and Bevy's own sampler selects the mip
// as `perceptual_roughness * max_mip`, so this term receives and forwards the
// perceptual roughness unchanged. The environment BRDF (`F_AB`) has the same
// contract.
//
// `world_position` is part of the term's contract, but a view environment
// probe does not depend on the sample position.
fn portrait_environment_specular(
    n: vec3<f32>,
    v: vec3<f32>,
    perceptual_roughness: f32,
    world_position: vec3<f32>,
) -> vec3<f32> {
    // The sample position does not affect a view environment probe, but it is
    // part of the term's contract.
    let probe_index = light_probes.view_cubemap_index;
    if (probe_index < 0) {
        return vec3<f32>(0.0);
    }
    // The same mip index rule as Bevy's own environment sampler: the
    // perceptual roughness against the smallest (coarsest) specular mip.
    let mip = perceptual_roughness * f32(light_probes.smallest_specular_mip_level_for_view);

    // Rotate the reflection by the probe rotation; cube maps are left-handed.
    let reflection = reflect(-v, n);
    let rotated = portrait_quat_rotate(light_probes.view_rotation, reflection);
    var direction = rotated;
    direction.z = -direction.z;
    let radiance = portrait_environment_sample(probe_index, direction, mip);

    // Split-sum approximation: the prefiltered radiance times the environment
    // BRDF for a non-metal dielectric. `F_AB` takes the perceptual roughness,
    // not the squared one. The environment diffuse is intentionally not
    // added; the standard MToon GI already carries it.
    let f_ab = F_AB(perceptual_roughness, saturate(dot(n, v)));
    let fss_ess = vec3<f32>(0.04) * f_ab.x + f_ab.y;
    return radiance * fss_ess * light_probes.intensity_for_view;
}

fn portrait_environment_sample(probe_index: i32, direction: vec3<f32>, mip: f32) -> vec3<f32> {
#ifdef ENVIRONMENT_MAP
#ifdef MULTIPLE_LIGHT_PROBES_IN_ARRAY
    return textureSampleLevel(
        mtoon_env_bindings::specular_environment_maps[probe_index],
        mtoon_env_bindings::environment_map_sampler,
        direction,
        mip,
    ).rgb;
#else
    return textureSampleLevel(
        mtoon_env_bindings::specular_environment_map,
        mtoon_env_bindings::environment_map_sampler,
        direction,
        mip,
    ).rgb;
#endif
#else
    return vec3<f32>(0.0);
#endif
}

fn portrait_quat_rotate(q: vec4<f32>, direction: vec3<f32>) -> vec3<f32> {
    return direction + 2.0 * cross(q.xyz, cross(q.xyz, direction) + q.w * direction);
}

// The added rim, tied to the light direction and color rather than to the view
// Fresnel alone, so it shades with the light instead of forcing a white edge.
fn portrait_rim(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    light_rgb: vec3<f32>,
    power: f32,
) -> vec3<f32> {
    let rim = pow(saturate(1.0 - saturate(dot(n, v))), max(power, 0.0001));
    return vec3<f32>(rim * saturate(dot(n, l))) * light_rgb;
}

// Composes the Rich result from the fixed Native result.
//
// The composition is the identity at `strength == 0`, so a Rich material with
// no added effect renders exactly the Native display. At 1 it is the Rich RGB
// with the Native alpha; in between the RGB is interpolated and the alpha is
// always the Native alpha, so coverage, cutout and depth never move with the
// look. This function is pure: it only reads its arguments.
fn compose_rich_mtoon(
    native: vec4<f32>,
    rich_rgb: vec3<f32>,
    strength: f32,
) -> vec4<f32> {
    if (strength <= 0.0) {
        return native;
    }
    if (strength >= 1.0) {
        return vec4<f32>(rich_rgb, native.a);
    }
    return vec4<f32>(mix(native.rgb, rich_rgb, strength), native.a);
}

// The Face-role diffuse lighting normal: the mesh normal at `amount == 0` and,
// above zero, slightly steered toward the head's world forward so the face's
// shading reads as one lit plane without flattening the whole head.
//
// The maximum blend of 0.25 at amount 1 is an art-tuning starting point, not a
// guaranteed optimum. This function is pure: it only reads its arguments and
// mirrors `vtuber_avatar`'s `face_lighting_normal` exactly.
fn portrait_face_normal(
    mesh_normal: vec3<f32>,
    head_forward: vec3<f32>,
    amount: f32,
) -> vec3<f32> {
    if (amount <= 0.0) {
        return mesh_normal;
    }
    return normalize(mix(mesh_normal, head_forward, 0.25 * amount));
}
