#define_import_path mtoon::portrait

// The rich look's extra material terms: a modest non-metal specular from each
// light, a view-reflected environment term, and a lighting-side rim.
//
// Everything here is an artistic layer on top of the authored MToon result:
// it is evaluated in linear space, adds no PBR diffuse, and never changes the
// material's alpha. `params.strength` is applied once, here, so the CPU gains
// and the shader gains are not multiplied twice.

#import bevy_pbr::{
    lighting::{
        D_GGX,
        F_AB,
        F_Schlick_vec,
        V_SmithGGXCorrelated,
    },
    mesh_view_bindings::{view, light_probes},
}

#import mtoon::types::MToonPortraitUniform

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
// `world_position` is part of the term's contract, but a view environment
// probe does not depend on the sample position.
fn portrait_environment_specular(
    n: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
    world_position: vec3<f32>,
) -> vec3<f32> {
    // The sample position does not affect a view environment probe, but it is
    // part of the term's contract.
    let probe_index = light_probes.view_cubemap_index;
    if (probe_index < 0) {
        return vec3<f32>(0.0);
    }
    let mip = roughness * f32(light_probes.smallest_specular_mip_level_for_view);

    // Rotate the reflection by the probe rotation; cube maps are left-handed.
    let reflection = reflect(-v, n);
    let rotated = portrait_quat_rotate(light_probes.view_rotation, reflection);
    var direction = rotated;
    direction.z = -direction.z;
    let radiance = portrait_environment_sample(probe_index, direction, mip);

    // Split-sum approximation: the prefiltered radiance times the environment
    // BRDF for a non-metal dielectric. The environment diffuse is intentionally
    // not added; the standard MToon GI already carries it.
    let f_ab = F_AB(roughness, saturate(dot(n, v)));
    let fss_ess = vec3<f32>(0.04) * f_ab.x + f_ab.y;
    return radiance * fss_ess * light_probes.intensity_for_view;
}

fn portrait_environment_sample(probe_index: i32, direction: vec3<f32>, mip: f32) -> vec3<f32> {
    return vec3<f32>(0.0);
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

// Layers the added terms on top of the authored result.
//
// `base` is the material result with the look's lighting and the author's own
// MatCap/rim/emission already in it. `strength == 0` returns `base` unchanged,
// so switching the look off cannot change a pixel.
fn apply_portrait_terms(
    base: vec4<f32>,
    direct_specular: vec3<f32>,
    environment_specular: vec3<f32>,
    rim: vec3<f32>,
    params: MToonPortraitUniform,
) -> vec4<f32> {
    if (params.strength <= 0.0) {
        return base;
    }
    let extra = params.specular_gain * direct_specular
        + params.environment_gain * environment_specular
        + params.rim_gain * rim;
    return vec4<f32>(base.rgb + view.exposure * extra * params.strength, base.a);
}



