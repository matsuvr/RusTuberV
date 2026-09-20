// The alpha-aware portrait finish for the avatar output view (Issue #74).
//
// The main pass leaves premultiplied linear HDR color and coverage in the
// view's main texture. This pass carries the preset's display transform to
// that content while keeping the alpha exactly as it arrived, so the final
// BGRA8 sRGB image stays a premultiplied image the preview can sample and the
// readback can unpremultiply exactly once.
//
// The camera exposure is not applied here: the look writes the resolved
// `Exposure` onto the camera, so the material shaders scale their light by it
// once, exactly as the standard display does.

#import bevy_core_pipeline::tonemapping::{
    ACESFitted,
    applyAgXLog,
    applyLUT3D,
    sample_blender_filmic_lut,
    sample_tony_mc_mapface_lut,
    tonemapping_pbr_neutral,
    tonemapping_reinhard,
    tonemapping_reinhard_luminance,
    somewhat_boring_display_transform,
}
#import bevy_core_pipeline::tonemapping_lut_bindings::{
    dt_lut_sampler,
    dt_lut_texture,
}
#import bevy_render::color_operations::{
    linear_to_srgb,
    srgb_to_linear,
}
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var finish_source: texture_2d<f32>;
@group(0) @binding(1) var finish_source_sampler: sampler;
// Bindings 2 and 3 are the shared `tonemapping_lut_bindings`, reindexed by the
// pipeline's `TONEMAPPING_LUT_*` definitions so the display transform reads
// the same LUT data as the standard tonemapping pass.
@group(0) @binding(4) var<uniform> finish: FinishUniform;

struct FinishUniform {
    // The resolved display transform, as the `Tonemapping` discriminant.
    tonemapping: u32,
    unused_1: f32,
    unused_2: f32,
    unused_3: f32,
}

// The finish for unassociated (straight) linear RGB.
//
// Bevy's display transforms, routed by the resolved curve. This is the one
// display transform in the rich path: the output camera's own `Tonemapping`
// is `None` while the finish pass runs.
fn finish_straight_linear_rgb(rgb: vec3<f32>, params: FinishUniform) -> vec3<f32> {
    switch params.tonemapping {
        case 1u: {
            return tonemapping_reinhard(rgb);
        }
        case 2u: {
            return tonemapping_reinhard_luminance(rgb);
        }
        case 3u: {
            return ACESFitted(rgb);
        }
        case 4u: {
            return applyLUT3D(applyAgXLog(rgb), 32.0);
        }
        case 5u: {
            return somewhat_boring_display_transform(rgb);
        }
        case 6u: {
            return sample_tony_mc_mapface_lut(rgb);
        }
        case 7u: {
            return sample_blender_filmic_lut(rgb);
        }
        case 8u: {
            return tonemapping_pbr_neutral(rgb);
        }
        default: {
            // Tonemapping::None: no display transform.
            return rgb;
        }
    }
}

// The finish for the main pass's premultiplied color at the final sRGB
// boundary.
//
// Non-linear tone mapping needs the unassociated color, so `a > 0` unprepares
// `C = Cp / a`, applies the finish, and re-associates with the same coverage.
// The linear internal result is `a * T(C)`, but the final image contract is
// `a * E(T(C))`. The finish pass still stores a linear value in its
// `Rgba16Float` post-process texture, so stage `D(a * E(T(C)))` here. Bevy's
// upscaling blit then writes that value to `Bgra8UnormSrgb`, whose attachment
// encode supplies the one `E` at the image boundary. Alpha never changes.
fn finish_premultiplied_linear(rgba: vec4<f32>, params: FinishUniform) -> vec4<f32> {
    let coverage = rgba.a;
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);
    }
    let color = finish_straight_linear_rgb(rgba.rgb / coverage, params);
    // The same linear association is retained conceptually as `a * T(C)`;
    // encode that associated value at the final image boundary instead of
    // encoding `a * T(C)` as one non-linear operation.
    let premultiplied_srgb = linear_to_srgb(color) * coverage;
    let staged_linear = srgb_to_linear(premultiplied_srgb);
    return vec4<f32>(staged_linear, coverage);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let source = textureSample(finish_source, finish_source_sampler, in.uv);
    return finish_premultiplied_linear(source, finish);
}
