// Avatar-only preview conversion.
//
// The shared Bgra8UnormSrgb image stores gamma-premultiplied RGB:
//     stored = a * E(C)
// An sRGB texture read returns S = D(stored). Convert each texel before the
// manual bilinear filter so transparent edges do not interpolate the encoded
// association. The egui target then receives linear-premultiplied RGB.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var avatar_texture: texture_2d<f32>;
@group(0) @binding(1) var avatar_sampler: sampler;

struct AvatarPreviewUniform {
    corner_radius: f32,
    width: f32,
    height: f32,
    unused: f32,
}

@group(0) @binding(2) var<uniform> avatar_preview_uniform: AvatarPreviewUniform;

const POSITIONS = array(
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(3.0, 1.0),
    vec2<f32>(-1.0, -3.0),
);

const UVS = array(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(2.0, 0.0),
    vec2<f32>(0.0, 2.0),
);

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    return VertexOutput(
        vec4<f32>(POSITIONS[vertex_index], 0.0, 1.0),
        UVS[vertex_index],
    );
}

fn srgb_encode(rgb: vec3<f32>) -> vec3<f32> {
    let cutoff = rgb <= vec3<f32>(0.0031308);
    let lower = rgb * vec3<f32>(12.92);
    let higher = vec3<f32>(1.055) * pow(rgb, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(higher, lower, cutoff);
}

fn srgb_decode(rgb: vec3<f32>) -> vec3<f32> {
    let cutoff = rgb <= vec3<f32>(0.04045);
    let lower = rgb / vec3<f32>(12.92);
    let higher = pow((rgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

// One sRGB texel: S is the RGB returned by the sRGB texture read.
// For a > 0 the required linear-premultiplied RGB is a * D(E(S) / a).
fn gamma_premultiplied_to_linear_premultiplied(sample: vec4<f32>) -> vec4<f32> {
    let alpha = sample.a;
    if (alpha == 0.0) {
        return vec4<f32>(0.0);
    }
    let gamma_straight = srgb_encode(sample.rgb) / alpha;
    return vec4<f32>(srgb_decode(gamma_straight) * alpha, alpha);
}

fn sample_nearest_texel(index: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
    let clamped_index = clamp(index, vec2<i32>(0), size - vec2<i32>(1));
    let uv = (vec2<f32>(clamped_index) + vec2<f32>(0.5)) / vec2<f32>(size);
    return textureSampleLevel(avatar_texture, avatar_sampler, uv, 0.0);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let display_size = vec2<f32>(
        avatar_preview_uniform.width,
        avatar_preview_uniform.height,
    );
    let radius = min(
        avatar_preview_uniform.corner_radius,
        min(display_size.x, display_size.y) * 0.5,
    );
    let local = in.uv * display_size;
    let nearest = clamp(local, vec2<f32>(radius), display_size - vec2<f32>(radius));
    if (distance(local, nearest) > radius) {
        discard;
    }

    let size = vec2<i32>(textureDimensions(avatar_texture));
    let texel_position = in.uv * vec2<f32>(size) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(texel_position));
    let fraction = fract(texel_position);

    // Decode/associate every texel first, then filter those linear values.
    let top_left = gamma_premultiplied_to_linear_premultiplied(
        sample_nearest_texel(base, size)
    );
    let top_right = gamma_premultiplied_to_linear_premultiplied(
        sample_nearest_texel(base + vec2<i32>(1, 0), size)
    );
    let bottom_left = gamma_premultiplied_to_linear_premultiplied(
        sample_nearest_texel(base + vec2<i32>(0, 1), size)
    );
    let bottom_right = gamma_premultiplied_to_linear_premultiplied(
        sample_nearest_texel(base + vec2<i32>(1, 1), size)
    );
    return mix(
        mix(top_left, top_right, fraction.x),
        mix(bottom_left, bottom_right, fraction.x),
        fraction.y,
    );
}
