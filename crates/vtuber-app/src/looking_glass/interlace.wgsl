#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct Calibration {
    optics: vec4<f32>, // normalized pitch, tilt, center, RGB-stripe increment
    screen: vec4<f32>,
    grid: vec4<u32>,   // columns, rows, tile width, tile height
    flags: vec4<u32>,  // cell count, pattern, inverted views, flips | quilt preview
}
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> calibration: Calibration;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var quilt: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var quilt_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<storage, read> cells: array<vec4<f32>>;

fn cell_at(pixel: vec2<u32>) -> u32 {
    var cell = 0u;
    switch calibration.flags.y {
        case 1u: { cell = (pixel.x + pixel.y) % 2u; }
        case 2u: { cell = pixel.x % 2u; }
        case 3u: { cell = (pixel.y + 2u * (pixel.x % 2u)) % 4u; }
        case 4u: { cell = pixel.y % 2u; }
        default: {}
    }
    return cell % calibration.flags.x;
}

fn quilt_uv(view: u32, local_uv: vec2<f32>) -> vec2<f32> {
    let grid = calibration.grid;
    let tile = vec2<u32>(view % grid.x, grid.y - 1u - view / grid.x);
    let half_texel = 0.5 / vec2<f32>(grid.zw);
    // Bilinear taps must stay inside this view, including along quilt borders.
    return (vec2<f32>(tile) + clamp(local_uv, half_texel, vec2<f32>(1.0) - half_texel)) / vec2<f32>(grid.xy);
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let flags = calibration.flags.w;
    if (flags & 8u) != 0u {
        return textureSampleLevel(quilt, quilt_sampler, mesh.uv, 0.0);
    }
    // Lens calibration uses bottom-left screen coordinates; Bevy image UVs
    // use top-left coordinates. Keep optical phase separate from image flips.
    let screen_uv = vec2<f32>(mesh.uv.x, 1.0 - mesh.uv.y);
    var image_uv = mesh.uv;
    if (flags & 1u) != 0u { image_uv.x = 1.0 - image_uv.x; }
    if (flags & 2u) != 0u { image_uv.y = 1.0 - image_uv.y; }
    let count = calibration.grid.x * calibration.grid.y;
    var result = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    for (var channel = 0u; channel < 3u; channel += 1u) {
        var subpixel = channel;
        if (flags & 4u) != 0u { subpixel = 2u - channel; }
        var offset = vec2<f32>(f32(subpixel) * calibration.optics.w, 0.0);
        if calibration.flags.x != 0u {
            let cell = cell_at(vec2<u32>(floor(screen_uv * calibration.screen.xy)));
            let rg = cells[2u * cell];
            switch subpixel {
                case 0u: { offset = rg.xy; }
                case 1u: { offset = rg.zw; }
                default: { offset = cells[2u * cell + 1u].xy; }
            }
        }
        let position = screen_uv + offset;
        var phase = fract((position.x + position.y * calibration.optics.y) * calibration.optics.x - calibration.optics.z);
        if calibration.flags.z != 0u { phase = 1.0 - phase; }
        let view = min(u32(floor(phase * f32(count))), count - 1u);
        let color = textureSampleLevel(quilt, quilt_sampler, quilt_uv(view, image_uv), 0.0);
        result[channel] = color[channel];
    }
    return result;
}
