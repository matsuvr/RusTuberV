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
    // Share cell lookup and offsets across RGB, but retain the original
    // coordinate addition order: reassociating phase arithmetic can change
    // the selected view near a lens boundary through f32 rounding.
    let pitch = calibration.optics.x;
    let tilt = calibration.optics.y;
    var offset_x = vec3<f32>(0.0, calibration.optics.w, 2.0 * calibration.optics.w);
    var offset_y = vec3<f32>(0.0);
    if calibration.flags.x != 0u {
        // Cell-pattern lookup and storage-buffer reads happen once per pixel,
        // rather than once for each color channel.
        let cell = cell_at(vec2<u32>(floor(screen_uv * calibration.screen.xy)));
        let rg = cells[2u * cell];
        let b = cells[2u * cell + 1u];
        offset_x = vec3<f32>(rg.x, rg.z, b.x);
        offset_y = vec3<f32>(rg.y, rg.w, b.y);
    }
    if (flags & 4u) != 0u {
        offset_x = offset_x.zyx;
        offset_y = offset_y.zyx;
    }
    let position_x = vec3<f32>(screen_uv.x) + offset_x;
    let position_y = vec3<f32>(screen_uv.y) + offset_y;
    var phases = fract((position_x + position_y * tilt) * pitch - vec3<f32>(calibration.optics.z));
    if calibration.flags.z != 0u { phases = vec3<f32>(1.0) - phases; }
    let views = min(vec3<u32>(floor(phases * f32(count))), vec3<u32>(count - 1u));
    let red = textureSampleLevel(quilt, quilt_sampler, quilt_uv(views.x, image_uv), 0.0).r;
    let green = textureSampleLevel(quilt, quilt_sampler, quilt_uv(views.y, image_uv), 0.0).g;
    let blue = textureSampleLevel(quilt, quilt_sampler, quilt_uv(views.z, image_uv), 0.0).b;
    return vec4<f32>(red, green, blue, 1.0);
}
