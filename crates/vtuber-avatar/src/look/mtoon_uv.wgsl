#define_import_path rustuberv::mtoon_uv

// Shared arithmetic; sampling and time masking stay in each shader stage.
fn animate_uv(uv: vec2<f32>, time: f32, scroll_speed: vec2<f32>, rotation_speed: f32) -> vec2<f32> {
    let translate = time * scroll_speed;
    let rotate_rad = fract(time * rotation_speed);
    let cos_rotate = cos(rotate_rad);
    let sin_rotate = sin(rotate_rad);
    let pivot = vec2<f32>(0.5, 0.5);
    return mat2x2(cos_rotate, -sin_rotate, sin_rotate, cos_rotate) * (uv - pivot) + pivot + translate;
}
