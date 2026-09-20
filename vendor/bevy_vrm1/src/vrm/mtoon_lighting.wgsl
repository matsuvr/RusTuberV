#define_import_path mtoon::lighting

// The Rich display's lighting primitives.
//
// These are used only by `mtoon_rich_fragment.wgsl`: the Native display owns
// its own expressions in `mtoon_native.wgsl`. They are pure functions that do
// not collect lights or sample shadows; the Rich fragment shader does that so
// one light can be evaluated at a time with its own linear radiance.

// `ndotl` is the signed dot product of the surface normal and the light
// direction (-1..=1), `shift` is `shadingShiftFactor` plus the
// `shadingShiftTexture` contribution, and `toony` is `shadingToonyFactor`.
//
// The specification collapses the ramp to a zero-width step when `toony` is 1.
// That endpoint is handled explicitly so the division by the ramp width is
// never taken with a zero width; it is the endpoint of the same formula, not
// an input correction.
fn mtoon_shading_weight(ndotl: f32, shift: f32, toony: f32) -> f32 {
    let shifted = ndotl + shift;
    if toony >= 1.0 {
        return select(0.0, 1.0, shifted >= 0.0);
    }
    return saturate((shifted - (-1.0 + toony)) / ((1.0 - toony) - (-1.0 + toony)));
}

// The Rich direct-light term for a single light: the base/shade color
// interpolation multiplied by that light's linear radiance.
fn mtoon_direct_term(
    lit: vec3<f32>,
    shade: vec3<f32>,
    shading: f32,
    light_rgb: vec3<f32>,
) -> vec3<f32> {
    return mix(shade, lit, shading) * light_rgb;
}
