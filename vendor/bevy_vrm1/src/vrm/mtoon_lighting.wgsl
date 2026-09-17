#define_import_path mtoon::lighting

// MToon lighting primitives shared by the mesh and outline fragment paths.
//
// These are pure functions: they do not collect lights, sample shadows or
// touch textures. Light collection and shadow sampling stay in the fragment
// shader so the same expression can be evaluated for one light at a time and
// multiplied by that light's linear radiance.

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

// The MToon direct-light term for a single light: the base/shade color
// interpolation multiplied by that light's linear radiance.
fn mtoon_direct_term(
    lit: vec3<f32>,
    shade: vec3<f32>,
    shading: f32,
    light_rgb: vec3<f32>,
) -> vec3<f32> {
    return mix(shade, lit, shading) * light_rgb;
}
