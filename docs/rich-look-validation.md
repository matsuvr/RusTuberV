# Rich look validation (Issues #68-#79)

This is the measurement record for the rich-look epic. It covers only what was
actually implemented and measured; everything else is listed as not
implemented and not measured, and is not claimed as working.

## Environment

- OS: Windows 11 Pro (kernel 26200)
- GPU: NVIDIA GeForce RTX 4090, driver 610.47
- Backend: Vulkan
- Renderer: `cargo xtask rich-look <case>`, which renders a synthetic scene
  through the production offscreen camera/readback path (`AVATAR_RENDER_LAYER`)
  and inspects CPU pixels. It exits 2 (`NOT RUN`) when no GPU/readback is
  available instead of reporting success. The `finish-alpha` case exercises
  the real finish shader, `Bgra8UnormSrgb` target write and
  `VideoOutputFrame::from_padded_bgra8`; `avatar-ui-alpha` exercises the
  avatar-only `EguiBevyPaintCallback` against a linear `Rgba16Float` target;
  `vrm-render` exercises the same output/readback path with imported VRM
  assets.
- macOS and other GPUs were not available and are not measured.

## Implemented and committed

| Issue | Commit | Scope |
|---|---|---|
| #69 standard MToon | `57977e2` | per-light direct term, signed NdotL/shift/toony endpoints, single exposure application, spec GI equalization, glTF/legacy normal texture at bindings 117/118 |
| #70 look boundary | `ea47b59` | `RichLookSettings`/`effective_look_strength`/`blend_look_scalar`, `StandardLookBase` capture, `AvatarLookSettings`+`LookSettingsChanged`, capture-once/unload lifecycle |
| #71 studio lighting | `83c5344` | `StudioPreset`/`StudioLight`/`StudioRig` solve+blend, key takeover + fill/rim, ambient/environment sync, restore on strength 0, MToon cutout prepass, generated studio cubemap |
| #73 Standard portrait | `a0d64f6` | `resolve_standard_portrait`/`apply_standard_portrait_settings` (roughness-only relative adjustment, unlit and MToon untouched) |
| #74 HDR finish | this PR | `PortraitFinish`/`PORTRAIT_FINISH`/`resolve_portrait_finish`, `sync_portrait_finish` capture-once/restore, `Hdr`+`Exposure`+`Tonemapping` camera components, and the alpha-aware finish pass (`finish.wgsl`: `finish_straight_linear_rgb`/`finish_premultiplied_linear`) with the explicit final contract `a * E(T(C))` |
| #79 avatar UI alpha boundary | this PR | avatar-only callback for the monitor and expanding transition; the shared `Bgra8UnormSrgb` image remains gamma-premultiplied while the callback supplies linear-premultiplied RGB to the UI blend |

Supporting reusable code: `tools/xtask/src/rich_look.rs` (the GPU cases,
including `finish-alpha` and `avatar-ui-alpha`),
`crates/vtuber-app/src/ui/avatar_preview.rs`/`avatar_preview.wgsl`,
`vendor/bevy_vrm1/src/vrm/mtoon_lighting.wgsl`, `mtoon_alpha.wgsl`,
`mtoon_prepass.wgsl`.

## Measured on the GPU (Windows/Vulkan/RTX 4090)

`mtoon-lighting` (64x64 BGRA readback, center pixel):

| case | pixel |
|---|---|
| zero illuminance | `[0, 0, 0, 255]` |
| 200 lx white | `[124, 124, 124, 255]` |
| 400 lx white | `[170, 170, 170, 255]` |
| 200 lx red | `[0, 0, 124, 255]` |
| 200 lx blue | `[124, 0, 0, 255]` |
| 2 x 100 lx white | `[124, 124, 124, 255]` |
| 100 lx white | `[89, 89, 89, 255]` |

The direct term follows each light's own radiance, light color is per light,
and two lights add. The pre-fix renderer (issue #69) multiplied no light
radiance at all, so it could not produce this table.

`mtoon-standard`: a fully lit white plane at the app's 650 lx key reads
`[211, 211, 211, 255]` (225 with the app's ambient), below white; at 1300 lx it
reaches `[255, 255, 255, 255]`, and a red light keeps the red channel. The
standard display follows the scene light's level and color and leaves headroom
instead of clipping. The standard path applies each light's radiance and the
camera exposure once, exactly as the rich path does; the difference between off
and on is the portrait extras and the studio rig, not the brightness formula.

The light level is the app's own `setup_scene` key (650 lx): a fully lit white
surface exposes to about 0.73 including the default ambient, so a material's
own rim or emission has room before white. Bevy applies `Tonemapping` only to
HDR views, so a real highlight roll-off needs the HDR output of issue #74; at
the current LDR output the headroom is what keeps the display off white.

## #74 measured: HDR finish, alpha and tone (Windows/Vulkan/RTX 4090)

`look::finish` resolves one preset (`PORTRAIT_FINISH`: HDR on, the app's
default `Exposure::BLENDER` ev100, Bevy's default `TonyMcMapface` display
transform) and `sync_portrait_finish` writes the resolved `Hdr`/`Exposure`/
`Tonemapping` onto both avatar cameras, marks the output view for the
alpha-aware finish pass and routes the output camera's own `Tonemapping`
through it (`None` on the component, the curve travels in the pass uniform) so
the display transform is applied exactly once. The window keeps Bevy's own
tonemapping pass, so the viewport and the output image share the same
exposure and curve.

Per-model `vrm-render` numbers with the finish in place (same fixture, same
frozen clock):

| model | off mean | on mean | mean abs diff |
|---|---|---|---|
| AvatarSample_C | 33.56 | 38.56 | 5.09 |
| IrisPart1(ShapeKey Reduce) | 31.21 | 33.49 | 3.60 |
| RearAliceLite_3.0 | 24.75 | 25.95 | 2.26 |
| Sapphy | 35.09 | 34.40 | 2.24 |

The ON state is now *dimmer on highlights than the pre-#74 look* while the
portrait rig still adds light: AvatarSample_C's bright jacket panel reads
`[245, 237, 236]` in the pre-#74 ON frame and `[193, 188, 188]` with the
finish, and the face keeps its tone, so the added light no longer clips to
white.

Alpha contract checks (byte comparisons of the readback frames):

| comparison | result |
|---|---|
| look off, this build vs origin/main (`648d981`) | byte-identical for AvatarSample_C, IrisPart1(ShapeKey Reduce), RearAliceLite_3.0 and Sapphy |
| look on vs off, alpha channel | AvatarSample_C and RearAliceLite_3.0: 0 differing pixels; Sapphy 112 and Iris 62 pixels differ by ±1/255 at MSAA-resolved semi-transparent edges (the pre-#74 build showed the same kind of 13/11-pixel differences at the same edges) |

The ±1 edge differences come from the HDR path quantizing the MSAA-resolved
premultiplied content through `Rgba16Float` before the final 8-bit write;
they are not a tone-dependent alpha change, and every consumer still receives
the same premultiplied BGRA8 sRGB image the preview samples and the readback
unpremultiplies exactly once.

### Final image color/alpha boundary

For a main-pass premultiplied linear pixel `Cp` with coverage `a`, the finish
pass first recovers `C = Cp / a` when `a > 0`, applies the tone curve `T(C)`,
and retains the linear association `a * T(C)`. The output target is a linear
`Rgba16Float` post-process texture until Bevy's upscaling blit writes the final
`Bgra8UnormSrgb` image. Therefore the finish shader stages
`D(a * E(T(C)))` (`E` is sRGB encode and `D = E^-1`) and the target attachment
performs the one final `E`, producing stored RGB bytes of exactly
`a * E(T(C))`.

This is deliberately not `E(a * T(C))`: the latter is the old gamma/alpha
ordering bug. Alpha is passed through unchanged, with an all-zero pixel for
`a = 0`. `VideoOutputFrame::from_padded_bgra8` then performs the existing
single byte-domain unpremultiply and exposes transport-neutral straight BGRA8
sRGB to NDI and other consumers. This readback result is the frame contract;
it is distinct from the UI blend input.

The preview and readback still share the same `AvatarOutputTarget` image. The
image is created as `Bgra8UnormSrgb` at the fixed `VideoOutputProfile`
dimensions. The avatar monitor and expanding transition read that same image
through a small avatar-only `EguiBevyPaintCallback`; normal egui text and
unrelated images keep their existing shader. If the sRGB texture sample is
`S`, the callback converts each texel to linear-premultiplied RGB as
`a * D(E(S) / a)` for `a > 0`, and to zero for `a = 0`, before its
premultiplied blend into the linear UI target. Exposure and tone are not
reapplied. With linear filtering, the callback uses nearest texel sampling,
per-texel conversion, then manual bilinear filtering, so transparent edges do
not interpolate gamma-premultiplied bytes first. Existing egui clip rectangles,
opacity and the monitor/transition rectangles remain in the egui pass.
Preview visibility does not add a readback, and deactivating transport leaves
the preview camera active.

Command: `cargo run -p xtask -j 1 -- rich-look finish-alpha`.

`finish-alpha` is the end-to-end GPU fixture:
`finish.wgsl` -> sRGB target write -> padded GPU readback ->
`VideoOutputFrame::from_padded_bgra8`. On the local Windows/Vulkan/RTX 4090
run, the tone input was linear `[0.5, 0.2, 0.05]` with Reinhard tone mapping:

| alpha | straight readback BGRA | expected straight BGRA | wrong `E(a * T(C))` comparison |
|---:|---|---|---|
| 0.00 | `[0, 0, 0, 0]` | transparent | n/a |
| 0.25 | `[60, 112, 155, 64]` | `[62, 113, 156, 64]` | `[114, 230, 255, 64]` |
| 0.50 | `[62, 114, 157, 127]` | `[62, 113, 156, 128]` | `[85, 163, 227, 128]` |
| 1.00 | `[62, 113, 156, 255]` | `[62, 113, 156, 255]` | n/a |

The one-byte alpha difference at `0.50` and the small RGB differences are
quantization. The much larger distance from the legacy comparison is color
amplification/saturation from encoding the associated value, not quantization.
Compositing the readback over black, white and `[24, 96, 180]` also passed:
the fixture's actual/expected pairs were respectively `[15,28,39]`/`[16,28,39]`,
`[206,219,230]`/`[207,219,230]` and `[33,100,174]`/`[34,100,174]` at `a=0.25`,
and `[31,57,78]`/`[31,57,78]`, `[159,185,206]`/`[158,184,205]` and
`[43,105,169]`/`[43,105,168]` at `a=0.50`; all differences are within the
fixture's four-byte quantization bound.

| boundary/check | status |
|---|---|
| alpha 0/0.25/0.5/1 through GPU finish, target and readback | PASS |
| readback RGB as straight sRGB after `VideoOutputFrame` packing | PASS (`finish-alpha`) |
| UI GPU composition from gamma-premultiplied image to linear-premultiplied blend input | PASS (`avatar-ui-alpha`) |
| straight RGB remains the same between opaque and partial coverage | PASS, within quantization; no legacy amplification/saturation |
| black/white/colored-background composition | PASS |
| shared preview/readback image, dimensions and `Bgra8UnormSrgb` format | PASS (source/unit check) |
| same-process OFF/ON/OFF, ON/strength-0 restoration and no repeated tone | PASS (`vrm-render`) |
| real NDI send/receive | NOT RUN |

Fixture note: the first `vrm-render` run after a rebuild can capture the look
-off frame before every material upload has landed (a model with missing
jacket/hair or a blank frame). Reruns of the same model are stable and were
the values recorded above; the flake is in the fixture's single-readback
capture, not in the look systems.

Not measured in this synthetic case: frame times, GPU time, NDI output, or any
GPU/OS other than the one above. The screenshots taken while reviewing the
result are visual inspection only; they are not a pixel measurement. No FPS or
image-quality threshold is claimed.

### #79 measured: avatar UI GPU composition

Command: `cargo run -p xtask -j 1 -- rich-look avatar-ui-alpha`.

This is a separate GPU fixture from `finish-alpha`. It paints the shared image
through the production avatar callback, draws the actual egui background shapes,
and reads a linear `Rgba16Float` target. The tone-after RGB is
`[0.5, 0.2, 0.05]`; the stored source is `a * E(C)` and the tested alpha values
are `0/0.25/0.5/1`. Reference pixels use the same stored frame's straight sRGB
value, decode it, composite in linear premultiplied space, and encode once at
the end.

| alpha | black | white | `[24,96,180]` | UI gray `228` |
|---:|---|---|---|---|
| 0 | `[0,0,0]` | `[255,255,255]` | `[24,96,180]` | `[228,228,228]` |
| 0.25 | `[99,63,30]` | `[240,231,226]` | `[102,104,161]` | `[219,208,202]` |
| 0.5 | `[137,89,44]` | `[224,203,191]` | `[138,111,138]` | `[209,185,172]` |
| 1 | `[188,124,63]` | `[188,124,63]` | `[188,124,63]` | `[188,124,63]` |

The normal-size opaque/transparent boundary was `[188,124,63]` and
`[228,228,228]`, both expected. The reduced 16x16 boundary was
`[209,186,172]`, exactly the decode-before-filter reference; the legacy
filter-first path would have produced `[253,205,176]`. Result: PASS.

The evidence is intentionally split: `finish-alpha` is the readback RGB and
straight-sRGB frame check; `avatar-ui-alpha` is the UI GPU composition check;
real NDI send/receive remains unrun.

### Local validation for this update

- `cargo test --workspace -j 1`: PASS.
- `cargo clippy --workspace --all-targets -j 1 -- -D warnings`: PASS.
- Existing GPU fixtures: `finish-alpha`, `mtoon-lighting`, `mtoon-shading`,
  `mtoon-normal`, `mtoon-standard`, `mtoon-portrait`, `mtoon-shadow`,
  `mtoon-cutout-shadow`, and `studio-environment`: PASS.
- Added `avatar-ui-alpha`: PASS.
- `git diff --check`: PASS.
- Real NDI send/receive and macOS hardware validation: NOT RUN.

`mtoon-portrait`: `standard` `[244,244,244]`, `rich_without_extras`
`[244,244,244]` (zeroed gains leave the standard pixel), `rich`
`[253,253,253]` (the gains add), and the same rich scene with a rotated key
`[244,244,244]` (the added specular follows the light).

`mtoon-shading`: front-lit `124`, 90-degree side `89`, side with +0.5 shift
`124`, light behind `0`; the toony=0 ramp has 17 intermediate samples on the
mid scanline and the toony=1 endpoint has 0.

`mtoon-normal`: no map `118`, identity map `118`, tilted map `95`, tilted map
with scale 0 `118`.

`mtoon-shadow`: darkest ground pixel under an MToon sphere is `0` with shadow
maps on and `267` with them off, so the MToon mesh casts into the shadow map.

`mtoon-cutout-shadow`: with an alpha-masked MToon quad, the ground under the
opaque half is `1` and under the transparent half `256`; without shadows both
are `256`. Before the MToon prepass shader, both halves were shadowed (a solid
quad shadow), which is the artifact issue #71 asked to remove.

`studio-environment`: a PBR sphere lit only by the generated studio cubemap is
`[87, 82, 82, 255]` against `[1, 0, 1, 255]` without it, so the GPU prefilter
produced usable diffuse/specular maps from the bundled 16x16x6 cubemap.

Not measured in these synthetic cases: frame times, GPU time, NDI output, or
any GPU/OS other than the one above. Real VRM coverage is recorded separately
below.

## Representative VRM hardware check

On the same Windows/Vulkan/RTX 4090 device, the production managed lifecycle
was run with two fixtures that are present in this repository. `vrm-render` now
captures `OFF -> ON -> OFF -> ON -> strength 0`, and requires the ON image to
change while both restoration paths and the repeated ON image remain within
`0.5` mean absolute byte difference.

| model | ON diff | OFF restore diff | ON repeat diff | strength 0 diff | opaque pixels off/on |
|---|---:|---:|---:|---:|---:|
| `inore-vrm1.vrm` | 3.117 | 0.001 | 0.000 | 0.001 | 19848 / 19848 |
| `tsukuyomi-chan.vrm` | 1.638 | 0.004 | 0.003 | 0.004 | 10455 / 10455 |

Commands:

```text
cargo run -p xtask -- vrm-render tests/fixtures/vrm/inore-vrm1.vrm target/rich-look-validation/inore-vrm1
cargo run -p xtask -- vrm-render tests/fixtures/vrm/tsukuyomi-chan.vrm target/rich-look-validation/tsukuyomi-chan
```

Both commands passed. The generated OFF/ON PNG pairs were visually inspected
for the available transparent hair, silhouette edges and clothing areas; this
is visual hardware evidence, not a claim that every VRM material role was
exhaustively classified. NDI send/receive was not run.

## Per-model rich-look check

`cargo xtask vrm-render <vrm-or-dir> <out-dir>` loads every model through the
production managed lifecycle with a frozen clock, a fixed camera and the
production offscreen readback, captures five 256x256 states (OFF, ON, restored
OFF, repeated ON and strength 0), and writes each frame as `.bgra` and `.png`.

| model | off mean | on mean | mean abs diff |
|---|---|---|---|
| 1565994099520778586 | 63.52 | 71.90 | 8.38 |
| AvatarSample_C | 33.56 | 39.65 | 6.08 |
| IrisPart1 | 31.26 | 35.62 | 4.36 |
| IrisPart1(ShapeKey Reduce) | 31.21 | 35.56 | 4.36 |
| IrisPart1(ShapeKey Reduce2) | 31.21 | 35.56 | 4.36 |
| RearAlice_3.0 | 24.73 | 28.34 | 3.63 |
| RearAliceLite_3.0 | 24.75 | 28.36 | 3.63 |
| Sapphy | 35.09 | 38.41 | 3.32 |
| SapphyPerfectSync | 35.09 | 38.41 | 3.32 |
| つくよみちゃん（タイプA・マテリアル数18） | 32.53 | 36.35 | 3.82 |
Every model keeps the same opaque pixel count in both states (the geometry and
alpha are untouched) and every model differs when the look is switched on, so
the switch has a measurable effect on all of them.

How much the difference reads as "rich" is subjective. What this records
mechanically is: off is the standard display, on is a different image, and the
difference is the studio rig plus the added MToon gloss, environment reflection
and rim of issue #72. `mtoon-portrait` additionally asserts that the added
specular follows the key light and that zeroing the gains removes the extra
terms.


Not measured: a byte comparison against a binary built from the pre-epic
commit. The equivalence above is established by the source mapping table and
the standard-path invariants, not by running the old build.



The "Enhanced look" section with the ON/OFF switch and the 0-100%
brightness/effect-strength slider is on the **Studio** pane and the settings
pane, in all four languages, wired through
`UiAction::SetRichLookEnabled`/`SetRichLookStrength` to `AvatarLookSettings`
and `LookSettingsChanged`. With the switch off, no look system writes
anything: the studio rig restores the scene's own key light, fill/rim stay at
zero, the ambient and environment return to their original values, and
Standard materials return to their captured values, so the plain
MToon/Standard/Unlit display is what renders.

Not done from #75: persistence to `settings.toml`, per-model settings, and
restore on model switch. The switch therefore starts OFF again after a
restart.

## Not implemented

| Issue | State |
|---|---|
| #72 MToon glossy/environment/extra rim | Implemented. `MToonPortraitParams` on the material, `resolve_mtoon_portrait`, `apply_mtoon_portrait_settings`, `mtoon_portrait.wgsl` and the view environment specular are in place; verified by `mtoon-portrait` and the per-model table above. |
| #74 HDR finish and transparency | Implemented. `PortraitFinish`, `resolve_portrait_finish`, `sync_portrait_finish`, `finish_straight_linear_rgb`/`finish_premultiplied_linear` and the alpha/color-space contract table exist in `crates/vtuber-avatar/src/look/finish.rs` and `finish.wgsl`; verified by the #74 measurements above. |
| #75 one-click UI, 4 languages, per-model save | Partially implemented: the switch and the strength slider exist in the settings screen in four languages (see above). Persistence, per-model settings and restore on model switch are not implemented. |
| #76 material roles | Not implemented. `MaterialRole`, `infer_material_role`, `resolve_material_role`, the role param resolvers and `face_lighting_normal` do not exist. |
| #77 final acceptance | Partially covered by this document (the measurements above). The #75 first-version and #76 role comparisons are not possible yet because those issues are not implemented. |

Consequences: the epic's completion criteria are not fully met yet. The
finish/tone change (#74) is in place and the look can be switched on and
adjusted from the settings screen; per-model look settings are not saved or
restored, and the role-based adjustments of #76 are not implemented.

## Known limitations in the implemented parts

- The MToon shadow prepass has no access to the view time globals, so a cutout
  shadow uses the static UV transform while the main pass uses the animated UV.
  A cutout material whose alpha depends on UV animation can therefore have a
  shadow that does not exactly match the lit alpha.
- `Blend` MToon materials keep Bevy's existing shadow behavior; they are not
  converted to Opaque/Mask.
- The studio cubemap is 16x16 per face. It is filtered on the GPU at startup by
  Bevy's environment-map filter (proper roughness prefiltering, not a plain mip
  chain), but it is a very low-frequency environment by construction.
- `RICH_ROUGHNESS_SCALE = 0.95` and the `STUDIO_PRESET` values are adjustment
  starting points, not measured optima. They were not tuned against a real
  model.
- `PORTRAIT_FINISH` pins the display transform to the app's default exposure
  and Bevy's default `TonyMcMapface` curve. Both are starting points; the
  preset has not been tuned on a real model, and the finish's runtime curve
  dispatch supports every `Tonemapping` variant but only the preset's value is
  reachable through `resolve_portrait_finish`.
- With the look on, the output view composites in linear `Rgba16Float`, so
  MSAA-resolved semi-transparent edges can differ from the plain display by
  ±1/255 after the final 8-bit write (see the #74 alpha checks above).
- `StandardLookBases` assumes the loader gives each avatar its own
  `StandardMaterial` assets; no per-avatar material cloning is performed.
- The rich path applies each light's radiance without a Lambert normalization,
  which is the specification's own toon behavior; the preset keeps every light
  at or below 900 lx so a fully lit surface does not clip.
- The cutout shadow prepass has no access to the view time globals, so a
  cutout shadow uses the static UV transform while the main pass uses the
  animated UV (only while the look is on).

## Settings screen controls (part of #75)

(see above)


