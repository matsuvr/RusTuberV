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
| #69 Native/Rich MToon | this change (working tree on `0018277`) | restores the upstream `f9593fd7` Native fragment (`mtoon_native.wgsl` + `mtoon_fragment.wgsl`), adds `MToonShadingMode`/`MToonMaterialKey::RICH_SHADING` shader selection, `mtoon_rich_fragment.wgsl` and `compose_rich_mtoon`; the per-light radiance, normal texture, GI equalization and added gloss are now Rich-only. The review fixes are included: the Rich transparent discard uses the authored base alpha (Blend outline), and the prepass activates the Rich cutout only for `RICH_SHADING` materials. `mtoon-reference` compares against the independent fixed-revision reference. |
| #70 look boundary | `ea47b59` | `RichLookSettings`/`effective_look_strength`/`blend_look_scalar`, `StandardLookBase` capture, `AvatarLookSettings`+`LookSettingsChanged`, capture-once/unload lifecycle |
| #71 studio lighting | `83c5344` | `StudioPreset`/`StudioLight`/`StudioRig` solve+blend, key takeover + fill/rim, ambient/environment sync, restore on strength 0, MToon cutout prepass, generated studio cubemap |
| #71 cutout shadow UV | `ac69489`/`b59c651` (PR #81) | extracts the UV animation into `mtoon_uv.wgsl` (pure expression, caller-supplied clock), has the prepass read the prepass view globals at binding 1 and reuse the animated UV, and adds animated scenes to `mtoon-cutout-shadow` (exact-phase cutout shadow) and `mtoon-reference` (scrolling Mask sphere); the fixture capture returns `Ok` only on 40 identical frames (expected-empty measurements opt in explicitly) and otherwise reports the unmet condition |
| #71 cutout base alpha | this change (working tree on `4bc53a2`) | `mtoon::alpha::mtoon_alpha_at_uv` now returns the authored base alpha (`material.base_color.a` times the base color texel); the Rich fragment discard imports that shared function instead of its private copy and the prepass tests the same value, so a cutout whose lit Mask test discards for a zero authored base alpha no longer casts the opaque texel's shadow. `mtoon-cutout-shadow` adds the zero-base-alpha scene. Native (and Rich(0)) keep the upstream depth-only shadow. |
| #73 Standard portrait | `a0d64f6` | `resolve_standard_portrait`/`apply_standard_portrait_settings` (roughness-only relative adjustment, unlit and MToon untouched) |
| #74 HDR finish | this PR | `PortraitFinish`/`PORTRAIT_FINISH`/`resolve_portrait_finish`, `sync_portrait_finish` capture-once/restore, `Hdr`+`Exposure`+`Tonemapping` camera components, and the alpha-aware finish pass (`finish.wgsl`: `finish_straight_linear_rgb`/`finish_premultiplied_linear`) with the explicit final contract `a * E(T(C))` |
| #79 avatar UI alpha boundary | this PR | avatar-only callback for the monitor and expanding transition; the shared `Bgra8UnormSrgb` image remains gamma-premultiplied while the callback supplies linear-premultiplied RGB to the UI blend |

Supporting reusable code: `tools/xtask/src/rich_look.rs` (the GPU cases,
including `mtoon-reference`, `mtoon-rich-zero`, `mtoon-blend-depth`,
`standard-look`, `finish-alpha` and `avatar-ui-alpha`),
`tools/xtask/src/mtoon_upstream_reference.wgsl` (the independent
fixed-revision reference),
`crates/vtuber-app/src/ui/avatar_preview.rs`/`avatar_preview.wgsl`,
`vendor/bevy_vrm1/src/vrm/mtoon_native.wgsl` (the fixed Native reference),
`mtoon_rich_fragment.wgsl`, `mtoon_lighting.wgsl`, `mtoon_alpha.wgsl`,
`mtoon_uv.wgsl`, `mtoon_prepass.wgsl`.

## Measured on the GPU (Windows/Vulkan/RTX 4090)

`mtoon-lighting` (64x64 BGRA readback, center pixel; Rich display, strength 1):

| case | pixel |
|---|---|
| zero illuminance | `[0, 0, 0, 255]` |
| 200 lx white | `[125, 125, 125, 255]` |
| 400 lx white | `[172, 172, 172, 255]` |
| 200 lx red | `[0, 0, 125, 255]` |
| 200 lx blue | `[125, 0, 0, 255]` |
| 2 x 100 lx white | `[125, 125, 125, 255]` |
| 100 lx white | `[90, 90, 90, 255]` |

The Rich direct term follows each light's own radiance, light color is per
light, and two lights add. These numbers describe the Rich display only; the
Native display deliberately does not apply the light level or color.

`mtoon-standard` (the restored Native display, `MToonShadingMode::Native`): a
fully lit white plane at 650 lx and at 1300 lx both read `[255, 255, 255, 255]`,
and the same scene under a red light also reads `[255, 255, 255, 255]`, so the
Native display shows the authored base color and does not apply the light's
level or color. A light travelling at 90° reads `[187, 187, 187, 255]`, so the
light direction still shapes the ramp. The same scene on the Rich display at
strength 1 reads `[253, 253, 253, 255]` at the same probe.

## #69 measured: Native vs Rich with zero added effect (Windows/Vulkan/RTX 4090)

All comparisons below run in one build (same `Cargo.lock`), backend (Vulkan) and
input scene, through the production offscreen readback path. None of them
compares the OFF switch state.

### Independent upstream reference

`cargo run -p xtask -j 1 -- rich-look mtoon-reference` renders each scene three
ways: the fixed-revision reference (`tools/xtask/src/mtoon_upstream_reference.wgsl`,
copied verbatim from `f9593fd7` and substituted for the Native fragment handle,
so it does not share `mtoon::native`), the production Native display
(`MToonShadingMode::Native`) and Rich with the nominal gains and `strength = 0`.

| scene inputs | reference vs Native | Native vs Rich(0) | Rich(1) |
|---|---:|---:|---:|
| sphere, two colored lights, authored parametric rim + MatCap, tilted normal map, UV transform, world outline | 0 | 0 | 468 |
| Mask sphere (cutoff 0.5) with tilted normal map | 0 | 0 | 452 |
| Blend + `transparentWithZWrite` sphere | 0 | 0 | 409 |
| Mask sphere with a scrolling base texture at a half-period clock | 0 | 0 | 1058 |

Every scene is byte-identical across the reference, the production Native
display and Rich(0). The animated scene exercises the shared UV expression
under animation: its Native image differs from the same scene at a frozen
clock in 2304 pixels, so the zero difference is not a comparison of two static
images. The compared inputs are observable on the same fixture:
the tilted normal map changes the Rich image in 440 pixels and the
Native/reference image in 0 (normal evaluation is Rich-only), removing the
parametric rim changes 312 pixels, removing the MatCap texture 468, and the UV
transform 314. The sphere outline adds visible pixels (514 opaque pixels with
the outline against 468 without).

### Zero-effect identity without the reference

`cargo run -p xtask -j 1 -- rich-look mtoon-rich-zero` compares Native and
Rich(0) directly on the same scenes and checks that a positive strength changes
them.

| scene inputs | differing pixels | max channel difference | Rich(1) differing pixels |
|---|---:|---:|---:|
| sphere, two colored lights, authored parametric rim, tilted normal map, world outline | 0 | 0 | 468 |
| alpha Mask base color texture at cutoff 0.5 | 0 | 0 | 1058 |
| alpha Blend + `transparentWithZWrite` base color texture | 0 | 0 | 966 |

The tilted normal map changes the Rich image in 449 pixels, so the normal input
is not an identity comparison of identical images. `mtoon-portrait` additionally
checks the same identity on a single light: `native=[255,255,255,255]`,
`rich_strength_zero=[255,255,255,255]`, `rich=[253,253,253,255]`,
`rich_rotated_light=[244,244,244,255]`.

### Blend outline and Z-write

`cargo run -p xtask -j 1 -- rich-look mtoon-blend-depth`:

| scene | Native | Rich(0) | Rich(1) |
|---|---:|---:|---:|
| fully transparent outlined Blend+Z-write sphere, opaque pixels | 92 | 92 | 0 |
| outlined opaque sphere behind a transparent Blend+Z-write quad, black outline pixels | 0 | 0 | 58 (58 without the quad) |

The transparent Blend outline is drawn on the Native display and not drawn when
the Rich display adds an effect; Rich(0) is byte-identical to Native. The
transparent quad's depth write hides the outline behind it on the Native
display (58 outline pixels become 0); Rich(1) discards the transparent fragment,
so the scene is byte-identical to the same scene without the quad. This is the
Rich-side discard introduced for issue #69: it evaluates the authored base
alpha (base color x base texture), not the outline-pass alpha that the Native
`lit_color` forces to 1 for Blend materials.

### Shadow/prepass split

`cargo run -p xtask -j 1 -- rich-look mtoon-cutout-shadow` renders the same
alpha-masked quad on a lit ground in several states. The Native display is
byte-identical with `portrait.strength` 0 and 1 (0 differing pixels), and
Rich(0) equals it; Rich(1) activates the cutout shadow (darkest ground band
under the opaque half `1`, under the transparent half `256`, against `256/256`
without shadows). The prepass uses the GPU-side
`MtoonFlags::RICH_SHADING` value derived from `shading_mode`, so a saved
positive strength cannot activate the Rich cutout while Native is selected.

The same case also animates the cutout's UV. The mask texture repeats and the
material scrolls one UV unit per second; the fixture advances its otherwise
frozen clock once by an exact half period, so the opaque texel moves to the
other half of the quad. The lit pass and the shadow pass evaluate the same UV
expression (`mtoon::uv`) at the same frame clock through their own view
bindings (the forward pass reads `globals` at binding 11, the prepass reads
its view globals at binding 1). Measured on the Windows/Vulkan/RTX 4090
device:

| state | darkest left ground band | darkest right ground band |
|---|---:|---:|
| animated material, frozen phase | 1 | 256 |
| animated material, half-period phase | 256 | 0 |
| same scene without shadow maps | 256 | 256 |

The animated material at the frozen phase is byte-identical to the static
material (0 differing pixels), and the Native display with the animated
material is still byte-identical with strength 0 and 1 (0 differing pixels).
Before the shared UV change, the shadow used only the static UV transform and
stayed on the frozen phase's half at the half-period phase; the fixture fails
on that result.

The same case also covers the authored base alpha. The lit Mask test discards
on `material.base_color.a` times the base color texel, and the shadow/prepass
test now tests that same shared `mtoon::alpha::mtoon_alpha_at_uv` value. A
cutout with the same opaque texture but `base_color.a = 0` is invisible in the
lit pass and must not cast the opaque texel's shadow either. Measured on the
Windows/Vulkan/RTX 4090 device:

| scene | darkest left ground band | darkest right ground band |
|---|---:|---:|
| Rich(1), opaque base alpha, opaque/transparent texels | 1 | 256 |
| Rich(1), zero base alpha, same texture | 256 | 256 |
| same scene without shadow maps | 256 | 256 |

The Rich fragment's transparent-fragment discard imports the same shared
function, so the Blend discard and the shadow cutout cannot drift apart.
Before this change the shadow test read only the texture alpha and the
zero-base-alpha scene left the opaque texel's shadow behind (`left=1`); the
fixture fails on that result. The Native display is unaffected: the prepass
alpha test stays gated on `RICH_SHADING` and `portrait_strength > 0`, so
Native (and Rich(0)) keep the upstream depth-only shadow.

The MToon fixture capture waits for 40 byte-identical frames (`SETTLED_FRAMES`)
before sampling. The previous 3-non-empty-frames wait could sample a frame
before the first shadow-map and pipeline setup had landed; re-running the
unchanged `mtoon-cutout-shadow` case failed intermittently for that reason
(Native-vs-Rich comparisons of random app instances differed by the missing
shadow). All MToon fixture cases re-run with the settled capture. The capture
returns `Ok` only when the 40-identical-frames condition is met; a run that
ends first reports the unmet condition (no readback at all stays `NOT RUN`)
instead of using the last image. Frames must also be non-empty, except for the
explicit expected-empty capture that `mtoon-blend-depth` uses for the Rich(1)
discard of the fully transparent Blend outline (a stable transparent image is
otherwise still not accepted). The CPU tests in `tools/xtask/src/rich_look.rs`
cover the rule: identical frames settle; changing frames, fewer than the
required frames and an early `AppExit` do not; no readback is `NOT RUN`.

Local validation for the base-alpha change (working tree on `4bc53a2`):
`cargo test --workspace -j 4` and
`cargo clippy --workspace --all-targets -j 4 -- -D warnings` PASS; the 14 GPU
cases re-ran PASS on the Windows/Vulkan/RTX 4090 device, including the new
zero-base-alpha scene in `mtoon-cutout-shadow`. With the pre-change
texture-only alpha the new scene fails (`left=1`), so the fixture measures the
leak it fixes. Not run in this change: real NDI send/receive, macOS, and any
GPU other than the RTX 4090 above. The #77 final acceptance remains with #77
and is not claimed here.

Local validation for the PR #81 change: `cargo test --workspace -j 4` and
`cargo clippy --workspace --all-targets -j 4` PASS; the 14 GPU cases above
re-ran PASS on the same device with the settled capture; `vrm-render` on
`inore-vrm1.vrm` and `tsukuyomi-chan.vrm` re-ran PASS with the same numbers as
the table in "Representative VRM hardware check" (off/on means `72.21`/`61.07`
and `39.01`/`31.25`, OFF restore/ON repeat/strength-0 differences `<= 0.004`),
so the Native/Rich boundary and the OFF restoration are unchanged on the real
models. Not run in this change: real NDI send/receive, macOS, and any GPU
other than the RTX 4090 above. The #77 final acceptance (head turn, camera
orbit, model size differences, moving bangs/hand shadows, first-version device
tuning) remains with #77 and is not claimed here.

## #72 measured: environment reflection and the outline line (Windows/Vulkan/RTX 4090)

The 2026-09-20 review found two gaps in the merged portrait work, and both are
closed here:

1. `mtoon_portrait.wgsl::portrait_environment_sample` returned
   `vec3<f32>(0.0)` unconditionally, so the added MToon environment reflection
   never sampled anything. It now samples the view's prefiltered specular
   cubemap (`specular_environment_maps` binding array or the single
   `specular_environment_map`) through `environment_map_sampler` at the
   roughness-selected mip, guarded by the same `ENVIRONMENT_MAP`/`MULTIPLE_LIGHT_PROBES_IN_ARRAY`
   shader defs Bevy's own PBR shader uses; without a view environment map the
   term stays zero. The sample reuses Bevy's split-sum BRDF (`F_AB`) and the
   view probe intensity; no environment diffuse is added (the MToon GI reads
   only `lights.ambient_color`, so attaching an environment map cannot
   double-add diffuse).
2. The Rich `OUTLINE_PASS` composed the portrait terms into the color the
   author's `outline_lighting_mix_factor` mixes into the line. It now keeps
   the author's own outline result: the line color is mixed from the fixed
   Native lit color, so the added specular/IBL/rim never enters the line and
   the line is identical to the Native display's at any strength.

The environment term's mip index uses the *physical* roughness the GPU
prefilter assigned per mip (`generate.rs`: mip k ↔ roughness `k/(mips-1)`),
so `roughness * smallest_specular_mip_level_for_view` selects the intended
prefiltered level.

`cargo run -p xtask -j 1 -- rich-look mtoon-portrait` (extension): with the
added direct specular and rim at zero and only the generated studio cubemap
attached (no directional light), the MToon plane reads `[0, 0, 0, 255]` at
environment gain 0 and `[12, 10, 10, 255]` at gain 1. Before this change the
gain-one frame was also `[0, 0, 0, 255]` (the sample was hard zero), so the
fixture measures the gap it fixes; the studio dome floor texel is the
reflection the plane's center normal reaches.

`cargo run -p xtask -j 1 -- rich-look mtoon-outline-mix` (new): a sphere with
an outline whose `outline_lighting_mix_factor` is 1.0 and a gray outline
color. The fixture masks the 46 outline-only pixels outside the silhouette
(opaque only in the outlined render) and requires them byte-identical between
the Native display and Rich at strength 1 with the nominal gains:
`line_differing_pixels=0`, while the lit body differs in 468 pixels, so the
added terms are active in the same comparison. With the pre-change Rich
outline color (composed including the portrait terms) the same fixture fails:
46/46 outline pixels differ (max channel difference 247), so it measures the
leak it fixes.

Re-run after both fixes (same build, device and cases as the sections above):

| check | result |
|---|---|
| `mtoon-reference` (reference vs Native, Native vs Rich(0)) | 0 differing pixels on all four scenes; Rich(1) values unchanged (468/452/409/1058) |
| `mtoon-rich-zero` (Native vs Rich(0)) | 0 differing pixels on all three scenes; Rich(1) values unchanged |
| `mtoon-blend-depth`, `mtoon-cutout-shadow` | PASS, same numbers as above (outline/alpha/prepass regressions none) |
| `mtoon-portrait` single light | `native=[255,255,255,255]`, `rich_strength_zero=[255,255,255,255]`, `rich=[253,253,253,255]`, `rich_rotated_light=[244,244,244,255]` |
| `vrm-render inore-vrm1.vrm` | off `72.21`, on `61.12`, OFF restore `0.001`, ON repeat `0.000`, strength 0 `0.001`, opaque `19848/19848` |
| `vrm-render tsukuyomi-chan.vrm` | off `39.01`, on `31.28`, OFF restore `0.004`, ON repeat `0.003`, strength 0 `0.004`, opaque `10455/10455` |

The two ON means moved from the earlier `61.07`/`31.25` by less than a tenth of
a mean byte: outlines whose materials use a nonzero
`outline_lighting_mix_factor` no longer receive the added gloss, which is the
fix itself. OFF, OFF-restore and strength-0 are unchanged within the same
`<= 0.004` bound as before, so the Rich-zero/Native identity still holds on
the real models.

Not measured in this change: the key-light shadow through-check for the added
direct specular as a separate GPU scene (the visibility multiplication that
gates it is the same `fetch_directional_shadow` sample the Native direct term
reuses in `mtoon_rich_fragment.wgsl::calc_rich_light_visibility`), real NDI
send/receive, macOS, and any GPU other than the RTX 4090 above. The initial
value tuning on real models remains with #77.

Local validation for this change: `cargo test --workspace -j 4` and
`cargo clippy --workspace --all-targets -j 4 -- -D warnings` PASS; the 15 GPU
cases (`mtoon-outline-mix` is new) re-ran PASS on the Windows/Vulkan/RTX 4090
device, plus the two `vrm-render` models above. Both new checks were verified
against the pre-fix behavior they replace (the hard-zero environment sample
and the composed Rich outline color) and fail on it, so they measure the gaps
they close.

### Other reused fixtures

`mtoon-normal` runs on the Rich display (normal texture, scale and TBN wiring
are Rich-only): flat `[118,118,118,255]`, identity map `[118,118,118,255]`,
tilted map `[95,95,95,255]`, tilted map with scale 0 `[118,118,118,255]`.

`mtoon-shading` (Rich display): front-lit `125`, 90-degree side `89`, side with
+0.5 shift `124`, light behind `0`; the toony=0 ramp has 18 intermediate
samples on the mid scanline and the toony=1 endpoint has 0.

`mtoon-shadow` (Rich display): darkest ground pixel under an MToon sphere is `0`
with shadow maps on and `267` with them off, so the MToon mesh casts into the
shadow map.

### Standard/Unlit

`cargo run -p xtask -j 1 -- rich-look standard-look` compares the production
`initialize_look_materials`/`apply_standard_portrait_settings` against a control
app that has no look systems at all:

| material | control vs Rich(0) | control vs Rich(1) |
|---|---:|---:|
| lit `StandardMaterial` sphere | 0 | 99 (recorded; #73 owns the positive Standard change) |
| unlit `StandardMaterial` sphere | 0 | 0 |

The lit and unlit Standard materials are byte-identical to the untouched Bevy
material at zero effect, and Unlit stays byte-identical at full strength.

### Not measured here

- Time-driven UV animation is verified on the GPU by `mtoon-cutout-shadow`
  (exact-phase animated cutout shadow) and `mtoon-reference` (scrolling Mask
  sphere against the fixed-revision reference); both advance the frozen clock
  once to an exact phase. The other fixtures keep the
  `TimeUpdateStrategy::ManualDuration(Duration::ZERO)` clock, so they compare
  only the shared UV transform path. The animation functions are shared by both
  display paths through `mtoon::native`/`mtoon::uv`; expression and animation
  behavior is also covered by the workspace tests.
- The light level of the scene is the app's own `setup_scene` key (650 lx). The
  Native display is deliberately the upstream authored display; the app's look
  switch selects the Rich display, which applies the light level and colors.
  Bevy applies `Tonemapping` only to HDR views, so a real highlight roll-off
  needs the HDR output of issue #74; the Rich preset keeps every light at or
  below 900 lx so a fully lit surface does not clip.

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

These four rows were measured before the #69 Native/Rich split, so both the OFF
(Native) and ON values are historical; the two models re-measured after the
split are in the #69 section above.

The ON state is now *dimmer on highlights than the pre-#74 look* while the
portrait rig still adds light: AvatarSample_C's bright jacket panel reads
`[245, 237, 236]` in the pre-#74 ON frame and `[193, 188, 188]` with the
finish, and the face keeps its tone, so the added light no longer clips to
white.

Alpha contract checks (byte comparisons of the readback frames; also measured
before the #69 split, when OFF was the modified standard path):

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

### Local validation for this change (#69)

- `cargo test --workspace -j 4`: PASS.
- `cargo clippy --workspace --all-targets -j 1 -- -D warnings`: PASS.
- GPU fixtures run on the Windows/Vulkan/RTX 4090 above: `mtoon-reference`
  (new), `mtoon-rich-zero`, `mtoon-blend-depth` (new), `mtoon-cutout-shadow`,
  `mtoon-lighting`, `mtoon-shading`, `mtoon-normal`, `mtoon-standard`,
  `mtoon-portrait`, `mtoon-shadow`, `standard-look` (new), `studio-environment`,
  `finish-alpha`, `avatar-ui-alpha`: PASS (14 cases).
- `vrm-render` on `inore-vrm1.vrm` and `tsukuyomi-chan.vrm`: PASS (numbers
  below).
- Real NDI send/receive and macOS hardware validation: NOT RUN.

The numbers for every GPU case are in the sections above; the previous
`mtoon-standard` numbers (`[211,211,211]` at 650 lx) belonged to the modified
standard path that this change removes and are superseded.

`studio-environment`: a PBR sphere lit only by the generated studio cubemap is
`[87, 82, 82, 255]` against `[1, 0, 1, 255]` without it, so the GPU prefilter
produced usable diffuse/specular maps from the bundled 16x16x6 cubemap.

Not measured in these synthetic cases: frame times, GPU time, NDI output, or
any GPU/OS other than the one above. Real VRM coverage is recorded separately
below.

## Representative VRM hardware check

On the same Windows/Vulkan/RTX 4090 device, the production managed lifecycle
was run with two fixtures that are present in this repository. `vrm-render`
captures `OFF -> ON -> OFF -> ON -> strength 0`, and requires the ON image to
change while both restoration paths and the repeated ON image remain within
`0.5` mean absolute byte difference. OFF is the restored Native display and
strength 0 is Rich with no added effect, so the strength-0 restoration is also
the switch-independent Native/Rich identity check on real models.

| model | off mean | on mean | ON diff | OFF restore diff | ON repeat diff | strength 0 diff | opaque pixels off/on |
|---|---:|---:|---:|---:|---:|---:|---:|
| `inore-vrm1.vrm` | 72.21 | 61.07 | 11.176 | 0.001 | 0.000 | 0.001 | 19848 / 19848 |
| `tsukuyomi-chan.vrm` | 39.01 | 31.25 | 7.780 | 0.004 | 0.003 | 0.004 | 10455 / 10455 |

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

The per-model table below was measured with the previous display path and was
not re-measured after this change; it is kept as a historical record. The
models re-measured for this change are the two rows above.

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
| #72 MToon glossy/environment/extra rim | Implemented and measured (see the "#72 measured" section): `MToonPortraitParams` on the material, `resolve_mtoon_portrait`, `apply_mtoon_portrait_settings`, `mtoon_portrait.wgsl`, the Rich `OUTLINE_PASS` keeps the author's line, and the added environment reflection actually samples the view's prefiltered specular cubemap; verified by `mtoon-portrait`, `mtoon-outline-mix` and the re-run identity fixtures. The 2026-09-20 review's two gaps (a hard-zero environment sample and the added gloss flowing into the outline line) are fixed and measured. |
| #74 HDR finish and transparency | Implemented. `PortraitFinish`, `resolve_portrait_finish`, `sync_portrait_finish`, `finish_straight_linear_rgb`/`finish_premultiplied_linear` and the alpha/color-space contract table exist in `crates/vtuber-avatar/src/look/finish.rs` and `finish.wgsl`; verified by the #74 measurements above. |
| #75 one-click UI, 4 languages, per-model save | Partially implemented: the switch and the strength slider exist in the settings screen in four languages (see above). Persistence, per-model settings and restore on model switch are not implemented. |
| #76 material roles | Not implemented. `MaterialRole`, `infer_material_role`, `resolve_material_role`, the role param resolvers and `face_lighting_normal` do not exist. |
| #77 final acceptance | Partially covered by this document (the measurements above). The #75 first-version and #76 role comparisons are not possible yet because those issues are not implemented. |

Consequences: the epic's completion criteria are not fully met yet. The
finish/tone change (#74) is in place and the look can be switched on and
adjusted from the settings screen; per-model look settings are not saved or
restored, and the role-based adjustments of #76 are not implemented.

## Known limitations in the implemented parts

- `Rich` MToon materials only discard fully transparent blend fragments while
  the added effect amount is positive; at strength 0 (and on the Native display)
  the upstream depth behavior is kept, so the shape-key overlay fix of `7708ac1`
  does not apply to the plain display. The discard evaluates the authored base
  alpha (base color x base texture) at the animated UV, so it covers the outline
  pass too, where the Native `lit_color` forces Blend alpha to 1.
- The Native MToon display is the fixed upstream `f9593fd7` display: it does
  not apply the light's level or color, does not evaluate the normal texture,
  and keeps the upstream alpha/depth behavior. Those are Rich effects, not
  missing Native fixes; issue #69 removed the earlier Native improvements on
  purpose.
- The MToon shadow prepass now shares the material's UV animation with the lit
  pass (`mtoon::uv`) and reads the same frame clock, so an animated cutout
  casts the silhouette it shows. Its alpha test uses the shared authored base
  alpha (`material.base_color.a` times the base color texel), the same value
  the lit Mask test uses, so a zero authored base alpha discards the shadow
  too. This stays a Rich effect: the shadow/prepass alpha test is active only
  while the look is on (`portrait.strength > 0` on the Rich display). The
  Native display deliberately keeps the upstream depth-only shadow (a Mask
  quad casts a solid shadow), and Rich(0) restores that same behavior.
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

## Settings screen controls (part of #75)

(see above)


