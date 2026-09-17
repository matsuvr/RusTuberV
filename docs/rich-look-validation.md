# Rich look validation (Issues #68-#77)

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
  available instead of reporting success.
- macOS and other GPUs were not available and are not measured.

## Implemented and committed

| Issue | Commit | Scope |
|---|---|---|
| #69 standard MToon | `57977e2` | per-light direct term, signed NdotL/shift/toony endpoints, single exposure application, spec GI equalization, glTF/legacy normal texture at bindings 117/118 |
| #70 look boundary | `ea47b59` | `RichLookSettings`/`effective_look_strength`/`blend_look_scalar`, `StandardLookBase` capture, `AvatarLookSettings`+`LookSettingsChanged`, capture-once/unload lifecycle |
| #71 studio lighting | `83c5344` | `StudioPreset`/`StudioLight`/`StudioRig` solve+blend, key takeover + fill/rim, ambient/environment sync, restore on strength 0, MToon cutout prepass, generated studio cubemap |
| #73 Standard portrait | `a0d64f6` | `resolve_standard_portrait`/`apply_standard_portrait_settings` (roughness-only relative adjustment, unlit and MToon untouched) |

Supporting reusable code: `tools/xtask/src/rich_look.rs` (six GPU cases),
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

`mtoon-standard`: the standard path (`MToonMaterial::look_strength == 0`)
renders a fully lit white plane at `[255, 255, 255, 255]` for 200 lx, 1500 lx
and 10000 lx, and for a red light as well as a white one; a light behind the
surface gives `[187, 187, 187, 255]`. The light's intensity and color do not
scale the standard display and a fully lit surface shows the authored base
color. `mtoon-lighting` is the same scene with `look_strength == 1`, where the
intensity and color do matter.

`MToonMaterial::look_strength` selects the path, and the zero path is a
verbatim reproduction of the renderer before this epic:

| standard path | source before the epic (`db364e0`) |
|---|---|
| `mtoon_standard_shading` | `calc_mtoon_lighting_shading` (saturate + `mtoon_linearstep` ramp) |
| `apply_standard_directional_lights` | `apply_directional_lights` (accumulate the shading, one `mix`) |
| `apply_standard_global_illumination` | `apply_global_illumination` (`view.exposure * ambient_light`) |
| `apply_standard_mtoon_lighting` | `apply_mtoon_lighting` (`direct + gi + emissive + rim`) |

The normal-map evaluation and the cutout shadow prepass are also gated on
`look_strength > 0`, so the standard display keeps the plain geometric normal
and the plain depth-only shadow.

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

Not measured: frame times, GPU time, NDI output, any GPU/OS other than the one
above, and any real VRM model. The screenshots taken while reviewing the
result are visual inspection only; they are not a pixel measurement. No FPS or
image-quality threshold is claimed.

## Per-model rich-look check

`cargo xtask vrm-render <vrm-or-dir> <out-dir>` loads every model through the
production managed lifecycle with a frozen clock, a fixed camera and the
production offscreen readback, captures one 256x256 frame with the look off and
one with it on, and writes both frames (`.bgra` and `.png`).

| model | off mean | on mean | mean abs diff |
|---|---|---|---|
| 1565994099520778586 | 72.03 | 75.07 | 3.13 |
| AvatarSample_C | 36.99 | 43.58 | 6.59 |
| IrisPart1 | 36.89 | 38.47 | 2.60 |
| IrisPart1(ShapeKey Reduce) | 36.89 | 38.42 | 2.53 |
| IrisPart1(ShapeKey Reduce2) | 36.89 | 38.42 | 2.53 |
| RearAlice_3.0 | 29.41 | 30.49 | 2.22 |
| RearAliceLite_3.0 | 29.43 | 30.51 | 2.22 |
| Sapphy | 38.68 | 40.80 | 3.34 |
| SapphyPerfectSync | 38.68 | 40.80 | 3.34 |
| つくよみちゃん（タイプA・マテリアル数18） | 39.01 | 38.95 | 0.83 |

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
| #74 HDR finish and transparency | Not implemented. `PortraitFinish`, `resolve_portrait_finish`, `sync_portrait_finish`, `finish_straight_linear_rgb`/`finish_premultiplied_linear` and the alpha/color-space contract table do not exist. |
| #75 one-click UI, 4 languages, per-model save | Partially implemented: the switch and the strength slider exist in the settings screen in four languages (see above). Persistence, per-model settings and restore on model switch are not implemented. |
| #76 material roles | Not implemented. `MaterialRole`, `infer_material_role`, `resolve_material_role`, the role param resolvers and `face_lighting_normal` do not exist. |
| #77 final acceptance | Partially covered by this document (the measurements above). The #75 first-version and #76 role comparisons are not possible yet because those issues are not implemented. |

Consequences: the epic's completion criteria are not met. In particular MToon
materials get no added gloss/environment/rim, no finish/tone change happens,
and per-model look settings are not saved or restored. The rich look itself can
be switched on and its strength adjusted from the settings screen.

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

