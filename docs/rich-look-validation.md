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
above, and any real VRM model. No FPS or image-quality threshold is claimed.

## Not implemented

| Issue | State |
|---|---|
| #72 MToon glossy/environment/extra rim | Not implemented. `MToonPortraitParams`, `resolve_mtoon_portrait`, `apply_mtoon_portrait_settings`, `mtoon_portrait.wgsl` and the MToon specular IBL connection do not exist. |
| #74 HDR finish and transparency | Not implemented. `PortraitFinish`, `resolve_portrait_finish`, `sync_portrait_finish`, `finish_straight_linear_rgb`/`finish_premultiplied_linear` and the alpha/color-space contract table do not exist. |
| #75 one-click UI, 4 languages, per-model save | Not implemented. There is no UI surface, `RichLookChange`/`reduce_rich_look`/`merge_model_look_settings` or settings persistence for the look. The look can currently only be driven by writing `AvatarLookSettings`/`LookSettingsChanged`. |
| #76 material roles | Not implemented. `MaterialRole`, `infer_material_role`, `resolve_material_role`, the role param resolvers and `face_lighting_normal` do not exist. |
| #77 final acceptance | Partially covered by this document (the measurements above). The #75 first-version and #76 role comparisons are not possible yet because those issues are not implemented. |

Consequences: the epic's completion criteria are not met. In particular there
is no one-click switch in the normal screen, MToon materials get no added
gloss/environment/rim, no finish/tone change happens, and per-model look
settings are not saved or restored.

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
