# Avatar rendering and Spout validation

## Who owns each operation

| Operation | Implementation in the tested build |
| --- | --- |
| Camera and tracking input | RusTuberV chooses expression values. On a per-eye avatar it sets `blinkLeft` and `blinkRight` independently. Camera mirroring is an input setting. |
| VRM expressions and shading | The pinned Bevy 0.20 port of `bevy_vrm1` resolves the VRM bindings and supplies MToon. Bevy owns mesh allocation, morph storage, skinning and drawing. |
| Material and UV expressions | The v0.10.0 runtime exposes the effective expression weights. RusTuberV uses them to update native material properties for the binds this version does not apply. The six standard material color targets include `matcapColor`. |
| Preview, NDI and Spout | Bevy renders a transparent BGRA image. The studio registers that same asset with `bevy_egui`. Output subscriptions use Bevy's `Readback`, and Spout submits the original packed Bevy pixels. The NDI worker converts only the frames it sends to straight alpha, copying on write if the original pixels are still shared. Preview alone requests no readback. Spout uses the SDK's D3D11 upload on its worker thread. |

The original VRM and its authored binds are the inputs to the comparisons. The pinned loader needs a compatibility adapter for omitted expression fields and author-defined expressions. Legacy VRM 0.x files go through the existing conversion into the 1.0 runtime. The comparison below covers VRM 1.0 expressions and morph rendering. It does not certify the entire VRM or MToon specifications.

## Closed-eye comparison

[VRMC_vrm expressions](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/expressions.md) identify a target by node and morph index. The expression value scales its bind weight. [glTF morph targets](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#geometry-morph-targets) add the weighted displacement of each corresponding vertex before skinning. The expression names do not prescribe an eyelid shape.

On the original AvatarSample_C, node 113 binds target 15 for the left eye, 14 for the right eye and 13 for the combined blink. All bind weights are 1. The presets are continuous and have no procedural overrides. The managed input and original file had the same SHA-256 (`193d614b524001c27ec401a5a3e21f03f039bee8422a1c9006b39c0d4aa20009`).

* Loaded the original model in Unity 2022.3.22f1 with UniVRM v0.131.2. Set the two per-eye inputs and ran `Vrm10Runtime.Process`. Compared `SkinnedMeshRenderer.BakeMesh` vertex positions against neutral copies with the authored glTF displacements applied to the base geometry. All 19,750 vertices agreed exactly for open, both individual winks, closed and reopened. The maximum deviation at half closure was 1.20e-7 m. The active Unity blend-shape weights were 50 at half closure and 100 at full closure. The same six comparisons on the Inore fixture agreed exactly across 35,744 vertices, including half closure.
* Loaded AvatarSample_C and the Inore VRM 1.0 fixture through the production import, tracking-command, material and output camera systems. Captured open, left, right, half, closed and reopened. With the default renderer (Vulkan on the tested RTX 4090), both winks, half and full closure matched all RGBA pixels of the independently baked reference images on the same backend at 512 × 512 for each model. Reopening restored each model's initial image. Unity and Bevy use different rendering pipelines; the cross-engine comparison above is geometric.
* Disabled tracking mirroring and fixed the camera and clock for these comparisons. The reference camera was positioned at each model's face. The on-disk results and Unity logs for this run are in `target/morph-root/`.

## Upstream report for bevy_vrm1

**Procedural expressions incorrectly suppress members of their own group.** The VRM specification ignores an override targeting the same procedural category. [UniVRM's validator](https://github.com/vrm-c/UniVRM/blob/v0.131.2/Packages/VRM10/Runtime/Components/Expression/DefaultExpressionValidator.cs) implements this separately for blink, mouth and gaze.

1. Start with a working continuous per-eye model. Set `blinkLeft.overrideBlink` to `block` and `blinkRight.overrideBlink` to `blend`. Keep other expression inputs at zero.
2. Set the left/right inputs to 0.5 each. Both effective values should be 0.5. An override from an emotion in a different group must still take effect.
3. The unmodified v0.10.0 pass sums self-directed overrides. In the focused regression a 0.5 expression with its own group's block produced 0.0, as reproduced against the unpatched pass. v0.11.0 also sums all three override rates independent of category in [CategoryMultipliers](https://github.com/not-elm/bevy_vrm1/blob/v0.11.0/src/vrma/animation/properties.rs). The v0.11.0 comparison is source-level.
4. With the three category checks in the pinned expression pass, the modified asset matches UniVRM and the original asset in every tested pose. The regression covers both `block` and `blend` in all three groups. Existing cross-category block/blend tests still pass.

This is a distinct upstream finding. AvatarSample_C's original presets do not request these overrides.

## DX12 reproduction for wgpu

The app previously selected Direct3D 12 for Spout sharing. On wgpu/Naga 30.0.1 with DXC the same WGSL input builtin yields different results when read directly and when passed through a struct. This reproduces with one point, no VRM, Bevy or Spout. It also affects instance indices, as described in [wgpu issue 4564](https://github.com/gfx-rs/wgpu/issues/4564).

```wgsl
struct Input {
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
}
fn unpack(input: Input) -> vec2u {
    return vec2u(input.vertex, input.instance);
}
```

In the vertex entry point, pass `vec4u(input.vertex, input.instance, unpack(input))` as a flat unsigned varying to an RGBA8Uint attachment. Use an index buffer containing 3 and call `draw_indexed(0..1, 5, 7..8)`. Expected bytes are `[8, 7, 8, 7]`. The observed bytes were `[8, 7, 3, 0]`. The zero-offset control yielded `[3, 0, 3, 0]`. Bevy's vertex decompression takes a struct. Its morph addressing then subtracts the mesh's shared-buffer base, so the lost offset selects the wrong displacement.

The application now uses Bevy's normal backend selection. An explicit `WGPU_BACKEND=dx12` can still encounter the driver/compiler path described above. The wgpu dependency, shader translator, Bevy allocator and native VRM shaders contain no local eyelid fix.

## Spout receiver

The Windows hardware test `vtuber-spout --test roundtrip` submits premultiplied BGRA through the same bounded worker as the application. A native Spout receiver checks transparent and semi-transparent pixels, row order and a live resize from 64×32 to 80×32. This transport incurs GPU-to-CPU readback and a D3D11 upload.

The application test renders two asymmetric, differently colored Bevy meshes, one opaque and one translucent, on the studio's transparent target. It captures a reference directly from that GPU texture before starting either transport, then compares all 8,294,400 bytes with a native Spout receiver, first with Spout alone and then with the NDI worker active. It also checks that both outputs use the same readback entity and that stopping the last output removes it. Run on Windows with the sender name available:

```powershell
.\tools\cargo-windows.ps1 -CargoArguments @('test', '--locked', '-p', 'vtuber-app', '--lib', 'receiver_matches_bevy_target_with_spout_alone_and_with_ndi', '--', '--ignored')
.\tools\cargo-windows.ps1 -CargoArguments @('test', '--locked', '-p', 'vtuber-spout', '--test', 'roundtrip', '--', '--ignored')
```

Started Spout from the packaged application's output panel with AvatarSample_C loaded. A separate native Spout receiver read five new 1920×1080 BGRA avatar frames. The fifth contained 1,647,180 transparent, 424,074 opaque and 2,346 semi-transparent pixels. Stopping transmission from the same panel returned the sender to its idle state.

## Preview performance

The optimized tracking replay used AvatarSample_C, a 640×360 recording at 30 Hz, a 1920×1080 VSync window and process affinity restricted to three logical processors. After the five-second warmup, 1,592 frames over 26.54 seconds averaged 59.94 fps on the RTX 4090 (Vulkan). Face and pose/hand inference each ran at 29.99 Hz. The 95th percentile frame time was 18.54 ms. This run measures normal preview with both network and local video senders off.
