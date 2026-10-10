# Looking Glass output prototype

Rust/Bevy 0.20 plus an embedded WESL interlacer. No Bridge DLL, C/C++, JavaScript,
texture-sharing API or CPU image readback is added. This is a direct-display
experiment, not an implementation of the Bridge SDK API.

The user reported approximately **27fps with Looking Glass enabled**, versus
comfortably 60fps with it disabled. That is the motivation for the array/capture
optimization below, not a benchmark of the new implementation. The new path's
build, GPU image comparison, optical output and frame rate remain unverified.

## Run

Copy the **actual device's** `LKG_calibration/visual.json` to your PC. The file
is exposed on supported devices' USB storage; the manufacturer's calibration
instructions are linked below. Do not substitute another display's values.
This prototype reads that raw JSON, not Bridge's already-normalized shader values.

Create a configuration file next to the copied `visual.json`. For example, this
Go-shaped layout uses 66 views at half the reference quilt's linear resolution:

```json
{
  "calibration": "visual.json",
  "columns": 11,
  "rows": 6,
  "view_width": 186,
  "view_height": 341,
  "depth_scale": 1.0
}
```

The virtual quilt is 2046 x 2046. A Portrait-shaped starting point is 8 columns,
6 rows, 210 x 280 per view. The grid/tile size is configurable, not inferred from
a device-name heuristic. Calibration paths are relative to the configuration
file; absolute paths also work. `depth_scale` multiplies the view-cone baseline:
0 removes parallax, 1 uses the calibrated view cone. Changing settings requires
restarting this prototype. Existing configuration files need no changes.

```sh
cargo run --release -p vtuber-desktop -- --looking-glass /path/to/looking-glass.json
```

For the packaged macOS application (built with `bash tools/build-macos.sh`):

```sh
open -n target/release/RusTuberV.app --args --looking-glass /path/to/looking-glass.json
```

For the Windows executable (PowerShell, after a normal Windows release build):

```powershell
.\target\release\RusTuberV.exe --looking-glass 'C:\path\to\looking-glass.json'
```

The existing `--model /path/to/avatar.vrm` argument can be combined with this.
Without `--looking-glass`, no output systems, cameras or textures are installed.

Complete the device manufacturer's normal desktop/video-input setup, including
its Bridge runtime where required by the device. This application does not link
to or call that runtime; it does not replace device setup or firmware.
Use the OS extended-desktop mode. Move the new window to the Looking Glass and
press **F11** for borderless fullscreen on that monitor. Its actual framebuffer
must match `screenW` x `screenH` in `visual.json`: choose the panel's native display
resolution, not a mirrored/scaled desktop. Windowed output is only a preview;
scaling an interlaced image destroys its pixel-to-lens correspondence. Move the
pointer off the display. The physical video connection can be HDMI or another
connector supported by that particular device; no HDMI-specific API is needed.

**F10** toggles a virtual quilt for inspection, sampled directly from the array.
**Escape** closes this output while it has focus. These keys act only on the
auxiliary window. Closing it also removes every associated camera and image
handle; closing the main window closes the auxiliary window too. There is no
automatic device detection, hot-plug restart, calibration download, settings UI
or fallback display mode.

## Rendering and scope

The existing live avatar world is rendered from parallel, horizontally translated
cameras. An asymmetric perspective projection cancels disparity at the camera
control's focus plane. Orbit/pan/dolly in the existing preview also move the
multiview rig; the avatar, tracking and animation are not duplicated.

### Array capture instead of quilt assembly

The previous path allocated an output image for every view, blitted each camera's
main texture to that image, and drew all those images as sprites into a quilt
through another 2D camera. The interlacer then sampled the quilt.

The new path keeps the same view count and per-view resolution, but changes the
storage and hand-off:

1. Every 3D view uses the same **small logical render target**, with MSAA off and
   an opaque-black clear. Bevy can share the small main/depth allocations instead
   of allocating them separately for all 66 cameras. The logical target image is
   retained because a clearing camera without an output attachment has no
   `ViewTarget` in Bevy 0.20; it is not a second output path.
2. `CameraOutputMode::Skip` removes each camera's final output blit. A `Core3d`
   capture system runs after post-processing and the skipped upscaling system,
   copying `ViewTarget::main_texture()` into that view's texture-array layer.
   This runs **inside each camera schedule**, after Native/Rich MToon outlines
   and before the next camera overwrites the shared scratch image. Each view
   has a distinct negative camera order; the display camera runs at order zero.
3. The interlacer samples `texture_2d_array` directly. There is no quilt texture,
   quilt camera, or quilt sprite entity. F10 computes the grid presentation in
   the same display shader. Missing avatar framing disables the view cameras
   and hides the output mesh, so retained array pixels are not displayed.

For 66 views, this removes **66 final-view raster blits**, all **66 quilt sprites**,
and the **quilt camera's rendering/output work**. It replaces the hand-off with
66 GPU texture-copy commands. Copies still use GPU bandwidth; this is not a
zero-copy renderer. No CPU image allocation is made for the array, and no
pixels are read back to the CPU. Texture-array dimensions are subject to the
active GPU's actual texture-size/layer limits, not an application quality cap.

Output excludes the floor and egui and does not change preview or NDI paths.
No custom changes to the avatar's MToon/Standard/unlit materials or scene
lighting are made. Cameras retain fixed SDR exposure, disabled tonemapping
and disabled dithering. The scratch and array formats are both explicitly
`Rgba8UnormSrgb`, so capture is a texture copy rather than a shader color
conversion. The shader samples completed colors rendered over black directly;
it no longer repeats the former sprite blend and intermediate sRGB writes.
Bit-identical output is not claimed: alpha-edge behavior and rounding after
removing those stages need GPU image comparison.

Array layer zero is optical view zero; the virtual quilt shows it bottom-left.
Lens coordinates are bottom-left while Bevy image UVs are top-left. Sampling
stays half a texel inside each layer. Unlike an atlas, filtering cannot cross
into a neighboring view. The existing optical phase calculations, arithmetic
order, RGB selection, orientation flags and cell-pattern handling are retained.
Nearest-view selection is used; no view interpolation or depth-based view
synthesis is implemented. Optical transforms are separate from image orientation.

### What this does not optimize

This is **not instanced multiview scene rendering**. All 66 camera schedules,
visibility/material preparation, scene draws, vertex skinning/morph work and
view-dependent shading still run. Texture arrays alone do not eliminate them.
True cross-view instancing or hardware multiview requires passing the correct
per-view projection and eye position through Native/Rich MToon, Standard,
skinning/morph and outline shaders, and preserving per-view visibility and
transparent ordering. It is not implemented or advertised as completed here.

The Go example still shades 4,186,116 source pixels, plus the display interlace
pass. If the measured bottleneck is repeated scene submission or vertex work,
removing composition and output passes may be insufficient for 60fps. No speedup
factor is assumed. Looking Glass remains optional and exempt from the normal
application's 60fps rendering requirement; the optimization does not lower the
configured view count, resolution, material quality or animation update rate.

## Verification

For the array/capture change, source-level checks cover the Bevy 0.20 camera
schedule, texture-format and shared-target contracts. Added Rust tests check
array descriptors without CPU pixel storage, one shared scratch/one array with
66 distinct layers and no quilt camera/sprites, camera ordering/output mode,
and output hiding. Existing optical projection/calibration tests remain.
**The new Rust tests have not been executed in the authoring environment.**
No Rust compiler, shader compiler or connected GPU was available there. Rust
build, rustfmt, Clippy, GPU output and FPS checks are still required on the normal
development machine; Python equation checks are not a substitute for them.

Historical verification on 2026-10-10 (Bevy 0.19, before both the 0.20 migration
and this array change; these results do not verify the new build):

- The original 12 Looking Glass tests passed, including optical calculations and
  auxiliary-window cleanup; formatting and application/desktop Clippy passed
  with existing unrelated warnings.
- A macOS release `.app` built and passed strict codesign verification. On Apple
  M4 / Metal, AvatarSample_A rendered into all 66 views with synthetic calibration;
  F10, F11, Escape and continued main-window rendering were checked.
- A focused Windows source check passed in a temporary manifest, but the full
  Windows application and real Looking Glass optical output were not verified.

Run on the existing development checkout, using its normal release configuration:

```sh
cargo fmt --all -- --check
cargo test -p vtuber-app --lib looking_glass
cargo build --release -p vtuber-desktop
cargo clippy -p vtuber-app -p vtuber-desktop --all-targets
```

Compare the same VRM, pose/tracking load, Native/Rich mode, window sizes,
Looking Glass grid/resolution and NDI state before and after the change. Allow
initial shader compilation to complete. Compare ordinary interlaced output
with ordinary interlaced output, not fullscreen output with a small F10 preview.
Record FPS/frame time and CPU/GPU utilization; the 27fps report alone does not
identify which side is limiting performance.

Check F10 first: every layer must update, view zero must be bottom-left, and
foreground/background objects must move oppositely around the focus plane.
Check Native/Rich outlines, transparent hair/eyelashes, image orientation,
color-channel registration and native-pixel alignment. Replace/unload the VRM,
move the camera, and close the auxiliary window with NDI also running. FPS
improvement and real-device optical equivalence remain to be measured.

## References (not bundled SDK code)

- Bridge SDK: https://lfdocs.lookingglassfactory.com/software/looking-glass-bridge-sdk
- Calibration-file instructions: https://lfdocs.lookingglassfactory.com/software/index/unity-plugin-4.0-alpha
- Raw calibration and quilt conventions: https://github.com/Looking-Glass/looking-glass-webxr/blob/b44ed27fea197f28fbe4cb669bc350d141a9f8b1/src/LookingGlassConfig.ts
- Bevy custom projections: https://github.com/bevyengine/bevy/blob/v0.20.0/examples/camera/custom_projection.rs
- Camera output mode: https://github.com/bevyengine/bevy/blob/v0.20.0/crates/bevy_camera/src/camera.rs
- Shared main texture allocation: https://github.com/bevyengine/bevy/blob/v0.20.0/crates/bevy_render/src/view/mod.rs
- Camera-driven schedule: https://github.com/bevyengine/bevy/blob/v0.20.0/crates/bevy_core_pipeline/src/schedule.rs

The Rust and WESL here implement the geometric/pixel-mapping equations directly;
no manufacturer SDK binaries or JavaScript source are redistributed.
