# Looking Glass output prototype

Rust/Bevy 0.19 plus an embedded WGSL interlacer. No Bridge DLL, C/C++, JavaScript,
texture-sharing API or CPU image readback is added. This is a direct-display
experiment, not an implementation of the Bridge SDK API. Hardware output and
frame rate have not yet been verified.

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

The resulting quilt is 2046 x 2046. A Portrait-shaped starting point is 8 columns,
6 rows, 210 x 280 per view. The grid/tile size is configurable, not inferred from
a device-name heuristic. Calibration paths are relative to the configuration
file; absolute paths also work. `depth_scale` multiplies the view-cone baseline:
0 removes parallax, 1 uses the calibrated view cone. Changing settings requires
restarting this prototype.

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

**F10** toggles the raw quilt for inspection. **Escape** closes this output while
it has focus. These keys act only on the auxiliary window. Closing it also
removes every associated camera and render target; closing the main window
closes the auxiliary window too. There is no automatic device detection,
hot-plug restart, calibration download, settings UI or fallback display mode.

## Rendering and scope

The existing live avatar world is rendered from parallel, horizontally translated
cameras. An asymmetric perspective projection cancels disparity at the camera
control's focus plane. Orbit/pan/dolly in the existing preview also move the
multiview rig; the avatar, tracking and animation are not duplicated.

Each view uses a small render-target image with MSAA off. A 2D camera assembles
the images into one quilt, and another applies calibrated RGB-subpixel selection
to the output window. All image work stays on the GPU. Output excludes the floor
and egui and does not change the preview or NDI paths. No custom changes to the
avatar's MToon/Standard/unlit materials or scene lighting are made.
The multiview cameras use the same fixed SDR exposure, disabled tonemapping and
disabled dithering as the existing avatar preview and NDI camera.

View zero is bottom-left in the quilt. Lens coordinates are bottom-left while
Bevy image UVs are top-left. Tile sampling stays half a texel inside the selected
tile to avoid borrowing a neighboring view. The shader handles classic RGB
stripes and calibration cell patterns 0–4, including per-cell RGB offsets.
This does **not** constitute tested support for every Looking Glass model.
Nearest-view selection is used; no view interpolation or depth-based view
synthesis is implemented. Optical transforms are separate from image orientation.

The Go example shades 4,186,116 source pixels, plus the quilt-assembly pass and
one native-resolution interlace pass. Low pixel count does not remove the
66-view draw-call, vertex/skinning, culling or shadow cost. Measure with the actual
VRM and GPU before drawing a frame-rate conclusion. The base application's
single-view performance target is not a measured guarantee for this experiment.

## Verification

Review verification on 2026-10-10:

- Rust tests cover argument opt-in, absent calibration, lens normalization, cell
  normalization, quilt order, focus-plane invariance, front/back disparity,
  rotated rigs, frustum corners and output entity cleanup: **12 passed**.
- Formatting and Clippy for the application and desktop, including all targets,
  passed. Existing unrelated warnings remain.
- The macOS release `.app` built with `tools/build-macos.sh` and passed
  `codesign --verify --deep --strict`.
- On an Apple M4 / Metal, `AvatarSample_A` rendered into all 66 views at
  186 x 341. The interlace shader, F10 quilt toggle, F11 fullscreen round trip,
  Escape closing only the output, continued main-window preview and application
  quit were checked. The application exited with code 0.
  This used **synthetic calibration on the internal LCD**, only to validate the
  GPU pipeline; it does not validate any panel's optical output.
- The unchanged Looking Glass source module and actual avatar crate passed a
  focused `x86_64-pc-windows-msvc` check in a temporary manifest with
  `blake3/pure` enabled to avoid the unavailable MSVC assembler. Shipping
  dependencies were not changed. The normal full Windows application check on
  this Mac was blocked by native dependencies requiring Windows build tools and
  SDK headers. Windows execution remains unverified.

```sh
cargo fmt --all -- --check
cargo test --locked --offline --release -p vtuber-app --lib looking_glass
cargo clippy --locked --offline --release -p vtuber-app -p vtuber-desktop --all-targets
bash tools/build-macos.sh
codesign --verify --deep --strict target/release/RusTuberV.app
```

On the actual display, check the quilt first: all tiles must update together,
view zero must be bottom-left, and a foreground/background object must move in
opposite directions around the focus plane. Then disable quilt preview and
check left/right parallax, upright image, color-channel registration and native
pixel alignment. Test VRM replacement/unload, main-window controls, NDI running
at the same time, and auxiliary-window closure. Measure the same scene with the
option absent and present. Looking Glass hardware, Windows execution, model
replacement/unload, NDI simultaneous output and comparative frame rates remain
pending.

## References (not bundled SDK code)

- Bridge SDK: https://lfdocs.lookingglassfactory.com/software/looking-glass-bridge-sdk
- Calibration-file instructions: https://lfdocs.lookingglassfactory.com/software/index/unity-plugin-4.0-alpha
- Raw calibration and quilt conventions: https://github.com/Looking-Glass/looking-glass-webxr/blob/b44ed27fea197f28fbe4cb669bc350d141a9f8b1/src/LookingGlassConfig.ts
- Bevy custom projections: https://github.com/bevyengine/bevy/blob/v0.19.0/examples/camera/custom_projection.rs

The Rust and WGSL here implement the geometric/pixel-mapping equations directly;
no manufacturer SDK binaries or JavaScript source are redistributed.
