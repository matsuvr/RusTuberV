# Looking Glass output

Rust/Bevy 0.20 plus an embedded WESL interlacer. No Bridge DLL, C/C++, JavaScript,
texture-sharing API or CPU image readback is added. The application queries
Bridge's local HTTP API for device data and renders directly to the display.
The Windows GPU path
has been measured. Physical Portrait output and natural left/right viewpoint
changes were confirmed by its owner on 2026-10-11; other panels remain unverified.

## Independent plugin (2026-10-11)

The renderer, calibration parser, optics, device catalog and embedded WESL
live in [bevy-looking-glass-rs](https://github.com/matsuvr/bevy-looking-glass-rs),
crate `bevy_looking_glass` for Bevy 0.20. RusTuberV consumes a pinned Git revision.
The public `bridge::discover()` API was added on 2026-10-12; RusTuberV uses its
device data and typed errors. There is no second renderer or Bridge client copy
in this workspace.

`crates/vtuber-app/src/looking_glass/mod.rs` keeps application settings and file
I/O, the avatar-to-`ViewPose` adapter, UI status, shortcuts, and the existing
Core3d executor policy. The independent plugin never changes the shared executor
or the host's other cameras/windows. RusTuberV explicitly selects its existing
layers, exposure, direct draws and single light cluster through plugin settings.
See the plugin README for the model-free example, scheduling hooks and packaging.
The historical results below describe their original builds; extraction-specific
validation is recorded separately.


### Extraction validation

- Independent repository: 21 tests passed; all-target Clippy with warnings denied,
  formatting, package file listing and `cargo publish --dry-run` passed. The
  dry-run compiled the extracted archive using only its normal dependencies.
- A separate model-free consumer outside the RusTuberV workspace rendered quilt,
  moved-camera and interlaced screenshots with ordinary registry Bevy (no window
  patch). `pose: None` produced all-black RGB pixels; stop and exit completed with
  code 0. Calibration was synthetic. Unpatched Bevy logged late window-close events.
- RusTuberV's three CLI/settings tests passed during the workspace extraction.
  The final pinned Git dependency passed the default-feature release desktop build
  and all-target application/desktop Clippy. Existing unrelated `chunks_exact`
  warnings remain; formatting and whitespace checks passed.
- The WESL interlacer is byte-for-byte unchanged from the source implementation.

Fresh Windows / i9-13900 / RTX 4090 / Vulkan presentation measurements used the
preserved pre-extraction executable and the final Git-dependency release build.
Both used Sapphy, 66 views at 186 x 341, a 720 x 1280 output, VSync, affinity
`0x15`, and the same synthetic calibration. Camera/inference and NDI output were
not started. Each capture ran for 30 seconds after 20 seconds of warmup, with no
build running. Values show the slower swapchain and worse p95 of the two windows.

| Build | Display fps | Display p95 ms | Mean CPU cores | Working set GiB |
|---|---:|---:|---:|---:|
| Preserved executable | 52.25 | 33.35 | 1.69 | 1.62 |
| External plugin | 57.83 | 16.70 | 1.85 | 1.56 |

Both applications exited with code 0. This single pair did not show a throughput
regression; it does not establish a performance improvement or a 60fps guarantee.
The measurements are OS presentation events on a desktop, not optical validation.
Raw CSVs, logs, hashes and summaries are in `target/looking-glass-extraction/`
(local, ignored). Pre-extraction binary SHA256: `d700c54e5342cd3dc1601741d2ab0f206299962b5a73650c4f309d0e24a7ec15`;
final binary SHA256: `dc5039ab47e218a84e3795448008de2e32a3091eeb0e46bdb538adf2ed48246f`.
Physical optics were not checked during extraction; the later Portrait check is
recorded below. macOS execution of the extracted crate, live camera/inference
acceptance and simultaneous NDI output remain unrun.

## Settings UI and device presets (2026-10-11)

Open **Settings → Looking Glass出力**, directly below NDI出力 at the same
navigation level. With Looking Glass Bridge installed and running on Windows or
macOS, the page automatically queries connected light-field displays. Select the
physical unit from the list and click **出力開始**. One connected unit is selected
automatically. No calibration filename, export command or file picker is needed.
On Portrait, use Desktop Mode with the manufacturer's video and USB/power
connections. **再検出** refreshes the list after changing connections.
Move the output window onto the panel and press F11. Use an extended desktop
at the reported native resolution; a scaled window is only a preview.
F10 previews the quilt for light field displays. **出力停止**, Escape, or the
window close button releases the output; it can be started again immediately.

Only the physical unit's serial (or the HLD model identifier) is saved in the
application settings on successful start. Output does not start automatically next launch. Legacy
saved calibration paths are ignored and removed on the next successful save.
Start queries Bridge again, so the current unit's calibration is always used.
An unreachable Bridge displays **Bridgeが見つかりません**. A reachable Bridge with
no light-field unit displays **Looking Glassが見つかりません**. Acquisition runs on
a worker; it does not block the UI or rendering. Stop cancels a pending start.

The application uses Bridge's raw optical calibration, native screen size and
`defaultQuilt` geometry. Model names are not an allowlist: a model absent from
the bundled catalog uses the same validated data path. Optical cell tables are
retained. Unknown optical modes, missing calibration/quilt or invalid geometry
are reported without inventing another unit's lens values or a quilt preset.
Existing single-view HLD presets remain selectable; they need no lenticular
calibration and do not use the multiview Bridge acquisition path.

### Factory calibration and optional diagnostic export

Looking Glass Portrait accepts HDMI as a monitor, but its light field optics
need the unit's factory calibration to interlace the views correctly. This is
**not user-performed calibration**. The official
[Bridge SDK](https://lfdocs.lookingglassfactory.com/software/looking-glass-bridge-sdk)
describes optical parameters as unique to each display. The
[Core SDK header](https://github.com/Looking-Glass/LookingGlassCoreSDK/blob/master/HoloPlayCore/include/HoloPlayCore.h)
also documents that calibration is transmitted over USB.

`device.json` is this project's chosen export filename, not a mandatory official
SDK filename. Its contents are the raw `visual.json` object. The embedded
`devices.json` catalog is separate and contains model presets, not factory lens
values for a physical unit.

Connect the manufacturer's video and data-capable USB connections, install and
start [Looking Glass Bridge](https://lookingglassfactory.com/software/looking-glass-bridge)
2.2 or newer, and confirm it detects the unit. On Portrait, a USB power adapter
alone cannot supply calibration to the PC; follow the manufacturer's
[Portrait setup](https://lfdocs.lookingglassfactory.com/getting-started/portrait/get-started-with-looking-glass-portrait).

The application and optional diagnostic exporter call the public pure Rust API
`bevy_looking_glass::bridge::discover()` from `bevy-looking-glass-rs`. Discovery
returns a device list, or `bridge::Error::Unavailable` when Bridge cannot be
reached; RusTuberV supplies the Japanese UI message. The protocol implementation
lives entirely in that independent crate. It follows the official Bridge.js protocol:
HTTP PUT to `http://localhost:33334/`, `enter_orchestration`,
`available_output_devices`, then `exit_orchestration`. It retains the returned
`calibration.value` JSON string unchanged, preserving the raw `{ "value": ... }`
fields and `subpixelCells`/`CellPatternMode`. It does not use Bridge.js's flattened
calibration or the native SDK's normalized shader uniforms. The renderer receives
parsed values in memory and calls no Bridge DLL. The HTTP protocol and worker
are the same on Windows and macOS. See the official
[client](https://github.com/Looking-Glass/Bridge.js/blob/main/src/library/client/BridgeClient.ts),
[HTTP endpoints](https://github.com/Looking-Glass/Bridge.js/blob/main/src/library/components/endpoints.ts)
and [calibration parser](https://github.com/Looking-Glass/Bridge.js/blob/main/src/library/components/displays.ts).
The normal UI creates no calibration file. Developers can optionally export one
for diagnostics or the advanced CLI below:

```sh
cargo run -p looking-glass-device --
cargo run -p looking-glass-device -- /path/to/device.json --serial YOUR_DEVICE_SERIAL
```

The default diagnostic path is `run/looking-glass/device.json` (local, ignored).
No file is written if Bridge or the selected calibration is unavailable. USB
`LKG_calibration/visual.json`, where exposed by the unit, can also be inspected
for diagnostics. These files are not part of the normal settings workflow.

Initial export validation on 2026-10-11: local tests exercise the official HTTP envelopes,
byte-preserving export, serial selection, and session exit with no device found.
The no-Bridge path was checked on this PC while Portrait was disconnected. The
later connected-unit export and optical check are recorded below.

The plugin's [`src/devices.json`](https://github.com/matsuvr/bevy-looking-glass-rs/blob/main/src/devices.json) is the embedded catalog. It
records panel size/aspect, output format, recommended quilt geometry, HLD safe
area, legacy status, research date and source URLs. All light field presets
share the view-array/off-axis/interlacing code. Projection uses the physical
display aspect, not the stretched tile aspect. The 65-inch reference quilt
has unused edge pixels: tiles are 1024 × 910, using 8192 × 8190 pixels.

Catalog coverage:

- Current light field lineup: Go, 16-inch landscape/portrait and 27-inch
  landscape/portrait.
- Existing-owner presets: Portrait, original 8.9-inch, 15.6-inch/Pro, legacy 16-inch,
  32-inch light field landscape/portrait, legacy 32-inch/8K and 65-inch.
- HLD 16/27/86-inch: a single portrait view with a white matte and documented
  safe margins. No lenticular calibration or quilt is applied. The same
  renderer uses one array layer. Frame the avatar inside the physical alcove;
  this app does not render the alcove or add a shadow-catcher scene.
  The store lists 86-inch for October shipping.
- musubi is listed with live output disabled: the manufacturer explicitly
  says it cannot serve as a secondary screen.

The current store lineup was checked separately from the documentation's
historical device list. These presets implement specifications; they are
**not evidence of optical validation on every model**. No physical panel or
its calibration was available during the initial preset implementation.

Official sources (checked 2026-10-11):

- [Current lineup](https://lookingglassfactory.com/) and
  [shipping / Portrait discontinued](https://looking-glass.helpscoutdocs.com/article/236-when-do-looking-glass-displays-ship).
- [Quilt specifications](https://lfdocs.lookingglassfactory.com/keyconcepts/quilts).
- [Official WebXR 15.6-inch quilt preset](https://github.com/Looking-Glass/looking-glass-webxr/blob/main/src/LookingGlassConfig.ts)
  uses 5 × 9 at 4096 square (4095 square used by integer-sized tiles).
- [Native resolutions and orientation](https://lfdocs.lookingglassfactory.com/software/looking-glass-bridge/display-settings-on-windows).
- [Portrait setup](https://lfdocs.lookingglassfactory.com/getting-started/portrait/get-started-with-looking-glass-portrait).
- [HLD desktop output](https://hlddocs.lookingglassfactory.com/content-pipelines/creating-real-time-apps)
  and [safe areas](https://hlddocs.lookingglassfactory.com/resources/media-specs-and-encoding).
- [musubi input limitation](https://checkout.lookingglassfactory.com/products/musubi).

### Windows settings smoke check (2026-10-11)

The release application was exercised with Sapphy.vrm and synthetic Portrait
calibration on an ordinary Windows monitor. The sidebar placement, model
dropdown, native JSON picker, 48-view quilt, interlaced output, settings stop,
restart without moving the camera, F11 fullscreen and Escape stop worked.
Switching to HLD 16 produced one portrait view with a white matte; closing the
main application also closed its output window. No shader error or panic was
logged. Bevy logged late window events during close, and gilrs logged its
existing controller-mapping warning. The original user settings were restored
byte-for-byte after this check.

Local validation passed: `cargo build --release -p vtuber-desktop -j 1`,
`cargo test --release -p vtuber-app --lib -j 1 -- --test-threads=1`
(323 tests), `cargo clippy -p vtuber-app -p vtuber-desktop --all-targets -j 1`,
formatting and whitespace checks. Clippy still reports pre-existing
`chunks_exact` suggestions outside this change.

This initial check covered application behavior with synthetic data. The later
Portrait check below used the actual unit's factory calibration. Other panels
and macOS execution of these presets remain unverified. Synthetic lens values
must not be used on a real display.

### Physical Portrait check (2026-10-11)

The connected unit was `LKG-PORT-01742` (display/storage name `LKG-P01742`).
`cargo run -p looking-glass-device --` successfully acquired its raw factory
calibration from the installed, running Bridge and wrote
`run/looking-glass/device.json`. Its eleven checked lens/display fields matched
`F:\LKG_calibration\visual.json` exactly. Bridge's JSON formatting differed from
the USB file; its optical values did not. The user did not perform calibration.

The existing final Git-dependency release executable above rendered Sapphy on
the physical Portrait. The output window was confirmed at desktop origin
`(5760, 0)` with a native `1536 x 2048` framebuffer and no window decorations.
The UI preset and Bridge both reported an 8 x 6 quilt: 48 views, `3360 x 3360`.
The CLI trial used 420 x 560 source views and the same grid. The owner confirmed
that the image was stereoscopic and that left/right viewpoint changes were
natural. This is owner-reported optical evidence, separate from screenshots.

F10 displayed the 48-view quilt and returned to interlaced output. Escape closed
only the auxiliary window. Choosing the Bridge-exported `device.json` in the
settings UI and starting again succeeded, saved the Portrait selection and path,
and returned to native fullscreen. The output was left running. No panic or
shader error was logged; the existing gilrs mapping warning and late Bevy window
events during auxiliary closure remained.

A 30-second PresentMon capture at native output resolution with process affinity
`0x15` measured the two swapchains at 55.50 and 55.56 display fps, both with a
33.34 ms p95 display interval. Mean process CPU use was 1.62 cores and working
set was 1.34 GiB. Other desktop applications remained open. Camera/inference and
NDI were off. This short measurement does not establish a 60fps guarantee.
Looking Glass output is optional and exempt from that target.

Local, ignored evidence is in `run/looking-glass/verification.json`,
`usb-bridge-comparison.json`, `bridge-devices.json`, the application logs and
`portrait-native-measurement/`. The tested executable predates the export tool
and calibration-label wording changes; it already contains the renderer and
settings behavior exercised here.

### Automatic Bridge acquisition validation (2026-10-11)

The new Windows release was launched with Sapphy and no `--looking-glass`
argument. The previous `run/looking-glass/device.json` was temporarily renamed;
the old saved path therefore pointed to a missing file. Opening the settings
page automatically detected `LKG-PORT-01742`, selected it, and displayed its
native 1536 x 2048 size and Bridge's 3360-square, 8 x 6 quilt.

Start acquired calibration from Bridge and rendered successfully while the JSON
file remained absent. F10 showed all 48 views and returned to interlaced output.
UI stop left the main avatar preview working; refresh retained the selected
unit; start again succeeded through a fresh acquisition. The restarted output
was left in native 1536 x 2048 fullscreen on Portrait at desktop `(5760, 0)`.
The saved settings contained only its serial in `[looking_glass]`; all other
preferences were preserved. The optional diagnostic JSON was restored afterward.
No panic or shader error appeared. Logs contain the existing gilrs mapping and
force-feedback warnings and a late Bevy window event during stop.

- Shared Bridge client: four tests passed, covering the official HTTP envelopes,
  session release on parse failure, raw optical-cell preservation, unknown model
  names, multiple serials and the exact unavailable-Bridge message.
- Application: seven focused tests passed, including uncatalogued models,
  landscape/portrait calibration, cell modes 1–4, unused quilt edge pixels,
  serial selection after reconnect, cancellation during acquisition and legacy
  settings migration. These synthetic tests establish data handling, not optics
  on those other models.
- Shared client/exporter all-target Clippy with warnings denied, app/desktop
  all-target Clippy, formatting, whitespace checks and the Windows release build
  passed. The existing unrelated Clippy suggestions remain.
- The original shared client/exporter passed an `aarch64-apple-darwin` compile
  check from Windows before the client was moved into the public plugin crate.
  This checked common acquisition code for Apple Silicon; full Mac app linking
  and Mac hardware execution were not performed. Other physical Looking Glass models, live
  inference/camera and simultaneous NDI were not exercised in this trial.

Local evidence and the preserved settings are in
`run/looking-glass/bridge-automatic/`. Tested executable SHA256:
`3096ab8e5cbb97c43a1215b9bbc334bf5250592c450878fb53601b50645ff8ed`.
The earlier optical confirmation and performance capture above belong to the
earlier release; there was no new owner optical report or presentation timing
capture for this build.

### Public Bridge API migration validation (2026-10-12)

Bridge detection, HTTP sessions and device parsing now live in
`bevy_looking_glass::bridge`, at pinned commit
`c7eb5eec90d3b565634e0dc3c16eb0ed18f10896`. The former application-local
client crate was removed. RusTuberV and its optional exporter consume the public
API; background scheduling, UI messages and saved selections stay in the app.

- The independent crate passed all 25 library tests and all-target Clippy with
  warnings denied. Formatting, whitespace and package-content checks passed.
- RusTuberV passed eight focused release tests against the relocated source,
  including consumption of `bridge::Error::Unavailable` and the required
  `Bridgeが見つかりません` message. App, desktop and exporter all-target Clippy
  then passed against the published Git commit, with existing unrelated warnings.
- The exact upstream HTTP module retrieved the connected Portrait's serial,
  raw 410-byte factory calibration and 3360-square, 8 x 6 quilt. It also passed
  an isolated `aarch64-apple-darwin` compile check.
- The app library and diagnostic exporter release build passed using the pinned
  Git dependency. That exporter acquired `LKG-PORT-01742`'s 1536 x 2048
  calibration from Bridge; its factory data matched the previous working export.
  No new desktop executable or fullscreen optical test was run for this refactor.
- A full-crate cross-check from Windows stopped at Bevy's `blake3` build script
  because the Mac C compiler is unavailable. Mac execution and other physical
  models remain unverified. The renderer was unchanged; the fullscreen UI and
  optical evidence above belong to their recorded builds.

## CLI custom layout (advanced, optional)

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
restarting for CLI custom layouts. The settings UI can switch to a catalog
preset during the same session.

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
Without `--looking-glass`, dormant output systems are installed for the settings
UI, but no output cameras, windows or textures are allocated.

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
closes the auxiliary window too. The normal settings UI discovers devices and
retrieves calibration through Bridge. There is no automatic hot-plug restart;
refresh the list and start output again after changing devices.

## Rendering and scope

The existing live avatar world is rendered from parallel, horizontally translated
cameras. An asymmetric perspective projection cancels disparity at the camera
control's focus plane. Orbit/pan/dolly in the existing preview also move the
multiview rig; the avatar, tracking and animation are not duplicated.

The cameras share one small logical image target and Bevy's main/depth work
textures, with MSAA off. At the end of each view's Core3d schedule, its finished
main texture is copied into the corresponding layer of a 2D texture array.
Distinct camera orders ensure the copy finishes before the next view reuses
the work textures. `CameraOutputMode::Skip` removes the final raster blit for
all but the first view. Bevy 0.20 omits the target attachment when every camera
skips output, so the first view keeps one blit to establish the shared target.
The capture runs after post-processing, upscaling and the MToon
outline passes. The interlacer selects array layers for the RGB subpixels directly.
There is no quilt image, quilt assembly camera, or tile sprite draw. F10 displays
a virtual quilt from the same array. All image work stays on the GPU. Output excludes the floor
and egui and does not change the preview or NDI paths. No custom changes to the
avatar's MToon/Standard/unlit materials or scene lighting are made.
The multiview cameras use the same fixed SDR exposure, disabled tonemapping and
disabled dithering as the existing avatar preview and NDI camera.

The view cameras use one lighting cluster for the avatar's small light rig and
direct batched draws (`NoIndirectDrawing`). CPU visibility culling still runs;
per-view GPU indirect-argument construction is unnecessary for these small
scenes. Bevy's material, skinning, morph, transparency-sorting and MToon outline
paths remain in use. No cross-view instancing or view synthesis is introduced.
When light field output is enabled, `Core3d` runs on the render thread using
`SingleThreadedExecutor`, avoiding repeated dispatch to the desktop's single
compute worker for every view. The ordinary path with the option absent is
unchanged; stopping restores the normal executor. Closing the output drops
its image/material handles; unloading an
avatar hides the interlacer mesh, leaving the output black until the rig is ready.

Array layer zero is view zero, bottom-left in the virtual quilt. Lens coordinates
are bottom-left while Bevy image UVs are top-left. Clamp-to-edge filtering within
each array layer prevents borrowing pixels from a neighboring view. The shader handles classic RGB
stripes and calibration cell patterns 0–4, including per-cell RGB offsets.
This does **not** constitute tested support for every Looking Glass model.
Nearest-view selection is used; no view interpolation or depth-based view
synthesis is implemented. Optical transforms are separate from image orientation.

The Go example shades 4,186,116 source pixels and one native-resolution interlace
pass. The old quilt assembly additionally read and wrote the 4,186,116-pixel
quilt and required 66 sprite draws; those operations have been removed.
Of the 66 per-view output raster blits, 65 are removed; finished views are
transferred by 66 GPU texture copies. This still transfers pixels; it is not a
zero-copy path or cross-view scene instancing.
Low pixel count does not remove the
66-view draw-call, vertex/skinning or culling cost. Measure with the actual
VRM and GPU before drawing a frame-rate conclusion. The base application's
single-view performance target is not a measured guarantee for this experiment.
Looking Glass is optional and exempt from the 60fps rendering requirement;
the normal application still targets rendering at 60fps and inference at 30Hz
on three CPU cores with this output disabled.

## Verification

### Initial Windows optimization measurement (2026-10-11)

Release build, Bevy 0.20, Windows, Core i9-13900, RTX 4090 / NVIDIA 617.14 /
Vulkan. Process affinity `0x15` selects logical CPUs 0/2/4; the Bevy pool budget
is also three threads. Sapphy, 66 views at 186 x 341, 720 x 1280 auxiliary
window, VSync on, camera and NDI off. Synthetic calibration was used for this
GPU workload, not for optical validation. The model's existing idle animation
and spring/skin rendering remained active. Each interval below is 30 seconds,
after loading and shader warmup, with no build running during capture.

PresentMon 2.6 records both windows separately; they have essentially the same
cadence. Intermediate rows use one release executable with temporary startup
switches. Those switches are removed from the final implementation.

| Change (cumulative only where stated) | OS display fps |
|---|---:|
| Original, two measurements | 19.50 / 20.15 |
| Texture array; no quilt assembly | 20.30 |
| Array + serial Core3d scheduling | 23.58 |
| Array + direct draws | 35.35 |
| Array + direct draws + serial scheduling | 44.30 |
| Array + direct draws + serial scheduling + one light cluster | 59.80 |
| Direct-to-array-layer implementation, two measurements | 60.00 / 60.00 |

The array alone did not resolve the frame-rate bottleneck. The useful gains
came from reducing work repeated for each of the 66 views. GPU ETW attribution
is retained in the raw data, but is not an isolated measurement of each pass.
The final build reaches the display's 60Hz VSync ceiling, about three times the
original cadence. The worst of the two windows' p95 display intervals was
16.68ms in both final captures, versus 83.36ms / 66.68ms originally. Resident
working set was about 1.58GiB versus 1.82GiB. Mean CPU use increased from
1.30–1.33 to 1.50–1.52 cores while delivering about three times as many frames.
This does not establish 60fps on a native-resolution physical panel, with live
inference/NDI, on macOS, or on other GPUs.

A separate size stress test requested a 1440 x 2560 auxiliary window, keeping
the same 66 source views. It measured 20.97fps originally and 60.00fps with the
final build (30 seconds each). For this test only, the synthetic calibration
used `screenW=2880`, `screenH=5120`, `DPI=600`, because the prototype initially
requests a half-size window. The window extended below the desktop's visible
area. This verifies the enlarged desktop workload, not a correctly calibrated
native-panel presentation.

The final Windows release build and the 12 focused Looking Glass tests passed.
`cargo fmt --all -- --check`, `git diff --check` and release Clippy for
`vtuber-app` / `vtuber-desktop` with all targets passed; existing unrelated
`chunks_exact` warnings remain. On this GPU the final build rendered the
interlaced output and F10 virtual quilt, completed the F11 fullscreen round
trip, and closed only the auxiliary window on Escape. The remaining preview
was measured at 60.00fps for 15 seconds. Rich lighting could be enabled and
disabled, and unloading the avatar turned the auxiliary output black.
Closing the main window also closed the auxiliary window. These are desktop
rendering checks using synthetic calibration, not physical-panel verification.

Reproduce with the same release executable, model, calibration, window sizes
and affinity. After the avatar is visible and shaders have warmed up:

```powershell
.\tools\measure-looking-glass.ps1 -ProcessId <RusTuberV-process-id> `
  -PresentMon <path-to-PresentMon-2.6.0-x64.exe> `
  -Output data/performance/<new-run-directory> -Seconds 30
```

The script writes raw per-swapchain events, executable SHA256, process affinity,
CPU usage, working set, FPS and frame-interval percentiles. Local artifacts for
this run are in `data/performance/20261011-looking-glass/` (ignored by Git).

### PR #271 integration (2026-10-11)

The shared-scratch capture from
[PR #271](https://github.com/matsuvr/RusTuberV/pull/271), commit
`647986da8bf0cf5bfcb232a9b35eb089a4cf92da`, is incorporated with the measured
direct-draw, single-cluster and serial-schedule settings above. Image assets
now own the array and scratch target, removing the bespoke raw-texture
material bindings and manual texture-view registration/cleanup.

Two issues in the unvalidated PR were corrected locally:

- The Bevy 0.20 `ExtractComponent` derive requires `#[extract_app(RenderApp)]`.
- All views using `CameraOutputMode::Skip` produced a black window. Runtime
  diagnosis found 66 extracted cameras but zero `ViewTarget`s: Bevy omits
  attachments for skip-only targets and then omits clearing view targets
  without an attachment. Keeping the first camera's output blit fixes this;
  the other 65 blits are skipped. The camera-order regression test covers this
  requirement. No engine patch or extra attachment-management path is added.

With the same 66-view desktop workload, a fresh 30-second comparison measured:

| Metric | Before PR integration | Integrated build |
|---|---:|---:|
| OS display fps (slower window) | 59.93 | 59.97 |
| Display interval p95 (slower window), ms | 16.6772 | 16.6775 |
| Process working set, GiB | 1.650 | 1.560 |
| Mean CPU cores | 1.473 | 1.447 |
| Process dedicated GPU memory, MiB | 968.86 | 962.80 |

Both builds remain at approximately the 60Hz VSync limit; this is not evidence
of additional uncapped frame-rate improvement. These are one comparison pair
and process totals, not isolated allocation/pass measurements. GPU memory is
the Windows `GPU Process Memory` counter, summed over the process's adapter
instances and sampled five times after each presentation capture. No build
was running during these captures. Rich mode, camera input and NDI were off.

The integrated release build, all 16 Looking Glass tests, formatting, whitespace
checks and release Clippy for app/desktop with all targets passed (existing
unrelated Clippy warnings remain). Desktop
checks covered the interlacer, all 66 distinct quilt views, F11 round trip,
Rich materials/outline display, unload-to-black and Escape closing only the
auxiliary output. Pixel-for-pixel image equivalence and physical-panel optics
remain unverified. The local raw captures, hashes, provenance and report are
in `data/performance/20261011-pr271/` (ignored by Git). The earlier measurements
above refer to the direct-to-array-layer implementation before this integration.

### Earlier prototype checks

Review verification on 2026-10-10 (Bevy 0.19, before the 0.20 migration;
these results do not verify the migrated build):

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
at the same time, and auxiliary-window closure. Other Looking Glass hardware,
NDI simultaneous output and macOS performance remain pending; the earlier
Windows timing measurements above concern their specified desktop workloads.

## References (not bundled SDK code)

- Bridge SDK: https://lfdocs.lookingglassfactory.com/software/looking-glass-bridge-sdk
- Calibration-file instructions: https://lfdocs.lookingglassfactory.com/software/index/unity-plugin-4.0-alpha
- Raw calibration and quilt conventions: https://github.com/Looking-Glass/looking-glass-webxr/blob/b44ed27fea197f28fbe4cb669bc350d141a9f8b1/src/LookingGlassConfig.ts
- Bevy custom projections: https://github.com/bevyengine/bevy/blob/v0.20.0/examples/camera/custom_projection.rs

The Rust and WESL here implement the geometric/pixel-mapping equations directly;
no manufacturer SDK binaries or JavaScript source are redistributed.
