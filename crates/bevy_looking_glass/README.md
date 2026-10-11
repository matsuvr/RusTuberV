# bevy_looking_glass

A small Bevy **0.20** plugin for direct Looking Glass display output. CPU code is
safe Rust; GPU code is embedded WESL. No Looking Glass Bridge SDK, C/C++ bindings,
DLL loading, JavaScript, or CPU image readback is added by this crate.

This does **not** replace manufacturer device setup, firmware, USB calibration
retrieval, or any manufacturer software needed to put a device into desktop mode.
The application supplies the physical unit's raw `visual.json` calibration bytes.

## Usage

```toml
[dependencies]
bevy = "0.20"
bevy_looking_glass = { git = "https://github.com/matsuvr/bevy-looking-glass-rs" }
```

```rust,no_run
use bevy::{prelude::*, camera::visibility::RenderLayers};
use bevy_looking_glass::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let calibration = parse_calibration(&std::fs::read("visual.json")?)?;
    let config = LookingGlassConfig::new(
        LookingGlassOutput::LightField {
            calibration,
            layout: ViewLayout::new(8, 6, 210, 280)?,
            depth_scale: 1.0,
        },
        RenderLayers::layer(0), // scene
        1,                     // reserved exclusively for the output mesh
    );
    let mut control = LookingGlassControl::default();
    control.start(config);
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LookingGlassPlugin)
        .insert_resource(control)
        .insert_resource(LookingGlassView {
            pose: Some(ViewPose {
                transform: Transform::from_xyz(0.0, 0.0, 5.0),
                focus_distance: 5.0,
                fov_y: 45_f32.to_radians(),
                near: 0.1,
                far: 100.0,
            }),
        })
        // Add your scene/animation systems here.
        .run();
    Ok(())
}
```

For a complete scene requiring no external model:

```sh
cargo run --example minimal -- /path/to/visual.json
```

The example places a red cube in front of the focus plane, a green sphere on it,
and a blue cube behind it. Arrow keys orbit/dolly. With the output window focused,
F10 toggles the quilt, F11 toggles fullscreen, and Escape stops output. Closing
the main window closes the example's output. These policies are example code,
not plugin behavior. The sample requires real calibration; synthetic calibration
is confined to tests and desktop pipeline validation.

## Integration contract

- Install after Bevy's render/asset/window plugins. Supply the window backend in
  your app (e.g. `bevy_winit` plus the appropriate Linux backend). Normal plugin
  dependencies enable only needed rendering APIs, not Bevy's complete default
  feature set. The example also enables PBR, Winit and tonemapping LUTs.
- One instance manages one output. `LookingGlassControl::start(config)` replaces
  it; `stop()` releases the window, cameras and owned asset handles. Invalid
  requests preserve existing output and report `LookingGlassStatus::error`.
  Inspect `status.window` for the output entity. Closing it releases its children.
- The host owns scene animation, files, dialogs, settings persistence, keyboard
  input and application exit. The plugin does not find or modify a primary window,
  other cameras, task pools or the shared Core3d executor.
- Write `LookingGlassView::pose` each time your camera changes. The transform is
  world space, with unit scale and normalized rotation; focus distance is positive,
  `0 < fov_y < pi`, `0 < near < far`. `pose: None` disables the scene cameras and
  hides old pixels. Panel aspect determines projection, not tile/preview aspect.
- `scene_layers` selects the scene. Reserve `display_layer` for the interlace mesh;
  it must not intersect scene layers or any other application's camera layers.
- `ViewRenderSettings` configures exposure, tonemapping, dithering, light clusters,
  direct/indirect draws and clear color on owned view cameras only. Output is SDR
  and MSAA is off. Enable Bevy LUT features for tonemappers that require them.
  HLD uses a white matte independently of the light-field clear color setting.
- `config.window` is a Bevy `Window`: set title, position, monitor/mode, resolution
  and presentation policy there. The constructor requests native physical pixels
  with scale factor 1. Windowed/scaled output is only a preview. For optics the
  actual framebuffer and panel must have one-to-one native pixel correspondence;
  use an extended desktop in the correct orientation, not display mirroring/scaling.
- Schedule request producers before `LookingGlassSystems::Control` in `Update`;
  read status after it. Publish camera poses before `LookingGlassSystems::SyncViews`
  in `PostUpdate` (the plugin reads after transform propagation).
- Custom render passes belong before `LookingGlassRenderSet::Capture` in the
  render sub-app's `Core3d` schedule. Capture already follows post-processing and
  upscaling. It must run once per view before the next camera reuses scratch.

## Rendering

Parallel off-axis cameras share one small render target and copy completed views
into a texture array. The first view retains Bevy 0.20's output blit to establish
the shared attachment; remaining cameras use `CameraOutputMode::Skip`. Interlacing
samples the array directly. Quilt preview uses the same array without a quilt
texture, assembly camera or tile sprites. View zero is bottom-left in a quilt.

Raw lens values are normalized on the CPU; RGB stripes and cell-pattern modes
0–4 use the embedded shader. No interpolation, view synthesis, cross-view GPU
instancing, device detection, Bridge integration or Bevy 0.19 compatibility is
included.

## Catalog and HLD

`device_presets()` reads `src/devices.json`. It preserves manufacturer specification
references and the research date. `DevicePreset::light_field_output(calibration)`
uses that preset's reference grid and rejects a calibration with different native
panel dimensions. Or use `ViewLayout` directly for custom view resolution.

Catalog entries cover Go, Portrait, original 8.9/15.6/Pro, current and legacy 16,
27, 32, 65-inch light-field variants, and HLD 16/27/86. These are **specification
presets, not a list of hardware-tested devices**. `hld_output()` uses one portrait
view and the catalog's safe area without any lens calibration. musubi is catalog
information only: it does not offer live desktop output.

## Validation scope

Extracted from [RusTuberV a198c1b](https://github.com/matsuvr/RusTuberV/tree/a198c1b9)
under MIT, preserving the array/copy/interlace path. Previous RusTuberV measurements
of roughly 60 fps concern a specified Windows desktop workload with synthetic
calibration. They do not guarantee plugin performance or physical-panel optical
correctness. Physical Looking Glass panels and all-device compatibility remain
unverified. See RusTuberV's [integration record](https://github.com/matsuvr/RusTuberV/blob/main/docs/looking-glass.md)
for the historical workload and limitations.

Use current stable Rust (minimum 1.99), Bevy 0.20, and normal Cargo checks:

```sh
cargo fmt --all -- --check
cargo test --lib -j 1
cargo clippy --all-targets -j 1
cargo package --list
cargo publish --dry-run
```

## License and provenance

MIT; see [LICENSE](LICENSE). Copyright notices from RusTuberV are retained.
The code directly implements geometry and pixel-mapping equations; manufacturer
SDK binaries and JavaScript source are not redistributed.

- [Raw calibration and quilt conventions](https://github.com/Looking-Glass/looking-glass-webxr/blob/b44ed27fea197f28fbe4cb669bc350d141a9f8b1/src/LookingGlassConfig.ts)
- [Quilt specifications](https://lfdocs.lookingglassfactory.com/keyconcepts/quilts)
- [Calibration file instructions](https://lfdocs.lookingglassfactory.com/software/index/unity-plugin-4.0-alpha)
- [Native panel resolutions/orientation](https://lfdocs.lookingglassfactory.com/software/looking-glass-bridge/display-settings-on-windows)
- [Bridge SDK scope](https://lfdocs.lookingglassfactory.com/software/looking-glass-bridge-sdk)
- [Bevy custom projections](https://github.com/bevyengine/bevy/blob/v0.20.0/examples/camera/custom_projection.rs)

Per-device specification sources are included in `src/devices.json`.
