# RusTuberV

[日本語](README.md) · **English** · [简体中文](README.zh-CN.md) · [한국어](README.ko.md)

RusTuberV is a webcam-driven VRM avatar application built with Rust and Bevy. It applies MediaPipe face, pose, and hand inference to an avatar and provides a preview, transparent video output over NDI®, and Looking Glass output.

## Download

To use the app, download the Windows x64 release or the Apple Silicon Mac test build from [GitHub Releases](https://github.com/matsuvr/RusTuberV/releases/latest). Distribution packages include the inference models and required runtimes. The development setup below is needed only when building from source.

On Windows, extract the entire ZIP and double-click `RusTuberV.exe`. Keep all bundled files together. The Mac build requires macOS 14 or later and is intended for users who can diagnose and resolve technical problems themselves. See the [release notes](https://github.com/matsuvr/RusTuberV/releases/latest) for Mac limitations, tested environments, and first-launch instructions.

## User guide

**[English user guide](manual/en/README.md)**

The illustrated guide covers loading an avatar, selecting a camera, adjusting framing and expressions, and sending video to OBS.

## Features

- VRM 0.x / 1.0 import
- Webcam tracking of the face, blinks, gaze, mouth, upper body, arms, and hands
- Avatar control constrained by bone lengths, skeletal hierarchy, joint axes, degrees of freedom, and ranges of motion
- Neutral-pose and arm calibration, mirroring, and framing controls
- Keyboard expression switching and per-model key bindings
- Enhanced rendering and per-model settings
- Avatar video output over NDI
- Looking Glass device discovery and output through Looking Glass Bridge
- Japanese, English, Simplified Chinese, and Korean UI

## Build requirements

| Component | Requirement |
| --- | --- |
| Rust | 1.99 or later; `rust-toolchain.toml` selects stable |
| Windows | MSYS2 CLANG64 Rust / Clang / LLD, x86_64-pc-windows-gnullvm target |
| macOS | Apple Silicon and Xcode Command Line Tools |
| Builds with NDI output | NDI Standard SDK and Clang / libclang |
| Rendering | GPU and drivers compatible with Bevy's rendering backend |
| Tracking | Webcam and OS camera permission |

Camera capture uses Media Foundation on Windows and AVFoundation on macOS.

The desktop build enables `ndi-output` by default. To build without the NDI SDK dependency, use `--no-default-features` as shown below.

## Get the source

```sh
git clone https://github.com/matsuvr/RusTuberV.git
cd RusTuberV
```

Dependencies are pinned in `Cargo.lock`. Use `--locked` when building.

## Prepare the inference models

Place these three files in `assets/models/`. The model binaries are excluded from Git, but they are embedded in the executable at build time and must be present before building.

| File | Purpose |
| --- | --- |
| `face_landmarker.task` | Face, expression, and gaze inference |
| `pose_landmarker_full.task` | Upper-body and arm-pose inference |
| `hand_landmarker.task` | Hand and finger inference |

On Windows, run the following in PowerShell from the repository root.

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "assets/models/face_landmarker.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "assets/models/pose_landmarker_full.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "assets/models/hand_landmarker.task"
```

On macOS, run:

```sh
curl -fL -o assets/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -fL -o assets/models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
curl -fL -o assets/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Approved SHA-256 hashes and download sources are recorded in the [model manifest](assets/models/manifest.toml). The application checks each model's hash when loading it. If an upstream `latest` download changes and its hash no longer matches, obtain the version corresponding to the manifest.

For source builds without a bundled native library, the MediaPipe native library is downloaded from the official PyPI wheel and cached when inference starts for the first time. This initial download requires a network connection. To supply the library yourself, set `MEDIAPIPE_LIB` to the path of a compatible MediaPipe 0.10.35 native library. See the [mediapipe-rs README](vendor/mediapipe-rs/README.md) for details.

## Build on Windows

Install the packages listed in the [LLVM setup and MSVC rollback guide](tools/WINDOWS-LLVM.md).
Use `tools/cargo-windows.ps1` for Windows builds. It uses MSYS2 Rust without
Visual Studio Build Tools; plain `cargo` retains the existing rustup environment.

### Default build with NDI output

Install the Windows Standard SDK from the [official NDI SDK download page](https://ndi.video/for-developers/ndi-sdk/). NDI binding generation uses Clang / libclang.

This example assumes the standard NDI SDK location and MSYS2 at `C:\msys64`.

```powershell
$env:NDI_SDK_DIR = "C:\Program Files\NDI\NDI 6 SDK"
.\tools\cargo-windows.ps1 build --release --locked --package vtuber-desktop -j 1
.\tools\cargo-windows.ps1 run --release --locked --package vtuber-desktop -j 1
```

The build script copies the NDI runtime DLL matching the SDK import library beside the executable. If the DLL is stored outside the SDK, set `NDI_RUNTIME_DLL` to its path before building.

### Build without NDI output

```powershell
.\tools\cargo-windows.ps1 build --release --locked --package vtuber-desktop --no-default-features -j 1
.\tools\cargo-windows.ps1 run --release --locked --package vtuber-desktop --no-default-features -j 1
```

This configuration does not require the NDI SDK or libclang for NDI bindings. The inference models and MediaPipe native library are still required.

## Build on macOS

For NDI output, install the macOS Standard SDK and specify its location.

```sh
export NDI_SDK_DIR="/Library/NDI SDK for Apple"
cargo build --release --locked -p vtuber-desktop -j 1
./target/release/RusTuberV
```

Ensure that Clang / libclang is available to the build environment. If the default search cannot find it, set `LIBCLANG_PATH` to the library directory.

To create a `.app` for local use, run these commands in the same environment:

```sh
bash tools/build-macos.sh
open target/release/RusTuberV.app
```

The [build script](tools/build-macos.sh) installs the icon, model manifest, and tracking settings and applies an ad hoc signature. The NDI library is loaded from the installed SDK. This `.app` does not include NDI runtime redistribution packaging or Apple notarization.

Without NDI output, run:

```sh
cargo build --release --locked -p vtuber-desktop --no-default-features -j 1
./target/release/RusTuberV
```

## Startup options

You can specify a VRM file at startup.

```powershell
.\target\release\RusTuberV.exe --model "C:\Avatars\avatar.vrm"
```

```sh
./target/release/RusTuberV --model /path/to/avatar.vrm
```

If omitted, load a model with **Choose VRM…** or by dropping a file into the application. See the [user guide](manual/en/README.md) for camera selection and streaming setup.

## Project structure

| Path | Responsibility |
| --- | --- |
| `apps/desktop` | Desktop executable and OS-specific startup |
| `crates/vtuber-app` | UI, settings, model import, and orchestration |
| `crates/vtuber-core` | Shared data types and frame exchange |
| `crates/vtuber-camera` | Camera capture |
| `crates/vtuber-inference` | Inference and MediaPipe native boundary |
| `crates/vtuber-tracking` | Observation processing and conversion to avatar controls |
| `crates/vtuber-avatar` | VRM rendering, expressions, skeletons, and joints |
| `crates/vtuber-ndi` | NDI sender native boundary |
| `crates/vtuber-platform` | OS-specific functionality |
| `tools/xtask` | Development commands and distribution packaging |
| `manual/ja` | Default Japanese user guide |

Application Rust code uses `#![forbid(unsafe_code)]`. Required native operations are isolated at binding boundaries. See [AGENTS.md](AGENTS.md) for development constraints.

## Packaging

See the [packaging instructions](tools/RELEASING.md) (Japanese) for the Windows build-and-package workflow and the Mac runtime bundling, signing, ZIP creation, and verification steps.

## License

The project's code is released under the [MIT License](LICENSE).

Dependencies, fonts, inference models, and reference data are covered by their respective licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The NDI SDK and runtime are separate components with their own license terms. Distributions containing the NDI runtime must include the license and notice documents from the SDK used and comply with the [NDI SDK use and redistribution terms](https://docs.ndi.video/all/developing-with-ndi/sdk/licensing). Windows packages can be created with `cargo xtask ndi package`.

Follow each rights holder's terms for avatar VRM files, textures, and streaming assets. The application's MIT License does not grant permission to use those assets.

NDI® is a registered trademark of Vizrt NDI AB. See the [official NDI website](https://ndi.video/) for NDI information.

## Disclaimer

This software is provided as is under the MIT License. Operation, inference and tracking accuracy, performance, and compatibility with particular hardware or streaming environments are not guaranteed.

The authors and copyright holders disclaim liability, to the extent set out in the MIT License, for damages, interrupted streaming or recording, loss of data, and other consequences of using the software.

Users are responsible for checking the rights to streaming and recording assets and whether their face or room appears in the broadcast.
