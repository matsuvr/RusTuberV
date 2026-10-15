# Third-party notices

## Bundled UI fonts

The UI embeds the same OTF files on Windows and macOS, without loading system
fonts. Japanese and English prefer LINE Seed JP, Korean prefers LINE Seed KR,
and Chinese prefers Noto Sans CJK SC. All three remain available for mixed text.

- `assets/fonts/LINESeedJP_A_OTF_Rg.otf` and `assets/fonts/LINESeedKR-Rg.otf`:
  © LY Corporation. [LINE Seed](https://seed.line.me/) is licensed under
  SIL OFL 1.1; see `assets/fonts/LICENSE-LINESeed.txt`.
- `assets/fonts/NotoSansCJKsc-VF.otf`: © 2014-2021 Adobe
  (http://www.adobe.com/), with Reserved Font Name 'Source'.
  [Noto Sans CJK](https://github.com/notofonts/noto-cjk/tree/main/Sans) is
  licensed under SIL OFL 1.1; see `assets/fonts/LICENSE-NotoSansCJK.txt`.

The license texts are also embedded in the application's OSS licenses view.

## UmebocDC hand poses (converted data)

`crates/vtuber-tracking/src/hand_poses/catalog.rs` includes 20 adapted poses
derived from UmebocDC's `UmebocDC_Hand.211010` (2021-10-10). The user-supplied
archive's `Memo.txt` identifies the distribution, including its animation clips,
as CC0. See [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

Only converted hand-local joint angles are included. The original Unity package,
animation clips, models, and textures are not redistributed.

## bevy_vrm1 (temporary Bevy 0.20 runtime port)

The VRM runtime is based on [not-elm/bevy_vrm1](https://github.com/not-elm/bevy_vrm1)
tag `v0.10.0` (commit `671b71fca7c24f905f71d1be7698a350da262f0e`),
licensed under MIT OR Apache-2.0. The Cargo patch uses
[matsuvr/bevy_vrm1](https://github.com/matsuvr/bevy_vrm1) commit
`96678a0c761c2839ccbf6a0d9a92421339fdbafc`, which ports the runtime to
Bevy 0.20: WESL modules, vertex decompression and morph descriptor indexing,
outline depth-stencil views, and lifecycle observer syntax.
The transparency discard fix (upstream PR #68) and final expression weights
(upstream PR #70) are already included in v0.10.0. There is no vendored copy
of this crate in this repository.

The VRM 0.x import-time normalization, the direct-pose / direct-gaze input
components and writers, and the expression bind-status records in
`crates/vtuber-avatar` are application code owned by this repository (MIT),
ported from the retired `vendor/bevy_vrm1` patch so the upstream runtime's
other behavior can stay untouched. They carry no separate upstream license beyond
the MIT terms of this application.

`crates/vtuber-avatar/src/look/mtoon_rich.wesl` imports the Native MToon
fragment helpers from the patched dependency instead of copying that base.
The app-side added terms remain this repository's MIT code.
`crates/vtuber-avatar/src/look/mtoon_rich_vertex.wesl` adapts the MToon
vertex path from bevy_vrm1 v0.10.0 and Bevy 0.20's mesh vertex path; it keeps
the upstream MIT OR Apache-2.0 terms. The shared UV arithmetic in
`mtoon_uv.wesl` is retained from the MToon implementation under those terms.

`crates/vtuber-avatar/src/look/standard_rich.wesl` adapts the Bevy 0.20.0
`bevy_pbr` Standard fragment path so the app can add a narrow Rich term to the
unchanged PBR result. It keeps Bevy's MIT OR Apache-2.0 terms; the app-side
additive rim is this repository's MIT code.

## NDI® runtime

This application uses the NDI Standard SDK runtime for
transparent avatar output. The runtime DLL is not included in the normal
source tree and must only be staged from the exact SDK package used for a
release, after its license agreement and SDK documentation have been checked.
An SDK-free build remains available via `--no-default-features`.

NDI® is a registered trademark of Vizrt NDI AB. See the official
[NDI developer site](https://ndi.video/) and the
[NDI SDK documentation](https://docs.ndi.video/all/developing-with-ndi/sdk).

The NDI runtime is a separate proprietary component and is not covered by the
MIT terms of this application. A release that bundles it must
include the exact SDK license/notice material required by that SDK package.
The repository's cargo xtask ndi package command requires that material as
an explicit input and records its SHA-256 in the generated package manifest.
A ZIP that includes the runtime must also include
`Processing.NDI.Lib.Licenses.txt` from the same SDK `Bin\x64` directory.

This project does not redistribute NDI Tools, NDI Advanced/HX components,
audio codecs, SDK headers, import libraries, or build artifacts. The runtime
is loaded application-locally; the package process does not install anything
into a system directory or edit PATH.

## LLVM libunwind and MinGW-w64 CRT (Windows release ZIP)

CLANG64 Windows release ZIPs include MSYS2's unmodified `libunwind.dll` and
its license at `licenses/LLVM-libunwind-LICENSE.txt`. The executable also links
MinGW-w64 CRT code; its notices are included under `licenses/MinGW-w64-CRT/`.
These components retain their own licenses. Windows system libraries are
provided by the operating system and are not bundled.

## grafton-ndi

The Rust sender boundary uses
[grafton-ndi v1.0.0](https://github.com/GrantSparks/grafton-ndi), licensed
under Apache-2.0. Its source and license are obtained through Cargo; the NDI
SDK runtime remains a separately governed distribution component.

## bevy_window (monitor disconnect lifetime)

`vendor/bevy_window` contains the local bevy_window port to Bevy 0.20.0,
licensed under MIT OR Apache-2.0. Both license texts are retained. Its
monitor-to-window relationship does not despawn windows when a display
disconnects or sleeps; see `vendor/bevy_window/PATCHES.md` for the upstream
source provenance, retained differences and existing regression test.

## nokhwa (bounded macOS frame reads)

`vendor/nokhwa` contains nokhwa 0.10.11 from crates.io, licensed under
Apache-2.0. Its `LICENSE` is retained. The AVFoundation frame reads have a
two-second receive timeout so an interrupted camera can be stopped and
reopened after system sleep; see `vendor/nokhwa/PATCHES.md`.

## mediapipe-rs (vendored)

`vendor/mediapipe-rs` is a copy of
[nikicat/mediapipe-rs](https://github.com/nikicat/mediapipe-rs) at revision
`527037fa0fe1339750140283930bbb9560460e9e`, licensed under Apache-2.0. Its
`LICENSE` and `NOTICE` files are retained in the vendored directory. The copy
adds the MediaPipe Pose Landmarker bindings alongside the existing face tasks;
the upstream face API is unchanged. The loader also supports a native library
shipped beside the executable. Windows release ZIPs include the official
MediaPipe Tasks 0.10.35 `libmediapipe.dll`, extracted from the SHA-256-verified
PyPI wheel, with the wheel's license in `licenses/MediaPipe-LICENSE.txt`.
Source builds without a packaged library retain the upstream download behavior.

## MyoHub MyoArm anatomical parameters

The CMC, MCP and IP joint axes, ranges and hand offsets in `vtuber-avatar/src/thumb.rs`,
elbow/radioulnar axes and offsets in `vtuber-avatar/src/arm_anatomy.rs`,
and girdle axes and offsets in `vtuber-avatar/src/girdle.rs`
are adapted from `myo_sim/models/arm/assets/myoarm_r_chain.xml` at
[MyoHub/myo_sim revision 93b0ca8f4ec90c9899ee7f05fee561e9911da91b](https://github.com/MyoHub/myo_sim/tree/93b0ca8f4ec90c9899ee7f05fee561e9911da91b).
Copyright 2025 Vikash Kumar, Vittorio Caggiano. Apache License, Version 2.0;
see [the retained license](vendor/mediapipe-rs/LICENSE).
The adaptation maps source joint axes to the VRM's authored thumb shaft/palm
frames, including left-hand reflection. The forearm adapter retains the radial offset and neutral carrying angle,
while mapping the resulting direction onto the fixed VRM forearm length. No MuJoCo runtime or meshes are included.

## CMU Graphics Lab shoulder range-of-motion data and Joint Limits

`crates/vtuber-avatar/src/data/cmu_shoulder_rom.{json,rs}` contains a derived
full-rotation shoulder k-DOP, fitted by
`tools/research/convert_cmu_shoulder_limits.py` to CMU ROM clips 74/13, 126/14,
127/02 and 143/21. The JSON records SHA-256 hashes and sample counts. The raw
ASF/AMC recordings are not included in this repository.

The data used in this project was obtained from mocap.cs.cmu.edu.
The database was created with funding from NSF EIA-0196217.

CMU permits copying, modification and redistribution without permission, and
inclusion in commercially sold products. The data itself may not be resold,
including in converted form. See [CMU](https://mocap.cs.cmu.edu/) and its
[FAQ](https://mocap.cs.cmu.edu/faqs.php). This derived constraint is incorporated
into the application; it is not a standalone data product.

The full-rotation k-DOP method and 13 directions follow Daniel Holden's
[Joint Limits](https://github.com/orangeduck/Joint-Limits/tree/951358f73a90c82ff4fa33c6d11969c1eb0b02e1),
Copyright (c) 2021 Daniel Holden, MIT. The implementation was adapted to Rust
and an offline ASF/AMC converter. See [the MIT terms](LICENSE).
Its LAFAN1-derived demonstration database and PosePrior data are not included
or used to fit these limits.

## Skeleton collision proxy references

`crates/vtuber-avatar/src/collision.rs` independently implements skeleton-based
capsule dimensions informed by [Wicked Engine, wiPhysics_Jolt.cpp](https://github.com/turanszkij/WickedEngine/blob/7f776d02094571dd2c8166c7e5879d5e93d2b642/WickedEngine/wiPhysics_Jolt.cpp#L1471-L1536)
(MIT, Copyright (c) 2026 Turánszki János). No Jolt or ragdoll code is included.
Upper-arm and forearm radii use .20 and .15 of their rest lengths; the torso
radius is .1 times object rest scale, at unit fatness. This is a standard proxy,
not a measurement of every avatar's body or clothing.

RusTuberV deliberately differs in capsule placement: torso links follow the
animated hips/spine/chest hierarchy; forearm centre lines reach the elbow and
wrist so the elbow has a full rounded cap. Existing MCP-based palm proxies are
retained. Optional missing MCPs leave the wrist protected by its forearm cap,
without claiming coverage of unknown fingers. Upper arms may overlap the torso;
forearms, palms and opposite arms are still checked, including swept paths.

Elbow target correction uses the shoulder--wrist swivel idea from Unzueta et al.,
[Full-body performance animation with Sequential Inverse Kinematics](https://doi.org/10.1016/j.gmod.2008.03.002),
Graphical Models 70 (2008), section 8. This is an independent bounded target
search, not their mesh collision detector or full SIK solver. It changes neither
the wrist target nor skeletal translations. Joint limits and collision checks
remain authoritative in the existing solver. Straight arms and circles with no
clear sampled target are left to that solver; a sampled target is not a proof
of anatomical or path feasibility. No extra contact clearance is imposed.
