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

## bevy_vrm1 (temporary transparency patch)

The VRM runtime is based on [not-elm/bevy_vrm1](https://github.com/not-elm/bevy_vrm1)
tag `v0.9.3` (commit `bf0ec103970fdf9091134fe0f208cf3590031c80`,
Bevy 0.19 support), licensed under MIT OR Apache-2.0. The temporary Cargo
patch uses [matsuvr/bevy_vrm1](https://github.com/matsuvr/bevy_vrm1) commit
`6ee2f610d6c95b542a3317028da10aba26c9e8de`. Its only difference from
the tag is a BLEND transparent-fragment discard in `mtoon_fragment.wgsl`,
proposed upstream in [PR #68](https://github.com/not-elm/bevy_vrm1/pull/68).
There is no vendored copy of this crate.

The VRM 0.x import-time normalization, the direct-pose / direct-gaze input
components and writers, and the expression bind-status records in
`crates/vtuber-avatar` are application code owned by this repository (MIT),
ported from the retired `vendor/bevy_vrm1` patch so the upstream runtime's
other behavior can stay untouched. They carry no separate upstream license beyond
the MIT terms of this application.

`crates/vtuber-avatar/src/look/mtoon_rich.wgsl` and
`crates/vtuber-avatar/src/look/mtoon_rich_vertex.wgsl` copy the MToon base
fragment/vertex WGSL from the pinned fork commit (the vertex code is unchanged
from upstream `v0.9.3`) so the Rich material can extend the corrected Native
result without modifying the dependency's shader assets. The copies
keep the upstream MIT OR Apache-2.0 terms; the app-side added terms are this
repository's MIT code.

`crates/vtuber-avatar/src/look/standard_rich.wgsl` copies the Bevy 0.19.0
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

## grafton-ndi

The Rust sender boundary uses
[grafton-ndi v1.0.0](https://github.com/GrantSparks/grafton-ndi), licensed
under Apache-2.0. Its source and license are obtained through Cargo; the NDI
SDK runtime remains a separately governed distribution component.

## mediapipe-rs (vendored)

`vendor/mediapipe-rs` is a copy of
[nikicat/mediapipe-rs](https://github.com/nikicat/mediapipe-rs) at revision
`527037fa0fe1339750140283930bbb9560460e9e`, licensed under Apache-2.0. Its
`LICENSE` and `NOTICE` files are retained in the vendored directory. The copy
adds the MediaPipe Pose Landmarker bindings alongside the existing face tasks;
the upstream face API and loader are unchanged. The native
`libmediapipe` library that binds to is the official MediaPipe Tasks 0.10.35
build and is fetched, not redistributed, by this repository.

## Google Neural Mesh (GNM) sparse landmark data

`crates/vtuber-gnm/assets/head_sparse_68.txt` is copied from the Google GNM
repository at revision
`f76519f4c0340e5333146c0a8f011c56879ae5e3`, matching the model and schema
revision, and is distributed under the upstream Apache-2.0 license.
`assets/models/gnm_head.npz` is the official GNM Head v3 archive from the same
revision. It is 53,305,389 bytes with SHA-256
`1DFF6A319C2FA28377D7669C30AA533CC0799B45E6049AF18E709B0CB8F122DB` and is
redistributed under the upstream Apache-2.0 terms with this notice retained.
The exact source URL, schema path, and redistribution record are maintained in
`assets/models/manifest.toml`.

## MyoHub MyoArm anatomical parameters

The CMC joint axes, ranges and hand offsets in `vtuber-avatar/src/thumb.rs`,
elbow/radioulnar axes and offsets in `vtuber-avatar/src/arm_anatomy.rs`,
and girdle axes and offsets in `vtuber-avatar/src/girdle.rs`
are adapted from `myo_sim/models/arm/assets/myoarm_r_chain.xml` at
[MyoHub/myo_sim revision 93b0ca8f4ec90c9899ee7f05fee561e9911da91b](https://github.com/MyoHub/myo_sim/tree/93b0ca8f4ec90c9899ee7f05fee561e9911da91b).
Copyright 2025 Vikash Kumar, Vittorio Caggiano. Apache License, Version 2.0;
see [the retained license](vendor/mediapipe-rs/LICENSE).
The adaptation maps source joint axes to the VRM's authored metacarpal/palm
frame, including left-hand reflection. The forearm adapter retains the radial offset and neutral carrying angle,
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
