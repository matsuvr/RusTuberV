# Third-party notices

## NDI® runtime

This application can optionally use the NDI Standard SDK runtime for
transparent avatar output. The runtime DLL is not included in the normal
source tree and must only be staged from the exact SDK package used for a
release, after its license agreement and SDK documentation have been checked.

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
