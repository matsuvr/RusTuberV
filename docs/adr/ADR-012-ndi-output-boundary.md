# ADR-012: Optional NDI output boundary

## Status

Accepted for Issue #47 on 2026-08-18.

## Decision

The NDI sender lives in the independent `vtuber-ndi` crate. Its default
feature set is SDK-free and exposes a deterministic typed-disabled result.
`vtuber-desktop` can opt into the explicit `ndi-output` feature, which enables
`vtuber-ndi/ndi-sdk`; no application or avatar crate contains NDI types or
hand-written FFI.

The backend accepts only the `vtuber-core::VideoOutputFrame` contract from
Issue #46: packed BGRA8, straight alpha, exact profile dimensions and stride.
It maps to standard NDI High Bandwidth BGRA progressive video with a
profile-derived frame rate and square-pixel aspect ratio. It does not
premultiply, unpremultiply, chroma-key, convert to BGRX, add audio, or use NDI
Advanced/HX features.

The controller owns one sender worker, one `Mutex<Option<VideoOutputFrame>>` +
`Condvar` latest-value mailbox, and the worker join handle. Stop closes the
mailbox, requests cooperative stop, joins the worker, and drops the sender and
runtime handles through the binding's RAII path. A slow sender can replace
pending frames but cannot block the render/readback producer or grow a queue.

## Dependency audit

| item | fixed source/version | purpose | license / distribution |
| --- | --- | --- | --- |
| `grafton-ndi` | crates.io `=1.0.0`, upstream `GrantSparks/grafton-ndi` tag `v1.0.0` | safe Rust API for the Standard NDI 6 sender/runtime | Apache-2.0; the NDI SDK headers and runtime remain developer-installed and are not committed or bundled here |
| NDI Standard SDK | NDI 6.3 documentation target | High Bandwidth BGRA sender, source discovery, connection count, synthesized/default timecode | SDK license must be checked at packaging time; Issue #49 owns notices, DLL placement, and redistribution |

The wrapper's default image-encoding feature is disabled. `advanced_sdk`,
audio, PTZ, receiver, and async-runtime features are not enabled. The wrapper
build still requires the local NDI SDK headers and bindgen toolchain only when
`ndi-sdk` is explicitly enabled. The default workspace build does not require
that SDK.

References:

- [NDI SDK documentation](https://docs.ndi.video/all/developing-with-ndi/sdk)
- [NDI send API](https://docs.ndi.video/all/developing-with-ndi/sdk/ndi-send)
- [NDI frame types](https://docs.ndi.video/all/developing-with-ndi/sdk/frame-types)
- [`grafton-ndi` v1.0.0](https://github.com/GrantSparks/grafton-ndi/releases/tag/v1.0.0)
- [`grafton-ndi` license](https://github.com/GrantSparks/grafton-ndi/blob/v1.0.0/LICENSE)

## Consequences

- Normal format, check, test, clippy, and deny gates remain SDK-free.
- An NDI-enabled local build and machine sender smoke require a separately
  installed NDI SDK/runtime and are not claimed by the default automated gate.
- Runtime packaging, trademark attribution, SDK license notices, and OBS
  interoperability remain Issue #49 work.
