# ADR-024: NDI output enabled by default in the desktop product

## Status

Accepted on 2026-09-07.

## Context

ADR-012 kept the NDI sender behind the explicit `ndi-output` feature and
ADR-013 kept the default workspace build SDK-free. As a result, a plain
`cargo build -p vtuber-desktop` / `cargo run -p vtuber-desktop` produced a
binary with no NDI sender: the NDI pane could never reach Live and OBS never
listed the source, even on machines with the NDI runtime installed. The
product requirement is that NDI output works out of the box.

## Decision

- `vtuber-desktop` enables `ndi-output` (hence `vtuber-ndi/ndi-sdk`) in its
  default features. A plain `cargo build -p vtuber-desktop`,
  `cargo run -p vtuber-desktop`, `cargo check --workspace`, and
  `cargo test --workspace` now build the NDI sender.
- Default builds therefore require the locally installed NDI SDK headers, a
  bindgen toolchain, and — on Windows x86_64 — the x64 import library that
  `apps/desktop/build.rs` uses to stage the matching runtime DLL beside the
  executable. This is the same requirement the explicit feature already had;
  it now applies to the default build.
- An SDK-free build remains available as an explicit opt-out:
  `cargo build -p vtuber-desktop --no-default-features` and
  `cargo check --workspace --no-default-features`.
- The library crates keep their opt-in features unchanged
  (`vtuber-ndi/ndi-sdk`, `vtuber-app/ndi-output`). The `vtuber-ndi` crate
  boundary, the BGRA transport contract, and the application-local runtime
  distribution rules from ADR-012 and ADR-013 are unchanged.
- The UI keeps distinguishing the two failure modes: a build without the
  feature reports that NDI output is not included, while an NDI-enabled
  build without a discoverable runtime reports
  「NDIランタイムがインストールされていません」 and stays startable so a
  later Start retries initialization.

## Consequences

- Contributors and machines without the NDI SDK must pass
  `--no-default-features` for desktop/workspace builds. macOS default builds
  also require an installed NDI SDK.
- `docs/NDI_RELEASE.md` commands were updated to the plain default build;
  the 2026-08-20 acceptance table is kept as a dated record.

References:

- ADR-012 (sender boundary, unchanged)
- ADR-013 (runtime distribution, unchanged)
- [NDI SDK documentation](https://docs.ndi.video/all/developing-with-ndi/sdk)
