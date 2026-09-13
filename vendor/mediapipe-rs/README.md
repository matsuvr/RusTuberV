# mediapipe

[![crates.io](https://img.shields.io/crates/v/mediapipe.svg)](https://crates.io/crates/mediapipe)
[![docs.rs](https://docs.rs/mediapipe/badge.svg)](https://docs.rs/mediapipe)
[![CI](https://github.com/nikicat/mediapipe-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/nikicat/mediapipe-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/rustc-1.85%2B-orange.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html)
[![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.35-4285F4.svg)](https://github.com/google-ai-edge/mediapipe/releases/tag/v0.10.35)

Safe Rust bindings to the [MediaPipe Tasks C API][c-api]. Face detection and face
landmarks today; the same shape extends to the rest of MediaPipe.

Every other Rust binding to MediaPipe is either abandoned (2020–2022) or
WebAssembly-only, and most wrap the Solutions API that Google deprecated in 2023.
This one targets the C API that Google's *own* Python bindings now run on, so it
sits on a maintained surface rather than a neglected one.

```rust
use mediapipe::{FaceDetector, Image, IouThreshold, ModelSource};

let mut detector = FaceDetector::builder(ModelSource::path("blaze_face_short_range.tflite"))
    .min_suppression_threshold(IouThreshold::new(0.3)?)
    .build()?;

let image = Image::from_file("portrait.jpg")?;
for face in detector.detect(&image)? {
    println!("{:?} {:?}", face.bounding_box, face.score());
    for kp in &face.keypoints {
        // Keypoints are normalized, the box is in pixels. The types make you convert.
        println!("  {:?}", kp.point.to_pixels(image.size()));
    }
}
# Ok::<(), mediapipe::Error>(())
```

## Nothing to install

The native library is **not linked** — it is `dlopen`ed on first use, so the crate
builds on any machine (docs.rs included) without MediaPipe present. On first run
it fetches `libmediapipe` (~34 MB) from the official PyPI wheel, verifies a
compile-time-pinned SHA-256, and caches it under `$XDG_CACHE_HOME/mediapipe-rs/`.

| variable | effect |
|---|---|
| `MEDIAPIPE_LIB` | path to a `libmediapipe` to use instead of downloading |
| `MEDIAPIPE_ABI` | `v0_10_35` or `renamed`, overriding ABI auto-detection |

Disable the default `download` feature for offline builds; then `MEDIAPIPE_LIB`
or a system path must supply the library.

Prebuilt libraries exist for linux-x86_64, macos-aarch64, windows-x86_64 and
windows-aarch64. **There is no linux-aarch64 wheel** — on a Pi or Jetson, build
`//mediapipe/tasks/c:libmediapipe.so` from source and point `MEDIAPIPE_LIB` at it.

## Two MediaPipe ABIs, one binary

MediaPipe renamed every C type to an `Mp*` prefix in commit `75c9711a3`
(2026-05-14), still unreleased. The rename is cosmetic, but the same change
inserted `int file_descriptor` into the middle of `BaseOptions`, growing it
56 → 72 bytes and shifting every field of every task options struct after it.

Headers from the wrong side of that commit still *link* and then silently misread
`running_mode`. So this crate carries both layouts and picks at load time, keyed
on a symbol that only exists post-rename
(`MpInteractiveSegmenterLegacyCreate`). The practical consequence: **the next
MediaPipe release will not require a new release of this crate.**

That one struct is the entire compatibility surface — see `src/sys/compat.rs`.
`scripts/abi-diff.sh <old-ref> <new-ref>` re-checks that claim across every
public header at version-bump time.

## Types carry the coordinate space

MediaPipe's C API reports a detection's bounding box in **pixels** and its
keypoints, landmarks and regions of interest **normalized to 0..1** — with
nothing in the types to tell them apart. Here `PixelPoint`/`PixelRect` and
`NormalizedPoint2`/`NormalizedPoint3` are distinct types with private fields, and
the only route between them is `to_pixels(size)` — which forces you to name the
image the normalization is relative to. `Size` has no `new(w, h)`, because two
positional `u32`s are the swap it exists to prevent; write
`Size { width, height }`. Same-space geometry (`distance_to`, `contains`,
`NormalizedPoint3::xy`) lives on the types so there is no reason to pull raw
floats out and mix spaces by hand.

Likewise `Timestamp` for presentation times, `Rotation` for the multiples of 90°
MediaPipe actually accepts, and `Transform4x4` for the column-major 4×4 face
transform.

Each of these is held in place by a `compile_fail` doctest pinned to the expected
error code, so they prove the intended failure rather than any failure.

The builder thresholds are all bare `float`s in `0..1` upstream, but they are two
different quantities, and one of them is misnamed:

| parameter | what it actually configures | type |
|---|---|---|
| `min_detection_confidence` | `TensorsToDetections.min_score_thresh` | `Confidence` |
| `min_face_detection_confidence` | detector-stage score threshold | `Confidence` |
| `min_face_presence_confidence` | `Thresholding` on a presence score | `Confidence` |
| `min_suppression_threshold` | NMS with `overlap_type = INTERSECTION_OVER_UNION` | `IouThreshold` |
| `min_tracking_confidence` | `AssociationNormRect`, tested against `CalculateIou` | `IouThreshold` |

`min_tracking_confidence` is not a confidence at all — upstream's own doc comment
calls it "the minimum confidence score for the face tracking", but it reaches
`AssociationCalculator`, which compares it to the IoU between this frame's face
box and the previous one. Typing it as `IouThreshold` means a score read off a
detection cannot be passed where an overlap ratio belongs; there is a
`compile_fail` doctest holding that line.

There is no region-of-interest type: both face tasks are built upstream with
`roi_allowed=false`, so an ROI is not a runtime error to guard against — it is a
request that cannot be expressed. It will arrive with the tasks that accept one.

## Running modes

| builder | method | notes |
|---|---|---|
| `build()` | `detect`, `detect_rotated` | single images |
| `build_for_video()` | `detect_for_video` | stateful tracking, increasing timestamps |
| `build_stream(cb)` | `send`, `send_rotated` | results on a worker thread |

Live-stream mode is **lossy by design**: MediaPipe flow-limits the graph and
drops frames that arrive while it is busy. Use video mode if you need every
frame. Dropping a `Stream` closes the task, which flushes and joins the worker
before releasing its callback slot, so no callback can fire afterwards. There are
8 concurrent live-stream slots — the C callback has no `user_data`, so the
trampolines have to be statically allocated.

## Development

```sh
./scripts/fetch-fixtures.sh    # models + test image into models/
cargo test
cargo run --example detect_face -- models/portrait.jpg

./scripts/gen-bindings.sh      # regenerate src/sys/bindings.rs (needs libclang)
./scripts/abi-diff.sh v0.10.35 HEAD
```

Bindings are generated from headers vendored in `vendor/headers/` (see
`vendor/REF` for the exact upstream commit) and checked in, so consumers need no
libclang.

- [CONTRIBUTING.md](CONTRIBUTING.md) — conventions, and the silent failures they
  exist to prevent
- [docs/adding-a-task.md](docs/adding-a-task.md) — recipe for wrapping the
  remaining 14 tasks, with per-task modes/ROI/gotchas

## License

Apache-2.0, matching MediaPipe. The vendored headers are Google's, under the same
license.

[c-api]: https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/tasks/c
