# Wrapping another MediaPipe task

Every MediaPipe task exposes the same six C functions, so adding one is mostly
mechanical. This is the recipe, the per-task facts you need before starting, and
the order the remaining tasks are worth doing in.

Read [../CONTRIBUTING.md](../CONTRIBUTING.md) first — the conventions there are
what keep the mechanical parts from going quietly wrong.

## The shape every task shares

```c
MpStatus MpXCreate(struct MpXOptions*, MpXPtr*, char** error_msg);
MpStatus MpXDetectImage(MpXPtr, MpImagePtr, const MpImageProcessingOptions*,
                        MpXResult*, char** error_msg);
MpStatus MpXDetectForVideo(MpXPtr, MpImagePtr, const MpImageProcessingOptions*,
                           int64_t timestamp_ms, MpXResult*, char** error_msg);
MpStatus MpXDetectAsync(MpXPtr, MpImagePtr, const MpImageProcessingOptions*,
                        int64_t timestamp_ms, char** error_msg);
void     MpXCloseResult(MpXResult*);
MpStatus MpXClose(MpXPtr, char** error_msg);
```

The verb changes per task — `Detect`, `Recognize`, `Classify`, `Segment`,
`Embed` — but nothing else does. `src/face_detector.rs` is the reference
implementation; copy it rather than starting fresh.

## Before you start: check these three things

**1. Does the task accept a region of interest?** Most do not, and passing one
returns `kMpInvalidArgument: This task doesn't support region-of-interest`. If
it does not, expose `*_rotated(&image, Rotation)` and do **not** invent an ROI
parameter — an impossible request should not be representable. Verify with:

```sh
grep 'roi_allowed' ../mediapipe/mediapipe/tasks/cc/vision/<task>/<task>.cc
```

`/*roi_allowed=*/false` means no. No mention at all means the default `true`
applies, and the task does accept one — you will need a `NormalizedRect` type
(deleted from this crate when the face tasks turned out not to want it; see git
history for the previous implementation).

**2. What running modes does it have?** Text tasks are single-shot; audio and
vision mostly have all three.

**3. What is each threshold actually thresholding?** Trace every `float` option
to the calculator that consumes it, in
`../mediapipe/mediapipe/tasks/cc/vision/<task>/<task>_graph.cc`. `min_score_thresh`
and `ThresholdingCalculator` mean `Confidence`; `NonMaxSuppression`,
`AssociationCalculator` or anything comparing to `CalculateIou` mean
`IouThreshold`. Do not go by the parameter's name.

## The trap waiting in phase 2: world landmarks

`landmark.h` declares two structs with **identical layouts and different
meanings**:

```c
struct MpLandmark           { float x, y, z; bool has_visibility; … };
struct MpNormalizedLandmark { float x, y, z; bool has_visibility; … };
```

`MpNormalizedLandmark` is relative to the image, in `0..1`.
`MpLandmark` — used for `pose_world_landmarks` and the holistic hand world
landmarks — is in **metres**, in a real-world frame centred on the subject's hips
or wrist. They are not convertible into one another, there is no image `Size`
that relates them, and nothing structural tells them apart. Mixing them yields
numbers that look plausible and are meaningless.

So the first task that returns world landmarks needs a third point type —
`WorldPoint3` in metres, with no `to_pixels` — sitting beside the existing
`NormalizedPoint3`. Do not reuse `NormalizedPoint3` for them, however tempting
the identical field list makes it.

## Per-task facts

| task | modes | ROI | notes |
|---|---|---|---|
| `face_detector` | image, video, stream | no | done |
| `face_landmarker` | image, video, stream | no | done |
| `hand_landmarker` | image, video, stream | no | `MpCategories* handedness` alongside the landmarks |
| `pose_landmarker` | image, video, stream | no | `MpImagePtr* segmentation_masks` + world landmarks |
| `gesture_recognizer` | image, video, stream | no | two `MpClassifierOptions` sub-structs |
| `holistic_landmarker` | image, video, stream | no | face + pose + both hands; fields are **by value**, not pointer+count |
| `object_detector` | image, video, stream | no | `typedef MpDetectionResult`, identical to face_detector |
| `image_classifier` | image, video, stream | **yes** | needs a normalized-rect type |
| `image_embedder` | image, video, stream | **yes** | plus `MpImageEmbedderCosineSimilarity` |
| `image_segmenter` | image, video, stream | no | returns `MpImage` masks, not a plain struct |
| `interactive_segmenter_legacy` | image only | no | takes a separate `MpRegionOfInterest` argument |
| `audio_classifier` | single, stream | n/a | needs an audio-buffer type and sample rate |
| `text_classifier` | single | n/a | simplest of all |
| `text_embedder` | single | n/a | **API changed upstream**, see below |
| `language_detector` | single | n/a | trivial |

## The recipe

1. **Add the headers to `vendor/wrapper.h`** — the task header plus its
   `*_result.h`. They are already vendored; all 38 public headers are present.

2. **Regenerate.** `./scripts/gen-bindings.sh`. Check the new
   `bindgen_test_layout_*` assertions appear for the structs you added.

3. **Add the v0.10.35 options layout** to `src/sys/compat.rs`: the task's options
   struct with `MpBaseOptionsV35` as its first field and the rest copied
   verbatim, plus a `const _: () = assert!(size_of::<…>() == N)` for both
   variants. Derive `N` from the generated layout assertions minus 16.

4. **Write `src/<task>.rs`**, copying `face_detector.rs`:
   - owned result types, converted with `unsafe fn *_from_raw`, then
     `Mp*CloseResult` immediately
   - a builder whose threshold setters take `Confidence` or `IouThreshold` as
     step 3 above determined
   - `build()` / `build_for_video()` / `build_stream(cb)`
   - `close_once` + `Drop` + `impl AsyncTask`
   - `unsafe impl Send`, `&mut self` on every inference method
   - a manual `Debug` impl

5. **Export** from `src/lib.rs`.

6. **Fixtures and tests.** Add the model URL to `scripts/fetch-fixtures.sh`
   (Google hosts them under `storage.googleapis.com/mediapipe-models/<task>/…`).
   Write the same test set as `tests/face.rs`: a pinned known-good result, a
   coordinate-space check, video mode, live stream, and the error paths.

7. **Run everything** in CONTRIBUTING.md's pre-commit list.

## What is genuinely new per phase

Phases 1 and 2 are the same work repeated. The later ones need something built
first.

**Phase 2 — remaining vision.** `hand_landmarker`, `pose_landmarker`,
`gesture_recognizer`, `holistic_landmarker`, `object_detector`,
`image_classifier`, `image_embedder`, `image_segmenter`,
`interactive_segmenter_legacy`.

New work, in rough order of how much thought it needs:

- **`WorldPoint3`** for pose and holistic world landmarks — see the trap above.
- **Segmentation masks**, which come back as `MpImagePtr*` rather than plain
  structs, so they need a `Mask` type over the existing `Image` (and category
  masks and confidence masks mean different things per pixel).
- **`NormalizedRect`** plus a region-of-interest path, for the two tasks that
  accept one. This existed and was deleted when the face tasks turned out not to
  want it; recover it from git history rather than rewriting.
- **`CosineSimilarity`** helpers for the embedders.
- **By-value result fields** in `holistic_landmarker`, which breaks the
  pointer+count conversion pattern every other task uses.

**Phase 3 — text.** `text_classifier`, `language_detector`, `text_embedder`. No
image plumbing at all, so these are the smallest.

⚠ `text_embedder` is the one place `abi-diff.sh` found a *real* API change
between v0.10.35 and master, not just a rename: `MpTextEmbedderEmbed` gained a
`const MpTextEmbedderFormatContext*` parameter, and two new enums
(`MpTextEmbedderEmbeddingType`, `MpTextEmbedderRole`) arrived with it. That is a
signature change, so the dual-ABI trick used for options structs does not cover
it — the two ABIs need different call sites. Do this task *after* the version
bump that ships the change, or be ready to branch on `lib.abi` at the call.

**Phase 4 — audio.** `audio_classifier`. Needs an audio-buffer type carrying a
sample rate, and its async mode is the stream machinery you already have.

**Phase 5 — metadata.** `MpFlatbufferParser*` to read `.task` bundles, so users
can discover labels and input normalization without Python.

**Phase 6 — optional.** GPU delegate (the prebuilt library does contain
`InferenceCalculatorGl*` and initialises EGL, so it may be a flag flip — untested);
Linux aarch64, which needs a source build because no wheel exists; and the LLM
converter/bundler, which exists in v0.10.35 but was **dropped on master**, so it
is probably not worth binding.

## When to introduce a macro

Not yet. Two implementations is not enough evidence of the shape — the third and
fourth will reveal which parts genuinely vary (result conversion, extra option
sub-structs, mask outputs). Around task four or five, a `task! { }` macro
generating builder + handle + `Drop` + `AsyncTask` + stream type, with each task
supplying only its options fields and a result converter, is where the leverage
is. Writing it earlier means designing around two data points and then fighting
the macro.

## Bumping the pinned MediaPipe version

```sh
scripts/abi-diff.sh v0.10.35 <new-tag>      # act on anything it flags
git -C ../mediapipe archive <new-tag> mediapipe/tasks/c | tar -x -C vendor/headers
# refresh vendor/REF, then:
scripts/gen-bindings.sh
```

Then update `MEDIAPIPE_VERSION` and the wheel URL + SHA-256 table in
`src/loader.rs` (hashes come from `https://pypi.org/pypi/mediapipe/<ver>/json`),
and re-run the test suite.

Because the crate carries both ABIs and detects at load, a bump does **not**
break existing users, and the previous `*V35` structs stay until you are willing
to drop support for libraries that old. When the rename ships, the currently
"renamed" branch becomes the common path and `Abi::V0_10_35` becomes the legacy
one — no code moves, only which branch is taken.
