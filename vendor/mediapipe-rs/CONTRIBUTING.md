# Conventions

Rules this codebase actually follows, and the reasoning behind them. They exist
because MediaPipe's C API is easy to bind *wrongly in ways that do not fail
loudly* — most of what follows is a defence against a specific silent failure we
have already hit.

For the recipe to add a new task, see [docs/adding-a-task.md](docs/adding-a-task.md).

## The two invariants everything else serves

**1. The loaded library's ABI is not knowable at compile time.** MediaPipe grew
`BaseOptions` by 16 bytes in an unreleased commit. Headers from the wrong side
still link, and the C side then reads `running_mode` from the wrong offset. Any
new options struct must be built through the `match lib.abi` in `build_mode`, and
any layout difference must be mirrored in `src/sys/compat.rs` with a `const`
size/offset assertion. Never add a field to a `*OptionsV35` struct without
running `scripts/abi-diff.sh`.

**2. C owns its memory and reclaims it on its own schedule.** Results are freed
by `Mp*CloseResult`, and in live-stream callbacks by MediaPipe itself the instant
the callback returns. Convert into owned Rust values immediately; never let a
pointer into C memory outlive the call that produced it. See
`face_landmarker.rs::detect_inner` for the shape, and `stream.rs` for the
callback rules.

## Types

**Make a domain type when the primitive is ambiguous, not merely when it is
primitive.** The test: *could a value of this type be confused with a different
quantity that has the same representation?* MediaPipe spells detection scores,
IoU overlap ratios, normalized coordinates and pixel coordinates all as bare
`float`s. Those became `Confidence`, `IouThreshold`, `NormalizedPoint2` and
`PixelPoint`. Category indices and label strings did not get newtypes, because
nothing else in the API is an index or a label.

**Do not trust upstream names.** `min_tracking_confidence` is not a confidence;
it reaches `AssociationNormRectCalculator` and is compared against
`CalculateIou`. Trace a parameter to the calculator that consumes it before
deciding its type, and record what you found in the doc comment.

**Coordinate types have private fields.** A public `x: f32` lets a pixel value be
written into a normalized type. Provide accessors plus the same-space operations
people would otherwise extract raw floats for (`distance_to`, `contains`,
`xy()`). Types that only ever come *out* of MediaPipe get no public constructor
at all — that is what makes constructing one from pixel values impossible.

**No positional constructors for same-typed pairs.** `Size` has no
`new(width, height)`; `Size { width, height }` cannot be transposed silently.

**Fallible construction, no `From`.** `Confidence::new` returns `Result`. There
is deliberately no `From<f32>` and no arithmetic on these types: a clamping or
inferring conversion would hide exactly the mistake the type exists to catch.

## Enforcing invariants

An invariant that is only described in prose will be violated. If the type system
enforces something worth stating, add a `compile_fail` doctest **pinned to the
expected error code**, so it proves the intended failure rather than any failure:

````rust
/// ```compile_fail,E0308
/// # use mediapipe::{Confidence, FaceDetector, ModelSource};
/// FaceDetector::builder(ModelSource::path("m.tflite"))
///     .min_suppression_threshold(Confidence::HALF);  // a score is not an overlap ratio
/// ```
````

An unpinned `compile_fail` passes when the example fails to compile for an
unrelated reason, such as a renamed import — which makes it worse than no test.

## `unsafe`

Every `unsafe` block carries a `// SAFETY:` comment saying what makes it sound —
not what it does. Three rules specific to this API:

- **Never free what MediaPipe owns.** The `MpImagePtr` passed to a live-stream
  callback points at a stack `MpImageInternal` inside MediaPipe's dispatch
  lambda; `MpImageFree` on it frees a stack address. That is why `ImageRef`
  exists and has no `Drop`.
- **Never double-close.** Where both `Drop` and an explicit `close` exist, guard
  with a `close_once` / `closed` flag. `Stream` closes its task before releasing
  the callback slot, and `Drop` must not then close it again.
- **Never unwind across FFI.** User callbacks run inside `catch_unwind`.

## Errors

Typed variants, never stringly-typed. Validate at the boundary *before* calling
into C when C would accept the value and misbehave — a short pixel buffer is an
out-of-bounds read inside MediaPipe, so `Image::from_u8` checks the length first.

No `unwrap` in library code. `expect` is permitted where the invariant is local
and provable; its message follows the standard-library convention of describing
**why success is expected**, as a "should" statement:

```rust
NonZeroU32::new(1).expect("1 should be non-zero")
dest.parent().expect("cache path should have a parent, it is built by joining onto a cache dir")
```

Not `expect("get parent")`, which describes the operation and tells a reader
nothing when it fires.

## Comments

Explain *why*, and cite the upstream file when the reason lives in MediaPipe's
source rather than ours — `face_landmarker.cc`, `matrix_converter.cc`,
`association_calculator.h`. A reader cannot rederive "MediaPipe frees this the
moment we return" from our code alone.

Mark deliberate shortcuts with a `ponytail:` comment naming the ceiling and the
upgrade path, so `scripts/`-free debt stays visible:

```rust
// ponytail: 8 concurrent live-stream tasks. Raise this or move to libffi
// closures if anyone ever needs more; the C API gives us no per-callback
// context to key off, so a fixed pool is the price.
```

## Tests

**Assert the invariant, not an incidental number.** Live-stream mode drops frames
by design, so asserting "10 sent, 10 received" is flaky and wrong; assert that
results are delivered, in order, drawn from the frames sent, and that delivery
stops at drop. A test that must be loosened later was testing the wrong thing.

**Some numbers are worth pinning.** The expected face box `(283,115) 234×234` at
score 0.922 was measured through the raw C API before this crate existed, so it
doubles as an ABI check: a wrong options layout does not shift it, it fails
outright.

Fixtures are downloaded by `scripts/fetch-fixtures.sh` and never committed. Tests
return early with an explanatory message when they are absent, so a fresh
checkout without network reports no failures.

Tests run in parallel: anything touching the 8-slot live-stream pool must assert
a range, not an exact count.

## Generated and vendored code

`src/sys/bindings.rs` is generated and checked in so consumers need no libclang.
Regenerate only via `scripts/gen-bindings.sh` — never hand-edit it. The headers
in `vendor/headers/` are Google's, unmodified apart from deleting the C++-only
`*_converter.h`; `vendor/REF` records the exact upstream commit they came from.

## Lints

Levels are set in `Cargo.toml` at whatever the codebase currently satisfies, so
each one is a convention from this document that cannot decay:

| lint | why |
|---|---|
| `unsafe_op_in_unsafe_fn` = deny | an `unsafe fn` body is not a blanket licence |
| `clippy::undocumented_unsafe_blocks` = deny | the SAFETY-comment rule above, mechanised |
| `clippy::cast_possible_truncation` = deny | a truncated length handed to C is read as a length |
| `clippy::cast_possible_wrap` = deny | ditto for a wrapped `u32` → `i32` |
| `missing_debug_implementations` = warn | public handles should be printable |

Generated code cannot satisfy the first three meaningfully, so
`scripts/gen-bindings.sh` emits `#![allow(…)]` for them at the top of
`bindings.rs`. Widen that allow-list rather than lowering a crate-level lint.

`missing_docs` is deliberately **not** enabled: it would demand 144 doc comments,
almost all on self-describing accessors and error variants whose `thiserror`
messages already say more than a doc line would.

## Checking memory correctness

The test suite structurally cannot see the failures that matter most here: a
missed `Mp*CloseResult` still returns correct results, and a use-after-free on a
result MediaPipe has already reclaimed usually returns plausible numbers.
`examples/stress.rs` drives every FFI path in a loop so a sanitizer can.

**AddressSanitizer — use-after-free, double-free, overflows.** The important
property is that ASAN replaces `malloc`/`free` process-wide, so allocations made
*inside* the uninstrumented `libmediapipe.so` are covered too. That is what makes
it able to catch a double-close, or reading a result after MediaPipe freed it.

```sh
RUSTFLAGS=-Zsanitizer=address \
ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=1 \
  cargo +nightly run --target x86_64-unknown-linux-gnu --example stress -- 25
```

**LeakSanitizer — leaks.** MediaPipe leaks ~152 KB at startup no matter what
(Mesa's EGL driver and MediaPipe's own `GlContext`/`GpuResources` never free
their one-time state), so the absolute number is meaningless. **The check is that
the total does not grow with the iteration count.**

```sh
RUSTFLAGS=-Zsanitizer=leak cargo +nightly run --target x86_64-unknown-linux-gnu \
    --example stress -- 1     # and again with 100
```

At the time of writing: ASAN reports 0 errors at N=25, and LSAN reports ~152 KB
at both N=1 and N=100 with no leak trace passing through this crate's code.

**Valgrind Memcheck — uninitialised reads.** This is the one thing the
sanitizers cannot do: catch a read of uninitialised memory *inside* the
uninstrumented `libmediapipe.so`. It matters because every task zeroes its result
struct and trusts MediaPipe to fill it (`mem::zeroed()` in `detect_inner`), so a
field the C side never writes would be read as zero rather than flagged.

Run it with leak checking **off**, so `ERROR SUMMARY` means memory errors only —
MediaPipe's static initialisers leak unconditionally and would otherwise drown
the signal:

```sh
valgrind --tool=memcheck --track-origins=yes --leak-check=no --error-exitcode=1 \
    ./target/debug/examples/stress 1
```

At the time of writing that reports **0 errors from 0 contexts**, with no
suppressions needed — no invalid reads or writes, no invalid or mismatched
frees, no uninitialised values, and no false positives from XNNPACK's hand
vectorised code. With `--leak-check=full` it additionally reports ~6 KB
definitely lost in 184 blocks, every one of them inside MediaPipe's own
`_GLOBAL__sub_I_*` calculator registration or `GpuResources`/`GlContext` setup,
none passing through this crate.

On Arch, Memcheck needs `glibc-debug`, which lives in the `[core-debug]` repo
that is not enabled by default. **The versions must match**: `pacman -Sy
glibc-debug` on its own installs debug symbols for a newer glibc than the one
installed, and Memcheck then rejects them with a `.gnu_debuglink` CRC mismatch
and refuses to start — for every binary, `/bin/true` included. `pacman -Syu` is
the fix. Neither `--enable-debuginfod=yes` nor `--allow-mismatched-debuginfo=yes`
works around it; the latter governs a later check than the debuglink CRC.

## Hooks

`prek install` (or `pre-commit install`) wires up `.pre-commit-config.yaml`:
`cargo fmt`, `cargo clippy -D warnings`, shellcheck, whitespace fixers, and a
check that vendored headers were not staged without regenerated bindings. Only
fast local checks live there — `cargo test` needs a 34 MB download and the
fixtures, so it is CI's job.

`scripts/*.sh` are POSIX `sh` unless they declare otherwise; shellcheck enforces
it, and it has already caught one script using bash-only process substitution
under a `#!/bin/sh` shebang.

## Before committing

```sh
cargo test                                    # 15 integration + 7 doc tests
cargo clippy --all-targets                    # must be silent
cargo clippy --no-default-features --all-targets
MEDIAPIPE_LIB=… cargo test --no-default-features
cargo run --example detect_face -- models/portrait.jpg
```

After bumping the pinned MediaPipe version, also run
`scripts/abi-diff.sh <old> <new>` and act on anything it flags.
