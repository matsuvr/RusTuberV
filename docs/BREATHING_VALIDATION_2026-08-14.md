# Always-on idle breathing validation report

Date: 2026-08-15
Scope: GitHub Issue #20 (always-on procedural idle breathing)
Code baseline: `b2fe9e4`
Status: normative automated/headless acceptance PASS; model/root-space managed compatibility PASS for three local VRM 1.0 files; optional manual visual observation NOT RUN in this audit

## Environment

| Field | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 (x86_64) |
| Shell | PowerShell |
| rustc | 1.97.1 (8bab26f4f 2026-07-14) |
| cargo | 1.97.1 (c980f4866 2026-06-30) |
| Bevy | 0.19.0 |
| bevy_vrm1 | f9593fd78136fb9e0507bcae111e09291ec9b82a (unchanged vendor patch) |
| GPU | NVIDIA GeForce RTX 4090 (Vulkan, driver 610.47) |
| CPU | 13th Gen Intel Core i9-13900 |

## Final default profile

| Field | Value |
|---|---|
| period_seconds | 5.0 |
| vertical amplitude | clamp(0.010 * rest_hips_height, 0.006 m, 0.0125 m) |
| forward amplitude | clamp(0.008 * rest_hips_height, 0.004 m, 0.010 m) |
| waveform | breath = sin(pi * phase)^2, phase = (elapsed / period) mod 1 |
| phase semantics | evaluated before advancing the f64 accumulator; first Ready frame is exactly phase 0 (no pop) |
| owner | additive hips.translation only (after AnimationSystems, before direct-pose BodyTracking) |

Reference rationale: established desktop-avatar implementations apply a small non-negative
sin-squared body offset added to the body IK target; a typical checked-in prefab uses 0.01 m offsets.
The issue profile deliberately stays at or below those small offsets.

## Model/root-space contract

`RestGlobalTransform` is a global/world-space affine, not VRM model/root
space. At binding time the implementation caches an immutable root rest/global
affine `G_root`; when the root has no `RestGlobalTransform`, the root's
binding-time `GlobalTransform` is cached as the explicit fallback authority.
It then resolves:

```text
G_parent = G_hips * inverse(hips RestTransform)
hips_model_position = inverse(G_root) * G_hips.translation
rest_hips_height = hips_model_position.y
parent_in_model = inverse(G_root) * G_parent
model_to_parent_local = inverse(linear(parent_in_model))
up_local = model_to_parent_local * +Y
forward_local = model_to_parent_local * +Z
```

The cached root authority is never replaced by an animated/current root
transform. Root rotation, translation, and scale therefore do not change the
model-space amplitudes or semantic breathing direction. Non-finite or
non-invertible affines disable breathing as a safe no-op.

## Real-model headless motion evidence (production plugin path)

The managed compatibility runner now verifies breathing after Ready
(tools/xtask/src/vrm_managed_compatibility.rs): it requires
BreathingProfile/BreathingBinding/BreathingState on the Ready root, runs 150
wall-paced frames, and fails unless the hips translation moves more than
1.0e-4 m while staying finite.

| Model | Result | rest_hips_height | vertical | forward | up_local | forward_local |
|---|---|---|---|---|---|---|
| apps/desktop/assets/models/inore-vrm1.vrm | PASS | 0.893817 m | 0.008938 m | 0.007151 m | (0,1,0) | (0,0,1) |
| AvatarSample_C.vrm | PASS | 1.026783 m | 0.010268 m | 0.008214 m | (0,1,0) | (0,0,1) |
| 1565994099520778586.vrm | PASS | 0.893817 m | 0.008938 m | 0.007151 m | (0,1,0) | (0,0,1) |

Command per model:

```text
cargo run -p xtask -- vrm-managed-compat <path-to-model.vrm>
```

Pre-existing fixture limitations, unrelated to breathing:

- tests/fixtures/vrm/alicia-solid.vrm and seed-san.vrm fail import with
  MODEL_FILE_INVALID (GLB parse) before any runtime work.
- tests/fixtures/vrm/tsukuyomi-chan.vrm and the local Tsukuyomi Type A model
  are VRM 0.x and are correctly rejected by the VRM 1.0 gate
  (MODEL_NOT_VRM1).

## Automated regression matrix

| Acceptance behavior | Local evidence | Result |
|---|---|---|
| Neutral exactly at phase 0/wrap, peak exactly at 0.5, clamped finite invalid input | cargo test -p vtuber-avatar breathing::tests | PASS |
| Continuity across wrap/peak, inhale/exhale monotonicity, true 5 s cycle | cargo test -p vtuber-avatar breathing::tests | PASS |
| 30/60/120 fps equivalence | cargo test -p vtuber-avatar breathing::tests --test breathing | PASS |
| Amplitude scaling and min/max clamps; invalid geometry rejection | cargo test -p vtuber-avatar breathing::tests | PASS |
| Profile validation (period, factors, bounds) | cargo test -p vtuber-avatar breathing::tests | PASS |
| Model/root-to-parent-local conversion for rotated root/rest and non-uniform scale | cargo test -p vtuber-avatar breathing::tests | PASS |
| First ready frame exactly equals authored base | cargo test -p vtuber-avatar --test breathing | PASS |
| No accumulation over 10 s of cycles; exact base at cycle boundaries | cargo test -p vtuber-avatar --test breathing | PASS |
| Animation-base replacement detected and preserved | cargo test -p vtuber-avatar --test breathing | PASS |
| Non-identity root/parent/rest rotations preserve model-space up/forward; world displacement is root-rest-linear * semantic model delta | cargo test -p vtuber-avatar --test breathing | PASS |
| Root translation and non-uniform scale do not change model-space height, amplitudes, or local delta | cargo test -p vtuber-avatar --test breathing | PASS |
| Root rotation preserves model-space height even when world Y is zero/negative | cargo test -p vtuber-avatar --test breathing | PASS |
| Two non-commuting intermediate nodes plus non-identity root preserve analytic global composition and semantic model displacement | cargo test -p vtuber-avatar --test breathing | PASS |
| Binding-time root RestGlobalTransform preference and GlobalTransform fallback | cargo test -p vtuber-avatar --test binding | PASS |
| Non-finite/non-invertible root affine disables breathing without NaN/panic | cargo test -p vtuber-avatar breathing::tests | PASS |
| Non-humanoid intermediate node; same-frame global consumption by body tracking | cargo test -p vtuber-avatar --test breathing | PASS |
| No scale/rotation writes; root and camera untouched | cargo test -p vtuber-avatar --test breathing | PASS |
| Continues without control frame and inactive tracking; transitions do not snap phase | cargo test -p vtuber-avatar --test breathing | PASS |
| Unload/replacement clears state; replacement starts neutral | cargo test -p vtuber-avatar --test breathing | PASS |
| Non-Ready lifecycle stops writing; NaN hips is a bounded no-op | cargo test -p vtuber-avatar --test breathing | PASS |
| Schedule: after animation, before body tracking/arm pose/gaze/constraints | cargo test -p vtuber-avatar --test schedule | PASS |

Complete local run for this repair:

```text
cargo fmt --all -- --check            PASS
cargo check --workspace               PASS
cargo clippy --workspace --all-targets -- -D warnings PASS
cargo test --workspace --no-fail-fast PASS
cargo deny check                      PASS
git diff --check                      PASS
cargo test -p vtuber-avatar --test breathing PASS
cargo test -p vtuber-avatar breathing::tests PASS
cargo test -p vtuber-avatar --test schedule PASS
cargo run -p xtask -- vrm-managed-compat apps/desktop/assets/models/inore-vrm1.vrm PASS
cargo run -p xtask -- vrm-managed-compat AvatarSample_C.vrm PASS
cargo run -p xtask -- vrm-managed-compat 1565994099520778586.vrm PASS
```

`cargo deny check` emitted only existing repository policy/dependency warnings:
the unmatched `Unicode-DFS-2016` allowance, duplicate transitive crate entries,
and the existing yanked `wide` advisory. No dependency or lockfile change was
made by this repair, so no new warning was introduced. Advisory, bans, license,
and source checks passed. No GitHub Actions workflow, trigger, or remote CI
result was used.

## Historical windowed observation (non-normative; not rerun in this audit)

The product desktop app was launched locally with
--model tests/fixtures/vrm/inore-vrm1.vrm. The model imported, the VRM
runtime attached, binding completed (lifecycle Ready), and the compatibility
report printed has_body_tracking_component: true with 36 spring roots.

Windowed visual observation of the breathing motion could not be confirmed on
this machine: repeated timed captures of the application window (CopyFromScreen
and PrintWindow with PW_RENDERFULLCONTENT, 0.5 s cadence over a full 5 s
cycle) were pixel-identical, while the app process shows multiple
busy-spinning worker threads (about 250 percent CPU), consistent with the
pre-existing no-webcam startup behavior of the application on this machine.
The windowed app therefore could not serve as a live visual evidence source
here; the headless managed runner above exercises the identical production
plugin path and measures real hips displacement directly.

The requested human visual checks (front, 45 degree, side views; short/chibi
stylized model with materially different rest rotations) remain NOT RUN. No
third-party VRM was copied into the repository for this purpose and no visual
PASS is claimed. The three managed runs above are normative headless runtime
evidence, not human visual acceptance; the historical windowed observation is
also non-normative.

## Files changed

- crates/vtuber-avatar/src/breathing.rs: profile, waveform, model/root-space
  amplitude and direction resolution, runtime compositor, unit tests.
- crates/vtuber-avatar/src/binding.rs: captures immutable root rest/global
  authority (with binding-time GlobalTransform fallback), resolves hips rest
  data and ancestor path once at binding, and inserts the breathing components.
- crates/vtuber-avatar/src/plugin.rs: registers apply_breathing_hips_translation
  after AnimationSystems and before apply_direct_body_tracking.
- crates/vtuber-avatar/src/lib.rs: module + re-exports.
- crates/vtuber-avatar/tests/breathing.rs (new): ECS integration/lifecycle/schedule tests.
- crates/vtuber-avatar/tests/schedule.rs: breathing ordering assertions.
- tools/xtask/src/vrm_managed_compatibility.rs: post-Ready breathing verification.
- docs/adr/ADR-004-avatar-control-order.md: breathing ownership/order/profile contract.
- DESIGN.md: system order diagram and breathing section.

## Known limitations and follow-up

- No macOS compile or runtime validation was performed for this issue.
- The windowed visual gate remains open for a human observer on a machine
  where the desktop app renders continuously.
- Non-identity hips-parent rest orientations on real models were not observed
  in the available VRM 1.0 fixtures; that path is covered by synthetic tests
  (rotated root/intermediate/hips rest, non-uniform parent scale, non-identity
  root rotation/translation/scale).
- The current audit reran the managed compatibility command for the three
  representative local VRM 1.0 files listed above; no camera or windowed
  visual acceptance was performed.
- The historical desktop observation of a static frame alongside busy worker
  threads is retained as context only and is unrelated to the breathing
  contract.
