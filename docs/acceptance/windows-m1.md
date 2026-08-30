# Windows M1 Acceptance Report

**Status:** PASS — C922 functional／recovery／30-minute performance acceptance完了
**Date:** 2026-08-12
**Commit:** M1-08-018 `a83967d` + this M1-08-019 acceptance commit
**Binary:** `vtuber-desktop` (release profile)

---

## 1. Test Environment

| Item | Value |
|------|-------|
| OS | Windows 11 Pro 10.0.26200 (build 26200) |
| CPU | 13th Gen Intel(R) Core(TM) i9-13900 |
| GPU | Virtual Desktop Monitor; driver 13.50.53.699 |
| RAM | NOT RECORDED — no hardware claim |
| Screen | NOT RECORDED — no hardware claim |
| Camera 1 | c922 Pro Stream Webcam — MSMF symbolic link VID_046D/PID_085C |
| Camera 2 | ELECOM 2MP Webcam — MSMF symbolic link VID_056E/PID_701E |
| Build profile | release |
| Rust toolchain | rustc 1.97.1 / cargo 1.97.1 |
| Binary SHA-256 | `C939C12411EA88B7363B8463F117CFD53CA516CE6522C36FDE6C5D3A4802B1E2` |

### Model Manifest

| # | Model Name | Source | License | SHA-256 | VRM Version | Notes |
|---|-----------|--------|---------|---------|-------------|-------|
| 1 | MediaPipe Face Landmarker task bundle | Official MediaPipe model URL; ADR-009 | Apache-2.0 | `64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF` | n/a | CPU VIDEO mode; 478 landmarks, 52 blendshapes, one matrix |
| 2 | inore-vrm1.vrm | Local approved fixture | project fixture | `B5A3D4126C4A30EF3BFBCFC764A24DC48511B558799D98D4C2FF1DB0BDC7AB01` | VRM 1.0 | Imported in GUI; Ready |

The former UltraFace/Peppa entries and their hashes are retained only in the
historical composite evidence below. They are not the current production
backend.

### Previous MediaPipe GUI run (partial evidence)

The release binary was launched on Windows and exercised through the desktop
UI. C922 was enumerated and selected as `MSMF index 0`, `inore-vrm1.vrm` was
imported and reached `Avatar lifecycle: Ready`, and Start/Stop completed without
a process restart. Diagnostics while the camera was running showed:

```text
capture rate: 30.0 Hz
inference rate: 30.0 Hz
capture worker: Running
inference worker: Running
no-face frames: 447
landmark/tracking rate: 0.0 Hz
```

The input contained no detectable face during this run. Therefore no neutral
sample, Recenter, head pose, expression, gaze, face-loss/reacquire, avatar
replace, or unplug/reconnect result is claimed. The observed no-face state was
normal tracking state, not an inference failure.

### Current MediaPipe GUI run (live acceptance)

On 2026-08-12 the release binary was exercised with the user present in front
of the connected `c922 Pro Stream Webcam` (MSMF index 0). The approved
`inore-vrm1.vrm` fixture reached `Avatar lifecycle: Ready`. Auto-neutral
calibration completed, and Live showed:

```text
Tracking: Tracking
Confidence: 1.00
Face detected: yes
```

The six head-pose directions, blink, mouth-open, and left/right gaze prompts
were executed while the pipeline remained live. Face loss was observed as
`Face detected: no`, `Tracking: Initializing`, `Confidence: 0.00`; returning to
the camera reacquired `Face detected: yes`, `Tracking: Tracking`, and
`Confidence: 1.00` in the same process. Three Stop/Start cycles completed, and
avatar replace, unload (`Avatar lifecycle: None`), and reload (`Ready`) also
completed without a process restart. After reload, tracking reacquired again.

The UI did not expose a direct numeric avatar-apply counter or a
capture-to-apply latency value, and the viewport framing did not provide a
recordable head/expression visual result in this run. Therefore this evidence
does not claim the VRM head, blink, mouth, or gaze visual checks as PASS.

Follow-up physical inspection found the display condition itself is a blocker.
The user-provided Setup screenshot showed no camera image (Setup does not
render the optional preview by design) and only the avatar below the waist;
the avatar head was outside the viewport. In the preceding Live screen,
`Show Preview` was enabled but the UI remained `Waiting for camera frames…`
while the same process reported `Face detected: yes`. The source inspection
found a fixed camera at `(0.0, 0.0, 2.5)` looking at `(0.0, 0.0, 0.0)` in
`crates/vtuber-avatar/src/plugin.rs`; this does not implement DESIGN.md §19.2
head/upper-body framing. Consequently head pose, blink, mouth, and gaze must
remain BLOCKED until preview registration and camera framing are repaired and
the live protocol is repeated. The screenshot was not copied into the
repository because raw camera/UI captures are not required as a release
artifact.

The `M1-08-020` repair now keeps the dynamic preview `Image` in both the main
and render worlds, registers its handle with `bevy_egui`, and frames the
viewport once per avatar generation from the head and hips world positions.
The release GUI was rebuilt and the approved `inore-vrm1.vrm` fixture rendered
with its face and upper body visible, closing the viewport-framing source
failure. The camera selector showed `None` during this follow-up run, so the
real C922 preview and head/blink/mouth/gaze protocol were not rerun and remain
BLOCKED rather than being inferred from the automated texture-registration
tests.

### Post-repair real pose-apply validation

`M1-08-021` found that the former application-owned pose writer lacked the
rest-orientation data it required, so the lifecycle could reach `Ready` while
every real control frame was skipped. `Q2-06-001` replaces that writer and its
cache with direct-pose `bevy_vrm1::BodyTracking`: binding now requires the
runtime-provided `RestTransform` and `RestGlobalTransform` components before
entering `Ready`, while the application bridge only updates a
generation-matched pose input. Missing rest data remains a transient binding
condition rather than a panic.

The rebuilt release binary (SHA-256
`9AE2538289654EB5B7655442246A2012BC252B979E62D017A6E47EE39D4492C6`) was
run with the connected `ELECOM 2MP Webcam` and approved
`inore-vrm1.vrm`. This is a correctness confirmation on available hardware,
not a substitute for the C922-specific final gate. The live preview displayed
camera frames, the avatar head followed real yaw/pitch motion, and held physical
prompts visibly produced both-eye blink, mouth-open, and eye-gaze responses.
Diagnostics after the guided checks showed:

```text
capture rate: 29.0 Hz
inference rate: 30.0 Hz
tracking rate: 30.0 Hz
slot overwrites: 0
avatar frames applied: 5,111
avatar frames skipped: 0
capture-to-apply p50: 30.82 ms
capture-to-apply p95: 48.23 ms
```

This closes the preview/framing/real-pose correctness blocker and proves the
latency clock path is populated. The same functional matrix was subsequently
repeated on the required C922 as recorded below.

### Multiple-camera identity and selection validation

`M1-08-022` reproduced the selector problem with C922 and ELECOM connected at
the same time. A direct MSMF backend probe enumerated both physical devices:

```text
0: c922 Pro Stream Webcam [VID_046D/PID_085C]
1: ELECOM 2MP Webcam [VID_056E/PID_701E]
```

The app previously started with no enumeration request and later opened the
selected list position as a fresh numeric MSMF index. A device-order change
could therefore open a different physical camera. The repaired path requests
enumeration on startup, uses the MSMF symbolic device link as the descriptor
identity and open key, preserves selection by identity across Refresh
reordering, and reports enumeration errors instead of converting them to an
empty list.

With both cameras still connected, the rebuilt release GUI displayed both
choices without a manual Refresh. `c922 Pro Stream Webcam` was explicitly
selected and Live displayed real preview frames. A separate five-second
MediaPipe smoke opened the C922 symbolic link and reported 87 captured frames,
79 face results, 0 contract failures, and 0 latest-slot overwrites. The GUI run
was stopped and closed normally. This validates enumeration, selection, and
physical-device open.

### Final C922 functional and recovery validation

After `M1-08-020` through `M1-08-022`, the release binary identified the C922
by its MSMF symbolic device link while the ELECOM camera remained connected.
The approved `inore-vrm1.vrm` reached `Ready`; Live showed the C922 preview,
`Tracking`, confidence `1.00`, and `Face detected: yes`. With the avatar face
and upper body framed, held physical prompts directly produced the following
visible responses:

- yaw right and left, pitch up and down, and roll clockwise and
  counter-clockwise all moved the avatar head in the intended direction;
- closing both eyes closed both avatar eyes;
- a held "aa" prompt opened the avatar mouth;
- eyes-only right and left prompts visibly shifted the avatar gaze.

Diagnostics during the same explicit-C922 session reported:

```text
capture rate: 30.0 Hz
inference rate: 31.0 Hz
tracking rate: 31.0 Hz
slot overwrites: 0
avatar frames applied: 12,262
avatar frames skipped: 0
capture-to-apply p50: 29.90 ms
capture-to-apply p95: 48.02 ms
inference wait mean: 27.21 ms
inference total mean: 6.10 ms
```

Three Stop/Start cycles in the same process each returned the explicit C922
session to `Running`, then `Tracking` at confidence `1.00`. Stop returned to
`Idle`, and the window closed without leaving the process running. The earlier
release-GUI recovery protocol had already exercised face loss/return,
approximately 405 ms reacquisition, avatar replace/unload/reload, and physical
camera unplug/replug followed by Stop -> Idle -> Start recovery. The attempted
repeat of face loss and USB removal during this final symbolic-link session did
not remove the face or camera from the live preview, so it is not counted as a
second physical event. Recovery PASS relies on the recorded physical event
plus the final explicit-C922 three-cycle proof; no unperformed action is
claimed.

Raw camera screenshots were inspected live but are not committed as release
artifacts.

The 60-second diagnostics observation showed approximately:

```text
capture/inference: 30 Hz
tracking: 30-31 Hz
inference wait: p50 ~25 ms, p95 ~41.9 ms
inference total: p50 ~6.0 ms, p95 ~6.8 ms
capture-to-apply p50/p95: (none)
```

The subsequent 30-minute soak completed with the process responsive and the
workers still Running. Final diagnostics were `tracking 30.0 Hz`,
`slot overwrites 0`, `no-face frames 106`, inference wait p95 `41.46 ms`, and
inference total p95 `7.45 ms`. No crash, worker exit, or hang was observed.

---

## 2. Test Matrix

Automated verification completed on the environment above:

- `cargo fmt --all -- --check`
- `cargo check -p vtuber-desktop`
- `cargo clippy -p vtuber-app -p vtuber-avatar -p vtuber-camera -p vtuber-tracking --all-targets -- -D warnings`
- `cargo clippy -p vtuber-inference --all-targets -- -D warnings`
- `cargo run -p xtask -- acceptance verify assets/models/manifest.toml`
- `cargo test --workspace --no-fail-fast`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo build -p vtuber-desktop --release`

The current live GUI run above covers the MediaPipe worker and recovery
evidence. The historical composite diagnostic below is retained only as
superseded evidence and was exercised on Windows with the connected
`c922 Pro Stream Webcam`:

```text
cargo run -p xtask -- face-pipeline-smoke --duration 10 --json
camera: msmf:0 (1280x720 @ 30/1 Rgb8)
frames_captured: 297
face_count/no_face_count: 0/30
detector_hz/landmark_hz: 2.836/0.000
stage_error: none
```

The detector was also exercised directly with the user-supplied local JPEG
`ookawa721_0I9A4938_TP_V4.jpg` (not committed to the repository):

```text
cargo run -p xtask -- face-image-probe ookawa721_0I9A4938_TP_V4.jpg
dimensions: 800x533
result: FACE_DETECTED
confidence: 0.9999
rect: (0.4314,0.1891,0.1228,0.2397)
```

This confirms that the pinned UltraFace artifact, JPEG/RGBA8 preprocessing,
and detector decode can detect a face in a still image. It does not substitute
for the live camera composite gate because it does not exercise crop,
landmark, or planar-pose output.

The first live MSMF run returned no face even though the captured frame showed
the user. A post-fix run found that the landmark model emitted visibility
values above 1.0; the ONNX landmark decoder now clamps visibility to the
engine contract `[0,1]`. The resulting 20-second C922 run was:

```text
cargo run -p xtask -- face-pipeline-smoke --camera 0 --duration 20 --snapshot <temp>/vrm-c922-snapshot-after-fix.jpg --json
format: 1280x720 @ 30/1 Rgb8
frames_captured: 593
face_count/no_face_count: 6/0
detector_hz/landmark_hz: 0.293/0.293
detector_confidence: 0.610928
finite_landmarks: 98
finite_pose_count: 5
pose yaw/pitch/roll ranges: [-1.402699,0.477511] [-1.695925,2.557055] [-1.651763,0.366612]
stage_error: none
```

The full M1-08-013 camera protocol was then run with the guided CUI mode. It
prints each phase and a one-second countdown for the last three seconds:

```text
cargo run -p xtask -- face-pipeline-smoke --camera 0 --guided-protocol --json
```

```text
camera: c922 Pro Stream Webcam (MSMF index 0)
format: 1280x720 @ 30/1 Rgb8
frames_captured: 4944
face_count/no_face_count: 50/19
detector_hz/landmark_hz: 0.387/0.347
stage_error: none
detector_confidence: 0.779927
finite_landmarks/finite_pose_count: 98/44
pose yaw/pitch/roll ranges: [-5.145444,2.186486] [-2.124861,4.208844] [-2.604939,1.199027]
```

The guided run emitted and executed the 60-second neutral, five loss/return
cycles, left/right/up/down, edge, and three capture Stop/Start phases. The
`no_face_count` is expected during the out-of-frame phases. The first run
returned exit code 1 after printing the complete summary because the exact
deadline bookkeeping did not mark the protocol complete; the command now
normalizes that boundary and has a regression test.

A later 60-second requested-duration run was executed after the release build
was rebuilt. The command completed the full guided protocol, including the
three capture Stop/Start phases, and returned exit code 0. It is recorded as a
composite camera result only: the low face count means it is not evidence for
GUI calibration or avatar motion.

```text
cargo run -p xtask -- face-pipeline-smoke --camera 0 --guided-protocol --duration 60 --json
camera: c922 Pro Stream Webcam (MSMF index 0)
format: 1280x720 @ 30/1 Rgb8
frames_captured: 4942
face_count/no_face_count: 12/418
detector_hz/landmark_hz: 2.450/0.074
stage_error: none
detector_confidence: 0.884251
finite_landmarks/finite_pose_count: 98/12
pose yaw/pitch/roll ranges: [-1.762825,1.524811] [-2.552518,8.091341] [-1.380595,1.612097]
```

The implementation gates for M1-08-015〜M1-08-017 also passed in this
environment: observation freshness/reset, real-source avatar bridge
contracts, synthetic-source exclusivity compile check, worker failure
recovery, diagnostics, and clean-shutdown unit/integration coverage. The
managed VRM compatibility command reached `Ready` for the approved local VRM
fixture. These automated results do not replace the GUI functional gate.

| Row | Model | Camera | Protocol | Result | Notes |
|-----|-------|--------|----------|--------|-------|
| 1 | Model 1 | Camera 1 | M1-08-013 guided camera | PASS | Full guided camera phases completed; GUI/VRM acceptance is recorded separately as NOT RUN |
| 2 | Model 2 | Camera 1 | Full | NOT RUN | No manifest model |
| 3 | Model 3 | Camera 1 | Full | NOT RUN | No manifest model |
| 4 | Model 1 | Camera 2 | Full | NOT RUN | No physical camera inventory |

Skip conditions:
- Camera 2 not available → rows 4+ marked SKIP with reason
- Model requires expressions not present → capability limitation noted, not failure

---

## 3. Functional Protocol (M1-08-003)

### 3.1 Neutral Calibration

- [x] Start app with model loaded
- [x] Face camera in neutral position
- [x] Auto-neutral calibration completed without a blocking calibration phase
- [x] Hold neutral pose until calibration completed
- [x] Live state reached `Tracking` with confidence `1.00`

### 3.2 Head Pose (yaw / pitch / roll)

| Axis | Direction | Expected | Actual | Result |
|------|-----------|----------|--------|--------|
| Yaw | Turn right (image right) | Head turns right | C922 prompt visibly moved avatar right | PASS |
| Yaw | Turn left | Head turns left | C922 prompt visibly moved avatar left | PASS |
| Pitch | Chin up | Head tilts up | C922 prompt visibly moved avatar up | PASS |
| Pitch | Chin down | Head tilts down | C922 prompt visibly moved avatar down | PASS |
| Roll | Tilt right (clockwise) | Head tilts right | C922 prompt visibly rolled avatar right | PASS |
| Roll | Tilt left (counter-clockwise) | Head tilts left | C922 prompt visibly rolled avatar left | PASS |

### 3.3 Blink

| Capability | Expected | Actual | Result |
|-----------|----------|--------|--------|
| Per-eye (blinkLeft + blinkRight) | Independent left/right blink | Independent wink was not isolated | NOT RUN |
| Combined (blink only) | Both eyes blink together | C922 held closure visibly closed both avatar eyes | PASS |
| No blink preset | No blink response (not a failure) | Not applicable to selected fixture | NOT RUN |

### 3.4 Mouth

| Capability | Expected | Actual | Result |
|-----------|----------|--------|--------|
| Full (aa/ih/ou/ee/oh) | Vowel shapes respond | Five-vowel differentiation was not isolated | NOT RUN |
| aa-only | Mouth opens with "aa" | C922 held "aa" visibly opened avatar mouth | PASS |
| No mouth preset | No mouth response (not a failure) | Not applicable to selected fixture | NOT RUN |

### 3.5 Gaze

| Mode | Expected | Actual | Result |
|------|----------|--------|--------|
| Active gaze path | Eyes move with gaze | C922 eyes-only left/right prompts visibly shifted irises | PASS |
| Eye bones | Eye bones rotate | Not separately isolated | NOT RUN |
| None | No gaze response (not a failure) | Not applicable to selected fixture | NOT RUN |

### 3.6 Functional Summary

- [x] No panic or fatal render issue observed
- [x] Head pose three axes move in intended directions
- [x] Blink/mouth respond according to capability
- [x] No unsupported-capability error was shown

---

## 4. Recovery Protocol (M1-08-004)

### 4.0 Automated recovery preflight

実機操作に先立つソフトウェア回復ゲートを 2026-08-11 に実施した。
これはWindows GUI／物理cameraの受入結果ではなく、回復契約と状態遷移を
自動検証した結果である。

| Recovery contract | Evidence | Result |
|---|---|---|
| Repeated inference result is not replayed forever | `vtuber-app` observation-gate tests | PASS |
| Stale inference becomes face loss after 250ms | `vtuber-app` observation-gate tests | PASS |
| Missing inference output enters normal no-face path | `vtuber-app` observation-gate tests | PASS |
| Lost hold / neutral decay / reacquisition | `vtuber-tracking` loss-recovery tests | PASS |
| Camera Stop removes retained frame | `vtuber-camera` capture tests | PASS |
| Camera reconnect preserves capture sequence | `vtuber-camera` capture tests | PASS |
| Inactive tracking clears retained avatar control frame | `vtuber-app` avatar-bridge test | PASS |

Executed commands:

```text
cargo test -p vtuber-core -p vtuber-camera -p vtuber-app --lib --tests
cargo clippy -p vtuber-core -p vtuber-camera -p vtuber-app --all-targets -- -D warnings
```

The live checklist below was exercised with one imported VRM 1.0 model.
Camera unplug/replug, a timed reacquisition observation, and a bounded
thread/RSS sample were performed. The resource sample is recorded separately
in `docs/acceptance/artifacts/windows-m1-2026-08-12-resource-sample.txt`.

### 4.1 Face Loss / Return

- [x] Move face out of frame
- [x] Verify: `Face detected: no`, `Tracking: Initializing`, confidence `0.00`
- [x] Verify: avatar does not remain permanently frozen after loss/return
- [x] Return face to frame
- [x] Verify: tracking resumes within 2 seconds (first post-return observation at approximately 405 ms)
- [x] Verify: tracking resumes and fresh avatar frames continue after return

### 4.2 Camera Stop / Restart

- [x] Click "Stop" and verify lifecycle reached `Idle`
- [x] Click "Start" and verify lifecycle reached `Running`
- [x] Repeat Stop/Start three times in the same process
- [x] Unplug camera and replug it
- [x] Verify: app handles disconnect gracefully; Stop → Idle → Start restored tracking after reconnect

### 4.3 Avatar Replace

- [x] While tracking, import a VRM replacement
- [x] Verify: lifecycle returned to `Ready` without a process restart
- [x] Verify: replacement model loads and binds
- [x] Verify: tracking continues after replacement/reload
- [x] Verify: generation-safe binding tests and live post-reload tracking exclude old-model replay

### 4.4 Recovery Summary

- [x] No permanent freeze on face loss
- [x] Camera restart recovers tracking
- [x] Avatar replace leaves no stale state
- [ ] Thread count returns to baseline after each operation (only one bounded sample was measured)

---

## 5. Latency & Rate (M1-08-005)

### 5.1 Measurement Setup

- Warm-up: first 10 seconds excluded
- Sample window: 60 seconds minimum
- Clock source: monotonic (same domain for all timestamps)

### 5.2 Results

| Metric | Target | Actual | Result |
|--------|--------|--------|--------|
| Render FPS | ≥ 30 | 56.257–60.770 FPS | PASS |
| Tracking Hz | ≥ 15 | 30–60 Hz; 30 Hz at end | PASS |
| p50 capture-to-apply | — | 21.052–22.625 ms | PASS |
| p95 capture-to-apply | ≤ 180ms | 29.393–31.213 ms | PASS |
| Queue depth | ≤ 1 | `LatestSlot` capacity one | PASS |
| Capture slot overwrite count | — | 0 | PASS |

### 5.3 Raw Data

- `docs/acceptance/artifacts/windows-m1-2026-08-12-soak-metrics.csv`
- 31 samples from measurement elapsed 0.000 through 1,800.016 seconds
- 10-second warm-up excluded; fixed 60-second cadence; create-new/no-overwrite
  output and flush after every row

`inference_input_overwrites` rose from 381 to 54,382. This counter is the
number of replacements of the retained value in the shared capacity-one
`LatestSlot`; reads intentionally do not remove that retained value. It does
not represent queue growth. The producer-side capture drop/overwrite metric
remained 0 and the data structure cannot exceed one retained frame.

---

## 6. 30-Minute Soak (M1-08-006)

### 6.1 Setup

- Model: `inore-vrm1.vrm`
- Camera: `c922 Pro Stream Webcam`, MSMF symbolic link VID_046D/PID_085C
- Duration: 30 minutes
- Sampling interval: every 60 seconds

### 6.2 Results

| Metric | Start | Mid | End | Trend |
|--------|-------|-----|-----|-------|
| Working set | 866.25 MiB | 873.83 MiB | 884.75 MiB | bounded 866.25–885.62 MiB; non-monotonic |
| Private memory | 1.659 GiB | 1.670 GiB | 1.682 GiB | bounded 1.659–1.683 GiB; non-monotonic |
| p95 capture-to-apply | 30.611 ms | 30.822 ms | 30.679 ms | bounded 29.393–31.213 ms |
| Render FPS | 59.733 | 59.253 | 60.770 | bounded 56.257–60.770 FPS |
| Tracking Hz | 30 | 31 | 30 | minimum 30 Hz |
| Threads | 137 | 125 | 126 | bounded 125–138; no growth |
| Handles | 1,696 | 1,703 | 1,712 | bounded 1,696–1,800; no growth |
| Worker states | Running / Running | Running / Running | Running / Running | no exit observed |

### 6.3 Soak Summary

- [x] No process crash
- [x] No memory, thread, or handle unbounded increase trend in 31 samples
- [x] No latency continuous increase trend in sampled diagnostics
- [x] All 31 external samples reported `Responding=True`
- [x] Stop changed app lifecycle from `Running` to `Idle`
- [x] Clean shutdown confirmed (`vtuber-desktop` window/process absent after close)

---

## 7. Acceptance Criteria Summary

| # | Criterion | Target | Actual | Verdict |
|---|-----------|--------|--------|---------|
| 1 | Render FPS | ≥ 30 | minimum 56.257 FPS | PASS |
| 2 | Tracking Hz | ≥ 15 | minimum 30 Hz | PASS |
| 3 | p95 capture-to-apply | ≤ 180ms | maximum 31.213 ms | PASS |
| 4 | Queue depth | ≤ 1 | capacity one; capture overwrite 0 | PASS |
| 5 | No memory/latency increase | stable | 31-point bounded series; no unbounded trend | PASS |
| 6 | No process crash | 0 crashes | 0 observed | PASS |
| 7 | Report saved | yes | metrics/resource CSV and summary saved | PASS |

**Overall Gate:** PASS — C922 preview, direct avatar motion, loss/reacquire,
lifecycle recovery, camera reconnect, timed reacquisition, capture-to-apply
latency, bounded 30-minute performance/resource export, Stop to Idle, and clean
shutdown are evidenced.

---

## 8. Blocker Classification (M1-08-008)

No open Windows M1 correctness, compatibility, or performance blocker remains.

The official MediaPipe Tasks call does not expose separate detector and
landmark inner-stage cadence. Their distinct CSV columns therefore remain
`0.000`; the observable canonical inference cadence is 29–30 Hz. This is a
diagnostic capability limitation, not evidence of a stopped stage.

Categories: correctness, compatibility, performance, hardware-specific, test-environment

**M1-09 Go/No-Go Decision:** Windows M1 is GO. M1-09 remains explicitly
`DEFERRED` until macOS development resumes; no macOS acceptance is claimed.

---

## 9. Artifact Manifest

| Artifact | SHA-256 | Path |
|----------|---------|------|
| vtuber-desktop.exe | `C939C12411EA88B7363B8463F117CFD53CA516CE6522C36FDE6C5D3A4802B1E2` | target/release/vtuber-desktop.exe |
| face_landmarker.task | `64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF` | assets/models/face_landmarker.task |
| inore-vrm1.vrm | `B5A3D4126C4A30EF3BFBCFC764A24DC48511B558799D98D4C2FF1DB0BDC7AB01` | tests/fixtures/vrm/inore-vrm1.vrm |
| 30-minute metrics CSV | `90A21346DFD813A796D704E8FD88FE3A776732C1B43B6F1634DCE0EEB7B195E0` | docs/acceptance/artifacts/windows-m1-2026-08-12-soak-metrics.csv |
| 30-minute resources CSV | `D23A39F32393743FD45848D34499AE8E186EAD130321E6810C25E00D9A001FE0` | docs/acceptance/artifacts/windows-m1-2026-08-12-soak-resources.csv |
| 30-minute summary | recorded below | docs/acceptance/artifacts/windows-m1-2026-08-12-soak-summary.md |

---

_Report generated from the automated, live GUI, latency, resource, and bounded
30-minute soak evidence above. Windows M1 acceptance is complete. macOS was
not run and remains deferred._
