# ADR-009: MediaPipe Face Landmarker production runtime

Status: Accepted for M1-08-015 Windows implementation and gate
Date: 2026-08-11
Supersedes: the production decision in ADR-001; ADR-001 remains historical evidence

## Context

The previous UltraFace + PeppaPig-98 + custom planar-pose path cannot provide a
reliable production neutral or expression contract. Its observed Windows C922
landmark rate was 0.074–0.347 Hz against a 30-sample/5-second calibration
gate, its pose output reached physically implausible multi-radian ranges, its
manifest normalization disagreed with the upstream PeppaPig contract, and its
expression mapping used MediaPipe indices outside the 98-point result. The
failure baseline is recorded in ADR-001.

The application still owns camera capture, worker lifecycle, orchestration,
tracking, filtering, and VRM control in Rust. The user-authorized design
change permits the official MediaPipe native Tasks runtime for the production
face task through one reviewed Rust binding.

## Decision

Use the official MediaPipe Face Landmarker Tasks pipeline as the sole production
face backend:

| Item | Fixed value |
| --- | --- |
| MediaPipe Tasks | `0.10.35` |
| Rust binding | [`nikicat/mediapipe-rs`](https://github.com/nikicat/mediapipe-rs) |
| Binding revision | `527037fa0fe1339750140283930bbb9560460e9e` |
| Binding crate version | `0.1.0` |
| Binding license | Apache-2.0 |
| Delegate | CPU |
| Mode | synchronous VIDEO / `detect_for_video` |
| Faces | one |
| Task bundle | `assets/models/face_landmarker.task` |
| Task bundle SHA-256 | `64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF` |

The worker constructs, uses, and drops `FaceLandmarker` inside the existing
inference worker. It does not move the runtime across a thread boundary and it
does not access Bevy ECS. The existing capacity-one `LatestSlot<VideoFrame>`
and latest-only result boundary remain in force. MediaPipe and the legacy
UltraFace/PeppaPig stack must never run concurrently.

The task bundle is consumed as a task bundle. Its internal TFLite files are not
extracted into an independent application pipeline. The binding's verified
download path may fetch the official MediaPipe 0.10.35 native library on first
use for the Windows research build. The library source and version, task
bundle hash, result contract, and relevant timing are exposed in diagnostics.
Offline packaging without first-use network access is explicitly deferred to a
later Q2 packaging task.

## Output contract

One-face mode accepts only a result containing exactly 478 finite landmarks,
52 finite typed blendshape scores in `[0,1]`, and one finite affine 4x4 face
transformation matrix. The matrix is column-major; its rotation block is
validated and converted to the nearest proper rotation before quaternion
extraction. Matrix determinant and orthogonality error are retained as quality
diagnostics.

Zero faces is normal `NoFace` and retains source sequence and timestamps. More
than one face, malformed counts, unknown or duplicate blendshape categories,
non-finite values, invalid score ranges, or a malformed matrix are typed
output-contract errors. They must not be relabeled as `NoFace`, and the app
must not fabricate a detector confidence of `1.0`.

The engine-independent sample contract stores camera-to-face rotation and
translation, normalized face centre, 478 landmarks, a typed 52-category
blendshape set, source capture timestamp, inference start/end timestamps, and
matrix quality. MediaPipe category names are parsed once at the inference
boundary rather than stored as arbitrary strings throughout the app.

## Neutral and tracking consequences

The first valid result establishes the initial neutral automatically. Manual
`Recenter` uses up to 15 valid transforms no older than 300 ms; three or more
samples use robust quaternion and component-wise median statistics, otherwise
the newest valid sample is committed. Recenter never rejects ordinary head or
expression motion and waits for a face when no face is available.

Current pose is computed as `inverse(T0) * Tt`, with rotation and translation
composed as rigid transforms before the application basis conversion. Euler
angle subtraction is not used. Preview mirroring does not change inference or
tracking coordinates.

## Privacy and native-runtime boundary

The application processes face images and landmarks on-device and must not
transmit frames or landmark arrays. The MediaPipe project states that Tasks
APIs may send performance/utilization metrics; therefore this project does not
claim “no network activity” without an explicit measurement. The approved
native exception is limited to MediaPipe Tasks 0.10.35 through the pinned
binding revision above. Application crates retain `#![forbid(unsafe_code)]`;
Python subprocesses, OpenCV, Unity, ONNX Runtime, arbitrary native plugins,
and sidecar inference processes remain prohibited.

## Revalidation and consequences

M1-08-016 and M1-08-017 are reset to `PENDING` and must be revalidated against
the new outcome contract and runtime. M1-08-018 and M1-08-019 remain blocked;
M1-09 remains deferred. The old model artifacts remain only as explicitly
identified legacy evaluation evidence. M1-08-015-010 removes their default
production reachability by making the legacy detector/crop/ONNX feature opt-in
and keeping its commands explicitly research-only. The standalone Windows gate
must prove at least 15
results per second after warm-up, the 478/52/one-matrix contract, clean
Stop/Start, and no queue growth before later leaves are accepted.

## References

- [MediaPipe FaceLandmarkerOptions](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions)
- [MediaPipe Face Landmarker C++ API](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h)
- [MediaPipe Face Mesh wiki](https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh)
- [mediapipe-rs pinned revision](https://github.com/nikicat/mediapipe-rs/tree/527037fa0fe1339750140283930bbb9560460e9e)
