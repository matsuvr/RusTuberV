# Windows M1 30-Minute Soak Summary

- Task: `M1-08-019`
- Date: 2026-08-12
- OS: Windows 11 Pro 10.0.26200
- Camera: C922 Pro Stream Webcam, MSMF symbolic identity VID_046D/PID_085C
- Concurrent camera: ELECOM 2MP Webcam remained connected
- Avatar: `inore-vrm1.vrm`, SHA-256 `B5A3D4126C4A30EF3BFBCFC764A24DC48511B558799D98D4C2FF1DB0BDC7AB01`
- Face task: SHA-256 `64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF`
- Release binary: SHA-256 `C939C12411EA88B7363B8463F117CFD53CA516CE6522C36FDE6C5D3A4802B1E2`
- Warm-up: 10 seconds excluded
- Measurement: 1,800.016 seconds, 31 samples, 60-second cadence
- External resource measurement: 1,800.027 seconds, 31 samples

## Results

| Metric | Minimum | Maximum | Start | Mid | End |
|---|---:|---:|---:|---:|---:|
| Render FPS | 56.257 | 60.770 | 59.733 | 59.253 | 60.770 |
| Capture Hz | 29 | 31 | 30 | 29 | 31 |
| Inference Hz | 29 | 30 | 29 | 30 | 30 |
| Tracking Hz | 30 | 60 | 30 | 31 | 30 |
| Capture-to-apply p50 ms | 21.052 | 22.625 | 22.625 | 21.052 | 21.879 |
| Capture-to-apply p95 ms | 29.393 | 31.213 | 30.611 | 30.822 | 30.679 |
| Working set MiB | 866.25 | 885.62 | 866.25 | 873.83 | 884.75 |
| Threads | 125 | 138 | 137 | 125 | 126 |
| Handles | 1,696 | 1,800 | 1,696 | 1,703 | 1,712 |

- Capture slot overwrites/drops: 0
- Avatar frames applied at end: 108,601
- Avatar frames skipped: 0
- Capture worker: Running for all 31 application samples
- Inference worker: Running for all 31 application samples
- External process responding: True for all 31 resource samples
- Crash/hang: 0
- Shutdown: UI Stop changed Running to Idle; close removed the process/window

`inference_input_overwrites` is the cumulative replacement count of the
retained value in the capacity-one `LatestSlot`; it is not queue depth. The
slot cannot grow beyond one value, and the producer-side capture overwrite
metric remained zero.

The MediaPipe Tasks runtime exposes the face-landmarker operation as one call,
so detector and landmark internal cadences are not separately observable and
remain `0.000` in the CSV. Canonical inference remained 29–30 Hz.

## Verdict

PASS. Render FPS, tracking rate, capture-to-apply p95, capacity-one transport,
resource stability, responsiveness, and clean shutdown meet the Windows M1
acceptance criteria.

## Raw Artifacts

- `docs/acceptance/artifacts/windows-m1-2026-08-12-soak-metrics.csv`
- `docs/acceptance/artifacts/windows-m1-2026-08-12-soak-resources.csv`
