# Architecture Decision Records

基準日: 2026-09-01

| ID | Status | Decision |
|---|---|---|
| ADR-001 | Superseded by ADR-009 | 旧顔推論runtimeとmodel artifactの評価履歴 |
| ADR-002 | Superseded in format scope by ADR-011 | `bevy_vrm1`を唯一のVRM runtimeとして採用 |
| ADR-003 | Accepted | Windows／macOS限定とcamera backend |
| ADR-004 | Accepted | tracking結果のVRM適用順と座標変換 |
| ADR-005 | Accepted | frame pipelineのlatest-value semantics |
| ADR-006 | Accepted | user VRM importとBevy asset source |
| ADR-007 | Proposed; Q2-04で確定 | Windows／macOS packaging |
| ADR-009 | Accepted for M1-08-015 | MediaPipe Face Landmarker production runtime |
| ADR-010 | Accepted | Head-relative gaze coordination |
| ADR-011 | Accepted | VRM 0.x/1.0 normalization into the existing `bevy_vrm1` runtime |
| ADR-012 | Accepted | Optional NDI sender boundary and bounded latest-value transport |
| ADR-013 | Accepted for Issue #49; release gate pending | Guarded application-local NDI runtime staging |
| ADR-014 | Accepted for Issue #51 | ARKit52 contract and effective Perfect Sync capability inspection |
| ADR-015 | Accepted for Issue #50 child #52 | Rust GNM Head v3 model boundary and sparse evaluator |
| ADR-017 | Accepted for Issue #50 / #56 | Perfect Sync 52 の VRM custom expression 適用境界 |
| ADR-019 | Accepted for Issue #167 | Position-aware upper-body solve / root-body motion writer (vendored `bevy_vrm1` extension) |
| ADR-020 | Accepted | Idle motion policy; legacy procedural breathing writer retired |
| ADR-021 | Accepted | 長時間tracking喪失時のdefault pose復帰とloss-scoped breathing（micro-motion層） |
| ADR-022 | Accepted | Perfect Sync 52はMediaPipe直接駆動とする（学習動的prior不採用） |

番号は再利用しない。採用済み判断を変更するときは元ADRを削除せず、Statusを`Superseded`へ変更し、新しいADRから参照する。
