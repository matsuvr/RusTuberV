# Performance and Latency Test Plan

基準日: 2026-08-04

## 1. 目的

render FPSだけでなく、camera captureからavatarへ反映されるまでの遅延とjitterを測定する。queueが蓄積して見かけ上のFPSだけを維持する失敗を検出する。

## 2. 記録するstage

```text
captured_at
inference_started_at
inference_finished_at
received_by_main_at
control_generated_at
applied_to_avatar_at
```

算出する値:

- capture interval
- inference duration
- result transfer delay
- tracking／main-thread delay
- capture-to-apply

## 3. 固定条件

各runで次を記録する。

- git commit
- Bevy version
- `bevy_vrm1` revision
- inference model id／SHA-256
- VRM SHA-256
- OS version
- CPU
- GPU
- RAM
- camera model
- camera backend
- resolution／fps／pixel format
- preview ON／OFF
- diagnostics ON／OFF
- release／debug

## 4. protocol

### A. Neutral jitter

30秒正面を維持する。

指標:

- head rotation RMS
- gaze RMS
- blink false positive
- mouth false positive

### B. Slow sine

yawを左右へゆっくり往復する。

指標:

- phase lag
- amplitude attenuation
- continuity

### C. Step response

正面から素早く約30°横を向き、停止する。

指標:

- 10–90% rise time
- overshoot
- settle time

### D. Tracking loss

顔を1秒隠し、戻る。

指標:

- tracking-state transition
- neutral decay time
- reacquisition time
- output discontinuity

### E. Continuous run

30分動作させる。

指標:

- memory every 60s
- p50／p95／p99 latency every 60s
- total captured／processed／overwritten
- worker restart
- error count

## 5. acceptance levels

### 5.1 Milestone 1 minimum gate

WindowsとmacOSの縦断MVPが最低限満たす値である。

- render: 30 FPS以上
- tracking output: 15 Hz以上
- capture-to-apply p95: 180 ms以下
- `LatestSlot` retained count: 常に1以下
- 30分でlatencyの単調増加なし
- 30分でunbounded memory growthなし
- camera／inference workerが終了時にjoinされる

### 5.2 Quality target

最適化後に目指す値であり、`DESIGN.md`の性能目標と一致させる。

| 指標 | Windows Tier 1 | macOS Apple Silicon Tier 1 |
|---|---:|---:|
| render | 60 FPS目標 | 60 FPS目標 |
| tracking output | 25 Hz以上 | 25 Hz以上 |
| capture-to-apply p50 | 70 ms以下 | 80 ms以下 |
| capture-to-apply p95 | 110 ms以下 | 120 ms以下 |

minimum gateを通ってもquality targetを満たさない場合、未達stage、hardware、model、設定をreportへ明記する。filterを強くして遅延を隠してはならない。

## 6. output

```text
docs/experiments/<date>-<os>-<machine>/
├─ environment.toml
├─ metrics.csv
├─ summary.md
└─ optional-user-recorded-video-reference.txt
```

camera frameそのものを自動保存しない。映像記録が必要な場合はユーザーが明示的に外部screen recorderを使い、repository artifactへ自動添付しない。
