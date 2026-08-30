# ADR-005: frame pipelineにlatest-value semanticsを採用する

Status: Accepted  
Date: 2026-08-04

## Context

camera capture、face inference、renderは異なるrateで動く。通常queueへ全frameを積むと、inferenceがcaptureより遅い場合に古い顔姿勢を順番に再生し、latencyとmemoryが時間とともに増える。

## Decision

- cameraからinference、inferenceからBevy main threadの両境界に容量1件の`LatestSlot<T>`を使う。
- publish時に未消費値があれば置換する。
- source frame sequenceとmonotonic timestampを全stageで保持する。
- consumerは最後に読んだsequenceより新しい値だけを採用する。
- overwritten count、publish count、consume countをmetricsへ出す。
- stop tokenとslot closeでwaitを解除し、workerを明示joinする。
- control commandは小容量bounded channelとし、frame dataをcontrol channelへ混ぜない。

## Consequences

throughput不足時にはframeがdropされるが、latencyは累積しない。全frame処理が必要なoffline analysisには別runnerを使う。performance評価ではFPSだけでなく、drop数とcapture-to-apply latencyを同時に記録する。

## Rejected alternatives

- unbounded queue: latencyとmemoryが増える。
- large bounded queue: 古いframeを処理する問題が残る。
- inferenceをBevy main threadで実行: renderをblockする。
- frameごとのthread／task生成: schedulingとallocation overheadが増える。
