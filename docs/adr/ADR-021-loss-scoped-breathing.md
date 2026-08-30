# ADR-021: 長時間トラッキング喪失時のデフォルトポーズ復帰とロストスコープド呼吸

Status: Accepted
Date: 2026-08-30
Related: Issue #172, Issue #20, Issue #180, ADR-004, ADR-019, ADR-020

## Context

配信者はカメラ信号の消失（ケーブル抜け・デバイス停止・capture stop）や、薄暗い環境・逆光などでの長時間の顔認識不能に直面する。従来の挙動:

- 顔認識不能時は tracking 側の glide/decay と `bevy_vrm1` の half-life スムージングによりデフォルトポーズへ戻るが、その後の姿勢は完全に静止する。
- カメラ信号が消失した場合は control frame が途絶え、position/pose 両 bridge が neutral input を書くため、loss idle episode（Issue #172 の micro-motion）は一度も開始されない。アバターは完全な静止画となり、配信が「死んで」見える。

ADR-020 は常に動作するプロシージャル idle オシレータ（旧 #20 breathing writer）を retire し、「新たな idle source は明示的な設計判断・型付き所有権・決定論的テスト・合成ポリシーを伴って追加する」ことを求めている。本 ADR はその明示的な設計判断である。

## Decision

1. **長時間喪失時のデフォルトポーズ復帰は既存機構で維持する。** 顔喪失時は tracking 側の慣性 glide → ニュートラル decay（vtuber-tracking `LossRecovery`）と、rotation/position bridge の非 tracked 状態（`active: false`）への half-life スムージング収束により、avatar はゆっくり既定の rest ポーズへ戻る。追加の writer は作らない。
2. **ロストスコープド呼吸を micro-motion 層へ追加する。** `MicroMotionProfile` に `breath_period`（seed 4.0 s ≈ 15 回/分）と `breath_amplitude_ratio`（seed 0.010 × body scale）を追加し、`IdleTarget.translation_y` を「喪失からの経過時間の純粋な正弦」として供給する。これは常に動作するオシレータではなく、**loss idle episode の envelope（既存の 4 秒 smoothstep fade）の内側でのみ動く**bounded 追加レイヤーであり、ADR-020 の禁止する always-on oscillator とは異なる。
3. **カメラ信号消失を loss episode として扱う。** control frame が存在しない tick でも position bridge は `LossIdleState` を untracked として進め、blend が正のあいだ idle sway + breathing を publish する。pose bridge も同様に、frame 無しでも blend が正のあいだ idle yaw/pitch を publish する。これにより「顔が識別できない」「カメラが無い」のいずれでもデフォルトポーズで呼吸と揺らぎが続く。
4. **hips 書き込みは行わない。** 呼吸は `BodyTrackingPositionInput.head_offset.y`（semantic +y 上方向）経由で流れ、`hips.translation` の所有権は ADR-020 のまま authored/animated pose にある。新しい system は追加せず、writer 所有権は既存の 2 bridge に留まる。

### 合成ポリシー

- 呼吸は sway（x/z, yaw/pitch）と同一の blend envelope でスケールされる。blend 0 で無害、episode 開始から 4 秒でフル振幅。
- episode の再開（短い再取得後の再ロスト）では blend が 0 からやり直すため、位相のリセットが可視のジャンプになることはない。
- 決定論: 呼吸は `elapsed_since_loss` のみの関数であり、30/60/120 FPS 等価評価で同一曲線になる。OS RNG は使わない。

## Writer ownership（ADR-020 表への追記）

| Channel | Owner |
| --- | --- |
| `BodyTrackingPositionInput.head_offset.y`（ロスト中の呼吸） | micro-motion 層 → `update_body_tracking_position_input`（既存の唯一 writer 内） |

## Validation

- `vtuber-tracking` unit tests: 呼吸の原点ゼロ・周期振動・長時間エピソードでの振幅 bound・profile validation。
- `vtuber-avatar/tests/idle_contract.rs`: カメラ無音（control frame 無し）でも idle episode が ramp し、position input が idle 値で active になること。hips translation は従来どおり不変。
- 既存の「retired breathing writer は PostUpdate schedule に存在しない」検証は、新 system を追加しないためそのまま成立する。

## Consequences

- カメラ・顔のいずれが不調でも、配信はデフォルトポーズで呼吸と揺らぎを続け、完全な静止画にならない。
- 意図的な capture stop 中もアバターが動き続ける。静止が望ましい運用では capture ではなく avatar unload を行う。
- 呼吸の周期・振幅は `MicroMotionProfile` の typed パラメータとして検証可能であり、将来の idle source とは独立に調整できる。
