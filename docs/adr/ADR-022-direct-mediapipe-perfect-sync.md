# ADR-022: Perfect Sync 52 は MediaPipe 直接駆動とする（学習動的 prior の不採用）

Status: Accepted
Date: 2026-09-01
Related: ADR-009, ADR-014, ADR-015, ADR-017, ADR-018,
`docs/arkit-ablation-report-20260828.md`, `docs/arkit-ablation-report-20260830.md`

## Context

ARKit52 / Perfect Sync 係数の生成源として、ARKit teacher との同一タイムライン
比較で学習した causal linear prior（直近履歴＋速度＋残差の特徴から次フレームの
係数を予測する線形モデル）を direct MediaPipe 観測や GNM 投影の上に重ねる案が
あった。採否は数値で判断する前提で、held-out ablation を 2 ラウンド実施した。

- 2026-08-28（1 人・1,196 行）: 値 MAE 0.2183 vs no-prior GNM 0.2168 — 改善なし。
  出力はほぼ定数に崩壊し、validation テイクの blink pulse 17/17 が測定不能。
- 2026-08-30（2 人・1,734 行＋クロス人物評価）: 1 人目の held-out でも
  0.2169 vs 0.2168（改善なし）。2 人目の held-out では 0.1875 vs 0.1744
  （劣化 +7.5%）。1 人だけで学習した prior を別人物に適用しても 0.1876 と
  数値不変。blink pulse は 0/16 で全滅。
- direct MediaPipe は両人物で MAE ≈ 0.098 かつ blink pulse をすべて測定
  （peak 減衰 0.068–0.38、timing 誤差 13–33 ms）。

学習 prior の値は訓練行数（+45%）と評価人物の訓練への包含の双方に不感であり、
状態に条件付けしない平均回帰へ収束していることが確定した。

## Decision

1. production の Perfect Sync 52 expression 源は、MediaPipe Face Landmarker
   の blendshape 観測を直接 ARKit52 係数へ写像する direct 経路とする。
   temporal smoothing や学習 prior を介さず、観測値をそのまま検証済み
   `Arkit52Coefficients` として `AvatarControlFrame` に流し、ADR-017 の
   authority 境界で VRM expression に適用する。既定 backend は
   `Direct MediaPipe` のままとする。
2. 学習動的 prior（causal linear prior）は不採用とする。`PriorRuntime`
   （`vtuber-tracking`）と xtask の `teacher-fit-prior` / `teacher-ablation`
   は offline 評価基盤として保持するが、production のライブ経路には接続しない。
3. GNM Head v3（ADR-015）と identity calibration lifecycle（ADR-018）は
   `GNM Temporal (Experimental)` / `GNM Shadow` backend として実験・品質監視用
   に維持する。production の式源には採用しない。
4. 将来、動的 prior 系を再評価する場合は、同じ特徴量の行数を増やすのではなく
   モデルクラスの変更（identity-aware・非線形な状態条件付け）を前提とし、
   同一の teacher-ablation 手順（値誤差・微分誤差・jitter・blink/step timing、
   take 非依存 split、cross-person 評価）で direct 経路と比較する。

## Consequences

- アバターの Perfect Sync 52 は追加の学習アーティファクトなしで動作し、
  モデル配布・起動時間・式生成の推論コストは MediaPipe のみで完結する。
- direct 経路の観測ノイズは temporal layer なしで avatar に届く。平滑化が
  必要になった場合は learned prior ではなく、決定論的で観測同期のフィルタ
  （ADR-020 / ADR-021 の micro-motion 層と同様の明示設計）として別途判断する。
- `vtuber-gnm` と `PriorRuntime` は production の式源から外れるが、crate と
  評価ツール群は維持される。GNM shadow は引き続き direct の品質監視に使える。
- 学習 prior への投資（特徴量設計、fit/ablation tooling、2 人分の teacher
  収録データと ablation 手順）は、将来の動力学モデル再評価の基盤として維持する。

## Alternatives considered

- GNM Temporal ＋ learned prior を production 採用: 2 ラウンドの ablation で
  値誤差改善なし・blink 全滅のため却下。
- GNM 単体（no-prior）を production 採用: 値 MAE 0.174–0.217 で direct
  （0.098）に劣り、採用根拠がない。実験 backend に留める。
- direct に固定ルールの smoothing を今追加する: 効果の測定なしにレイテンシと
  追従遅れを払う根拠がないため見送る。必要になったら Decision 4 の手順で
  direct と比較して判断する。
