# ADR-023: GNM 実装の main ツリーからの完全除去

Status: Accepted
Date: 2026-09-05
Related: ADR-015, ADR-017, ADR-018, ADR-022, `docs/arkit-ablation-report-20260828.md`,
`docs/arkit-ablation-report-20260830.md`

## Context

ADR-022 で Perfect Sync 52 の式源は MediaPipe direct 経路に確定し、GNM は
production 採用を見送った（値 MAE 0.174–0.217 vs direct 0.098、blink pulse
全滅）。その時点では `vtuber-gnm` クレート、`GNM Temporal (Experimental)` /
`GNM Shadow` backend、teacher 系評価ツール群を実験・品質監視用に維持する方針
だった。

その後の研究でも GNM が direct を上回る結果は得られず、GNM を採用しないことが
最終決定された。実験 backend と評価ツールを main に残り続けさせる合理的な根拠
が消えたため、コードベースから完全に切り離すこととした。

## Decision

1. main ツリーから GNM 実装とその依存をすべて削除する。
   - `crates/vtuber-gnm` クレート
   - `vtuber-tracking` の GNM 依存モジュール群（`gnm_*`、`ab_backend` /
     `authority_gate`、`causal_prior*`、`teacher_*`、`reduced_*`、
     `unified_ablation`、`calibration/gnm_identity` 等）
   - UI の backend 切り替え（`FaceTrackingBackendSelection`、設定の
     `face_tracking_mode` 永続化、diagnostics の requested / authority /
     fallback 表示）。production backend は Direct MediaPipe のみとなる。
   - xtask の teacher 系コマンドと `ab-report` / `temporal-report`、
     standalone の `tools/teacher-capture`
   - `assets/models` の `gnm_head.npz`、`canonical_face_model.obj`、
     dense mapping の manifest エントリ
2. 削除前の実装は `archive/gnm` ブランチ（main HEAD `7226c25` 時点）に保存し、
   履歴参照はそこへ向ける。main では ADR を含む歴史記録（本 ADR、ADR-015、
   ADR-018、ablation レポート）は削除しない。
3. ADR-015（GNM Head v3 model boundary）と ADR-018（GNM identity calibration
   lifecycle）は `Superseded by ADR-023` とする。ADR-022 の Decision 3
   （GNM 実験 backend の維持）は本 ADR が上書きする。ADR-022 の Decision 1・2
   （direct 経路の production 採用、学習 prior 不採用）は有効のまま。
4. MediaPipe direct 経路（ADR-009 / ADR-022）が唯一の production 式源であり、
   `vtuber-tracking` は direct 経路に必要なモジュール（pipeline、pose、
   filter、calibration、state machine 等）のみを持つ。

## Consequences

- workspace から `vtuber-gnm` 依存が消え、tracking/app/xtask の
  GNM 関連コード約 5.6 万行が除去される。manifest には GNM モデルが残らず、
  配布・起動時の GNM アーティファクト検証が不要になる。
- GNM 実装を再評価する場合は `archive/gnm` ブランチから復元する。その際は
  ADR-022 Decision 4 の手順（同一 ablation 手順で direct と比較）を踏む。
- GNM 導入のために追加された設定項目・UI・diagnostics 欄は削除済みで、
  設定ドキュメントの `face_tracking_mode` キーは既存ファイルでも無視される。

## Alternatives considered

- 実験 backend / 評価ツールだけ残す（ADR-022 維持）: 採用しないことが確定した
  以降は保守コストだけが残るため却下。
- クレートだけ残して runtime 経路を削る: `vtuber-tracking` の GNM 依存型が
  多く、部分削除はコンパイルが通らず、結局依存閉包の全削除が必要になるため
  却下。
