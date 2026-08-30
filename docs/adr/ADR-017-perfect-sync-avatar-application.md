# ADR-017: Perfect Sync 52 の VRM expression 適用境界

- Status: Accepted for Issue #50 / #56
- Date: 2026-08-21

## Context

Issue #50 の GNM パイプラインが生成する ARKit52 coefficient を、VRM の
custom expression に適用する境界が必要になった。モデルごとに expression
の有無と morph bind の解決結果が異なるため、tracking 層が VRM/Bevy の型や
モデル固有名を知ることはできない。また、既存の粗い blink/mouth/gaze 経路と
詳細係数を同時に適用すると二重駆動になる。

## Decision

`AvatarControlFrame` に、検証済み `Arkit52Coefficients` の optional な値を
追加する。`None` は従来の coarse expression 経路を維持する。係数の名前解決は
avatar adapter の `ExpressionEntityMap` に限定し、ARKit52 の canonical
PascalCase と明示された lower-camel alias の完全一致だけを受け付ける。
fuzzy matching、任意の大文字小文字変換、MediaPipe の `_neutral` 名は使わない。

モデルの capability は `present` と `effective` を分離する。metadata に
expression があっても resolved morph bind が 0 件なら送信対象にしない。
部分的な Perfect Sync は正常な状態として扱い、coarse writer を置き換えられる
だけの coverage がある domain だけを detailed authority とする。

`PerfectSyncFaceAuthority` は blink、jaw/lip の mouth/lower-face、eye-look を
別々に判定する。blink は左右2 channel、eye-look は左右8方向すべて、
mouth/lower-face は jaw/lip 27 channel すべてが effective のときだけ authority
を持つ。TongueOut、brow、cheek、eyelid、nose のように既存 coarse writer が
ない supplemental channel は、partial model でも有効 channel を適用する。
authority のない coarse domain は既存 blink/mouth/VRM LookAt pathへ fallback
し、同一 domain の detailed と coarse を同時に駆動しない。

既存の expression coalescing tracker が detailed/coarse 経路の切り替え、明示的
な 0 への遷移、avatar generation の変更を処理する。tracking lost/neutral と
avatar unload では detailed state を保持せず、前フレームに残った custom
expression にはゼロ更新を発行できる状態にする。

## Consequences

- `vtuber-core` は固定長の検証済み ARKit52 payload だけを扱い、VRM runtime に
  依存しない。
- `vtuber-avatar` が capability 検出、exact name 解決、二重駆動防止、LookAt
  fallback を所有する。
- GNM のモデル、fitter、decoder はこの適用層に依存せず、後続 Issue #53--#55
  で別境界として追加する。
- 実 VRM とカメラを接続した視覚確認、Windows/macOS 実機確認はこの変更では
  実施していないため、受入根拠として扱わない。
