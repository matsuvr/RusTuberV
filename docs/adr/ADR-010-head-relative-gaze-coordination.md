# ADR-010: head-relative gaze coordinationとVRM LookAt統合

Status: Accepted and implemented by Q2-06-002 (2026-08-12)
Date: 2026-08-12

## Context

従来はhead poseとeye gazeを「独立経路」と記述し、`Option<GazePose>`、未校正係数のradian化、独自eye-bone writerを組み合わせていた。この構成では有効な正面0値と計測不能を区別できず、loss時に前回回転が残り、モデル作者の`LookAtType`とVRM range mapを無視し得る。MediaPipeのeye coefficientsとhead transformation matrixは別の観測だが、眼球を独立world transformへする根拠にはならない。

## Decision

tracking契約は正規化された`GazeSignal`を用いる。horizontal／verticalは`[-1,1]`、confidenceは`[0,1]`で、stateはTracked／Degraded／Unavailableを明示する。正面はTrackedかつ0、計測不能はUnavailableであり同じ値にしない。

MediaPipeの左右眼を個別に算出し、blink／occlusion weight付き平均と左右一致度から共通eye-in-head信号とconfidenceを得る。neutral取得時には左右のhorizontal／vertical baselineを保存し、差し引き後にfinite検査とclampを行う。眼球専用のフレームレート非依存指数filterはtracked half-life 0.055秒、neutral return 0.150秒、unavailable hold 0.080秒を初期値とする。

アバター適用は次の階層合成と等価でなければならない。

```text
eye_world = current_head_world
          * eye_socket_chain
          * animated_eye_base_local
          * eye_in_head_delta_local
```

`vtuber-avatar`は正規化信号をVRM LookAt空間のdegreeへ変換し、符号を反転する。direct inputはVRM rootへ置き、world targetを生成しない。`bevy_vrm1`は`LookAtProperties`のinner／outer／up／down range mapとrest orientationを正本としてBone出力を適用し、Expression出力は既存のcoalesced `ModifyExpressions`経路へ渡す。Eye translation、scale、`GlobalTransform`は書き換えない。

能力集合と選択backendを分ける。宣言された`LookAtType`が利用可能ならそれを使用し、壊れている場合だけdiagnostic付きalternate fallbackを選ぶ。metadataがない場合は完全な4方向Expression、両eye bone、partial Expression、Noneの順とする。同一frameでBoneとExpressionを同時適用しない。

実行順はAnimation、direct BodyTracking、direct LookAt、Expression update/apply、Node Constraint、transform propagation、SpringBoneの意味順序を固定する。legacy cursor／target LookAtは維持する。

loss時は最後の有効gazeを短くholdし、明示的なneutralへ戻す。Searchingはneutralを出力し、reacquisitionは現在値から新しいtracked値へ連続補間する。avatar replacement、reset、recalibrationでfilter／additive gaze stateを破棄する。

## Rejected alternatives

- webcam gazeからworld-space target entityを作る方式: head-relativeなeye-in-head観測をworld targetへ偽装し、ownershipと符号を曖昧にする。
- gazeから追加head rotationを生成する方式: cameraで計測済みhead poseへ二重加算になる。
- raw係数を物理radianと呼ぶ方式: calibration上の根拠がない。
- eye boneをrestへ毎frame上書きする方式: animation baseを破壊し、非identity restで誤る。
- BoneとExpressionの同時適用: model作者の意図とVRM overrideを破る。

## Consequences and migration

`Option<GazePose>`は正本から外れ、centerとunavailableが型で区別される。永続化されている旧profileがgaze baselineを持たない場合は0 baselineを後方互換defaultとし、次のrecenterで更新する。既存`Q2-06-001`のBodyTracking履歴は変更しない。vendored patchのbase revisionとlicenseは維持する。

## Validation

typed core contract、左右符号／一致度／blink、neutral baseline、30／60／120fps応答、loss／reacquisition、non-identity rest、range map、backend排他、animation base、state cleanup、legacy LookAt、実行順を自動testする。Windows実camera／licensed VRMのvisual acceptanceは自動testと分離し、未実施時はPASSとしない。
