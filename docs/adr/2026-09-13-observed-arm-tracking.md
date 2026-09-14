# 観測した腕の追跡を既存IKへ接続する

状態: 方針採用。Pose FFI（#46）、純粋なtracking状態更新（#48）、Pose workerと
カメラ配信（#47）、既存compositorへの統合（#49）、評価関数（#50）を実装済み。
実機での映像品質とmacOS実行は未検証。

## 範囲

単眼WebカメラのMediaPipe Poseから肩・肘・手首を取得する。既存の
`solve_two_bone_arm` を再利用し、新しい全身IKライブラリは導入しない。
指・掌の姿勢推定、Hand/Holisticへの置換、全身追跡は今回の範囲外。
MediaPipe以外のネイティブランタイム、常駐Python、汎用プラグイン基盤は増やさない。

## 境界

- `mediapipe-rs`: 固定revを`vendor/mediapipe-rs`へ取り込み、所有するPose結果と
  安全な動画APIを追加した。FFI、ABI、結果解放だけを担当。アプリ側へunsafeを移さない。
- `vtuber-inference::pose_decode`: 世界座標33点から左右の6点を抽出する。
  不正な外部データはResult、人物なしは`PoseArmFrame.observation = None`。
  visibility/presenceの欠損を1へ補完しない。低信頼時の演出はここで決めない。
- `vtuber-core::arm_tracking`: 固定サイズの観測/目標/撮影時刻。Bevy、MediaPipeの
  ハンドル、骨Entityを持ち込まない。顔の`Landmark3`へPoseを詰め替えない。
- `vtuber-tracking::arm_tracking`: 固定キャリブレーション長、肩相対座標、
  render clock上のcritically damped平滑化（頭・身体と共通の`filter::damped`）。
  `ArmTrackingState`/`ArmTrackingProfile`と
  `step_arm_tracking`は入力値と前状態と`now`から値を返す純粋関数。左右独立の
  短い維持→仮想腕への復帰と再取得、伸び切り時の肘平面の連続性を含む。
- `vtuber-avatar::tracked_arm`: モデル長・rest-spaceへの変換と既存IKのみ。
  Transformを書かない。適用は既存`apply_default_arm_pose`の単一writerへ統合する。
- `vtuber-app`: カメラ共有、worker/slotの接続、既存の開始・停止・設定UIだけ。

## 座標と時間

Poseのメートル値は腰中心の推定値であり、カメラ画像のnormalized値や実測の
部屋座標ではない。肩位置を引き、PoseのY/Z符号を変換して、腕ターゲットは
「X=非ミラー画像右、Y=上、Z=カメラ側」とする。既存のhead translationは
Z=カメラから遠ざかる向きなので混用しない。ネイティブ接続時のfixtureで軸を確認する。

腕長は信頼できるキャリブレーション標本の上腕長+前腕長を固定する。毎frameの
推定長で割り直さない。演者の絶対位置をVRMへコピーせず、相対目標をモデルの腕長で
拡大する。肘は曲げ平面の観測であり、手首と肘を両方厳密一致させる制約ではない。

ミラーは`ArmTrackingTargets::mirrored`に集約し、X反転と左右交換を同時に行う。
入力画像のプレビュー反転とは独立。既存のmirror経路から二重適用しない。

追跡座標からIK rest-spaceへは明示的な単位Quaternionを渡す。同じモデル座標で
B=肩親のrest回転、A=現在回転、V=追跡viewからmodelへの回転なら
`tracking_to_rest = B * inverse(A) * V`。肩親の現在姿勢を除去せずにローカル回転を
適用すると胴体動作を二重適用する。原点は鎖骨でなく上腕骨の起点。

Poseは既存顔推論を待たせず、カメラの`Arc`画像を共有する。各結果に元画像の
`FrameSeq`と`captured_at`を残し、顔と腕の結果を同時撮影だったように偽装しない。
平滑化は顔・身体と同じrender clockに統一する: 最新観測を保持して毎tick再投入し、
頭・身体と同じ時間定数のcritically damped 2次フィルタで追従させる。新しい観測の
取り込み（キャリブレーション、再ターゲット、観測blend）だけを1回に限定し、
描画補間は同じ純粋状態更新の中で行う。

## ライブ接続

- `vtuber-inference::backend::mediapipe::MediaPipePoseRuntime`が1台のカメラ映像を
  顔とは別のcapacity-one slotで推論する。`run_pose_worker`は顔workerを待たせない。
- `vtuber-app`の`pose_runtime`がカメラのfan-out slotを1回だけ配線し、ON/OFFで
  workerを開始・停止する。`step_arm_tracking`の結果を`TrackedArmControl`へ渡す。
- `vtuber-avatar`の`update_tracked_arm_targets`が`ArmPoseSourceKind::TrackedPose`
  のときだけ`DynamicArmTargets`を書き、`apply_default_arm_pose`が唯一のwriterのまま
  最終Transformを書く。virtual targetとは`ArmBlendWeight`で連続合成する。
- 設定UIに「腕のトラッキング」の有効/無効と再校正を追加した（初期OFF）。

左右のvisibility判断、キャリブレーション標本の選別、伸び切り時の肘平面の連続性、
ロスト/復帰は純粋関数として実装・テスト済み。実測モードへ仮想腕用のtorso lag/
swivel/shoulder trim/twist緩和を無条件に重ねず、観測目標を保つ。

映像評価後のロスト/復帰は顔パイプラインと同じ形にした。保持した観測を毎tick再投入
してもチャンネルは在席のまま扱い、欠測（人物なし・低visibility）または検出飛びだけが
復帰タイムラインを進める。手首が1観測で校正腕長の`max_wrist_step`（初期0.75）を
超えて飛んだ場合は外れ値として棄却し、以降も最後に採用した目標から離れている限り
採用しない。手首を本当に見失った後の最初の観測は距離にかかわらず再取得として受け、
現在の権威から連続的に取得する。遮蔽時は片腕単位でhold 150 ms→既存の仮想腕へ
2 sで復帰し、再取得は`acquire` 500 msを上限に残り分だけブレンドする。
欠損を原点や既定長で捏造せず、無期限維持・別推論器への自動切替は入れない。

## 実機評価

`vtuber-tracking::arm_evaluation::evaluate_arm_sequence`が記録した腕系列から
到達誤差、骨長誤差、静止区間の変動、肘平面の最大変化、capture→display遅延を集計する。
計測値の取得だけをBevy/時刻側に置き、集計は純粋関数にする。

Windows x86_64では`cargo fmt --all -- --check`、`cargo test --workspace`、
`cargo clippy --workspace --all-targets`、nativeのPose workerテストが通る。
macOSでのビルド・FFI fixture・着座/机/腕交差/伸び切り/復帰の映像確認は未実施。
hold/return/acquireと`max_wrist_step`は実機映像のフィードバックで更新した値であり、
品質保証値ではない。
