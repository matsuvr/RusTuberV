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
  観測時刻差での適応平滑化。`ArmTrackingState`/`ArmTrackingProfile`と
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
平滑化は新しい観測だけで進め、正のcapture時刻差を`NonZeroU64`で渡す。
render時刻で同じ観測を再投入しない。描画補間は別段階。

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
遮蔽時の動作は片腕単位の短い維持→既存の仮想腕への連続復帰と、再観測からの連続取得。
欠損を原点や既定長で捏造せず、無期限維持・別推論器への自動切替は入れない。

## 実機評価

`vtuber-tracking::arm_evaluation::evaluate_arm_sequence`が記録した腕系列から
到達誤差、骨長誤差、静止区間の変動、肘平面の最大変化、capture→display遅延を集計する。
計測値の取得だけをBevy/時刻側に置き、集計は純粋関数にする。

Windows x86_64では`cargo fmt --all -- --check`、`cargo test --workspace`、
`cargo clippy --workspace --all-targets`、nativeのPose workerテストが通る。
macOSでのビルド・FFI fixture・着座/机/腕交差/伸び切り/復帰の映像確認は未実施。
平滑化係数と時間定数は実測前の出発値であり、品質保証値ではない。
