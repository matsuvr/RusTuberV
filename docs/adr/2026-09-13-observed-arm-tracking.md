# 観測した腕の追跡を既存IKへ接続する

状態: 方針採用。Pose FFIと純粋なtracking状態更新を実装済み。カメラ配信・
avatar合成・実機評価は後続Issue。

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

## ライブ接続の残作業

先行コードは左右のvisibility判断、キャリブレーション標本の選別、伸び切り時の
肘平面の連続性、ロスト/復帰、描画補間、実機での可愛さを完成させたものではない。
これらを実装・検証するまで実測モードを有効化済みとして表示しない。

明示的に採用する遮蔽時の動作は、片腕単位の短い維持→既存の仮想腕への連続復帰と、
再観測からの連続取得。肘だけが隠れた場合は手首追従と曲げ平面を分ける。
欠損を原点や既定長で捏造せず、無期限維持・別推論器への自動切替は入れない。

実測モードへ仮想腕用のtorso lag/swivel/shoulder trim/twist緩和を無条件に重ねない。
追跡中の肩・捻り補正は実際のFK後の手首/肘が目標を保つように設計する。
可愛さは観測動作を潰す一律内寄せやランダム揺れではなく、安定した肘・適切な
肩追従・身体に埋まらない手の軌道・遮蔽時の連続性で評価する。

Windows/macOSでのビルド・FFI fixture・着座/机/腕交差/伸び切り/復帰の映像確認は未実施。
先行の平滑化係数は実測前の出発値であり、品質保証値ではない。
