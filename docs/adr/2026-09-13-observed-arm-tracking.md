# 観測した腕の追跡を既存IKへ接続する

状態: 方針採用。Pose FFI（#46）、純粋なtracking状態更新（#48）、Pose workerと
カメラ配信（#47）、既存compositorへの統合（#49）、評価関数（#50）を実装済み。
実機での映像品質とmacOS実行は未検証。

## 範囲

単眼WebカメラのMediaPipe Poseから肩・肘・手首を取得する。既存の
`solve_two_bone_arm` を再利用し、新しい全身IKライブラリは導入しない。
指の関節角の推定、Hand/Holisticへの置換、全身追跡は今回の範囲外。
MediaPipe以外のネイティブランタイム、常駐Python、汎用プラグイン基盤は増やさない。

追記（掌の向き）: Poseの粗いhand keypointsでは掌平面がほぼ固まり、時々大きく
跳ぶため実用にならなかった。同じカメラフレームにMediaPipe Hand Landmarker
（`hand_landmarker.task`、Poseと同じworker内で同期実行）を追加し、21点のworld
landmarksを`ArmLandmarks.hand`へ載せる。手はhandednessではなく、同じ画像の
Pose手首に最も近い側へ対応付ける。掌平面法線はwrist/index MCP/pinky MCPから
求め、`vtuber-avatar::tracked_arm::align_palm_twist`が回旋を前腕と手首へ
折半して重ねる（下の追記を参照）。指の関節角の消費は引き続き対象外。

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

映像評価後のロスト/復帰は顔パイプラインと同じ形にした。追記（2026-09-16）: この
「同じ形」を共有コードに引き上げた。`vtuber-tracking::loss_blend` の
`LossBlendProfile`（hold/return/acquire）が腕チャンネルと顔パイプラインの両方の
唯一のタイムラインであり、腕は `LossBlend` の重みランプ、顔は同じ時間定数での
重み緩和（保持フレームの confidence を `loss_return_factor` でスケール）と
`acquire` 時間の smoothstep 再取得ブレンドで同じ挙動をする。既定値は
150 ms / 5 s / 1 s で両者共通。保持した観測を毎tick再投入
してもチャンネルは在席のまま扱い、欠測（人物なし・低visibility）または検出飛びだけが
復帰タイムラインを進める。手首が1観測で校正腕長の`max_wrist_step`（初期0.75）を
超えて飛んだ場合は外れ値として棄却し、以降も最後に採用した目標から離れている限り
採用しない。手首を本当に見失った後の最初の観測は距離にかかわらず再取得として受け、
現在の権威から連続的に取得する。遮蔽時は片腕単位でhold 150 ms→既存の仮想腕へ
5 sで復帰し、再取得は`acquire` 1 sでブレンドする。どちらもsmoothstepで始点と
終点の速度を0にする（飛びの接続を優先し、追従性は意図的に落とす）。腕位置の
平滑化時定数は0.15 s、掌平面は0.35 sとする。

掌の向きは、観測掌法線をrest空間へ写し、その手の解剖学的な掌法線
（index/little proximalのrest位置）との、solveが出した前腕長軸まわりの符号付き
角度を求めて追う。回旋は前腕と手首へ折半して重ねる: 角度の半分を`lower_arm`
へ（前腕長軸まわりのrollとしてsolutionへ書き戻す）、残り半分を`hand`のrest相対
deltaにする。前腕長軸は肘と手首を通るので、どちらの回転も骨位置を動かさず、
手と指は前腕に追従する。

折半するのは、このVRMが前腕のtwistボーンを持たず（`J_Bip_*_LowerArm`1本に
上腕の`J_Sec_*`だけ）、1関節に回旋を全部載せるとそこのskinningが潰れるため。
手首（hand/lowerArm混在の頂点）に全量を載せると「キャンディラッパー」状に
ねじれて手首がちぎれて見える。実測したこのモデルでは手首の混在頂点182、肘の
混在頂点248で、肘側の方が広い。折半で各関節のせん断が半減する。角度はpalm
チャンネルの重みでスケールするので、掌を見失えば復帰時間（5 s）で既定rollへ
戻る。滑らかさはtracking層の掌平面平滑化（0.35 s）と重みのランプが担い、
フレーム毎の速度制限や累積状態は持たない。

掌法線が前腕長軸と平行に近い（掌が前腕方向を向く）縮退時だけは回転軸が定まらない
ので補正をスキップし、solveのrollを残す。

肘は手の大きな移動に引きずられて腋が開く方向へ跳ぶことがあった。人間は意識
しない限り腋を大きく開かないため、肘の曲げ平面（肩-手首軸まわりの方向）の
角速度を`ELBOW_PLANE_MAX_RATE_RAD_PER_SEC`（3 rad/s）で制限する。制限は
観測肘の平面方向にだけ掛かり、手首位置とリーチは観測の平滑化結果のまま
変えないため、頭・身体の平滑化とは独立で、腕の追従が遅れるだけで済む。
観測肘が動かなければ平面も動かない。

観測肘の方向からリーチを逆算して肘を観測へ一致させる案（`seat_elbow_angle`）は
試作して破棄した。腕を上げると肩→手首方向と肩→肘方向のなす角が90度を超え、
リーチが不正に縮んで前腕が反転する（実機ログで前腕delta 178度）ためで、
肘の角度は解析solveのリーチに任せるのが正しい。

追記（2026-09-16）: 伸び切りの減衰は曲げ平面の回転量にだけ掛ける。腕が伸びるほど
観測平面のチャンネル重みを0へ落とす実装は、生のPose手首リーチが校正腕長の
まわりで揺れるたびに重みを0と1の間で切り替え、肘を観測平面と仮想平面の間で
1フレームに反転させていた（記録済み`mediapipe_pose_debug.csv`の再生で上腕
0.5 rad/フレーム、肘0.58 rad/フレーム）。減衰は平面の更新量にだけ掛け、
チャンネルの権威は観測の在席（`LossBlend`）だけに従わせる。腕が伸びて平面が
定まらない間は最後の整定した平面を保持し、曲がれば観測へ滑らかに戻る。入力の
リーチが腕長のまわりで揺れてもチャンネル重みは動かない。

追記（2026-09-16, 肘平面の反対側）: 保持する平面は観測方向へ線形ブレンドせず、
肩-手首軸まわりに短い向きへ回転させて寄せる。以前は観測平面が保持平面と反対側
（内積が負）のときブレンドを0にして保持平面を固定していたため、腕を体の前へ
大きく動かすと平面が反対側に残り続け、最短弧ソルバが上腕を約154度まで巻き
込んでいた（記録済み`propagation_debug.log`の`rUpperArm=153.74deg`、同時に
`pole=[-0.22,-0.16,-0.16]`が観測肘の前方に対し後方）。回転で寄せれば、軸を通り
抜けて反転することも、誤った側に固定されることもない。可視の連続性は描画
クロック側の`ELBOW_PLANE_MAX_RATE_RAD_PER_SEC`が担う。伸び切りの減衰（回転量
のスケール）はそのまま残る。

追記（2026-09-16, 上腕の可動域制限）: 観測経路のsolveにも仮想腕と同じ
`clamp_upper_arm_swing`（`MAX_ARM_DROP_RADIANS`=85度）を掛ける。観測手首が体を
またぐと解析solveは最短弧で上腕を肩の可動域の外へ回してしまうが、この段は
鎖全体を肩まわりに剛体回転するので肘の曲げとリーチは保たれる。観測経路が
`arm_pose::solve_stage`を通らず生の`solve_two_bone_arm`を呼んでいたため、
この段だけが欠けていた。

腕の採用判定は`ArmAdoptionGate`のヒステリシスで行う: 採用は
`ARM_ENTER_VISIBILITY`（0.7）が`ARM_GOOD_FRAMES`（4）連続、喪失は
`ARM_EXIT_VISIBILITY`（0.4）が`ARM_BAD_FRAMES`（6）連続、間の値は現状維持、
欠損スコアは悪い側に数える。判定に使うスコアは肩・手首のvisibilityと
Hand Landmarkerの検出スコアの最小値とする。単眼Poseは見えない腕を
見えている腕と対称に捏造する（実機ログで左手首が右手首とほぼ鏡像で動き、
visは0.5前後、Hand Landmarkerはその側の手を検出しない）ため、自分の手が
検出されないPose腕は追跡しない。これで反対腕が勝手に動かず、手を見失えば
5 sで仮想腕へ戻る。
これで片手を下ろしたら復帰が単調に進み、もう一方の腕の操作に巻き込まれない。

MediaPipeの生観測はdebugビルドで`mediapipe_pose_debug.csv`へ1カメラフレーム
1行（片側あたり肩・肘・手首・hand score）で残す。`propagation_debug.log`の
アバター側と突き合わせ、観測のvisibility低下と追跡の反応を切り分ける。

欠損を原点や既定長で捏造せず、無期限維持・別推論器への自動切替は入れない。

## 実機評価

`vtuber-tracking::arm_evaluation::evaluate_arm_sequence`が記録した腕系列から
到達誤差、骨長誤差、静止区間の変動、肘平面の最大変化、capture→display遅延を集計する。
計測値の取得だけをBevy/時刻側に置き、集計は純粋関数にする。

Windows x86_64では`cargo fmt --all -- --check`、`cargo test --workspace`、
`cargo clippy --workspace --all-targets`、nativeのPose workerテストが通る。
macOSでのビルド・FFI fixture・着座/机/腕交差/伸び切り/復帰の映像確認は未実施。
hold/return/acquire、平滑化時定数、ツイスト速度上限と`max_wrist_step`は実機映像の
フィードバックで更新した値であり、品質保証値ではない。自然な見え方を優先し、
モーションキャプチャ的な即時追従は目標にしない。
