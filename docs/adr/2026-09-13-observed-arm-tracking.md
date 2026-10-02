# 観測した腕の追跡を既存IKへ接続する

状態: 方針採用。Pose FFI（#46）、純粋なtracking状態更新（#48）、Pose workerと
カメラ配信（#47）、既存compositorへの統合（#49）、評価関数（#50）を実装済み。
2026-10-01の利用者による実機確認で、人体骨格の自由度とFKへの統一後の改善と
安定感を確認した。macOS実行は未検証。

現在の関節制御は末尾の2026-10-01「人体骨格の自由度とFKへの統一」を適用する。
全身への適用と2026-10-02の共通化は
[人体骨格の共通解決ADR](2026-10-02-shared-skeleton-resolution.md)に従う。
平滑化の共有計算・保持状態は[共通平滑化ADR](2026-10-02-shared-motion-smoothing.md)に従う。
以前の保持幅、肩から肘への追加配分、前腕/手首への回旋折半は撤去済み。
以下の日時付き追記は経緯であり、撤去された方式を現在の仕様として使わない。

## 採用済みの安定性と維持する原則（2026-10-01）

21:42:48更新のデバッグビルドを案内した後、利用者から
「よくなった。この安定感を守りたい」と実機での改善報告を受けた。
この時点の骨格・関節制御を、今後の変更でも維持する基準とする。
これは今回の使用条件での利用者評価であり、全モデル・全姿勢・全OSの確認ではない。

- モデルのrest骨長・初期回転・親子階層を保ち、上腕の3自由度、肘の固定屈伸軸、
  前腕の回内/回外を分離する。追跡腕と仮想腕は同じ固定軸two-bone IKを使う。
- 親の回転はFKで子へ一度だけ継承する。肩から上腕0.3・肘0.15への追加配分、
  前腕回旋のhandへの折半、観測できない鎖骨姿勢の独自追従を復活させない。
- 減衰は各関節の観測座標へ適用し、固定長の骨格を再構成する。後段で生の手首へ
  IKを解き直したり、補正回転を重ねたりして減衰した姿勢を上書きしない。
  独自の3度/5度保持幅や、根拠のない筋肉・関節連動の係数も使わない。
  欠測復帰にも同じ関節構成を使う。解決済みの初期姿勢と観測姿勢の肩・肘・前腕・
  指の関節座標を戻し、手首位置とpoleの空間補間をIKへ再投入しない。
- TrackedPoseでは観測stageだけがtargetsを所有する。同じカメラseqの描画tickでも
  関節フィルタの状態を継続し、仮想stageによるclearで初期化しない。
- 回内/回外はneutralを原点とする有限の関節座標として、目標と保持状態の両方を
  同じ範囲へ置く。自由回転の累積角を保持せず、正端・負端・neutralへ戻れることを保つ。

変更時は該当する既存テストで、本番の仮想→観測stage順序、固定骨長と肘軸、
非identityのrest回転、掌回旋からの復帰を確認する。実機では静止挙手の肩・肘の
落ち着きと、手を回して戻したときの自然な復帰を短く確認する。単独関数の結果を
最終合成後の安定性確認の代わりにしない。これを新たな長時間待機や全workspace・
全モデルの一律品質ゲートにはしない。

根拠となる公開モデル・公式仕様と選択した可動域は、末尾の
「人体骨格の自由度とFKへの統一」に記載する。変更する場合も同じ一次資料に照合し、
今回の実機評価と区別して変更理由・確認結果を残す。

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
求め、`vtuber-avatar::tracked_arm::align_palm_twist`が回内/回外を前腕へ適用する。
手首へ軸回旋は足さない。指の消費は後の追記で追加済み。

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
- `vtuber-avatar::tracked_arm`: モデル長・rest-spaceへの変換、既存IKと関節角の減衰。
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
角度を求めて追う。回内/回外は`lower_arm`の前腕長軸へ全量を適用し、
handは親子階層でその回転を継承する。handのlocalには軸回旋を追加しない。
折半してskinningのねじれを隠す旧処理は、手首に余分な自由度を作るため撤去した。
掌の欠測時は既存のpalmチャンネルの重みでneutralへ戻る。

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
欠損スコアは悪い側に数える。判定に使うvisibilityは肩・手首のPose値とする。
単眼Poseは見えない腕を見えている腕と対称に捏造する（実機ログで左手首が
右手首とほぼ鏡像で動き、visは0.5前後、Hand Landmarkerはその側の手を
検出しない）ため、自分の手が検出されないPose腕は追跡しない。これで反対腕が
勝手に動かず、手を見失えば5 sで仮想腕へ戻る。
これで片手を下ろしたら復帰が単調に進み、もう一方の腕の操作に巻き込まれない。

MediaPipeの生観測はdebugビルドで`mediapipe_pose_debug.csv`へ1カメラフレーム
1行（片側あたり肩・肘・手首・handedness score）で残す。`propagation_debug.log`の
アバター側と突き合わせ、観測のvisibility低下と追跡の反応を切り分ける。

追加（2026-09-27, handednessを品質に使わない）: Hand Landmarkerの
`HandWorldLandmarks.score`は`handedness_score`へ改名し、左手/右手ラベルを
推測した確信度だと明記した。ラベルの確信度は手の形から導かれるので、判定が
曖昧な姿勢では下がるが座標の良さは変わらない。従来はこの値を腕の採用判定と
掌チャンネルの信頼度として使っていたため、左右の判定が曖昧なだけで腕全体が
落ち、掌も得られなくなっていた。採用判定はPoseの肩・手首のvisibilityと
「その手首に手が検出されているか」だけを使う。`ArmTrackingProfile`の
`palm_confidence`は削除し、掌は検出の有無だけで観測の可否を決める（平面が
定まらない場合は法線が`None`になる）。左右の対応付けもhandednessの降順ではなく
報告順のまま最近傍の空き側を取り、1件の検出が1つの手首に割り当てられる。
`ArmAdoptionGate::update`はPoseのvisibilityと手検出の有無の2つを受け取る。

追加（2026-09-27, 再取得の重複を1つにする）: 再取得までの遅れが2つ重なっていた。
1つ目は採用判定の連続フレーム待ち（`ARM_GOOD_FRAMES`=4）で、PoseとHand Landmarkerが
同じworkerで順次処理されるため1フレームは数十msになり、推論fpsが落ちると待ち時間が
その分伸びる。2つ目は再取得時にslow応答を選んでいた平滑化で、これは#179の固定応答
変更で解消した。残る滞后は共有の`LossBlend`のacquireランプ（1 s）だけになり、判定の
揺れを抑えることと復帰を遅らせることは別の役割なので分けた。冷起動は校正サンプルの
都合で4フレーム、追跡したことのある腕が戻る場合は`ARM_REACQUIRE_GOOD_FRAMES`=2
フレームにした（1フレームだけの誤検出は依然として棄却される）。重みと
hold/return/acquireの時間は変更していないので、戻った腕が飛ばないことは変わらない。
新しい喪失モード・別推論器・代替経路は追加していない。`ArmAdoptionGate`は
「一度でも採用されたか」を`returned`で持ち、実際に採用されていた腕が失われたときに
だけ立てるので、採用されたことのない腕は引き続き冷起動の待ちを使う。
手を画面外へ出して戻す動作と片側だけの欠測で反対側が止まらないことは
**実機未確認**。実測fpsの記録も残っていないため、待ち時間をms換算した値は記載して
いない。

追加（2026-09-27, #190 レビュー R1–R3 の修正）:
`HandFingerPose` は Hand Landmarker の21点から、各指の屈曲と開きを別々に保持する。
掌の基準は wrist→index MCP と wrist→little MCP の単位方向の二等分線をforward、
そのindex cross littleをnormal、forward cross normalをacrossとする。
手ローカルXYZは(across, forward, normal)。4指のMCPは掌平面からの符号付き仰角、
spreadはforwardからacrossへの面内角であり、wrist→MCPとの3D角度をcurlにしない。
PIP/DIPも各指の節から個別に測る。親指の比較区間はcore・tracking・avatarすべて
CMC→MCP、VRMではproximal.position - metacarpal.positionとする。

指の屈曲は各指の節と掌normalが作る軸まわりの符号付き角度で、正はnormal側。
normalは軸性ベクトルなので、左右ミラーでは屈曲と手ローカルZを反転し、spreadと
ローカルXYを保存する。各関節の角度の絶対値には既存の上限を使う。
`resolve_finger_joint`と既存bindingを再利用し、観測経路は節とrest掌normalから
曲げ軸を求め、著作者のrest回転でローカルdeltaへ戻す。仮想腕の緩いcurlは従来の
著作者軸を使う。4指の付け根はrestの仰角・開きとの差をproximalへ適用する。
親指もCMC→MCPのrestと観測の符号付き仰角差を`resolve_finger_joint`へ渡し、
面内の開きと`opening.delta * elevation.delta`の順に合成してmetacarpalへ適用する
（#190 再レビューR4）。4指のspread用`opening_delta`は面内処理を維持する。
隣接するrest節がある関節は
その曲がりを差し引く。VRMにtip骨はないため末節の軸は直前の節から取る。

指形状は平滑化より前に手ローカル化する。`observed_finger_deltas`はIK解や
`tracking_to_rest`を参照しないため、肩親だけを除いた方向を手ローカルと扱う経路は
ない。掌twistの前腕分・hand_delta分はどちらも指の親側の姿勢にだけ反映される。
最終掌姿勢を確認するテストではhand_deltaも含める。Transform writerは引き続き
`apply_default_arm_pose`のみで、指回転は肘・手首のIK位置を変更しない。

指の平滑化は既存のrender clock・固定時定数0.04 s、欠測時の復帰は既存のLossBlendを
使う。掌基準または指節が退化している観測は指チャンネルを生成しない。掌が読めても
指節が読めない場合は掌だけを採用できる。新しい推論器・汎用リターゲット基盤・
監視基盤は追加していない。

修正時のローカル確認:
- `cargo check -p vtuber-core -p vtuber-tracking -p vtuber-avatar -j 1`
- `cargo test -p vtuber-core --lib arm_tracking::tests -j 1`（4件）
- `cargo test -p vtuber-tracking --lib arm_tracking::tests -j 1`（55件）
- `cargo test -p vtuber-avatar --lib tracked_arm::tests -j 1`（24件）
- `cargo test -p vtuber-avatar --lib arm_pipeline::tests::tracked_side_applies_the_observed_fingers_through_the_finger_weight -j 1`（1件）
- 対象3クレートの `cargo clippy --lib --tests -j 1`、fmt、diff check。
  Clippyは変更外のavatarコードに10件のwarningがある。

小さな合成ランドマークで、指定の非放射状伸展指、開きだけ/屈曲だけ、非共線の親指rest
（metacarpalを含む）、剛体回転・掌twist、左右ミラー、非identityのrest軸での掌側への
曲がり、指weightを変えても腕IKの回転が変わらないことを確認した。
**実機は未確認**: 左右の開く・握る・Vサイン・親指を立てる動作、欠測復帰、
VRM 0.x / 1.0の表示、macOS。使用VRM形式・モデル名は実機未実施のため記録なし。
合成ランドマークの結果をVRMの実機受入結果とは扱わず、#183はOPENを維持する。

追加（2026-09-30, 上腕の1フレーム単位計測プローブを追加）: `propagation_debug.log`は
1秒ごとに1行なので、単一フレームの不連続を解決できなかった。上腕が比較的不連続に
到達しうるかを1行で決定的読めるように、`debug_arm_frame_probe`を追加した。
`arm_frame_debug.log`に**毎レンダーティック**1行を書く（既存のプローブと
同じ`Update`、作業ディレクトリ相対、GUI subsystemなのでstderrには出ない）。

1行に両側を出し、1サイトごとに次を書く。
- `sh` `up` `lo` `hand`: 合成後のボーン回転（`rest.rotation.inverse() * transform.rotation`の
  角度）。見えているのはこれである。
- `up_dot`: 解いた上腕方向とrestの骨方向の内積。-1に近づくほど、solverが取る
  最短弧に軸が定まらなくなる姿勢である。
- `desc`: 符号付き冠状面descent（度）。半回転で巻き戻る。
- `w`: `wrist/pole/palm`の3重み。
- `obs`: ミラー適用**後**の観測wrist・elbow pole・palm normal。

2点、既存プローブの誤りを意図的に直している。
1つめ、観測はミラー適用後に書く。既存プローブはunmirroredな値を出しており、
pipelineが実際にどちらの観測側をどちらのボーンに適用しているかを逆に読んでしまう。
2つめ、`written`はボーンのrest相対**ローカル**回転なので、model空間の方向へ適用する前に
restのglobal回転で共役する必要がある。共役しないと`up_dot`/`desc`が別の軸まわりの
回転を測ることになり、実機での切分けには無意味な値になる。
`arm_probe_diagnostics_read_the_composed_bone_correctly`が非identityのrest global
回転でこれを固定する。

`coronal_descent_radians(restの骨方向, 解いた骨方向)`を`arm_pipeline`に切り出して
`upper_arm_descent_radians`と共用するようにした。プローブが制限と別の定義を
持つと切り分け自体が壊れるため。

回帰: 1件（上記）。ローカル確認: `cargo test -p vtuber-avatar --lib -j 1`（375件）、
`--tests`、`cargo clippy -p vtuber-avatar --lib --tests -j 1`（変更外の10件のみ）、fmt。
**実機は未確認**: プローブの実出力は見ていない。上記の共役とミラーはtestと
コードの定義で担保しているが、ファイルが実際に有用な形式で書かれることは
実機で確認していない。#183はOPENを維持する。

欠損を原点や既定長で捏造せず、無期限維持・別推論器への自動切替は入れない。

追加（2026-09-30, 掌ロールが上腕の回転を肩へ逃がされ、肘が跳ねた実機症状）:
`arm_frame_debug.log`（毎レンダーティック）で、左右とも肘（`lo`）が単一フレームで
42〜77deg跳ねるのが記録されていた。**前日の修正が混入した回帰**である。

記録から-: 跳ねた13フレームすべてで `hand`（手の1关节分のロール）が
87.4〜90.0deg に収まっていた。`hand` の上限は前日入れた
`PALM_TWIST_TOTAL_LIMIT_RAD`(=180deg)の半分、つまり**ちょうど90deg**である。
跳ねたフレームは1600個中13個で、±180degの`desc`の反転でも最短弧の対向点でも
ない（`up_dot`は最小0.12で-1に寄っておらず、`desc`は最大147degで半回転を
跨いでいない）。前日の2つの仮説（冠状面descentの巻き戻り、solverの最短弧）は
どちらも外れており、実犯は前日のクランプ自身である。

なぜ跳ねるのか: 前日は合計ロールを180degで頭打ちにして**前腕と手**に50/50で
分けていた。合計ロールは手首の回旋だけでなく、方向のみのsolveが表現できない
上腕の軸回転を含むので180degまで達する。すると前腕のボーンに最大90degの
回旋が乗り、VRMの前腕は1本なのでその回転がそのまま前腕のスキューになる。
腕を曲げた姿勢ではその合成が急峻になり、飽和の瞬間に肘が50〜77deg飛ぶ。

方針: **手首は手首の可動域だけ、超過分は上腕へ**。`PALM_TWIST_TOTAL_LIMIT_RAD`を
廃止し、`PALM_WRIST_LIMIT_RAD`(=90deg、前腕と手のpairが示す手首の回旋範囲)に
変えた。`align_palm_twist`は `wrist_roll`（頭打ち）と `humeral_roll`（残余）に分け、
残余は新しい`roll_humerus`が上腕自身の長軸まわりに回して下位を剛体で追従させる。
上腕を自身の軸まわりに回すと上腕のlocal deltaだけが変わらず、前腕のlocal deltaは
変わらない（同じworld回転が両方に掛かるため`upper⁻¹ ⊗ lower`が不変）。肘の屈曲と
手首のロールはそのまま残り、合計も保たれるので掌の平面は観測に一致したままに
なる。前腕のボーンに乗る回旋が90degを超えないので、肘は跳ねない。

回帰: `a_roll_past_the_wrist_range_goes_to_the_humerus_not_the_forearm`。
ロールを0→240→0degで回して、上腕と前腕が連続であること、前腕のlocal deltaが
`roll_humerus`で一切変わらないこと、超過分が実際に上腕へ回っていることを固定する。
前腕に全部を戻す（=前日の挙動にすると）このテストは落ちる。

**前日の記述の訂正**: 「上腕が半回転する実機症状」として書いたものは観測腕の冠状面
descent制限の撤去で対処したが、上のとおり実機ログの`lUpperArm`は最大85.54degで
対向点にも達しておらず、今回の症状とは別物だった。同じ日に別の不連続が2つ
あったことになる。`clamp_upper_arm_swing`の`rotation_arc`由来の94.69degの跳ねは
実在するので撤去は妥当だが、**今回の症状の防止にはなっていなかった**。

ローカル確認:
- `cargo check -p vtuber-core -p vtuber-tracking -p vtuber-avatar -p vtuber-app -j 1`
- `cargo test -p vtuber-avatar --lib -j 1`（376件）、`--tests`（integrationは全て成功）
- `cargo clippy -p vtuber-avatar --lib --tests -j 1`（変更外の10件のみ）、fmt。
**実機は未確認**: `arm_frame_debug.log`による修正前後の実測比較はしていない。
`lo`が連続になったことは合成ランドマークの回帰でしか担保していない。左右の
手首の捻り、捻った状態での手の開閉、欠測復帰、VRM 0.x / 1.0の表示、macOS。
合成ランドマークの回帰をVRMの実機受入結果とは扱わず、#183はOPENを維持する。

追加（2026-09-30, 観測腕の冠状面descent制限を撤去）: `AvatarSample_C.vrm`で動かした
`propagation_debug.log`の後に、観測腕のStage 3b（`clamp_upper_arm_swing` /
`MAX_ARM_DROP_RADIANS`）を撤去した。理由は、この制限が上腕を半回転で飛ばすためである。
`clamp_upper_arm_swing`は上腕の冠状面descentを**符号付き**角度で測り、
`descent > MAX_ARM_DROP_RADIANS`という片側条件でchain全体を剛体回転させる。
その角度は`atan2`なので半回転で巻き戻り、上腕が体側を横切って回ると
+179.8°→-179.8°で制限の適用の有無が反転する。実測で1°のステップあたり
上腕のdeltaが94.69°、肘の位置が376.5mm跳ねた。

この制限が抑えていたのは**解いた腕**のposeである。`solve_two_bone_arm`は
authored rest poseから最短弧で外れるので、その产物が肩の可動域を超えることはある。
観測腕は測定値であり、測定値を抑えない。置き換えたテストは
`tracked_side_shows_the_measured_arm_rather_than_clamping_it`で、制限を
超える観測ターゲットでも解いたままの腕になることを固定する。制限そのものは
仮想ハンド(source)には残す。

**未解決**: 撤去だけでは連続性は回復しない。同じ「腕が体に対して水平に横切る」
姿勢で、`solve_two_bone_arm`の`rotation_arc(restの骨方向, 解いた骨方向)`が最短弧の
軸を定められず`stable_perpendicular`へ切り替わるため、上腕のdeltaが1°の
ステップあたり151°跳ねた（測定済み）。上腕のmodel deltaが最短弧である限り、
その対向点では必ず不連続になる。これは2ボーンIKのelbow flipと同じ問題で、
bend平面の軸から回転を組み立てるなど別の構成に変える必要がある。実機ログの
`lUpperArm`は最大85.54°で対向点には達していないため、この151°が今回報告された
症状なのか、そうでないのかは断定できない。`propagation_debug.log`は1秒ごとの
サンプルなので単一フレームの事象を解決できず、この切り分けには`palm_trace`
（30tickごと、stdout）が必要である。

ローカル確認:
- `cargo check -p vtuber-core -p vtuber-tracking -p vtuber-avatar -p vtuber-app -j 1`
- `cargo test -p vtuber-avatar --lib -j 1`（374件）、`--tests`（integrationは全て成功）
- `cargo clippy -p vtuber-avatar --lib --tests -j 1`、fmt。Clippyは変更外の
  avatarコードの10件のみ。
**実機は未確認**: 左右の手首の捻り、捻った状態での手の開閉、腕を体側に横切った
ときの挙動、欠測復帰、VRM 0.x / 1.0の表示、macOS。合成ランドマークの回帰を
VRMの実機受入結果とは扱わず、#183はOPENを維持する。

追加（2026-09-30, 手首を捻った途中に上腕が非連続的に回転する実機症状）: `debug`ビルドの
`propagation_debug.log`で、左手の `hand=`（適用された手首ロール）が +8〜+86deg を行き来し、
同じ区間の `lLowerArm` が 71〜167deg 動いていた。`hand`は1秒ごとのサンプルなので
どのフレームで跳ねたかは決まらないが、要求されるロールの大きさは測れば決まる。

原因は1つで、`align_palm_twist`が要求ロールを符号付き角度として測っていたことである。
要求ロールは手首の回旋だけではない。`solve_two_bone_arm`は上腕・前腕のmodel deltaを
「restの骨方向から解いた骨方向への最短弧」で作るため、方向だけでは表せない上腕の
内旋・外旋（上腕自身の軸まわりの回転）を表現できない。だから掌の姿勢の誤差は
「上腕の軸回転＋手首の回旋」になる。実測では、合成した腕姿勢を手首捻り0°で測ると
要求はT-poseで0°、腕を下ろした姿勢で0°、腕を下ろした姿勢で上腕を90°捻ると90°だった。
つまり要求は腕の姿で0°から180°まで動く。符号付き角度は180°で巻き戻り、`atan2`の値が
+179°から-179°へ飛んだ瞬間、前腕と手はそれぞれ180°ずつ階段状に回った。50/50に分ける
ため1关节あたりの飛びは180°で、ちょうど実測ログの`hand`の分享（最大86°）の2倍である。

符号なし角度と外積の軸で連続な回転を作っても直らない。合計の回転は連続になるが、
2关节の分割は連続にならない。180°で軸が反転しその瞬間だけ分割が反転するためで
（`slerp(IDENTITY, roll, 0.5)` は180°をまたぐと最短弧を取り直す）、「どちらの
半回転か」は単一フレームの観測には無い本質的な帰結である。つまり直前の丸めを
覚えておくしかない。

方針: 枝を前フレームから持ち越す。`align_palm_twist`は `previous_roll: Option<f32>`
を受け、前回の合計ロールの近傍に測定値を `rem_euclid(TAU)` で持ち上げ、
`PalmTwist { hand, roll }` を返す。`roll`は持ち上げた値を保持し、boneへ書く分だけを
`PALM_TWIST_TOTAL_LIMIT_RAD`（=PI、つまり1関節あたり手首1回分の可動域）で頭打ちに
する。branchを丸めずに持つのは、腕が上限で止まっている間にもbranchを抱えており、
観測が戻ったときに上限から連続して戻るためである。持ち上げる対象は
`measured` として保持し、符号付き角度に戻す処理は入れない。
枝は `update_tracked_arm_targets` の `Local<PalmTwistBranches>` が左右で持ち、
control frameが無いとき・lifecycle が Ready でないとき・avatar が置き換わったとき
（generationが変わったとき）に clear する。side の resolve が飛ばされた場合はposeを
保持するのと同じく枝も保持する。

回帰: avatarに「手首を捻っても前腕と手が半回転で飛ばない」1件を追加した。0→200→0度の
手首捻りで、1度あたりの前腕・手の回転が5度未満であることを確認する。この回帰は
修正を外すと最悪179.4度で落ちる（実際の飛びの再現）。既存の「palm planeを観測に
合わせる」「50/50に分ける」「weightで単調に変わる」はそのまま通る。

ローカル確認:
- `cargo check -p vtuber-core -p vtuber-tracking -p vtuber-avatar -p vtuber-app -j 1`
- `cargo test -p vtuber-avatar --lib -j 1`（374件）、`--tests`（integrationは全て成功）
- `cargo test -p vtuber-core --lib -j 1`（85件）、`-p vtuber-tracking --lib -j 1`（317件）、
  `-p vtuber-app --lib -j 1`（292件）
- `cargo clippy -p vtuber-avatar --lib --tests -j 1`、fmt。Clippyは前回同様
  変更外のavatarコードの10件のみ。`resolve_tracked_side`が8引数になったので
  この1箇所に理由付きの`#[expect]`を置いた。
**実機は未確認**: 左右の手首の捻り、捻った状態での手の開閉、欠測復帰、VRM 0.x / 1.0の
表示、macOS。合成ランドマークの回帰をVRMの実機受入結果とは扱わず、#183はOPENを
維持する。

追加（2026-09-30, 親指の根本が動きすぎて伸びきる実機症状）: 実機で親指を動かすと
根本の位置そのものが一緒に動き、親指が伸びきった状態になる。原因は1つで、Hand
Landmarkerのlandmark 1（THUMB_CMC）は検出された関節ではなく手のモデルが配置する点
であり、親指全体が動くと一緒に動く。そのためCMC→MCPのrayは親指全体の向き変化の
合計であり、CMC→MCPをrest相対でVRMのthumbMetacarpalへ適用すると
(a) rayの向き変化がそのまま付け根の回転になり、親指を動かすと根本が一緒に動き、
(b) 同じrayに3D全体角を第二の曲げとしてproximalとdistalへ重ねるため、親指
の先が伸びきる。両方とも「付け根の回転に実在しない自由度を与えていた」ことの帰結である。
人間の手で親指が関節正确的是MCPとIPであり、付け根はほとんど動かない。

方針: CMCは観測に一切使わない。`THUMB_CHAIN`をlandmark 1,2,3,4の4点から
2,3,4の3点（実在するMCP/IP/TIP）に改め、親指も4指と同じ2関節のchainとして測る。
MCP→IPの節が付け根の面外仰角と面内開きを与え、IP関節が自身の曲げを与える。
`HandFingerPose.thumb_direction`は使わなくなったので削除し、同じ役割の
`thumb_spread`（f32）に置き換えた。`HandFingerPose.thumb`の意味も変わるので、
[0]は4指の`fingers[i][0]`と同じくMCP→IP節の掌平面からの符号付き仰角、[1]はIPの
曲げになる。
avatar側: `observed_finger_deltas`は親指に`three_joint_deltas`を使わず
`thumb_deltas`を使う。`thumb_metacarpal_delta`は削除し、metacarpalには
rest姿勢のdeltaを書かないので付け根はVRMのrest poseのままになる。VRMの親指は
`metacarpal, proximal, distal`で`intermediate`が無くproximal→distalが付け根の
節になるため、4指と同じ`three_joint_deltas`をそのまま使うと付け根が
`None`になり動かない。proximalはdistalまでの節を軸に付け根の回転を受け持つ。
`THUMB_FLEXION_LIMIT_RAD`を削除したのは、MCPの値が4指の付け根上限と同じ測定に
なったためで、MCPは`FINGER_FLEXION_LIMIT_RAD[0]`を共用する。親指はこの測定が
掌平面から大きく傾いているので、小さい上限では通常の動きを制限してしまう。
IPだけ`THUMB_IP_FLEXION_LIMIT_RAD`を持つ。
指方向の`DirectionSmootherState`はpalm normal用に残り、親指方向用の状態は
使わなくなった。最短弧回転のhelperはpalm normalが使うので残す。

回帰: trackingに「landmark 1を動かしても親指poseが一切変わらない」と
「まっすぐな脚は屈曲0」の2件。avatarには「どんな観測でも付け根は
metacarpalへ書かれない」と、実tracking経路を通して親指全体が斜めに回っても
その motion が2関節だけに届くことを確かめる1件を追加。四指の回帰は
そのまま通る。

ローカル確認:
- `cargo check -p vtuber-core -p vtuber-tracking -p vtuber-avatar -p vtuber-app -j 1`
- `cargo test -p vtuber-core --lib arm_tracking -j 1`（4件）
- `cargo test -p vtuber-tracking --lib -j 1`（315件）
- `cargo test -p vtuber-avatar --lib -j 1`（373件）、`--tests`（integrationは全て成功）
- `cargo clippy -p vtuber-core -p vtuber-tracking -p vtuber-avatar --lib --tests -j 1`、fmt。
  Clippyは前回同様変更外のavatarコードの10件のみで、新たなwarningはない。
**実機は未確認**: 左右の開く・握る・Vサイン・親指を立てる動作、欠測復帰、
VRM 0.x / 1.0の表示、macOS。合成ランドマークの回帰をVRMの実機受入結果とは
扱わず、#183はOPENを維持する。

追加（2026-09-27, 遅延の比較に必要な観測の可視化）: 入力解像度・推論モデル・
描画負荷の比較を実機で行うには、実際にどのformatが取れたかを見る必要がある。
`CameraRequest`は要求であって約束ではなく、`select_format`はハードコードされた
優先順位（1280x720 → 640x480、30fps）で実デバイスの候補から選ぶので、
要求値と実formatは違うことがある。`CaptureMetrics.format`には記録されて
いたがUIにもCSVにも出ていなかった（ADR-003が「実際に選択されたformatをUIと
performance reportへ記録する」と決めている事項）。そこで既存
`DiagnosticsSnapshot`に`camera_format`を1項追加し、`sync_capture_diagnostics`が
formatがある場合は同期のたびに`format.to_string()`を実行する。文字列の比較後、
違いがある場合だけsnapshotへ代入し、既存のopt-in CSVにも列を追加した。
stream開始時だけ文字列を生成する実装ではない。新しい監視基盤は追加していない。

更新順序の監査結果（ADR-004に反映）: 顔の経路はPoseの結果を待っていない。
captureスレッドが同じ1フレームをface用とPose用の別々の`LatestSlot`へfan-outし、
出力スレッドも別々で、`Update`内に両者を結ぶ順序指定はない。共有しているのは
腕の`ArmSourceSelection`だけで、頭・胴体・表情の経路はそれに依存しない。
一方、腕のターゲット解決は`update_dynamic_arm_targets`が胴体骨の
`GlobalTransform`を読んでおり、このシステムが`update_body_tracking_pose_input`
より前にスケジュールされていたため回転writerより前、つまり前フレームの胴体姿勢を
読んでいた。lean writer（`apply_direct_body_position`）との前後関係はexecutorの
アクセス競合解決に任されており、決まっていなかった。両armターゲット段を
両writerの後ろへ移し、`tests/schedule.rs`にその5本の辺を固定した。
body writerが先、arm-target段が後である。循環はテスト実行中のschedule初期化時に
検出されるもので、Rustのコンパイル時検出ではない。

実機での比較（Release、同じカメラ・VRM・画角・照明・動作で条件を1つずつ変える、
640x360と1280x720、Pose FullとHeavy、通常表示/プレビュー/配信有効）は**未実施**。
`pose_landmarker_heavy.task`は`assets/models`に同梱されておらず、選択する経路も
無いのでHeavyの比較にはアセットと一時的な選択手順が要る。ファイルを置くだけでは
比較できない。640x360が選べない場合は非対応として記録し、640x480等の結果を
640x360と表記しない。`Delegate::Cpu`が3つのlandmarkerで
ハードコードされていてGPU delegateは到達不能だが、委譲の切替は要求されておらず、
負荷が測定されるまで恒久的な分岐を残さない方針なので変更していない。
`inference_input_skipped_frames`はfaceワーカーのみであり、Poseワーカーの
drop率・stage timing・結果は`DiagnosticsSnapshot`に一切出ていない。これは今回の
比較で欲しければ扱うべき点として記録するにとどめ、新しい計測経路は追加しない。
`capture_to_apply`は顔側の骨姿勢の適用までを測る値で、腕や指の遅延や画面提示までの
遅延ではない点も再確認した（ADR-019/022の既存方針と一致）。

#191は修正済み#190（`83a318e3c06bc663e29f2d3ebedc087d905843f4`）へ追従し、
更新順序とCSVの固有変更を維持した。追従後のローカル確認は
`cargo check -p vtuber-app -p vtuber-avatar -j 1`、avatarの`--test schedule`（3件）、
appの`metrics_export::tests::csv_records_the_negotiated_capture_format`（1件）、
avatarの`tracked_arm::tests`（24件）が成功。fmtとdiff checkも成功。
これは比較用コードの確認であり、#184の実測比較と採用判断は未実施のまま残す。

追加（2026-09-27, 掌の回転連続性）: 手の位置0.05 sに対して掌は0.30 sで6倍遅く、
手先位置は追いついていても掌の向きだけが後からついていく状態になっていた。時定数を
0.10 s（位置の2倍）に縮めた。加えて掌法線は3ベクトルを平滑化して最後に正規化する
方式をやめ、`DirectionSmootherState`で回転として追従する方式へ置き換えた。誤差は
現在方向から観測方向への最短弧の回転ベクトルなので、手のひらを返して法線が正反対
になっても、原点付近を経由せず実際の動いた側を回って到達する。正反対では軸が不定
になるため、方向と最も平行でないワールド軸との外積で決めた安定な垂直軸を使う。
回転の適用はRodriguesの式で、補正が0のフレームでも退化しない。頭の回転フィルタと
同じ接平面表現と共通のステップだけを共有しており、頭用の`max_step_rad`（1.25 rad、
2観測間の首の物理上限）は掌には意味がないため流用していない。ねじれの折半配分と
IKのrest空間変換は`align_palm_twist`のままで、向き補正で肘・手首の位置は動かない。
実機での映像確認は未実施。

追加（2026-09-27, 速度と外れ値を混ぜない固定応答）: `ResidualResponse`は連続する
観測の変位を観測間隔で割り、その速度が大きいほど時定数をfastからslowへ伸ばして
いた。一定速度の速い運動でも残差が大きくなるため「継続する速い動作では残差が
小さい」という以前のコメントと計算が一致せず、速く動かす手や首振りだけが
遅れていた。頭は同じターゲットを保持する描画tickで速度を0にして応答を戻す一方、
腕は次の観測まで速度を保持しており、部位ごとに挙動が違ってもいた。
`filter::damped::ResidualResponse`と`observation_rate`を削除し、頭回転・頭平行
移動・腕の手首/肘・掌平面を部位ごとの固定時定数の二次フィルタにした。時定数は
既存のfast側（頭回転0.025 s、頭平行移動0.10 s、腕の位置0.05 s、掌0.30 s）。
不連続は速度ではなく既存の判定で弾く: 頭は`max_step_rad`（1.25 rad）、腕は
`max_wrist_step`（校正腕長の0.75倍）、欠測と復帰は`LossBlend`のhold/return/acquire
が担当する。掌の回転連続性の改善は別の追記（2026-09-27, 掌の回転連続性）で行う。

追加（2026-10-01, 掌ロール分解を肘平面基準へ変更し、上腕への後付けロールを撤去）:
`arm_frame_debug.log`の実機症状（手首を回すと上腕ごと回る）の原因は、平滑化・
MediaPipeではなく`align_palm_twist`の分解だった。記録では追跡した951フレーム中
276フレームで`hand=+45.0`（手首ロールの上限90度×折半分）に張り付き、手首ターゲット
`w=`と`palm=`がほぼ静止した区間で`up`が33.6度振れていた。frame 1763→1765
（同一seq=354）では観測palmの変化が1〜3度の間に`up`が80.3→43.3、`hand`が45.0→4.9
と1フレームで飛ぶ。`mediapipe_pose_debug.csv`の生landmarkも同じ区間で静止しており、
入力ではなくアバター側の再分解である。

原因は1つで、`solve_two_bone_arm`は上腕・前腕をそれぞれ独立な最短弧
（`rotation_arc`）で回すため、解の軸回りロールが任意になる。`align_palm_twist`は
その任意ロールを基準に掌法線との差を測っていたので、「要求ロール」は手首の回内
ではなく「ソルバの任意オフセット＋上腕twist＋手首回内」の混合物になり、常に90度を
超えて飽和した。さらに`roll_humerus`は`lower_arm_delta`を据え置いたまま上腕を
肩-肘軸回りに前回転するので、肘は軸上で動かないが手首は軸周りを公転し、IKの手首
位置制約そのものを壊していた。腕全体が肩から回る見た目はこれである。
`lift_roll`と`PalmTwistBranches`は90度の配分境界ではなく180度の分岐しか連続化
できず、配分の不連続は残っていた。

方針: `orient_tracked_arm`を追加し、掌を当てる前に解の姿勢を作り直す。上腕は
「restの曲げ平面（曲がったrest姿勢では上腕×前腕、直線T-poseでは掌法線の射影）を
観測の曲げ平面（解いた上腕×前腕）へ写す回転」で向き付ける。前腕は上腕に対する
純粋なヒンジswingとして組み直す。これで解の軸回りロールが観測の肘位置で決まり、
掌法線との残差は手首自身の回内になる。`align_palm_twist`は残差を
`PALM_WRIST_LIMIT_RAD`（±90度）で頭打ちにし、前腕と手首へ折半するだけにした。
範囲を超えた分は捨て、上腕には決して載せない。位置は変えない（上腕は解いた肘を、
前腕は解いた手首を向いたまま）。`roll_humerus`、`lift_roll`、`PalmTwistBranches`、
`previous_roll`は削除した。

回帰: `a_wrist_twist_moves_the_wrist_not_the_upper_arm`（掌を±120度回して上腕が
1e-3度も動かないこと、手首が±90度で飽和すること、前腕と手が連続であること）、
`orient_tracked_arm_keeps_the_solved_positions_and_take_the_bend_plane`。
既存の掌テストは新しい基準（`orient_tracked_arm`後の参照掌法線まわりの回転）で
観測を作る形に更新した。

ローカル確認:
- `cargo test -p vtuber-avatar --lib -j 1`（376件）
- `cargo test -p vtuber-avatar --tests -j 1`（全suite成功）
- `cargo check -p vtuber-app -j 1`
- `cargo clippy -p vtuber-avatar --lib --tests -j 1`（変更外の10件のみ）
**実機は未確認**: 手首の捻り、捻った状態での手の開閉、腕を体の前へ回したときの
肘平面、欠測復帰、VRM 0.x / 1.0の表示、macOS。修正前後の`arm_frame_debug.log`
比較はしていない。合成ランドマークの回帰をVRMの実機受入結果とは扱わず、#183は
OPENを維持する。

追加（2026-10-01, 手首ロールのジッター: ロールチャンネルをレンダークロックで
平滑化）: 上の分解修正後、肘のぶん回りは消えたが、今度は手首の回転がジッターを
起こす実機症状が出た。新しい`arm_frame_debug.log`（18:41）では、同一カメラseqの
まま観測palmが1〜3度しか動いていないのに`hand`が1tickで10〜18度動き、2tickで
29度（frame 2648→2650）進んでいた。`mediapipe_pose_debug.csv`の生掌法線は1
カメラフレームで中央値4.1度、p90で14.1度、p99で28.6度、最大48.4度動いており、
手ランドマークの掌法線はもともとノイズが大きい。

原因は、掌法線の*向き*はrender clockで平滑化されていたのに、そこから導く
*ロール角*は一度も平滑化されていなかったこと。ロールは「観測掌法線」と「基準
掌法線」の前腕軸まわりの射影同士の角度なので、掌法線が前腕軸に近い姿勢では
1/sin(角度)で増幅され、基準（解いた前腕のロール）もwrist/poleの微小な動きで
回るため、平滑化済みの掌法線から数十度/tickのロールが出ていた。旧実装では
±90度で頭打ちにして超過分を上腕へ逃がしていたため、手首が張り付いて目立って
いなかっただけである。

方針: ロールチャンネルにも`tracked_arm::PalmRollFilter`を追加し、頭・身体・
掌平面と同じ`filter::damped`のcritically damped 2次フィルタ（時定数0.10 s、
掌平面と同じ）をrender clockで回す。測定角はフィルタの現在値に最も近いturnへ
`rem_euclid`で持ち上げてから入れる（これは分岐の持ち越しではなく、-π/πの
ラップが1tickで1回転として入らないためのunwrapである）。`update_tracked_arm_targets`
が左右のフィルタを`Local`で持ち、control frameが無い・lifecycleがReadyでない・
avatarが置き換わったときは`reset`する。フィルタは測定値だけでなく上腕の
再構成にも依存しないため、`orient_tracked_arm`の出力をそのまま平滑化する。
`align_palm_twist`は`&mut PalmRollFilter`と`dt_sec`を受け取り、平滑化した
ロールを±90度で頭打ちにして前腕と手へ折半する。上腕には引き続き一切載せない。
新しい分岐・代替経路・計測基盤は追加していない。`filter::damped`はクレート外
から使うため`pub`にした。

回帰: `the_roll_filter_suppresses_tick_noise_but_still_follows_a_turn`（20度の
交互ノイズで1tickの出力が5度未満、持続する45度の回転には追従する）。既存の
掌テストはフィルタを1tickで収束させる`dt_sec=1.0`で従来どおりの応答を確認する。

ローカル確認:
- `cargo test -p vtuber-avatar --lib -j 1`（377件）
- `cargo test -p vtuber-avatar --tests -j 1`（全suite成功）
- `cargo check -p vtuber-app -j 1`
- `cargo clippy -p vtuber-tracking -p vtuber-avatar --lib --tests -j 1`
  （変更外のavatar 10件のみ）
**実機は未確認**: 手首の捻り速度・追従ラグ・残留ジッター、欠測復帰、VRM 0.x / 1.0、
macOS。時定数0.10 sは掌平面と同じ値であり、品質保証値ではない。

追加（2026-10-01, 19:11ログの親指ルートずれ・肘ジッター回帰）:
`propagation_debug.log`のモデルは`edf052…a205`（千駄ヶ谷 渋、VRM 0.x由来）。
その管理コピーの`VRMC_vrm.humanoid.humanBones`には、左親指が
`leftThumbProximal=77 / leftThumbIntermediate=78 / leftThumbDistal=79`
というVRM 0.xの名前のまま残っていた。上流はVRM 1.0の
`metacarpal / proximal / distal`でバインドするため、CMCをproximalとして回し、
MCPをバインドできていなかった。9月30日の合成テストは最初から正しい
VRM 1.0のバインドを作っていたので、この経路を検出できなかった。

VRM 0.x変換で`ThumbProximal→ThumbMetacarpal`、
`ThumbIntermediate→ThumbProximal`へ左右とも名前を変換する。
`prepare_managed_vrm_bytes`も、既存の管理コピーに残った`ThumbIntermediate`
を同じ対応へ直す。既存の`ensure_managed_model_ready`が読込前に適用するので、
再インポートは不要。正常なVRM 1.0の名前とGLBのbinary chunkは維持する。
また、metacarpalに`None`を返すとコンポジタは以前のdefault curlを残すため、
観測指のmetacarpalにはidentity deltaを渡して付け根を元の姿勢へ戻す。

肘は同ログの追跡中、wrist/poleの小さな変化に合わせて上腕・前腕が往復していた。
たとえばframe 5463→5464（seq 1264→1265）は左前腕2.5度・右前腕3.6度/tick。
位置平滑化を手首・肘で共有していた0.05 sから、肘だけ0.10 sへ変更した。
肘のpoleはIKの曲げ平面を決めるので、その位置ノイズが腕の回転になる。
手首は0.05 sを維持し、reachの追従を遅くしない。新しいフィルタ経路は追加しない。

回帰確認は、VRM 0.xの左右親指の変換、既存管理コピーの修復と二重適用、
default curl後に実際のコンポジタで親指MCP位置がrestへ戻ること、
30 Hzの肘ノイズの抑制と持続する平面変化への追従。
ローカル確認: avatar lib 380件、trackingのarm_tracking 57件、appのimport 33件が成功。
`cargo build -p vtuber-desktop -j 1`で`target/debug/RusTuberV.exe`を更新。
fmt、diff check、変更クレートのClippyも成功（既存のavatar警告10件のみ）。
**修正後の実機映像は未確認**。ログとモデルは修正前の実測であり、
合成テストの成功を実機での解消確認とは扱わない。

追加（2026-10-01, 19:31の静止挙手ログ、関節全体の減衰）:
19:11の位置フィルタ調整だけでは解消しなかった。最新ログのseq 350–450では、
演者が静止していても前腕のlocal角は左53.3–64.7度、右75.4–90.2度で往復した。
Poseの推定骨長・手首距離も揺れており、位置を減衰してからIKで角度へ変換する
だけでは、逆算の感度が再び角度ノイズを作る。また観測経路が使っていた
`resolved_from_solution`は鎖骨の弱い追従に加えて上腕0.3・前腕0.15の追加回転を
載せ、親子階層の鎖骨回転と重複していた。これは解いた骨格を後段で書き換える。

現在の観測経路は次の一方向の順序とする。

1. 観測位置の固定応答・欠測重み → neutralとの目標合成 → 既存two-bone IK。
2. 曲げ平面から上腕の球関節と肘のヒンジを分離する。
3. `TrackedArmFilter`で上腕orientationと肘flexionをrender clock上で減衰し、
   モデルの固定長の上腕・前腕を一緒に再構成する。
4. 減衰した上腕挙上から鎖骨を弱く追従させる。鎖骨回転を上腕local deltaから
   除去し、階層が一度だけ適用する。観測経路へ仮想腕のdownstream sharesを足さない。
5. 掌回旋は前腕長軸と手首へ折半する。肘・手首位置と上腕intentを変えない。
   指は手ローカルで適用し、親指CMCのrest保持も維持する。

筋緊張による姿勢保持の最小表現として、上腕3度・肘5度以内の観測変化は
retained intentへ足さず、帯域を越えた分だけcritically damped応答（0.15 s）で
追う。これは解剖学的な可動域制限ではなく、カメラの不確かさに対するアニメーション
許容幅である。小さな意図的動きもこの幅内では保留されるため、即時の位置一致より
静止保持を優先する。欠測重みが0へ向かうと保持幅も0にし、neutralへの復帰に
オフセットを残さない。フィルタの上腕状態は胴体補償前のcanonical基準へ保存し、
胴体回転の除去を遅延させない。モード終了・観測frame消失・モデル交換で状態を落とす。
位置フィルタは手首・肘とも0.05 sへ揃え、19:11の肘だけ0.10 sの変更を置き換えた。
角度を減衰した後に生の手首へIKを解き直す処理は置かない。

最新CSV（567観測）と実際の管理VRMのrest骨格を一時プローブで60 Hz再投入した。
ミラーON、胴体回転固定、同じtracking出力を旧・新の観測合成経路へ渡した比較である。
seq 350–450（205 render ticks）の合成後の骨方向変化は次のとおり。

| 側・骨 | RMS 度/tick 旧→新 | 最大 度/tick 旧→新 |
| --- | --- | --- |
| 左・上腕 | 0.378 → 0.015 | 0.767 → 0.044 |
| 左・前腕 | 0.408 → 0.025 | 0.994 → 0.064 |
| 右・上腕 | 0.360 → 0.021 | 0.969 → 0.063 |
| 右・前腕 | 0.396 → 0.041 | 1.022 → 0.099 |

一時プローブは削除し、カメラログを常設fixtureや新しい計測基盤にはしていない。
常設の回帰確認は、静止挙手の上腕・肘ノイズが鎖骨/掌/指の合成後にも抑制されること、
鎖骨合成が解いた両骨の向きを変えないこと、意図的動作と胴体補償への追従、欠測後の
neutral復帰である。骨長不変と、手首・指の動作が固い関節を動かさないことも同時に確認する。
ローカル確認: avatar lib 384件、trackingのarm_tracking 56件、arm_pose 9件と
arm_tracked_mirror 3件が成功。変更クレートのClippy（既存avatar警告10件のみ）、
fmt、diff checkも成功。`cargo build -p vtuber-desktop -j 1`で20:06にデバッグ実行ファイルを更新。
**新ビルドの実機映像は未確認**。再投入では動作ログの変動抑制を確認したが、
実際の見え方・追従遅延まで確認したとは扱わない。

追加（2026-10-01, 20:52ログの手の甲反転と、標準的な関節構成への統一）:
今回のモデルは`899c7c…6f9c`（AvatarSample_C）。親指の確認に使った
`edf052…a205`とは別の骨格で、肩・上腕のrest回転もidentityではない。
seq 800–1020のログでは、観測掌法線は変化しているのに、手のlocal回旋は
ほぼ0–1度だった。原因は本番の順序`update_dynamic_arm_targets` →
`update_tracked_arm_targets`で、前者がTrackedPose時にも共有targetsを毎tick消し、
後者がgenerationなしをモデル交換と判定して関節・回旋のフィルタを毎tick
初期化していたこと。これまでの単独tracked-systemテストと純粋関数の再投入は
この不具合を含んでいなかった。20:06ビルドで減衰状態が継続するとした判断は誤り。

TrackedPoseでは観測stageだけがtargetsを所有し、仮想stageは書かない。
frame消失・generation不一致・Ready終了時のclearは観測stageで行う。
仮想stageの同じseqを理由としたcache skipも外し、同じカメラseqでsourceを
切り替えたときに以前のsourceの姿勢をそのまま使わない。

関節構成は[ozz-animationのtwo-bone IK](https://guillaumeblanc.github.io/ozz-animation/documentation/ik/)
と同じ固定長2骨・3関節、固定middle-axisとpoleを基準にする。上腕の球関節、
肘の屈伸ヒンジ、前腕の回内/回外を分離し、親子階層を一度だけ適用する。
これは[OpenSimの上肢モデル](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53087772/Upper%2BExtremity%2BModel)
が区別する肩・肘・前腕・手首の構成に従う。回旋の折半は下の骨格監査で撤去した。
未知の筋力や筋肉係数を推測する物理モデルは実装しない。

前回の独自保持幅（上腕3度・肘5度）は撤去した。上腕Quaternionの差を
rotation vectorにして臨界減衰し、目標Quaternionへ再構成する、公開された
[Quaternion springの式](https://theorangeduck.com/page/spring-roll-call#quaternion-spring)
に揃える。肘は同じ臨界減衰のscalarで追う。0.15 sはアプリの応答時定数であり、
解剖学的な角度幅や筋力の推定値ではない。小さな意図的変化もneutralへの復帰も
オフセットなしで収束する。

回旋も出力だけでなく目標・フィルタの保持状態を同じ±90度の関節座標内に置く。
角度の原点はneutralとし、現在値に近い自由回転のturnへ持ち上げない。
たとえば現在+90度で観測-120度を+240度へ変換すると、正側の端に張り付き続ける。
正負の制限とneutralへ戻る観測をそのままbounded jointとして減衰する。
可動域の超過を上腕へ移す補正は加えない。

骨格監査前の中間修正で、最新CSVの1046観測と今回のモデル骨格を、60 Hz、ミラーON、胴体固定で
本番と同じ仮想→観測stageへ一時的に再投入した。seq 800–1020、palm weight>0.95
における観測法線と合成掌法線の前腕軸上の誤差が90度を越えるtickは、左351/446→
54/446、右114/324→21/324。左のseq 900は59.14→2.67度、右のseq 1000は
26.02→3.43度。比較側は旧clearを挿入した同一入力である。残る誤差には応答の
遅延と可動域外の観測が含まれ、ログ再投入を実機映像の完全一致とは扱わない。
一時プローブは撤去した。

常設確認は、両stageの実行順序で60度の掌回旋へ追従して元へ戻ること、
frame消失時のtargets clear、角度wrap・可動域の正端→負端→neutralで保持状態が
張り付かないこと。既存の固定骨長、肩・肘の減衰、指・親指、ミラーも確認する。
**更新後の実機映像は未確認**。

追加（2026-10-01, 人体骨格の自由度とFKへの統一）:
人体の構成は[Holzbaur, Murray, Delp (2005), pp. 830–831](https://nmbl.stanford.edu/publications/pdf/Holzbaur2005.pdf)
を採用する。これはOpenSim公開上肢モデルの原論文で、肩の3自由度、肘の屈伸、
前腕の回内/回外、手首の屈伸と橈尺屈を区別する。肘屈伸は固定軸で0–130度、
回内/回外は前腕長軸で±90度という同モデルの範囲を使う。これらを万人の実測
可動域や筋力と見なさず、このアプリで選択した公開モデルの関節制約とする。

- 肩: 上腕のorientationの3自由度を観測曲げ平面で解く。VRMのshoulderボーンは
  鎖骨側であり、上腕起点の球関節とは別物。現在の入力には鎖骨/肩甲骨の独立姿勢が
  無いため、モデルのrestを保つ。上腕の回転を0.35倍して鎖骨へ写す処理は撤去した。
  これは肩甲帯の筋骨格運動を再現したという意味ではない。
- 肘: `ArmIkInput.elbow_axis`をimmutable rest骨格から定義し、poleが指定する
  曲げ平面へ上腕を向け、前腕をその固定軸の屈伸だけで解く。独立shortest-arcで
  両骨を向ける方式と、追跡だけが後から曲げ軸を修正する方式を廃止した。
  仮想腕・追跡腕とも同じ[ozz IKTwoBoneJobの固定middle-axis方式](https://guillaumeblanc.github.io/ozz-animation/documentation/ik/)
  に合わせる。130度を超す近すぎるtargetは骨を折り返さず、制限内のreachへ置く。
- 前腕: 回内/回外を前腕の軸に適用し、handはFKで継承する。handへ0.5を追加する
  回旋分配とその補助関数を削除した。手首の軸回旋という余分な自由度を作らない。
  入力は掌法線だけなので、手首の屈伸・橈尺屈の2座標を完全には観測できない。
  それらを推測で生成せず、handのrestを保持する。
- 親子階層: 親骨の回転は子骨へ剛体として一度だけ継承する。親の回転をlocalの
  上腕0.3・肘0.15へさらに配分する独自処理は、仮想/追跡双方から削除した。
  shoulder trimは鎖骨のlocal姿勢変更としてFKだけで伝わり、肘角を勝手に変えない。
  ノイズ減衰は各関節の観測座標に対して行い、剛体の親子継承の倍率には使わない。
- モデル対応: [VRM humanoidの階層](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/humanoid.md)
  と[公式のrest回転を使う姿勢変換](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm_animation-1.0/how_to_transform_human_pose.md)
  に従い、位置と骨長はモデルのrestから固定する。model-space回転を各骨のrest-globalで
  conjugateしてrest-localへ合成する。VRM1のrest回転をidentityと仮定しない。

不要になったshoulder-followの実装、プロファイル値、保存値、設定スライダーを削除した。
筋肉係数を創作する代わりに、観測を臨界減衰した関節座標へ変換し、制約を守った骨格を
再構成する。独自の3度/5度保持幅も使わない。

最新1046観測とAvatarSample_Cの実際のrest回転・位置・指参照を、60 Hz、ミラーON、
胴体固定で本番の仮想→追跡stageへ再投入した。解が出た1331 arm ticksで最大骨長誤差は
1.20e-7 m、最大肘屈曲109.998度、hand-local軸回旋はidentityだった。一時プローブは削除した。
常設では左右・非identity骨軸・複数poleでも肘屈伸軸が変わらず、FKがIKの肘/手首へ
一致すること、130度制限、掌回旋への追従と復帰、同一実行順序でフィルタ状態が保たれる
ことを確認する。これらは人体そのものの筋骨格シミュレーションの再現確認ではない。

ローカル確認: avatar lib 384件、arm_ik 9件、arm_pose 9件、arm_tracked_mirror 4件、
arm_virtual_hand 3件（計409件）とapp settings 31件が成功。
変更avatarのClippyは成功（既存の警告10件のみ）。fmt、diff checkも成功。
`cargo build -p vtuber-desktop -j 1`で`target/debug/RusTuberV.exe`を21:42:48に更新した。
ビルド完了時点では修正後の実機映像は未確認だった。その後、利用者から改善と
安定感の確認を受けた。現在の採用判断と維持する原則は冒頭の節を適用する。

追加（2026-10-02, カメラ外への退避を関節座標で初期姿勢へ戻す）:
手をカメラから外したときの復帰は、追跡中の安定性を確認した関節フィルタの前で、
観測手首と初期手首を直線補間し、poleをその移動中の手首からの方向として補間していた。
両端が自然な姿勢でも、手首が肩近傍を通ると到達距離が小さくなって肘が深く折れ、
曲げ平面も反転する。対向poleは途中を退化と扱って前姿勢を保持していたため、
初期姿勢へ戻る途中そのものが停止する場合もあった。さらに、復帰先には仮想腕の
最終姿勢ではなく、その計算前の目標点だけを取り出して再度IKを解いていた。

復帰は[ozzのIK後のlocal姿勢合成と階層再計算](https://guillaumeblanc.github.io/ozz-animation/documentation/ik/)
に従い、既に採用した[公開上肢モデルの関節自由度](https://nmbl.stanford.edu/publications/pdf/Holzbaur2005.pdf)
を保持する。`resolve_tracked_side`は通常の仮想腕経路が解決した初期姿勢を取得し、
観測IKの肩・肘を既存フィルタで減衰した後、肩のQuaternionと固定軸の肘屈伸角を
欠測重みで合成する。その座標からモデルの固定骨長でFKを再構成し、胴体・鎖骨・
上腕・前腕・手・指の実際の親子階層を唯一のコンポジタで適用する。
初期姿勢の鎖骨trimも同じ重みで戻す。親の回転の再配分や二重適用は追加しない。

掌回旋は初期姿勢へ戻した腕ではなく、減衰済みの観測腕を基準に測定する。
カメラ外で保持される掌法線を、戻り動作に合わせて別の回旋目標に解釈し直さない。
その有限の回内/回外座標を欠測重みでneutralへ戻し、再構成した前腕長軸へ一度だけ
適用する。手首localはrest、指は既存の各関節の重みでrestへ戻る。観測腕の重みが0に
なったら解決済み初期姿勢をそのまま書き、失われた観測のフィルタ状態を終了する。
初めから観測が無い側も同じ初期姿勢を使う。

肘だけが欠測し手首が見える場合は、同じ観測手首に対する初期poleと観測poleを
それぞれ固定軸IKで解き、肩のQuaternionとして合成する。poleの絶対重みは腕の
重みで割って条件付き寄与にする。腕と肘が一緒に欠測した際、同じ重み減少を
肩へ二重適用しない。手首を保持した肘だけの復帰でも固定骨長と肘軸を保つ。
不要になったCartesianの`blend_arm_targets`/`blend_pole`とその経路専用テストは削除した。
hold/return/acquireと追跡中の応答時定数は変更しない。

再現確認では、肩の反対側まで挙げた腕の復帰で、旧処理が両端より肘を深く折ることを
既存の欠測テストで検出した。修正後は両端の屈伸角の間で戻る。実際の
仮想→観測→コンポジタ順序にも、左右、非identityの胸・鎖骨・腕・手のrest回転、
掌回旋と指屈曲、復帰中の胴体回転、共有LossBlendのhold/5秒復帰を投入した。
全tickで骨長・local位置・scaleと手首restを保ち、末尾で肩から指まで実際の
初期local姿勢へ戻ることを確認する。対向poleも復帰を停止せず、ミラーと追跡中の
掌回旋への追従も既存テストで確認した。ローカル確認はavatar lib 380件、arm_ik 9件、
arm_pose 10件、arm_tracked_mirror 4件、arm_virtual_hand 3件が成功。変更範囲のClippyは
成功（既存avatar警告10件のみ）、`cargo build -p vtuber-desktop -j 1`で
`target/debug/RusTuberV.exe`を更新した。これらは合成入力の確認であり、
修正後のカメラ映像・VRM実モデルの見え方・macOSは未確認。

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
