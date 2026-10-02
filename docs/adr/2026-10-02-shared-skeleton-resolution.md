# 人体骨格の解決を共通化する

状態: 採用。2026-10-01に実機で確認した腕の安定性を維持する。

## 問題

腕の観測・欠測復帰には固定肘軸が適用されていたが、接地は大腿と下腿を
別々のrotation arcで合わせていた。足先の位置が合っても膝へ独立した回旋が入る。
仮想腕にも解決後の前腕回旋を上腕へ配分する旧補正が残っていた。
指の初期curlは骨のlocal軸、観測curlは掌のrest形状から軸を選んでいた。

胴体のleanはworld回転をlocal回転へ直接掛け、モデル配置の回転を角度計算と
world軸に二度使っていた。大きなleanを前フレームのアニメーション姿勢として
取り込むこともあった。接地による骨盤補正は腕の解決後に行われ、解決に使った
胸の姿勢と最終的に継承する姿勢が異なっていた。

## 採用する構成

`vtuber-avatar::skeleton`を独立した共通モジュールとする。追跡や接地の入力生成、
時間フィルタ、欠測の重み、設定の読み取りは各処理が担当する。骨格の解決は
共通のrest形状・関節座標・座標変換・実際の親子階層から行う。

- 腕と脚は同じball joint + 固定middle hingeのtwo-bone IKを使う。
  骨長と軸はrest由来とし、位置の合わせ込みで関節自由度を増やさない。
  中間の補助ノードを含む親子関係も維持する。
  まっすぐなVRM T-poseの肘はモデル前方（+Z）へ屈曲する固定軸を使う。
  掌の外積の向きや前腕の回内/回外から肘屈伸の正方向を決めない。
  初期姿勢・仮想腕・肘欠測時のpoleも同じ固定軸から作る。T-poseの上腕を下げた
  基準から手首方向へ屈曲面を運び、法線をpole方向に流用しない。
  初期姿勢はモデルの肩位置と上腕長+前腕長から作る45度のAポーズとし、
  静的姿勢・仮想腕neutral・全欠測復帰で共有する。腰からの固定比率で手首を置かない。
- 観測の減衰、初期姿勢への欠測復帰、仮想腕の下降制限、前腕回旋は、同じ
  関節座標からFKを再構成する。肘屈伸と前腕回内/回外を分離し、手首へ回旋を配分しない。
- 指の初期curlと観測curlは、同じ掌のrest形状と共通hinge解決を使う。
  観測指は既存のMCP屈伸・開きとPIP/DIP屈伸を維持する。掌の形状が不足する
  モデルへ架空の曲げ軸を足さない。観測されない親指基部はrestに置く。
- 胴体と眼のrest-relative回転、world回転からparent-local回転への変換、
  ancestor/subtreeのGlobalTransform再計算を共通化する。local位置とscaleは動かさない。
  rootのbody位置と接地の骨盤移動は既存の位置チャンネルが所有する。
- leanは意味座標で角度を求め、配置を含むworld軸を現在の親座標へ変換して合成する。
  前回加算したleanは次のanimation評価前に外す。接地も同じ既存の復元方式を維持する。
- 仮想腕の旧twist relaxer、上腕への回旋配分、その設定・専用geometry・専用テストを削除する。

既存の初期姿勢/設定変更の遷移は同一rest軸のhinge同士をlocal空間で補間する。
観測腕の欠測はこの遷移を通さず、解決済み初期姿勢と観測の関節座標を共通FKで合成する。
指の欠測も屈伸・開きの各角度へ重みを掛けてから解決する。

更新順序は、加算姿勢の復元 → animation → 胴体回転/位置/lean → 接地と骨盤補正 →
仮想/観測腕の解決 → 腕と指のcompositor → VRMのgaze/constraints/通常のtransform伝播とする。
腕は最終的な骨盤・胸の姿勢を一度だけ継承する。

## 位置入力の段階と仮想腕（#222）

位置チャンネルは`update_body_tracking_position_input`が一度だけ生成し、
同じrootの`BodyTrackingPositionInput`に次の二段階を公開する。

- `tracked_head_target` / `tracked_body_target`: translation shaping、軸別split、
  mirrorを終えた目標。idle混合・body-follow・表示用confidence適用の前。
- `head_offset` / `body_offset`: idle混合後に既存`BodyFollowFilter`を一度通した
  胴体用入力。既存の`weight`はdirect position writerが一度適用し、root移動と
  bounded leanを描画する。これらも最終boneのworld位置ではない。

仮想手のhips-relative anchor補償は前者を使う。これは既存の位置応答を維持する
判断であり、意図的に遅れる胴体の表示位置へ参照先を変えない。手先の補償目標に
body-followをもう一度入れず、`(tracked_head_target + tracked_body_target) *
compensation_gains`を既存のanchor生成へ渡す。胴体側のidleとconfidenceは腕へ
再適用しない。腕自身に新しいフィルタや補正は足さない。

一方、torso lagは目標回転ではなく、胴体・lean・接地後の実際の胸回転を参照する。
仮想anchorの既存counter-rotationと固定骨長two-bone解決を保ち、root・骨盤・胸の
表示移動は親子FKで一度だけ継承する。表示済みroot/world位置をanchorへ再加算しない。
目標位置と表示回転は役割が異なる入力であり、段階差を不具合とは断定しない。

ロスト中も既存control frameが持つ減衰目標を共有する。frameがなくなればtracked
目標はゼロとなり仮想overrideは解除され、胴体だけが既存idleへ移る。世代不一致・
非Readyでは両段階を無効化する。`TrackedPose`では観測stageの所有権・減衰状態を
維持し、仮想stageは入力を読まずreturnする。#183の指追従と#184の性能比較の受入は
この整理に含めない。

## 一次資料

[ozz IKTwoBoneJob](https://guillaumeblanc.github.io/ozz-animation/documentation/ik/)の
腕・脚に共通する固定骨長、middle axis、local補正と階層再計算に従う。
腕の肘0–130度、回内/回外と指の既存範囲の根拠は
[観測腕ADR](2026-09-13-observed-arm-tracking.md)を維持する。

脚の股関節ball/膝の一つの屈伸座標は
[OpenSim Gait2392の公式説明](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53086215/Gait+2392+and+2354+Models)
に合わせる。[公式モデルのknee_angle_r](https://github.com/opensim-org/opensim-models/blob/master/Models/Gait2392_Simbody/gait2392_millard2012muscle.osim)
の屈曲端120度を使い、接地で過伸展させない。伸び切りでもpoleから膝の固定平面を
解決できるようにする。足のrest接地位置・向きと既存の弱い骨盤反応は維持する。
筋肉モデルや新しい連動係数は導入しない。

任意のrest回転とhips移動/local関節回転による人体姿勢の表現は
[VRM公式のhumanoid pose説明](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm_animation-1.0/how_to_transform_human_pose.md)
を参照する。VRMのモデル定義に基づく眼のyaw/pitch範囲、constraint、spring bone、
import時のrest正規化はそれぞれの仕様上の役割を維持する。

## 確認

本番の骨格Transform writerはdirect pose、direct position、腕compositor、接地、
眼のdirect lookを監査した。idleやpose入力の生成は別の骨格writerを持たない。

既存の腕/指/欠測復帰と接地の確認に、膝軸・固定骨長、120frameの大きいleanと解除、
配置回転と非identityの関節rest軸、接地→腕の本番schedule順序を加える。
合成入力とローカルビルドの結果は実機の見え方の確認と区別する。

ローカル確認: avatar libと関連8 integration targetの423件、設定の22件が成功。
接地は直立/屈曲rest、非identityの骨と補助ノード、配置yawとscaleを確認した。
旧胴体処理へ戻す比較では追加した2件が失敗し、修正後は成功する。
変更クレートのClippyは成功（既存警告を除く追加警告なし）。
`cargo build -p vtuber-desktop -j 1`で`target/debug/RusTuberV.exe`を更新した。
今回の変更後の実カメラ映像、VRM実モデルの目視、macOSは未確認。

#222のローカル確認: body_motion 8件、arm_pipeline 26件、tracked_arm絞込32件、
関連integration 45件が成功。公開tracked目標への参照、body-follow出力との分離、
idle時の目標ゼロ、世代・ロスト復帰、FKと固定骨長を既存の合成入力で確認した。
観測腕・idleのfixtureは実lifecycle世代へ移行した。実カメラとVRM目視、macOS、
NDI送受信は未実施であり、2026-10-01の実機評価を今回の実機証拠として流用しない。
