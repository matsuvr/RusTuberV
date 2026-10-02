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
