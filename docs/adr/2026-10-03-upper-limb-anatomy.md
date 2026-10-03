# 上肢の関節座標と制約（Epic #248）

状態: 実装中。初期姿勢と #249 の胸郭基準座標を導入。Epic の完了記録ではない。

## 初期姿勢

利用者の指定により、2026-10-03から水平から80度下げ、鉛直から外側へ10度残す。
`ArmPoseProfile` の一つの値を初期表示・仮想腕・欠測復帰で共有する。
腕長0.5 mなら手首は肩から約0.087 m外側になる。これは衣服を含む全VRMへの
非貫通保証ではなく、#256 の表示形状による判定と #257 の制約解決が必要である。

## 肩の座標

`shoulder.rs` は骨localのEuler角ではなく、親の現在回転を一度だけ除いた
胸郭基準の回転を扱う。+Yが上、+Zが前、左の外側が+X。左右反射では
方向をpolar vector、肘屈曲軸をaxial vectorとして扱う。
restの上腕方向と固定肘軸から基準フレームを作り、非identityのbone restを
共役変換で保存する。neutralは上腕が下、90度曲げた肘の前腕が前を向く姿勢。

挙上面 p は外側0度・前方90度、挙上 e は真下0度・真上180度、軸回旋 a は
内旋を正とするglobe座標。側の符号 s を左+1・右-1として、neutralからの回転は
`Ry(-s*p) Rz(s*e) Ry(s*p) Ry(-s*a)`。
最初の3項を下向きベクトルからの最短swingとして計算し、残差twistから a を回収する。
ISBの反復軸3回転と比べ、最後の軸回転は挙上面の逆回転を含むため、a を
ISBの第3角へそのまま代入しない。肩甲帯が未実装の段階ではこの角度は
thoracohumeralであり、glenohumeralの独立観測とは呼ばない。

Xu et al. が記載するHolzbaurの範囲を採用する: p=-90〜130度、e=0〜180度、
a=-90〜20度。これは同モデルの定義域であり、全個人の生理的最大値ではない。
制約後に同じ固定骨長FKで肘/手首を再構成し、内外旋超過を前腕回旋へ転嫁しない。
観測・初期・仮想・復帰は同じ肩座標の制約を通る。

下向きの極では挙上面を `None` とし、方向だけから架空の面を測定しない。
正確な上向きの極では面とtwistの分離が一意でなく、変換結果を `None` として
既存の未解決姿勢の保持へ伝える。極の判定はf32精度に基づくもので、角度保持幅ではない。
禁止領域の枝選択、速度状態、非凸領域での遷移は #258 が未完了。

## 一次資料と利用範囲

- [Holzbaur et al. (2005), pp.830–831](https://nmbl.stanford.edu/publications/pdf/Holzbaur2005.pdf):
  neutral、15自由度、肘/前腕/手首の区別。論文の数式・定義を独立実装し、骨形状やコードは転載しない。
- [Xu et al. (2012)](https://doi.org/10.1016/j.jbiomech.2012.08.018):
  HolzbaurとISBの回転規約が同一でないこと、肩の3角の範囲。
- [Contrasting action and posture coding (2023), Eq.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC10361732/):
  挙上面の逆回転を含むglobe回転積の照合。
- [Maier et al. (2024)](https://biomechanics.stanford.edu/paper/JBIOM24.pdf) と
  [公開コード dfcb9dc](https://github.com/stanfordnmbl/shoulder-personalization/tree/dfcb9dcda9fc86288cc2df849e04a77612a6b1ce):
  肩甲帯の個人化には肩甲骨の触診・専用マーカー測定が必要。Poseの肩点だけで同等の個人化を済ませたとはしない。
- [PosePrior利用条件](https://poseprior.is.tue.mpg.de/license.html):
  非商用科学研究向け。学習済み制約・データを本OSSへ転用していない。

## 確認

Windows x86_64で初期姿勢変更後の `vtuber-avatar` lib、arm_ik、arm_virtual_hand の
395テストと `cargo check -p vtuber-app -j 1` が成功。
肩座標追加後は既存到達確認を制約内の目標へ更新し、制約外のtwistが肘/前腕へ
転嫁されないこと、左右と非identity rest、上下の極を局所数値確認する。
実カメラ、表示メッシュと手の非貫通、VRM形式ごとの目視、macOSは未確認。
