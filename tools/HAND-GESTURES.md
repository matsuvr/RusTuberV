# Camera hand gesture recognition / カメラ画像からの手形選択

## 日本語

手形は HaGRID の軽量画像分類器 `crops_classifier.onnx` で選びます。
既存の MediaPipe Hand Landmarker が見つけた手の画像を切り抜き、45ラベルの
最大ロジットを採用します。向きの違いをまとめた23種類の指形へ対応づけます。
指の関節角度を測って近いポーズを探索する処理は廃止しています。

- モデルは411,683バイトで実行ファイルに埋め込みます。起動時のモデル取得や
  Python / ONNX Runtime の追加インストールは不要です。
- 既存のPose推論ワーカー内で一度ロードし、検出された手ごとにtractで実行します。
  手の位置合わせ、左右の対応づけ、手首・掌の向きは既存のトラッキングを使います。
- 同じ指形が新しいカメラフレームで4回続いたら切り替えます。保持された1枚を
  描画で繰り返しても確認回数は増えません。左右の手の状態は独立です。
- 補間と手の消失時の復帰処理を引き継ぎ、指はカタログの解剖学的範囲内の角度で動かします。

この45ラベルには「該当なし」がありません。手が検出されている間は最も近いラベルを
選びます。45種類すべてを異なるアバターポーズとして再現するものではありません。
掌の向きが違うラベルは同じ指形へまとめ、ハートの片手部分などは既存ポーズで近似します。
小さいモデルファイルでも推論の速さはCPUに依存します。対象端末での30Hzとカメラ映像での
認識精度は実測が必要です。

## English

The existing Pose worker runs the embedded HaGRID crop classifier through tract.
The MediaPipe hand detections locate each crop; each classification is attached to
its detection **before** anatomical wrist pairing. The maximum of 45 logits selects
one of 23 authored finger configurations. Landmark-angle nearest-pose fitting is
removed. The model does not expose a no-gesture class or calibrated confidence;
there is no additional score threshold.

A pose changes after four consecutive new camera observations, independently per
hand. Orientation variants are folded together before this confirmation. Existing
render-clock interpolation and hand-loss blending remain in charge of transitions.
Palm orientation comes from landmarks, while bounded finger articulation comes
from the catalog. The two heart-half variants share an authored approximation.

### Model and preprocessing

- Source: [ai-forever/dynamic_gestures](https://github.com/ai-forever/dynamic_gestures/tree/557a836bf82fce8d36494d0c17e95ca236f5611c).
- File: `models/crops_classifier.onnx`, copied unchanged; 411,683 bytes.
- Git blob SHA-1: `d11dd4f110e4e71ff480e8d2de3195bfbc48d749`.
- SHA-256: `12a02344f63a7c4f2a2ca90f8740ca10a08c17b683b5585d73c3e88323056762`.
- License: [Apache-2.0](../assets/models/LICENSE.hagrid.txt), also shown in the app.
- Input: RGB float32 `[1, 3, 128, 128]`; output: 45 float32 logits.
- Crop: axis-aligned bounds of all 21 normalized image landmarks, expanded to a
  square with a 1.2 scale to include the skin outline, then clipped to the image.
  No rotation or horizontal reflection is applied. Near the image border,
  clipping can make the crop rectangular.
- Resize: half-pixel bilinear interpolation to 128 × 128, round interpolated byte
  values, then normalize each RGB channel as `(pixel - 127) / 128` in CHW order.
  OpenCV's fixed-point uint8 interpolation may differ by one byte value.
- The 1.2 crop margin is an integration choice for landmark bounds; it is not an
  upstream accuracy guarantee. The upstream implementation uses detector boxes.

The model is loaded once, with batch size one, and runs sequentially for each
hand on the existing worker. Tract's unused transformer operators are disabled
in this path. There is no new worker or extra detector. The small
file size is not a throughput measurement: measure camera recognition and the
30Hz budget on the target CPU, including the existing MediaPipe workload.

### Label-to-pose mapping

Label spellings and order come from upstream `utils/enums.py`. These are static
hand configurations; dynamic swipe/click event recognition is not part of this path.
The catalog contains the existing authored poses plus the missing middle finger,
half-closed hand, grip, click shape, and three-finger gun configurations.

| Classifier labels | Authored finger pose |
|---|---|
| `palm`, `hand_down`, `hand_right`, `hand_left` | `open` |
| `thumb_index` | `finger_gun` |
| `like`, `dislike`, `thumb_left`, `thumb_right`, `thumb_down` | `thumbs_up` |
| `half_up`, `half_left`, `half_right`, `half_down` | `half_closed` |
| `part_hand_heart`, `part_hand_heart2` | `half_heart` |
| `fist`, `fist_inverted` | `fist` |
| `two_left`, `two_right`, `two_down`, `two_up`, `two_up_inverted` | `two_together` |
| `grabbing` | `claw` |
| `grip` | `grip` |
| `point` | `point_click` |
| `call` | `shaka` |
| `three3` | `i_love_you` |
| `little_finger` | `little_only` |
| `middle_finger` | `middle_only` |
| `four` | `four` |
| `mute`, `one`, `one_left`, `one_right`, `one_down` | `point` |
| `ok` | `ok` |
| `peace`, `peace_inverted` | `peace` |
| `rock` | `horns` |
| `stop`, `stop_inverted` | `open_together` |
| `three` | `three_thumb_index_middle` |
| `three2` | `three` |
| `three_gun` | `three_gun` |

### Focused verification

On Windows, use the repository's CLANG64 wrapper:

```powershell
./tools/cargo-windows.ps1 test -p vtuber-inference hand_gesture::tests --lib
./tools/cargo-windows.ps1 test -p vtuber-inference pose_decode::tests --lib
./tools/cargo-windows.ps1 test -p vtuber-tracking --lib
./tools/cargo-windows.ps1 test -p vtuber-avatar tracked_arm::tests --lib
```

The tests cover RGB/BGR stride handling and bilinear normalization, the real
embedded graph's neutral-input reference logits, output-label order, detection
pairing, confirmation on new camera frames, per-side reflection, loss blending,
and bounded avatar finger articulation. Synthetic neutral-input inference is a
runtime-contract check, not a gesture-accuracy benchmark.
