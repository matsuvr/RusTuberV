# MToon互換のセル陰影リッチ表示（実験）

`codex/experimental-mtoon-cel-rich` は、既存の「リッチ表示」のMToon描画を
置き換える実験。OFFは既存のNative表示、ONはこの方式。表示方式の追加や
設定ファイルの変更はなく、既存の強度スライダーを使う。

## 採用する考え方

公開版 [NiloToonのLightingEquation.hlsl](https://github.com/ColinLeung-NiloCat/UnityURPToonLitShaderExample/blob/fbe276e0df1d1f9379e13581c20b6a080a1a6977/SimpleURPToonLitOutlineExample_LightingEquation.hlsl)
から、柔らかいセル境界、追加光の寄与制限（0.25）、近距離減衰の上限（4）、
環境光と直接光のRGB成分ごとの最大値合成を取り込む。
製品版NiloToonURPの再現や公開版の忠実移植を目的にしない。

- セル境界の位置・幅は、MToonのshading shift（テクスチャを含む）とtoonyから決める。
  符号付きN·Lを使い、画素微分で境界のちらつきと幅0の除算を避ける。
- 色は作者のlit/shadeカラー・テクスチャを使う。固定の影色で上書きしない。
- Bevyの光量は露出を一度適用してから制限する。直接光の総エネルギーが1を
  超える場合だけRGB共通の係数で正規化し、色相を維持する。
  テクスチャ色や発光色自体を0〜1へ制限しない。
- 追加のPBR光沢と自動Fresnelリムを除去する。作者指定のmatcap・リム・発光は残す。
- Nativeとセル陰影の最終色を強度で補間する。共有の追加ライトはStandard材質のために
  光量も強度に比例するので、MToon側はそれを戻してから補間し、強度の二重適用を避ける。
  強度0ではNativeの計算結果をそのまま返す。

## 互換性の境界

`ExtendedMaterial<MToonMaterial, RichMtoonExtension>` と既存のNative所有権を維持する。
alpha mask/blend、両面、深度、テクスチャ、UVアニメーション、表情材質同期、
モーフ・スキニング、作者の輪郭線幅・色・幅テクスチャを引き継ぐ。
輪郭色のlighting mixにも同じセル陰影が反映される。

顔材質の自動判定は行わない。顔用SDFや専用テクスチャを要求せず、作者のshade/shift/toonyを
使う。公開版の顔専用影位置・奥行き補正や画角連動の輪郭線は、MToonの作者設定を
変えてしまうため取り込まない。上流にない輪郭のscreen-spaceモードも追加しない。

ライトは現在の正面方向光と、既存プリセットの2つのスポット光を使う。
双方のシャドウマップ有効フラグを尊重する（既存プリセットは影無効）。
点光源・矩形光源・ライト投影テクスチャの新規対応は含まない。
非MToonのStandard/unlit材質は既存経路を使う。

## 輪郭描画の切り替え修正

上流のNative輪郭描画は、材質を取り外したメッシュの登録を保持していた。
旧Richでは同じbind group構成だったため見えにくかったが、強度uniformを追加すると
NativeパイプラインにRich材質が渡り、GPU検証エラーになる。
現在使用中の依存コミットを `vendor/bevy_vrm1` に固定し、抽出時に登録をクリアして
そのフレームのNative材質だけを登録する修正を加えた。元の依存からの機能差分は
`src/vrm/mtoon/outline_pass.rs` のこの修正だけ。全体の描画方式や公開APIは変えていない。

## 検証

Windowsのコマンドは `tools/cargo-windows.ps1` 経由で実行する。

```powershell
.\tools\cargo-windows.ps1 -CargoArguments @('test', '--package', 'vtuber-avatar', '-j', '1', '--lib', 'look::')
.\tools\cargo-windows.ps1 -CargoArguments @('run', '--package', 'xtask', '-j', '1', '--', 'vrm-render', 'tests/fixtures/vrm/inore-vrm1.vrm', 'target/cel-rich-after')
.\tools\cargo-windows.ps1 -CargoArguments @('run', '--package', 'xtask', '-j', '1', '--', 'vrm-render', 'tests/fixtures/vrm/tsukuyomi-chan.vrm', 'target/cel-rich-after')
```

`vrm-render` は同じモデル・姿勢・カメラで OFF → ON → 0.5 → OFF → ON → 0 を
GPU描画し、PNG保存と画素差の検査を行う。製品版の作例との一致を判定するものではない。
実施結果はこの文書の末尾に記録する。

この作業時の `alicia-solid.vrm` と `seed-san.vrm` は中身がHTMLであり、描画検証対象から
除外している。既存ファイルの取得や修正はこの実験には含めない。

CPU 3コアで描画60fps・推論30Hz、カメラ追従中、Looking Glass多視点、macOS実機での
性能・品質は別途実測が必要。画面全体の後処理・新規描画パス・CPUの新規システムは追加しない。

実験を採用しない場合は、変更をこのブランチに保存した状態で `git switch main` へ戻せる。

## 2026-10-16の実施結果

Windows / MSYS2 CLANG64で `look::` 25件、ライセンス4件のテストが成功。
`cargo fmt --all -- --check` と `git diff --check` が成功。
`clippy --all-targets -- -D warnings` は既存の `vtuber-core/src/frame.rs:148` と
`video_output.rs:260` の `chunks_exact_to_as_chunks` 警告で失敗した。
通常のClippy（avatar / app / xtask、全ターゲット）は既存警告ありで成功した。

ユーザー指定の2体について、GLB内の材質宣言も確認した。

| モデル | MToon | Standard | unlit |
| --- | ---: | ---: | ---: |
| 6208967401843641193.vrm (VRM 0.x) | 6 | 1 | 1 (UnlitCutout) |
| AvatarSample_C.vrm (VRM 1.0) | 10 | 1 | 1 |

```powershell
.\tools\cargo-windows.ps1 -CargoArguments @('run', '--package', 'xtask', '-j', '1', '--', 'vrm-render', 'data/vrm_data/6208967401843641193.vrm', 'target/cel-rich-user-models')
.\tools\cargo-windows.ps1 -CargoArguments @('run', '--package', 'xtask', '-j', '1', '--', 'vrm-render', 'data/vrm_data/AvatarSample_C.vrm', 'target/cel-rich-user-models')
```

GPU描画の平均絶対画素差（0〜255、256×256、同一姿勢・時刻・カメラ）:

| モデル | OFF対ON | OFF復帰 | ON再切替 | 強度0対OFF | OFF対強度0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6208967401843641193 | 1.395 | 0.000 | 0.000 | 0.000 | 0.720 |
| AvatarSample_C | 1.446 | 0.000 | 0.000 | 0.000 | 0.751 |
| inore-vrm1 | 5.111 | 0.000 | 0.136 | 0.000 | 2.517 |
| tsukuyomi-chan | 1.884 | 0.000 | 0.000 | 0.000 | 0.907 |

指定2体では不透明画素数もOFF/ONで一致した。PNGとログはローカルの
`target/cel-rich-user-models/` と `target/cel-rich-user-model*.log` に保存される。
画素差は復元・変化の検査であり、美しさの指標としては使っていない。

Computer Useで実アプリを操作し、指定2体を読み込んでリッチ表示を切り替え、
全画面の拡大表示をVisionで確認した。1体目は目・リボンの色、ベストとスカートの
折り目、髪の束を確認。2体目はOFF/ONの同じ構図で、目元、肌の階調、
ジャケットの折り目・襟・袖を比較した。作者の配色を残した柔らかな立体感として、
見た目は良好と判断した。派手な光沢を増やす方向ではない。

自動ドラッグでは視点が大きく上方へ跳び、狙った斜め角度を安定して設定できなかった。
上方からの描画は見たが、任意角度の品質保証とはしない。正面へはUIの画角リセットで戻した。
リッチ表示の初回切り替え直後には、パイプライン準備中に一部材質が一時的に消える場合がある。
準備後の表示とGPU検査では復帰した。カメラ推論・CPU3コア性能の実測は未実施。

## 2026-10-17の性能測定

このPCでは、リッチ表示ONで約60fps、FaceとPose/Handの実推論で約30Hzを維持した。
OFFに対するGPU処理時間の増加は小さい。ただし単発の約33msの画面提示間隔があり、
全フレームが16.67ms以内だったという意味ではない。

### 条件と範囲

- Core i9-13900 / RTX 4090、Vulkan、NVIDIA 617.42、Windows 11 Pro 10.0.26300。
- MSYS2 CLANG64のreleaseビルド。`opt-level=3 / thin LTO / codegen-units=1`。
- 起動前からプロセスaffinityを `0x15` に制限。論理CPU 0・2・4の3個を使用。
  実行中の `xtask.exe` でもaffinity 21を確認。Bevyの3プールは各1スレッド。
- VSync/FIFO、ウィンドウ1920×1080。通常プレビューは約972×546。
  通常のUI、VRM描画、Face・Pose・Hand推論、トラッキング、解剖制約ソルバを同時実行。
- 録画入力はRGB 640×360 / 30fps。実カメラのセンサー・ドライバ、NDI、Looking Glassは含まない。
- PresentMon 2.6のOS画面提示イベントを集計。アプリ内fpsだけで判定していない。
  光学的な表示遅延の測定ではない。既存計測器の2秒ごとのスクリーンショット取得を含む。

### 通常プレビュー：各条件3回

`WIN_20261004_22_48_07_Pro.mp4`（31.57秒）を使い、先頭5秒を除外。
OFF/ONを交互に実行し、2巡目は順番を逆にした。ONは強度100%。
fpsと推論Hzは3回の範囲、提示p95は3回の最悪値、GPU平均は各回平均の平均。

| モデル | リッチ | 画面提示fps | 提示p95 ms | GPU平均 ms | Face Hz | Pose/Hand Hz | 25ms超 / 提示数 |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| 6208967401843641193 | OFF | 59.96–60.00 | 16.71 | 2.392 | 29.91–30.00 | 29.98–30.00 | 1 / 4,781 |
| 6208967401843641193 | ON | 59.92–60.00 | 16.68 | 2.447 | 29.98–30.02 | 29.98–30.02 | 3 / 4,779 |
| AvatarSample_C | OFF | 59.85–59.96 | 16.70 | 2.534 | 29.96–30.00 | 29.96–30.02 | 6 / 4,776 |
| AvatarSample_C | ON | 60.00–60.00 | 16.70 | 2.737 | 29.98–29.98 | 29.98–30.02 | 0 / 4,782 |

GPU平均の増分はそれぞれ約0.055ms、0.203ms。各回のばらつきもあるため、
この差を他のGPUやモデルへそのまま適用はしない。25ms超はOFF側でも発生しており、
この小標本だけでシェーダーとの因果関係は確定できない。

### 全画面ON：各モデル1回の追加確認

Computer Useで全画面表示に切り替え、Visionで追従中の描画を確認した。
`WIN_20261004_22_29_52_Pro.mp4`（60.6秒）を使い、操作の影響を除くため
30秒以降の約30.6秒だけを集計。全画面を確認した時刻は入力開始から20.51秒／22.04秒。
元の全区間サマリーを残し、`fullscreen-presentation.json` に対象区間を別途集計した。
通常プレビューとは入力・UI描画量も違うため、両者のGPU時間を直接比較しない。

| モデル | 画面提示fps | 提示p95 ms | GPU平均 / p95 ms | Face Hz | Pose/Hand Hz | 25ms超 / 提示数 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 6208967401843641193 | 60.00 | 16.68 | 1.180 / 2.036 | 30.00 | 30.00 | 0 / 1,836 |
| AvatarSample_C | 59.97 | 16.69 | 1.411 / 2.263 | 29.98 | 29.98 | 1 / 1,835 |

### 保存先と再実行

集計値・ホスト情報・実行中affinity確認・集計スクリプトは
`data/performance/cel-rich-20261017-report/` に保存。
`cel-rich-performance-runs.csv` と `cel-rich-performance-fullscreen.json` の `source` に、
各回のmetadata・PresentMon CSV・フレームトレース・描画画像の保存先を記録している。
生データはローカルのみ。入力動画・VRM・描画画像はGitに追加していない。

```powershell
.\tools\measure-replay.ps1 -Video data/video/WIN_20261004_22_48_07_Pro.mp4 `
  -Model data/vrm_data/6208967401843641193.vrm -Runs 3 `
  -LogicalProcessors 3 -ProcessorAffinity 21 `
  -PresentMon data/performance/PresentMon-2.6.0-x64.exe -Rich -Tag cel-rich-on
```

`-Rich` を省くとOFF。`-NoBuild` は同じソースのreleaseビルドがある場合に使う。
build/runはいずれも `cargo-windows.ps1` を通す。初回releaseビルドは28分35秒だったが、
ビルド時間は測定値に含まれない。今回の結果は短時間の録画再生による性能確認であり、
長時間稼働、実カメラ入力、他のGPU・VRMでの性能保証ではない。
