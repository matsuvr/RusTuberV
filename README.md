# RusTuberV

**日本語** · [English](README.en.md) · [简体中文](README.zh-CN.md) · [한국어](README.ko.md)

RustとBevyで実装したWebカメラ入力のVRMアバターアプリケーションです。MediaPipeによる顔・姿勢・手の推論結果をアバターへ適用し、プレビュー、NDI®による背景透過映像の送信、Looking Glassへの出力を行います。

## ダウンロード

利用するだけなら、[GitHub Releases](https://github.com/matsuvr/RusTuberV/releases/latest)からWindows x64版またはApple Silicon Mac向けテスト版をダウンロードしてください。配布版には推論モデルと必要なランタイムを同梱しています。ソースからビルドする場合のみ、後述の開発環境の準備が必要です。

WindowsではZIPを「すべて展開」し、`RusTuberV.exe`をダブルクリックします。同梱ファイルはまとめて置いてください。Mac版はmacOS 14以降で、技術的な問題を自分で対処できる方向けのテスト版です。制限・確認済み環境・初回起動手順は[リリース説明](https://github.com/matsuvr/RusTuberV/releases/latest)を参照してください。

## 利用マニュアル

**[日本語の利用マニュアル](manual/ja/README.md)**

アバターの読み込み、カメラ設定、画角・表情の調整、OBSへの出力をスクリーンショット付きで説明しています。

## 主な機能

- VRM 0.x / 1.0の読み込み
- Webカメラによる顔、まばたき、視線、口、上半身・腕・手のトラッキング
- 骨長、骨の親子階層、関節軸、自由度、可動域を制約とするアバター制御
- 中立姿勢と腕の校正、左右反転、画角調整
- キーボードによる表情切り替えとモデルごとのキー割り当て
- リッチ表示とモデルごとの設定保存
- NDIによるアバター映像の送信
- Looking Glass Bridgeによる機器検出と出力
- 日本語、英語、中国語、韓国語のUI


## ビルド環境

| 項目 | 要件 |
| --- | --- |
| Rust | 1.99以上。`rust-toolchain.toml`はstableを指定 |
| Windows | MSYS2 CLANG64のRust / Clang / LLD、x86_64-pc-windows-gnullvmターゲット |
| macOS | Apple Silicon、Xcode Command Line Tools |
| NDI出力を含むビルド | NDI Standard SDKとClang / libclang |
| 描画 | Bevyの描画バックエンドに対応するGPUとドライバー |
| トラッキング | WebカメラとOSのカメラ利用許可 |

カメラ実装はWindowsのMedia FoundationとmacOSのAVFoundationを使用します。

通常のデスクトップビルドは`ndi-output`を有効にします。NDIを使用しない場合は、後述の`--no-default-features`でSDKへの依存を外せます。

## ソースの取得

```sh
git clone https://github.com/matsuvr/RusTuberV.git
cd RusTuberV
```

依存関係は`Cargo.lock`で固定しています。ビルドには`--locked`を指定してください。

## 推論モデルの準備

以下の3ファイルを`assets/models/`に配置します。モデル本体はGit管理の対象外ですが、ビルド時に実行ファイルへ埋め込まれるため、ビルド前に必要です。

| ファイル | 用途 |
| --- | --- |
| `face_landmarker.task` | 顔、表情、視線の推論 |
| `pose_landmarker_full.task` | 上半身と腕の姿勢の推論 |
| `hand_landmarker.task` | 手と指の推論 |

WindowsのPowerShellでは、リポジトリのルートで次を実行します。

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "assets/models/face_landmarker.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "assets/models/pose_landmarker_full.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "assets/models/hand_landmarker.task"
```

macOSでは次を実行します。

```sh
curl -fL -o assets/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -fL -o assets/models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
curl -fL -o assets/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

承認済みのSHA-256と取得元は[モデルマニフェスト](assets/models/manifest.toml)に記載しています。アプリはモデル読み込み時にハッシュを照合します。取得先の`latest`が更新されてハッシュが一致しなくなった場合は、マニフェストに対応するモデルが必要です。

MediaPipeのネイティブライブラリは、実行ファイルの隣にあるものを読み込みます。同梱されていないソースビルドでは、初回の推論開始時に公式PyPI wheelから取得してキャッシュするため、ネットワーク接続が必要です。ライブラリを別途指定する場合は、`MEDIAPIPE_LIB`にMediaPipe 0.10.35の対応ライブラリへのパスを設定します。

## Windowsでのビルド

[LLVMビルド環境の準備とMSVCへの切り戻し](tools/WINDOWS-LLVM.md)を参照してください。
Windowsでは`tools/cargo-windows.ps1`を通してMSYS2のRustを使用します。
Visual Studio Build Toolsは不要です。通常の`cargo`は従来のrustup環境を使用します。

### NDI出力を含む通常のビルド

[NDI SDKの公式配布ページ](https://ndi.video/for-developers/ndi-sdk/)からWindows用のStandard SDKをインストールします。NDIバインディングの生成にClang / libclangを使用します。

NDI SDKを標準の場所、MSYS2を`C:\msys64`にインストールした場合の例です。

```powershell
$env:NDI_SDK_DIR = "C:\Program Files\NDI\NDI 6 SDK"
.\tools\cargo-windows.ps1 build --release --locked --package vtuber-desktop -j 1
.\tools\cargo-windows.ps1 run --release --locked --package vtuber-desktop -j 1
```

ビルドスクリプトは、SDKのインポートライブラリに対応するNDIランタイムDLLを実行ファイルの隣へコピーします。DLLをSDK外に置く場合は、ビルド前に`NDI_RUNTIME_DLL`へそのファイルのパスを指定します。

配布ZIPは、次のコマンドでLLVMビルドから梱包まで続けて作成します。
LLVMの`libunwind.dll`とライセンスも同梱します。

```powershell
.\tools\package-windows.ps1 -Version 0.1.2
```

`release/`に`win-RusTuberV-0.1.2-x64.zip`と同名の`.zip.sha256`ファイルが生成されます。パッケージにはライセンス文書、設定、利用マニュアルも含まれます。Macの配布ZIPは`mac-RusTuberV-<バージョン>-arm64.zip`形式です。

### NDI出力なしのビルド

```powershell
.\tools\cargo-windows.ps1 build --release --locked --package vtuber-desktop --no-default-features -j 1
.\tools\cargo-windows.ps1 run --release --locked --package vtuber-desktop --no-default-features -j 1
```

この構成ではNDI SDKとNDIバインディング用のlibclangは不要です。推論モデルの準備とMediaPipeネイティブライブラリは必要です。

## macOSでのビルド

NDI出力を含む場合はmacOS用のStandard SDKをインストールし、SDKの配置先を指定します。

```sh
export NDI_SDK_DIR="/Library/NDI SDK for Apple"
cargo build --release --locked -p vtuber-desktop -j 1
./target/release/RusTuberV
```

Clang / libclangをビルド環境から参照できるようにしてください。標準の探索で見つからない場合は、`LIBCLANG_PATH`でライブラリのディレクトリを指定します。

ローカル実行用の`.app`を作る場合は、同じ環境で次を実行します。

```sh
bash tools/build-macos.sh
open target/release/RusTuberV.app
```

[ビルドスクリプト](tools/build-macos.sh)はアイコン、モデルマニフェスト、トラッキング設定を配置し、アドホック署名を行います。NDIライブラリはインストール済みSDKを参照します。この`.app`にはNDIランタイムの再配布パッケージ化とAppleの公証は含まれません。

NDI出力なしでは次を実行します。

```sh
cargo build --release --locked -p vtuber-desktop --no-default-features -j 1
./target/release/RusTuberV
```

## 起動オプション

起動時にVRMファイルを指定できます。

```powershell
.\target\release\RusTuberV.exe --model "C:\Avatars\avatar.vrm"
```

```sh
./target/release/RusTuberV --model /path/to/avatar.vrm
```

指定しない場合は、アプリの「VRMを読み込む…」またはファイルのドロップから読み込みます。カメラの選択と配信設定は[利用マニュアル](manual/ja/README.md)を参照してください。

## プロジェクト構成

| パス | 役割 |
| --- | --- |
| `apps/desktop` | デスクトップ実行ファイルとOSごとの起動処理 |
| `crates/vtuber-app` | UI、設定、モデル取り込み、処理の統括 |
| `crates/vtuber-core` | 共通のデータ型とフレーム受け渡し |
| `crates/vtuber-camera` | カメラ入力 |
| `crates/vtuber-inference` | 推論とMediaPipeネイティブ境界 |
| `crates/vtuber-tracking` | 観測値の処理とアバター制御値への変換 |
| `crates/vtuber-avatar` | VRM描画、表情、骨格と関節の制御 |
| `crates/vtuber-ndi` | NDI送信のネイティブ境界 |
| `crates/vtuber-platform` | OS固有の機能 |
| `tools/xtask` | 開発用コマンドと配布パッケージ作成 |
| `manual/ja` | 日本語の利用マニュアル |

アプリケーションのRustコードは`#![forbid(unsafe_code)]`を使用し、必要なネイティブ処理はバインディング境界に隔離しています。開発時の制約は[AGENTS.md](AGENTS.md)を参照してください。

## 配布物の作成

配布ZIPの作成と確認は[配布手順](tools/RELEASING.md)を参照してください。Windowsのビルドと梱包、Macのランタイム同梱・署名・ZIP化の手順を記載しています。

## ライセンス

本プロジェクトのコードは[MIT License](LICENSE)で公開しています。

依存ライブラリ、フォント、推論モデル、参照データにはそれぞれのライセンスが適用されます。詳細は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。

NDI SDKとランタイムは独自のライセンス条件を持つ別コンポーネントです。NDIランタイムを含む配布物を作る場合は、使用したSDKのライセンス・通知文書を同梱し、[NDI SDKの利用・再配布条件](https://docs.ndi.video/all/developing-with-ndi/sdk/licensing)を確認してください。Windowsのパッケージ作成には`cargo xtask ndi package`を使用できます。

アバターのVRMファイル、テクスチャ、配信素材の利用条件は各権利者の指定に従ってください。アプリケーションのMITライセンスは、それらの素材の利用許諾を与えるものではありません。

NDI® is a registered trademark of Vizrt NDI AB. NDIに関する情報は[NDI公式サイト](https://ndi.video/)を参照してください。

## 免責事項

本ソフトウェアはMIT Licenseに従い、現状のまま提供されます。動作、推論・トラッキングの精度、性能、特定のハードウェアや配信環境との互換性について保証しません。

利用によって生じた損害、配信・録画の中断、データの損失などについて、作者および著作権者はMIT Licenseに定める範囲で責任を負いません。

配信・録画に使用する素材の権利と、顔や室内の映像が配信に含まれるかを利用者自身で確認してください。
