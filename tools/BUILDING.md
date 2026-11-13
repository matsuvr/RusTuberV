# ソースからのビルド

RustとBevyを使ったRusTuberVのビルド手順です。アプリの利用方法は[README](../README.md)を参照してください。

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

通常のデスクトップビルドは`ndi-output`を有効にします。基本構成でビルドする場合は、後述の`--no-default-features`を指定します。

## ソースの取得

```sh
git clone https://github.com/matsuvr/RusTuberV.git
cd RusTuberV
```

依存関係は`Cargo.lock`で固定しています。ビルドには`--locked`を指定してください。

## 推論モデルの準備

ビルド前に、以下の3ファイルを`assets/models/`に配置します。モデル本体はビルド時に実行ファイルへ埋め込まれます。

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

取得するモデルのバージョンは、[モデルマニフェスト](../assets/models/manifest.toml)のSHA-256と対応させてください。アプリはモデル読み込み時にハッシュを照合します。

MediaPipeのネイティブライブラリは、実行ファイルの隣に配置できます。初回の推論開始時にネットワーク経由で公式PyPI wheelから取得し、キャッシュする方法にも対応しています。ライブラリを指定する場合は、`MEDIAPIPE_LIB`にMediaPipe 0.10.35の対応ライブラリへのパスを設定します。

## Windowsでのビルド

[LLVMビルド環境の準備とMSVCへの切り戻し](WINDOWS-LLVM.md)を参照してください。
Windowsでは`tools/cargo-windows.ps1`を通してMSYS2のRustを使用します。
通常の`cargo`はrustup環境を使用します。

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
.\tools\package-windows.ps1 -Version 0.2.3
```

`release/`に`win-RusTuberV-0.2.3-x64.zip`と同名の`.zip.sha256`ファイルが生成されます。パッケージにはライセンス文書、設定、利用マニュアルも含まれます。Macの配布ZIPは`mac-RusTuberV-<バージョン>-arm64.zip`形式です。

### 基本構成でのビルド

```powershell
.\tools\cargo-windows.ps1 build --release --locked --package vtuber-desktop --no-default-features -j 1
.\tools\cargo-windows.ps1 run --release --locked --package vtuber-desktop --no-default-features -j 1
```

この構成では、推論モデルとMediaPipeネイティブライブラリを用意します。

## macOSでのビルド

NDI出力を含む場合はmacOS用のStandard SDKをインストールし、SDKの配置先を指定します。

```sh
export NDI_SDK_DIR="/Library/NDI SDK for Apple"
cargo build --release --locked -p vtuber-desktop -j 1
./target/release/RusTuberV
```

Clang / libclangをビルド環境から参照できるようにしてください。`LIBCLANG_PATH`でライブラリのディレクトリを指定できます。

ローカル実行用の`.app`を作る場合は、同じ環境で次を実行します。

```sh
bash tools/build-macos.sh
open target/release/RusTuberV.app
```

[ビルドスクリプト](build-macos.sh)はアイコン、モデルマニフェスト、トラッキング設定を配置し、アドホック署名を行います。NDIライブラリはインストール済みSDKを参照します。配布向けのランタイム同梱、署名、ZIP作成は[配布手順](RELEASING.md)を参照してください。

基本構成では次を実行します。

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

アプリの「VRMを読み込む…」やファイルのドロップからも読み込めます。カメラの選択と配信設定は[利用マニュアル](../manual/ja/README.md)を参照してください。

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

アプリケーションのRustコードは`#![forbid(unsafe_code)]`を使用し、必要なネイティブ処理はバインディング境界に隔離しています。開発時の制約は[AGENTS.md](../AGENTS.md)を参照してください。

## 配布物の作成

配布ZIPの作成と確認は[配布手順](RELEASING.md)を参照してください。Windowsのビルドと梱包、Macのランタイム同梱・署名・ZIP化の手順を記載しています。

