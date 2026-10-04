# ADR-003: Windows／macOS限定とcamera backend

Status: Accepted  
Date: 2026-08-04

## Context

完成可能な範囲へ実装と検証を限定しながら、WindowsとmacOSで同じtracking coreを利用する必要がある。両OSではcamera API、permission、package形式が異なるが、capture後のframe契約は共通化できる。

## Decision

サポート対象:

- Windows 11 x86_64、MSVC
- macOS 13以降、Apple SiliconをTier 1
- macOS Intelはcompileと可能な範囲のsmokeをTier 2

camera:

- `nokhwa = 0.10.11`を使用する。
- Windowsでは`input-msmf`を有効化する。
- macOSでは`input-avfoundation`を有効化する。
- 共通で`decoding-yuv`と`decoding-mjpeg`だけを必要に応じて有効化する。
- `input-native`で全backendを一括有効化せず、Cargo target dependencyを分ける。
- OpenCV backendを使わない。
- camera objectはcapture worker内で生成・所有し、OS handleやborrowed bufferを外へ出さない。
- worker境界でowned RGB8 frameへ正規化する。

macOS:

- camera利用前に`nokhwa_initialize`を呼び、callback完了までopenしない。
- app bundleへ`NSCameraUsageDescription`を含める。
- permission状態は`NotChecked / Requesting / Granted / DeniedOrRestricted`へ正規化する。
- 正式なmanual camera testは`.app` bundleで行う。

## Format policy

候補formatへscoreを付け、30fpsを優先する。初期候補は1280×720／30fpsと640×480／30fpsで、実際に選択されたformatをUIとperformance reportへ記録する。camera indexだけを永続device IDにしない。

## Consequences

OS差は`vtuber-camera`とpackagingに閉じる。GitHub Actionsは利用せず、camera hardwareのない開発者環境ではmock、format selection、compile testだけを行い、実機testは明示的なignored testまたはxtaskでローカル実行する。対象外platformの将来移植だけを目的とした抽象化は作らない。

## macOS実装（2026-10-04）

- macOSの既定backendをAVFoundationに接続。`nokhwa_initialize`の非同期callback完了まで列挙・openを行わず、拒否は既存の`CameraError::PermissionDenied`で伝える。許可待ちで描画threadを止めない。
- device IDは`AVCaptureDevice.uniqueID`。camera objectはcapture workerで生成・停止・破棄し、AVFoundationのYUYV出力をnokhwaのRGB decoderでowned RGB8へ変換する。
- nokhwa 0.10.11の`compatible_list_by_resolution`はfourccのfilterが逆、`Closest`は未対応解像度のfps検索に元の要求解像度を使う。そのため同じ依存の`nokhwa-bindings-macos` 0.2.4のsafe APIで実format一覧を取得し、既存の`select_format`で選んだformatを`Exact`で渡す。独自の代替入力は追加しない。
- `tools/build-macos.sh`は`NSCameraUsageDescription`を含むローカル`.app`を作る。NDIはインストール済みSDKのdylibをrpathから読む。配布用SDK同梱・署名・公証はこのローカルビルドの範囲外。

根拠: [nokhwa初期化API](https://docs.rs/nokhwa/0.10.11/nokhwa/fn.nokhwa_initialize.html)、[Appleのカメラ利用理由キー](https://developer.apple.com/documentation/bundleresources/information-property-list/nscamerausagedescription)、固定依存の`src/backends/capture/avfoundation.rs`と`nokhwa-core/src/types.rs`。
