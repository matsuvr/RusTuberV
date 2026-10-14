# 2026-10-14 配布案内・起動通知修正の確認

ソースの基点はmain / v0.1.0の
`855b6dd09389d82ce7da316ac350b63dbedaa2c4`。
修正後のコード確認は、この基点に今回の未コミット変更を加えたWindowsローカルビルドで実施。
公開済みv0.1.0のZIP確認とは区別する。新しいリリースの公開・既存アセットの置換は行っていない。

## 修正後のコード

環境: Windows 11 Pro 10.0.26300、rustc 1.99.0 (b940084d7 2026-09-28)。

| 確認 | 結果 |
| --- | --- |
| `cargo test --locked -p vtuber-desktop -p vtuber-platform -j 1` | desktop 4件、platform 1件成功 |
| `cargo test --locked -p vtuber-app --lib -j 1 licenses::` | 4件成功。nokhwaを含む同梱一覧・ライセンス本文を確認 |
| `cargo test --locked -p vtuber-app --lib -j 1 settings::` | 22件成功。不正な設定・非対応スキーマで元ファイルを保持するテストを含む |
| `cargo test --locked -p vtuber-app --lib -j 1 tracking_file::` | 3件成功。不正なTOML・非対応スキーマで元ファイルのバイト列を保持 |
| `cargo build --locked -p vtuber-desktop -j 1` | 成功。既定機能、開発プロファイル |
| `cargo clippy --locked -p vtuber-desktop -p vtuber-platform -p vtuber-app --all-targets -j 1` | 成功。変更外の既存 `chunks_exact` スタイル警告4件（inference 1件、avatar 2件、appのテスト1件） |
| `cargo fmt --all -- --check` / `git diff --check` | 成功 |
| Windows梱包スクリプトのPowerShell構文解析 | 成功 |
| build-macos.shとRELEASING.md内のBashコマンドの `bash -n` | 成功。Macでの実行確認ではない |

Windowsの画面確認は、修正したexeとNDI DLL、モデルマニフェストを
`release/validation/startup-error 日本語/`へコピーして実施した。
exeのSHA-256:
`a27542b830d8457171d9ee1cb7499fd6c3c301180f7a6988f19f83ad1d91d84e`。

- テスト用tracking_profile.tomlを不正なTOMLにすると、OSダイアログに
  パース失敗の原因・行列番号・日本語と空白を含むファイルパスが表示された。
- 同ファイルのschema_versionを999にすると、非対応バージョンとパスが表示された。
- どちらもOKを押すと終了コード1で終了し、ファイルのSHA-256は起動前後で一致した。
- 利用者のsettings.tomlは変更していない。settings.toml自体を壊す画面試験は行わず、
  上記の設定テストとstartup errorのパス・原因を保持するテストで確認した。

## 公開済みv0.1.0のZIP

[公開リリース](https://github.com/matsuvr/RusTuberV/releases/tag/v0.1.0)から
ZIPと対応する.zip.sha256を取得した。両方ともハッシュファイルおよび
GitHub Release APIのdigestと一致した。ZIP内のSHA256SUMS.txtも照合した。

| ZIP | SHA-256 | 内部ファイル照合 |
| --- | --- | --- |
| RusTuberV-0.1.0-windows-x64.zip | `84b2eba083e9db22864179d04e5f878ce444c5cae2ec73d07a1dcaf7adbec0ae` | 91件一致 |
| RusTuberV-0.1.0-macos-arm64.zip | `4dc85ad30ef4beec6abc9a5b4cd3a0ab856e9057b877d2933f2a1996543b667b` | 92件一致 |

Windows ZIPは別フォルダーへ展開し、そのexeから「VRM・カメラ選択」画面が
表示されることと、閉じるボタンによる通常終了（終了コード0）を確認した。
今回の起動通知修正はこの公開済みexeには含まれていない。

Mac ZIPのInfo.plistはLSMinimumSystemVersion=14.0だった。
Contents/Frameworks/libndi.dylib、Contents/MacOS/libmediapipe.dylib、
ライセンス・マニュアル・BUILD-INFO.txtの同梱を確認した。
tools/macos-runtime-terms.txtはこのZIPのRUNTIME-TERMS.txtをそのまま保存したもの。
配布手順は[RELEASING.md](RELEASING.md)に記載した。

## 未確認

- 修正後のWindows梱包処理の全実行と新しいRelease ZIPの作成。
- Macでの今回のビルド・梱包手順の再実行、起動・署名検証・Gatekeeper操作。
- DLL/dylibの依存全件と実行時のロード元の照合。
- 公開ZIPでのVRM読込、実カメラ追従、OBSへのNDI受信を通した一巡確認。
- Looking Glass実機、他OS/GPU/VRMでの確認。

過去のリリース説明にある実行結果を今回再確認したものとは扱わない。
Issue [#183](https://github.com/matsuvr/RusTuberV/issues/183)と
[#184](https://github.com/matsuvr/RusTuberV/issues/184)の本文は実装済み部分と
残る実機確認へ整理した。新しい指追従の受入結果・性能比較結果は追加していない。
