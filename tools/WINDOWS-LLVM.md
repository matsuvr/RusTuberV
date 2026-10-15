# WindowsのLLVMビルドと切り戻し

Windowsの標準ビルドではMSYS2 CLANG64のRust、Clang、LLD、MinGW-w64ヘッダーを使う。
アプリだけでなくCargoのビルドスクリプトとproc-macroも
`x86_64-pc-windows-gnullvm`でビルドする。
Visual Studio Build ToolsとWindows SDKをコンパイル・リンクに使わない。
macOSは従来の手順を使う。

## 準備

MSYS2を`C:\msys64`にインストールし、MSYS2のシェルで実行する。

```sh
pacman -Syu
```

更新でシェルが閉じた場合は開き直し、次を実行する。

```sh
pacman -Syu --needed mingw-w64-clang-x86_64-rust mingw-w64-clang-x86_64-clang mingw-w64-clang-x86_64-llvm mingw-w64-clang-x86_64-cmake mingw-w64-clang-x86_64-ninja
```

Rustは1.99以上が必要。推論モデルとNDI SDKは[README](../README.md)に従って準備する。
MSYS2のRustを直接呼ぶので、rustupのstableや`rust-toolchain.toml`の変更は不要。

## ビルド

通常のPowerShellから実行する。

```powershell
.\tools\cargo-windows.ps1 build --release --locked --package vtuber-desktop -j 1
```

NDIなしは`--no-default-features`を追加する。開発用ビルドは`--release`を省略する。
`check`、`test`、`run`などのCargoコマンドも同じスクリプトを通す。
PowerShellの共通パラメーターと衝突する`-p`などは、`--package`などの長いオプションを使う。
テストやアプリへの引数区切り`--`を渡す場合は、PowerShellに消費されないよう
`-CargoArguments @('test', '--package', 'vtuber-inference', '--lib', '--', '--nocapture')`の形式で指定する。
MSYS2の場所が異なる場合は`-MsysRoot D:\msys64`を指定する。
出力先は`target/llvm/x86_64-pc-windows-gnullvm/{debug,release}/`。

```powershell
.\tools\cargo-windows.ps1 run --release --locked --package vtuber-desktop -j 1
```

このコマンドではCLANG64のDLL検索パスも設定される。
NDI SDKのヘッダーとインポートライブラリは`target/llvm/ndi-sdk`へコピーし、
RustのGNU系ターゲットが探す`libProcessing.NDI.Lib.x64.a`という名前でも配置する。
中身はSDKの`.lib`と同一で、インストール済みSDKは変更しない。
NDI・MediaPipeの既成DLLは引き続き使用する。アプリのビルドツールをOSSへ
切り替えることと、これらの外部DLLやWindows自身のランタイムの依存は別である。

## MSVCへの切り戻し

スクリプトは終了時にPATH、コンパイラ指定などの環境変数を復元する。
システムのPATHやrustupの既定ツールチェーンは変更しない。
MSVCの成果物は`target/llvm`と別なので残る。

```powershell
$env:LIBCLANG_PATH = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\Llvm\x64\bin'
cargo +stable-x86_64-pc-windows-msvc build --locked -p vtuber-desktop -j 1
```

上のlibclangはこの環境で確認した既存の配置先。他の環境では既存LLVMの場所を指定する。
LLVM移行前のソースはコミット`11d9c57`に残っている。
ソース全体を移行前へ戻して確認する場合は、未コミットの変更を保存してから
`git switch --detach 11d9c57`を実行する。確認後は`git switch main`でLLVM標準のソースへ戻る。
MSYS2に追加したパッケージはMSVCのビルドに影響しないため、切り戻し時の削除は不要。
今回のセットアップで行ったMSYS2本体の更新も、そのまま残してよい。

## 2026-10-15のローカル確認

MSYS2 Rust 1.99.0 Rev2、Clang / LLD 23.1.3、NDI SDK 6.3.2.0を使用。

| 確認 | 結果 |
| --- | --- |
| デスクトップの既定機能（NDI有効） | dev / releaseともビルド成功 |
| camera / inference / ndi / platformの既存単体テスト | 198件成功、6件は既定でignore |
| MediaPipeのネイティブPose/Handワーカーテスト | ignore対象から1件を個別実行し成功。合成フレームを使用 |
| NDI合成フレーム送受信 | 128×128 BGRAを3フレーム受信、停止成功。MSVC版でも同じ結果 |
| PEリソース | ICON / GROUP_ICON / VERSIONINFO、合計11リソースを確認 |
| 配布ZIP | 作成、SHA-256照合、展開成功 |
| 展開したrelease実行ファイル | PATHをWindowsのシステムディレクトリだけにして起動。ウィンドウ生成後、閉じる要求で終了コード0 |
| 読み込みDLL | libunwind / NDIは展開フォルダー内。MSYS2配下からの読み込みなし |
| 切り戻し | MSVCでNDIスモークを再ビルド・実行してPASS。ラッパーの環境変数復元は成功時・失敗時とも確認 |

NDIの既存スモークでは`stop_source_absent=false`、`transparent_rgb_zero=false`も
出力された。MSVC版でも同じ値であり、これらを達成済みとは扱わない。
カメラ追従、VRM表示の目視、OBS受信、Looking Glass、CPU 3コアでの60fps/30Hz、
macOSの動作は上記の自動・ローカル検証では未確認。

その後、ユーザーから実動作で問題なさそうとの報告があり、LLVMを標準とすることと
`main`へのマージが承認された。個別の確認項目や性能測定値の報告はないため、
上記の未確認項目をすべて合格したという意味ではない。

検証用ZIPは`release/RusTuberV-0.1.1-llvm-validation-windows-x64.zip`。
SHA-256は`93567e7dc7251758b0fe64967e554124b43ff376578c347cdbc70fc2551aab45`。
これはコミット前の変更を含むローカル検証物で、`BUILD-INFO.txt`の`source_dirty=True`は意図した値。
ログは`target/toolchain-migration/`に保存している。

参照: [Rustのgnullvmターゲット](https://doc.rust-lang.org/rustc/platform-support/windows-gnullvm.html)、
[MSYS2のCLANG64 Rust](https://packages.msys2.org/packages/mingw-w64-clang-x86_64-rust)。
