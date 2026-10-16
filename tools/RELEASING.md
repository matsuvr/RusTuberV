# 配布の作成と確認

配布は各OSのローカル環境で作成する。開発環境・推論モデルの準備は
[README](../README.md)を参照。ビルド中からZIP作成終了まではソース、コンパイラ、
SDKを変更しない。公開時にはコミット済みのソースを使い、ZIPとSHA-256を一組で扱う。
GitHub ReleasesとBOOTHで使う配布ZIPは、プラットフォーム名を先頭に付ける。
Windowsは`win-RusTuberV-<version>-x64.zip`、Macは`mac-RusTuberV-<version>-arm64.zip`とし、
対応するハッシュファイルはそれぞれのZIP名に`.sha256`を付ける。公開時もこの名前を維持する。

## Windows x64

[MSYS2 CLANG64環境](WINDOWS-LLVM.md)を準備する。

```powershell
.\tools\package-windows.ps1 -Version 0.1.1
```

この処理は指定NDI SDKとMSYS2 CLANG64で x86_64-pc-windows-gnullvm 向けの既定機能をビルドし、
`target/llvm/x86_64-pc-windows-gnullvm/release`の実行ファイルを梱包する。
LLVMの`libunwind.dll`とそのライセンス、MinGW-w64 CRTの通知も同梱する。
MSYS2が標準パス以外にある場合は`-MsysRoot`を指定する。
BUILD-INFO.txtにはビルド時のコミット、未コミット変更の有無、rustc、
ビルド完了時刻と同梱実行ファイルのSHA-256を記録する。
同名の出力フォルダーがある場合は停止するため、既存配布物を上書きしない。

## Apple Silicon Mac

v0.1.0の公開ZIPを2026-10-14に取得し、GitHubの公開SHA-256
`4dc85ad30ef4beec6abc9a5b4cd3a0ab856e9057b877d2933f2a1996543b667b`
との一致を確認した。実物は最低OS 14.0、NDIはContents/Frameworks、
MediaPipeはContents/MacOSに同梱されていた。BUILD-INFO.txtには
tools/build-macos.shによるビルドとNDI参照先の相対化が記録されている。

以下はその構成を次の配布で作るための手順。過去のシェル操作の逐語的な記録ではなく、
この手順自体のMac上での再実行は未確認。
tools/build-macos.shだけで作る.appはインストール済みSDKを使うローカル実行用なので、
配布には以下の梱包処理も必要になる。

Apple Silicon Mac上で、リポジトリのルートから同じBashセッションで順に実行する。
NDI SDKはビルドに使ったものを指定し、そのSDKのライセンスPDFと
libndi_licenses.txtの実際の場所を設定する。

```bash
set -euo pipefail
test "$(uname -m)" = arm64
export NDI_SDK_DIR="/Library/NDI SDK for Apple"
export NDI_SDK_LICENSE="/path/to/NDI SDK License Agreement.pdf"
export NDI_RUNTIME_NOTICES="/path/to/libndi_licenses.txt"
test -f "$NDI_SDK_LICENSE"
test -f "$NDI_RUNTIME_NOTICES"
version=0.1.1
package="$PWD/release/mac-RusTuberV-$version-arm64"
test ! -e "$package"
test ! -e "$package.zip"
source_commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain --untracked-files=normal)"
compiler=$(rustc -V)
unset CARGO_BUILD_TARGET
bash tools/build-macos.sh
built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

app="$package/RusTuberV.app"
resources="$app/Contents/Resources"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Frameworks" \
    "$resources/assets/models" "$resources/licenses" release/inputs
# 新しい配布フォルダーへ必要なファイルだけをコピーする。
cp target/release/RusTuberV "$app/Contents/MacOS/"
cp apps/desktop/macos/Info.plist "$app/Contents/"
cp target/release/RusTuberV.app/Contents/Resources/rustuberv-app.icns "$resources/"
cp tracking_profile.toml LICENSE THIRD_PARTY_NOTICES.md "$resources/"
cp assets/models/manifest.toml assets/models/LICENSE.mediapipe.txt "$resources/assets/models/"
cp -R manual "$resources/"
cp assets/fonts/LICENSE-LINESeed.txt assets/fonts/LICENSE-NotoSansCJK.txt "$resources/licenses/"
cp vendor/mediapipe-rs/NOTICE "$resources/licenses/MediaPipe-rs-NOTICE.txt"
cp "$NDI_SDK_LICENSE" "$resources/licenses/NDI_SDK_LICENSE_AGREEMENT.pdf"
cp "$NDI_RUNTIME_NOTICES" "$resources/licenses/libndi_licenses.txt"
cp tools/macos-runtime-terms.txt "$resources/RUNTIME-TERMS.txt"
cp "$NDI_SDK_DIR/lib/macOS/libndi.dylib" "$app/Contents/Frameworks/libndi.dylib"
ndi_original_sha256=$(shasum -a 256 "$app/Contents/Frameworks/libndi.dylib" | awk '{print $1}')

wheel=release/inputs/mediapipe-0.10.35-py3-none-macosx_11_0_arm64.whl
wheel_url=https://files.pythonhosted.org/packages/aa/c2/439e948d2a9a542498aee5d8fa3fb91ac6f4478be5d502f0c78ca3fb2333/mediapipe-0.10.35-py3-none-macosx_11_0_arm64.whl
wheel_sha256=3b31376f34ca3665e34b834565996464cd66c9c91316e914fa7f149c891ce7ac
curl -fL "$wheel_url" -o "$wheel"
printf '%s  %s\n' "$wheel_sha256" "$wheel" | shasum -a 256 -c -
unzip -p "$wheel" mediapipe/tasks/c/libmediapipe.dylib > "$app/Contents/MacOS/libmediapipe.dylib"
unzip -p "$wheel" mediapipe-0.10.35.dist-info/licenses/LICENSE > "$resources/licenses/MediaPipe-LICENSE.txt"

binary="$app/Contents/MacOS/RusTuberV"
install_name_tool -change @rpath/libndi.dylib \
    @executable_path/../Frameworks/libndi.dylib "$binary"
install_name_tool -delete_rpath "$NDI_SDK_DIR/lib/macOS" "$binary"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $version" "$app/Contents/Info.plist"
test "$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$app/Contents/Info.plist")" = 14.0
otool -L "$binary" "$app/Contents/Frameworks/libndi.dylib" "$app/Contents/MacOS/libmediapipe.dylib"
```

otoolの出力で、NDIが上記のアプリ内パスを参照し、その他の依存に開発機だけの
絶対パスがないことを確認する。予期しない依存があれば解決してから進む。
モデル本体は実行ファイルに埋め込まれる。アプリ内のファイルを揃えてから
アドホック署名する。Developer ID署名・Apple公証を意味しない。

```bash
codesign --force --sign - "$app/Contents/Frameworks/libndi.dylib"
codesign --force --sign - "$app/Contents/MacOS/libmediapipe.dylib"
codesign --force --sign - "$app"
codesign --verify --deep --strict --verbose=2 "$app"
executable_sha256=$(shasum -a 256 "$binary" | awk '{print $1}')
cat > "$package/BUILD-INFO.txt" <<EOF
version=$version
source_commit=$source_commit
source_dirty=false
target=aarch64-apple-darwin
minimum_macos=14.0
features=ndi-output
compiler=$compiler
built_at_utc=$built_at
executable_sha256=$executable_sha256
ndi_original_sha256=$ndi_original_sha256
mediapipe_version=0.10.35
mediapipe_wheel_url=$wheel_url
mediapipe_wheel_sha256=$wheel_sha256
app_signature=ad-hoc
apple_notarization=none
EOF
cat > "$package/README.txt" <<'EOF'
RusTuberV for Apple Silicon Mac (macOS 14+), experimental build.
Models, MediaPipe and NDI runtimes are bundled. Keep the .app intact.
Move RusTuberV.app to Applications. This build has no Developer ID signature
or Apple notarization. See the release notes for first-launch instructions
and the actual tested environment:
https://github.com/matsuvr/RusTuberV/releases/latest
Manuals: RusTuberV.app/Contents/Resources/manual/
Read Contents/Resources/RUNTIME-TERMS.txt, LICENSE, THIRD_PARTY_NOTICES.md
and licenses/ before using the bundled runtimes.
NDI® is a registered trademark of Vizrt NDI AB. https://ndi.video/
EOF
(
    cd "$package"
    find . -type f ! -name SHA256SUMS.txt -exec shasum -a 256 {} \; |
        LC_ALL=C sort > SHA256SUMS.txt
)
ditto -c -k --keepParent "$package" "$package.zip"
(
    cd release
    shasum -a 256 "mac-RusTuberV-$version-arm64.zip" > "mac-RusTuberV-$version-arm64.zip.sha256"
)
```

## 配布ZIPに対応した短い確認記録

開発用実行ファイルではなく、配布ZIPを別フォルダーに展開したものを使う。
公開後に確認する場合は公開アセットを取得し直してSHA-256を照合する。
同じZIPについて以下を記録し、未実施の欄は未確認とする。長時間のカメラ待機は不要。

- リリースURL、ZIP名、ZIPのSHA-256、BUILD-INFO.txt、確認日。
- OSバージョン、CPU/GPU、カメラ、使用VRMの形式・モデル、OBS/DistroAVの版。
- 起動 → VRM読込 → カメラ追従 → OBSでNDI受信 → NDI停止 → 通常終了の結果。
- Windowsの起動エラー確認は別のテスト用設定で行う。壊れたTOML・非対応スキーマで
  原因とパスを表示し、終了後もファイル内容が保持されることを確認する。
- macOSの署名検証、実行時に読まれたdylibの場所、初回起動の警告操作の実施有無。
- 実施していないOS、VRM、カメラ、OBS、Looking Glassなどの範囲。

v0.1.0の過去の実行確認範囲は[公開リリース説明](https://github.com/matsuvr/RusTuberV/releases/tag/v0.1.0)
を参照。ZIPのハッシュ一致・内容確認や空フレーム推論を、実カメラ追従やOBS受信の成功とは扱わない。

2026-10-14のZIP照合・Windows起動・起動エラー通知の確認結果は
[検証記録](RELEASE-VALIDATION-2026-10-14.md)を参照。
