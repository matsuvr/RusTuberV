# Build and package the default-feature Windows x64 executable in one invocation.
param(
    [string]$Version = '0.1.0',
    [string]$NdiSdk = 'C:\Program Files\NDI\NDI 6 SDK',
    [Parameter(Mandatory = $true)][string]$VcRuntime
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$root = (Get-Location).Path
$package = Join-Path $root "release/RusTuberV-$Version-windows-x64"
$inputs = Join-Path $root 'release/inputs'
if (Test-Path -LiteralPath $package) { throw "Package already exists: $package" }

# Capture provenance for the build we actually package, never a pre-existing exe.
$sourceCommit = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot read source commit' }
$sourceStatus = git status --porcelain --untracked-files=normal
if ($LASTEXITCODE -ne 0) { throw 'Cannot read source status' }
$sourceDirty = [bool]$sourceStatus
$compiler = rustc -V
if ($LASTEXITCODE -ne 0) { throw 'Cannot read compiler version' }
$env:NDI_SDK_DIR = $NdiSdk
$env:NDI_RUNTIME_DLL = Join-Path $NdiSdk 'Bin/x64/Processing.NDI.Lib.x64.dll'
cargo build --release --locked -p vtuber-desktop -j 1 --target x86_64-pc-windows-msvc --target-dir "$root/target"
if ($LASTEXITCODE -ne 0) { throw 'Desktop release build failed; package was not created' }
$builtAt = [DateTime]::UtcNow.ToString('o')
$executable = Join-Path $root 'target/x86_64-pc-windows-msvc/release/RusTuberV.exe'
New-Item -ItemType Directory -Path "$package/assets/models", "$package/licenses", $inputs -Force | Out-Null

Copy-Item -LiteralPath $executable, 'tracking_profile.toml', 'LICENSE', 'THIRD_PARTY_NOTICES.md' -Destination $package
Copy-Item -LiteralPath 'assets/models/manifest.toml', 'assets/models/LICENSE.mediapipe.txt' -Destination "$package/assets/models"
Copy-Item -LiteralPath 'assets/fonts/LICENSE-LINESeed.txt', 'assets/fonts/LICENSE-NotoSansCJK.txt', 'vendor/mediapipe-rs/NOTICE' -Destination "$package/licenses"
Copy-Item -LiteralPath 'manual' -Destination $package -Recurse
Copy-Item -LiteralPath "$NdiSdk/Bin/x64/Processing.NDI.Lib.x64.dll", "$NdiSdk/Bin/x64/Processing.NDI.Lib.Licenses.txt" -Destination $package
Copy-Item -LiteralPath "$NdiSdk/NDI SDK License Agreement.pdf" -Destination "$package/NDI_SDK_LICENSE_AGREEMENT.pdf"
Copy-Item -LiteralPath $VcRuntime -Destination "$package/vcruntime140.dll"
Copy-Item -LiteralPath 'tools/windows-runtime-terms.txt' -Destination "$package/RUNTIME-TERMS.txt"

# Pin the official wheel rather than repackaging an unverified local cache.
$wheel = Join-Path $inputs 'mediapipe-0.10.35-py3-none-win_amd64.whl'
$wheelUrl = 'https://files.pythonhosted.org/packages/5b/f6/763477e9aeed98accc984ed6ee3f11a21a0c5fd1d1c6586b8d07067748ff/mediapipe-0.10.35-py3-none-win_amd64.whl'
$wheelHash = 'B08F001CF3C3CD0D88D9ED68F3368DC8A4913F568281A93117F083115AA672BA'
if (-not (Test-Path -LiteralPath $wheel)) { Invoke-WebRequest -Uri $wheelUrl -OutFile $wheel }
if ((Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash -ne $wheelHash) { throw 'MediaPipe wheel SHA-256 mismatch' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($wheel)
try {
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($archive.GetEntry('mediapipe/tasks/c/libmediapipe.dll'), "$package/libmediapipe.dll")
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($archive.GetEntry('mediapipe-0.10.35.dist-info/licenses/LICENSE'), "$package/licenses/MediaPipe-LICENSE.txt")
} finally { $archive.Dispose() }

$vcLicense = Join-Path $inputs 'Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx'
Invoke-WebRequest -Uri 'https://visualstudio.microsoft.com/wp-content/uploads/2025/10/Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx' -OutFile $vcLicense
$archive = [System.IO.Compression.ZipFile]::OpenRead($vcLicense)
try {
    $reader = [System.IO.StreamReader]::new($archive.GetEntry('word/document.xml').Open())
    try { [xml]$document = $reader.ReadToEnd() } finally { $reader.Dispose() }
    $namespaces = [System.Xml.XmlNamespaceManager]::new($document.NameTable)
    $namespaces.AddNamespace('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
    $text = $document.SelectNodes('//w:p', $namespaces) | ForEach-Object { $_.InnerText }
    $text | Set-Content -LiteralPath "$package/licenses/Visual-C-Runtime-LICENSE.txt" -Encoding UTF8
} finally { $archive.Dispose() }

@'
RusTuberV for Windows (x64)

日本語: ZIPを「すべて展開」して、RusTuberV.exeをダブルクリックします。
「VRMを読み込む…」でご自身のVRMを選び、「入力カメラ」を選択してください。
モデル・推論ライブラリ・NDIランタイムは同梱済みです。初回ダウンロードは不要です。
実行ファイルと同梱ファイルは同じフォルダーに置いたまま使用してください。
利用マニュアル: manual/ja/README.md
https://github.com/matsuvr/RusTuberV/tree/main/manual/ja

English: Extract the entire ZIP and double-click RusTuberV.exe.
Load your own VRM avatar and select your webcam. All inference models and
runtime libraries are included; no first-run download is required.
Keep the executable and bundled files together. Manual: manual/en/README.md
Chinese manual: manual/zh-CN/README.md; Korean manual: manual/ko/README.md

Requires Windows 10/11 x64, a compatible GPU/driver, and a webcam for tracking.
Windows N editions require the Microsoft Media Feature Pack for webcam input.
An avatar is not included. Looking Glass output requires Looking Glass Bridge.

Read RUNTIME-TERMS.txt before using the bundled proprietary runtimes.
Application source: MIT (LICENSE). Other components: THIRD_PARTY_NOTICES.md
and licenses/. NDI SDK terms: NDI_SDK_LICENSE_AGREEMENT.pdf.
NDI® is a registered trademark of Vizrt NDI AB. https://ndi.video
'@ | Set-Content -LiteralPath "$package/README.txt" -Encoding UTF8

$executableHash = (Get-FileHash -LiteralPath "$package/RusTuberV.exe" -Algorithm SHA256).Hash.ToLowerInvariant()
$ndiVersion = (Get-Item -LiteralPath "$package/Processing.NDI.Lib.x64.dll").VersionInfo.FileVersion
@"
version=$Version
source_commit=$sourceCommit
source_dirty=$sourceDirty
built_at_utc=$builtAt
executable_sha256=$executableHash
target=x86_64-pc-windows-msvc
features=ndi-output
compiler=$compiler
ndi_version=$ndiVersion
mediapipe_version=0.10.35
mediapipe_wheel_url=$wheelUrl
mediapipe_wheel_sha256=$wheelHash
"@ | Set-Content -LiteralPath "$package/BUILD-INFO.txt" -Encoding UTF8

Get-ChildItem -LiteralPath $package -File -Recurse -Exclude SHA256SUMS.txt | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($package.Length + 1).Replace('\', '/')
    "{0}  {1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $relative
} | Set-Content -LiteralPath "$package/SHA256SUMS.txt" -Encoding ASCII
$zip = "$package.zip"
Compress-Archive -LiteralPath $package -DestinationPath $zip
"{0}  {1}" -f (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant(), (Split-Path $zip -Leaf) |
    Set-Content -LiteralPath "$zip.sha256" -Encoding ASCII
Write-Output "Created $zip"
