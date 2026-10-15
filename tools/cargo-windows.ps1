# Run Cargo with the MSYS2 CLANG64 host and target, including build scripts.
# Environment changes are scoped to this invocation; MSVC remains available.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$MsysRoot = 'C:\msys64',
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$CargoArguments
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$bin = Join-Path $MsysRoot 'clang64/bin'
$environment = @{
    PATH = "$bin;$env:PATH"
    RUSTC = Join-Path $bin 'rustc.exe'
    RUSTDOC = Join-Path $bin 'rustdoc.exe'
    CC = Join-Path $bin 'clang.exe'
    CXX = Join-Path $bin 'clang++.exe'
    AR = Join-Path $bin 'llvm-ar.exe'
    LIBCLANG_PATH = $bin
    CMAKE_GENERATOR = 'Ninja'
    CARGO_BUILD_TARGET = 'x86_64-pc-windows-gnullvm'
    CARGO_TARGET_DIR = Join-Path $root 'target/llvm'
}
# grafton-ndi declares a static link even though the SDK ships a COFF import
# archive. LLVM accepts its contents, but Rust's GNU target searches for .a.
# Keep both names in a local SDK copy; never change the installed SDK.
$ndiSdk = if ($env:NDI_SDK_DIR) { $env:NDI_SDK_DIR } else { 'C:\Program Files\NDI\NDI 6 SDK' }
$ndiLibrary = Join-Path $ndiSdk 'Lib/x64/Processing.NDI.Lib.x64.lib'
if (Test-Path -LiteralPath $ndiLibrary) {
    $stagedSdk = Join-Path $root 'target/llvm/ndi-sdk'
    $stagedLib = Join-Path $stagedSdk 'Lib/x64'
    New-Item -ItemType Directory -Path $stagedLib -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $ndiSdk 'Include') -Destination $stagedSdk -Recurse -Force
    Copy-Item -LiteralPath $ndiLibrary -Destination $stagedLib -Force
    Copy-Item -LiteralPath $ndiLibrary -Destination (Join-Path $stagedLib 'libProcessing.NDI.Lib.x64.a') -Force
    $environment.NDI_SDK_DIR = $stagedSdk
    $environment.NDI_RUNTIME_DLL = if ($env:NDI_RUNTIME_DLL) { $env:NDI_RUNTIME_DLL } else {
        Join-Path $ndiSdk 'Bin/x64/Processing.NDI.Lib.x64.dll'
    }
}
$saved = @{}
foreach ($name in $environment.Keys) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location $root
try {
    foreach ($name in $environment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $environment[$name], 'Process')
    }
    & (Join-Path $bin 'cargo.exe') @CargoArguments
    if ($LASTEXITCODE -ne 0) { throw "CLANG64 Cargo failed with exit code $LASTEXITCODE" }
} finally {
    foreach ($name in $saved.Keys) {
        if ($null -eq $saved[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
        }
    }
    Pop-Location
}
