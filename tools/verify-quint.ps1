# Model check every finite specification, then replay its traces against Rust.
[CmdletBinding()]
param(
    [switch]$ModelsOnly,
    [string]$MsysRoot = 'C:\msys64'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$specs = Join-Path $root 'verification'
$models = @(Get-ChildItem -LiteralPath $specs -Filter '*.qnt' -File | Sort-Object Name)
if (!$models) { throw "No Quint models in $specs" }
# A separate directory per run also isolates Apalache logs and old witnesses.
$output = Join-Path $root ('target/quint/' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $output -Force

$npx = (Get-Command npx -CommandType Application | Select-Object -First 1).Source
function Invoke-Quint {
    & $npx --yes '--package=@informalsystems/quint@0.34.0' -- quint @args
    if ($LASTEXITCODE -ne 0) { throw "Quint failed: $args" }
}

$previousTraces = $env:RUSTUBERV_QUINT_TRACES
Push-Location $output
try {
    foreach ($model in $models) {
        Write-Host "Checking $($model.Name)"
        $dir = New-Item -ItemType Directory -Path $model.BaseName
        Invoke-Quint verify $model.FullName --backend tlc --tlc-config (Join-Path $specs 'tlc.json') --invariant safety --verbosity 3
        Invoke-Quint test $model.FullName --backend typescript --seed 2026 --out-itf (Join-Path $dir.FullName '{test}.json')
        Invoke-Quint run $model.FullName --backend typescript --invariant safety --seed 2026 --max-samples 12 --n-traces 12 --max-steps 25 --verbosity 1 --out-itf (Join-Path $dir.FullName 'walk-{seq}.json')
    }
    Write-Host "Quint traces: $output"
    if (!$ModelsOnly) {
        $env:RUSTUBERV_QUINT_TRACES = $output
        Set-Location $root
        $packages = @('app', 'avatar', 'camera', 'inference', 'ndi', 'platform', 'spout', 'syphon', 'tracking')
        $cargoArgs = @('test', '--locked') + @($packages | ForEach-Object { '-p'; "vtuber-$_" }) +
            @('--lib', '-j', '1', 'quint_', '--', '--ignored')
        if ($env:OS -eq 'Windows_NT') {
            & (Join-Path $PSScriptRoot 'cargo-windows.ps1') -MsysRoot $MsysRoot -CargoArguments $cargoArgs
        } else {
            & cargo @cargoArgs
            if ($LASTEXITCODE -ne 0) { throw 'Rust conformance tests failed' }
        }
        foreach ($model in $models) {
            if (!(Test-Path -LiteralPath (Join-Path $output "$($model.BaseName)/replayed"))) {
                throw "No Rust conformance test completed for $($model.Name)"
            }
        }
    }
} finally {
    $env:RUSTUBERV_QUINT_TRACES = $previousTraces
    Pop-Location
}
