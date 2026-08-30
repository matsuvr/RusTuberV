# Strict build-artifact cleanup for RusTuberV.
#
# Policy (see AGENTS.md "Build artifact disk policy"):
#   - Only the most recent build may occupy disk. Stale artifacts from older
#     builds are treated as accidents and must not survive a work session.
#   - Source lives in git; there is nothing to roll back via build caches.
#   - This deletes ALL target directories (workspace, teacher-capture,
#     vendored bevy_vrm1). The next build is a full rebuild by design.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\clean-target.ps1

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

$targets = @(
    (Join-Path $repoRoot "target"),
    (Join-Path $repoRoot "tools\teacher-capture\target"),
    (Join-Path $repoRoot "vendor\bevy_vrm1\target")
)

function Get-DirSize {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return [long]0 }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { [long]0 } else { [long]$sum }
}

$totalFreed = [long]0
$totalBefore = [long]0

foreach ($t in $targets) {
    if (Test-Path -LiteralPath $t) {
        $size = Get-DirSize $t
        $totalBefore += $size
        Remove-Item -LiteralPath $t -Recurse -Force
        Write-Host ("removed  {0}  ({1:N2} GiB)" -f $t, ($size / 1GB))
    }
    else {
        Write-Host ("clean    {0}" -f $t)
    }
}

$totalAfter = [long]0
foreach ($t in $targets) {
    if (Test-Path -LiteralPath $t) { $totalAfter += Get-DirSize $t }
}

$totalFreed = $totalBefore - $totalAfter
Write-Host ("freed total: {0:N2} GiB" -f ($totalFreed / 1GB))
