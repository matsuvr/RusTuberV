param(
    [Parameter(Mandatory = $true)][string]$Csv,
    [Parameter(Mandatory = $true)][string]$ReplaySummary,
    [Parameter(Mandatory = $true)][string]$Metadata,
    [Parameter(Mandatory = $true)][string]$Output
)

$ErrorActionPreference = 'Stop'
$replay = Get-Content -LiteralPath $ReplaySummary -Raw | ConvertFrom-Json
$clock = (Get-Content -LiteralPath $Metadata -Raw | ConvertFrom-Json).presentmon_clock
if (!$clock) { throw 'Replay metadata has no PresentMon QPC clock mapping.' }
$trace = @(Import-Csv -LiteralPath $Csv)
if (!$trace -or !$trace[0].PSObject.Properties['TimeInQPC']) {
    throw 'Expected a non-empty PresentMon 2.6 CSV recorded with --qpc_time.'
}
$rows = @($trace | Where-Object {
    $seconds = ([double]$_.TimeInQPC - $clock.qpc) / $clock.qpc_frequency +
        $clock.unix_seconds - $replay.video_started_unix_seconds
    $seconds -ge $replay.warmup_seconds -and $seconds -lt $replay.video_duration_seconds
})
if (!$rows) { throw 'No presentation events overlap the measured video interval.' }

function Measure-Column([string]$Name) {
    $values = @($rows | Where-Object { $_.$Name -and $_.$Name -ne 'NA' } |
        ForEach-Object { [double]::Parse($_.$Name, [Globalization.CultureInfo]::InvariantCulture) } | Sort-Object)
    if (!$values) { return $null }
    return [ordered]@{
        samples = $values.Count
        mean = ($values | Measure-Object -Average).Average
        p50 = $values[[int][Math]::Round(($values.Count - 1) * 0.5)]
        p95 = $values[[int][Math]::Round(($values.Count - 1) * 0.95)]
        p99 = $values[[int][Math]::Round(($values.Count - 1) * 0.99)]
        max = $values[-1]
    }
}
$display = Measure-Column 'MsBetweenDisplayChange'
if (!$display) { throw 'PresentMon recorded no display timings; inspect its log.' }
[ordered]@{
    scope = 'OS presentation events during video [warmup, duration); not optical photon measurement'
    requested_interval_seconds = $replay.video_duration_seconds - $replay.warmup_seconds
    first_video_seconds = ([double]$rows[0].TimeInQPC - $clock.qpc) / $clock.qpc_frequency + $clock.unix_seconds - $replay.video_started_unix_seconds
    last_video_seconds = ([double]$rows[-1].TimeInQPC - $clock.qpc) / $clock.qpc_frequency + $clock.unix_seconds - $replay.video_started_unix_seconds
    display_time_seconds = $display.mean * $display.samples / 1000.0
    display_fps = 1000.0 / $display.mean
    display_frame_ms = $display
    present_interval_ms = Measure-Column 'MsBetweenPresents'
    gpu_busy_ms = Measure-Column 'MsGPUBusy'
    present_to_display_ms = Measure-Column 'MsUntilDisplayed'
    presented_frames = $rows.Count
    frames_without_display_time = @($rows | Where-Object { $_.MsUntilDisplayed -eq 'NA' }).Count
    display_over_25_ms = @($rows | Where-Object { $_.MsBetweenDisplayChange -ne 'NA' -and [double]$_.MsBetweenDisplayChange -gt 25 }).Count
    allows_tearing_frames = @($rows | Where-Object { $_.AllowsTearing -eq '1' }).Count
    present_modes = @($rows.PresentMode | Sort-Object -Unique)
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Output -Encoding UTF8
