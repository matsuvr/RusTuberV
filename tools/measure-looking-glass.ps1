param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [Parameter(Mandatory = $true)][string]$PresentMon,
    [Parameter(Mandatory = $true)][string]$Output,
    [ValidateRange(5, 60)][int]$Seconds = 30
)

# Attach after the avatar and shader pipelines have warmed up. Keep the same
# visible windows, model, output dimensions and CPU affinity for both builds.
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $Output | Out-Null
$Output = (Resolve-Path -LiteralPath $Output).Path
$app = Get-Process -Id $ProcessId
$exe = $app.Path
$session = "RusTuberV-LookingGlass-$PID"
$csv = Join-Path $Output 'presentmon.csv'
$collector = Start-Process -FilePath $PresentMon -WindowStyle Hidden -PassThru -ArgumentList @(
    '--process_id', $ProcessId, '--session_name', $session,
    '--qpc_time', '--no_console_stats', '--timed', $Seconds, '--terminate_after_timed',
    '--output_file', ('"{0}"' -f $csv)
) -RedirectStandardOutput (Join-Path $Output 'presentmon.log') `
  -RedirectStandardError (Join-Path $Output 'presentmon-error.log')
$begin = [Diagnostics.Stopwatch]::GetTimestamp()
$cpu = $app.TotalProcessorTime.TotalSeconds
if (!$collector.WaitForExit(($Seconds + 10) * 1000)) {
    & $PresentMon --session_name $session --terminate_existing_session
    throw 'PresentMon capture did not finish.'
}
if ($collector.ExitCode -ne 0) { throw "PresentMon exit $($collector.ExitCode)" }
$app.Refresh()
$elapsed = ([Diagnostics.Stopwatch]::GetTimestamp() - $begin) / [Diagnostics.Stopwatch]::Frequency
$cpuCores = ($app.TotalProcessorTime.TotalSeconds - $cpu) / $elapsed
$trace = @(Import-Csv -LiteralPath $csv)
if (!$trace) { throw 'No presentation events captured.' }

function Distribution($Rows, [string]$Column) {
    $values = @($Rows | Where-Object { $_.$Column -and $_.$Column -ne 'NA' } |
        ForEach-Object { [double]::Parse($_.$Column, [Globalization.CultureInfo]::InvariantCulture) } | Sort-Object)
    if (!$values) { return $null }
    [ordered]@{
        samples = $values.Count
        mean = ($values | Measure-Object -Average).Average
        p50 = $values[[int][Math]::Round(($values.Count - 1) * 0.5)]
        p95 = $values[[int][Math]::Round(($values.Count - 1) * 0.95)]
        p99 = $values[[int][Math]::Round(($values.Count - 1) * 0.99)]
        max = $values[-1]
    }
}
# PresentMon can emit transient "Other" events with no swapchain (0x0).
# Keep them in the raw CSV, but do not report them as a third output window.
$unattributed = @($trace | Where-Object SwapChainAddress -eq '0x0').Count
$chains = @($trace | Where-Object SwapChainAddress -ne '0x0' | Group-Object SwapChainAddress | ForEach-Object {
    $display = Distribution $_.Group 'MsBetweenDisplayChange'
    $present = Distribution $_.Group 'MsBetweenPresents'
    [ordered]@{
        swapchain = $_.Name
        frames = $_.Count
        display_fps = if ($display.mean -gt 0) { 1000 / $display.mean } else { $null }
        present_fps = if ($present.mean -gt 0) { 1000 / $present.mean } else { $null }
        display_ms = $display
        present_ms = $present
        gpu_busy_ms = Distribution $_.Group 'MsGPUBusy'
        present_modes = @($_.Group.PresentMode | Sort-Object -Unique)
    }
})
$summary = [ordered]@{
    executable = $exe
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    processor_affinity = $app.ProcessorAffinity.ToInt64()
    elapsed_seconds = $elapsed
    mean_cpu_cores = $cpuCores
    working_set_gib = $app.WorkingSet64 / 1GB
    swapchains = $chains
    unattributed_present_events = $unattributed
    scope = 'OS presentation events, separate swapchains; GPU timings are ETW attribution, not isolated render-pass timestamps.'
}
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Output 'summary.json')
[ordered]@{
    output = $Output
    mean_cpu_cores = $cpuCores
    working_set_gib = $summary.working_set_gib
    swapchains = @($chains | ForEach-Object {
        [ordered]@{ swapchain = $_.swapchain; display_fps = $_.display_fps; present_fps = $_.present_fps; display_p95_ms = $_.display_ms.p95 }
    })
} | ConvertTo-Json -Depth 4
