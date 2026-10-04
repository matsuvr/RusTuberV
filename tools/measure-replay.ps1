param(
    [Parameter(Mandatory = $true)][string[]]$Video,
    [Parameter(Mandatory = $true)][string]$Model,
    [ValidateRange(1, 20)][int]$Runs = 3,
    [ValidateRange(1, 63)][int]$LogicalProcessors = 3,
    [long]$ProcessorAffinity = 0,
    [string]$Tag = 'replay',
    [string]$Ffmpeg,
    [string]$PresentMon,
    [switch]$Uncapped,
    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$presentation = if ($Uncapped) { 'uncapped' } else { 'vsync' }
if ($PresentMon) { $PresentMon = (Resolve-Path -LiteralPath $PresentMon).Path }
if (!$Ffmpeg) {
    $command = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($command) { $Ffmpeg = $command.Source }
    else {
        $Ffmpeg = Get-ChildItem -Path "$env:LOCALAPPDATA/Microsoft/WinGet/Packages/Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
    }
}
if (!$Ffmpeg) { throw 'FFmpeg is required only to prepare the video. Specify -Ffmpeg <ffmpeg.exe>.' }
if (!$NoBuild) {
    & cargo build --release -p xtask -j 8
    if ($LASTEXITCODE -ne 0) { throw "Release build failed: $LASTEXITCODE" }
}
foreach ($clip in $Video) {
    $videoPath = (Resolve-Path -LiteralPath $clip).Path
    $videoHash = (Get-FileHash -LiteralPath $videoPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $modelHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $cache = Join-Path $repo "data/performance/input-$videoHash-rgb640x360-30"
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $rgb = Join-Path $cache 'frames.rgb'
    if (!(Test-Path -LiteralPath $rgb)) {
        $pending = Join-Path $cache 'frames.pending'
        & $Ffmpeg -hide_banner -loglevel error -y -i $videoPath -an -vf 'fps=30,scale=640:360' -pix_fmt rgb24 -f rawvideo $pending
        if ($LASTEXITCODE -ne 0) { throw "FFmpeg preparation failed: $LASTEXITCODE" }
        Move-Item -LiteralPath $pending -Destination $rgb
    }
    $exe = Join-Path $repo 'target/release/xtask.exe'
    $exeHash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    $hostProcess = Get-Process -Id $PID
    $originalAffinity = $hostProcess.ProcessorAffinity
    $measurementAffinity = 0L
    $selectedProcessors = 0
    $allowedAffinity = $originalAffinity.ToInt64()
    if ($ProcessorAffinity -ne 0) {
        if ($ProcessorAffinity -lt 0 -or ($ProcessorAffinity -band $allowedAffinity) -ne $ProcessorAffinity) {
            throw 'ProcessorAffinity must be a positive subset of the current process affinity.'
        }
        $allowedAffinity = $ProcessorAffinity
    }
    for ($bit = 0; $bit -lt 63 -and $selectedProcessors -lt $LogicalProcessors; $bit++) {
        $bitMask = 1L -shl $bit
        if (($allowedAffinity -band $bitMask) -ne 0) {
            $measurementAffinity = $measurementAffinity -bor $bitMask
            $selectedProcessors++
        }
    }
    if ($selectedProcessors -ne $LogicalProcessors) { throw 'The requested logical CPUs are not available in the current affinity mask.' }
    if ($ProcessorAffinity -ne 0 -and $measurementAffinity -ne $ProcessorAffinity) {
        throw 'ProcessorAffinity must contain exactly LogicalProcessors bits.'
    }
    $runRoot = Join-Path $repo ('data/performance/{0}-{1}-{2}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), ($Tag -replace '[^a-zA-Z0-9_-]', '_'), [IO.Path]::GetFileNameWithoutExtension($videoPath))
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $metadata = [ordered]@{
        video = $videoPath; video_sha256 = $videoHash; model = $modelPath; model_sha256 = $modelHash
        executable_sha256 = $exeHash; git_head = (& git rev-parse HEAD); git_status = @(& git status --short)
        input = 'RGB8 640x360 30fps'; runs = $Runs; adapter_request = $env:WGPU_ADAPTER_NAME
        backend_request = $env:WGPU_BACKEND
        logical_processors = $LogicalProcessors; processor_affinity = $measurementAffinity
        presentation = $presentation
        presentmon_sha256 = if ($PresentMon) { (Get-FileHash -LiteralPath $PresentMon -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
        presentmon_clock = if ($PresentMon) { @{
            qpc = [Diagnostics.Stopwatch]::GetTimestamp()
            qpc_frequency = [Diagnostics.Stopwatch]::Frequency
            unix_seconds = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
        } } else { $null }
        tracking_profile_sha256 = if (Test-Path -LiteralPath 'tracking_profile.toml') { (Get-FileHash -LiteralPath 'tracking_profile.toml').Hash.ToLowerInvariant() } else { 'generated default on first run' }
        model_manifest_sha256 = (Get-FileHash -LiteralPath 'assets/models/manifest.toml').Hash.ToLowerInvariant()
        measurement = 'Real Face/Pose/Hand, tracking, anatomy solver and desktop rendering. Physical camera driver and NDI are excluded.'
    }
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot 'metadata.json') -Encoding UTF8
    & git diff HEAD | Set-Content -LiteralPath (Join-Path $runRoot 'working-tree.patch') -Encoding UTF8
    $results = @()
    for ($run = 1; $run -le $Runs; $run++) {
        $out = Join-Path $runRoot "run-$run"
        $log = Join-Path $runRoot "run-$run.log"
        Write-Host "Replay $run / $Runs -> $out"
        $presentProcess = $null
        $replayProcess = $null
        try {
            # Windows children inherit affinity; set it BEFORE Bevy/MediaPipe start.
            $hostProcess.ProcessorAffinity = [IntPtr]$measurementAffinity
            # This is the measured GUI, which must be visible for display events.
            # Only the external collector runs hidden.
            $replayProcess = Start-Process -FilePath $exe -WindowStyle Normal -PassThru `
                -ArgumentList @('tracking-replay', ('"{0}"' -f $rgb), ('"{0}"' -f $modelPath), `
                    ('"{0}"' -f $out), $LogicalProcessors, $presentation) `
                -RedirectStandardOutput $log -RedirectStandardError (Join-Path $runRoot "run-$run-error.log")
            $hostProcess.ProcessorAffinity = $originalAffinity
            if ($PresentMon) {
                # Keep the external ETW observer outside the app's CPU budget.
                $presentProcess = Start-Process -FilePath $PresentMon -WindowStyle Hidden -PassThru `
                    -ArgumentList @('--process_id', $replayProcess.Id, '--session_name', "RusTuberV-Replay-$PID", `
                        '--qpc_time', '--no_console_stats', `
                        '--output_file', ('"{0}"' -f (Join-Path $runRoot "run-$run-presentmon.csv"))) `
                    -RedirectStandardOutput (Join-Path $runRoot "run-$run-presentmon.log") `
                    -RedirectStandardError (Join-Path $runRoot "run-$run-presentmon-error.log")
            }
            $replayProcess.WaitForExit()
            $replayExit = $replayProcess.ExitCode
            if ($presentProcess) {
                # End only the trace session started above. Process-exit events
                # are not reliable without elevation, even when CSV capture works.
                & $PresentMon --session_name "RusTuberV-Replay-$PID" --terminate_existing_session *> (Join-Path $runRoot "run-$run-presentmon-stop.log")
                if (!$presentProcess.WaitForExit(5000)) { throw 'PresentMon did not finish after replay exit; inspect its log.' }
                if ($presentProcess.ExitCode -ne 0) { throw "PresentMon failed: $($presentProcess.ExitCode); inspect its log." }
            }
        } finally {
            $hostProcess.ProcessorAffinity = $originalAffinity
            if ($presentProcess -and !$presentProcess.HasExited) {
                # Killing the collector alone leaves its ETW session alive.
                & $PresentMon --session_name "RusTuberV-Replay-$PID" --terminate_existing_session *> (Join-Path $runRoot "run-$run-presentmon-stop.log")
                if (!$presentProcess.WaitForExit(5000)) { Stop-Process -Id $presentProcess.Id }
            }
            if ($replayProcess -and !$replayProcess.HasExited) { Stop-Process -Id $replayProcess.Id }
        }
        if ($replayExit -ne 0) { throw "Replay failed: see $log" }
        Copy-Item -LiteralPath 'tracking_profile.toml' -Destination (Join-Path $out 'tracking_profile.toml')
        $summary = Get-Content -LiteralPath (Join-Path $out 'summary.json') -Raw | ConvertFrom-Json
        $display = $null
        if ($PresentMon) {
            $displayPath = Join-Path $out 'presentation-summary.json'
            & "$PSScriptRoot/summarize-presentation.ps1" `
                -Csv (Join-Path $runRoot "run-$run-presentmon.csv") `
                -ReplaySummary (Join-Path $out 'summary.json') `
                -Metadata (Join-Path $runRoot 'metadata.json') -Output $displayPath
            $display = Get-Content -LiteralPath $displayPath -Raw | ConvertFrom-Json
        }
        $results += [pscustomobject]@{
            Run = $run; FPS = $summary.fps; FrameP95ms = $summary.frame_ms.p95
            FaceHz = $summary.face_hz; PoseHandHz = $summary.pose_hand_hz
            ArmFirstP95ms = $summary.arm_first_apply_ms.p95; ArmAgeP95ms = $summary.arm_displayed_age_ms.p95
            DisplayFPS = $display.display_fps; DisplayP95ms = $display.display_frame_ms.p95
            DisplayOver25ms = $display.display_over_25_ms
            DisplayCoverageSeconds = $display.display_time_seconds
        }
        $results | Export-Csv -LiteralPath (Join-Path $runRoot 'comparison.csv') -NoTypeInformation -Encoding UTF8
    }
    $results | Format-Table -AutoSize
    Write-Host "Results: $runRoot"
}
