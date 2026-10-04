param(
    [Parameter(Mandatory = $true)][string[]]$Video,
    [Parameter(Mandatory = $true)][string]$Model,
    [ValidateRange(1, 20)][int]$Runs = 3,
    [ValidateRange(1, 63)][int]$LogicalProcessors = 3,
    [string]$Tag = 'replay',
    [string]$Ffmpeg,
    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$modelPath = (Resolve-Path -LiteralPath $Model).Path
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
    for ($bit = 0; $bit -lt 63 -and $selectedProcessors -lt $LogicalProcessors; $bit++) {
        $bitMask = 1L -shl $bit
        if (($originalAffinity.ToInt64() -band $bitMask) -ne 0) {
            $measurementAffinity = $measurementAffinity -bor $bitMask
            $selectedProcessors++
        }
    }
    if ($selectedProcessors -ne $LogicalProcessors) { throw 'The requested logical CPUs are not available in the current affinity mask.' }
    $runRoot = Join-Path $repo ('data/performance/{0}-{1}-{2}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), ($Tag -replace '[^a-zA-Z0-9_-]', '_'), [IO.Path]::GetFileNameWithoutExtension($videoPath))
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    $metadata = [ordered]@{
        video = $videoPath; video_sha256 = $videoHash; model = $modelPath; model_sha256 = $modelHash
        executable_sha256 = $exeHash; git_head = (& git rev-parse HEAD); git_status = @(& git status --short)
        input = 'RGB8 640x360 30fps'; runs = $Runs; adapter_request = $env:WGPU_ADAPTER_NAME
        logical_processors = $LogicalProcessors; processor_affinity = $measurementAffinity
        measurement = 'Real Face/Pose/Hand, tracking, anatomy solver and desktop rendering. Physical camera driver and NDI are excluded.'
    }
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot 'metadata.json') -Encoding UTF8
    & git diff HEAD | Set-Content -LiteralPath (Join-Path $runRoot 'working-tree.patch') -Encoding UTF8
    $results = @()
    for ($run = 1; $run -le $Runs; $run++) {
        $out = Join-Path $runRoot "run-$run"
        $log = Join-Path $runRoot "run-$run.log"
        Write-Host "Replay $run / $Runs -> $out"
        # Direct invocation keeps arguments separate and waits for a complete run.
        # Windows PowerShell treats native stderr diagnostics as error records.
        try {
            # Windows children inherit affinity; set it BEFORE Bevy/MediaPipe start.
            $hostProcess.ProcessorAffinity = [IntPtr]$measurementAffinity
            $ErrorActionPreference = 'Continue'
            & $exe tracking-replay $rgb $modelPath $out $LogicalProcessors *> $log
            $replayExit = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = 'Stop'
            $hostProcess.ProcessorAffinity = $originalAffinity
        }
        if ($replayExit -ne 0) { throw "Replay failed: see $log" }
        $summary = Get-Content -LiteralPath (Join-Path $out 'summary.json') -Raw | ConvertFrom-Json
        $results += [pscustomobject]@{
            Run = $run; FPS = $summary.fps; FrameP95ms = $summary.frame_ms.p95
            FaceHz = $summary.face_hz; PoseHandHz = $summary.pose_hand_hz
            ArmFirstP95ms = $summary.arm_first_apply_ms.p95; ArmAgeP95ms = $summary.arm_displayed_age_ms.p95
        }
        $results | Export-Csv -LiteralPath (Join-Path $runRoot 'comparison.csv') -NoTypeInformation -Encoding UTF8
    }
    $results | Format-Table -AutoSize
    Write-Host "Results: $runRoot"
}
