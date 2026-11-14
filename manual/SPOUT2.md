# Spout2 output / Spout2出力 (Windows)

「出力」画面の「Spout2送信を開始」で、透過アバターのみを同じPC内の
Spout2対応アプリへ送信します。ソース名は `RusTuberV`。NDIとは独立して
開始・停止でき、同時使用も可能です。macOSは従来のNDIを使用してください。
SDK/DLLの追加インストールは不要です。OBSの場合は別途
[OBS Spout2 plugin](https://github.com/Off-World-Live/obs-spout2-plugin)を
導入し、Spout2 Captureでソースを選択してください。

In the Output page, select **Start Spout2**, then select `RusTuberV` in a
Spout2 receiver on the same Windows PC. The frame contains the avatar, not
camera images or UI. The current render profile is 1920×1080, nominally 60 Hz,
BGRA8 with straight alpha. Actual submission rate depends on rendering/readback;
this is not a guarantee of 60 delivered frames per second. No encoding/network
transport is involved. The source name may be suffixed by Spout if another
instance already owns it; avoid multiple RusTuberV senders when selecting it.

Stop, avatar unload, model replacement, sleep and shutdown release the sender.
Replacement/sleep recovery preserves a requested start, but output waits for a
ready avatar and native teardown. Initialization/send errors stop automatic
retries; inspect the message, then explicitly start again. Use a receiver on
the same graphics adapter if frames are not visible. Device-loss recovery may
require restarting output or the application. Do not interpret “Sending” as
proof that OBS received a frame: the SDK may skip a frame when its texture is busy.

## Implementation and cost

Uses the existing bounded, asynchronous GPU readback and a separate D3D11
`SendImage` worker. NDI and Spout share the pixel allocation; the Spout mailbox
retains only the newest pending frame. No UI-thread native send or join occurs.
This initial path **is not GPU zero-copy**: GPU→CPU readback plus CPU→D3D11
upload costs bandwidth (~8.3 MB per 1080p frame in each direction). Measure
with output off, NDI alone, Spout alone and both, using the same avatar/camera,
GPU, driver and resolution. Record rendering FPS, CPU usage, submission rate,
and camera-to-receiver latency separately. No performance gain is claimed.

Native source: Spout 2.007.017 via spout2-rs/sys 0.1.1 (BSD-2-Clause).
The native crate is patched for CLANG64; no change to the standard Windows
build toolchain. `tools/package-windows.ps1` includes its license notices.

## Validation on Windows (not a claim of completed hardware testing)

```
.\tools\cargo-windows.ps1 -CargoArguments @('test', '--locked', '-p', 'vtuber-spout', '--lib')
.\tools\cargo-windows.ps1 -CargoArguments @('test', '--locked', '-p', 'vtuber-app', '--lib', 'ndi_output')
.\tools\cargo-windows.ps1 -CargoArguments @('run', '--locked', '-p', 'vtuber-spout', '--bin', 'spout-smoke')
```

The smoke sender runs about 20 seconds, changes 640→800-pixel width halfway,
and provides a transparent upper-left quadrant and animated color. Check
alpha, red/blue order, orientation, resizing, receiver reconnect and source
removal after exit. Then check the actual app: NDI only, Spout only, both;
stop one while the other stays active; replace/unload the avatar; sleep/wake;
repeated Start/Stop; exit with a receiver attached. Hardware tests need a real
Windows graphics driver and receiver; cross-compilation cannot replace them.
