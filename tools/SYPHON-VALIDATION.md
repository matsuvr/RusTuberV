# Syphon output development checks

The macOS backend reuses the app's existing OS-local sender intent, bounded
latest-frame mailbox, avatar-readiness and sleep lifecycle. Unsafe code is
isolated in vtuber-syphon-native. Windows continues to use vtuber-spout.

This is CPU readback plus a Metal texture upload and Syphon GPU copy, not
zero-copy Bevy texture sharing. A worker waits for one command buffer to finish
before reusing the texture. At most one pending frame is retained; UI does not
join the native worker. NDI shares the input Arc and performs its own alpha
conversion. Syphon receives the original premultiplied BGRA pixels.

## Required native checks

On Apple Silicon/macOS 14+, with full Xcode:

1. `cargo test --locked -p vtuber-syphon --lib`.
2. `bash tools/build-macos.sh`; verify nested code signatures.
3. Open app, load VRM, start Syphon; receive RusTuberV in an independent
   Syphon client (e.g. OBS Syphon source). Verify top/bottom orientation, red/blue
   channels, transparent edges and resolution against the preview.
4. Start/stop repeatedly; stop while loading a replacement avatar, unload,
   sleep/wake. Ensure old frames/senders disappear and explicit stop persists.
5. NDI alone, Syphon alone, both: compare output pixels, render fps, readback
   cadence, end-to-end latency, worker CPU and resident memory under same scene.
6. Remove framework from a disposable app copy: app must launch and Start must
   show a recoverable error. Restore and retry. Never modify the installed app.

The sender's Live state records successful GPU submission/completion, not
confirmation that a receiver displayed the frame. Actual macOS GPU/client
checks cannot be substituted by Linux unit tests or a native compile check.
