Upstream runtime: bevy_window 0.20.0 (MIT OR Apache-2.0).

The 0.19.0 registry copy was ported against bevyengine/bevy tag v0.20.0:
Cargo dependencies, lib.rs, raw_handle.rs and cursor/mod.rs are updated.
Other runtime source is unchanged upstream except for the patch below.
window.rs retains the old cursor-grab documentation (the two upstream
macOS/X11 fallback documentation edits have no code changes).
Cargo.toml.orig retains the upstream 0.20 manifest; Cargo.toml resolves its
sibling Bevy crates through crates.io and keeps local lint declarations.

Remove `linked_spawn` from `HasWindows`: a monitor disconnect must remove
`OnMonitor`, not despawn the windows on that monitor. macOS display sleep
otherwise removes the last application window and triggers `AppExit`.
The fix also applies to monitor disconnects on Windows and other platforms.
The existing regression test covers repeated disconnect/reconnect with two
windows, no automatic exit, and normal explicit window closing afterward.
Bevy 0.20.0 still has `linked_spawn`; this patch remains necessary.
Remove this patch when an upstream release preserves windows on disconnect.
