Upstream: bevy_window 0.19.0 from crates.io (MIT OR Apache-2.0).

Remove `linked_spawn` from `HasWindows`: a monitor disconnect must remove
`OnMonitor`, not despawn the windows on that monitor. macOS display sleep
otherwise removes the last application window and triggers `AppExit`.
The fix also applies to monitor disconnects on Windows and other platforms.
The regression test covers repeated disconnect/reconnect with two windows,
no automatic exit, and normal explicit window closing afterward.
Remove this patch when an upstream release preserves windows on disconnect.
