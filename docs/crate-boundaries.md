# Crate boundaries and API contracts

## Responsibilities

| Member | Responsibility |
| --- | --- |
| vtuber-core | Engine- and platform-independent data, validated output frames/signals, slots and worker handles. |
| vtuber-tracking | Engine-independent calibration, filtering, pose solving and tracking state. |
| vtuber-camera | Camera ownership and capture worker; Windows Media Foundation and explicit development Mock. |
| vtuber-inference | Rust preprocessing and inference workers, including the pinned native MediaPipe Tasks boundary. |
| vtuber-avatar | VRM/Bevy scene, material, expression and tracking adapters. |
| vtuber-app | Bevy orchestration, UI, settings, import and worker coordination. |
| vtuber-ndi | Optional sender boundary; SDK types stay inside the crate. |
| vtuber-desktop | Application entry point and platform resource setup. |
| xtask | Repository development and validation commands. |

The application also uses Bevy entities. Only core/tracking are engine-independent.
MediaPipe calls native code: the complete inference dependency stack is not pure Rust.
macOS currently uses an explicit development Mock; a production macOS camera backend
is unimplemented/deferred. These changes do not claim new hardware or platform support.

## Acceptance, completion and ownership

A successful capture/inference control request means the command was accepted, not
that the device/model transition completed. Inference controls never wait for queue
capacity: a full or disconnected bounded queue is a typed error. Worker-owned status
reports completion. NDI submission similarly acknowledges its one-frame mailbox, not
network delivery; replacing a pending frame differs from failing to publish a frame.

LatestSlot retains values even after reading. Its replacement count is not frame loss;
each inference reader computes skipped generations from its own cursor. Clearing a
session removes its retained value without manufacturing a generation or reopening a
closed slot. Closing a slot permanently rejects publication and wakes readers.

WorkerHandle::join consumes the handle and blocks, returning the worker result or
Panicked. WorkerHandle Drop only detaches: it does not stop or join. Controllers own
shutdown policy; their Drop stops/joins and can block. Explicit shutdown is preferred
when the caller must observe errors. Inference has an explicit preserving-input
shutdown for a shared capture slot; reaping a worker is separate from reading status.

Settings load defaults only when the file is absent. Unreadable, malformed or unsupported
files remain errors and are not silently overwritten. Saves replace the file through a
same-directory temporary file; failed writes/replacement keep the prior bytes. Mutating
settings setters update memory only after persistence succeeds. Save methods accepting
a store do not change the resource's startup snapshot. This is not a crash-durability or
multi-process read-modify-write synchronization guarantee.

Rich settings validate finite strength in 0..=1 at construction/deserialization. OFF
retains strength and selects baseline materials/front light. ON keeps that light and
adds the Rich paths and key/rim lights; zero disables the additional contribution.
The approved transparent-BLEND MToon correction remains in the baseline. Rendering,
tracking math, saved TOML keys/schema and four-language messages are unchanged.

## User-visible source changes

Control-delivery and worker-shutdown failures are now observable instead of being
reported as successful transitions. Reconnection continues its bounded retry plan and
can be cancelled. Development mock frames have monotonic timestamps and fps pacing.
Startup preserves OS path representations and reports explicit model/settings failures
with a failure exit status. CLI Help, NotRun and failures carry typed exit codes rather
than using message text as control flow. NDI errors retain native technical details
without guessing their category from words such as "library" or "dll".

## Publication metadata

All nine members were inspected. Existing version/edition/license/rust-version/authors/
repository inheritance is retained. vtuber-desktop and xtask are marked publish=false
because they are the application and repository tool. The repository does not establish
a crates.io publication plan for the seven library crates: their publication policy is
unresolved and their publish settings remain unchanged. No documentation URLs or
keywords have been invented. The inference description now acknowledges its native
MediaPipe boundary.
