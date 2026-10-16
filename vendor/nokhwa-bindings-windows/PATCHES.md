Upstream: nokhwa-bindings-windows 0.4.6 from crates.io, Apache-2.0.
Registry provenance: nokhwa commit fa5a19208ade2b22113bfaa48f20dfdd3758b751,
subdirectory nokhwa-bindings-windows. LICENSE is the Apache-2.0 license text
from the already vendored nokhwa package (the bindings archive omits it).

Local capture lifetime fixes:

- Replace global initialization/refcount flags with a thread-bound runtime
  owner per enumeration/device. Balance COM initialization on the same thread;
  reuse an existing apartment without uninitializing its owner's reference.
  Balance MFStartup(MF_VERSION) with MFShutdown after COM objects are released.
  The unused, unscoped initialize_mf/de_initialize_mf Windows entry points are
  removed; all actual public capture operations manage their own lifetime.
- Match symbolic identity and activate within one enumeration, avoiding a
  second enumeration by an index that can change during hot-plugging.
- Release enumerated activation references and their COM-allocated array and
  strings. Detach the selected activation's cached source, retain the source,
  and call Shutdown on successful sessions and partial construction failures.
- Receive samples into an empty out-parameter instead of leaking a
  preallocated sample on every frame. Unlock each copied frame buffer,
  including empty-buffer error paths.
- Return a read error on source-reader end/error flags instead of looping
  indefinitely on a stream that has ended. Synchronous ReadSample itself can
  still block in a driver; this patch does not introduce asynchronous capture.

Camera-free Windows tests cover nested and overlapping runtime owners,
existing MTA initialization, and repeated enumeration cleanup. Physical camera
opening, unplugging and driver-specific blocking require separate hardware
checks.

References:
- https://learn.microsoft.com/en-us/windows/win32/api/combaseapi/nf-combaseapi-coinitializeex
- https://learn.microsoft.com/en-us/windows/win32/api/mfapi/nf-mfapi-mfstartup
- https://learn.microsoft.com/en-us/windows/win32/medfound/audio-video-capture-in-media-foundation

Remove the local patch when upstream provides equivalent ownership fixes.
