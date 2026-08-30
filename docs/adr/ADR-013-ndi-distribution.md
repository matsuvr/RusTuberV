# ADR-013: Guarded NDI runtime distribution

Status: Accepted for Issue #49 research ZIP packaging
Date: 2026-08-20

## Context

Issue #47 intentionally keeps the NDI SDK out of the normal workspace build
and Git history. Issue #49 needs a reproducible Windows x64 release path
without silently downloading or redistributing a proprietary SDK component.
The exact SDK package and agreement, not a summary in an issue, determine
whether application-local runtime distribution is permitted.

## Decision

- Keep the default workspace and source package SDK-free.
- Enable the sender only with the explicit ndi-output feature.
- For an NDI-enabled local build, inspect the x64 import library and stage
  only its matching Standard SDK runtime filename beside the executable. The
  supported names are `Processing.NDI.Lib.x64.dll` and the legacy
  `Processing.NDI.Lib_x64.dll`; the runtime is supplied by the build operator
  or the local SDK installation and is never downloaded.
- Require the exact SDK license agreement file, SDK version, and SDK package
  SHA-256 as package inputs.
- Place the runtime beside vtuber-desktop.exe; never install it into a
  system directory, edit PATH, or include NDI Tools.
- Generate NDI_RUNTIME_MANIFEST.txt with runtime/license hashes and explicit
  exclusions for Advanced/HX, audio, and system installation.
- Include the approved MediaPipe Face Landmarker task bundle at
  `assets/models/face_landmarker.task`, together with its source manifest and
  Apache-2.0 license, and record their hashes in NDI_RUNTIME_MANIFEST.txt.
- Require THIRD_PARTY_NOTICES.md to contain the NDI attribution and official
  link. The Live UI also links to the official NDI site.
- Verify an allow-listed package layout before it is treated as a staging
  result; the only nested resource path is `assets/models/`.
- Do not commit runtime DLLs, SDK headers, import libraries, or the exact
  agreement to the normal source tree.

cargo xtask ndi package performs staging and structural/hash checks. It is a
guardrail, not legal advice and not proof that a particular SDK agreement
permits distribution.

## Consequences

The release operator must obtain and review the exact SDK agreement and
record its package hash before a release. If application-local distribution is
not permitted, the release is blocked and Issue #45 must be updated; the
repository does not switch to an unlicensed workaround.

Numeric NDI receiver/alpha and clean-machine checks remain separate from
automated Rust tests. OBS and DistroAV are receiver-side dependencies and are
not redistributed here.

References:

- [NDI for Developers](https://ndi.video/for-developers/)
- [NDI SDK license agreement](https://downloads.ndi.tv/SDK/NDI_SDK/NDI%20SDK%20License%20Agreement.pdf)
- [NDI application-local runtime guidance](https://docs.ndi.video/all/developing-with-ndi/sdk/dynamic-loading-of-ndi-libraries)
- [DistroAV](https://github.com/DistroAV/DistroAV)
