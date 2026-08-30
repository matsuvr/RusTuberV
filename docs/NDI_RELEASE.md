# NDI release and OBS interoperability

Status as of 2026-08-20: the research Windows x64 ZIP path stages the
Standard SDK x64 runtime DLL application-locally after reviewing the exact
SDK License Agreement shipped with the installed NDI 6 SDK. Clean-machine
QA, DistroAV GUI smoke, and the original installer-archive hash remain
optional extras, not close gates for Issue #49's current research ZIP
acceptance.

NDI® is a registered trademark of Vizrt NDI AB. The application uses NDI only
to identify compatibility; it is not an NDI product and this repository does
not claim sponsorship. The official link is present in the Live UI and in
THIRD_PARTY_NOTICES.md.

## Fixed release boundary

The first release target is Windows x86_64. The normal workspace build remains
SDK-free. Only the explicit ndi-output feature requires the locally installed
NDI SDK headers and bindgen environment:

~~~~text
cargo build -p vtuber-desktop --release --features ndi-output
~~~~

For an NDI-enabled local build, `apps/desktop/build.rs` reads the x64 import
library and stages the matching Standard SDK runtime beside
`vtuber-desktop.exe` in `target/debug` or `target/release`. It does not download
or install anything. Set `NDI_RUNTIME_DLL` to an explicitly selected SDK DLL
when the runtime is not at the SDK's `Bin\x64` directory; otherwise the build
fails before producing an unlaunchable NDI artifact. Both the current
`Processing.NDI.Lib.x64.dll` name and the legacy
`Processing.NDI.Lib_x64.dll` name are resolved from the import library.

The runtime is not committed to Git and must not be copied to System32, PATH,
or an NDI Tools directory. The package contains the application, the
explicitly supplied Standard SDK x64 runtime DLL, the exact license/notice
file supplied by that SDK package, the project notices, the approved MediaPipe
task bundle with its manifest and license, and a generated hash manifest.

## Exact SDK license decision (installed NDI 6 SDK)

Reviewed on 2026-08-20 from the installed package, not from an issue summary.

| item | value |
|---|---|
| SDK tree | `C:\Program Files\NDI\NDI 6 SDK` |
| License Agreement | `NDI SDK License Agreement.pdf` (163805 bytes) SHA-256 `11AE13AE038D5AF06D2FD5F9E55CBBDB97E75D2526656A0DBAA4DAFCB7BF48E6` |
| Redistributable runtime | `Bin\x64\Processing.NDI.Lib.x64.dll` SHA-256 `2B6602075868BA4401F82F417D72424805D69B11CA86078023D0D489FF45DD84` |
| Required 3rd-party notices | `Bin\x64\Processing.NDI.Lib.Licenses.txt` (“This file should be included with all distribution of the binary files”) |
| Original installer archive SHA-256 | not available on this machine (`--sdk-package-unavailable`) |

Agreement §2.a licenses distribution of SDK object code solely as used by the
Product, in accordance with the SDK Documentation. The Software Distribution
page states that `NDI_SDK_DIR\BIN\*.*` may be distributed inside the
application when the product meets the License section (ndi.video link,
trademark wording, no NDI Tools). The Licensing page additionally says to keep
NDI DLLs in the application folder and not on the system path.

Bundled in the research ZIP:

- `Processing.NDI.Lib.x64.dll`
- `NDI_SDK_LICENSE_AGREEMENT.pdf`
- `Processing.NDI.Lib.Licenses.txt`

Not bundled (not permitted, not needed, or not a Product binary):

- NDI Tools
- `Redist\NDI 6 Runtime.exe` (optional redistributable installer; users are
  pointed to the official link only if the application-local DLL fails)
- Advanced SDK / HX / audio codecs / headers / import libraries

This is not legal advice. If a later SDK agreement withdraws BIN redistribution,
stop bundling the DLL and keep only the official Runtime install instructions.

## Exact SDK license gate

The license is a release gate, not an assumption made by the packaging tool.
The exact license agreement shipped with the SDK package used for the build
must be checked again before each release. In particular, the agreement and
SDK documentation determine which object-code files may be distributed and
which end-user restrictions, NDI notices, trademark wording, export
conditions, and SDK freshness requirements apply.

The current official developer page identifies the NDI SDK as version 6.3.2
on this date. The local package command still requires the actual package
version and package SHA-256 from the build machine; it never infers these from
the web page.

References:

- [NDI for Developers](https://ndi.video/for-developers/)
- [NDI SDK license agreement](https://downloads.ndi.tv/SDK/NDI_SDK/NDI%20SDK%20License%20Agreement.pdf)
- [NDI SDK dynamic loading and application-local runtime guidance](https://docs.ndi.video/all/developing-with-ndi/sdk/dynamic-loading-of-ndi-libraries)
- [NDI SDK documentation](https://docs.ndi.video/all/developing-with-ndi/sdk)

If the exact agreement does not permit this application's intended
application-local distribution, do not bundle the DLL. Record the exact
blocker against Issue #45 and stop the release path.

## Reproducible package staging

Obtain the exact Standard SDK x64 runtime DLL and license agreement through the
approved SDK package channel. Do not use NDI Tools as a substitute. Record the
SDK package archive SHA-256, then run:

~~~~powershell
cargo run -p xtask -- ndi package --output target/ndi-package --runtime-dll 'C:\Program Files\NDI\NDI 6 SDK\Bin\x64\Processing.NDI.Lib.x64.dll' --sdk-license 'C:\Program Files\NDI\NDI 6 SDK\NDI SDK License Agreement.pdf' --sdk-version '6.3.2' --sdk-package-unavailable --zip target/RusTuberV-ndi-windows-x64.zip --force
cargo run -p xtask -- ndi verify-package target/ndi-package
~~~~

The staging command accepts only the runtime name imported by the supplied
executable (`Processing.NDI.Lib.x64.dll` or the legacy
`Processing.NDI.Lib_x64.dll`), requires the project attribution/link, records
runtime/license/face-task hashes, and rejects extra files outside the
allow-listed model resource directory. It does not download or delete the SDK,
and it does not make a legal determination.

Expected top-level package:

~~~~text
vtuber-desktop.exe
Processing.NDI.Lib.x64.dll
# or Processing.NDI.Lib_x64.dll when that is the executable's import name
NDI_SDK_LICENSE_AGREEMENT.pdf
Processing.NDI.Lib.Licenses.txt
README_NDI.md
THIRD_PARTY_NOTICES.md
NDI_RUNTIME_MANIFEST.txt
assets/
  models/
    face_landmarker.task
    manifest.toml
    LICENSE.mediapipe.txt
~~~~

The generated manifest must say application_local=true,
system_path_install=false, and must explicitly say that NDI Tools,
Advanced/HX, and audio components are absent.

## OBS and DistroAV receiver procedure

The receiver is not redistributed by this repository. Install OBS and
DistroAV separately on the receiver machine, then add an NDI Source and
select RusTuberV. As of this report, the DistroAV README states OBS
31.1.1 or newer and NDI Runtime 6.3 or newer as requirements; re-check the
current receiver release before a release report is signed.

- [DistroAV repository and installation requirements](https://github.com/DistroAV/DistroAV)
- [DistroAV NDI source mapping](https://github.com/DistroAV/DistroAV/blob/master/src/ndi-source.cpp)

No firewall rule is installed by this application. If discovery fails, check
that the two hosts are on the intended LAN and that local firewall policy
permits the receiver workflow.

## GPU offscreen pixel validator

The Issue #46 GPU contract is exercised locally with:

~~~~powershell
cargo run -p xtask -- ndi validate-render --evidence target/ndi-gpu-render.txt
~~~~

This command renders synthetic empty, opaque, and translucent scenes through the
production offscreen camera/readback path. It exits 0 on PASS, 1 on FAIL, and 2
on NOT RUN when the local GPU/readback path is unavailable. Unit tests are not
a substitute for this validator.

Environment inventory (SDK, license file, OBS, DistroAV, clean-machine) is
probed without claiming release success:

~~~~powershell
cargo run -p xtask -- ndi probe-environment
~~~~

## Machine-readable roundtrip evidence

The receiver-side harness must write a UTF-8 key=value capture manifest after
it has discovered the source, captured frames, and observed sender shutdown.
The repository validates the normative assertions without pretending that a
Rust-only unit test received an NDI packet:

~~~~powershell
cargo run -p xtask -- ndi verify-roundtrip path\to\ndi-roundtrip.txt
~~~~

The manifest must contain source_name=RusTuberV, four_cc=BGRA,
width=1920, height=1080, fps=60, connection_count at least one, frame_count
at least two, distinct_frame_hashes at least two, positive counts for alpha
zero/opaque/partial pixels, transparent_rgb_zero=true, sender_stopped=true,
stop_source_absent=true, render_blocked=false, and queue_depth_max no greater
than one. This is the machine-runnable assertion boundary; an NDI
SDK/runtime receiver harness must supply the manifest on a validation
machine. No such receiver harness was available for this local run.

## Acceptance evidence

| Gate | Result on 2026-08-20 | Evidence |
|---|---|---|
| SDK-free default workspace build | PASS | cargo check --workspace |
| SDK-free unit/workspace tests | PASS | cargo test --workspace --no-fail-fast |
| GPU offscreen pixel validator | PASS | cargo run -p xtask -- ndi validate-render |
| package layout/hash validator | PASS | cargo xtask ndi verify-package target/ndi-package |
| exact SDK License Agreement | PASS | installed `NDI SDK License Agreement.pdf` SHA-256 `11AE13AE038D5AF06D2FD5F9E55CBBDB97E75D2526656A0DBAA4DAFCB7BF48E6`; BIN redistribution allowed by SDK Software Distribution + License §2.a |
| NDI runtime version | PASS | `NDI SDK WIN64 16:38:09 Apr 14 2026 6.3.2.0` |
| exact SDK package archive hash | NOT RUN | original installer/zip was not on this machine; packaged with `--sdk-package-unavailable` |
| NDI-enabled release build | PASS | cargo build -p vtuber-desktop --release --features ndi-output |
| research ZIP | PASS | cargo xtask ndi package … --zip target/RusTuberV-ndi-windows-x64.zip |
| ZIP extract + app launch | PASS | extracted under %TEMP%; `vtuber-desktop.exe` stayed alive 8s then was stopped |
| sender start + finder discovery + video receive | PASS | `cargo run -p vtuber-ndi --features ndi-sdk --bin ndi-smoke`; source `DESKTOP-4090 (RusTuberV)`, 128x128 BGRA 60/1, 3 changing frames, A=0 / A=255 / partial present |
| clean Windows machine without SDK/Tools/toolchain | NOT RUN | this host has NDI SDK, Runtime, and Tools; not required by Issue #49 research ZIP acceptance |
| OBS + DistroAV smoke | NOT RUN | OBS is present; DistroAV plugin is not. Procedure is in README and docs/NDI_USER.md |

The following are required before calling the release self-contained:

1. Capture multiple frames through an NDI receiver and assert configured
   dimensions/FPS, BGRA/RGBA alpha, transparent/opaque/semitransparent pixels,
   changing frame hashes, and clean sender stop.
2. Repeat with a stopped or intentionally slow receiver and record that the
   application render loop remains responsive and the latest-value mailbox
   stays bounded.
3. Run the package on a clean Windows x64 machine with no SDK, NDI Tools,
   system-wide NDI Runtime, Visual Studio, LLVM, bindgen, or PATH edits.
4. Run the OBS/DistroAV smoke and attach the machine, package, SDK, receiver,
   commands, and results to the release report.

These physical/network checks are intentionally not represented as automated
PASS results.
