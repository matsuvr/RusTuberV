# VRM 0.x/1.0 Model Compatibility Report

## Identification

| Field | Value |
|---|---|
| Report date | |
| Model filename | |
| SHA-256 | |
| Model name | |
| Generation | VRM 0.x / VRM 1.0 |
| Author | |
| Source／exporter | |
| Exporter version | |
| VRM specVersion | |
| License URL | |
| bevy version | 0.19.0 |
| bevy_vrm1 revision | f9593fd78136fb9e0507bcae111e09291ec9b82a |

## Preflight

| Check | Result | Notes |
|---|---|---|
| GLB parse | | |
| exactly one VRM root exists | | `VRM` or `VRMC_vrm` |
| generation accepted | | `VRM 0.x` or `VRM 1.0` |
| external URI absent | | |
| hips present | | |
| head present | | |
| file size within limit | | |

## Runtime capabilities

| Capability | Result | Notes |
|---|---|---|
| load without panic | | |
| `Initialized` observed | | |
| normalized generation | | |
| head entity | | |
| neck entity | | |
| left eye | | |
| right eye | | |
| blinkLeft | | |
| blinkRight | | |
| blink | | |
| aa／ih／ou／ee／oh | | |
| look expressions | | |
| lookAt type | | |
| SpringBone | | |
| Node Constraint | | |
| First Person metadata | | |

## Rendering

| Check | Windows | macOS | Notes |
|---|---|---|---|
| base color／texture | | | |
| MToon shade | | | |
| outline | | | |
| transparent material | | | |
| emissive | | | |
| duplicate material names | | | |
| SpringBone appearance | | | |

## Tracking control

| Test | Windows | macOS | Notes |
|---|---|---|---|
| neutral | | | |
| yaw | | | |
| pitch | | | |
| roll | | | |
| blink left／right | | | |
| mouth open | | | |
| gaze | | | |
| face loss／return | | | |

## Errors／workarounds

- Error code:
- Stack trace:
- Reproduction command:
- Minimal fixture:
- Upstream issue／PR:
- App workaround:
- Fork patch required: Yes / No

## Lifecycle regression

| Check | Result | Notes |
|---|---|---|
| load VRM 0.x | | |
| load VRM 1.0 | | |
| replace 0.x -> 1.0 | | |
| replace 1.0 -> 0.x | | |
| unload cleanup | | |
| automated lifecycle transition matrix | | repeated VRM 0.x/1.0 load/replace/unload; no stale runtime state |
| manual replacement cycles | | optional platform evidence; record `NOT VERIFIED` when not run |
| bounded SpringBone soak | | optional platform evidence; record `NOT VERIFIED` when not run |

## Conclusion

- Supported: Yes / Conditional / No
- Required settings:
- Known limitations:
