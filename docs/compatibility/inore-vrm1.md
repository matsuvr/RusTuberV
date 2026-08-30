# Historical VRM 1.0 Model Compatibility Report (2026-08-06)

This report predates ADR-011 and records the VRM 1.0-only runtime revision at
that time. It is retained as historical evidence; current VRM 0.x/1.0 scope
and the Issue #31 gate are recorded in
`docs/compatibility/vrm-0x-1x-2026-08-14.md`.

## Identification

| Field | Value |
|---|---|
| Report date | 2026-08-06 |
| Model filename | inore-vrm1.vrm |
| SHA-256 | `b5a3d4126c4a30ef3bfbcfc764a24dc48511b558799d98d4c2ff1db0bdc7ab01` |
| Model name | 天詩いのれ（VRM1.0） |
| Author | 松 |
| Source／exporter | VRoid Studio または同等の VRM 1.0 エクスポーター |
| Exporter version | 不明（メタデータに未記載） |
| VRM specVersion | 1.0 |
| License URL | https://vrm.dev/licenses/1.0/ |
| bevy version | 0.19.0 |
| bevy_vrm1 revision | f9593fd78136fb9e0507bcae111e09291ec9b82a |

## Preflight

| Check | Result | Notes |
|---|---|---|
| GLB parse | OK | glTF 2.0, JSON chunk valid |
| `VRMC_vrm` exists | OK | - |
| VRM 1.0 accepted | OK | `specVersion == "1.0"` |
| external URI absent | OK | images / buffers共にGLB内包 |
| hips present | OK | node 1: J_Bip_C_Hips |
| head present | OK | node 18: J_Bip_C_Head |
| file size within limit | OK | 14.57 MiB < 256 MiB |

## Runtime capabilities

| Capability | Result | Notes |
|---|---|---|
| load without panic | OK | - |
| `Initialized` observed | OK | - |
| head entity | OK | `J_Bip_C_Head` |
| neck entity | OK | `J_Bip_C_Neck` |
| left eye | OK | `J_Adj_L_FaceEye` |
| right eye | OK | `J_Adj_R_FaceEye` |
| blinkLeft | OK | expression preset available |
| blinkRight | OK | expression preset available |
| blink | OK | expression preset available |
| aa／ih／ou／ee／oh | OK | all 5 vowels available |
| look expressions | OK | `lookLeft` / `lookRight` / `lookUp` / `lookDown` are **NOT** present |
| lookAt type | bone | `LookAt` component intentionally avoided in adapter |
| SpringBone | OK | 36 `SpringRoot` components detected |
| Node Constraint | N/A | not present in model |
| First Person metadata | present | annotations exist in `VRMC_vrm` |

## Rendering

| Check | Windows | macOS | Notes |
|---|---|---|---|
| base color／texture | OK | TBD | 36 embedded images loaded |
| MToon shade | OK | TBD | 18 MToon materials |
| outline | TBD | TBD | requires visual inspection |
| transparent material | present | TBD | MASK/BLEND materials present |
| emissive | TBD | TBD | requires visual inspection |
| duplicate material names | OK | N/A | no duplicates detected |
| SpringBone appearance | OK | TBD | 36 spring roots initialized |

## Tracking control

| Test | Windows | macOS | Notes |
|---|---|---|---|
| neutral | TBD | TBD | pending calibration implementation |
| yaw | TBD | TBD | pending head pose integration |
| pitch | TBD | TBD | pending head pose integration |
| roll | TBD | TBD | pending head pose integration |
| blink left／right | TBD | TBD | pending expression integration |
| mouth open | TBD | TBD | pending `aa` expression integration |
| gaze | TBD | TBD | `lookAt.type = bone`; expression lookAt is `todo!()` in bevy_vrm1 |
| face loss／return | TBD | TBD | pending tracking state machine |

## Errors／workarounds

- Error code: なし
- Stack trace: なし
- Reproduction command: `cargo run -p vtuber-desktop -- --model <absolute-path>/inore-vrm1.vrm`
- Minimal fixture: `tests/fixtures/vrm/inore-vrm1.vrm`
- Upstream issue／PR: なし
- App workaround:
  - `LookAt` componentは挿入せず、adapter内で眼球ボーンまたはlook Expressionへ直接適用する。
  - `lookLeft`/`lookRight`/`lookUp`/`lookDown` Expressionが無いため、gazeはbone制御でfallbackする。
- Fork patch required: No（現時点）

## Conclusion

- Supported: **Yes**
- Required settings: なし
- Known limitations:
  - `bevy_vrm1` pinned revision で `LookAtType::Expression` は `todo!()` のため使用不可。
  - 本モデルは `lookAt.type = bone` かつ look-direction Expression preset を持たないため、MVP gaze は eye bone 直接制御に依存する。
  - VRM 0.x support was outside the scope of this historical run. Current
    generation normalization is governed by ADR-011.
