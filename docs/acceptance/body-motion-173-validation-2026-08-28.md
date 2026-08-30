# Body Motion Final Validation — Issue #173 (2026-08-28)

Status: PASS (machine-runnable gates)
Scope: Body Motion final headless trace validation and control-order confirmation.

| item | value |
| --- | --- |
| OS | Windows 11 Pro (kernel 26200) |
| CPU / GPU | Intel i9-13900 / NVIDIA RTX 4090 (Vulkan) |
| Bevy | 0.19.0 |
| bevy_vrm1 | 0.9.1 (vendored, pinned revision) |
| commands | see "Commands" below |

## Gate → evidence mapping

| acceptance gate | machine-runnable evidence | result |
| --- | --- | --- |
| lateral head X → root X suppression + torso response | `body_motion_integration.rs::lateral_target_keeps_root_x_and_produces_torso_lean`, ADR-019 vendor tests | PASS |
| Y/Z → root/body compensation + head-target residual | `body_motion_integration.rs::depth_and_vertical_targets_move_the_root_per_compensation`, ADR-019 vendor tests | PASS |
| no accumulation / no drift | `body_motion_integration.rs::repeated_evaluation_of_the_same_inputs_does_not_accumulate`, `::literal_same_tick_reevaluation_is_bit_stable`, `::output_is_deterministic_across_identical_runs` | PASS |
| deterministic X/Y/Z + rotation trace at 30/60/120 fps | `body_motion_trace.rs::trace_is_deterministic_across_30_60_and_120_fps_equivalents`, `::rotation_and_position_trace_is_deterministic_across_fps_equivalents` | PASS |
| virtual hand hips-relative constraint + arm reach | `arm_ik.rs`, `arm_virtual_hand.rs`, `arm_motion_geometry.rs` | PASS |
| legacy static arm-drop source not default authority | `body_motion_trace.rs::virtual_hand_authority_drives_the_compositor_not_the_legacy_source` | PASS |
| elbow flip / forearm twist bound / shoulder trim isolation | `arm_ik.rs`, `arm_motion_geometry.rs`, `arm_profile.rs` | PASS |
| legacy fixed wrist bias / finger curl not double-applied in dynamic mode | arm-pipeline regression tests; trace compositor never writes hand orientation | PASS |
| idle procedural amplitude = zero; composition with tracking-loss motion | `idle_contract.rs`, `body_motion_trace.rs::idle_amplitude_in_the_trace_is_zero_by_policy`, ADR-020 | PASS |
| tracking loss / reacquire continuity | `body_motion_trace.rs::tracking_loss_reacquire_replaces_state_without_snaps_or_stale_entities`, managed `verify_control_episode` phases | PASS |
| animation base preservation / no accumulation | `body_motion_trace.rs` hold-frame assertions | PASS |
| avatar replacement generation cleanup | `body_motion_trace.rs::avatar_generation_cleanup_rejects_stale_frames_and_targets`, managed replacement assertions | PASS |
| same-frame order: body/root, idle, arm compositor, gaze, expressions, constraints, SpringBone | `schedule.rs::avatar_schedule_ordering_matches_design` + retired-writer absence assertion | PASS |
| writer ownership confirmed in authoritative docs | ADR-004, ADR-019, ADR-020 | PASS |
| representative VRM 0.x managed headless run | `cargo xtask -- vrm-managed-compat tests\fixtures\vrm\tsukuyomi-chan.vrm` | PASS |
| representative VRM 1.0 managed headless run | `cargo xtask -- vrm-managed-compat tests\fixtures\vrm\inore-vrm1.vrm` | PASS |
| workspace focused tests / fmt / check / clippy / deny / diff check | `cargo test --workspace`, `cargo fmt --all -- --check`, `cargo check --workspace --all-targets`, `cargo clippy --workspace --all-targets`, `cargo deny check`, `git diff --check` | PASS |

## Managed runner output summary

Both fixture generations print, in order:

```text
managed avatar reached Ready: root=… visibility=Inherited
arm pose verified: side=left/right … bend_sine=…
idle contract verified: hips=… amplitude=0m
control episode motion verified (45 frames)
control episode loss verified (40 frames)
control episode reacquire verified (20 frames)
replacement verified: old_root=… new_root=… generation AvatarGeneration(1) -> AvatarGeneration(2)
idle contract verified: hips=… amplitude=0m
control episode motion / loss / reacquire verified
```

`alicia-solid.vrm` and `seed-san.vrm` are unmaterialized LFS placeholders (HTML payloads) and are excluded; they are not valid glTF fixtures.

## Not verified / out of scope

- Physical webcam evidence, human visual A/B comparison, macOS run: **NOT VERIFIED** (optional evidence, not a close gate).
- Subjective visual quality and GNM/ARKit52 quality: out of scope.
- The idle contract intentionally omits always-on procedural breathing motion; any future procedural idle source requires a new explicit design decision (ADR-020).
