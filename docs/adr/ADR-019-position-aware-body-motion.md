# ADR-019: position-aware upper-body solveとroot/body motion writer

Status: Accepted (2026-08-26)
Date: 2026-08-26
Amended: 2026-08-28 (Issue #180: breathing writer retired, ADR-020)
Amended: 2026-08-30 (binding wiring fix + torso propagation profile tuning)
Related: Issue #167, Epic #162, ADR-004, Q2-06-001 vendored patch

## Context

`bevy_vrm1::BodyTracking`のdirect-pose経路（Q2-06-001）はhead/neck/upper-chest/chest/spineへのyaw/pitch/roll重み配分のみを行うrotation-only pathである。#163〜#165で確立したneutral-relative head translation（shaping → axis-selective split）を消費し、仮想head targetを実際の骨運動へ変換するposition-aware solveが必要になった。

制約:

- FinalIK/VRIKは移植しない。既存rest-relative additive rotation、animation-base preservation、bottom-up propagationを維持する。
- Q2-06-001により`bevy_vrm1::BodyTracking`のdirect-pose writerがtracked head/neck/upper-chest/chest/spine rotationの唯一writerである。
- #20 breathingは当時additive `hips.translation`を所有していたため、body/root compensationを同じhips translation channelへ書く2つ目のsystemは存在できなかった（この制約はIssue #180／ADR-020でbreathing writerがretireされたことで解消され、root translation channelとの分離は維持される）。
- camera Transform/Projection/FOVへ作用しない。VRM0/VRM1分岐をsolver層に追加しない。
- AGENTS.mdのvendor拡張ルール（target-model reproducer、regression test、spec引用、ADR）を満たすこと。

## Decision

vendored `bevy_vrm1`のbody trackingモジュールに、Q2-06-001と同一パターンのsource-derived extensionとして`BodyTrackingPositionInput` / `BodyTrackingPositionProfile`コンポーネントと`apply_direct_body_position` systemを追加する。これはADR-002で固定されたruntime境界内の拡張であり、replacement runtimeや2つ目のECS solverを作らない。

### 入力契約

`BodyTrackingPositionInput`はsemantic meter値のみを保持する: `head_offset`（#165のhead残差）、`body_offset`（#165のroot/body補正）、`weight`、`active`。Bevy world-spaceデータを持たない。唯一のinput writerは`vtuber-avatar::update_body_tracking_position_input`で、ActiveControlFrameを#164 shaping profile→#165 `VirtualBodyProfile`へ通し、mirror設定ではlateral軸のみ符号反転する。pose bridgeと同一のlifecycle/generation skip規則に従う。

### 変換境界

semantic frame（+x画像右、+y上、+zカメラ遠ざかる）からmodel/root空間への変換は`semantic_offset_to_model`一箇所のみで行い、immutable root rest rotation `(x, y, -z)`写像を共役させる。VRM0.x legacy Y=180 basis rootはこのcapture済みrest rotationで吸収され、generation別分岐は存在しない。

### 出力channelとownership

1. **avatar root translation**: `rest_translation + clamped(body_offset_model)`の絶対書き込み。rest値は初回評価時に一度だけcaptureされ、再評価でdelta蓄積が起きない。このchannelのruntime writerは本systemのみ（arm compositorはarm bonesを書き、hips bone translationのruntime writerはIssue #180／ADR-020以降存在しない、camera系は読み取り専用）。
2. **torso lean rotation**: spine/chest/upperChest/neck（利用可能な骨のみ、head除外）へworld-space premultiplyで等分配。角度は`atan2(head_offset, lever)`由来でprofile上限（seed 15°）にclampされる。head rotationのsole writerはQ2-06-001のまま不変。
3. 書き換えたchain骨とheadのcached `GlobalTransform`は手動更新し、GazeControl〜Constraints間の既存fresh-global契約を維持する。それ以外の子孫は既存`PropagateAfterExpressions`/`PropagateAfterConstraints`段で解決する。

### schedule order

```text
AnimationSystems
  -> update_body_tracking_pose_input   (rotation input; sole writer)
  -> update_body_tracking_position_input (position input; sole writer)
  -> apply_direct_body_tracking        (bone rotations)
  -> apply_direct_body_position        (root translation + torso lean)
  -> VrmSystemSets::GazeControl
  -> VrmSystemSets::Expressions
  -> PropagateAfterExpressions
  -> VrmSystemSets::Constraints
  -> ...
```

各Transform channelのwriterは必ず一意である。position inputが無い・inactive・confidence 0の場合、lean/offsetは0になりrotation-only挙動は従来と完全に一致する。translation観測がUnavailableでもrotation poseは破壊されない。

### bounds

`BodyTrackingPositionProfile`: max_lean_radians seed = 15°、max_body_translation_meters = 0.25 m。非finite入力は0化され、magnitudeはclampされる。出力は常にbounded finiteである。

## Reproducer and regression tests

- vendor単体テスト8件: semantic→model写像（identity / Y=180 rest）、lean角導出とclamp、degenerate lever fail-closed、sanitize（inactive/zero-confidence/non-finite）。
- `vtuber-avatar/tests/body_motion_integration.rs` 9件: synthetic humanoid rigでのX lateral（root X不変＋torso lean＋head横変位）、Y/Z root追従、同frame再評価のbit安定性、跨frame非蓄積、rotation-only後方互換、inactive inert、optional骨欠損時のsafe degrade、実行決定性、巨大/非有限入力のbounded finite。
- `tests/schedule.rs`: position input writerがanimation後・direct tracking前に登録され、既存ownership順序を壊さないことをschedule graph検証に追加。

target-model reproducerはsynthetic rig統合テストが担い、実機モデルでの視覚確認はIssue #173のheadless trace validationで別途行う（本ADRの時点ではNOT VERIFIEDのままでよい）。

## Rejected alternatives

- breathingと同じhips translation channelへの追記: 2つのsystemが同一channelを競合writeするため禁止（Issue #180／ADR-020でbreathing writer自体がretireされた）。
- leanをPoseInputのyaw/pitch/rollへ混入: head計測回転との二重加算になり、配分weightsもhead優先でlean意味論と矛盾する。
- feedback型IK solver: bounded deterministic要件とframe-rate invarianceを損なう。Webcam用途には小さい直接solveで十分。
- vtuber-app経由でtargetsを渡す方式: control frame契約の変更が不要な一方でavatar層だけで完結し、tracking純粋型をそのまま消費できるためavatar→tracking依存（DESIGN.md §9.3に追加）を採用。

## Amendment 2026-08-30: binding wiring fix + torso propagation profile

実機検証で「顔を動かすと腕だけが追従し胸・腰が静止する」不自然さが報告された。原因は2つ:

1. **位置channelの未接続（wiring bug）**: `bind_humanoid_bones`が`BodyTrackingPositionInput` / `BodyTrackingPositionProfile`をrootへinsertしていなかったため、`update_body_tracking_position_input`は常にskipし、`apply_direct_body_position`の`run_if(any_with_component::<BodyTrackingPositionInput>)`が一度も成立しなかった。つまり本ADRのposition-aware solve（torso lean + root/body translation追従）はプロダクションで死んでおり、head translationに追従するのはarm pipeline（`update_dynamic_arm_targets`が同一channelsを直接消費）のみだった。修正: bindingが両component（inactive初期値）をinsertする。writer ownershipは変更しない。
2. **回転配分の胴体シェア不足**: vendored `BodyTrackingProfile::default()`はpitch/rollの胴体合計が約5-7%かつtorso limitsが厳しく、頭部yaw/pitch/rollが胸・腰まで視覚的に伝播しない。修正: `vtuber-avatar::natural_body_tracking_profile()`を追加し、binding時にlibrary defaultの代わりにinsertする。yaw方針（小yawは頭・首のみ、12°から45°へengagement ramp）はlibrary defaultを維持し、pitch (0.58/0.20/0.12/0.06/0.04) とroll (0.60/0.20/0.12/0.05/0.03) で胴体合計を約20-22%へ引き上げ、torso limitsもupperChest (10° pitch / 8° roll)、chest (6° / 5°)、spine (5° / 4°) まで拡張する。配分は測定head姿勢のchain合計が測定値と一致する方式を維持するため、head方向は保ったまま可視の回転が胴体へ流れる。

head translation → torso lean (lateral) + root translation (Y/Z) + 頭部回転 → 首・胸・腰の配分、という合成により、顔の動きが腰（root translationはhips以下を含む全身）まで伝播する。vendor自体は変更しない（library defaultはfallbackとして不変）。回帰テストは`crates/vtuber-avatar/tests/binding.rs`（位置component挿入とtorso pitch shareの検証）と`crates/vtuber-avatar/src/pose/mod.rs`（配分合計・胴体シェア・limit整合）に追加した。

位置solve有効化の副作用として、hand targetのbody-follow補償（`DynamicArmProfile::compensation_gains`）が腕を体の反対側へ引き込むケースでupper armがT-pose基準90°（きをつけ）を超えて降り、胴体にめり込むことが確認された。対策としてarm pipelineにstage 3b（`arm_pipeline::clamp_upper_arm_swing`、定数`MAX_ARM_DROP_RADIANS` = 85°）を追加した: solvedなupper arm方向のcoronal面（model forward +Zに垂直）でのT-poseからの降下角が85°を超える場合、elbow/wrist/回転/deltaを肩周りで剛体回転させて上限へ戻す。腕を上げる方向と前後スイングは制限しない。85°は90°ではなく服の厚みぶんのマージンである。
