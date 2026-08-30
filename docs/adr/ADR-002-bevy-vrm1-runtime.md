# ADR-002: VRM 1.0 runtimeとしてbevy_vrm1を採用する

Status: Superseded in format scope by ADR-011; runtime decision retained
Date: 2026-08-04

## Context

VRM 1.0モデルをBevyで実用的に扱うには、glTF loadだけでなくHumanoid、Expression、MToon、SpringBone、Node Constraint等が必要になる。これらをアプリ側で独自実装すると、自由研究の主題である顔追跡と低遅延制御より、VRM runtime開発の比重が大きくなる。

`bevy_vrm1`はVRM 1.0専用のBevy pluginで、固定対象revisionはBevy 0.19を利用し、`.vrm` loader、Humanoid bone、Expression、MToon、SpringBone、Node Constraint、First Person、VRMA関連機能を含む。`.vrm` loaderは内部でBevyの`GltfLoader`を利用するため、「Bevy glTF loaderかbevy_vrm1か」の二者択一ではなく、Bevy loader上のVRM runtimeとして採用する。

## Decision

- `bevy_vrm1`を唯一のVRM execution systemとする。対象formatの世代判定とVRM 0.x extension normalizationはADR-011へ委譲する。
- Bevyを`=0.19.0`へ固定する。
- `bevy_vrm1`を次のGit revisionへ固定する。

```text
f9593fd78136fb9e0507bcae111e09291ec9b82a
```

- VRM runtime dependencyは`bevy_vrm1`へ一本化する。
- app側で`.vrm` AssetLoader、Humanoid runtime、MToon、SpringBone、Node Constraint、Expression accumulatorを実装しない。
- `bevy_vrm1`への直接依存は`vtuber-avatar`へ隔離する。
- model取込前の検査は安全性と互換性gateに限定し、VRM runtime schemaの複製にしない。

ADR-011はこのruntime選定を変更せず、legacy `extensions.VRM`を同じregistry contractへ入れるvendor境界だけを追加する。

## Public integration surface

使用する主な公開API:

- `VrmPlugin`
- `VrmHandle`
- `Vrm`
- `Initialized`
- `HeadBoneEntity`等のHumanoid bone entity holder
- `RestTransform`／`RestGlobalTransform`
- `ExpressionEntityMap`
- `ModifyExpressions`
- `VrmSystemSets`
- model replacement時のdetach API

product pathでは顔姿勢用の`LookAt`を使用しない。`Q2-06-002`は固定revisionの未実装Expression経路を埋め、webcamのeye-in-head信号だけをhead-relative直接LookAtへ渡す。cursor／target entityは生成しない。

`Q2-06-001`では、固定revision由来の最小vendored patchに直接yaw／pitch／roll入力を追加した`BodyTracking`を使用する。app側adapterは入力componentだけを更新し、head、neck、upperChest、chest、spineへ直接書き込まない。直接入力がない場合のupstream `LookAt + BodyTracking`挙動は維持する。

## Compatibility and patch policy

- 公式sample、VRoid Studio export、実利用modelをG0-08で検査する。
- target modelで再現しない問題のためにforkしない。ただし`Q2-06-001`の直接姿勢入力は、stock公開APIだけでは表現できない承認済み機能拡張として扱う。
- valid target modelでbugを再現した場合、minimal fixture、spec根拠、regression test、upstream issueを先に作る。
- forkが必要ならGit commit SHAへ固定し、app repositoryへsource断片をcopyしない。
- `Q2-06-001`と`Q2-06-002`ではbase revision、license、差分を記録したvendored patchを許可する。変更範囲はdirect BodyTracking、direct LookAt、range map再利用、選択backend出力、VRM順序、cleanup、必要な公開export／system登録と関連testに限定する。

## G0-08 findings

G0-08 compatibility gate（`cargo xtask vrm-compat`）を実施した結果、ピン留めした `bevy_vrm1` revision は以下の制約を持つことが確定した。

- `LookAtType::Expression` は `src/vrm/look_at.rs` で `todo!("Expression look at is not supported yet")` に到達する。したがって product path では `LookAt` component を挿入しない方針を維持する。
- `inore-vrm1.vrm`（実利用予定モデル）は head/neck/leftEye/rightEye と `blink`/`blinkLeft`/`blinkRight`、`aa`/`ih`/`ou`/`ee`/`oh` を含み、MVP capability をすべて満たす。
- 同モデルは `lookAt.type = bone` かつ look-direction Expression preset（`lookLeft`/`lookRight`/`lookUp`/`lookDown`）を持たないため、MVP gaze は eye bone 直接制御に依存する。
- VRM 0.x モデル（`tsukuyomi-chan.vrm`）は `VRMC_vrm` を持たないため、`MODEL_NOT_VRM1` で正しく拒否される。
- fixture に含まれる `alicia-solid.vrm` と `seed-san.vrm` は HTML ファイルであり `MODEL_FILE_INVALID` で正しく拒否される。

これらの所見に加え、stock `BodyTracking`は`LookAt`必須で、roll、upperChest、軸別weight、bone別half-life、direct pose pathを持たないことを確認した。`Q2-06-001`ではこの不足に限ったvendored patchを採用する。
