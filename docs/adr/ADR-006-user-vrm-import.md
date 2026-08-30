# ADR-006: user VRM importとBevy asset source

Status: Accepted  
Date: 2026-08-04

## Context

`bevy_vrm1::VrmHandle`はBevy AssetServerからloadする。一方、ユーザーは任意directoryのVRMを選択する。asset root外を無制限に許可すると、再起動時のpath管理、macOS bundle、security、transactional model replacementが不明瞭になる。

## Decision

- `<AppData>/user-assets/`をnamed asset source `user`として登録する。
- user-selected fileをpreflight後、content hash名で管理directoryへcopyする。
- AssetServer pathは`user://avatars/<sha256>/model.vrm`とする。
- global `UnapprovedPathMode::Allow`は使わない。
- temp fileへcopy・fsync相当・atomic renameの順でimportする。
- default file size上限は256 MiB、hard capは1 GiBとする。
- 同一SHA-256は再利用する。

## Preflight

- regular fileであること
- `.vrm` extension
- size limit
- GLB parse
- `VRM`または`VRMC_vrm`のどちらか一方が存在すること
- `VRMC_vrm`の場合は`specVersion == "1.0"`
- required `hips`／`head`
- humanoid node index範囲と重複
- external buffer／image URIなし
- meta summary
- SHA-256

## Error mapping

- missing or ambiguous VRM generation: `MODEL_NOT_VRM`
- unsupported specVersion: `MODEL_UNSUPPORTED_VERSION`
- missing required bone: `MODEL_MISSING_REQUIRED_BONE`
- invalid GLB／index／URI: `MODEL_FILE_INVALID`

## Consequences

import時に一回copyが発生するが、load path、recent list、hash-based cache、transactional replacementが安定する。preflightは安全gateであり、VRM runtimeを再実装しない。VRM 0.xはADR-011のvendor互換レイヤーへ渡すため、cache metadataにはgenerationを保存する。
