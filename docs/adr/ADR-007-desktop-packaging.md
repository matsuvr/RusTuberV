# ADR-007: Windows／macOS packaging

Status: Proposed; Q2-04で確定する  
Date: 2026-08-04

## Context

Windowsではportable ZIPが扱いやすい。一方、macOS camera permissionはbundle identifier、`Info.plist`、署名identityと結び付くため、raw binaryだけの検査では不十分である。推論modelとlicense resourceの配置もworking directoryへ依存させない必要がある。

## Candidate decision

- packaging entrypointは`cargo xtask package-windows`と`cargo xtask package-macos`。
- Windowsはx86_64 portable ZIP。
- macOSはApple Silicon `.app` bundle。
- resource lookupは`ResourceLocator`へ集約する。
- inference model、license、UI assetをpackage resourcesへcopyし、起動時にhash検証する。
- macOS `Info.plist`へ`NSCameraUsageDescription`を含める。
- local research buildはad-hoc signingを許容する。
- installer、App Store、Developer ID notarization、universal binaryはMVP外とする。

## Q2-04 acceptance evidence

- working directoryに依存せず起動する。
- clean Windows machineでmodel loadとcamera smokeを行う。
- `.app` bundleでcamera permissionが対象appへ付与される。
- bundled model hashとlicense reportが一致する。
- package内容と再現commandをrelease reportへ記録する。
