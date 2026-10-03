# ADR-011: Convert VRM 0.x at import, run only VRM 1.0

Status: Accepted  
Date: 2026-08-14; revised 2026-10-03

Supersedes: the VRM 1.0-only format restriction in ADR-002 and the former
generation-independent descriptor / runtime basis design of this ADR.

## Context

The application already uses Bevy's generic GLB/scene/image loading path and
the pinned `bevy_vrm1` runtime registries for Humanoid, Expression, MToon,
LookAt, Node Constraint, and SpringBone. VRM 0.x stores its model contract in
the legacy `extensions.VRM` object (`blendShapeMaster`, `materialProperties`,
and `secondaryAnimation`), while VRM 1.0 stores equivalent capabilities in
`VRMC_vrm` and related extensions. Adding a second loader or avatar runtime
would duplicate lifecycle and writer ownership.

The pinned specification references for this work are:

- VRM specification: `vrm-c/vrm-specification@821c11b250d8c70d5804ee13431e42bee56ea9c0`
- Reference implementation: `vrm-c/UniVRM@52e1250813f370783351788b5c4cd0332e59c9c3`

## Decision

1. Accept exactly VRM 0.x (`extensions.VRM`) and VRM 1.0
   (`extensions.VRMC_vrm` with `specVersion == "1.0"`). A file containing
   both roots or neither root is rejected before loading.
2. Keep Bevy's existing `GltfLoader`-backed `.vrm` asset path. No custom
   `.vrm` AssetLoader, conversion subprocess, Python, Unity, or new crate is
   introduced.
3. `vtuber-avatar::vrm::prepare_vrm_document` is the format boundary. Convert
   the legacy `VRM` extension into `VRMC_vrm`, `VRMC_materials_mtoon`, and
   `VRMC_springBone`, remove the legacy extension, then run the same VRM 1.0
   preparation used for native VRM 1.0. The private `vrm0` module owns legacy
   parsing and conversion; there is no shared runtime descriptor or VRM 1.0
   parser in that module.
4. Inspect the converted document and store those same bytes as the managed
   copy. Humanoid, expression, material, first-person and spring capability
   inspection reads only VRM 1.0. Morph reduction accepts prepared VRM 1.0
   and remaps only its node-indexed binds. Import, existing managed copies
   and compatibility tools share preparation and the morph-limit decision.
   The existing `VrmHandle -> Vrm -> Initialized -> AvatarBinding` lifecycle
   remains the only execution path. Its request, ECS root and runtime report
   do not carry an expected source format.
5. Bake the VRM 0.x forward-axis correction once into the GLB scene roots
   during conversion. There is no runtime basis entity. No source-format
   sign correction is allowed in tracking, gaze, camera, pose or breathing.
6. Normalize legacy expression groups, material properties, and secondary
   animation at the boundary. A secondary-animation terminal receives a
   7 cm synthetic joint; resolved node indices are deduplicated before writer
   registration, and gravity is transformed once with the same basis.
7. Treat glTF node index as runtime identity. Legacy mesh references are
   validated against `meshes`, expanded to every node that instantiates the
   mesh, and morph indices are validated against primitive target counts.
   The converter writes these references as VRM 1.0 node indices. Binding
   then follows the same pinned upstream contract as native VRM 1.0.
8. Read VRM 0.x LookAt only from `firstPerson`: `lookAtTypeName` is mapped
   from `Bone`/`BlendShape`, `firstPersonBoneOffset` is the official `{x,y,z}`
   object, and the four DegreeMap objects use direct `xRange`/`yRange` values
   with an optional numeric `curve` array. The obsolete synthetic
   `lookAtMaster` shape is not accepted.
9. Legacy materialProperties are indexed by glTF material index, never by
   material name or occurrence. Known Unlit and unknown shaders retain the
   generic glTF material fallback with a warning; valid MToon properties are
   converted into the existing renderer, including validated texture indices,
   alpha/cull/queue, UV, emission, outline, and color-space conversion.
10. The converter emits ordinary VRM 1.0 `preset` and `custom` expression
    maps. Only the common `vrm1` preparation merges custom expressions into
    the upstream registry's preset map and fills omitted expression defaults.
    Its custom-origin record and material binds remain available to the app.
    Existing converted copies with legacy thumb names are repaired inside
    the conversion module before common preparation, without the original file.

The resulting pipeline is:

```text
VRM 0.x source -> vrm0 conversion --+
                                  +-> VRM 1.0 preparation -> common inspection
VRM 1.0 source -------------------+      -> morph reduction -> managed model
                                         -> one bevy_vrm1 runtime
```

Source format, exporter and conversion warnings remain import metadata.
License review continues to read the original source's permissions, following
the [official migration guidance](https://vrm.dev/en/univrm1/migrate_vrm0/feature/).
It must not derive legal permissions from rendering data in the converted copy.
The source file and its hash-based model identity are unchanged.

## Consequences

- Preflight and import metadata can report a common summary while retaining
  the detected generation for diagnostics and cache invalidation.
- VRM 1.0 remains a mandatory regression target for every legacy compatibility
  change.
- Format-specific fixes belong in `vrm0`; common loading and behavior fixes
  belong in the VRM 1.0 path and apply to both sources. A malformed legacy
  field is a typed preparation/import error, not `NoFace` and not a panic.
- Missing, ambiguous or unsupported extensions are rejected both on initial
  import and when preparing an existing managed copy. Preparation is
  idempotent and does not require the original file for cached model loading.
- Full physical camera, MToon appearance, and SpringBone acceptance remain
  platform/model evidence and cannot be inferred from unit tests.

## Rejected alternatives

- A second VRM 0.x loader or ECS runtime: duplicates lifecycle, transforms,
  and writer ownership.
- Pre-converting files with Python, Unity, or a sidecar: violates the
  full-Rust boundary and makes the cached artifact non-transparent.
- Scattering a 180-degree correction through each feature: makes tracking and
  avatar semantics generation-dependent and prevents reliable regression
  tests.

## Verification record (2026-10-03)

- Import/preparation: 37 tests passed, including equivalent inspection and
  runtime bytes for both formats, idempotence, typed invalid-format failures,
  legacy binds, material colors, source preservation and morph remapping.
- Avatar VRM/binding tests: 12 passed; load/unload tests: 11 passed;
  lifecycle integration: 7 passed.
- Targeted Clippy for avatar/app/xtask, formatting and diff checks passed.
- Desktop and xtask debug builds succeeded.
- Existing `vrm-compat tests/fixtures/vrm` initialized the VRM 0.x
  `tsukuyomi-chan.vrm` and VRM 1.0 `inore-vrm1.vrm`, both MVP-capable.
  The directory-wide command exited with failure because `alicia-solid.vrm`
  and `seed-san.vrm` contain HTML (`<!DOCTYPE html>`) rather than GLB data;
  their runtime loading was not attempted. These fixture files were not changed.
- Camera, model appearance and macOS hardware were not checked.

## Verification record (2026-08-15)

Historical Issue #31 evidence (not additional requirements for later changes):

- the SpringBone terminal-direction fixed-step tests pass;
- the automated VRM 0.x/1.0 lifecycle transition matrix passes twice without
  stale roots, registries, generation state, or `SpringJointState`;
- the automated real-model matrix initializes five VRM 0.x and two VRM 1.0
  models as MVP-capable;
- three additional valid VRM 0.x artifacts hit the bounded 600-frame timeout
  and are explicitly not counted as PASS.

The former 20 real-model replacements and physical SpringBone soak are
superseded follow-up evidence, not current Issue #31 acceptance conditions.
Human visual/camera evidence and macOS hardware evidence are optional for this
automated gate; when not run they remain `NOT VERIFIED`. See
`docs/compatibility/vrm-0x-1x-2026-08-14.md` for the exact commands and matrix.
