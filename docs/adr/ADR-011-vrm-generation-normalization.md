# ADR-011: VRM 0.x/1.0 normalization into the existing runtime

Status: Accepted  
Date: 2026-08-14  
Supersedes: the VRM 1.0-only format restriction in ADR-002, DESIGN.md, and AGENTS.md

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
3. Add a pure vendor parser/normalizer that turns both extension layouts into
   a generation-independent `VrmRuntimeDescriptor`. It is data-only and does
   not expose Bevy entities, tracking types, or worker state.
4. Feed that descriptor into the existing `bevy_vrm1` registry contract. The
   existing `VrmHandle -> Vrm -> Initialized -> AvatarBinding` lifecycle and
   the existing Expression, MToon, LookAt, Node Constraint, and SpringBone
   systems remain the only execution paths.
5. Apply the VRM 0.x forward-axis correction once by placing its spawned scene
   below an adapter-owned basis entity rotated around Y by `pi`. VRM 1.0 has
   no extra basis rotation. No generation-dependent sign correction is
   allowed in tracking, gaze, camera, default pose, or breathing code.
6. Normalize legacy expression groups, material properties, and secondary
   animation at the boundary. A secondary-animation terminal receives a
   7 cm synthetic joint; resolved node indices are deduplicated before writer
   registration, and gravity is transformed once with the same basis.
7. Treat glTF node index as runtime identity. Legacy mesh references are
   validated against `meshes`, expanded to every node that instantiates the
   mesh, and morph indices are validated against primitive target counts.
   Runtime scene entities retain the source node index through Bevy's existing
   GLTF extension hook, so duplicate or missing node names cannot change the
   binding target.
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

## Consequences

- Preflight and import metadata can report a common summary while retaining
  the detected generation for diagnostics and cache invalidation.
- VRM 1.0 remains a mandatory regression target for every legacy compatibility
  change.
- The normalized descriptor must be versioned and tested as a compatibility
  boundary. A malformed legacy field is a typed import/runtime descriptor
  error, not `NoFace` and not a panic.
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

## Verification record (2026-08-15)

The current Issue #31 acceptance record is the deterministic local gate:

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
