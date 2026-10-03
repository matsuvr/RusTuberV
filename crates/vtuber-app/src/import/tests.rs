#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)] // tests may panic (AGENTS.md)
use super::*;
use serde_json::Value;
use tempfile::TempDir;
use vtuber_avatar::glb::Glb;

#[test]
fn rejects_non_vrm_extension() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.txt");
    fs::write(&path, b"not a vrm").unwrap();
    let err = import_vrm(&path, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
    assert!(matches!(err, ModelImportError::InvalidExtension));
}

#[test]
fn accepts_uppercase_vrm_extension_for_preflight() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.VRM");
    fs::write(&path, b"not a glb").unwrap();
    let err = import_vrm(&path, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
    assert!(!matches!(err, ModelImportError::InvalidExtension));
}

#[test]
fn rejects_directory() {
    let dir = TempDir::new().unwrap();
    let subdir = dir.path().join("model.vrm");
    fs::create_dir(&subdir).unwrap();
    let err = import_vrm(&subdir, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
    assert!(matches!(err, ModelImportError::NotRegularFile));
}

#[test]
fn rejects_oversized() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.vrm");
    fs::write(&path, b"x").unwrap();
    let err = import_vrm(&path, dir.path(), 0).unwrap_err();
    assert!(matches!(err, ModelImportError::SizeExceeded { .. }));
}

#[test]
fn rejects_hard_cap_config() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.vrm");
    fs::write(&path, b"x").unwrap();
    let err = import_vrm(&path, dir.path(), HARD_SIZE_CAP + 1).unwrap_err();
    assert!(matches!(err, ModelImportError::LimitExceedsHardCap { .. }));
}

#[test]
fn rejects_invalid_bytes() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.vrm");
    fs::write(&path, b"not glb").unwrap();
    let err = import_vrm(&path, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
    assert!(matches!(err, ModelImportError::GlbParse(_)));
}

#[test]
fn import_metadata_file_round_trips() {
    let dir = TempDir::new().unwrap();
    let meta_path = dir.path().join("import.toml");
    let imported = ImportedModel {
        id: "test".into(),
        name: "x".into(),
        asset_path: dir.path().join("model.vrm"),
        meta_path: meta_path.clone(),
        summary: VrmInspectionSummary::default(),
        original_path: dir.path().join("source.vrm"),
        size: 1,
    };
    let meta = ImportMeta {
        imported: imported.clone(),
        mtime: None,
    };
    fs::write(&meta_path, toml::to_string_pretty(&meta).unwrap()).unwrap();

    let read: ImportMeta = toml::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(read.imported, imported);
    assert_eq!(read.mtime, meta.mtime);
}

const NON_VRM_GLTF_JSON: &str = r#"{
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{}]
    }"#;

const VRM0_GLTF_JSON: &str = r#"{
        "asset": {"version": "2.0", "generator": "vtuber-app hermetic test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "buffers": [{"byteLength": 12}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}],
        "meshes": [{"name": "Face", "primitives": [{"attributes": {"POSITION": 0}, "targets": [{"POSITION": 0}, {"POSITION": 0}]}]}],
        "materials": [{"name": "Body"}, {"name": "Body"}],
        "nodes": [
            {"name": "Hips", "children": [1, 2, 3, 4]},
            {"name": "Head"},
            {"name": "Neck"},
            {"name": "Face", "mesh": 0},
            {"name": "Face", "mesh": 0}
        ],
        "extensionsUsed": ["VRM"],
        "extensions": {
            "VRM": {
                "exporterVersion": "UniVRM 0.123",
                "meta": {"title": "Hermetic VRM 0.x", "author": "Legacy Author", "exporterVersion": "nonstandard-meta"},
                "humanoid": {
                    "humanBones": [
                        {"bone": "hips", "node": 0},
                        {"bone": "head", "node": 1},
                        {"bone": "neck", "node": 2}
                    ]
                },
                "firstPerson": {
                    "firstPersonBone": 1,
                    "firstPersonBoneOffset": {"x": 0.0, "y": 0.1, "z": 0.2},
                    "meshAnnotations": [{"mesh": 0, "firstPersonFlag": "Both"}],
                    "lookAtTypeName": "BlendShape",
                    "lookAtHorizontalInner": {"curve": [0.0, 0.0, 0.0, 0.0], "xRange": 90.0, "yRange": 10.0},
                    "lookAtHorizontalOuter": {"xRange": 90.0, "yRange": 10.0},
                    "lookAtVerticalDown": {"xRange": 90.0, "yRange": 10.0},
                    "lookAtVerticalUp": {"xRange": 90.0, "yRange": 10.0}
                },
                "blendShapeMaster": {
                    "blendShapeGroups": [
                        {"name": "vowel-a", "presetName": "A", "binds": [{"mesh": 0, "index": 1, "weight": 100}]},
                        {"name": "blink", "presetName": "Blink_L"},
                        {"name": "joy", "presetName": "Joy"},
                        {"name": "customSmile", "presetName": "unknown"}
                    ]
                },
                "materialProperties": [
                    {"name": "Body", "shader": "VRM/MToon", "floatProperties": {"_Cull": 0.0}},
                    {"name": "Body", "shader": "VRM/MToon", "floatProperties": {"_Cull": 2.0}}
                ],
                "secondaryAnimation": {
                    "colliderGroups": [{"node": 2, "colliders": [{"offset": {"x": 0.0, "y": 0.1, "z": 0.0}, "radius": 0.02}]}],
                    "boneGroups": [{"bones": [3], "center": 1, "colliderGroups": [0], "gravityDir": {"x": 0.0, "y": -1.0, "z": 0.0}, "gravityPower": 0.5, "stiffiness": 0.8, "dragForce": 0.2, "hitRadius": 0.01}]
                }
            }
        }
    }"#;

const VRM1_GLTF_JSON: &str = r#"{
        "asset": {"version": "2.0", "generator": "vtuber-app hermetic test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "Hips", "children": [1]},
            {"name": "Head"}
        ],
        "extensionsUsed": ["VRMC_vrm", "VRMC_springBone"],
        "extensions": {
            "VRMC_vrm": {
                "specVersion": "1.0",
                "meta": {"name": "Hermetic VRM 1.0"},
                "humanoid": {
                    "humanBones": {
                        "hips": {"node": 0},
                        "head": {"node": 1}
                    }
                },
                "expressions": {
                    "preset": {
                        "happy": {"isBinary": false},
                        "neutral": {"isBinary": false}
                    },
                    "custom": {
                        "JawOpen": {"isBinary": false},
                        "笑顔": {"isBinary": false}
                    }
                }
            },
            "VRMC_springBone": {}
        }
    }"#;

fn write_glb_fixture(dir: &TempDir, file_name: &str, json: &str) -> PathBuf {
    let bytes = Glb::new(serde_json::from_str(json).unwrap(), Some(&[0; 12]))
        .to_vec()
        .unwrap();

    let path = dir.path().join(file_name);
    fs::write(&path, bytes).unwrap();
    path
}

fn legacy_fixture(dir: &TempDir) -> PathBuf {
    write_glb_fixture(dir, "legacy.vrm", NON_VRM_GLTF_JSON)
}

fn vrm0_fixture(dir: &TempDir) -> PathBuf {
    write_glb_fixture(dir, "legacy-0.x.vrm", VRM0_GLTF_JSON)
}

fn vrm1_fixture(dir: &TempDir) -> PathBuf {
    write_glb_fixture(dir, "hermetic.vrm", VRM1_GLTF_JSON)
}

#[test]
fn generated_non_vrm_glb_is_rejected_with_generation_error() {
    let dir = TempDir::new().unwrap();
    let err = inspect_vrm(legacy_fixture(&dir)).unwrap_err();
    assert!(matches!(&err, ModelImportError::NotVrm { .. }));
    assert_eq!(err.code(), "MODEL_NOT_VRM");
}

#[test]
fn generated_non_vrm_glb_import_is_rejected_with_generation_error() {
    let dir = TempDir::new().unwrap();
    let source = legacy_fixture(&dir);
    let err = import_vrm(source, dir.path(), DEFAULT_SIZE_LIMIT).unwrap_err();
    assert!(matches!(err, ModelImportError::NotVrm { .. }));
}

#[test]
fn inspects_generated_minimal_vrm0_fixture() {
    let dir = TempDir::new().unwrap();
    let summary = inspect_vrm(vrm0_fixture(&dir)).expect("fixture should be valid VRM 0.x");
    assert_eq!(summary.generation, VrmGeneration::Vrm0);
    assert_eq!(summary.spec_version, "0.x");
    assert_eq!(summary.exporter_version.as_deref(), Some("UniVRM 0.123"));
    assert_eq!(summary.name, "Hermetic VRM 0.x");
    assert_eq!(summary.authors, vec!["Legacy Author"]);
    assert_eq!(summary.look_at_type.as_deref(), Some("expression"));
    assert_eq!(
        summary.expression_presets,
        vec!["aa", "blinkLeft", "customSmile", "happy"]
    );
    assert!(summary.has_spring_bone);
    assert!(summary.has_mtoon_materials);
    assert_eq!(summary.humanoid_nodes.neck, Some(2));
}

#[test]
fn inspects_generated_minimal_vrm1_fixture() {
    let dir = TempDir::new().unwrap();
    let summary = inspect_vrm(vrm1_fixture(&dir)).expect("fixture should be valid VRM 1.0");
    assert_eq!(summary.generation, VrmGeneration::Vrm1);
    assert_eq!(summary.spec_version, "1.0");
    assert_eq!(summary.exporter_version, None);
    assert!(!summary.name.is_empty(), "model name should be present");
    assert!(summary.humanoid_nodes.hips < 1000);
    assert!(summary.humanoid_nodes.head < 1000);
    assert!(summary.has_spring_bone);
    // Both standard presets and author-defined custom expressions are
    // surfaced; Unicode custom names stay exact.
    assert_eq!(
        summary.expression_presets,
        vec!["JawOpen", "happy", "neutral", "笑顔"]
    );
}

#[test]
fn rejects_legacy_mesh_and_morph_indices_during_preflight() {
    let dir = TempDir::new().unwrap();
    let invalid_mesh = VRM0_GLTF_JSON.replace(
        "\"meshAnnotations\": [{\"mesh\": 0",
        "\"meshAnnotations\": [{\"mesh\": 99",
    );
    let mesh_path = write_glb_fixture(&dir, "invalid-mesh.vrm", &invalid_mesh);
    assert!(matches!(
        inspect_vrm(mesh_path),
        Err(ModelImportError::InvalidMeshIndex { index: 99 })
    ));

    let invalid_morph =
        VRM0_GLTF_JSON.replace("\"index\": 1, \"weight\"", "\"index\": 99, \"weight\"");
    let morph_path = write_glb_fixture(&dir, "invalid-morph.vrm", &invalid_morph);
    assert!(matches!(
        inspect_vrm(morph_path),
        Err(ModelImportError::InvalidMorphTargetIndex { mesh: 0, index: 99 })
    ));
}

#[test]
fn rejects_duplicate_legacy_human_bones_during_preflight() {
    let dir = TempDir::new().unwrap();
    let duplicate = VRM0_GLTF_JSON.replace(
        "{\"bone\": \"head\", \"node\": 1}",
        "{\"bone\": \"head\", \"node\": 1}, {\"bone\": \"head\", \"node\": 2}",
    );
    let path = write_glb_fixture(&dir, "duplicate-bone.vrm", &duplicate);
    let err = inspect_vrm(path).unwrap_err();
    assert!(matches!(&err, ModelImportError::DuplicateHumanBone(name) if name == "head"));
    assert_eq!(err.code(), "MODEL_DUPLICATE_HUMAN_BONE");
}

#[test]
fn accepts_legacy_bone_look_at_during_preflight() {
    let dir = TempDir::new().unwrap();
    let bone = VRM0_GLTF_JSON.replace(
        "\"lookAtTypeName\": \"BlendShape\"",
        "\"lookAtTypeName\": \"Bone\"",
    );
    let path = write_glb_fixture(&dir, "bone-look-at.vrm", &bone);
    let summary = inspect_vrm(path).expect("Bone LookAt should be valid");
    assert_eq!(summary.look_at_type.as_deref(), Some("bone"));
    let zero_range = bone.replace("\"xRange\": 90.0", "\"xRange\": 0.0");
    let path = write_glb_fixture(&dir, "zero-range.vrm", &zero_range);
    assert_eq!(
        inspect_vrm(&path).unwrap().look_at_type.as_deref(),
        Some("bone")
    );
    let converted = vtuber_avatar::convert_vrm0_to_vrm1(&fs::read(path).unwrap())
        .unwrap()
        .unwrap();
    let document = Glb::parse(&converted).unwrap().document;
    assert_eq!(
        document
            .pointer("/extensions/VRMC_vrm/lookAt/rangeMapHorizontalInner/inputMaxValue")
            .and_then(Value::as_f64),
        Some(90.0)
    );
}

#[test]
fn rejects_malformed_legacy_degree_map_during_preflight() {
    let dir = TempDir::new().unwrap();
    let malformed = VRM0_GLTF_JSON.replace(
        "\"curve\": [0.0, 0.0, 0.0, 0.0]",
        "\"curve\": \"not-an-array\"",
    );
    let path = write_glb_fixture(&dir, "malformed-degree-map.vrm", &malformed);
    assert!(matches!(
        inspect_vrm(path),
        Err(ModelImportError::InvalidVrmField { path, .. })
            if path.ends_with("lookAtHorizontalInner.curve")
    ));
}

#[test]
fn rejects_ambiguous_vrm_generation() {
    let dir = TempDir::new().unwrap();
    let both = VRM1_GLTF_JSON.replace("\"VRMC_vrm\": {", "\"VRM\": {}, \"VRMC_vrm\": {");
    let path = write_glb_fixture(&dir, "ambiguous.vrm", &both);
    let err = inspect_vrm(path).unwrap_err();
    assert!(matches!(&err, ModelImportError::AmbiguousVrmVersion { .. }));
    assert_eq!(err.code(), "MODEL_AMBIGUOUS_VRM_VERSION");
}

#[test]
fn old_summary_defaults_to_vrm1_for_cache_compatibility() {
    let summary: VrmInspectionSummary = toml::from_str(
        r#"spec_version = "1.0"
name = "old cache"
authors = []
expression_presets = []
has_spring_bone = false
has_node_constraint = false
humanoid_nodes = { hips = 0, head = 1 }
"#,
    )
    .expect("old cache summary should remain readable");
    assert_eq!(summary.generation, VrmGeneration::Vrm1);
    assert!(!summary.has_first_person);
}

#[test]
fn imports_generated_minimal_vrm1_fixture() {
    let dir = TempDir::new().unwrap();
    let source = vrm1_fixture(&dir);
    let asset_root = dir.path().join("asset-root");
    let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
        .expect("fixture should import successfully");
    assert_eq!(imported.summary.spec_version, "1.0");
    assert!(imported.asset_path.exists());
    assert!(imported.meta_path.exists());
    // Re-import with same file should be idempotent.
    let reimported =
        import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT).expect("re-import should succeed");
    assert_eq!(imported.id, reimported.id);
    assert_eq!(imported.asset_path, reimported.asset_path);
}

#[test]
fn repairs_corrupt_existing_cached_file() {
    let dir = TempDir::new().unwrap();
    let source = vrm1_fixture(&dir);
    let asset_root = dir.path().join("asset-root");
    let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
        .expect("fixture should import successfully");

    fs::write(&imported.asset_path, b"corrupt cached model").unwrap();
    let repaired = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
        .expect("re-import should repair the cached file");

    assert_eq!(repaired.id, imported.id);
    // The managed copy carries the VRM 1.0 expression adaptation (custom
    // expressions merged into `preset`, optional fields filled), so it is
    // the adapted runtime bytes rather than the raw source.
    assert_eq!(
        fs::read(&repaired.asset_path).unwrap(),
        runtime_ready_source_bytes(&fs::read(source).unwrap(), VrmGeneration::Vrm1).unwrap()
    );
}

fn over_limit_vrm0_fixture(dir: &TempDir, target_count: usize, bind_indices: &[usize]) -> PathBuf {
    let mut root: serde_json::Value = serde_json::from_str(VRM0_GLTF_JSON).unwrap();
    let targets: Vec<serde_json::Value> =
        (0..target_count).map(|_| serde_json::json!({})).collect();
    let names: Vec<serde_json::Value> = (0..target_count)
        .map(|index| serde_json::Value::from(format!("m{index}")))
        .collect();
    let mut weights: Vec<serde_json::Value> = vec![serde_json::Value::from(0.0); target_count];
    weights[5] = serde_json::Value::from(0.25);
    weights[250] = serde_json::Value::from(0.75);

    let mesh = root["meshes"][0].as_object_mut().unwrap();
    mesh["primitives"][0]["targets"] = serde_json::Value::Array(targets);
    mesh.insert(
        "extras".to_string(),
        serde_json::json!({"targetNames": names}),
    );
    mesh.insert("weights".to_string(), serde_json::Value::Array(weights));

    let binds: Vec<serde_json::Value> = bind_indices
        .iter()
        .map(|index| serde_json::json!({"mesh": 0, "index": index, "weight": 50.0}))
        .collect();
    let groups = root["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]
        .as_array_mut()
        .unwrap();
    groups[0]["binds"] = serde_json::Value::Array(binds);

    write_glb_fixture(dir, "vrm0-over-limit.vrm", &root.to_string())
}

fn stored_glb_json(imported: &ImportedModel) -> serde_json::Value {
    let stored = fs::read(&imported.asset_path).unwrap();
    let glb = Glb::parse(&stored).expect("stored copy is a valid GLB");
    let (json, bin) = (glb.document, glb.bin);
    assert!(bin.is_some(), "BIN chunk must be preserved");
    json
}

#[test]
fn import_reduces_morph_targets_beyond_bevy_limit() {
    let dir = TempDir::new().unwrap();
    let source = over_limit_vrm0_fixture(&dir, 300, &[5, 250]);
    let asset_root = dir.path().join("asset-root");

    let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
        .expect("over-limit model should import");

    // Identity remains keyed to the original bytes; the stored copy is
    // the VRM 1.0-shaped conversion, morph-normalized.
    let source_bytes = fs::read(&source).unwrap();
    assert_eq!(imported.id, format!("{:x}", Sha256::digest(&source_bytes)));
    assert_ne!(fs::read(&imported.asset_path).unwrap(), source_bytes);

    let json = stored_glb_json(&imported);
    assert!(
        json["extensions"].get("VRM").is_none(),
        "managed copy is VRM 1.0-shaped"
    );
    let mesh = &json["meshes"][0];
    let targets = mesh["primitives"][0]["targets"].as_array().unwrap();
    assert_eq!(targets.len(), 2);
    let names = mesh["extras"]["targetNames"].as_array().unwrap();
    assert_eq!(names[0], "m5");
    assert_eq!(names[1], "m250");
    let weights = mesh["weights"].as_array().unwrap();
    assert_eq!(weights[0], 0.25);
    assert_eq!(weights[1], 0.75);

    // The legacy mesh/morph binds were converted to node binds (mesh 0 is
    // instanced by nodes 3 and 4) and remapped to the reduced targets.
    let binds = json["extensions"]["VRMC_vrm"]["expressions"]["preset"]["aa"]["morphTargetBinds"]
        .as_array()
        .unwrap();
    let pairs = binds
        .iter()
        .map(|bind| {
            (
                bind["node"].as_u64().unwrap(),
                bind["index"].as_u64().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(pairs, vec![(3, 0), (4, 0), (3, 1), (4, 1)]);

    // Re-import is idempotent and keeps the normalized copy.
    let reimported =
        import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT).expect("re-import succeeds");
    assert_eq!(imported.id, reimported.id);
    assert_eq!(
        fs::read(&imported.asset_path).unwrap(),
        fs::read(&reimported.asset_path).unwrap()
    );
}

#[test]
fn import_converts_vrm0_within_limits_to_vrm1_shape() {
    let dir = TempDir::new().unwrap();
    let source = vrm0_fixture(&dir);
    let asset_root = dir.path().join("asset-root");
    let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT).expect("fixture imports");
    // Identity stays keyed to the source bytes, but the managed copy is
    // the converted VRM 1.0 shape with the binary chunk preserved.
    let source_bytes = fs::read(&source).unwrap();
    assert_eq!(imported.id, format!("{:x}", Sha256::digest(&source_bytes)));
    let json = stored_glb_json(&imported);
    assert!(
        json["extensions"].get("VRM").is_none(),
        "managed copy drops the legacy extension"
    );
    assert_eq!(
        json["extensions"]["VRMC_vrm"]["specVersion"], "1.0",
        "managed copy carries the normalized descriptor"
    );
    assert!(
        json["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
            .as_object()
            .is_some_and(|bones| !bones.is_empty()),
        "humanoid survives conversion"
    );
}

/// Builds a VRM 0.x fixture whose meta carries explicit permission
/// strings and whose expressions declare legacy `materialValues`, to
/// exercise the R2/R3/R5 managed-copy behavior end to end.
fn vrm0_permissions_fixture(dir: &TempDir) -> PathBuf {
    let mut root: serde_json::Value = serde_json::from_str(VRM0_GLTF_JSON).unwrap();
    root["extensions"]["VRM"]["meta"]["allowedUserName"] = serde_json::json!("OnlyAuthor");
    root["extensions"]["VRM"]["meta"]["violentUsageName"] = serde_json::json!("Disallow");
    root["extensions"]["VRM"]["meta"]["sexualUsageName"] = serde_json::json!("Allow");
    root["extensions"]["VRM"]["meta"]["commercialUsageName"] = serde_json::json!("Allow");
    root["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"][0]["materialValues"] = serde_json::json!([
        {
            "materialName": "Body",
            "propertyName": "_Color",
            "targetValue": [0.8, 0.2, 0.1, 1.0]
        }
    ]);
    write_glb_fixture(dir, "vrm0-permissions.vrm", &root.to_string())
}

#[test]
fn import_preserves_permissions_and_material_binds_in_the_managed_copy() {
    let dir = TempDir::new().unwrap();
    let source = vrm0_permissions_fixture(&dir);
    let imported = import_vrm(&source, dir.path().join("asset-root"), DEFAULT_SIZE_LIMIT)
        .expect("fixture imports");
    let json = stored_glb_json(&imported);
    let meta = &json["extensions"]["VRMC_vrm"]["meta"];

    // R5: the managed copy records what the source actually permitted —
    // mapped strings pass through, VRM 0.x fields with no VRM 1.0
    // counterpart record no permission instead of `true`.
    assert_eq!(meta["allowExcessivelySexualUsage"], true);
    assert_eq!(meta["allowExcessivelyViolentUsage"], false);
    assert_eq!(meta["allowRedistribution"], false);
    assert_eq!(meta["allowAntisocialOrHateUsage"], false);
    assert_eq!(meta["allowPoliticalOrReligiousUsage"], false);
    assert_eq!(meta["avatarPermission"], "OnlyAuthor");
    assert_eq!(meta["commercialUsage"], "Allow");

    // R3: legacy `materialValues` survive conversion as VRM 1.0 color
    // binds, with the glTF material index of the named material.
    let happy = &json["extensions"]["VRMC_vrm"]["expressions"]["preset"]["aa"];
    let color_binds = happy["materialColorBinds"].as_array().unwrap();
    assert_eq!(color_binds.len(), 1);
    assert_eq!(color_binds[0]["material"], 0);
    assert_eq!(color_binds[0]["type"], "color");
    assert_eq!(
        color_binds[0]["targetValue"],
        serde_json::json!([0.8_f32, 0.2_f32, 0.1_f32, 1.0_f32])
    );

    // R2: the custom-origin record is retained next to the merged
    // `preset` entry.
    let custom = &json["extensions"]["VRMC_vrm"]["expressions"]["custom"];
    assert!(custom.get("customSmile").is_some());
}

#[test]
fn import_adapts_vrm1_custom_expressions_and_omitted_defaults() {
    let dir = TempDir::new().unwrap();
    let source = vrm1_fixture(&dir);
    let imported = import_vrm(&source, dir.path().join("asset-root"), DEFAULT_SIZE_LIMIT)
        .expect("fixture imports");
    let json = stored_glb_json(&imported);
    let expressions = &json["extensions"]["VRMC_vrm"]["expressions"];
    let preset = expressions["preset"].as_object().unwrap();

    // Custom expressions reach the `preset` map the runtime reads, with
    // their names exact, and keep the custom-origin record.
    assert!(preset.contains_key("JawOpen"));
    assert!(preset.contains_key("笑顔"));
    assert!(expressions["custom"].get("JawOpen").is_some());
    // Optional spec fields are filled so the upstream serde contract
    // accepts the file.
    assert_eq!(preset["JawOpen"]["isBinary"], false);
    assert_eq!(preset["JawOpen"]["overrideBlink"], "none");

    // The runtime facts read back from the same managed copy keep the
    // provenance exactly.
    let facts = read_runtime_expression_facts(&imported.asset_path).unwrap();
    assert!(!facts.entry("JawOpen").unwrap().declared_as_preset);
    assert!(facts.entry("happy").unwrap().declared_as_preset);
}

#[test]
fn ensure_managed_model_ready_adapts_stale_vrm1_copies_without_the_original() {
    let dir = TempDir::new().unwrap();
    let source = vrm1_fixture(&dir);
    let asset_root = dir.path().join("asset-root");
    let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT).expect("fixture imports");
    let source_bytes = fs::read(&source).unwrap();
    let expected = runtime_ready_source_bytes(&source_bytes, VrmGeneration::Vrm1).unwrap();

    // Simulate a managed copy stored before the VRM 1.0 adaptation (the
    // raw source passthrough) with the original moved away: the managed
    // copy alone must still be adapted.
    fs::write(&imported.asset_path, &source_bytes).unwrap();
    fs::remove_file(&source).unwrap();
    assert!(
        ensure_managed_model_ready(&imported.asset_path).expect("adaptation succeeds"),
        "the stale copy must be rewritten"
    );
    assert_eq!(fs::read(&imported.asset_path).unwrap(), expected);

    // Custom origin and omitted defaults survive the managed-copy-only
    // adaptation.
    let facts = read_runtime_expression_facts(&imported.asset_path).unwrap();
    assert!(!facts.entry("JawOpen").unwrap().declared_as_preset);
    assert!(facts.entry("happy").unwrap().declared_as_preset);

    // A second run is a no-op.
    assert!(
        !ensure_managed_model_ready(&imported.asset_path).expect("second run succeeds"),
        "an adapted copy must not be rewritten again"
    );
}

#[test]
fn ensure_managed_model_ready_converts_vrm0_copies_without_the_original() {
    let dir = TempDir::new().unwrap();
    let source = vrm0_fixture(&dir);
    let asset_root = dir.path().join("asset-root");
    let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT).expect("fixture imports");
    let source_bytes = fs::read(&source).unwrap();
    let expected = runtime_ready_source_bytes(&source_bytes, VrmGeneration::Vrm0).unwrap();

    // Simulate an unconverted VRM 0.x managed copy with no original.
    fs::write(&imported.asset_path, &source_bytes).unwrap();
    fs::remove_file(&source).unwrap();
    assert!(
        ensure_managed_model_ready(&imported.asset_path).expect("conversion succeeds"),
        "the unconverted copy must be rewritten"
    );
    assert_eq!(fs::read(&imported.asset_path).unwrap(), expected);

    let json = stored_glb_json(&imported);
    assert!(
        json["extensions"].get("VRM").is_none(),
        "managed copy is VRM 1.0-shaped"
    );
    assert!(json["extensions"].get("VRMC_vrm").is_some());
}

/// Builds a VRM 0.x fixture whose only `happy` expression is an author
/// custom (`presetName: "unknown"`); the standard Joy group is removed,
/// so a later adaptation could mistake the merged copy for a genuine
/// standard preset collision.
fn vrm0_custom_happy_fixture(dir: &TempDir) -> PathBuf {
    let mut root: serde_json::Value = serde_json::from_str(VRM0_GLTF_JSON).unwrap();
    let groups = root["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]
        .as_array_mut()
        .unwrap();
    groups.retain(|group| group["presetName"] != "Joy");
    groups.push(serde_json::json!({"name": "happy", "presetName": "unknown"}));
    write_glb_fixture(dir, "vrm0-custom-happy.vrm", &root.to_string())
}

#[test]
fn ensure_managed_model_ready_keeps_a_vrm0_custom_origin_and_classification() {
    let dir = TempDir::new().unwrap();
    let source = vrm0_custom_happy_fixture(&dir);
    let asset_root = dir.path().join("asset-root");
    let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT).expect("fixture imports");

    // The VRM0 conversion emits the same output contract as the VRM 1.0
    // adaptation: the custom origin record carries the merged-custom
    // marker, while the runtime `preset` copy does not.
    let json = stored_glb_json(&imported);
    let expressions = &json["extensions"]["VRMC_vrm"]["expressions"];
    assert!(expressions["preset"].get("happy").is_some());
    assert_eq!(
        expressions["custom"]["happy"][vtuber_avatar::vrm1::MERGED_CUSTOM_MARKER],
        serde_json::json!(true)
    );

    // Load preparation re-runs the shared adaptation on the converted
    // copy. The marked record must survive: origin and classification
    // stay custom and the managed copy is not rewritten.
    let managed_before = fs::read(&imported.asset_path).unwrap();
    assert!(!ensure_managed_model_ready(&imported.asset_path).expect("ensure succeeds"));
    assert_eq!(fs::read(&imported.asset_path).unwrap(), managed_before);
    let facts = read_runtime_expression_facts(&imported.asset_path).unwrap();
    let happy = facts.entry("happy").expect("happy fact");
    assert!(!happy.declared_as_preset);
    assert_eq!(
        vtuber_avatar::classify_expression("happy", happy.declared_as_preset),
        vtuber_avatar::ExpressionKind::Custom
    );

    // The managed copy alone carries everything: repeated ensure and a
    // deleted original change neither the copy nor the facts.
    fs::remove_file(&source).unwrap();
    assert!(!ensure_managed_model_ready(&imported.asset_path).expect("ensure succeeds"));
    assert_eq!(fs::read(&imported.asset_path).unwrap(), managed_before);
    assert_eq!(
        read_runtime_expression_facts(&imported.asset_path).unwrap(),
        facts
    );
}

#[test]
fn normalization_bails_when_referenced_binds_alone_exceed_the_limit() {
    let dir = TempDir::new().unwrap();
    let bind_indices: Vec<usize> = (0..MAX_MORPH_TARGETS + 4).collect();
    let source = over_limit_vrm0_fixture(&dir, 300, &bind_indices);
    let bytes = fs::read(&source).unwrap();
    assert!(normalize_vrm_morph_targets(&bytes).unwrap().is_none());

    let error = import_vrm(&source, dir.path().join("asset-root"), DEFAULT_SIZE_LIMIT)
        .expect_err("an over-limit model that cannot be reduced must not be cached raw");
    assert!(matches!(
        error,
        ModelImportError::InvalidVrmField { ref path, .. }
            if path == "meshes[*].primitives[*].targets"
    ));
}

#[test]
fn normalization_rejects_non_glb_bytes() {
    assert!(normalize_vrm_morph_targets(b"not glb").is_err());
}

#[test]
fn legacy_expression_ids_match_inspection_diagnostics_and_runtime() {
    let dir = TempDir::new().unwrap();
    let mut root: Value = serde_json::from_str(VRM0_GLTF_JSON).unwrap();
    root["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"] = serde_json::json!([
        {"presetName": "A", "name": "author-vowel"},
        {"presetName": "Joy", "name": ""},
        {"presetName": "vendorPreset", "name": "smile"},
        {"presetName": "Unknown", "name": "joy"},
        {"presetName": "", "name": "A"},
        {"name": "Blink_L"},
        {"presetName": "unknown", "name": " "},
        {"presetName": "vendorPreset"},
        {"presetName": "vendorOther", "name": "smile"}
    ]);
    let source = write_glb_fixture(&dir, "identities.vrm", &root.to_string());
    let summary = inspect_vrm(&source).unwrap();
    let expected = [
        "A", "Blink_L", "aa", "custom_6", "custom_7", "happy", "joy", "smile",
    ];
    assert_eq!(summary.expression_presets, expected);
    let warnings = &summary.compatibility_warnings;
    assert_eq!(
        warnings
            .iter()
            .filter(|warning| warning.code
                == vtuber_avatar::VrmCompatibilityWarningCode::EmptyLegacyExpressionName)
            .count(),
        2
    );
    assert!(warnings.iter().any(|warning| warning.code
        == vtuber_avatar::VrmCompatibilityWarningCode::DuplicateLegacyExpression
        && warning.context.contains("canonical=smile")));
    let imported = import_vrm(&source, dir.path().join("managed"), DEFAULT_SIZE_LIMIT).unwrap();
    let facts = read_runtime_expression_facts(&imported.asset_path).unwrap();
    assert_eq!(
        facts
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        expected
    );
    for entry in facts.entries {
        assert_eq!(
            entry.declared_as_preset,
            matches!(entry.name.as_str(), "aa" | "happy")
        );
    }
}

#[test]
fn inspection_and_conversion_share_container_rules_and_keep_unknown_chunks() {
    let dir = TempDir::new().unwrap();
    let source = over_limit_vrm0_fixture(&dir, 300, &[5]);
    let mut bytes = fs::read(&source).unwrap();
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&0x1234_u32.to_le_bytes());
    bytes.extend_from_slice(&[9, 8, 7, 6]);
    let length = bytes.len() as u32;
    bytes[8..12].copy_from_slice(&length.to_le_bytes());
    fs::write(&source, &bytes).unwrap();
    assert!(inspect_vrm(&source).is_ok());
    let imported = import_vrm(&source, dir.path().join("managed"), DEFAULT_SIZE_LIMIT).unwrap();
    let stored = fs::read(imported.asset_path).unwrap();
    assert_eq!(&stored[stored.len() - 12..], &bytes[bytes.len() - 12..]);
    assert_eq!(
        Glb::parse(&stored).unwrap().bin,
        Glb::parse(&bytes).unwrap().bin
    );
    bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
    fs::write(&source, &bytes).unwrap();
    assert!(inspect_vrm(&source).is_err());
    assert!(vtuber_avatar::prepare_managed_vrm_bytes(&bytes).is_err());
    assert!(normalize_vrm_morph_targets(&bytes).is_err());
}

#[test]
fn invalid_image_range_returns_a_reason_without_panicking() {
    let dir = TempDir::new().unwrap();
    let mut root: Value = serde_json::from_str(VRM1_GLTF_JSON).unwrap();
    root["buffers"] = serde_json::json!([{"byteLength": 12}]);
    root["bufferViews"] = serde_json::json!([{"buffer": 0, "byteOffset": 12, "byteLength": 8}]);
    root["images"] = serde_json::json!([{"bufferView": 0, "mimeType": "image/png"}]);
    let source = write_glb_fixture(&dir, "bad-image.vrm", &root.to_string());
    let error = import_vrm(&source, dir.path().join("managed"), DEFAULT_SIZE_LIMIT).unwrap_err();
    assert!(
        matches!(&error, ModelImportError::InvalidVrmField { path, .. } if path == "bufferViews[0]")
    );
    assert!(
        error
            .to_string()
            .contains("12 + 8 exceeds buffer 0 (12 bytes)")
    );
    assert!(!dir.path().join("managed").exists());
}

#[test]
fn reduction_preserves_mesh_and_node_default_shapes() {
    let dir = TempDir::new().unwrap();
    let source = over_limit_vrm0_fixture(&dir, 300, &[5]);
    let bytes = fs::read(source).unwrap();
    let mut glb = Glb::parse(&bytes).unwrap();
    let root = &mut glb.document;
    // The fixture's mesh defaults use 5 and 250; add a node-only default.
    let mut weights = vec![0.0_f32; 300];
    weights[299] = 1.0;
    root["nodes"][3]["weights"] = serde_json::json!(weights);
    let normalized = normalize_vrm_morph_targets(&glb.to_vec().unwrap())
        .unwrap()
        .unwrap();
    let after = Glb::parse(&normalized).unwrap().document;
    assert_eq!(
        after["meshes"][0]["weights"],
        serde_json::json!([0.25, 0.75, 0.0])
    );
    assert_eq!(
        after["nodes"][3]["weights"],
        serde_json::json!([0.0, 0.0, 1.0])
    );
    assert_eq!(
        after["meshes"][0]["primitives"][0]["targets"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn too_many_nonzero_defaults_are_rejected_without_changing_the_source() {
    let dir = TempDir::new().unwrap();
    let source = over_limit_vrm0_fixture(&dir, 300, &[5]);
    let bytes = fs::read(&source).unwrap();
    let mut glb = Glb::parse(&bytes).unwrap();
    let root = &mut glb.document;
    root["meshes"][0]["weights"] = serde_json::json!(vec![1.0; 300]);
    let bytes = glb.to_vec().unwrap();
    fs::write(&source, &bytes).unwrap();
    let error = import_vrm(&source, dir.path().join("managed"), DEFAULT_SIZE_LIMIT).unwrap_err();
    assert!(matches!(error, ModelImportError::InvalidVrmField { .. }));
    assert_eq!(fs::read(&source).unwrap(), bytes);
}

#[test]
fn normalization_bails_on_morph_weight_animations() {
    let dir = TempDir::new().unwrap();
    let source = over_limit_vrm0_fixture(&dir, 300, &[5]);
    let source_bytes = fs::read(source).unwrap();
    let mut glb = Glb::parse(&source_bytes).unwrap();
    let root = &mut glb.document;
    root["animations"] = serde_json::json!([{
        "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}],
        "samplers": [{"input": 0, "output": 0}]
    }]);
    let bytes = glb.to_vec().unwrap();
    assert_eq!(over_limit_morph_target_count(&bytes), Some(300));
    assert!(normalize_vrm_morph_targets(&bytes).unwrap().is_none());
}

fn over_limit_vrm1_fixture(dir: &TempDir) -> PathBuf {
    let mut root: serde_json::Value = serde_json::from_str(VRM1_GLTF_JSON).unwrap();
    root["buffers"] = serde_json::json!([{"byteLength": 12}]);
    root["bufferViews"] = serde_json::json!([{"buffer": 0, "byteOffset": 0, "byteLength": 12}]);
    root["accessors"] = serde_json::json!([{
        "bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3",
        "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]
    }]);
    root["nodes"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"name": "Face", "mesh": 0}));
    root["meshes"] = serde_json::json!([{
        "name": "Face",
        "primitives": [{
            "attributes": {"POSITION": 0},
            "targets": (0..300).map(|_| serde_json::json!({})).collect::<Vec<_>>()
        }],
        "extras": {"targetNames": (0..300).map(|index| format!("m{index}")).collect::<Vec<_>>()},
        "weights": (0..300).map(|_| 0.0).collect::<Vec<_>>()
    }]);
    root["extensions"]["VRMC_vrm"]["expressions"] = serde_json::json!({
        "preset": {
            "aa": {"morphTargetBinds": [{"node": 2, "index": 5}, {"node": 2, "index": 250}]}
        },
        "custom": {
            "smile": {"morphTargetBinds": [{"node": 2, "index": 250}]}
        }
    });
    write_glb_fixture(dir, "vrm1-over-limit.vrm", &root.to_string())
}

#[test]
fn import_remaps_vrm1_node_based_binds() {
    let dir = TempDir::new().unwrap();
    let source = over_limit_vrm1_fixture(&dir);
    let asset_root = dir.path().join("asset-root");
    let imported = import_vrm(&source, &asset_root, DEFAULT_SIZE_LIMIT)
        .expect("over-limit VRM 1.0 should import");

    let json = stored_glb_json(&imported);
    let expressions = &json["extensions"]["VRMC_vrm"]["expressions"];
    let preset_binds = &expressions["preset"]["aa"]["morphTargetBinds"];
    assert_eq!(preset_binds[0]["index"], 0);
    assert_eq!(preset_binds[0]["node"], 2);
    assert_eq!(preset_binds[1]["index"], 1);
    let custom_binds = &expressions["custom"]["smile"]["morphTargetBinds"];
    assert_eq!(custom_binds[0]["index"], 1);
    let targets = json["meshes"][0]["primitives"][0]["targets"]
        .as_array()
        .unwrap();
    assert_eq!(targets.len(), 2);
}
