//! Generation-independent VRM core data used at the runtime boundary.
//!
//! This module intentionally contains no Bevy entities, assets, or systems.
//! It parses the core model contract from the glTF JSON and gives the VRM 0.x
//! and VRM 1.0 paths the same descriptor before ECS initialization.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Maximum context length for a compatibility warning.
pub const MAX_COMPATIBILITY_WARNING_CONTEXT: usize = 256;

/// Stable machine-readable compatibility warning codes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VrmCompatibilityWarningCode {
    /// A legacy `DegreeMap` curve cannot be represented by the current linear
    /// runtime range map.
    NonLinearLegacyLookAtCurve,
    /// A legacy expression contains materialValues which are not applied by
    /// the existing expression runtime.
    LegacyExpressionMaterialValuesUnsupported,
    /// Two legacy expression groups normalize to the same canonical identity.
    DuplicateLegacyExpression,
    /// A legacy custom expression had no usable name.
    EmptyLegacyExpressionName,
    /// A legacy expression repeats the same node/morph bind.
    DuplicateLegacyExpressionBind,
    /// A legacy material property key is not part of the known migration set.
    UnknownLegacyMaterialProperty,
    /// Legacy materialProperties contains more entries than glTF materials.
    ExtraLegacyMaterialProperty,
    /// A legacy shader is not handled by the existing material path.
    UnknownLegacyShader,
    /// `StandardMaterial` has no equivalent for legacy transparent Z-write.
    UnlitZWriteNotRepresentable,
    /// A known legacy texture property contains an invalid glTF texture index.
    InvalidLegacyMaterialTexture,
}

/// Exact VRM 0.x shader classification shared by diagnostics and migration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyShaderKind {
    /// The only shader migrated through the `MToon` path.
    MToon,
    /// One of the four known `StandardMaterial` fallbacks.
    SupportedUnlit,
    /// A known glTF/legacy passthrough shader with no VRM migration.
    Passthrough,
    /// An exporter-specific or otherwise unknown shader.
    Unknown,
}

/// Classifies a VRM 0.x shader by exact fixed-schema name.
#[must_use]
pub fn classify_legacy_shader(shader: &str) -> LegacyShaderKind {
    match shader {
        "VRM/MToon" => LegacyShaderKind::MToon,
        "VRM/UnlitTexture"
        | "VRM/UnlitCutout"
        | "VRM/UnlitTransparent"
        | "VRM/UnlitTransparentZWrite" => LegacyShaderKind::SupportedUnlit,
        "VRM_USE_GLTFSHADER" | "Standard" | "UniGLTF/UniUnlit" => {
            LegacyShaderKind::Passthrough
        }
        _ => LegacyShaderKind::Unknown,
    }
}

impl VrmCompatibilityWarningCode {
    /// Returns the stable machine-readable spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonLinearLegacyLookAtCurve => "non_linear_legacy_look_at_curve",
            Self::LegacyExpressionMaterialValuesUnsupported => {
                "legacy_expression_material_values_unsupported"
            }
            Self::DuplicateLegacyExpression => "duplicate_legacy_expression",
            Self::EmptyLegacyExpressionName => "empty_legacy_expression_name",
            Self::DuplicateLegacyExpressionBind => "duplicate_legacy_expression_bind",
            Self::UnknownLegacyMaterialProperty => "unknown_legacy_material_property",
            Self::ExtraLegacyMaterialProperty => "extra_legacy_material_property",
            Self::UnknownLegacyShader => "unknown_legacy_shader",
            Self::UnlitZWriteNotRepresentable => "unlit_z_write_not_representable",
            Self::InvalidLegacyMaterialTexture => "invalid_legacy_material_texture",
        }
    }
}

/// A bounded, generation-owned compatibility diagnostic.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VrmCompatibilityWarning {
    /// Stable warning code.
    pub code: VrmCompatibilityWarningCode,
    /// Short source path or other bounded context.
    pub context: String,
}

impl VrmCompatibilityWarning {
    /// Constructs a warning while enforcing the bounded context contract.
    #[must_use]
    pub fn new(code: VrmCompatibilityWarningCode, context: impl Into<String>) -> Self {
        Self {
            code,
            context: context.into().chars().take(MAX_COMPATIBILITY_WARNING_CONTEXT).collect(),
        }
    }
}

/// VRM generation represented by a normalized descriptor.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VrmGeneration {
    /// Legacy VRM 0.x using the root VRM extension.
    Vrm0,
    /// VRM 1.0 using the root `VRMC_vrm` extension.
    Vrm1,
}

/// Single coordinate correction applied to a legacy VRM scene.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CoordinateBasis {
    /// VRM 0.x faces -Z and is placed below a Y=pi basis entity.
    Vrm0Y180,
    /// VRM 1.0 uses the application's canonical glTF basis unchanged.
    Vrm1Identity,
}

/// Data-only descriptor shared by the VRM 0.x and VRM 1.0 runtime paths.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VrmRuntimeDescriptor {
    /// Detected VRM generation.
    pub generation: VrmGeneration,
    /// Stable source version marker (0.x for VRM 0.x).
    pub spec_version: String,
    /// Coordinate basis to apply at the scene boundary.
    pub coordinate_basis: CoordinateBasis,
    /// Model metadata.
    pub meta: VrmMeta,
    /// Source-only VRM 0.x metadata retained for diagnostics and reports.
    #[serde(default)]
    pub legacy_meta: Option<Vrm0MetaDiagnostics>,
    /// Humanoid node references by canonical bone name.
    pub humanoid: VrmHumanoid,
    /// Optional first-person metadata.
    pub first_person: Option<VrmFirstPerson>,
    /// Optional gaze metadata.
    pub look_at: Option<VrmLookAt>,
    /// Compatibility diagnostics produced while normalizing the source.
    #[serde(default)]
    pub compatibility_warnings: Vec<VrmCompatibilityWarning>,
}

/// Source metadata which has no common VRM 1.0 runtime equivalent.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct Vrm0MetaDiagnostics {
    /// Exporter identifier/version from `VRM.exporterVersion`.
    pub exporter_version: Option<String>,
    /// Source metadata version.
    pub version: Option<String>,
    /// Contact information supplied by the exporter.
    pub contact_information: Option<String>,
    /// Reference/credit text supplied by the exporter.
    pub reference: Option<String>,
    /// Thumbnail texture index.
    pub texture_index: Option<usize>,
    /// Usage permission fields retained as source strings.
    pub allowed_user_name: Option<String>,
    pub violent_usage_name: Option<String>,
    pub sexual_usage_name: Option<String>,
    pub commercial_usage_name: Option<String>,
    pub other_permission_url: Option<String>,
    /// License names and both legacy URL spellings.
    pub license_name: Option<String>,
    pub license_url: Option<String>,
    pub other_license_url: Option<String>,
}

/// Common model metadata.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct VrmMeta {
    /// Display name, if supplied by the exporter.
    pub name: Option<String>,
    /// Author names.
    pub authors: Vec<String>,
    /// License URL, if supplied by the exporter.
    pub license_url: Option<String>,
}

/// Common Humanoid node references.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VrmHumanoid {
    /// Node index by canonical VRM Humanoid bone name.
    pub human_bones: BTreeMap<String, usize>,
}

/// First-person mesh annotation data.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct VrmFirstPerson {
    /// Optional first-person reference bone from VRM 0.x.
    pub first_person_bone: Option<usize>,
    /// Optional headset offset from the first-person bone.
    pub first_person_bone_offset: Option<[f32; 3]>,
    /// Mesh/node visibility annotations.
    pub mesh_annotations: Vec<VrmMeshAnnotation>,
}

/// One first-person visibility annotation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VrmMeshAnnotation {
    /// glTF node index for the annotated mesh.
    pub node: usize,
    /// Canonical visibility flag.
    pub flag: VrmFirstPersonFlag,
}

/// Canonical first-person visibility flag.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum VrmFirstPersonFlag {
    /// Let the runtime decide from head weights.
    Auto,
    /// Visible in both views.
    Both,
    /// Visible only in third-person view.
    ThirdPersonOnly,
    /// Visible only in first-person view.
    FirstPersonOnly,
}

/// Common gaze configuration.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VrmLookAt {
    /// Bone or expression gaze backend declared by the model.
    pub r#type: VrmLookAtType,
    /// Offset from the head bone.
    pub offset_from_head_bone: [f32; 3],
    /// Horizontal inner range map.
    pub range_map_horizontal_inner: VrmRangeMap,
    /// Horizontal outer range map.
    pub range_map_horizontal_outer: VrmRangeMap,
    /// Vertical down range map.
    pub range_map_vertical_down: VrmRangeMap,
    /// Vertical up range map.
    pub range_map_vertical_up: VrmRangeMap,
}

/// Canonical gaze backend.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum VrmLookAtType {
    /// Direct eye-bone gaze.
    Bone,
    /// Expression-based gaze.
    Expression,
}

/// Canonical input/output range map.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct VrmRangeMap {
    /// Maximum input magnitude.
    pub input_max_value: f32,
    /// Output magnitude in the runtime's canonical units.
    pub output_scale: f32,
}

/// Errors returned by the pure core descriptor parser.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VrmParseError {
    /// No supported root extension was found.
    MissingGeneration,
    /// Both generation roots were supplied.
    AmbiguousGeneration,
    /// VRM 1.0 declared an unsupported version.
    UnsupportedVersion(String),
    /// A required field was not present.
    MissingField(String),
    /// A field had an invalid JSON type or value.
    InvalidField { path: String, reason: String },
    /// A glTF index was outside the referenced array.
    InvalidIndex { path: String, index: usize },
    /// A legacy human bone was declared more than once.
    DuplicateBone(String),
}

impl fmt::Display for VrmParseError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::MissingGeneration => write!(f, "missing VRM or VRMC_vrm extension"),
            Self::AmbiguousGeneration => {
                write!(f, "both VRM and VRMC_vrm extensions are present")
            }
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported VRMC_vrm specVersion {version}")
            }
            Self::MissingField(path) => write!(f, "missing required field {path}"),
            Self::InvalidField { path, reason } => write!(f, "invalid field {path}: {reason}"),
            Self::InvalidIndex { path, index } => {
                write!(f, "invalid index {path}={index}")
            }
            Self::DuplicateBone(name) => write!(f, "duplicate Humanoid bone {name}"),
        }
    }
}

impl std::error::Error for VrmParseError {}

/// Parses either the glTF root JSON or a root JSON object containing the
/// extensions member into a common core descriptor.
pub fn parse_runtime_descriptor(root: &Value) -> Result<VrmRuntimeDescriptor, VrmParseError> {
    let extensions = root
        .get("extensions")
        .and_then(Value::as_object)
        .ok_or(VrmParseError::MissingGeneration)?;
    let legacy = extensions.get("VRM");
    let modern = extensions.get("VRMC_vrm");

    match (legacy, modern) {
        (Some(_), Some(_)) => Err(VrmParseError::AmbiguousGeneration),
        (Some(vrm), None) => parse_vrm0(root, vrm),
        (None, Some(vrmc)) => parse_vrm1(vrmc),
        (None, None) => Err(VrmParseError::MissingGeneration),
    }
}

fn parse_vrm0(
    root: &Value,
    vrm: &Value,
) -> Result<VrmRuntimeDescriptor, VrmParseError> {
    let meta = parse_vrm0_meta(vrm.get("meta"));
    let legacy_meta = Some(parse_vrm0_meta_diagnostics(vrm));
    let humanoid = parse_vrm0_humanoid(root, vrm.get("humanoid"))?;
    let first_person = vrm
        .get("firstPerson")
        .map(|value| parse_vrm0_first_person(root, value))
        .transpose()?;
    let look_at = vrm
        .get("firstPerson")
        .and_then(|value| {
            let has_look_at = value.get("lookAtTypeName").is_some()
                || [
                    "lookAtHorizontalInner",
                    "lookAtHorizontalOuter",
                    "lookAtVerticalDown",
                    "lookAtVerticalUp",
                ]
                .iter()
                .any(|field| value.get(*field).is_some());
            has_look_at.then_some(value)
        })
        .map(parse_vrm0_look_at)
        .transpose()?;

    let compatibility_warnings = collect_legacy_compatibility_warnings(root, vrm);

    Ok(VrmRuntimeDescriptor {
        generation: VrmGeneration::Vrm0,
        spec_version: "0.x".into(),
        coordinate_basis: CoordinateBasis::Vrm0Y180,
        meta,
        legacy_meta,
        humanoid,
        first_person,
        look_at,
        compatibility_warnings,
    })
}

fn parse_vrm1(vrmc: &Value) -> Result<VrmRuntimeDescriptor, VrmParseError> {
    let spec_version = required_string(vrmc, "specVersion")?;
    if spec_version != "1.0" {
        return Err(VrmParseError::UnsupportedVersion(spec_version));
    }
    let meta = parse_vrm1_meta(vrmc.get("meta"));
    let humanoid = parse_vrm1_humanoid(vrmc.get("humanoid"))?;
    let first_person = vrmc
        .get("firstPerson")
        .map(parse_vrm1_first_person)
        .transpose()?;
    let look_at = vrmc.get("lookAt").map(parse_vrm1_look_at).transpose()?;

    Ok(VrmRuntimeDescriptor {
        generation: VrmGeneration::Vrm1,
        spec_version,
        coordinate_basis: CoordinateBasis::Vrm1Identity,
        meta,
        legacy_meta: None,
        humanoid,
        first_person,
        look_at,
        compatibility_warnings: Vec::new(),
    })
}

fn parse_vrm0_meta(meta: Option<&Value>) -> VrmMeta {
    let Some(meta) = meta.and_then(Value::as_object) else {
        return VrmMeta::default();
    };
    VrmMeta {
        name: string_field(meta, "title").or_else(|| string_field(meta, "name")),
        authors: string_field(meta, "author").into_iter().collect(),
        license_url: string_field(meta, "otherLicenseUrl")
            .or_else(|| string_field(meta, "licenseUrl")),
    }
}

fn parse_vrm0_meta_diagnostics(vrm: &Value) -> Vrm0MetaDiagnostics {
    let meta = vrm.get("meta").and_then(Value::as_object);
    let string = |field: &str| meta.and_then(|meta| string_field(meta, field));
    let texture_index = meta
        .and_then(|meta| meta.get("texture"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    Vrm0MetaDiagnostics {
        exporter_version: string_field_from(vrm, "exporterVersion")
            .or_else(|| string("exporterVersion")),
        version: string("version"),
        contact_information: string("contactInformation"),
        reference: string("reference"),
        texture_index,
        allowed_user_name: string("allowedUserName"),
        violent_usage_name: string("violentUsageName"),
        sexual_usage_name: string("sexualUsageName"),
        commercial_usage_name: string("commercialUsageName"),
        other_permission_url: string("otherPermissionUrl"),
        license_name: string("licenseName"),
        license_url: string("licenseUrl"),
        other_license_url: string("otherLicenseUrl"),
    }
}

fn string_field_from(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(String::from)
}

/// Collects non-fatal VRM 0.x migration diagnostics in source order.
pub fn collect_legacy_compatibility_warnings(
    root: &Value,
    legacy: &Value,
) -> Vec<VrmCompatibilityWarning> {
    let mut warnings = Vec::new();

    if let Some(first_person) = legacy.get("firstPerson") {
        for field in [
            "lookAtHorizontalInner",
            "lookAtHorizontalOuter",
            "lookAtVerticalDown",
            "lookAtVerticalUp",
        ] {
            if let Some(range) = first_person.get(field).and_then(Value::as_object)
                && range
                    .get("curve")
                    .and_then(Value::as_array)
                    .is_some_and(|curve| !legacy_curve_is_linear(curve))
            {
                warnings.push(VrmCompatibilityWarning::new(
                    VrmCompatibilityWarningCode::NonLinearLegacyLookAtCurve,
                    format!("VRM.firstPerson.{field}.curve"),
                ));
            }
        }
    }

    if let Some(properties) = legacy
        .get("materialProperties")
        .and_then(Value::as_array)
    {
        let material_count = root
            .get("materials")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        let texture_count = root
            .get("textures")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        if properties.len() > material_count {
            warnings.push(VrmCompatibilityWarning::new(
                VrmCompatibilityWarningCode::ExtraLegacyMaterialProperty,
                format!(
                    "VRM.materialProperties length {} exceeds glTF materials length {material_count}",
                    properties.len()
                ),
            ));
        }
        for (index, property) in properties.iter().enumerate() {
            if let Some(shader) = property.get("shader").and_then(Value::as_str) {
                if classify_legacy_shader(shader) == LegacyShaderKind::Unknown {
                    warnings.push(VrmCompatibilityWarning::new(
                        VrmCompatibilityWarningCode::UnknownLegacyShader,
                        format!("VRM.materialProperties[{index}].shader={shader}"),
                    ));
                }
                if shader == "VRM/UnlitTransparentZWrite" {
                    warnings.push(VrmCompatibilityWarning::new(
                        VrmCompatibilityWarningCode::UnlitZWriteNotRepresentable,
                        format!("VRM.materialProperties[{index}].shader"),
                    ));
                }
            }
            for key in property
                .get("floatProperties")
                .and_then(Value::as_object)
                .into_iter()
                .flat_map(|values| values.keys())
                .chain(
                    property
                        .get("vectorProperties")
                        .and_then(Value::as_object)
                        .into_iter()
                        .flat_map(|values| values.keys()),
                )
                .chain(
                    property
                        .get("textureProperties")
                        .and_then(Value::as_object)
                        .into_iter()
                        .flat_map(|values| values.keys()),
                )
            {
                if !is_known_legacy_material_property(key) {
                    warnings.push(VrmCompatibilityWarning::new(
                        VrmCompatibilityWarningCode::UnknownLegacyMaterialProperty,
                        format!("VRM.materialProperties[{index}].{key}"),
                    ));
                }
            }
            if let Some(textures) = property
                .get("textureProperties")
                .and_then(Value::as_object)
            {
                for (name, source_index) in textures {
                    if is_known_legacy_texture_property(name)
                        && !valid_legacy_texture_index(source_index, texture_count)
                    {
                        warnings.push(VrmCompatibilityWarning::new(
                            VrmCompatibilityWarningCode::InvalidLegacyMaterialTexture,
                            format!(
                                "VRM.materialProperties[{index}].textureProperties.{name}={source_index}"
                            ),
                        ));
                    }
                }
            }
        }
    }

    if let Some(groups) = legacy
        .get("blendShapeMaster")
        .and_then(|master| master.get("blendShapeGroups"))
        .and_then(Value::as_array)
    {
        let mut seen = BTreeMap::new();
        for (group_index, group) in groups.iter().enumerate() {
            let name = normalized_legacy_expression_name(group, group_index, &mut warnings);
            if let Some(previous) = seen.insert(name.clone(), group_index) {
                warnings.push(VrmCompatibilityWarning::new(
                    VrmCompatibilityWarningCode::DuplicateLegacyExpression,
                    format!(
                        "VRM.blendShapeMaster.blendShapeGroups[{group_index}] canonical={name} first={previous}"
                    ),
                ));
            }
            if group.get("materialValues").is_some() {
                warnings.push(VrmCompatibilityWarning::new(
                    VrmCompatibilityWarningCode::LegacyExpressionMaterialValuesUnsupported,
                    format!("VRM.blendShapeMaster.blendShapeGroups[{group_index}].materialValues"),
                ));
            }
            let mut binds = std::collections::BTreeSet::new();
            if let Some(raw_binds) = group.get("binds").and_then(Value::as_array) {
                for (bind_index, bind) in raw_binds.iter().enumerate() {
                    let mesh = bind.get("mesh").and_then(Value::as_u64);
                    let morph = bind.get("index").and_then(Value::as_u64);
                    let Some((mesh, morph)) = mesh.zip(morph) else { continue };
                    let nodes = mesh_instance_nodes_for_warning(root, mesh as usize);
                    for node in nodes {
                        if !binds.insert((node, morph as usize)) {
                            warnings.push(VrmCompatibilityWarning::new(
                                VrmCompatibilityWarningCode::DuplicateLegacyExpressionBind,
                                format!(
                                    "VRM.blendShapeMaster.blendShapeGroups[{group_index}].binds[{bind_index}] node={node} morph={morph}"
                                ),
                            ));
                        }
                    }
                }
            }
        }
    }
    warnings
}

fn mesh_instance_nodes_for_warning(root: &Value, mesh: usize) -> Vec<usize> {
    root.get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, node)| {
            (node.get("mesh").and_then(Value::as_u64) == Some(mesh as u64)).then_some(index)
        })
        .collect()
}

fn is_known_legacy_material_property(key: &str) -> bool {
    matches!(
        key,
        "_Color"
            | "_MainColor"
            | "_ShadeColor"
            | "_MainTex"
            | "_ShadeTexture"
            | "_BumpScale"
            | "_BumpMap"
            | "_ReceiveShadowRate"
            | "_ReceiveShadowTexture"
            | "_ShadingGradeRate"
            | "_ShadingGradeTexture"
            | "_ShadeShift"
            | "_ShadeToony"
            | "_LightColorAttenuation"
            | "_IndirectLightIntensity"
            | "_EmissionColor"
            | "_EmissionMap"
            | "_OutlineColor"
            | "_OutlineWidthTexture"
            | "_OutlineWidth"
            | "_OutlineScaledMaxDistance"
            | "_OutlineLightingMix"
            | "_RimColor"
            | "_RimTexture"
            | "_RimLightingMix"
            | "_RimFresnelPower"
            | "_RimLift"
            | "_SphereAdd"
            | "_UvAnimMaskTexture"
            | "_UvAnimScrollX"
            | "_UvAnimScrollY"
            | "_UvAnimRotation"
            | "_MToonVersion"
            | "_DebugMode"
            | "_BlendMode"
            | "_OutlineWidthMode"
            | "_OutlineColorMode"
            | "_CullMode"
            | "_OutlineCullMode"
            | "_SrcBlend"
            | "_DstBlend"
            | "_ZWrite"
            // Compatibility aliases found in older non-official fixtures.
            | "_MainTexture"
            | "_MainTex_ST"
            | "_Cutoff"
            | "_Cull"
    )
}

fn is_known_legacy_texture_property(key: &str) -> bool {
    matches!(
        key,
        "_MainTex"
            | "_MainTexture"
            | "_ShadeTexture"
            | "_BumpMap"
            | "_ReceiveShadowTexture"
            | "_ShadingGradeTexture"
            | "_EmissionMap"
            | "_EmissionTexture"
            | "_OutlineWidthTexture"
            | "_RimTexture"
            | "_RimMultiplyTexture"
            | "_SphereAdd"
            | "_ShadingShiftTexture"
            | "_MatcapTexture"
            | "_MatCapTex"
            | "_UvAnimMaskTex"
            | "_UvAnimMaskTexture"
    )
}

fn valid_legacy_texture_index(value: &Value, texture_count: usize) -> bool {
    value
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .is_some_and(|index| index < texture_count)
}

fn legacy_curve_is_linear(curve: &[Value]) -> bool {
    if curve.len() < 8 || !curve.len().is_multiple_of(4) {
        return false;
    }
    let keys = curve.chunks_exact(4).collect::<Vec<_>>();
    let Some(first) = keys.first() else {
        return false;
    };
    let Some(last) = keys.last() else {
        return false;
    };
    const EPSILON: f64 = 1.0e-6;

    let Some(first_time) = first[0].as_f64() else {
        return false;
    };
    let Some(first_value) = first[1].as_f64() else {
        return false;
    };
    let Some(last_time) = last[0].as_f64() else {
        return false;
    };
    let Some(last_value) = last[1].as_f64() else {
        return false;
    };
    if (first_time - 0.0).abs() > EPSILON
        || (first_value - 0.0).abs() > EPSILON
        || (last_time - 1.0).abs() > EPSILON
        || (last_value - 1.0).abs() > EPSILON
    {
        return false;
    }

    // UniVRM serializes the identity AnimationCurve as keys on y=x. The
    // endpoint tangents may be zero because they are not used outside the
    // interval; the tangents joining each adjacent pair must be one.
    if keys.iter().any(|key| {
        let Some(time) = key[0].as_f64() else {
            return true;
        };
        let Some(value) = key[1].as_f64() else {
            return true;
        };
        (time - value).abs() > EPSILON
    }) {
        return false;
    }
    keys.windows(2).all(|pair| {
        let Some(left_time) = pair[0][0].as_f64() else {
            return false;
        };
        let Some(right_time) = pair[1][0].as_f64() else {
            return false;
        };
        if right_time <= left_time {
            return false;
        }
        let outgoing = pair[0][3].as_f64();
        let incoming = pair[1][2].as_f64();
        outgoing.zip(incoming).is_some_and(|(outgoing, incoming)| {
            (outgoing - 1.0).abs() <= EPSILON && (incoming - 1.0).abs() <= EPSILON
        })
    })
}

fn normalized_legacy_expression_name(
    group: &Value,
    group_index: usize,
    warnings: &mut Vec<VrmCompatibilityWarning>,
) -> String {
    let preset = group
        .get("presetName")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("unknown"));
    let name = group
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let Some(source) = preset.or(name) else {
        warnings.push(VrmCompatibilityWarning::new(
            VrmCompatibilityWarningCode::EmptyLegacyExpressionName,
            format!("VRM.blendShapeMaster.blendShapeGroups[{group_index}]"),
        ));
        return format!("custom_{group_index}");
    };
    match source {
        "A" | "a" => "aa",
        "I" | "i" => "ih",
        "U" | "u" => "ou",
        "E" | "e" => "ee",
        "O" | "o" => "oh",
        "Blink" | "blink" => "blink",
        "Blink_L" | "blink_l" => "blinkLeft",
        "Blink_R" | "blink_r" => "blinkRight",
        "Joy" | "joy" => "happy",
        "Angry" | "angry" => "angry",
        "Sorrow" | "sorrow" => "sad",
        "Fun" | "fun" => "relaxed",
        "LookUp" | "lookup" => "lookUp",
        "LookDown" | "lookdown" => "lookDown",
        "LookLeft" | "lookleft" => "lookLeft",
        "LookRight" | "lookright" => "lookRight",
        "Neutral" | "neutral" => "neutral",
        other => other,
    }
    .to_string()
}

fn parse_vrm1_meta(meta: Option<&Value>) -> VrmMeta {
    let Some(meta) = meta.and_then(Value::as_object) else {
        return VrmMeta::default();
    };
    VrmMeta {
        name: string_field(meta, "name"),
        authors: meta
            .get("authors")
            .and_then(Value::as_array)
            .map(|authors| {
                authors
                    .iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default(),
        license_url: string_field(meta, "licenseUrl"),
    }
}

fn parse_vrm0_humanoid(
    root: &Value,
    humanoid: Option<&Value>,
) -> Result<VrmHumanoid, VrmParseError> {
    let bones = humanoid
        .and_then(|value| value.get("humanBones"))
        .and_then(Value::as_array)
        .ok_or_else(|| VrmParseError::MissingField("VRM.humanoid.humanBones".into()))?;
    let mut human_bones = BTreeMap::new();
    for (index, bone) in bones.iter().enumerate() {
        let path = format!("VRM.humanoid.humanBones[{index}]");
        let name = bone
            .get("bone")
            .and_then(Value::as_str)
            .ok_or_else(|| VrmParseError::MissingField(format!("{path}.bone")))?;
        let node = required_usize(bone, "node", &format!("{path}.node"))?;
        validate_node_index(root, node, &format!("{path}.node"))?;
        if human_bones.insert(name.to_string(), node).is_some() {
            return Err(VrmParseError::DuplicateBone(name.into()));
        }
    }
    require_humanoid_bones(&human_bones)?;
    Ok(VrmHumanoid { human_bones })
}

fn parse_vrm1_humanoid(humanoid: Option<&Value>) -> Result<VrmHumanoid, VrmParseError> {
    let bones = humanoid
        .and_then(|value| value.get("humanBones"))
        .and_then(Value::as_object)
        .ok_or_else(|| VrmParseError::MissingField("VRMC_vrm.humanoid.humanBones".into()))?;
    let mut human_bones = BTreeMap::new();
    for (name, bone) in bones {
        let node = required_usize(
            bone,
            "node",
            &format!("VRMC_vrm.humanoid.humanBones.{name}.node"),
        )?;
        human_bones.insert(name.clone(), node);
    }
    require_humanoid_bones(&human_bones)?;
    Ok(VrmHumanoid { human_bones })
}

fn require_humanoid_bones(bones: &BTreeMap<String, usize>) -> Result<(), VrmParseError> {
    for required in ["hips", "head"] {
        if !bones.contains_key(required) {
            return Err(VrmParseError::MissingField(format!(
                "humanoid.humanBones.{required}"
            )));
        }
    }
    Ok(())
}

fn parse_vrm0_first_person(
    root: &Value,
    value: &Value,
) -> Result<VrmFirstPerson, VrmParseError> {
    let first_person_bone = value
        .get("firstPersonBone")
        .map(|_| {
            let index =
                required_usize(value, "firstPersonBone", "VRM.firstPerson.firstPersonBone")?;
            validate_node_index(root, index, "VRM.firstPerson.firstPersonBone")
        })
        .transpose()?;
    let first_person_bone_offset = value
        .get("firstPersonBoneOffset")
        .map(|offset| array3_object(offset, "VRM.firstPerson.firstPersonBoneOffset"))
        .transpose()?;
    let mut mesh_annotations = Vec::new();
    let mut seen_nodes = std::collections::BTreeSet::new();
    if let Some(raw_annotations) = value.get("meshAnnotations") {
        let annotations = raw_annotations
            .as_array()
            .ok_or_else(|| VrmParseError::InvalidField {
                path: "VRM.firstPerson.meshAnnotations".into(),
                reason: "expected an array".into(),
            })?;
        for (index, annotation) in annotations.iter().enumerate() {
            let path = format!("VRM.firstPerson.meshAnnotations[{index}]");
            let mesh = required_usize(annotation, "mesh", &format!("{path}.mesh"))?;
            let flag = parse_first_person_flag(
                annotation.get("firstPersonFlag"),
                &format!("{path}.firstPersonFlag"),
            )?;
            for node in mesh_instance_nodes(root, mesh, &format!("{path}.mesh"))? {
                if seen_nodes.insert(node) {
                    mesh_annotations.push(VrmMeshAnnotation { node, flag });
                }
            }
        }
    }
    Ok(VrmFirstPerson {
        first_person_bone,
        first_person_bone_offset,
        mesh_annotations,
    })
}

fn parse_vrm1_first_person(value: &Value) -> Result<VrmFirstPerson, VrmParseError> {
    let mut mesh_annotations = Vec::new();
    if let Some(annotations) = value.get("meshAnnotations").and_then(Value::as_array) {
        for (index, annotation) in annotations.iter().enumerate() {
            let path = format!("VRMC_vrm.firstPerson.meshAnnotations[{index}]");
            let node = required_usize(annotation, "node", &format!("{path}.node"))?;
            let flag = parse_first_person_flag(
                annotation.get("firstPersonFlag"),
                &format!("{path}.firstPersonFlag"),
            )?;
            mesh_annotations.push(VrmMeshAnnotation { node, flag });
        }
    }
    Ok(VrmFirstPerson {
        first_person_bone: None,
        first_person_bone_offset: None,
        mesh_annotations,
    })
}

fn parse_first_person_flag(
    value: Option<&Value>,
    path: &str,
) -> Result<VrmFirstPersonFlag, VrmParseError> {
    let value = value
        .and_then(Value::as_str)
        .ok_or_else(|| VrmParseError::MissingField(path.into()))?;
    match value {
        "Auto" | "auto" => Ok(VrmFirstPersonFlag::Auto),
        "Both" | "both" => Ok(VrmFirstPersonFlag::Both),
        "ThirdPersonOnly" | "thirdPersonOnly" | "third_person_only" => {
            Ok(VrmFirstPersonFlag::ThirdPersonOnly)
        }
        "FirstPersonOnly" | "firstPersonOnly" | "first_person_only" => {
            Ok(VrmFirstPersonFlag::FirstPersonOnly)
        }
        _ => Err(VrmParseError::InvalidField {
            path: path.into(),
            reason: format!("unknown first-person flag {value}"),
        }),
    }
}

fn parse_vrm0_look_at(value: &Value) -> Result<VrmLookAt, VrmParseError> {
    let has_look_at = value.get("lookAtTypeName").is_some()
        || [
            "lookAtHorizontalInner",
            "lookAtHorizontalOuter",
            "lookAtVerticalDown",
            "lookAtVerticalUp",
        ]
        .iter()
        .any(|field| value.get(*field).is_some());
    if !has_look_at {
        return Err(VrmParseError::MissingField(
            "VRM.firstPerson.lookAtTypeName".into(),
        ));
    }
    let r#type = match required_string(value, "lookAtTypeName")?.as_str() {
        "Bone" => VrmLookAtType::Bone,
        "BlendShape" => VrmLookAtType::Expression,
        other => {
            return Err(VrmParseError::InvalidField {
                path: "VRM.firstPerson.lookAtTypeName".into(),
                reason: format!("unknown look-at type {other}"),
            });
        }
    };
    Ok(VrmLookAt {
        r#type,
        offset_from_head_bone: array3_object(
            value
                .get("firstPersonBoneOffset")
                .ok_or_else(|| {
                    VrmParseError::MissingField(
                        "VRM.firstPerson.firstPersonBoneOffset".into(),
                    )
                })?,
            "VRM.firstPerson.firstPersonBoneOffset",
        )?,
        range_map_horizontal_inner: parse_vrm0_range_map(
            value,
            "lookAtHorizontalInner",
            "VRM.firstPerson.lookAtHorizontalInner",
        )?,
        range_map_horizontal_outer: parse_vrm0_range_map(
            value,
            "lookAtHorizontalOuter",
            "VRM.firstPerson.lookAtHorizontalOuter",
        )?,
        range_map_vertical_down: parse_vrm0_range_map(
            value,
            "lookAtVerticalDown",
            "VRM.firstPerson.lookAtVerticalDown",
        )?,
        range_map_vertical_up: parse_vrm0_range_map(
            value,
            "lookAtVerticalUp",
            "VRM.firstPerson.lookAtVerticalUp",
        )?,
    })
}

fn parse_vrm0_range_map(
    value: &Value,
    field: &str,
    path: &str,
) -> Result<VrmRangeMap, VrmParseError> {
    let range = value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| VrmParseError::MissingField(path.into()))?;
    if let Some(curve) = range.get("curve") {
        let values = curve
            .as_array()
            .ok_or_else(|| VrmParseError::InvalidField {
                path: format!("{path}.curve"),
                reason: "expected an array of curve coefficients".into(),
            })?;
        if values
            .iter()
            .any(|value| value.as_f64().is_none_or(|value| !value.is_finite()))
        {
            return Err(VrmParseError::InvalidField {
                path: format!("{path}.curve"),
                reason: "curve coefficients must be finite numbers".into(),
            });
        }
        if values.len() % 4 != 0 {
            return Err(VrmParseError::InvalidField {
                path: format!("{path}.curve"),
                reason: "curve must contain groups of time, value, inTangent, outTangent".into(),
            });
        }
    }
    let mut input_max_value = required_f32(
            &Value::Object(range.clone()),
            "xRange",
            &format!("{path}.xRange"),
        )?;
    let output_scale = required_f32(
            &Value::Object(range.clone()),
            "yRange",
            &format!("{path}.yRange"),
        )?;
    // UniVRM's fixed reference maps xRange == 0 to its default 90 degree
    // input range, permits yRange == 0, and rejects negative ranges. The
    // current runtime is linear, so any source curve remains a typed warning.
    if input_max_value < 0.0 {
        return Err(VrmParseError::InvalidField {
            path: format!("{path}.xRange"),
            reason: "expected zero or a positive degree range".into(),
        });
    }
    if output_scale < 0.0 {
        return Err(VrmParseError::InvalidField {
            path: format!("{path}.yRange"),
            reason: "expected a zero or positive degree range".into(),
        });
    }
    if input_max_value == 0.0 {
        input_max_value = 90.0;
    }
    Ok(VrmRangeMap {
        input_max_value,
        output_scale,
    })
}

fn parse_vrm1_look_at(value: &Value) -> Result<VrmLookAt, VrmParseError> {
    let r#type = match required_string(value, "type")?.as_str() {
        "bone" | "Bone" => VrmLookAtType::Bone,
        "expression" | "Expression" => VrmLookAtType::Expression,
        other => {
            return Err(VrmParseError::InvalidField {
                path: "VRMC_vrm.lookAt.type".into(),
                reason: format!("unknown look-at type {other}"),
            });
        }
    };
    Ok(VrmLookAt {
        r#type,
        offset_from_head_bone: array3(value, "offsetFromHeadBone", "VRMC_vrm.lookAt")?,
        range_map_horizontal_inner: parse_vrm1_range_map(
            value,
            "rangeMapHorizontalInner",
            "VRMC_vrm.lookAt.rangeMapHorizontalInner",
        )?,
        range_map_horizontal_outer: parse_vrm1_range_map(
            value,
            "rangeMapHorizontalOuter",
            "VRMC_vrm.lookAt.rangeMapHorizontalOuter",
        )?,
        range_map_vertical_down: parse_vrm1_range_map(
            value,
            "rangeMapVerticalDown",
            "VRMC_vrm.lookAt.rangeMapVerticalDown",
        )?,
        range_map_vertical_up: parse_vrm1_range_map(
            value,
            "rangeMapVerticalUp",
            "VRMC_vrm.lookAt.rangeMapVerticalUp",
        )?,
    })
}

fn parse_vrm1_range_map(
    value: &Value,
    field: &str,
    path: &str,
) -> Result<VrmRangeMap, VrmParseError> {
    let range = value
        .get(field)
        .ok_or_else(|| VrmParseError::MissingField(path.into()))?;
    Ok(VrmRangeMap {
        input_max_value: required_f32(range, "inputMaxValue", &format!("{path}.inputMaxValue"))?,
        output_scale: required_f32(range, "outputScale", &format!("{path}.outputScale"))?,
    })
}

fn string_field(
    values: &Map<String, Value>,
    field: &str,
) -> Option<String> {
    values.get(field).and_then(Value::as_str).map(String::from)
}

fn required_string(
    value: &Value,
    field: &str,
) -> Result<String, VrmParseError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(String::from)
        .ok_or_else(|| VrmParseError::MissingField(field.into()))
}

fn required_usize(
    value: &Value,
    field: &str,
    path: &str,
) -> Result<usize, VrmParseError> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| VrmParseError::MissingField(path.into()))
}

fn required_f32(
    value: &Value,
    field: &str,
    path: &str,
) -> Result<f32, VrmParseError> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
        .ok_or_else(|| VrmParseError::InvalidField {
            path: path.into(),
            reason: "expected a finite number".into(),
        })
}

fn array3_object(
    value: &Value,
    path: &str,
) -> Result<[f32; 3], VrmParseError> {
    let object = value
        .as_object()
        .ok_or_else(|| VrmParseError::InvalidField {
            path: path.into(),
            reason: "expected an object with x, y, z".into(),
        })?;
    let mut result = [0.0; 3];
    for (index, field) in ["x", "y", "z"].into_iter().enumerate() {
        result[index] = object
            .get(field)
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .ok_or_else(|| VrmParseError::InvalidField {
                path: format!("{path}.{field}"),
                reason: "expected a finite number".into(),
            })?;
    }
    Ok(result)
}

fn validate_node_index(
    root: &Value,
    index: usize,
    path: &str,
) -> Result<usize, VrmParseError> {
    let count = root
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| VrmParseError::MissingField("nodes".into()))?
        .len();
    (index < count)
        .then_some(index)
        .ok_or_else(|| VrmParseError::InvalidIndex {
            path: path.into(),
            index,
        })
}

fn mesh_instance_nodes(
    root: &Value,
    mesh: usize,
    path: &str,
) -> Result<Vec<usize>, VrmParseError> {
    let meshes = root
        .get("meshes")
        .and_then(Value::as_array)
        .ok_or_else(|| VrmParseError::MissingField("meshes".into()))?;
    meshes
        .get(mesh)
        .ok_or_else(|| VrmParseError::InvalidIndex {
            path: path.into(),
            index: mesh,
        })?;
    let nodes = root
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| VrmParseError::MissingField("nodes".into()))?;
    let instances = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            (node.get("mesh").and_then(Value::as_u64) == Some(mesh as u64)).then_some(index)
        })
        .collect::<Vec<_>>();
    Ok(instances)
}

fn validate_morph_target_index(
    root: &Value,
    mesh: usize,
    morph_index: usize,
    path: &str,
) -> Result<(), VrmParseError> {
    let meshes = root
        .get("meshes")
        .and_then(Value::as_array)
        .ok_or_else(|| VrmParseError::MissingField("meshes".into()))?;
    let mesh_value = meshes
        .get(mesh)
        .ok_or_else(|| VrmParseError::InvalidIndex {
            path: path.into(),
            index: mesh,
        })?;
    let count = mesh_value
        .get("primitives")
        .and_then(Value::as_array)
        .and_then(|primitives| {
            primitives
                .iter()
                .filter_map(|primitive| primitive.get("targets").and_then(Value::as_array))
                .map(Vec::len)
                .max()
        })
        .unwrap_or(0);
    (morph_index < count)
        .then_some(())
        .ok_or_else(|| VrmParseError::InvalidIndex {
            path: path.into(),
            index: morph_index,
        })
}

fn array3(
    value: &Value,
    field: &str,
    parent_path: &str,
) -> Result<[f32; 3], VrmParseError> {
    let values = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| VrmParseError::MissingField(format!("{parent_path}.{field}")))?;
    if values.len() != 3 {
        return Err(VrmParseError::InvalidField {
            path: format!("{parent_path}.{field}"),
            reason: "expected three numbers".into(),
        });
    }
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_f64()
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .ok_or_else(|| VrmParseError::InvalidField {
                path: format!("{parent_path}.{field}[{index}]"),
                reason: "expected a finite number".into(),
            })?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn vrm0_root() -> Value {
        json!({
            "nodes": [{}, {}, {"mesh": 2}, {"mesh": 2}],
            "meshes": [{}, {}, {"primitives": [{}]}],
            "extensions": {"VRM": {
                "exporterVersion": "UniVRM top-level",
                "meta": {"title": "legacy", "author": "author", "exporterVersion": "nonstandard-meta", "otherLicenseUrl": "https://example.test/license"},
                "humanoid": {"humanBones": [
                    {"bone": "hips", "node": 0},
                    {"bone": "head", "node": 1}
                ]},
                "firstPerson": {"firstPersonBone": 1, "meshAnnotations": [
                    {"mesh": 2, "firstPersonFlag": "ThirdPersonOnly"}
                ],
                "lookAtTypeName": "BlendShape",
                "firstPersonBoneOffset": {"x": 0.0, "y": 0.1, "z": 0.2},
                "lookAtHorizontalInner": {"curve": [0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 0.0], "xRange": 90.0, "yRange": 10.0},
                "lookAtHorizontalOuter": {"xRange": 90.0, "yRange": 10.0},
                "lookAtVerticalDown": {"xRange": 90.0, "yRange": 10.0},
                "lookAtVerticalUp": {"xRange": 90.0, "yRange": 10.0}
                }
            }}
        })
    }

    fn vrm1_root() -> Value {
        json!({
            "extensions": {"VRMC_vrm": {
                "specVersion": "1.0",
                "meta": {"name": "modern", "authors": ["author"]},
                "humanoid": {"humanBones": {
                    "hips": {"node": 0}, "head": {"node": 1}
                }},
                "firstPerson": {"meshAnnotations": [
                    {"node": 2, "firstPersonFlag": "thirdPersonOnly"}
                ]},
                "lookAt": {
                    "type": "bone",
                    "offsetFromHeadBone": [0.0, 0.1, 0.2],
                    "rangeMapHorizontalInner": {"inputMaxValue": 90.0, "outputScale": 10.0},
                    "rangeMapHorizontalOuter": {"inputMaxValue": 90.0, "outputScale": 10.0},
                    "rangeMapVerticalDown": {"inputMaxValue": 90.0, "outputScale": 10.0},
                    "rangeMapVerticalUp": {"inputMaxValue": 90.0, "outputScale": 10.0}
                }
            }}
        })
    }

    #[test]
    fn parses_legacy_core_into_common_descriptor() {
        let descriptor = parse_runtime_descriptor(&vrm0_root()).expect("VRM 0.x should parse");
        assert_eq!(descriptor.generation, VrmGeneration::Vrm0);
        assert_eq!(descriptor.coordinate_basis, CoordinateBasis::Vrm0Y180);
        assert_eq!(descriptor.meta.name.as_deref(), Some("legacy"));
        assert_eq!(
            descriptor.legacy_meta.as_ref().unwrap().exporter_version.as_deref(),
            Some("UniVRM top-level")
        );
        assert_eq!(descriptor.humanoid.human_bones["head"], 1);
        assert_eq!(
            descriptor.first_person.as_ref().unwrap().first_person_bone,
            Some(1)
        );
        assert_eq!(
            descriptor.first_person.as_ref().unwrap().mesh_annotations[0].flag,
            VrmFirstPersonFlag::ThirdPersonOnly
        );
        assert_eq!(
            descriptor
                .first_person
                .as_ref()
                .unwrap()
                .mesh_annotations
                .iter()
                .map(|annotation| annotation.node)
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(
            descriptor.look_at.as_ref().unwrap().r#type,
            VrmLookAtType::Expression
        );
        assert_eq!(
            descriptor
                .look_at
                .as_ref()
                .unwrap()
                .range_map_horizontal_inner
                .output_scale,
            10.0
        );
        assert_eq!(
            descriptor
                .first_person
                .as_ref()
                .unwrap()
                .first_person_bone_offset,
            Some([0.0, 0.1, 0.2])
        );
        assert_eq!(
            descriptor.legacy_meta.as_ref().unwrap().other_license_url.as_deref(),
            Some("https://example.test/license")
        );
        assert!(descriptor.compatibility_warnings.iter().any(|warning| {
            warning.code == VrmCompatibilityWarningCode::NonLinearLegacyLookAtCurve
        }));
    }

    #[test]
    fn parses_modern_core_into_the_same_descriptor_shape() {
        let descriptor = parse_runtime_descriptor(&vrm1_root()).expect("VRM 1.0 should parse");
        assert_eq!(descriptor.generation, VrmGeneration::Vrm1);
        assert_eq!(descriptor.coordinate_basis, CoordinateBasis::Vrm1Identity);
        assert_eq!(descriptor.spec_version, "1.0");
        assert_eq!(descriptor.meta.authors, vec!["author"]);
        assert_eq!(
            descriptor.first_person.as_ref().unwrap().mesh_annotations[0].node,
            2
        );
        assert_eq!(
            descriptor.look_at.as_ref().unwrap().r#type,
            VrmLookAtType::Bone
        );
    }

    #[test]
    fn rejects_ambiguous_and_unsupported_roots() {
        let mut ambiguous = vrm1_root();
        ambiguous["extensions"]["VRM"] = json!({});
        assert_eq!(
            parse_runtime_descriptor(&ambiguous),
            Err(VrmParseError::AmbiguousGeneration)
        );

        let mut unsupported = vrm1_root();
        unsupported["extensions"]["VRMC_vrm"]["specVersion"] = json!("1.1");
        assert_eq!(
            parse_runtime_descriptor(&unsupported),
            Err(VrmParseError::UnsupportedVersion("1.1".into()))
        );
    }

    #[test]
    fn rejects_legacy_duplicate_required_bones() {
        let mut root = vrm0_root();
        root["extensions"]["VRM"]["humanoid"]["humanBones"]
            .as_array_mut()
            .unwrap()
            .push(json!({"bone": "head", "node": 3}));
        assert_eq!(
            parse_runtime_descriptor(&root),
            Err(VrmParseError::DuplicateBone("head".into()))
        );
    }

    #[test]
    fn rejects_legacy_mesh_annotation_out_of_range() {
        let mut root = vrm0_root();
        root["extensions"]["VRM"]["firstPerson"]["meshAnnotations"][0]["mesh"] = json!(99);
        assert!(matches!(
            parse_runtime_descriptor(&root),
            Err(VrmParseError::InvalidIndex { .. })
        ));
    }

    #[test]
    fn rejects_legacy_look_at_without_official_bone_offset() {
        let mut root = vrm0_root();
        root["extensions"]["VRM"]["firstPerson"]
            .as_object_mut()
            .unwrap()
            .remove("firstPersonBoneOffset");
        assert!(matches!(
            parse_runtime_descriptor(&root),
            Err(VrmParseError::MissingField(path))
                if path == "VRM.firstPerson.firstPersonBoneOffset"
        ));
    }

    #[test]
    fn parses_legacy_bone_look_at() {
        let mut root = vrm0_root();
        root["extensions"]["VRM"]["firstPerson"]["lookAtTypeName"] = json!("Bone");
        let descriptor = parse_runtime_descriptor(&root).expect("Bone LookAt should parse");
        assert_eq!(
            descriptor.look_at.as_ref().unwrap().r#type,
            VrmLookAtType::Bone
        );
    }

    #[test]
    fn accepts_legacy_first_person_without_look_at() {
        let mut root = vrm0_root();
        let first_person = root["extensions"]["VRM"]["firstPerson"]
            .as_object_mut()
            .unwrap();
        for field in [
            "lookAtTypeName",
            "firstPersonBoneOffset",
            "lookAtHorizontalInner",
            "lookAtHorizontalOuter",
            "lookAtVerticalDown",
            "lookAtVerticalUp",
        ] {
            first_person.remove(field);
        }
        let descriptor = parse_runtime_descriptor(&root).expect("LookAt is optional");
        assert!(descriptor.look_at.is_none());
    }

    #[test]
    fn rejects_legacy_malformed_degree_map_curve() {
        let mut root = vrm0_root();
        root["extensions"]["VRM"]["firstPerson"]["lookAtHorizontalInner"]["curve"] =
            json!("not-an-array");
        assert!(matches!(
            parse_runtime_descriptor(&root),
            Err(VrmParseError::InvalidField { path, .. })
                if path.ends_with("lookAtHorizontalInner.curve")
        ));
    }

    #[test]
    fn rejects_legacy_degree_map_curve_with_malformed_length() {
        let mut root = vrm0_root();
        root["extensions"]["VRM"]["firstPerson"]["lookAtHorizontalInner"]["curve"] =
            json!([0.0, 0.0, 0.0, 1.0, 1.0]);
        assert!(matches!(
            parse_runtime_descriptor(&root),
            Err(VrmParseError::InvalidField { path, .. })
                if path.ends_with("lookAtHorizontalInner.curve")
        ));
    }

    #[test]
    fn linear_legacy_degree_map_curves_do_not_warn() {
        for curve in [
            json!([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            json!([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        ] {
            let mut root = vrm0_root();
            root["extensions"]["VRM"]["firstPerson"]["lookAtHorizontalInner"]["curve"] =
                curve;
            let warnings = collect_legacy_compatibility_warnings(
                &root,
                &root["extensions"]["VRM"],
            );
            assert!(!warnings.iter().any(|warning| {
                warning.code == VrmCompatibilityWarningCode::NonLinearLegacyLookAtCurve
            }));
        }
    }

    #[test]
    fn absent_legacy_degree_map_curve_uses_default_without_warning() {
        let mut root = vrm0_root();
        root["extensions"]["VRM"]["firstPerson"]["lookAtHorizontalInner"]
            .as_object_mut()
            .unwrap()
            .remove("curve");
        let warnings = collect_legacy_compatibility_warnings(
            &root,
            &root["extensions"]["VRM"],
        );
        assert!(!warnings.iter().any(|warning| {
            warning.code == VrmCompatibilityWarningCode::NonLinearLegacyLookAtCurve
        }));
    }

    #[test]
    fn empty_one_key_and_partial_legacy_degree_maps_warn() {
        for curve in [
            json!([]),
            json!([0.0, 0.0, 0.0, 0.0]),
            json!([0.2, 0.2, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0]),
        ] {
            let mut root = vrm0_root();
            root["extensions"]["VRM"]["firstPerson"]["lookAtHorizontalInner"]["curve"] =
                curve;
            let warnings = collect_legacy_compatibility_warnings(
                &root,
                &root["extensions"]["VRM"],
            );
            assert!(warnings.iter().any(|warning| {
                warning.code == VrmCompatibilityWarningCode::NonLinearLegacyLookAtCurve
            }));
        }
    }

    #[test]
    fn nonlinear_legacy_degree_map_curve_warns() {
        let mut root = vrm0_root();
        root["extensions"]["VRM"]["firstPerson"]["lookAtHorizontalInner"]["curve"] =
            json!([0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 0.0]);
        let warnings = collect_legacy_compatibility_warnings(
            &root,
            &root["extensions"]["VRM"],
        );
        assert_eq!(
            warnings
                .iter()
                .filter(|warning| {
                    warning.code == VrmCompatibilityWarningCode::NonLinearLegacyLookAtCurve
                })
                .count(),
            1
        );
    }

    #[test]
    fn official_legacy_mtoon_property_set_is_not_unknown() {
        let mut root = vrm0_root();
        root["materials"] = json!([{}]);
        let legacy = &mut root["extensions"]["VRM"];
        legacy["materialProperties"] = json!([{
            "shader": "VRM/MToon",
            "floatProperties": {
                "_Cutoff": 0.5, "_BumpScale": 1.0, "_ReceiveShadowRate": 1.0,
                "_ShadingGradeRate": 1.0, "_ShadeShift": 0.0, "_ShadeToony": 0.9,
                "_LightColorAttenuation": 0.5, "_IndirectLightIntensity": 0.5,
                "_RimLightingMix": 1.0, "_RimFresnelPower": 5.0, "_RimLift": 0.0,
                "_UvAnimScrollX": 0.0, "_UvAnimScrollY": 0.0, "_UvAnimRotation": 0.0,
                "_MToonVersion": 30.0, "_DebugMode": 0.0, "_BlendMode": 0.0,
                "_OutlineWidthMode": 0.0, "_OutlineColorMode": 0.0, "_CullMode": 2.0,
                "_OutlineCullMode": 1.0, "_SrcBlend": 1.0, "_DstBlend": 0.0, "_ZWrite": 1.0
            },
            "vectorProperties": {
                "_Color": [1.0, 1.0, 1.0, 1.0], "_ShadeColor": [0.5, 0.5, 0.5, 1.0],
                "_MainTex": [0.0, 0.0, 1.0, 1.0], "_ShadeTexture": [0.0, 0.0, 1.0, 1.0],
                "_BumpMap": [0.0, 0.0, 1.0, 1.0], "_ReceiveShadowTexture": [0.0, 0.0, 1.0, 1.0],
                "_ShadingGradeTexture": [0.0, 0.0, 1.0, 1.0], "_RimColor": [1.0, 1.0, 1.0, 1.0],
                "_RimTexture": [0.0, 0.0, 1.0, 1.0], "_EmissionColor": [0.0, 0.0, 0.0, 1.0],
                "_EmissionMap": [0.0, 0.0, 1.0, 1.0], "_OutlineWidthTexture": [0.0, 0.0, 1.0, 1.0],
                "_OutlineColor": [0.0, 0.0, 0.0, 1.0], "_UvAnimMaskTexture": [0.0, 0.0, 1.0, 1.0]
            },
            "textureProperties": {
                "_MainTex": 0, "_ShadeTexture": 0, "_BumpMap": 0,
                "_ReceiveShadowTexture": 0, "_ShadingGradeTexture": 0,
                "_EmissionMap": 0, "_RimTexture": 0, "_SphereAdd": 0,
                "_OutlineWidthTexture": 0, "_UvAnimMaskTexture": 0
            }
        }]);
        let legacy = root["extensions"]["VRM"].clone();
        let warnings = collect_legacy_compatibility_warnings(&root, &legacy);
        assert_eq!(
            warnings
                .iter()
                .filter(|warning| {
                    warning.code == VrmCompatibilityWarningCode::UnknownLegacyMaterialProperty
                })
                .count(),
            0
        );
    }

    #[test]
    fn legacy_shader_classification_requires_exact_names() {
        assert_eq!(classify_legacy_shader("VRM/MToon"), LegacyShaderKind::MToon);
        assert_eq!(
            classify_legacy_shader("VRM/UnlitTransparentZWrite"),
            LegacyShaderKind::SupportedUnlit
        );
        for shader in ["VRM_USE_GLTFSHADER", "Standard", "UniGLTF/UniUnlit"] {
            assert_eq!(classify_legacy_shader(shader), LegacyShaderKind::Passthrough);
        }
        for shader in ["Custom/MToon", "MyMToonShader", "VRM/MToonExtra"] {
            assert_eq!(classify_legacy_shader(shader), LegacyShaderKind::Unknown);
        }
    }

    #[test]
    fn custom_mtoon_shader_is_an_unknown_warning_not_a_mtoon_diagnostic() {
        let mut root = vrm0_root();
        root["materials"] = json!([{}]);
        root["extensions"]["VRM"]["materialProperties"] = json!([
            {"shader": "Custom/MToon"},
            {"shader": "MyMToonShader"},
            {"shader": "VRM/MToonExtra"}
        ]);
        let warnings = collect_legacy_compatibility_warnings(
            &root,
            &root["extensions"]["VRM"],
        );
        let contexts = warnings
            .iter()
            .filter(|warning| warning.code == VrmCompatibilityWarningCode::UnknownLegacyShader)
            .map(|warning| warning.context.as_str())
            .collect::<Vec<_>>();
        assert_eq!(contexts.len(), 3);
        assert!(contexts.iter().all(|context| context.contains("shader=")));
    }

    #[test]
    fn compatibility_warnings_are_typed_bounded_and_deterministic() {
        let mut root = vrm0_root();
        root["materials"] = json!([{}]);
        root["extensions"]["VRM"]["materialProperties"] = json!([
            {"shader": "VRM/UnlitTransparentZWrite", "floatProperties": {"_Unknown": 1.0}},
            {"shader": "Unknown/Shader"}
        ]);
        root["extensions"]["VRM"]["blendShapeMaster"] = json!({
            "blendShapeGroups": [
                {"presetName": "Blink", "name": "first", "binds": []},
                {"presetName": " blink ", "name": "second", "materialValues": []},
                {"presetName": "Unknown", "name": "   "}
            ]
        });
        let warnings = collect_legacy_compatibility_warnings(
            &root,
            &root["extensions"]["VRM"],
        );
        assert!(warnings.iter().any(|warning| {
            warning.code == VrmCompatibilityWarningCode::ExtraLegacyMaterialProperty
        }));
        assert!(warnings.iter().any(|warning| {
            warning.code == VrmCompatibilityWarningCode::DuplicateLegacyExpression
        }));
        assert!(warnings.iter().any(|warning| {
            warning.code == VrmCompatibilityWarningCode::EmptyLegacyExpressionName
        }));
        assert!(warnings.iter().any(|warning| {
            warning.code == VrmCompatibilityWarningCode::UnlitZWriteNotRepresentable
        }));
        assert!(warnings.iter().all(|warning| {
            warning.context.chars().count() <= MAX_COMPATIBILITY_WARNING_CONTEXT
        }));
    }

    #[test]
    fn invalid_legacy_texture_index_emits_typed_warning() {
        let mut root = vrm0_root();
        root["textures"] = json!([{}]);
        root["extensions"]["VRM"]["materialProperties"] = json!([{
            "shader": "VRM/MToon",
            "textureProperties": {
                "_MainTex": 1,
                "_ShadeTexture": -1,
                "_RimTexture": "not-an-index",
                "_MatcapTexture": 0.5,
                "_EmissionMap": 0
            }
        }]);

        let warnings = collect_legacy_compatibility_warnings(
            &root,
            &root["extensions"]["VRM"],
        );
        let invalid = warnings
            .iter()
            .filter(|warning| {
                warning.code == VrmCompatibilityWarningCode::InvalidLegacyMaterialTexture
            })
            .collect::<Vec<_>>();
        assert_eq!(invalid.len(), 4);
        assert!(invalid.iter().any(|warning| {
            warning.context.contains("textureProperties._MainTex=1")
        }));
        assert!(invalid.iter().any(|warning| {
            warning.context.contains("textureProperties._ShadeTexture=-1")
        }));
        assert!(invalid.iter().any(|warning| {
            warning.context.contains("textureProperties._RimTexture=\"not-an-index\"")
        }));
        assert!(invalid.iter().any(|warning| {
            warning.context.contains("textureProperties._MatcapTexture=0.5")
        }));

        let descriptor = parse_runtime_descriptor(&root).expect("VRM 0.x should parse");
        assert_eq!(descriptor.compatibility_warnings, warnings);
    }
}
