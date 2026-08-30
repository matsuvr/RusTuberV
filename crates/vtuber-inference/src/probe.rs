//! Model provenance probing for G0-05.

use std::path::Path;

use tract_tflite::prelude::*;

/// Result of loading a TFLite model and inspecting its inputs/outputs.
#[derive(Debug, Clone)]
pub struct ModelProbe {
    /// Model file path.
    pub path: String,
    /// SHA-256 hex of the model file.
    pub sha256: String,
    /// Input fact descriptors.
    pub inputs: Vec<IOFact>,
    /// Output fact descriptors.
    pub outputs: Vec<IOFact>,
}

/// Input or output tensor descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct IOFact {
    /// Tensor name from the model.
    pub name: String,
    /// Datum type string.
    pub dtype: String,
    /// Shape dimensions.
    pub shape: Vec<i64>,
}

/// Probe a TFLite model at `path` and return its input/output facts.
pub fn probe_tflite_model(path: impl AsRef<Path>) -> anyhow::Result<ModelProbe> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let sha256 = sha256_hex(&bytes);

    let model = tract_tflite::tflite()
        .model_for_read(&mut &bytes[..])
        .map_err(|e| anyhow::anyhow!("failed to load model: {e:?}"))?;

    let inputs = model
        .input_outlets()
        .map_err(|e| anyhow::anyhow!("failed to get inputs: {e}"))?
        .iter()
        .map(|outlet| {
            let fact = model
                .outlet_fact(*outlet)
                .map_err(|e| anyhow::anyhow!("failed to get input fact: {e}"))?;
            Ok(IOFact {
                name: model
                    .node_names()
                    .nth(outlet.node)
                    .unwrap_or("")
                    .to_string(),
                dtype: format!("{:?}", fact.datum_type),
                shape: fact
                    .shape
                    .as_concrete()
                    .map(|dims| dims.iter().map(|&d| d as i64).collect())
                    .unwrap_or_default(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let outputs = model
        .output_outlets()
        .map_err(|e| anyhow::anyhow!("failed to get outputs: {e}"))?
        .iter()
        .map(|outlet| {
            let fact = model
                .outlet_fact(*outlet)
                .map_err(|e| anyhow::anyhow!("failed to get output fact: {e}"))?;
            Ok(IOFact {
                name: model
                    .node_names()
                    .nth(outlet.node)
                    .unwrap_or("")
                    .to_string(),
                dtype: format!("{:?}", fact.datum_type),
                shape: fact
                    .shape
                    .as_concrete()
                    .map(|dims| dims.iter().map(|&d| d as i64).collect())
                    .unwrap_or_default(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(ModelProbe {
        path: path.to_string_lossy().to_string(),
        sha256,
        inputs,
        outputs,
    })
}

/// Probe an ONNX model at `path` and return its input/output facts.
#[cfg(feature = "onnx")]
pub fn probe_onnx_model(path: impl AsRef<Path>) -> anyhow::Result<ModelProbe> {
    use tract_onnx::prelude::*;

    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let sha256 = sha256_hex(&bytes);

    let model = tract_onnx::onnx()
        .model_for_read(&mut &bytes[..])
        .map_err(|e| anyhow::anyhow!("failed to load onnx model: {e:?}"))?
        .into_optimized()
        .map_err(|e| anyhow::anyhow!("failed to optimize onnx model: {e:?}"))?;

    let inputs = model
        .input_outlets()
        .map_err(|e| anyhow::anyhow!("failed to get inputs: {e}"))?
        .iter()
        .map(|outlet| {
            let fact = model
                .outlet_fact(*outlet)
                .map_err(|e| anyhow::anyhow!("failed to get input fact: {e}"))?;
            Ok(IOFact {
                name: model
                    .node_names()
                    .nth(outlet.node)
                    .unwrap_or("")
                    .to_string(),
                dtype: format!("{:?}", fact.datum_type),
                shape: fact
                    .shape
                    .as_concrete()
                    .map(|dims| dims.iter().map(|&d| d as i64).collect())
                    .unwrap_or_default(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let outputs = model
        .output_outlets()
        .map_err(|e| anyhow::anyhow!("failed to get outputs: {e}"))?
        .iter()
        .map(|outlet| {
            let fact = model
                .outlet_fact(*outlet)
                .map_err(|e| anyhow::anyhow!("failed to get output fact: {e}"))?;
            Ok(IOFact {
                name: model
                    .node_names()
                    .nth(outlet.node)
                    .unwrap_or("")
                    .to_string(),
                dtype: format!("{:?}", fact.datum_type),
                shape: fact
                    .shape
                    .as_concrete()
                    .map(|dims| dims.iter().map(|&d| d as i64).collect())
                    .unwrap_or_default(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(ModelProbe {
        path: path.to_string_lossy().to_string(),
        sha256,
        inputs,
        outputs,
    })
}

/// Stable identity of the only detector accepted by M1-08-013-002.
#[cfg(feature = "onnx")]
pub const ULTRAFACE_RFB_320_MODEL_ID: &str = "ultraface-rfb-320";

/// Fixed SHA-256 of the ONNX Model Zoo UltraFace RFB-320 artifact.
#[cfg(feature = "onnx")]
pub const ULTRAFACE_RFB_320_SHA256: &str =
    "34CD7E60AEFF28744C657DE7A3DC64E872D506741DE66987F3426F2B79F88017";

/// Fixed byte size of the ONNX Model Zoo UltraFace RFB-320 artifact.
#[cfg(feature = "onnx")]
pub const ULTRAFACE_RFB_320_BYTE_SIZE: usize = 1_270_727;

/// Fixed model input shape for UltraFace RFB-320.
#[cfg(feature = "onnx")]
pub const ULTRAFACE_RFB_320_INPUT_SHAPE: [usize; 4] = [1, 3, 240, 320];

/// Stage at which the exact UltraFace probe failed.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxProbeStage {
    /// Reading or verifying the exact local artifact.
    ArtifactRead,
    /// Loading the ONNX graph through tract-onnx.
    Load,
    /// Reading or validating the model input fact.
    InputFact,
    /// Capturing the source graph's operator inventory.
    OperatorInventory,
    /// Optimizing the graph.
    Optimize,
    /// Constructing the runnable plan.
    Runnable,
    /// Executing the runnable plan.
    Run,
    /// Reading output facts or values.
    OutputFact,
    /// Validating the fixed tensor contract.
    Validation,
}

#[cfg(feature = "onnx")]
impl std::fmt::Display for OnnxProbeStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::ArtifactRead => "artifact_read",
            Self::Load => "load",
            Self::InputFact => "input_fact",
            Self::OperatorInventory => "operator_inventory",
            Self::Optimize => "optimize",
            Self::Runnable => "runnable",
            Self::Run => "run",
            Self::OutputFact => "output_fact",
            Self::Validation => "validation",
        };
        f.write_str(name)
    }
}

/// Structured exact-probe failure. The model identity is retained even when
/// loading fails so an incompatible artifact cannot be silently substituted.
#[cfg(feature = "onnx")]
#[derive(Debug, thiserror::Error)]
#[error(
    "model_id={model_id} sha256={sha256} stage={stage} node={node:?} operator={operator:?}: {detail}"
)]
pub struct OnnxProbeError {
    /// Stable manifest model ID.
    pub model_id: &'static str,
    /// Actual SHA-256 when available, otherwise the expected SHA or unknown.
    pub sha256: Box<str>,
    /// Probe stage.
    pub stage: OnnxProbeStage,
    /// Optional graph node involved in the failure.
    pub node: Option<Box<str>>,
    /// Optional operator involved in the failure.
    pub operator: Option<Box<str>>,
    /// Technical failure detail.
    pub detail: Box<str>,
}

#[cfg(feature = "onnx")]
impl OnnxProbeError {
    fn new(stage: OnnxProbeStage, sha256: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            model_id: ULTRAFACE_RFB_320_MODEL_ID,
            sha256: sha256.into().into_boxed_str(),
            stage,
            node: None,
            operator: None,
            detail: detail.into().into_boxed_str(),
        }
    }
}

/// One stable source-graph operator entry.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorInventoryEntry {
    /// Source graph node index.
    pub index: usize,
    /// Source graph node name.
    pub node: String,
    /// Debug-stable tract operator representation.
    pub operator: String,
}

/// Summary of one output tensor from one deterministic probe input.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone, PartialEq)]
pub struct TensorValueSummary {
    /// Output name from the runnable model.
    pub name: String,
    /// Output dtype.
    pub dtype: String,
    /// Actual output shape.
    pub shape: Vec<usize>,
    /// Actual number of elements.
    pub element_count: usize,
    /// Whether every output value was finite.
    pub all_finite: bool,
    /// Minimum output value.
    pub min: f32,
    /// Maximum output value.
    pub max: f32,
}

/// Summary of a deterministic zero or mean-normalized input run.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeRunSummary {
    /// Human-readable input pattern.
    pub input: String,
    /// Output summaries in model order.
    pub outputs: Vec<TensorValueSummary>,
}

/// Complete result of the exact UltraFace load/optimize/runnable/run probe.
#[cfg(feature = "onnx")]
#[derive(Debug, Clone, PartialEq)]
pub struct UltraFaceProbeReport {
    /// Stable manifest model ID.
    pub model_id: String,
    /// Local artifact path.
    pub path: String,
    /// Verified artifact size.
    pub byte_size: usize,
    /// Verified artifact SHA-256.
    pub sha256: String,
    /// Input facts from the loaded model.
    pub inputs: Vec<IOFact>,
    /// Output facts from the optimized model.
    pub outputs: Vec<IOFact>,
    /// Stable source-graph operator inventory.
    pub operators: Vec<OperatorInventoryEntry>,
    /// Deterministic run summaries.
    pub runs: Vec<ProbeRunSummary>,
}

/// Construct the exact UltraFace runnable plan for the next detector leaf.
#[cfg(feature = "onnx")]
pub fn build_ultraface_runnable(
    path: impl AsRef<Path>,
) -> Result<std::sync::Arc<tract_onnx::prelude::TypedRunnableModel>, OnnxProbeError> {
    use tract_onnx::prelude::*;

    let (bytes, sha256) = read_exact_ultraface(path.as_ref())?;
    let model = tract_onnx::onnx()
        .model_for_read(&mut &bytes[..])
        .map_err(|error| OnnxProbeError::new(OnnxProbeStage::Load, &sha256, format!("{error:?}")))?
        .with_input_fact(0, f32::fact(ULTRAFACE_RFB_320_INPUT_SHAPE).into())
        .map_err(|error| {
            OnnxProbeError::new(OnnxProbeStage::InputFact, &sha256, format!("{error:?}"))
        })?
        .into_optimized()
        .map_err(|error| {
            OnnxProbeError::new(OnnxProbeStage::Optimize, &sha256, format!("{error:?}"))
        })?
        .into_runnable()
        .map_err(|error| {
            OnnxProbeError::new(OnnxProbeStage::Runnable, &sha256, format!("{error:?}"))
        })?;

    Ok(model)
}

/// Run the exact UltraFace artifact through tract-onnx and validate its fixed
/// input/output contract. No alternate model or runtime is attempted here.
#[cfg(feature = "onnx")]
pub fn probe_ultraface_model(
    path: impl AsRef<Path>,
) -> Result<UltraFaceProbeReport, OnnxProbeError> {
    use tract_onnx::prelude::*;

    let path = path.as_ref();
    let (bytes, sha256) = read_exact_ultraface(path)?;
    let model = tract_onnx::onnx()
        .model_for_read(&mut &bytes[..])
        .map_err(|error| {
            OnnxProbeError::new(OnnxProbeStage::Load, &sha256, format!("{error:?}"))
        })?;

    let inputs = model_io_facts(&model, true, &sha256)?;
    validate_ultraface_input(&inputs, &sha256)?;
    let operators = model
        .nodes()
        .iter()
        .map(|node| OperatorInventoryEntry {
            index: node.id,
            node: node.name.clone(),
            operator: format!("{:?}", node.op),
        })
        .collect::<Vec<_>>();

    let typed_model = model
        .with_input_fact(0, f32::fact(ULTRAFACE_RFB_320_INPUT_SHAPE).into())
        .map_err(|error| {
            OnnxProbeError::new(OnnxProbeStage::InputFact, &sha256, format!("{error:?}"))
        })?
        .into_optimized()
        .map_err(|error| {
            OnnxProbeError::new(OnnxProbeStage::Optimize, &sha256, format!("{error:?}"))
        })?;
    let outputs = model_io_facts(&typed_model, false, &sha256)?;
    validate_ultraface_outputs(&outputs, &sha256)?;
    let runnable = typed_model.into_runnable().map_err(|error| {
        OnnxProbeError::new(OnnxProbeStage::Runnable, &sha256, format!("{error:?}"))
    })?;

    let runs = [("zero", 0.0_f32), ("mean_normalized", 0.0_f32)]
        .into_iter()
        .map(|(input, value)| {
            let data = vec![value; ULTRAFACE_RFB_320_INPUT_SHAPE.iter().product()];
            let tensor =
                Tensor::from_shape(&ULTRAFACE_RFB_320_INPUT_SHAPE, &data).map_err(|error| {
                    OnnxProbeError::new(
                        OnnxProbeStage::Run,
                        &sha256,
                        format!("input tensor: {error:?}"),
                    )
                })?;
            let output_values = runnable.run(tvec![tensor.into()]).map_err(|error| {
                OnnxProbeError::new(OnnxProbeStage::Run, &sha256, format!("{error:?}"))
            })?;
            let output_summaries = output_values
                .iter()
                .enumerate()
                .map(|(index, value)| summarize_output(index, value, &outputs, &sha256))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ProbeRunSummary {
                input: input.to_string(),
                outputs: output_summaries,
            })
        })
        .collect::<Result<Vec<_>, OnnxProbeError>>()?;

    Ok(UltraFaceProbeReport {
        model_id: ULTRAFACE_RFB_320_MODEL_ID.to_string(),
        path: path.to_string_lossy().to_string(),
        byte_size: bytes.len(),
        sha256,
        inputs,
        outputs,
        operators,
        runs,
    })
}

#[cfg(feature = "onnx")]
fn read_exact_ultraface(path: &Path) -> Result<(Vec<u8>, String), OnnxProbeError> {
    let bytes = std::fs::read(path).map_err(|error| {
        OnnxProbeError::new(
            OnnxProbeStage::ArtifactRead,
            "unknown",
            format!("path={} error={error}", path.display()),
        )
    })?;
    let sha256 = sha256_hex(&bytes);
    if bytes.len() != ULTRAFACE_RFB_320_BYTE_SIZE {
        return Err(OnnxProbeError::new(
            OnnxProbeStage::ArtifactRead,
            &sha256,
            format!(
                "byte_size={} expected={}",
                bytes.len(),
                ULTRAFACE_RFB_320_BYTE_SIZE
            ),
        ));
    }
    if sha256 != ULTRAFACE_RFB_320_SHA256 {
        return Err(OnnxProbeError::new(
            OnnxProbeStage::ArtifactRead,
            &sha256,
            format!("sha256={} expected={}", sha256, ULTRAFACE_RFB_320_SHA256),
        ));
    }
    Ok((bytes, sha256))
}

#[cfg(feature = "onnx")]
fn model_io_facts<F, O>(
    model: &tract_core::model::Graph<F, O>,
    inputs: bool,
    sha256: &str,
) -> Result<Vec<IOFact>, OnnxProbeError>
where
    F: tract_core::model::Fact + Clone + 'static,
    O: std::fmt::Debug
        + std::fmt::Display
        + AsRef<dyn tract_core::ops::Op>
        + AsMut<dyn tract_core::ops::Op>
        + Clone
        + 'static,
{
    let outlets = if inputs {
        model.input_outlets()
    } else {
        model.output_outlets()
    }
    .map_err(|error| {
        OnnxProbeError::new(
            if inputs {
                OnnxProbeStage::InputFact
            } else {
                OnnxProbeStage::OutputFact
            },
            sha256,
            format!("{error:?}"),
        )
    })?;

    outlets
        .iter()
        .map(|outlet| {
            let fact = model.outlet_fact(*outlet).map_err(|error| {
                OnnxProbeError::new(
                    if inputs {
                        OnnxProbeStage::InputFact
                    } else {
                        OnnxProbeStage::OutputFact
                    },
                    sha256,
                    format!("{error:?}"),
                )
            })?;
            let typed_fact = fact.to_typed_fact().map_err(|error| {
                OnnxProbeError::new(
                    if inputs {
                        OnnxProbeStage::InputFact
                    } else {
                        OnnxProbeStage::OutputFact
                    },
                    sha256,
                    format!("{error:?}"),
                )
            })?;
            Ok(IOFact {
                name: model
                    .node_names()
                    .nth(outlet.node)
                    .unwrap_or("")
                    .to_string(),
                dtype: format!("{:?}", typed_fact.datum_type),
                shape: typed_fact
                    .shape
                    .as_concrete()
                    .map(|dims| dims.iter().map(|&d| d as i64).collect())
                    .unwrap_or_default(),
            })
        })
        .collect()
}

#[cfg(feature = "onnx")]
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn validate_ultraface_input(inputs: &[IOFact], sha256: &str) -> Result<(), OnnxProbeError> {
    if inputs.len() != 1 {
        return Err(OnnxProbeError::new(
            OnnxProbeStage::Validation,
            sha256,
            format!("input_count={} expected=1", inputs.len()),
        ));
    }
    if inputs[0].dtype != "F32" || inputs[0].shape != [1, 3, 240, 320] {
        return Err(OnnxProbeError::new(
            OnnxProbeStage::Validation,
            sha256,
            format!(
                "input name={} dtype={} shape={:?} expected dtype=F32 shape=[1, 3, 240, 320]",
                inputs[0].name, inputs[0].dtype, inputs[0].shape
            ),
        ));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn validate_ultraface_outputs(outputs: &[IOFact], sha256: &str) -> Result<(), OnnxProbeError> {
    let expected = [[1_i64, 4420, 2], [1, 4420, 4]];
    if outputs.len() != expected.len() {
        return Err(OnnxProbeError::new(
            OnnxProbeStage::Validation,
            sha256,
            format!("output_count={} expected=2", outputs.len()),
        ));
    }
    for (index, (output, expected_shape)) in outputs.iter().zip(expected).enumerate() {
        if output.dtype != "F32" || output.shape != expected_shape {
            return Err(OnnxProbeError::new(
                OnnxProbeStage::Validation,
                sha256,
                format!(
                    "output_index={} name={} dtype={} shape={:?} expected dtype=F32 shape={expected_shape:?}",
                    index, output.name, output.dtype, output.shape
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn summarize_output(
    index: usize,
    value: &tract_onnx::prelude::TValue,
    facts: &[IOFact],
    sha256: &str,
) -> Result<TensorValueSummary, OnnxProbeError> {
    let array = value.to_plain_array_view::<f32>().map_err(|error| {
        OnnxProbeError::new(
            OnnxProbeStage::OutputFact,
            sha256,
            format!("output_index={index} cannot read F32 values: {error:?}"),
        )
    })?;
    let values = array.iter().copied().collect::<Vec<_>>();
    let all_finite = values.iter().all(|value| value.is_finite());
    if !all_finite {
        return Err(OnnxProbeError::new(
            OnnxProbeStage::Validation,
            sha256,
            format!("output_index={index} contains non-finite values"),
        ));
    }
    let expected_count: usize = facts
        .get(index)
        .map(|fact| fact.shape.iter().map(|&dim| dim.max(0) as usize).product())
        .unwrap_or_default();
    if values.len() != expected_count {
        return Err(OnnxProbeError::new(
            OnnxProbeStage::Validation,
            sha256,
            format!(
                "output_index={index} element_count={} expected={expected_count}",
                values.len()
            ),
        ));
    }
    Ok(TensorValueSummary {
        name: facts
            .get(index)
            .map(|fact| fact.name.clone())
            .unwrap_or_default(),
        dtype: "F32".to_string(),
        shape: value.shape().to_vec(),
        element_count: values.len(),
        all_finite,
        min: values.iter().copied().fold(f32::INFINITY, f32::min),
        max: values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    })
}

/// Format the operator inventory as a stable, reviewable text artifact.
#[cfg(feature = "onnx")]
pub fn format_ultraface_operator_inventory(report: &UltraFaceProbeReport) -> String {
    let mut text = format!(
        "model_id={}\nsha256={}\nbyte_size={}\noperator_count={}\n",
        report.model_id,
        report.sha256,
        report.byte_size,
        report.operators.len()
    );
    for entry in &report.operators {
        text.push_str(&format!(
            "{}\t{}\t{}\n",
            entry.index,
            entry.node.replace(['\t', '\r', '\n'], " "),
            entry.operator.replace(['\t', '\r', '\n'], " "),
        ));
    }
    text
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode_upper(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> std::path::PathBuf {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.join("assets").join("models")
    }

    #[test]
    #[ignore = "requires downloaded model artifacts"]
    fn probe_face_detector() {
        let probe = probe_tflite_model(model_dir().join("face_detector.tflite")).unwrap();
        assert!(!probe.inputs.is_empty());
        assert!(!probe.outputs.is_empty());
    }

    #[test]
    #[ignore = "requires downloaded model artifacts"]
    fn probe_face_landmarks_detector() {
        let probe = probe_tflite_model(model_dir().join("face_landmarks_detector.tflite")).unwrap();
        assert!(!probe.inputs.is_empty());
        assert!(!probe.outputs.is_empty());
    }

    #[test]
    #[ignore = "requires downloaded model artifacts"]
    fn probe_face_blendshapes() {
        let probe = probe_tflite_model(model_dir().join("face_blendshapes.tflite")).unwrap();
        assert!(!probe.inputs.is_empty());
        assert!(!probe.outputs.is_empty());
    }

    #[test]
    #[ignore = "requires downloaded model artifacts"]
    fn probe_legacy_face_landmark() {
        let probe = probe_tflite_model(model_dir().join("face_landmark.tflite")).unwrap();
        assert!(!probe.inputs.is_empty());
        assert!(!probe.outputs.is_empty());
        println!("{probe:#?}");
    }

    #[test]
    #[cfg(feature = "onnx")]
    #[ignore = "requires downloaded model artifacts"]
    fn probe_peppapig_student_256() {
        let probe =
            probe_onnx_model(model_dir().join("peppapig_student_1x3x256x256.onnx")).unwrap();
        assert!(!probe.inputs.is_empty());
        assert!(!probe.outputs.is_empty());
        println!("{probe:#?}");
    }
}
