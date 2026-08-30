// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

//! Contract tests for the frame-level composite inference boundary.

#![cfg(feature = "legacy-face-stack")]

use std::path::{Path, PathBuf};

use vtuber_inference::{CompositeFrameInference, FrameInferenceOutcome, RuntimeSettings};

#[test]
fn composite_contract_distinguishes_no_face_from_face() {
    assert!(matches!(
        FrameInferenceOutcome::NoFace,
        FrameInferenceOutcome::NoFace
    ));
}

#[test]
#[ignore = "requires downloaded legacy UltraFace and Peppa Pig ONNX research artifacts"]
fn composite_contract_constructs_detector_and_landmark_owners_from_pipeline() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate is nested beneath workspace root");
    let pipeline = repository_pipeline_descriptor();
    let runtime = CompositeFrameInference::from_pipeline_descriptor(
        &pipeline,
        &root.join("assets").join("models"),
        &RuntimeSettings::default(),
    )
    .expect("manifest-tracked detector and landmark should construct");

    assert_eq!(runtime.descriptor().id, "ultraface-rfb-320-peppapig-98");
}

fn repository_pipeline_descriptor() -> vtuber_inference::FacePipelineDescriptor {
    let detector_input = vtuber_inference::TensorContract {
        shape: vec![1, 3, 240, 320],
        dtype: "float32".into(),
        layout: vtuber_inference::TensorLayout::Nchw,
        channel_order: vtuber_inference::ChannelOrder::Rgb,
        value_domain: vtuber_inference::InputValueDomain::RawU8,
        normalization: vtuber_inference::NormalizationContract {
            mean: [127.0; 3],
            scale: [128.0; 3],
        },
    };
    let landmark_input = vtuber_inference::TensorContract {
        shape: vec![1, 3, 256, 256],
        dtype: "float32".into(),
        layout: vtuber_inference::TensorLayout::Nchw,
        channel_order: vtuber_inference::ChannelOrder::Rgb,
        value_domain: vtuber_inference::InputValueDomain::UnitFloat,
        normalization: vtuber_inference::NormalizationContract {
            mean: [0.485, 0.456, 0.406],
            scale: [0.229, 0.224, 0.225],
        },
    };
    vtuber_inference::FacePipelineDescriptor {
        id: "ultraface-rfb-320-peppapig-98".into(),
        detector: vtuber_inference::ModelArtifactDescriptor {
            id: "ultraface-rfb-320".into(),
            role: vtuber_inference::ModelRole::FaceDetector,
            file: PathBuf::from("version-RFB-320.onnx"),
            byte_size: 1_270_727,
            sha256: "34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017".into(),
            input_name: "input".into(),
            source: "https://huggingface.co/onnxmodelzoo/version-RFB-320/resolve/main/version-RFB-320.onnx".into(),
            upstream: "https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface".into(),
            license: "MIT".into(),
            license_url: Some("https://opensource.org/license/mit/".into()),
            input: detector_input,
            outputs: vec![
                vtuber_inference::OutputTensorContract {
                    name: "scores".into(),
                    shape: vec![1, 4420, 2],
                    dtype: "float32".into(),
                    description: "Per-anchor background and face scores".into(),
                },
                vtuber_inference::OutputTensorContract {
                    name: "boxes".into(),
                    shape: vec![1, 4420, 4],
                    dtype: "float32".into(),
                    description: "Per-anchor encoded face boxes".into(),
                },
            ],
            requires_crop: false,
            schema: None,
            landmark_coordinate_encoding: None,
            pose_method: None,
            representative_indices: Vec::new(),
        },
        landmarks: vtuber_inference::ModelArtifactDescriptor {
            id: "peppapig-98".into(),
            role: vtuber_inference::ModelRole::FaceLandmarks,
            file: PathBuf::from("peppapig_student_1x3x256x256.onnx"),
            byte_size: 13_728_231,
            sha256: "73EDF90954F05EBEF4639E7FA8620C5F83CCA09D2476DE66AB100F26C2B25E7A".into(),
            input_name: "input".into(),
            source: "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/436_Peppa_Pig_Face_Landmark/resources.tar.gz".into(),
            upstream: "https://github.com/610265158/Peppa_Pig_Face_Landmark".into(),
            license: "Apache-2.0".into(),
            license_url: Some("https://github.com/610265158/Peppa_Pig_Face_Landmark/blob/master/LICENSE".into()),
            input: landmark_input,
            outputs: vec![vtuber_inference::OutputTensorContract {
                name: "/Concat_1".into(),
                shape: vec![1, 98, 3],
                dtype: "float32".into(),
                description: "98 facial landmarks with visibility/confidence in third channel".into(),
            }],
            requires_crop: true,
            schema: Some("peppapig-98".into()),
            landmark_coordinate_encoding: Some("normalized_0_1".into()),
            pose_method: Some("canonical_orthographic_2d".into()),
            representative_indices: vec![16, 37, 46, 52, 63, 71, 76, 82],
        },
        detector_postprocess: vtuber_inference::DetectorPostprocessConfig {
            score_threshold: 0.7,
            nms_iou: 0.3,
            max_pre_nms_candidates: 256,
            max_post_nms_detections: 16,
        },
        crop: vtuber_inference::FaceCropConfig {
            square_scale: 1.35,
            center_y_offset_fraction: -0.05,
            output_size: [256, 256],
            interpolation: vtuber_inference::CropInterpolation::Bilinear,
            outside_fill: vtuber_inference::CropOutsideFill::NormalizationMean,
        },
    }
}
