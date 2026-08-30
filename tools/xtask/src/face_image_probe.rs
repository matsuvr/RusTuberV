//! Still-image probe for the production UltraFace detector.

use std::path::Path;
use std::sync::Arc;

use image::ImageReader;
use vtuber_core::types::{
    FrameSeq, Landmark3, LandmarkSchemaId, MonoTimeNs, PixelFormat, VideoFrame,
};
use vtuber_inference::detector::{
    DetectorDecodeOutcome, UltraFaceDetector, UltraFacePreprocessBuffers, decode_detections,
    select_primary_face,
};
use vtuber_inference::{
    CompositeFrameInference, FaceCropPreprocessBuffers, FaceCropTransform, FrameFaceInference,
    FrameInferenceOutcome, LandmarkCoordinateEncoding, LandmarkStage, OnnxRuntime,
    ProductionLandmarkStage,
};

/// Runs the detector against one decoded still image.
pub fn run(args: &[String]) -> Result<(), String> {
    let mut image_path = None;
    let mut composite = false;
    for arg in args {
        match arg.as_str() {
            "--composite" => composite = true,
            value if value.starts_with('-') => return Err(format!("unknown option `{value}`")),
            value if image_path.is_none() => image_path = Some(value),
            value => return Err(format!("unexpected argument `{value}`")),
        }
    }
    let Some(path) = image_path else {
        print_help();
        return Err("face-image-probe requires an image path".into());
    };

    let path = Path::new(path);
    let project_root =
        std::env::current_dir().map_err(|error| format!("cannot resolve project root: {error}"))?;
    let manifest = project_root
        .join("assets")
        .join("models")
        .join("manifest.toml");
    let pipeline = vtuber_app::model_catalog::verify_research_pipeline_artifacts(&manifest)
        .map_err(|error| format!("model verification failed: {error}"))?;
    let artifact_root = project_root.join("assets").join("models");
    let detector_path = artifact_root.join(&pipeline.detector.file);
    let detector = UltraFaceDetector::from_path(&detector_path)
        .map_err(|error| format!("detector load failed: {error}"))?;

    let decoded = ImageReader::open(path)
        .map_err(|error| format!("image read failed for {}: {error}", path.display()))?
        .decode()
        .map_err(|error| format!("image decode failed for {}: {error}", path.display()))?
        .into_rgba8();
    let (width, height) = decoded.dimensions();
    let data = decoded.into_raw();
    let expected_len = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "decoded image dimensions overflow the RGBA8 buffer length".to_string())?;
    if data.len() != expected_len {
        return Err(format!(
            "decoded image pixel format is not RGBA8: width={width} height={height} bytes={}",
            data.len()
        ));
    }

    let frame = VideoFrame {
        seq: FrameSeq(1),
        captured_at: MonoTimeNs(1),
        width,
        height,
        stride_bytes: usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .ok_or_else(|| "image stride overflows usize".to_string())?,
        format: PixelFormat::Rgba8,
        data: Arc::<[u8]>::from(data),
    };
    let mut buffers = UltraFacePreprocessBuffers::new();
    let raw = detector
        .infer(&mut buffers, &frame)
        .map_err(|error| format!("detector inference failed: {error}"))?;
    let outcome = decode_detections(&raw, pipeline.detector_postprocess)
        .map_err(|error| format!("detector decode failed: {error}"))?;

    println!("Face image probe");
    println!("  image: {}", path.display());
    println!("  dimensions: {width}x{height}");
    println!("  pipeline: {}", pipeline.id);
    match &outcome {
        DetectorDecodeOutcome::NoFace => {
            println!("  result: NO_FACE");
        }
        DetectorDecodeOutcome::Detections(detections) => {
            println!("  result: FACE_DETECTED");
            println!("  detections: {}", detections.len());
            for (index, detection) in detections.iter().enumerate() {
                println!(
                    "  detection[{index}]: confidence={:.4} rect=({:.4},{:.4},{:.4},{:.4}) anchor={}",
                    detection.confidence,
                    detection.rect.x,
                    detection.rect.y,
                    detection.rect.width,
                    detection.rect.height,
                    detection.anchor_index,
                );
            }
        }
    }

    if composite {
        let primary = match &outcome {
            DetectorDecodeOutcome::Detections(detections) => select_primary_face(detections, None),
            DetectorDecodeOutcome::NoFace => None,
        };
        let mut runtime = CompositeFrameInference::from_pipeline_descriptor(
            &pipeline,
            &artifact_root,
            &vtuber_inference::RuntimeSettings::default(),
        )
        .map_err(|error| format!("composite runtime load failed: {error}"))?;
        let outcome = runtime
            .infer_frame(&frame)
            .map_err(|error| format!("composite inference failed: {error}"))?;
        let timing = runtime.take_timing();
        match outcome {
            FrameInferenceOutcome::NoFace => println!("  composite: NO_FACE timing={timing:?}"),
            FrameInferenceOutcome::Face(observation) => println!(
                "  composite: FACE_DETECTED landmarks={} confidence={:.4} roi=({:.4},{:.4},{:.4},{:.4}) timing={:?}",
                observation.landmarks.len(),
                observation.face_confidence,
                observation.roi.x,
                observation.roi.y,
                observation.roi.width,
                observation.roi.height,
                timing,
            ),
        }
        if let Some(detection) = primary {
            print_landmark_stage_diagnostics(&frame, detection, &pipeline, &artifact_root)?;
        }
    }
    Ok(())
}

fn print_landmark_stage_diagnostics(
    frame: &VideoFrame,
    detection: vtuber_inference::detector::FaceDetection,
    pipeline: &vtuber_inference::FacePipelineDescriptor,
    artifact_root: &Path,
) -> Result<(), String> {
    let transform = FaceCropTransform::from_detector_box(
        frame.width,
        frame.height,
        &detection.rect,
        pipeline.crop,
    )
    .map_err(|error| format!("diagnostic crop failed: {error}"))?;
    let mut crop_buffers = FaceCropPreprocessBuffers::new(pipeline.crop.output_size)
        .map_err(|error| format!("diagnostic crop buffers failed: {error}"))?;
    let tensor = crop_buffers
        .preprocess(frame, &transform, &pipeline.landmarks.input, pipeline.crop)
        .map_err(|error| format!("diagnostic crop preprocessing failed: {error}"))?;
    let input_shape: [usize; 4] = pipeline
        .landmarks
        .input
        .shape
        .clone()
        .try_into()
        .map_err(|_| "diagnostic landmark input shape is not rank four".to_string())?;
    let schema = match pipeline.landmarks.schema.as_deref() {
        Some("peppapig-98") => LandmarkSchemaId("peppapig-98"),
        Some(other) => return Err(format!("unsupported diagnostic landmark schema `{other}`")),
        None => return Err("diagnostic landmark schema is missing".into()),
    };
    let landmark_path = artifact_root.join(&pipeline.landmarks.file);
    let runtime = OnnxRuntime::new(landmark_path, schema)
        .map_err(|error| format!("diagnostic landmark runtime load failed: {error}"))?;
    let mut stage = ProductionLandmarkStage::new(runtime);
    let mut landmarks = stage
        .infer_landmarks(tensor, input_shape)
        .map_err(|error| format!("diagnostic landmark inference failed: {error}"))?;
    print_landmark_stats("raw crop output", &landmarks);

    let encoding = pipeline
        .landmarks
        .landmark_coordinate_encoding
        .as_deref()
        .and_then(LandmarkCoordinateEncoding::parse)
        .ok_or_else(|| "diagnostic landmark coordinate encoding is invalid".to_string())?;
    transform
        .map_landmarks_to_source_normalized(&mut landmarks, encoding)
        .map_err(|error| format!("diagnostic landmark mapping failed: {error}"))?;
    print_landmark_stats("mapped source output", &landmarks);
    Ok(())
}

fn print_landmark_stats(label: &str, landmarks: &[Landmark3]) {
    let finite = landmarks
        .iter()
        .filter(|landmark| {
            landmark.x.is_finite()
                && landmark.y.is_finite()
                && landmark.z.is_finite()
                && landmark.visibility.is_finite()
        })
        .count();
    let min_x = landmarks
        .iter()
        .map(|landmark| landmark.x)
        .fold(f32::INFINITY, f32::min);
    let max_x = landmarks
        .iter()
        .map(|landmark| landmark.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = landmarks
        .iter()
        .map(|landmark| landmark.y)
        .fold(f32::INFINITY, f32::min);
    let max_y = landmarks
        .iter()
        .map(|landmark| landmark.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let mean_visibility = if landmarks.is_empty() {
        0.0
    } else {
        landmarks
            .iter()
            .map(|landmark| landmark.visibility)
            .sum::<f32>()
            / landmarks.len() as f32
    };
    println!(
        "  {label}: count={} finite={} x=[{min_x:.4},{max_x:.4}] y=[{min_y:.4},{max_y:.4}] mean_visibility={mean_visibility:.4}",
        landmarks.len(),
        finite
    );
}

/// Prints command usage.
pub fn print_help() {
    println!("face-image-probe - run the production UltraFace detector on one image");
    println!();
    println!("USAGE:");
    println!("  cargo run -p xtask -- face-image-probe <image-path> [--composite]");
    println!("  --composite  Continue through crop and Peppa 98-landmark inference");
}
