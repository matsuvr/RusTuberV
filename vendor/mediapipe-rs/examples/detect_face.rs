//! Run with: cargo run --example detect_face -- models/portrait.jpg

use mediapipe::{FaceDetector, FaceLandmarker, Image, IouThreshold, ModelSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/portrait.jpg".to_owned());

    let image = Image::from_file(&path)?;
    println!("{path}: {}x{}", image.size().width, image.size().height);

    let mut detector =
        FaceDetector::builder(ModelSource::path("models/blaze_face_short_range.tflite"))
            .min_suppression_threshold(IouThreshold::new(0.3)?)
            .build()?;

    for (i, face) in detector.detect(&image)?.iter().enumerate() {
        let b = face.bounding_box;
        let score = face.score().map(|s| s.get()).unwrap_or(f32::NAN);
        println!(
            "face {i}: box=({},{} {}x{}) score={score:.3} keypoints={}",
            b.left(),
            b.top(),
            b.width(),
            b.height(),
            face.keypoints.len(),
        );
        // Keypoints are normalized; the box is in pixels. The type system makes
        // you convert rather than silently comparing the two.
        for kp in &face.keypoints {
            let p = kp.point.to_pixels(image.size());
            println!("    keypoint at ({:.0}, {:.0})", p.x(), p.y());
        }
    }

    let mut landmarker = FaceLandmarker::builder(ModelSource::path("models/face_landmarker.task"))
        .output_blendshapes(true)
        .output_transformation_matrixes(true)
        .build()?;

    let result = landmarker.detect(&image)?;
    for (i, face) in result.landmarks.iter().enumerate() {
        println!("face {i}: {} landmarks", face.len());
    }
    if let Some(bs) = result.blendshapes.first() {
        let mut top: Vec<_> = bs.iter().collect();
        top.sort_by(|a, b| b.score.get().total_cmp(&a.score.get()));
        println!(
            "top blendshapes: {}",
            top.iter()
                .take(3)
                .map(|c| format!(
                    "{}={:.2}",
                    c.category_name.as_deref().unwrap_or("?"),
                    c.score.get()
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}
