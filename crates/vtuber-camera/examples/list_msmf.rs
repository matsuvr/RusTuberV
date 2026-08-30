//! Lists camera descriptors produced by the Windows MSMF backend.

#[cfg(target_os = "windows")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use vtuber_camera::backend::msmf::MsmfBackend;
    use vtuber_camera::device::CameraBackend;

    for (position, camera) in MsmfBackend::new().enumerate()?.into_iter().enumerate() {
        println!("{position}: {} [{}]", camera.label, camera.id);
    }
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("MSMF camera enumeration is available only on Windows");
}
