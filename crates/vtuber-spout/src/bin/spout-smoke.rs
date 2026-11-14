//! Twenty-second transparent BGRA test source for a local Spout receiver.
#![forbid(unsafe_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !vtuber_spout::available() {
        return Err("Spout2 requires Windows".into());
    }
    let mut sender = vtuber_spout::Controller::default();
    sender.start();
    for seq in 0..1_200_u64 {
        if let vtuber_spout::Status::Error(message) = sender.status() {
            return Err(message.into());
        }
        let width = if seq < 600 { 640 } else { 800 };
        let height = 360;
        let mut pixels = vec![0_u8; width as usize * height as usize * 4];
        for (i, pixel) in pixels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let x = i % width as usize;
            let y = i / width as usize;
            pixel.copy_from_slice(&[
                255,
                (seq % 256) as u8,
                32,
                if x < width as usize / 2 && y < 180 {
                    0
                } else {
                    255
                },
            ]);
        }
        sender.submit(vtuber_core::VideoOutputFrame::new_bgra8(
            width,
            height,
            vtuber_core::FrameSeq(seq),
            vtuber_core::MonoTimeNs(seq * 16_666_667),
            pixels,
        )?);
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    sender.stop();
    Ok(())
}
