//! Check the transport used by the app against a native Spout receiver.
#![cfg(windows)]

use std::time::{Duration, Instant};
use vtuber_core::{FrameSeq, MonoTimeNs, VideoOutputFrame};

#[test]
#[ignore = "requires a Windows GPU and the RusTuberV sender name"]
fn receiver_preserves_rendered_bgra_across_resize() -> Result<(), Box<dyn std::error::Error>> {
    let mut sender = vtuber_spout::Controller::default();
    sender.start();
    let mut receiver = spout2::dx::Receiver::new(Some("RusTuberV"))?;
    for width in [64, 80] {
        let height = 32;
        let row = [0, 0, 0, 0, 31, 72, 104, 128].repeat(width as usize / 2);
        let expected = [
            row.repeat(16),
            [10, 20, 30, 255].repeat(width as usize * 16),
        ]
        .concat();
        let frame = VideoOutputFrame::from_rendered_bgra8(
            width,
            height,
            width as usize * 4,
            FrameSeq(1),
            MonoTimeNs(0),
            expected.clone(),
        )?;
        let mut received = vec![0; expected.len()];
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            sender.submit(frame.clone());
            if let vtuber_spout::Status::Error(error) = sender.status() {
                return Err(error.into());
            }
            if receiver.receive_image(&mut received, width, height, false, false)?
                && !receiver.is_updated()
                && receiver.is_frame_new()
                && received == expected
            {
                break;
            }
            if Instant::now() >= deadline {
                return Err(
                    format!("Spout did not deliver the {width}x{height} BGRA frame").into(),
                );
            }
            std::thread::sleep(Duration::from_millis(16));
        }
    }
    sender.stop();
    Ok(())
}
