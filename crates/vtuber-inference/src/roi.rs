//! Typed region-of-interest state for face inference.
//!
//! The face ROI is kept in frame coordinates so that detector output,
//! landmark crop planning, and confidence checks share the same unit.

use vtuber_core::types::NormalizedRect;

/// A face region-of-interest expressed in frame coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceRoi {
    /// Horizontal center in pixels.
    pub center_x: f32,
    /// Vertical center in pixels.
    pub center_y: f32,
    /// Clockwise rotation in radians as viewed in the unmirrored frame.
    pub rotation_rad: f32,
    /// Relative scale compared to the smaller frame dimension.
    ///
    /// A value of `1.0` means the ROI spans the entire smaller dimension.
    pub scale: f32,
    /// Confidence associated with this ROI, in `[0, 1]`.
    pub confidence: f32,
}

impl FaceRoi {
    /// Creates a `FaceRoi` from a normalized rectangle and frame size.
    pub fn from_normalized_rect(rect: &NormalizedRect, frame_w: u32, frame_h: u32) -> Self {
        let fw = frame_w as f32;
        let fh = frame_h as f32;
        let min_dim = fw.min(fh);
        let width_px = rect.width * fw;
        let height_px = rect.height * fh;

        Self {
            center_x: rect.x * fw + width_px / 2.0,
            center_y: rect.y * fh + height_px / 2.0,
            rotation_rad: rect.rotation_rad,
            scale: if min_dim > 0.0 {
                width_px.max(height_px) / min_dim
            } else {
                0.0
            },
            confidence: 1.0,
        }
    }

    /// Converts this ROI back to a normalized rectangle.
    pub fn to_normalized_rect(&self, frame_w: u32, frame_h: u32) -> NormalizedRect {
        let fw = frame_w.max(1) as f32;
        let fh = frame_h.max(1) as f32;
        let min_dim = fw.min(fh);
        let size = self.scale * min_dim;

        NormalizedRect {
            x: (self.center_x - size / 2.0) / fw,
            y: (self.center_y - size / 2.0) / fh,
            width: size / fw,
            height: size / fh,
            rotation_rad: self.rotation_rad,
        }
    }

    /// Returns true if the ROI center lies within the frame plus the given
    /// margin and the scale is positive and finite.
    pub fn is_in_bounds(&self, frame_w: u32, frame_h: u32, margin: f32) -> bool {
        let min_x = -margin;
        let max_x = frame_w as f32 + margin;
        let min_y = -margin;
        let max_y = frame_h as f32 + margin;

        self.center_x >= min_x
            && self.center_x <= max_x
            && self.center_y >= min_y
            && self.center_y <= max_y
            && self.scale.is_finite()
            && self.scale > 0.0
    }
}

/// Lifecycle of the tracked face ROI.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum RoiState {
    /// No face has been detected yet.
    #[default]
    Empty,
    /// A face is being tracked with an active ROI.
    Tracking(FaceRoi),
    /// The face was lost and the pipeline must detect again.
    Lost,
}

impl RoiState {
    /// Returns true if the state is tracking a face.
    pub fn is_tracking(&self) -> bool {
        matches!(self, RoiState::Tracking(_))
    }

    /// Returns true if the state is lost.
    pub fn is_lost(&self) -> bool {
        matches!(self, RoiState::Lost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, width: f32, height: f32, rotation_rad: f32) -> NormalizedRect {
        NormalizedRect {
            x,
            y,
            width,
            height,
            rotation_rad,
        }
    }

    #[test]
    fn roi_state_converts_normalized_rect_round_trip() {
        // Use a square frame so the single-scale reconstruction is exact.
        let original = rect(0.2, 0.3, 0.4, 0.4, 0.1);
        let roi = FaceRoi::from_normalized_rect(&original, 480, 480);

        assert!((roi.center_x - (0.2 + 0.4 / 2.0) * 480.0).abs() < 1e-4);
        assert!((roi.center_y - (0.3 + 0.4 / 2.0) * 480.0).abs() < 1e-4);
        assert_eq!(roi.rotation_rad, 0.1);
        assert!(roi.scale > 0.0);

        let back = roi.to_normalized_rect(480, 480);
        assert!((back.x - original.x).abs() < 1e-4);
        assert!((back.y - original.y).abs() < 1e-4);
        assert!((back.width - original.width).abs() < 1e-4);
        assert!((back.height - original.height).abs() < 1e-4);
        assert!((back.rotation_rad - original.rotation_rad).abs() < 1e-4);
    }

    #[test]
    fn roi_state_out_of_bounds_becomes_lost() {
        let roi = FaceRoi {
            center_x: 1000.0,
            center_y: 1000.0,
            rotation_rad: 0.0,
            scale: 0.5,
            confidence: 1.0,
        };
        assert!(!roi.is_in_bounds(640, 480, 32.0));
    }

    #[test]
    fn roi_state_negative_scale_is_invalid() {
        let roi = FaceRoi {
            center_x: 320.0,
            center_y: 240.0,
            rotation_rad: 0.0,
            scale: -0.5,
            confidence: 1.0,
        };
        assert!(!roi.is_in_bounds(640, 480, 32.0));
    }

    #[test]
    fn roi_state_transitions_empty_to_tracking() {
        let mut state = RoiState::Empty;
        assert!(!state.is_tracking());

        let roi = FaceRoi::from_normalized_rect(&rect(0.4, 0.4, 0.2, 0.2, 0.0), 640, 480);
        state = RoiState::Tracking(roi);
        assert!(state.is_tracking());
        assert!(!state.is_lost());

        state = RoiState::Lost;
        assert!(state.is_lost());
        assert!(!state.is_tracking());
    }
}
