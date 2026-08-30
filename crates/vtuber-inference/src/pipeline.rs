//! Face inference pipeline orchestration.
//!
//! The pipeline decides whether to run the face detector on a given frame,
//! maintains the tracked ROI, and handles lost/recovery transitions.
//!
//! At this milestone the pipeline gates a single combined detector/landmark
//! stage. A later task will split detector and landmark stages so that
//! landmark inference can run every frame while the detector runs only on
//! cadence.

use vtuber_core::types::{FrameSeq, RawFaceObservation};

use crate::descriptor::RuntimeSettings;
use crate::error::InferenceError;
use crate::roi::{FaceRoi, RoiState};

/// Minimum confidence to consider a face detection or observation valid.
const FACE_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Margin in pixels allowed around the frame when validating an ROI.
const ROI_BOUNDARY_MARGIN: f32 = 32.0;

/// Orchestrates detector cadence and ROI state for face inference.
#[derive(Debug)]
pub struct Pipeline {
    detector_interval_frames: u32,
    roi_state: RoiState,
    last_detector_seq: Option<FrameSeq>,
}

impl Pipeline {
    /// Creates a new pipeline from runtime settings.
    pub fn new(settings: &RuntimeSettings) -> Self {
        Self {
            // A zero interval would deadlock the tracker, so force at least 1.
            detector_interval_frames: settings.detector_interval_frames.max(1),
            roi_state: RoiState::Empty,
            last_detector_seq: None,
        }
    }

    /// Returns the current ROI state.
    pub fn roi_state(&self) -> &RoiState {
        &self.roi_state
    }

    /// Returns true if a face is currently being tracked.
    pub fn is_tracking(&self) -> bool {
        self.roi_state.is_tracking()
    }

    /// Returns true if the detector should run for `frame_seq`.
    ///
    /// The detector runs unconditionally while searching or lost, and on a
    /// fixed frame cadence while tracking. A low-confidence ROI also forces a
    /// detector run.
    pub fn should_run_detector(&self, frame_seq: FrameSeq) -> bool {
        match self.roi_state {
            RoiState::Empty | RoiState::Lost => true,
            RoiState::Tracking(roi) => {
                let elapsed = self
                    .last_detector_seq
                    .map_or(0, |last| frame_seq.0.saturating_sub(last.0));
                elapsed >= self.detector_interval_frames as u64
                    || roi.confidence < FACE_CONFIDENCE_THRESHOLD
            }
        }
    }

    /// Records that the detector ran on `frame_seq`.
    pub fn record_detector_run(&mut self, frame_seq: FrameSeq) {
        self.last_detector_seq = Some(frame_seq);
    }

    /// Records the result of a detector run and updates the ROI state.
    pub fn record_detector_result(
        &mut self,
        frame_seq: FrameSeq,
        frame_w: u32,
        frame_h: u32,
        detection: Option<FaceRoi>,
    ) {
        self.last_detector_seq = Some(frame_seq);
        self.roi_state = match detection {
            Some(roi)
                if roi.confidence >= FACE_CONFIDENCE_THRESHOLD
                    && roi.is_in_bounds(frame_w, frame_h, ROI_BOUNDARY_MARGIN) =>
            {
                RoiState::Tracking(roi)
            }
            _ => RoiState::Lost,
        };
    }

    /// Updates the ROI from a landmark observation.
    ///
    /// Returns `Err` and transitions to [`RoiState::Lost`] if the observation
    /// confidence is too low or the derived ROI is out of bounds.
    pub fn update_from_observation(
        &mut self,
        observation: &RawFaceObservation,
        frame_w: u32,
        frame_h: u32,
    ) -> Result<(), InferenceError> {
        if !observation.face_confidence.is_finite()
            || observation.face_confidence < FACE_CONFIDENCE_THRESHOLD
        {
            self.roi_state = RoiState::Lost;
            return Err(InferenceError::InvalidRoi(
                "face confidence below threshold".into(),
            ));
        }

        let roi = FaceRoi::from_normalized_rect(&observation.roi, frame_w, frame_h);
        if !roi.is_in_bounds(frame_w, frame_h, ROI_BOUNDARY_MARGIN) {
            self.roi_state = RoiState::Lost;
            return Err(InferenceError::InvalidRoi("ROI out of bounds".into()));
        }

        self.roi_state = RoiState::Tracking(roi);
        Ok(())
    }

    /// Marks the tracked face as lost.
    pub fn mark_lost(&mut self) {
        self.roi_state = RoiState::Lost;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vtuber_core::types::{
        FrameSeq, LandmarkSchemaId, MonoTimeNs, NamedCoefficient, NormalizedRect,
        RawExpressionObservation,
    };

    fn settings(interval: u32) -> RuntimeSettings {
        RuntimeSettings {
            frame_wait_timeout_ms: 100,
            detector_interval_frames: interval,
        }
    }

    fn centered_roi(confidence: f32) -> FaceRoi {
        FaceRoi {
            center_x: 320.0,
            center_y: 240.0,
            rotation_rad: 0.0,
            scale: 0.5,
            confidence,
        }
    }

    fn observation(confidence: f32, x: f32, y: f32, width: f32, height: f32) -> RawFaceObservation {
        RawFaceObservation {
            source_seq: FrameSeq(0),
            captured_at: MonoTimeNs(0),
            inference_started_at: MonoTimeNs(0),
            inference_finished_at: MonoTimeNs(0),
            face_confidence: confidence,
            landmarks: Vec::new(),
            blendshapes: Some(vec![NamedCoefficient {
                name: "blinkLeft".into(),
                value: 0.0,
            }]),
            expressions: RawExpressionObservation::default(),
            roi: NormalizedRect {
                x,
                y,
                width,
                height,
                rotation_rad: 0.0,
            },
            schema: LandmarkSchemaId("test-pipeline"),
        }
    }

    #[test]
    fn detector_cadence_initial_and_lost_run_every_frame() {
        let mut pipeline = Pipeline::new(&settings(5));

        assert!(pipeline.should_run_detector(FrameSeq(1)));
        pipeline.record_detector_result(FrameSeq(1), 640, 480, Some(centered_roi(1.0)));
        assert!(pipeline.is_tracking());

        // Tracking suppresses the detector until the cadence expires.
        assert!(!pipeline.should_run_detector(FrameSeq(2)));
        assert!(!pipeline.should_run_detector(FrameSeq(3)));
        assert!(!pipeline.should_run_detector(FrameSeq(4)));
        assert!(!pipeline.should_run_detector(FrameSeq(5)));
        assert!(pipeline.should_run_detector(FrameSeq(6)));

        pipeline.mark_lost();
        assert!(pipeline.roi_state().is_lost());
        assert!(pipeline.should_run_detector(FrameSeq(7)));
    }

    #[test]
    fn detector_cadence_low_confidence_triggers_detector() {
        let mut pipeline = Pipeline::new(&settings(5));
        pipeline.record_detector_result(FrameSeq(1), 640, 480, Some(centered_roi(1.0)));

        // A low-confidence observation forces the detector on the next frame.
        assert!(!pipeline.should_run_detector(FrameSeq(2)));
        let result =
            pipeline.update_from_observation(&observation(0.1, 0.4, 0.3, 0.2, 0.2), 640, 480);
        assert!(result.is_err());
        assert!(pipeline.roi_state().is_lost());
        assert!(pipeline.should_run_detector(FrameSeq(3)));
    }

    #[test]
    fn detector_cadence_counter_uses_frame_sequence() {
        let mut pipeline = Pipeline::new(&settings(3));
        pipeline.record_detector_result(FrameSeq(10), 640, 480, Some(centered_roi(1.0)));

        assert!(!pipeline.should_run_detector(FrameSeq(11)));
        assert!(!pipeline.should_run_detector(FrameSeq(12)));
        assert!(pipeline.should_run_detector(FrameSeq(13)));
    }

    #[test]
    fn roi_state_out_of_bounds_forces_lost() {
        let mut pipeline = Pipeline::new(&settings(5));
        let obs = observation(1.0, 2.0, 2.0, 0.5, 0.5);

        let result = pipeline.update_from_observation(&obs, 640, 480);
        assert!(result.is_err());
        assert!(pipeline.roi_state().is_lost());
        assert!(pipeline.should_run_detector(FrameSeq(1)));
    }

    #[test]
    fn detector_cadence_recovers_from_lost() {
        let mut pipeline = Pipeline::new(&settings(5));
        pipeline.mark_lost();
        assert!(pipeline.should_run_detector(FrameSeq(1)));

        pipeline.record_detector_result(FrameSeq(1), 640, 480, Some(centered_roi(1.0)));
        assert!(pipeline.is_tracking());
        assert!(!pipeline.should_run_detector(FrameSeq(2)));
    }
}
