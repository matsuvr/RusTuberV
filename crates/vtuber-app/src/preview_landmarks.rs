//! Display-only snapshot of the canonical face landmarks.
//!
//! The snapshot is the boundary between inference orchestration and the UI.
//! It shares the canonical landmark allocation through [`Arc`] and contains
//! no camera bytes, inference runtime, tracking state, or avatar controls.

use std::sync::Arc;

use bevy::prelude::*;
use vtuber_core::{
    FaceLandmark, FaceTrackingSample, FrameSeq, MEDIAPIPE_FACE_LANDMARK_COUNT, MonoTimeNs,
    monotonic_now,
};

/// Maximum age of a landmark snapshot for display purposes, in nanoseconds.
pub const PREVIEW_LANDMARK_FRESHNESS_NS: u64 = 500_000_000;

/// Read-only display data copied from one canonical face sample.
#[derive(Clone, Debug)]
pub struct PreviewLandmarkSnapshot {
    /// Source camera frame sequence.
    pub source_seq: FrameSeq,
    /// Capture timestamp carried by the canonical sample.
    pub captured_at: MonoTimeNs,
    /// Monotonic time at which this snapshot was published to the resource.
    pub published_at: MonoTimeNs,
    /// Shared canonical landmark storage; no per-frame `Vec` is created.
    pub landmarks: Arc<[FaceLandmark]>,
}

impl PreviewLandmarkSnapshot {
    /// Returns whether this snapshot is fresh at the supplied monotonic time.
    ///
    /// A timestamp exactly at the 500 ms limit is stale. A backwards clock
    /// value is also treated as stale, even though the production clock is
    /// monotonic, so callers never display a snapshot outside its interval.
    #[must_use]
    pub fn is_fresh_at(&self, now: MonoTimeNs) -> bool {
        now.0 >= self.published_at.0 && now.0 - self.published_at.0 < PREVIEW_LANDMARK_FRESHNESS_NS
    }
}

/// Bevy resource holding the latest valid landmark snapshot for the UI.
#[derive(Resource, Debug, Default)]
pub struct PreviewLandmarkState {
    /// Latest published snapshot, retained for diagnostics and freshness checks.
    pub latest: Option<PreviewLandmarkSnapshot>,
}

impl PreviewLandmarkState {
    /// Publishes a canonical sample when its display coordinates are valid.
    ///
    /// Returns `true` only when a new source sequence replaced the previous
    /// snapshot. The landmark backing allocation is shared through `Arc`.
    /// Invalid coordinates clear the old display snapshot and are not exposed.
    pub fn update_from_sample(
        &mut self,
        sample: &FaceTrackingSample,
        published_at: MonoTimeNs,
    ) -> bool {
        if !is_displayable_sample(sample) {
            self.clear();
            return false;
        }
        if self
            .latest
            .as_ref()
            .is_some_and(|latest| latest.source_seq == sample.source_seq)
        {
            return false;
        }

        self.latest = Some(PreviewLandmarkSnapshot {
            source_seq: sample.source_seq,
            captured_at: sample.captured_at,
            published_at,
            landmarks: Arc::clone(&sample.landmarks),
        });
        true
    }

    /// Clears the snapshot after no-face, stop, retry, or worker replacement.
    pub fn clear(&mut self) {
        self.latest = None;
    }

    /// Returns the latest snapshot only when it is fresh at `now`.
    #[must_use]
    pub fn latest_fresh_at(&self, now: MonoTimeNs) -> Option<&PreviewLandmarkSnapshot> {
        self.latest
            .as_ref()
            .filter(|snapshot| snapshot.is_fresh_at(now))
    }
}

fn is_displayable_sample(sample: &FaceTrackingSample) -> bool {
    sample.landmarks.len() == MEDIAPIPE_FACE_LANDMARK_COUNT
        && sample
            .landmarks
            .iter()
            .all(|landmark| landmark.x.is_finite() && landmark.y.is_finite())
}

/// Synchronizes the display-only snapshot after canonical inference output is read.
///
/// `InferenceRuntime` remains an orchestration resource: the UI renderer reads
/// only [`PreviewLandmarkState`], never the inference runtime or its worker.
pub fn sync_preview_landmark_system(
    inference: Res<crate::inference_runtime::InferenceRuntime>,
    mut landmarks: ResMut<PreviewLandmarkState>,
) {
    match inference.latest_face_sample.as_ref() {
        Some(sample) => {
            let _ = landmarks.update_from_sample(sample, monotonic_now());
        }
        None => landmarks.clear(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use vtuber_core::{
        CameraFaceTransform, FaceBlendshapeSet, FaceTrackingQuality, MediaPipeBlendshape,
    };

    fn sample(source_seq: u64) -> FaceTrackingSample {
        let pairs = MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| (category.as_str(), 0.0))
            .collect::<Vec<_>>();
        FaceTrackingSample {
            source_seq: FrameSeq(source_seq),
            captured_at: MonoTimeNs(10),
            inference_started_at: MonoTimeNs(11),
            inference_finished_at: MonoTimeNs(12),
            camera_to_face: CameraFaceTransform::identity(),
            face_center: [0.5, 0.5],
            landmarks: (0..MEDIAPIPE_FACE_LANDMARK_COUNT)
                .map(|index| FaceLandmark {
                    x: index as f32 / MEDIAPIPE_FACE_LANDMARK_COUNT as f32,
                    y: 0.5,
                    ..FaceLandmark::default()
                })
                .collect::<Vec<_>>()
                .into(),
            blendshapes: FaceBlendshapeSet::from_pairs(&pairs).expect("canonical categories"),
            quality: FaceTrackingQuality {
                landmark_presence_median: Some(1.0),
                matrix_orthogonality_error: 0.0,
                matrix_determinant: 1.0,
            },
        }
    }

    fn runtime() -> crate::inference_runtime::InferenceRuntime {
        crate::inference_runtime::InferenceRuntime::new(
            Arc::new(vtuber_core::LatestSlot::new()),
            PathBuf::from("."),
        )
    }

    #[test]
    fn new_source_sequence_publishes_shared_478_point_snapshot() {
        let sample = sample(1);
        let mut state = PreviewLandmarkState::default();

        assert!(state.update_from_sample(&sample, MonoTimeNs(100)));
        let snapshot = state.latest.as_ref().expect("snapshot published");
        assert_eq!(snapshot.source_seq, FrameSeq(1));
        assert_eq!(snapshot.captured_at, MonoTimeNs(10));
        assert_eq!(snapshot.published_at, MonoTimeNs(100));
        assert_eq!(snapshot.landmarks.len(), MEDIAPIPE_FACE_LANDMARK_COUNT);
        assert!(Arc::ptr_eq(&snapshot.landmarks, &sample.landmarks));
    }

    #[test]
    fn duplicate_source_sequence_does_not_replace_snapshot_or_publish_time() {
        let sample = sample(1);
        let mut state = PreviewLandmarkState::default();
        assert!(state.update_from_sample(&sample, MonoTimeNs(100)));
        let first = state.latest.clone().expect("first snapshot");

        assert!(!state.update_from_sample(&sample, MonoTimeNs(200)));
        let second = state.latest.as_ref().expect("snapshot retained");
        assert!(Arc::ptr_eq(&first.landmarks, &second.landmarks));
        assert_eq!(second.published_at, MonoTimeNs(100));
    }

    #[test]
    fn no_face_stop_and_retry_clear_the_display_snapshot() {
        let sample = sample(1);
        let mut app = App::new();
        app.insert_resource(runtime())
            .init_resource::<PreviewLandmarkState>()
            .add_systems(Update, sync_preview_landmark_system);
        app.world_mut()
            .resource_mut::<crate::inference_runtime::InferenceRuntime>()
            .latest_face_sample = Some(sample.clone());
        app.update();
        assert!(
            app.world()
                .resource::<PreviewLandmarkState>()
                .latest
                .is_some()
        );

        // NoFace clears the canonical latest sample and the display state in
        // the same synchronization path.
        app.world_mut()
            .resource_mut::<crate::inference_runtime::InferenceRuntime>()
            .latest_face_sample = None;
        app.update();
        assert!(
            app.world()
                .resource::<PreviewLandmarkState>()
                .latest
                .is_none()
        );

        // Retry and stop both call InferenceRuntime::stop_model, which clears
        // the runtime's canonical sample before this display sync runs.
        app.world_mut()
            .resource_mut::<crate::inference_runtime::InferenceRuntime>()
            .latest_face_sample = Some(sample);
        app.update();
        assert!(
            app.world()
                .resource::<PreviewLandmarkState>()
                .latest
                .is_some()
        );
        app.world_mut()
            .resource_mut::<crate::inference_runtime::InferenceRuntime>()
            .stop_model();
        app.update();
        assert!(
            app.world()
                .resource::<PreviewLandmarkState>()
                .latest
                .is_none()
        );
    }

    #[test]
    fn freshness_is_strictly_less_than_500_milliseconds() {
        let sample = sample(1);
        let mut state = PreviewLandmarkState::default();
        assert!(state.update_from_sample(&sample, MonoTimeNs(1_000)));

        assert!(
            state
                .latest_fresh_at(MonoTimeNs(1_000 + 499_999_999))
                .is_some()
        );
        assert!(
            state
                .latest_fresh_at(MonoTimeNs(1_000 + 500_000_000))
                .is_none()
        );
        assert!(state.latest_fresh_at(MonoTimeNs(999)).is_none());
    }

    #[test]
    fn malformed_landmark_count_or_coordinates_are_not_exposed() {
        let mut state = PreviewLandmarkState::default();
        let mut invalid_count = sample(1);
        invalid_count.landmarks = Arc::from(vec![FaceLandmark::default()]);
        assert!(!state.update_from_sample(&invalid_count, MonoTimeNs(100)));
        assert!(state.latest.is_none());

        let mut invalid_coordinate = sample(2);
        let mut landmarks = invalid_coordinate.landmarks.to_vec();
        landmarks[0].x = f32::NAN;
        invalid_coordinate.landmarks = landmarks.into();
        assert!(!state.update_from_sample(&invalid_coordinate, MonoTimeNs(100)));
        assert!(state.latest.is_none());
    }

    #[test]
    fn snapshot_state_is_independent_of_preview_visibility_and_mirror() {
        let sample = sample(1);
        let mut state = PreviewLandmarkState::default();
        assert!(state.update_from_sample(&sample, MonoTimeNs(100)));
        let before = state.latest.clone().expect("snapshot published");

        // The display resource has no visibility or mirror fields. Changing
        // PreviewState cannot mutate or replace its canonical snapshot.
        let mut preview = crate::preview::PreviewState::default();
        preview.toggle_visible();
        preview.toggle_mirrored();
        assert_eq!(
            state.latest.as_ref().expect("snapshot retained").source_seq,
            before.source_seq
        );
        assert!(Arc::ptr_eq(
            &state.latest.as_ref().expect("snapshot retained").landmarks,
            &before.landmarks
        ));
    }
}
