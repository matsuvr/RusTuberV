//! Per-render inputs to the arm filter and avatar. Store the face control in
//! its validated public representation; playback reconstructs bounded signals.

use serde::{Deserialize, Serialize};
use vtuber_core::{Arkit52Coefficients, arm_tracking::PoseArmFrame, types::*};

#[derive(Serialize, Deserialize)]
pub(super) struct ObservationTick {
    pub seconds: f64,
    pub delta: std::time::Duration,
    pub now: MonoTimeNs,
    pub pose_now: Option<MonoTimeNs>,
    pub raw: Option<PoseArmFrame>,
    pub basis: [f32; 4],
    pub face: Option<RecordedFace>,
    pub tracked: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub(super) struct RecordedFace(#[serde(with = "Face")] pub AvatarControlFrame);

#[derive(Serialize, Deserialize)]
#[serde(remote = "AvatarControlFrame")]
struct Face {
    source_seq: FrameSeq,
    captured_at: MonoTimeNs,
    produced_at: MonoTimeNs,
    confidence: f32,
    state: TrackingState,
    head: HeadPose,
    #[serde(getter = "Face::translation")]
    head_translation: Translation,
    #[serde(getter = "Face::gaze")]
    gaze: Gaze,
    expressions: ExpressionCoefficients,
    #[serde(with = "detailed")]
    detailed_face: Option<Arkit52Coefficients>,
}

#[derive(Serialize, Deserialize)]
struct Translation {
    x_meters: f32,
    y_meters: f32,
    z_meters: f32,
    state: HeadTranslationState,
}

impl From<Translation> for HeadTranslationSignal {
    fn from(t: Translation) -> Self {
        match t.state {
            HeadTranslationState::Tracked => Self::tracked(t.x_meters, t.y_meters, t.z_meters),
            HeadTranslationState::Degraded => Self::degraded(t.x_meters, t.y_meters, t.z_meters),
            HeadTranslationState::Unavailable => Self::UNAVAILABLE,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Gaze {
    horizontal: f32,
    vertical: f32,
    confidence: f32,
    state: GazeTrackingState,
}

impl From<Gaze> for GazeSignal {
    fn from(g: Gaze) -> Self {
        match g.state {
            GazeTrackingState::Tracked => Self::tracked(g.horizontal, g.vertical, g.confidence),
            GazeTrackingState::Degraded => Self::degraded(g.horizontal, g.vertical, g.confidence),
            GazeTrackingState::Unavailable => Self::UNAVAILABLE,
        }
    }
}

mod detailed {
    use super::*;

    pub fn serialize<S: serde::Serializer>(
        value: &Option<Arkit52Coefficients>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        value.as_ref().map(|v| v.as_array().as_slice()).serialize(s)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Arkit52Coefficients>, D::Error> {
        use serde::de::Error;
        Option::<Vec<f32>>::deserialize(d)?
            .map(|v| {
                let values = v
                    .try_into()
                    .map_err(|_| D::Error::custom("expected 52 face coefficients"))?;
                Arkit52Coefficients::try_from_array(values)
                    .map_err(|e| D::Error::custom(format!("{e:?}")))
            })
            .transpose()
    }
}

impl Face {
    fn translation(frame: &AvatarControlFrame) -> Translation {
        let value = frame.head_translation;
        Translation {
            x_meters: value.x_meters(),
            y_meters: value.y_meters(),
            z_meters: value.z_meters(),
            state: value.state(),
        }
    }
}

impl Face {
    fn gaze(frame: &AvatarControlFrame) -> Gaze {
        let value = frame.gaze;
        Gaze {
            horizontal: value.horizontal(),
            vertical: value.vertical(),
            confidence: value.confidence(),
            state: value.state(),
        }
    }
}

impl From<Face> for AvatarControlFrame {
    fn from(f: Face) -> Self {
        Self {
            source_seq: f.source_seq,
            captured_at: f.captured_at,
            produced_at: f.produced_at,
            confidence: f.confidence,
            state: f.state,
            head: f.head,
            head_translation: f.head_translation.into(),
            gaze: f.gaze.into(),
            expressions: f.expressions,
            detailed_face: f.detailed_face,
        }
    }
}
