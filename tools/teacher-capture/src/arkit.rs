// Unit tests may use unwrap/expect/panic (AGENTS.md: Production Rust panic policy).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]
//! ARKit face-anchor callback to canonical teacher record conversion
//! (GNM #68.2b).
//!
//! The iOS face-tracking callback reports a name-keyed blendshape dictionary
//! plus a head transform. This module converts one callback sample into a
//! canonical-order teacher frame record with fail-closed semantics:
//!
//! - channel names map explicitly onto canonical ARKit52 order; unknown,
//!   duplicate, and missing channels are typed errors (never silent fills),
//! - non-finite or out-of-range coefficient values are typed errors,
//! - frame sequence and monotonic timestamps must be strictly increasing,
//! - the head transform coordinate convention and units are fixed constants
//!   recorded beside every record.

use serde::{Deserialize, Serialize};

/// Canonical lowerCamel channel names exactly as the ARKit callback reports
/// them, in canonical ARKit52 index order.
pub const ARKIT_CALLBACK_CHANNEL_NAMES: [&str; 52] = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
];

/// Fixed head-transform convention recorded in serialized records.
pub const HEAD_TRANSFORM_CONVENTION: &str =
    "ARKit world: right-handed, Y-up, camera-forward -Z; translation in meters";

/// Typed conversion failures for one callback sample.
#[derive(Clone, Debug, PartialEq)]
pub enum TeacherRecordError {
    /// A channel name did not match any canonical ARKit52 channel.
    UnknownChannel {
        /// Offending callback name.
        name: String,
    },
    /// The same channel appeared more than once in one callback dictionary.
    DuplicateChannel {
        /// Duplicated canonical channel.
        name: &'static str,
    },
    /// At least one canonical channel was absent from the dictionary.
    MissingChannels {
        /// Number of missing channels.
        count: usize,
    },
    /// A coefficient was NaN or infinite.
    NonFiniteCoefficient {
        /// Channel carrying the bad value.
        name: String,
    },
    /// A coefficient fell outside `[0, 1]`.
    OutOfRangeCoefficient {
        /// Channel carrying the value.
        name: String,
        /// Rejected value.
        value: f32,
    },
    /// The rotation quaternion was not close to unit length.
    NonUnitQuaternion {
        /// Measured quaternion norm.
        norm: f32,
    },
    /// Frame sequences must be strictly increasing without duplicates.
    NonIncreasingFrameSeq {
        /// Previous sequence.
        previous: u64,
        /// Offending sequence.
        current: u64,
    },
    /// Timestamps must be strictly increasing on the monotonic timeline.
    NonMonotonicTimestamp {
        /// Previous timestamp in microseconds.
        previous: u64,
        /// Offending timestamp in microseconds.
        current: u64,
    },
}

/// Raw name-keyed coefficients exactly as delivered by the callback.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RawBlendshapeDictionary {
    /// `(callback_name, value)` pairs in callback delivery order.
    pub entries: Vec<(String, f32)>,
}

/// Serialized canonical teacher frame payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TeacherFramePayload {
    /// Canonical-order coefficients, index-aligned with
    /// [`ARKIT_CALLBACK_CHANNEL_NAMES`].
    pub coefficients_canonical: Vec<f32>,
    /// Unit rotation quaternion `(w, x, y, z)` under
    /// [`HEAD_TRANSFORM_CONVENTION`].
    pub rotation_quaternion_wxyz: [f32; 4],
    /// Translation in meters under [`HEAD_TRANSFORM_CONVENTION`].
    pub translation_meters: [f32; 3],
    /// Fixed convention string so units/handedness are never reinterpreted.
    pub head_transform_convention: String,
}

/// Stateful converter enforcing strict identity ordering across callbacks.
#[derive(Debug)]
pub struct TeacherFrameConverter {
    last_frame_seq: Option<u64>,
    last_timestamp_micros: Option<u64>,
}

impl Default for TeacherFrameConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl TeacherFrameConverter {
    /// Creates a converter at session start.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_frame_seq: None,
            last_timestamp_micros: None,
        }
    }

    /// Converts one callback sample into a serializable canonical record.
    ///
    /// # Errors
    ///
    /// Returns a typed [`TeacherRecordError`] for unknown/duplicate/missing
    /// channels, non-finite or out-of-range values, invalid quaternions, and
    /// regressed identities. Nothing is repaired silently.
    pub fn convert(
        &mut self,
        frame_seq: u64,
        timestamp_micros: u64,
        raw: &RawBlendshapeDictionary,
        rotation_quaternion_wxyz: [f32; 4],
        translation_meters: [f32; 3],
    ) -> Result<(u64, u64, TeacherFramePayload), TeacherRecordError> {
        if let Some(previous_seq) = self.last_frame_seq {
            if frame_seq <= previous_seq {
                return Err(TeacherRecordError::NonIncreasingFrameSeq {
                    previous: previous_seq,
                    current: frame_seq,
                });
            }
        }
        if let Some(previous_time) = self.last_timestamp_micros {
            if timestamp_micros <= previous_time {
                return Err(TeacherRecordError::NonMonotonicTimestamp {
                    previous: previous_time,
                    current: timestamp_micros,
                });
            }
        }

        let mut coefficients = [0.0_f32; 52];
        let mut seen = [false; 52];
        for (name, value) in &raw.entries {
            let Some(index) = ARKIT_CALLBACK_CHANNEL_NAMES
                .iter()
                .position(|candidate| candidate == name)
            else {
                return Err(TeacherRecordError::UnknownChannel { name: name.clone() });
            };
            if seen[index] {
                return Err(TeacherRecordError::DuplicateChannel {
                    name: ARKIT_CALLBACK_CHANNEL_NAMES[index],
                });
            }
            seen[index] = true;
            if !value.is_finite() {
                return Err(TeacherRecordError::NonFiniteCoefficient { name: name.clone() });
            }
            if !(0.0..=1.0).contains(value) {
                return Err(TeacherRecordError::OutOfRangeCoefficient {
                    name: name.clone(),
                    value: *value,
                });
            }
            coefficients[index] = *value;
        }
        let missing = seen.iter().filter(|present| !*present).count();
        if missing > 0 {
            return Err(TeacherRecordError::MissingChannels { count: missing });
        }

        let norm = (rotation_quaternion_wxyz[0] * rotation_quaternion_wxyz[0]
            + rotation_quaternion_wxyz[1] * rotation_quaternion_wxyz[1]
            + rotation_quaternion_wxyz[2] * rotation_quaternion_wxyz[2]
            + rotation_quaternion_wxyz[3] * rotation_quaternion_wxyz[3])
            .sqrt();
        if !(0.95..=1.05).contains(&norm) {
            return Err(TeacherRecordError::NonUnitQuaternion { norm });
        }

        self.last_frame_seq = Some(frame_seq);
        self.last_timestamp_micros = Some(timestamp_micros);

        Ok((
            frame_seq,
            timestamp_micros,
            TeacherFramePayload {
                coefficients_canonical: coefficients.to_vec(),
                rotation_quaternion_wxyz,
                translation_meters,
                head_transform_convention: HEAD_TRANSFORM_CONVENTION.to_owned(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_dictionary(value_at: impl Fn(usize) -> f32) -> RawBlendshapeDictionary {
        RawBlendshapeDictionary {
            entries: ARKIT_CALLBACK_CHANNEL_NAMES
                .iter()
                .enumerate()
                .map(|(index, name)| ((*name).to_owned(), value_at(index)))
                .collect(),
        }
    }

    fn unit_quat() -> [f32; 4] {
        [1.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn synthetic_callback_serializes_to_canonical_order() {
        let mut converter = TeacherFrameConverter::new();
        let jaw_index = ARKIT_CALLBACK_CHANNEL_NAMES
            .iter()
            .position(|name| *name == "jawOpen")
            .expect("jawOpen exists");
        let raw = full_dictionary(|index| if index == jaw_index { 0.6 } else { 0.01 });

        let (seq, time, payload) = converter
            .convert(7, 116_669, &raw, unit_quat(), [0.0, 0.0, 0.5])
            .expect("converts");
        assert_eq!((seq, time), (7, 116_669));
        // Canonical order is preserved by position, not by callback order.
        assert!((payload.coefficients_canonical[jaw_index] - 0.6).abs() < 1e-6);
        assert!((payload.coefficients_canonical[0] - 0.01).abs() < 1e-6);
        assert_eq!(payload.head_transform_convention, HEAD_TRANSFORM_CONVENTION);

        // Round-trip through JSON keeps values and convention stable.
        let json = serde_json::to_string(&payload).unwrap();
        let parsed: TeacherFramePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, payload);
    }

    #[test]
    fn callback_order_does_not_change_the_canonical_vector() {
        let mut converter_a = TeacherFrameConverter::new();
        let mut converter_b = TeacherFrameConverter::new();
        let forward = full_dictionary(|index| index as f32 / 100.0);
        let mut reversed_entries = forward.entries.clone();
        reversed_entries.reverse();
        let reversed = RawBlendshapeDictionary {
            entries: reversed_entries,
        };

        let (_, _, payload_a) = converter_a
            .convert(1, 10, &forward, unit_quat(), [0.0; 3])
            .unwrap();
        let (_, _, payload_b) = converter_b
            .convert(1, 10, &reversed, unit_quat(), [0.0; 3])
            .unwrap();
        assert_eq!(
            payload_a.coefficients_canonical,
            payload_b.coefficients_canonical
        );
    }

    #[test]
    fn unknown_duplicate_missing_and_bad_values_fail_closed() {
        let mut converter = TeacherFrameConverter::new();

        let mut unknown = full_dictionary(|_| 0.5);
        unknown.entries.push(("_neutral".to_owned(), 0.2));
        assert!(matches!(
            converter.convert(1, 10, &unknown, unit_quat(), [0.0; 3]),
            Err(TeacherRecordError::UnknownChannel { .. })
        ));

        let mut duplicate = full_dictionary(|_| 0.5);
        duplicate.entries.push(("jawOpen".to_owned(), 0.4));
        assert!(matches!(
            converter.convert(1, 10, &duplicate, unit_quat(), [0.0; 3]),
            Err(TeacherRecordError::DuplicateChannel { name: "jawOpen" })
        ));

        let partial = RawBlendshapeDictionary {
            entries: vec![("jawOpen".to_owned(), 0.4)],
        };
        assert!(matches!(
            converter.convert(1, 10, &partial, unit_quat(), [0.0; 3]),
            Err(TeacherRecordError::MissingChannels { count: 51 })
        ));

        let mut nan_value = full_dictionary(|_| 0.5);
        nan_value.entries[0].1 = f32::NAN;
        assert!(matches!(
            converter.convert(1, 10, &nan_value, unit_quat(), [0.0; 3]),
            Err(TeacherRecordError::NonFiniteCoefficient { .. })
        ));

        let mut out_of_range = full_dictionary(|_| 0.5);
        out_of_range.entries[0].1 = 1.5;
        assert!(matches!(
            converter.convert(1, 10, &out_of_range, unit_quat(), [0.0; 3]),
            Err(TeacherRecordError::OutOfRangeCoefficient { value, .. }) if value == 1.5
        ));

        assert!(matches!(
            converter.convert(
                1,
                10,
                &full_dictionary(|_| 0.5),
                [2.0, 0.0, 0.0, 0.0],
                [0.0; 3]
            ),
            Err(TeacherRecordError::NonUnitQuaternion { .. })
        ));
    }

    #[test]
    fn identity_ordering_is_strictly_increasing() {
        let mut converter = TeacherFrameConverter::new();
        let raw = full_dictionary(|_| 0.5);
        converter
            .convert(1, 10, &raw, unit_quat(), [0.0; 3])
            .unwrap();

        assert!(matches!(
            converter.convert(1, 20, &raw, unit_quat(), [0.0; 3]),
            Err(TeacherRecordError::NonIncreasingFrameSeq {
                previous: 1,
                current: 1
            })
        ));
        assert!(matches!(
            converter.convert(2, 10, &raw, unit_quat(), [0.0; 3]),
            Err(TeacherRecordError::NonMonotonicTimestamp {
                previous: 10,
                current: 10
            })
        ));
        converter
            .convert(2, 20, &raw, unit_quat(), [0.0; 3])
            .unwrap();
    }
}
