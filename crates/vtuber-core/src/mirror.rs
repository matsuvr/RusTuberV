//! One mirror convention at the presentation boundary. Inference and tracking
//! retain their canonical camera basis; pixel preview mirroring is independent.
//! A position reflects as R v, a cross-product normal as det(R) R n. Exchanging
//! anatomical sides must accompany reflection of an arm or face pair.

use crate::arm_tracking::{ArmBlendWeights, ArmTrackingTarget, ArmTrackingTargets, HandFingerPose};
use crate::{ARKIT_NON_TONGUE_LEFT_RIGHT_PAIRS, Arkit52Coefficients, ExpressionCoefficients};

/// Optional sagittal reflection shared by every avatar input channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionMirror {
    enabled: bool,
}

impl MotionMirror {
    /// Uses the user-selected presentation policy.
    #[must_use]
    pub const fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Reflects a horizontal position, yaw or roll coordinate.
    #[must_use]
    pub const fn horizontal(self, value: f32) -> f32 {
        if self.enabled { -value } else { value }
    }

    /// Reflects a position or polar direction across X.
    #[must_use]
    pub const fn polar(self, [x, y, z]: [f32; 3]) -> [f32; 3] {
        [self.horizontal(x), y, z]
    }

    /// Reflects a cross-product normal, including the determinant sign.
    #[must_use]
    pub const fn axial(self, [x, y, z]: [f32; 3]) -> [f32; 3] {
        [x, self.horizontal(y), self.horizontal(z)]
    }

    /// Transforms an arm pair and its channel weights together, exactly once.
    #[must_use]
    pub fn arms(
        self,
        targets: ArmTrackingTargets,
        weights: ArmBlendWeights,
    ) -> (ArmTrackingTargets, ArmBlendWeights) {
        if self.enabled {
            (targets.mirrored(), weights.mirrored())
        } else {
            (targets, weights)
        }
    }

    /// Exchanges anatomical face channels without changing symmetric channels.
    #[must_use]
    pub fn expressions(
        self,
        mut expressions: ExpressionCoefficients,
        mut detailed: Option<Arkit52Coefficients>,
    ) -> (ExpressionCoefficients, Option<Arkit52Coefficients>) {
        if self.enabled {
            std::mem::swap(&mut expressions.blink_left, &mut expressions.blink_right);
            if let Some(coefficients) = &mut detailed {
                for &(left, right) in ARKIT_NON_TONGUE_LEFT_RIGHT_PAIRS {
                    coefficients.swap(left, right);
                }
            }
        }
        (expressions, detailed)
    }
}

impl ArmTrackingTarget {
    /// Reflects positions, axial palm normal and signed finger bends. A pair
    /// must also exchange anatomical sides through [`ArmTrackingTargets::mirrored`].
    #[must_use]
    pub fn mirrored(self) -> Self {
        let mirror = MotionMirror::new(true);
        Self {
            wrist: mirror.polar(self.wrist),
            elbow_pole: mirror.polar(self.elbow_pole),
            palm_normal: self.palm_normal.map(|normal| mirror.axial(normal)),
            palm_forward: self.palm_forward.map(|forward| mirror.polar(forward)),
            fingers: self.fingers.map(|fingers| HandFingerPose {
                fingers: fingers
                    .fingers
                    .map(|angles| angles.map(|angle| mirror.horizontal(angle))),
                spread: fingers.spread,
                thumb: fingers.thumb.map(|angle| mirror.horizontal(angle)),
                thumb_spread: fingers.thumb_spread,
            }),
        }
    }
}

impl ArmTrackingTargets {
    /// Reflects positions and exchanges anatomical sides once.
    #[must_use]
    pub fn mirrored(self) -> Self {
        Self {
            left: self.right.map(ArmTrackingTarget::mirrored),
            right: self.left.map(ArmTrackingTarget::mirrored),
        }
    }
}

impl ArmBlendWeights {
    /// Exchanges anatomical sides with no change to the weights.
    #[must_use]
    pub fn mirrored(self) -> Self {
        Self {
            left: self.right,
            right: self.left,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflected_landmarks_and_reflected_palm_normal_agree() {
        let cross = |[ax, ay, az]: [f32; 3], [bx, by, bz]: [f32; 3]| {
            [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
        };
        // A tilted palm exercises all three components. Treating the normal
        // as a position instead would reverse which side of the hand is shown.
        let index = [0.4, 0.8, -0.2];
        let little = [-0.3, 0.7, 0.1];
        for enabled in [false, true] {
            let mirror = MotionMirror::new(enabled);
            let normal = cross(index, little);
            assert_eq!(
                cross(mirror.polar(index), mirror.polar(little)),
                mirror.axial(normal)
            );
            assert_eq!(mirror.polar(mirror.polar(index)), index);
            assert_eq!(mirror.axial(mirror.axial(normal)), normal);
        }
    }
}
