//! Adapter from region GNM geometry features to auxiliary residual inputs
//! (Issue #55.4 / #89).
//!
//! This module converts [`vtuber_gnm`] region auxiliary geometry features
//! (eye aperture, jaw/mouth, brow) into engine-neutral
//! [`PredictedAuxiliaryFeature`] values and evaluates the exact-frame
//! auxiliary residual once through the existing
//! [`evaluate_auxiliary_expression_loss`] contract.
//!
//! Mapping policy (fixed by this adapter, one geometry feature drives at most
//! one semantic so no semantic is ever duplicated):
//!
//! - Each eye-aperture delta becomes its anatomical `EyeClosure*` prediction
//!   with a negated value (closure-positive). The perfectly correlated
//!   `EyeWide*` channels are intentionally left unpredicted; configuring them
//!   surfaces as `missing_prediction_channels`, never fabricated values.
//! - `jaw_open` maps to `JawOpen`; `jaw_forward` (negative = forward) maps
//!   negated to `JawForward`; the signed lateral delta maps to `JawLeft`
//!   (positive = toward the subject's left); `JawRight` stays unpredicted.
//! - `corner_lift` (positive = smile-like lift) maps symmetrically to both
//!   `MouthSmile*` channels; frown/pucker/funnel/stretch stay unpredicted.
//! - Mean inner rise maps to `BrowInnerUp`; per-side lower/outer deltas map
//!   to their anatomical `BrowDown*` / `BrowOuterUp*` channels.
//!
//! Unavailable features (`None`) are skipped: the channel simply receives no
//! prediction and is reported through the existing missing-prediction
//! diagnostic. Values are never invented.

use crate::auxiliary_expression::{
    AuxiliaryExpressionError, AuxiliaryExpressionObservation, AuxiliaryExpressionSemantic,
    AuxiliaryLossConfig, AuxiliaryLossDiagnostics, PredictedAuxiliaryFeature,
    evaluate_auxiliary_expression_loss, validate_auxiliary_source_alignment,
};
use vtuber_gnm::{
    BrowAuxFeatures, DenseCorrespondenceSet, DenseRegionGroups, EyeAuxFeatures, FixedGnmIdentity,
    GnmIdentityCalibration, GnmModel, GnmReprojectionError, MouthAuxFeatures,
};

/// Borrowed region geometry features for one reconstructed frame.
#[derive(Clone, Copy, Debug)]
pub struct AuxiliaryGeometryFeatures<'a> {
    /// Eye-aperture features.
    pub eyes: &'a EyeAuxFeatures,
    /// Jaw/mouth features.
    pub mouth: &'a MouthAuxFeatures,
    /// Brow features.
    pub brows: &'a BrowAuxFeatures,
}

impl<'a> AuxiliaryGeometryFeatures<'a> {
    /// Converts the available geometry into one prediction per semantic.
    ///
    /// Semantics are unique by construction, so downstream duplicate
    /// validation cannot fire from geometry alone.
    pub fn predictions(&self) -> Vec<PredictedAuxiliaryFeature> {
        let mut predictions = Vec::new();
        let mut push = |semantic: AuxiliaryExpressionSemantic, value: f32| {
            predictions.push(PredictedAuxiliaryFeature { semantic, value });
        };

        // Eyes: aperture delta is positive-wide, closure is positive-closed.
        for (semantic, aperture) in [
            (
                AuxiliaryExpressionSemantic::EyeClosureRight,
                self.eyes.right.normalized_delta,
            ),
            (
                AuxiliaryExpressionSemantic::EyeClosureLeft,
                self.eyes.left.normalized_delta,
            ),
        ] {
            push(semantic, -aperture);
        }

        // Jaw/mouth.
        if let Some(jaw_open) = self.mouth.jaw_open {
            push(AuxiliaryExpressionSemantic::JawOpen, jaw_open);
        }
        // The measured chin-to-nose-tip delta shrinks (goes negative) when the
        // jaw moves forward, while the semantic is positive-forward.
        if let Some(jaw_forward) = self.mouth.jaw_forward {
            push(AuxiliaryExpressionSemantic::JawForward, -jaw_forward);
        }
        // Signed lateral delta is positive toward the subject's left.
        if let Some(jaw_lateral) = self.mouth.jaw_lateral {
            push(AuxiliaryExpressionSemantic::JawLeft, jaw_lateral);
        }
        // Symmetric corner lift feeds both anatomical smile channels once.
        if let Some(corner_lift) = self.mouth.corner_lift {
            push(AuxiliaryExpressionSemantic::MouthSmileLeft, corner_lift);
            push(AuxiliaryExpressionSemantic::MouthSmileRight, corner_lift);
        }

        // Brows: the single inner-up channel takes the mean of the available
        // sides instead of duplicating one side's evidence.
        let inner_sides = [self.brows.right.inner_rise, self.brows.left.inner_rise];
        let inner_count = inner_sides.iter().filter(|side| side.is_some()).count();
        if inner_count > 0 {
            let sum: f32 = inner_sides.iter().filter_map(|side| *side).sum();
            push(
                AuxiliaryExpressionSemantic::BrowInnerUp,
                sum / inner_count as f32,
            );
        }
        for (down, outer, side) in [
            (
                self.brows.right.brow_lower,
                self.brows.right.outer_rise,
                AuxiliaryExpressionSemantic::BrowDownRight,
            ),
            (
                self.brows.left.brow_lower,
                self.brows.left.outer_rise,
                AuxiliaryExpressionSemantic::BrowDownLeft,
            ),
        ] {
            if let Some(brow_lower) = down {
                push(side, brow_lower);
            }
            let outer_semantic = match side {
                AuxiliaryExpressionSemantic::BrowDownRight => {
                    AuxiliaryExpressionSemantic::BrowOuterUpRight
                }
                _ => AuxiliaryExpressionSemantic::BrowOuterUpLeft,
            };
            if let Some(outer_rise) = outer {
                push(outer_semantic, outer_rise);
            }
        }

        predictions
    }

    /// Evaluates the auxiliary residual exactly once for this frame.
    ///
    /// When an observation is supplied, exact source sequence/timestamp
    /// alignment against the primary dense frame is validated through the
    /// existing contract before any residual work.
    ///
    /// # Errors
    ///
    /// Returns the underlying typed error for alignment mismatch,
    /// invalid configuration, duplicate/non-finite predictions, or a
    /// non-finite accumulated loss.
    pub fn evaluate_residual(
        &self,
        dense_source_seq: u64,
        dense_captured_at_micros: u64,
        observation: Option<&AuxiliaryExpressionObservation>,
        config: AuxiliaryLossConfig,
    ) -> Result<AuxiliaryLossDiagnostics, AuxiliaryExpressionError> {
        if let Some(observation) = observation {
            validate_auxiliary_source_alignment(
                dense_source_seq,
                dense_captured_at_micros,
                observation,
            )?;
        }
        evaluate_auxiliary_expression_loss(observation, &self.predictions(), config)
    }
}

/// Caller-side auxiliary objective for the bounded expression/joint step
/// (Issue #64.2d / #121): pairs the exact-frame MediaPipe observation with
/// geometry-derived predictions recomputed at any candidate dynamic state and
/// reports the weighted robust loss plus a finite-difference gradient.
///
/// Construction validates exact source alignment between the auxiliary
/// observation and the dense frame being solved, so a stale or mismatched
/// observation can never enter the solver.
///
/// # Weighting contract
///
/// The effective solver weight of this term is the product of
/// [`AuxiliaryLossConfig::auxiliary_weight`] and the solver-level weight passed
/// alongside the trait object. Configure exactly one of the two; the other
/// should stay at `1.0` / be left at zero respectively. Per-channel
/// reliability, relative weight, and the Huber bound all come from the existing
/// auxiliary-loss contract; this objective adds no new output surface (in
/// particular, no ARKit52 output).
pub struct GeometryAuxiliaryObjective<'a> {
    /// Exact-frame MediaPipe observation (alignment proven at construction).
    observation: &'a AuxiliaryExpressionObservation,
    /// Geometry inputs used to recompute predictions at candidate states.
    model: &'a GnmModel,
    identity: &'a FixedGnmIdentity,
    mapping: &'a DenseCorrespondenceSet,
    calibration: &'a GnmIdentityCalibration,
    /// Eyelid-ring topology required by the eye-aperture features.
    eye_groups: &'a DenseRegionGroups,
    /// Robust-loss configuration (absolute `w_aux`, Huber delta,
    /// disagreement bound).
    loss_config: AuxiliaryLossConfig,
    /// Finite-difference step for the gradient, per parameter.
    step: f32,
}

impl<'a> GeometryAuxiliaryObjective<'a> {
    /// Assembles and validates one frame-bound auxiliary objective.
    ///
    /// Validation is fail-closed: the auxiliary observation must carry exactly
    /// the dense frame's source sequence and capture timestamp, and the
    /// finite-difference step must be finite and positive.
    ///
    /// # Errors
    ///
    /// Returns [`AuxiliaryExpressionError::SourceSequenceMismatch`]
    /// / [`AuxiliaryExpressionError::CaptureTimestampMismatch`] on alignment
    /// failure and [`AuxiliaryExpressionError::InvalidConfig`] for an invalid
    /// gradient step.
    #[allow(clippy::too_many_arguments)]
    // every argument is a distinct frame/model reference of the objective contract
    pub fn new(
        dense_source_seq: u64,
        dense_captured_at_micros: u64,
        observation: &'a AuxiliaryExpressionObservation,
        model: &'a GnmModel,
        identity: &'a FixedGnmIdentity,
        mapping: &'a DenseCorrespondenceSet,
        calibration: &'a GnmIdentityCalibration,
        eye_groups: &'a DenseRegionGroups,
        loss_config: AuxiliaryLossConfig,
        step: f32,
    ) -> Result<Self, AuxiliaryExpressionError> {
        validate_auxiliary_source_alignment(
            dense_source_seq,
            dense_captured_at_micros,
            observation,
        )?;
        if !step.is_finite() || step <= 0.0 {
            return Err(AuxiliaryExpressionError::InvalidConfig(
                "auxiliary gradient step must be finite and positive",
            ));
        }
        Ok(Self {
            observation,
            model,
            identity,
            mapping,
            calibration,
            eye_groups,
            loss_config,
            step,
        })
    }
}

/// Computes the weighted auxiliary loss for one candidate dynamic state.
fn geometry_auxiliary_loss(
    objective: &GeometryAuxiliaryObjective<'_>,
    expression_values: &[f32],
    joint_rotations: &[[f32; 3]],
    joint_translation: [f32; 3],
) -> Result<f32, GnmReprojectionError> {
    use vtuber_gnm::{
        GnmExpressionState, GnmJointState, compute_brow_aux_features,
        compute_eye_aperture_features, compute_mouth_aux_features,
    };
    let expression = GnmExpressionState::new(
        expression_values.to_vec(),
        objective.model.expression_dimension(),
    )
    .map_err(GnmReprojectionError::Model)?;
    let joints = GnmJointState::new(
        joint_rotations.to_vec(),
        joint_translation,
        objective.model.joint_count(),
    )
    .map_err(GnmReprojectionError::Model)?;

    let eyes = compute_eye_aperture_features(
        objective.model,
        objective.identity.state(),
        &expression,
        &joints,
        objective.mapping,
        objective.eye_groups,
        objective.calibration,
    )
    .map_err(|_| GnmReprojectionError::InvalidConfig("auxiliary eye geometry evaluation failed"))?;
    let mouth = compute_mouth_aux_features(
        objective.model,
        objective.identity.state(),
        &expression,
        &joints,
        objective.mapping,
        objective.calibration,
    )
    .map_err(|_| {
        GnmReprojectionError::InvalidConfig("auxiliary mouth geometry evaluation failed")
    })?;
    let brows = compute_brow_aux_features(
        objective.model,
        objective.identity.state(),
        &expression,
        &joints,
        objective.mapping,
        objective.calibration,
    )
    .map_err(|_| {
        GnmReprojectionError::InvalidConfig("auxiliary brow geometry evaluation failed")
    })?;

    let features = AuxiliaryGeometryFeatures {
        eyes: &eyes,
        mouth: &mouth,
        brows: &brows,
    };
    let diagnostics = evaluate_auxiliary_expression_loss(
        Some(objective.observation),
        &features.predictions(),
        objective.loss_config,
    )
    .map_err(|_| GnmReprojectionError::InvalidConfig("auxiliary loss evaluation failed"))?;
    Ok(diagnostics.weighted_loss)
}

impl vtuber_gnm::AuxiliaryObjectiveTerm for GeometryAuxiliaryObjective<'_> {
    fn evaluate(
        &self,
        expression_values: &[f32],
        joint_rotations: &[[f32; 3]],
        joint_translation: [f32; 3],
    ) -> Result<vtuber_gnm::AuxiliaryTermEvaluation, GnmReprojectionError> {
        // Step validity is enforced at construction; evaluate must not depend
        // on it silently.
        if !self.step.is_finite() || self.step <= 0.0 {
            return Err(GnmReprojectionError::InvalidConfig(
                "auxiliary gradient step must be finite and positive",
            ));
        }
        let loss =
            geometry_auxiliary_loss(self, expression_values, joint_rotations, joint_translation)?;
        if !loss.is_finite() {
            return Err(GnmReprojectionError::InvalidConfig(
                "auxiliary loss became non-finite",
            ));
        }

        // Forward-difference gradient over expression coefficients.
        let mut expression_gradient = Vec::with_capacity(expression_values.len());
        for index in 0..expression_values.len() {
            let mut perturbed = expression_values.to_vec();
            #[allow(clippy::indexing_slicing)] // index < len by loop bounds
            {
                perturbed[index] += self.step;
            }
            let perturbed_loss =
                geometry_auxiliary_loss(self, &perturbed, joint_rotations, joint_translation)?;
            expression_gradient.push((perturbed_loss - loss) / self.step);
        }

        // Forward-difference gradient over joint rotations and translation.
        let mut joint_gradient = Vec::new();
        for joint in 0..joint_rotations.len() {
            for axis in 0..3 {
                let mut perturbed = joint_rotations.to_vec();
                #[allow(clippy::indexing_slicing)] // joint < len, axis < 3 by loop bounds
                {
                    perturbed[joint][axis] += self.step;
                }
                let perturbed_loss = geometry_auxiliary_loss(
                    self,
                    expression_values,
                    &perturbed,
                    joint_translation,
                )?;
                joint_gradient.push((perturbed_loss - loss) / self.step);
            }
        }
        for axis in 0..3 {
            let mut perturbed_translation = joint_translation;
            #[allow(clippy::indexing_slicing)] // axis < 3 by loop bounds
            {
                perturbed_translation[axis] += self.step;
            }
            let perturbed_loss = geometry_auxiliary_loss(
                self,
                expression_values,
                joint_rotations,
                perturbed_translation,
            )?;
            joint_gradient.push((perturbed_loss - loss) / self.step);
        }

        Ok(vtuber_gnm::AuxiliaryTermEvaluation {
            loss,
            expression_gradient,
            joint_gradient,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auxiliary_expression::{
        AuxChannelReliability, AuxiliaryChannelConfig, AuxiliaryExpressionGroup,
    };
    use vtuber_gnm::{
        AuxiliaryObjectiveTerm, DenseArray, DenseMappingVersion, GNM_HEAD_V3_EXPRESSION_DIM,
        GNM_HEAD_V3_IDENTITY_DIM, GNM_HEAD_V3_VERSION, GnmIdentityCalibration, GnmModelData,
        GnmSparseVertices, IdentityFitDiagnostics, MEDIAPIPE_FACE_LANDMARK_COUNT,
    };
    use vtuber_gnm::{BrowSideAuxFeatures, EyeApertureFeature};

    fn eye_features(right: f32, left: f32) -> EyeAuxFeatures {
        let side = |delta: f32| EyeApertureFeature {
            side: vtuber_gnm::AnatomicalSide::Right,
            current_aperture: 1.0,
            neutral_aperture: 1.0,
            normalized_delta: delta,
        };
        EyeAuxFeatures {
            right: side(right),
            left: side(left),
        }
    }

    fn mouth_features(
        jaw_open: Option<f32>,
        jaw_forward: Option<f32>,
        jaw_lateral: Option<f32>,
        corner_lift: Option<f32>,
    ) -> MouthAuxFeatures {
        MouthAuxFeatures {
            jaw_open,
            jaw_forward,
            jaw_lateral,
            width_delta: None,
            corner_lift,
        }
    }

    fn brow_features(
        side: vtuber_gnm::AnatomicalSide,
        inner: Option<f32>,
        lower: Option<f32>,
        outer: Option<f32>,
    ) -> BrowSideAuxFeatures {
        BrowSideAuxFeatures {
            side,
            inner_rise: inner,
            brow_lower: lower,
            outer_rise: outer,
        }
    }

    fn features<'a>(
        eyes: &'a EyeAuxFeatures,
        mouth: &'a MouthAuxFeatures,
        brows: &'a BrowAuxFeatures,
    ) -> AuxiliaryGeometryFeatures<'a> {
        AuxiliaryGeometryFeatures { eyes, mouth, brows }
    }

    fn brows(right: BrowSideAuxFeatures, left: BrowSideAuxFeatures) -> BrowAuxFeatures {
        BrowAuxFeatures { right, left }
    }

    fn find(
        predictions: &[PredictedAuxiliaryFeature],
        semantic: AuxiliaryExpressionSemantic,
    ) -> f32 {
        predictions
            .iter()
            .find(|prediction| prediction.semantic == semantic)
            .unwrap()
            .value
    }

    #[test]
    fn every_semantic_is_emitted_at_most_once() {
        let eyes = eye_features(0.1, -0.2);
        let mouth = mouth_features(Some(0.3), Some(-0.4), Some(0.05), Some(0.6));
        let right = brow_features(
            vtuber_gnm::AnatomicalSide::Right,
            Some(0.1),
            Some(0.2),
            Some(0.3),
        );
        let left = brow_features(vtuber_gnm::AnatomicalSide::Left, Some(0.3), None, Some(0.5));
        let brows_set = brows(right, left);
        let all = features(&eyes, &mouth, &brows_set);

        let predictions = all.predictions();
        // Duplicate validation inside the evaluator must accept the set.
        assert!(
            crate::auxiliary_expression::evaluate_auxiliary_expression_loss(
                None,
                &predictions,
                AuxiliaryLossConfig::new(1.0, 0.2, 10.0).unwrap(),
            )
            .is_ok()
        );
        // And the count equals unique semantics actually covered.
        let mut seen: Vec<_> = predictions.iter().map(|p| p.semantic).collect();
        seen.sort_by_key(|semantic| {
            AuxiliaryExpressionSemantic::ALL
                .iter()
                .position(|candidate| candidate == semantic)
                .unwrap()
        });
        seen.dedup();
        assert_eq!(seen.len(), predictions.len());

        // Sign conventions.
        assert!(
            (find(&predictions, AuxiliaryExpressionSemantic::EyeClosureRight) + 0.1).abs() < 1.0e-6
        );
        assert!(
            (find(&predictions, AuxiliaryExpressionSemantic::EyeClosureLeft) - 0.2).abs() < 1.0e-6
        );
        assert!((find(&predictions, AuxiliaryExpressionSemantic::JawOpen) - 0.3).abs() < 1.0e-6);
        assert!((find(&predictions, AuxiliaryExpressionSemantic::JawForward) - 0.4).abs() < 1.0e-6);
        assert!((find(&predictions, AuxiliaryExpressionSemantic::JawLeft) - 0.05).abs() < 1.0e-6);
        assert!(
            (find(&predictions, AuxiliaryExpressionSemantic::MouthSmileLeft) - 0.6).abs() < 1.0e-6
        );
        assert!(
            (find(&predictions, AuxiliaryExpressionSemantic::BrowInnerUp) - 0.2).abs() < 1.0e-6
        );
        assert!(find(&predictions, AuxiliaryExpressionSemantic::BrowDownRight) - 0.2 < 1.0e-6);
        assert!(find(&predictions, AuxiliaryExpressionSemantic::BrowOuterUpLeft) - 0.5 < 1.0e-6);
        assert!(
            !predictions
                .iter()
                .any(|p| p.semantic == AuxiliaryExpressionSemantic::JawRight)
        );
        assert!(
            !predictions
                .iter()
                .any(|p| p.semantic == AuxiliaryExpressionSemantic::EyeWideLeft)
        );
    }

    #[test]
    fn unavailable_geometry_is_skipped_not_fabricated() {
        let eyes = eye_features(0.0, 0.0);
        let mouth = mouth_features(None, None, None, None);
        let none_brow = |side| brow_features(side, None, None, None);
        let right = none_brow(vtuber_gnm::AnatomicalSide::Right);
        let left = none_brow(vtuber_gnm::AnatomicalSide::Left);
        let brows_set = brows(right, left);
        let all = features(&eyes, &mouth, &brows_set);

        let predictions = all.predictions();
        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|prediction| prediction.value == 0.0));
    }

    fn observation_with_jaw_open(value: f32) -> AuxiliaryExpressionObservation {
        use vtuber_core::{FaceBlendshapeSet, MediaPipeBlendshape};
        let pairs: Vec<(&str, f32)> = MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| {
                let v = if category == MediaPipeBlendshape::JawOpen {
                    value
                } else {
                    0.0
                };
                (category.as_str(), v)
            })
            .collect();
        let scores = FaceBlendshapeSet::from_pairs(&pairs).unwrap();
        let config = AuxiliaryChannelConfig::new(
            AuxiliaryExpressionSemantic::JawOpen,
            AuxChannelReliability::TrustedForAux,
            1.0,
            None,
        )
        .unwrap();
        AuxiliaryExpressionObservation::from_mediapipe(7, 123_000, &scores, &[config]).unwrap()
    }

    #[test]
    fn exact_frame_alignment_is_enforced_before_evaluation() {
        let eyes = eye_features(0.0, 0.0);
        let mouth = mouth_features(Some(0.5), None, None, None);
        let right = brow_features(vtuber_gnm::AnatomicalSide::Right, None, None, None);
        let left = brow_features(vtuber_gnm::AnatomicalSide::Left, None, None, None);
        let brows_set = brows(right, left);
        let all = features(&eyes, &mouth, &brows_set);
        let observation = observation_with_jaw_open(0.5);

        let ok = all
            .evaluate_residual(
                7,
                123_000,
                Some(&observation),
                AuxiliaryLossConfig::new(1.0, 0.2, 0.01).unwrap(),
            )
            .unwrap();
        assert_eq!(ok.used_channels, 1);
        assert_eq!(
            ok.weighted_loss, 0.0,
            "matching prediction gives zero residual"
        );

        assert!(matches!(
            all.evaluate_residual(
                8,
                123_000,
                Some(&observation),
                AuxiliaryLossConfig::new(1.0, 0.2, 0.01).unwrap(),
            ),
            Err(AuxiliaryExpressionError::SourceSequenceMismatch {
                dense: 8,
                auxiliary: 7
            })
        ));
        assert!(matches!(
            all.evaluate_residual(
                7,
                123_001,
                Some(&observation),
                AuxiliaryLossConfig::new(1.0, 0.2, 0.01).unwrap(),
            ),
            Err(AuxiliaryExpressionError::CaptureTimestampMismatch { .. })
        ));
    }

    #[test]
    fn zero_weight_and_missing_observation_are_finite_no_ops() {
        let eyes = eye_features(0.4, 0.4);
        let mouth = mouth_features(None, None, None, None);
        let right = brow_features(vtuber_gnm::AnatomicalSide::Right, None, None, None);
        let left = brow_features(vtuber_gnm::AnatomicalSide::Left, None, None, None);
        let brows_set = brows(right, left);
        let all = features(&eyes, &mouth, &brows_set);
        let observation = observation_with_jaw_open(0.9);

        let zero_weight = all
            .evaluate_residual(
                7,
                123_000,
                Some(&observation),
                AuxiliaryLossConfig::new(0.0, 0.2, 0.01).unwrap(),
            )
            .unwrap();
        assert_eq!(zero_weight.weighted_loss, 0.0);
        assert_eq!(zero_weight.used_channels, 0);
        assert!(zero_weight.weighted_loss.is_finite());

        let no_observation = all
            .evaluate_residual(
                7,
                123_000,
                None,
                AuxiliaryLossConfig::new(1.0, 0.2, 0.01).unwrap(),
            )
            .unwrap();
        assert_eq!(no_observation.used_channels, 0);
        assert_eq!(no_observation.weighted_loss, 0.0);
    }

    #[test]
    fn group_residuals_disabled_and_disagreement_are_reported() {
        let eyes = eye_features(0.0, -0.9);
        let mouth = mouth_features(Some(0.8), None, None, None);
        let right = brow_features(vtuber_gnm::AnatomicalSide::Right, None, None, None);
        let left = brow_features(vtuber_gnm::AnatomicalSide::Left, None, None, None);
        let brows_set = brows(right, left);
        let all = features(&eyes, &mouth, &brows_set);

        use vtuber_core::{FaceBlendshapeSet, MediaPipeBlendshape};
        let pairs: Vec<(&str, f32)> = MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| {
                let v = match category {
                    MediaPipeBlendshape::EyeBlinkLeft => 0.0,
                    MediaPipeBlendshape::JawOpen => 0.1,
                    _ => 0.0,
                };
                (category.as_str(), v)
            })
            .collect();
        let scores = FaceBlendshapeSet::from_pairs(&pairs).unwrap();
        let configs = [
            // Disabled channel counts as diagnostic, not used.
            AuxiliaryChannelConfig::new(
                AuxiliaryExpressionSemantic::EyeWideRight,
                AuxChannelReliability::Disabled,
                0.0,
                None,
            )
            .unwrap(),
            AuxiliaryChannelConfig::new(
                AuxiliaryExpressionSemantic::EyeClosureLeft,
                AuxChannelReliability::TrustedForAux,
                1.0,
                None,
            )
            .unwrap(),
            AuxiliaryChannelConfig::new(
                AuxiliaryExpressionSemantic::JawOpen,
                AuxChannelReliability::Weak,
                1.0,
                None,
            )
            .unwrap(),
        ];
        let observation =
            AuxiliaryExpressionObservation::from_mediapipe(3, 50, &scores, &configs).unwrap();

        let diagnostics = all
            .evaluate_residual(
                3,
                50,
                Some(&observation),
                AuxiliaryLossConfig::new(1.0, 0.2, 0.5).unwrap(),
            )
            .unwrap();

        assert_eq!(diagnostics.disabled_channels, 1);
        // EyeClosureRight has a prediction (0.0 aperture) but no configured
        // observation channel, so it does not count; EyeClosureLeft and
        // JawOpen are used.
        assert_eq!(diagnostics.used_channels, 2);
        // EyeClosureLeft residual: prediction 0.9 vs observed 0.0.
        assert!(diagnostics.disagreement_count >= 1);
        let eye = diagnostics
            .group_residuals
            .get(AuxiliaryExpressionGroup::Eye);
        let jaw = diagnostics
            .group_residuals
            .get(AuxiliaryExpressionGroup::Jaw);
        assert_eq!(eye, Some(0.9));
        assert!((jaw.unwrap() - 0.7).abs() < 1.0e-5);
        assert!(diagnostics.max_abs_residual >= 0.9);
        assert!(diagnostics.weighted_loss.is_finite() && diagnostics.weighted_loss > 0.0);
    }

    // -- GeometryAuxiliaryObjective (Issue #64.2d / #121) -----------------------

    const RIGHT_UPPER_APEX_MP: usize = 159;
    const RIGHT_LOWER_MID_MP: usize = 145;
    const LEFT_UPPER_APEX_MP: usize = 386;
    const LEFT_LOWER_MID_MP: usize = 374;

    fn region_tag(mp: usize) -> vtuber_gnm::FaceRegion {
        use vtuber_gnm::topology;
        if topology::NOSE.contains(&mp) {
            vtuber_gnm::FaceRegion::Nose
        } else if topology::FACE_OVAL.contains(&mp) {
            vtuber_gnm::FaceRegion::Contour
        } else if mp == topology::IRIS_CENTER_RIGHT || mp == topology::IRIS_CENTER_LEFT {
            vtuber_gnm::FaceRegion::Iris
        } else if topology::is_eyelid(mp) {
            vtuber_gnm::FaceRegion::Eye
        } else if topology::is_brow(mp) {
            vtuber_gnm::FaceRegion::Brow
        } else if topology::LIPS.contains(&mp) {
            vtuber_gnm::FaceRegion::Mouth
        } else {
            vtuber_gnm::FaceRegion::Other
        }
    }

    fn anatomical_side_of(mp: usize) -> vtuber_gnm::AnatomicalSide {
        use vtuber_gnm::topology;
        if topology::EYE_RIGHT.contains(&mp)
            || topology::BROW_RIGHT.contains(&mp)
            || topology::LIPS.contains(&mp) && mp < 300
        {
            vtuber_gnm::AnatomicalSide::Right
        } else if topology::EYE_LEFT.contains(&mp) || topology::BROW_LEFT.contains(&mp) {
            vtuber_gnm::AnatomicalSide::Left
        } else {
            vtuber_gnm::AnatomicalSide::Midline
        }
    }

    fn aux_model() -> vtuber_gnm::GnmModel {
        let vertex_count = MEDIAPIPE_FACE_LANDMARK_COUNT + 3;
        let identity_dim = GNM_HEAD_V3_IDENTITY_DIM;
        let expression_dim = GNM_HEAD_V3_EXPRESSION_DIM;
        let mut vertices = Vec::with_capacity(vertex_count * 3);
        for index in 0..vertex_count {
            vertices.extend_from_slice(&[(index % 7) as f32, (index % 5) as f32, 0.0]);
        }
        for (apex, lower, x) in [
            (RIGHT_UPPER_APEX_MP, RIGHT_LOWER_MID_MP, 5.0f32),
            (LEFT_UPPER_APEX_MP, LEFT_LOWER_MID_MP, 1.0f32),
        ] {
            #[allow(clippy::indexing_slicing)] // fixture indices are compile-time known
            {
                vertices[apex * 3..apex * 3 + 3].copy_from_slice(&[x, 4.0, 0.0]);
                vertices[lower * 3..lower * 3 + 3].copy_from_slice(&[x, 0.0, 0.0]);
            }
        }
        let mut expression_basis = vec![0.0f32; expression_dim * vertex_count * 3];
        for apex in [RIGHT_UPPER_APEX_MP, LEFT_UPPER_APEX_MP] {
            // Channel 0 (closure) lowers both apexes, channel 1 raises them.
            #[allow(clippy::indexing_slicing)]
            {
                expression_basis[apex * 3 + 1] = -0.4;
                expression_basis[(vertex_count + apex) * 3 + 1] = 0.6;
            }
        }
        vtuber_gnm::GnmModel::from_data(GnmModelData {
            version: GNM_HEAD_V3_VERSION,
            variant: vtuber_gnm::GnmVariant::Head,
            template_vertices: DenseArray::new("vertices", vec![vertex_count, 3], vertices)
                .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0; 3]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity_dim, vertex_count, 3],
                vec![0.0; identity_dim * vertex_count * 3],
            )
            .unwrap(),
            joint_identity_basis: DenseArray::new(
                "joint_identity",
                vec![identity_dim, 1, 3],
                vec![0.0; identity_dim * 3],
            )
            .unwrap(),
            expression_basis: DenseArray::new(
                "expression",
                vec![expression_dim, vertex_count, 3],
                expression_basis,
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new(
                "weights",
                vec![1, vertex_count],
                vec![1.0; vertex_count],
            )
            .unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    fn mapping_row(mp: usize) -> vtuber_gnm::MediaPipeGnmDenseCorrespondence {
        use vtuber_gnm::{GnmSurfacePointRef, topology};
        let target = if mp == topology::IRIS_CENTER_RIGHT || mp == topology::IRIS_CENTER_LEFT {
            GnmSurfacePointRef::Barycentric {
                vertex_indices: [mp, mp + 1, mp + 2],
                weights: [0.5, 0.25, 0.25],
            }
        } else {
            GnmSurfacePointRef::Vertex { vertex_index: mp }
        };
        vtuber_gnm::MediaPipeGnmDenseCorrespondence {
            mediapipe_index: mp,
            target,
            region: region_tag(mp),
            anatomical_side: anatomical_side_of(mp),
            base_weight: 1.0,
            provenance: vtuber_gnm::CorrespondenceProvenance::RepositoryValidated,
            reliability: vtuber_gnm::CorrespondenceReliability::High,
        }
    }

    fn full_mapping(model: &vtuber_gnm::GnmModel) -> DenseCorrespondenceSet {
        use vtuber_gnm::topology;
        let mut mps: Vec<usize> = topology::NOSE
            .iter()
            .copied()
            .chain(topology::FACE_OVAL.iter().copied())
            .chain(topology::LIPS.iter().copied())
            .chain(topology::EYE_RIGHT.iter().copied())
            .chain(topology::EYE_LEFT.iter().copied())
            .chain(topology::BROW_RIGHT.iter().copied())
            .chain(topology::BROW_LEFT.iter().copied())
            .chain([
                topology::IRIS_CENTER_RIGHT,
                topology::IRIS_CENTER_LEFT,
                100,
                200,
            ])
            .collect();
        mps.sort_unstable();
        mps.dedup();
        DenseCorrespondenceSet::new(
            DenseMappingVersion {
                schema_revision: 1,
                model_version: GNM_HEAD_V3_VERSION,
            },
            mps.iter().map(|mp| mapping_row(*mp)).collect(),
            model,
        )
        .unwrap()
    }

    fn neutral_calibration(
        model: &vtuber_gnm::GnmModel,
        mapping: &DenseCorrespondenceSet,
    ) -> GnmIdentityCalibration {
        use vtuber_gnm::{
            FixedGnmIdentity, GnmJointState, NeutralNormalizationScales, NeutralPoseDiversity,
        };
        let mut surface = GnmSparseVertices::with_len(mapping.len());
        mapping
            .evaluate_surface(
                model,
                &model.neutral_identity(),
                &model.neutral_expression(),
                &GnmJointState::neutral(model.joint_count()),
                &mut surface,
            )
            .unwrap();
        let fixed = FixedGnmIdentity::new(model.neutral_identity(), model).unwrap();
        GnmIdentityCalibration::new(
            model,
            DenseMappingVersion {
                schema_revision: 1,
                model_version: GNM_HEAD_V3_VERSION,
            },
            fixed,
            model.neutral_expression(),
            surface.values().to_vec(),
            NeutralNormalizationScales {
                inter_ocular: Some(2.0),
                mouth_width: Some(2.0),
                eye_aperture: None,
            },
            IdentityFitDiagnostics {
                accepted_samples: 8,
                rejected_samples: 0,
                reprojection_rms: 0.01,
                active_identity_dimension: 8,
                condition_number: Some(12.0),
                pose_diversity: NeutralPoseDiversity {
                    yaw_span_radians: 0.2,
                    pitch_span_radians: 0.1,
                    near_duplicate_fraction: 0.1,
                },
            },
        )
        .unwrap()
    }

    fn media_pipe_scores(
        entries: &[(vtuber_core::MediaPipeBlendshape, f32)],
    ) -> vtuber_core::FaceBlendshapeSet {
        let pairs: Vec<(&str, f32)> = vtuber_core::MediaPipeBlendshape::ALL
            .into_iter()
            .map(|category| {
                let value = entries
                    .iter()
                    .find(|(candidate, _)| *candidate == category)
                    .map_or(0.0, |(_, value)| *value);
                (category.as_str(), value)
            })
            .collect();
        vtuber_core::FaceBlendshapeSet::from_pairs(&pairs).unwrap()
    }

    struct ObjectiveFixture {
        model: vtuber_gnm::GnmModel,
        mapping: DenseCorrespondenceSet,
        calibration: GnmIdentityCalibration,
        identity: vtuber_gnm::FixedGnmIdentity,
        eye_groups: DenseRegionGroups,
    }

    fn objective_fixture() -> ObjectiveFixture {
        let model = aux_model();
        let mapping = full_mapping(&model);
        let eye_groups = DenseRegionGroups::from_set(&mapping).unwrap();
        let calibration = neutral_calibration(&model, &mapping);
        let identity = vtuber_gnm::FixedGnmIdentity::new(model.neutral_identity(), &model).unwrap();
        ObjectiveFixture {
            model,
            mapping,
            calibration,
            identity,
            eye_groups,
        }
    }

    fn blink_observation(
        source_seq: u64,
        captured_at_micros: u64,
        blink_value: f32,
    ) -> crate::auxiliary_expression::AuxiliaryExpressionObservation {
        use crate::auxiliary_expression::{AuxChannelReliability, AuxiliaryChannelConfig};
        use vtuber_core::MediaPipeBlendshape;
        let scores = media_pipe_scores(&[
            (MediaPipeBlendshape::EyeBlinkLeft, blink_value),
            (MediaPipeBlendshape::EyeBlinkRight, blink_value),
        ]);
        let configs = [
            AuxiliaryChannelConfig::new(
                AuxiliaryExpressionSemantic::EyeClosureLeft,
                AuxChannelReliability::TrustedForAux,
                1.0,
                None,
            )
            .unwrap(),
            AuxiliaryChannelConfig::new(
                AuxiliaryExpressionSemantic::EyeClosureRight,
                AuxChannelReliability::TrustedForAux,
                1.0,
                None,
            )
            .unwrap(),
        ];
        crate::auxiliary_expression::AuxiliaryExpressionObservation::from_mediapipe(
            source_seq,
            captured_at_micros,
            &scores,
            &configs,
        )
        .unwrap()
    }

    #[test]
    fn objective_construction_validates_exact_source_alignment() {
        let fixture = objective_fixture();
        let observation = blink_observation(7, 123_000, 0.5);
        let config = AuxiliaryLossConfig::new(1.0, 0.2, 0.5).unwrap();

        assert!(
            GeometryAuxiliaryObjective::new(
                7,
                123_000,
                &observation,
                &fixture.model,
                &fixture.identity,
                &fixture.mapping,
                &fixture.calibration,
                &fixture.eye_groups,
                config,
                1.0e-2,
            )
            .is_ok()
        );
        assert!(matches!(
            GeometryAuxiliaryObjective::new(
                8,
                123_000,
                &observation,
                &fixture.model,
                &fixture.identity,
                &fixture.mapping,
                &fixture.calibration,
                &fixture.eye_groups,
                config,
                1.0e-2,
            ),
            Err(AuxiliaryExpressionError::SourceSequenceMismatch { .. })
        ));
        assert!(matches!(
            GeometryAuxiliaryObjective::new(
                7,
                123_001,
                &observation,
                &fixture.model,
                &fixture.identity,
                &fixture.mapping,
                &fixture.calibration,
                &fixture.eye_groups,
                config,
                1.0e-2,
            ),
            Err(AuxiliaryExpressionError::CaptureTimestampMismatch { .. })
        ));
        assert!(matches!(
            GeometryAuxiliaryObjective::new(
                7,
                123_000,
                &observation,
                &fixture.model,
                &fixture.identity,
                &fixture.mapping,
                &fixture.calibration,
                &fixture.eye_groups,
                config,
                0.0,
            ),
            Err(AuxiliaryExpressionError::InvalidConfig(_))
        ));
    }

    #[test]
    fn objective_evaluate_reports_finite_loss_and_full_dimensions() {
        use vtuber_gnm::AuxiliaryTermEvaluation;
        let fixture = objective_fixture();
        let observation = blink_observation(7, 123_000, 0.5);
        let objective = GeometryAuxiliaryObjective::new(
            7,
            123_000,
            &observation,
            &fixture.model,
            &fixture.identity,
            &fixture.mapping,
            &fixture.calibration,
            &fixture.eye_groups,
            AuxiliaryLossConfig::new(1.0, 0.2, 0.5).unwrap(),
            1.0e-2,
        )
        .unwrap();

        let expression =
            vtuber_gnm::GnmExpressionState::neutral(fixture.model.expression_dimension());
        let joints = vtuber_gnm::GnmJointState::neutral(fixture.model.joint_count());
        let evaluation: AuxiliaryTermEvaluation = objective
            .evaluate(
                expression.values(),
                joints.rotations(),
                joints.translation(),
            )
            .unwrap();

        assert!(evaluation.loss.is_finite() && evaluation.loss > 0.0);
        assert_eq!(
            evaluation.expression_gradient.len(),
            expression.values().len()
        );
        assert_eq!(
            evaluation.joint_gradient.len(),
            3 * (joints.rotations().len() + 1)
        );
        assert!(evaluation.expression_gradient.iter().all(|v| v.is_finite()));
        assert!(evaluation.joint_gradient.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn objective_loss_decreases_toward_observed_blink() {
        use vtuber_gnm::{GnmExpressionState, GnmJointState};
        let fixture = objective_fixture();
        let observation = blink_observation(7, 123_000, 0.5);
        let objective = GeometryAuxiliaryObjective::new(
            7,
            123_000,
            &observation,
            &fixture.model,
            &fixture.identity,
            &fixture.mapping,
            &fixture.calibration,
            &fixture.eye_groups,
            AuxiliaryLossConfig::new(1.0, 0.2, 0.5).unwrap(),
            1.0e-2,
        )
        .unwrap();

        let joints = GnmJointState::neutral(fixture.model.joint_count());
        let neutral = GnmExpressionState::neutral(fixture.model.expression_dimension());
        let mut closing = neutral.values().to_vec();
        // Channel 0 lowers the upper-lid apexes, i.e. closes the eyes.
        #[allow(clippy::indexing_slicing)]
        {
            closing[0] = 0.05;
        }
        let closed =
            GnmExpressionState::new(closing, fixture.model.expression_dimension()).unwrap();

        let neutral_loss = objective
            .evaluate(neutral.values(), joints.rotations(), joints.translation())
            .unwrap()
            .loss;
        let closed_loss = objective
            .evaluate(closed.values(), joints.rotations(), joints.translation())
            .unwrap()
            .loss;
        assert!(
            closed_loss < neutral_loss,
            "closing toward observed blink must reduce the auxiliary loss: {closed_loss} vs {neutral_loss}"
        );
    }
}
