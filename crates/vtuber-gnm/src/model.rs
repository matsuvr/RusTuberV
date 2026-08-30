//! Validated GNM Head v3 data and sparse evaluation.

use crate::error::GnmModelError;

/// Number of identity coefficients in the official GNM Head v3 model.
pub const GNM_HEAD_V3_IDENTITY_DIM: usize = 253;
/// Number of expression coefficients in the official GNM Head v3 model.
pub const GNM_HEAD_V3_EXPRESSION_DIM: usize = 383;

/// GNM model version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GnmVersion {
    /// Major schema version.
    pub major: u16,
    /// Minor schema version.
    pub minor: u16,
}

/// Exact GNM Head v3 version supported by the pinned official artifact.
pub const GNM_HEAD_V3_VERSION: GnmVersion = GnmVersion { major: 3, minor: 0 };

/// Supported GNM model variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GnmVariant {
    /// The v3 head model used by the retargeting pipeline.
    Head,
}

/// A finite, row-major numeric tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseArray {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl DenseArray {
    /// Creates a tensor after checking its dimensions and finiteness.
    pub fn new(
        field: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<Self, GnmModelError> {
        let field = field.into();
        let expected = shape
            .iter()
            .try_fold(1usize, |product, dimension| product.checked_mul(*dimension));
        let Some(expected) = expected else {
            return Err(GnmModelError::InvalidValue {
                field,
                reason: "shape size overflow".to_owned(),
            });
        };
        if expected != values.len() {
            return Err(GnmModelError::Shape {
                field,
                expected: format!("{} values", expected),
                actual: format!("{} values", values.len()),
            });
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(GnmModelError::NonFinite { field, index });
        }
        Ok(Self { shape, values })
    }

    /// Returns the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the flattened tensor values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn at3(&self, first: usize, second: usize, third: usize) -> f32 {
        let index = first * self.shape[1] * self.shape[2] + second * self.shape[2] + third;
        self.values[index]
    }
}

/// Identity coefficients for one GNM evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmIdentityState(Vec<f32>);

impl GnmIdentityState {
    /// Creates an identity state with the requested model dimension.
    pub fn new(values: Vec<f32>, expected_dimension: usize) -> Result<Self, GnmModelError> {
        validate_state("identity", values.len(), expected_dimension)?;
        validate_finite("identity", &values)?;
        Ok(Self(values))
    }

    /// Creates the neutral identity state for a model.
    pub fn neutral(expected_dimension: usize) -> Self {
        Self(vec![0.0; expected_dimension])
    }

    /// Returns the coefficients.
    pub fn values(&self) -> &[f32] {
        &self.0
    }
}

/// Expression coefficients for one GNM evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmExpressionState(Vec<f32>);

impl GnmExpressionState {
    /// Creates an expression state with the requested model dimension.
    pub fn new(values: Vec<f32>, expected_dimension: usize) -> Result<Self, GnmModelError> {
        validate_state("expression", values.len(), expected_dimension)?;
        validate_finite("expression", &values)?;
        Ok(Self(values))
    }

    /// Creates the neutral expression state for a model.
    pub fn neutral(expected_dimension: usize) -> Self {
        Self(vec![0.0; expected_dimension])
    }

    /// Returns the coefficients.
    pub fn values(&self) -> &[f32] {
        &self.0
    }
}

/// Optional joint pose state. Rotations are axis-angle vectors in radians.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmJointState {
    rotations: Vec<[f32; 3]>,
    translation: [f32; 3],
}

impl GnmJointState {
    /// Creates a joint state with one rotation per model joint.
    pub fn new(
        rotations: Vec<[f32; 3]>,
        translation: [f32; 3],
        expected_joint_count: usize,
    ) -> Result<Self, GnmModelError> {
        if rotations.len() != expected_joint_count {
            return Err(GnmModelError::Shape {
                field: "joint_rotations".to_owned(),
                expected: format!("[{expected_joint_count}, 3]"),
                actual: format!("[{}, 3]", rotations.len()),
            });
        }
        if rotations.iter().flatten().any(|value| !value.is_finite())
            || translation.iter().any(|value| !value.is_finite())
        {
            return Err(GnmModelError::NonFinite {
                field: "joint_state".to_owned(),
                index: 0,
            });
        }
        Ok(Self {
            rotations,
            translation,
        })
    }

    /// Creates a neutral pose state for a model.
    pub fn neutral(expected_joint_count: usize) -> Self {
        Self {
            rotations: vec![[0.0; 3]; expected_joint_count],
            translation: [0.0; 3],
        }
    }

    /// Returns joint axis-angle rotations.
    pub fn rotations(&self) -> &[[f32; 3]] {
        &self.rotations
    }

    /// Returns the global translation.
    pub fn translation(&self) -> [f32; 3] {
        self.translation
    }
}

/// Input arrays needed by the sparse Head v3 evaluator.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmModelData {
    /// Model schema version.
    pub version: GnmVersion,
    /// Model variant.
    pub variant: GnmVariant,
    /// Neutral vertices, shaped `[vertex, 3]`.
    pub template_vertices: DenseArray,
    /// Neutral joints, shaped `[joint, 3]`.
    pub template_joints: DenseArray,
    /// Identity vertex basis, shaped `[identity, vertex, 3]`.
    pub vertex_identity_basis: DenseArray,
    /// Identity joint basis, shaped `[identity, joint, 3]`.
    pub joint_identity_basis: DenseArray,
    /// Expression vertex basis, shaped `[expression, vertex, 3]`.
    pub expression_basis: DenseArray,
    /// Parent index for each joint; the root may use `-1` or the official
    /// self-parent sentinel `0`.
    pub joint_parent_indices: Vec<i32>,
    /// Linear blend skinning weights, shaped `[joint, vertex]`.
    pub skinning_weights: DenseArray,
    /// Optional pose-corrective basis, shaped `[9 * joint, vertex, 3]`.
    pub pose_correctives_regressor: Option<DenseArray>,
}

/// Validated GNM model.
#[derive(Clone, Debug, PartialEq)]
pub struct GnmModel {
    version: GnmVersion,
    variant: GnmVariant,
    template_vertices: DenseArray,
    template_joints: DenseArray,
    vertex_identity_basis: DenseArray,
    joint_identity_basis: DenseArray,
    expression_basis: DenseArray,
    joint_parent_indices: Vec<i32>,
    skinning_weights: DenseArray,
    pose_correctives_regressor: Option<DenseArray>,
}

impl GnmModel {
    /// Validates a Head v3 model and takes ownership of its numeric arrays.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn from_data(data: GnmModelData) -> Result<Self, GnmModelError> {
        if data.version != GNM_HEAD_V3_VERSION {
            return Err(GnmModelError::UnsupportedVersion(format!(
                "{}.{}",
                data.version.major, data.version.minor
            )));
        }
        if data.variant != GnmVariant::Head {
            return Err(GnmModelError::UnsupportedVariant(format!(
                "{:?}",
                data.variant
            )));
        }
        let [vertex_count, three] = shape2("template_vertices", data.template_vertices.shape())?;
        if three != 3 {
            return Err(shape_error(
                "template_vertices",
                "[vertex, 3]",
                data.template_vertices.shape(),
            ));
        }
        let [joint_count, joint_three] = shape2("template_joints", data.template_joints.shape())?;
        if joint_three != 3 {
            return Err(shape_error(
                "template_joints",
                "[joint, 3]",
                data.template_joints.shape(),
            ));
        }
        let [identity_count, identity_vertices, identity_three] =
            shape3("vertex_identity_basis", data.vertex_identity_basis.shape())?;
        if [identity_vertices, identity_three] != [vertex_count, 3] {
            return Err(shape_error(
                "vertex_identity_basis",
                "[identity, vertex, 3]",
                data.vertex_identity_basis.shape(),
            ));
        }
        let [
            joint_identity_count,
            joint_identity_joints,
            joint_identity_three,
        ] = shape3("joint_identity_basis", data.joint_identity_basis.shape())?;
        if [
            joint_identity_count,
            joint_identity_joints,
            joint_identity_three,
        ] != [identity_count, joint_count, 3]
        {
            return Err(shape_error(
                "joint_identity_basis",
                "[identity, joint, 3] matching vertex identity basis",
                data.joint_identity_basis.shape(),
            ));
        }
        let [expression_count, expression_vertices, expression_three] =
            shape3("expression_basis", data.expression_basis.shape())?;
        if [expression_vertices, expression_three] != [vertex_count, 3] {
            return Err(shape_error(
                "expression_basis",
                "[expression, vertex, 3]",
                data.expression_basis.shape(),
            ));
        }
        if data.joint_parent_indices.len() != joint_count {
            return Err(GnmModelError::Shape {
                field: "joint_parent_indices".to_owned(),
                expected: format!("[{joint_count}]"),
                actual: format!("[{}]", data.joint_parent_indices.len()),
            });
        }
        if joint_count == 0 {
            return Err(GnmModelError::InvalidValue {
                field: "template_joint_positions".to_owned(),
                reason: "at least one root joint is required".to_owned(),
            });
        }
        let root_parent = data.joint_parent_indices[0];
        if !matches!(root_parent, -1 | 0)
            || data
                .joint_parent_indices
                .iter()
                .enumerate()
                .skip(1)
                .any(|(index, parent)| (*parent < 0) || (*parent >= index as i32))
        {
            return Err(GnmModelError::InvalidValue {
                field: "joint_parent_indices".to_owned(),
                reason: "root must be -1 or 0 and children must reference an earlier joint"
                    .to_owned(),
            });
        }
        let weights_shape = data.skinning_weights.shape();
        if weights_shape != [joint_count, vertex_count] {
            return Err(shape_error(
                "skinning_weights",
                "[joint, vertex]",
                weights_shape,
            ));
        }
        if data
            .skinning_weights
            .values()
            .iter()
            .any(|weight| *weight < 0.0)
        {
            return Err(GnmModelError::InvalidValue {
                field: "skinning_weights".to_owned(),
                reason: "weights must be non-negative".to_owned(),
            });
        }
        if let Some(correctives) = &data.pose_correctives_regressor {
            let expected = [joint_count * 9, vertex_count, 3];
            if correctives.shape() != expected {
                return Err(shape_error(
                    "pose_correctives_regressor",
                    "[9 * joint, vertex, 3]",
                    correctives.shape(),
                ));
            }
        }
        if identity_count != GNM_HEAD_V3_IDENTITY_DIM {
            return Err(GnmModelError::InvalidValue {
                field: "vertex_identity_basis".to_owned(),
                reason: format!(
                    "Head v3 requires {GNM_HEAD_V3_IDENTITY_DIM} identity coefficients"
                ),
            });
        }
        if expression_count != GNM_HEAD_V3_EXPRESSION_DIM {
            return Err(GnmModelError::InvalidValue {
                field: "expression_basis".to_owned(),
                reason: format!(
                    "Head v3 requires {GNM_HEAD_V3_EXPRESSION_DIM} expression coefficients"
                ),
            });
        }
        Ok(Self {
            version: data.version,
            variant: data.variant,
            template_vertices: data.template_vertices,
            template_joints: data.template_joints,
            vertex_identity_basis: data.vertex_identity_basis,
            joint_identity_basis: data.joint_identity_basis,
            expression_basis: data.expression_basis,
            joint_parent_indices: data.joint_parent_indices,
            skinning_weights: data.skinning_weights,
            pose_correctives_regressor: data.pose_correctives_regressor,
        })
    }

    /// Returns the schema version.
    pub fn version(&self) -> GnmVersion {
        self.version
    }

    /// Returns the model variant.
    pub fn variant(&self) -> GnmVariant {
        self.variant
    }

    /// Returns the vertex count.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn vertex_count(&self) -> usize {
        self.template_vertices.shape()[0]
    }

    /// Returns the joint count.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn joint_count(&self) -> usize {
        self.template_joints.shape()[0]
    }

    /// Returns the identity dimension.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn identity_dimension(&self) -> usize {
        self.vertex_identity_basis.shape()[0]
    }

    /// Returns the expression dimension.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn expression_dimension(&self) -> usize {
        self.expression_basis.shape()[0]
    }

    /// Creates a neutral identity state.
    pub fn neutral_identity(&self) -> GnmIdentityState {
        GnmIdentityState::neutral(self.identity_dimension())
    }

    /// Creates a neutral expression state.
    pub fn neutral_expression(&self) -> GnmExpressionState {
        GnmExpressionState::neutral(self.expression_dimension())
    }

    /// Evaluates only the requested sparse landmark points into reusable output.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn evaluate_sparse(
        &self,
        identity: &GnmIdentityState,
        expression: &GnmExpressionState,
        joints: &GnmJointState,
        landmarks: &crate::SparseLandmarkSet,
        output: &mut GnmSparseVertices,
    ) -> Result<(), GnmModelError> {
        validate_state(
            "identity",
            identity.values().len(),
            self.identity_dimension(),
        )?;
        validate_state(
            "expression",
            expression.values().len(),
            self.expression_dimension(),
        )?;
        if joints.rotations.len() != self.joint_count() {
            return Err(GnmModelError::Shape {
                field: "joint_rotations".to_owned(),
                expected: format!("[{}, 3]", self.joint_count()),
                actual: format!("[{}, 3]", joints.rotations.len()),
            });
        }
        if let Some(vertex) = landmarks
            .unique_vertices()
            .iter()
            .find(|vertex| **vertex >= self.vertex_count())
        {
            return Err(GnmModelError::InvalidValue {
                field: "landmark vertex index".to_owned(),
                reason: format!("references vertex {vertex}"),
            });
        }
        output.resize_points(landmarks.len());
        output.resize_vertex_scratch(landmarks.unique_vertices().len());
        output.resize_joint_scratch(self.joint_count());
        self.deformed_sparse_vertices(
            identity,
            expression,
            joints,
            landmarks,
            &mut output.vertex_values,
            &mut output.pose_coefficients,
        );
        self.joint_transforms(
            joints,
            identity,
            &mut output.joint_positions,
            &mut output.joint_transforms,
        )?;
        for (point_index, ((output_point, landmark), slots)) in output
            .values
            .iter_mut()
            .zip(landmarks.points())
            .zip(landmarks.point_vertex_slots())
            .enumerate()
        {
            let mut point: [f32; 3] = [0.0; 3];
            for (corner, (slot, weight)) in slots.iter().zip(landmark.weights).enumerate() {
                let mut skinned = [0.0; 3];
                for (joint, transform) in output.joint_transforms.iter().enumerate() {
                    let vertex = landmark.indices[corner];
                    let skin_weight =
                        self.skinning_weights.values()[joint * self.vertex_count() + vertex];
                    let transformed = transform
                        .apply_skin(output.vertex_values[*slot], output.joint_positions[joint]);
                    for component in 0..3 {
                        skinned[component] += skin_weight * transformed[component];
                    }
                }
                for component in 0..3 {
                    point[component] += weight * skinned[component];
                }
            }
            if point.iter().any(|value| !value.is_finite()) {
                return Err(GnmModelError::NonFinite {
                    field: "sparse_vertices".to_owned(),
                    index: point_index,
                });
            }
            *output_point = point;
        }
        Ok(())
    }

    /// Computes per-point, per-corner linear maps from template-space vertex
    /// offsets to skinned world-space offsets under one pose.
    ///
    /// Because sparse evaluation is affine in the pre-skinning vertex position
    /// for fixed identity and joint state, these maps are exactly the Jacobian
    /// of [`Self::evaluate_sparse`] with respect to template-space offsets.
    /// Callers combine them with a basis array (for example the expression
    /// basis) through [`SparseSkinningDerivatives::expression_point_offsets`]
    /// to obtain analytic derivatives without re-evaluating the surface.
    ///
    /// # Errors
    ///
    /// Fails when the state dimensions or landmark vertices do not match the
    /// model, mirroring [`Self::evaluate_sparse`] validation.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn sparse_skinning_derivatives(
        &self,
        identity: &GnmIdentityState,
        joints: &GnmJointState,
        landmarks: &crate::SparseLandmarkSet,
    ) -> Result<SparseSkinningDerivatives, GnmModelError> {
        validate_state(
            "identity",
            identity.values().len(),
            self.identity_dimension(),
        )?;
        if joints.rotations.len() != self.joint_count() {
            return Err(GnmModelError::Shape {
                field: "joint_rotations".to_owned(),
                expected: format!("[{}, 3]", self.joint_count()),
                actual: format!("[{}, 3]", joints.rotations.len()),
            });
        }
        if let Some(vertex) = landmarks
            .unique_vertices()
            .iter()
            .find(|vertex| **vertex >= self.vertex_count())
        {
            return Err(GnmModelError::InvalidValue {
                field: "landmark vertex index".to_owned(),
                reason: format!("references vertex {vertex}"),
            });
        }

        let mut joint_positions = vec![[0.0f32; 3]; self.joint_count()];
        let mut world_transforms = vec![JointTransform::identity(); self.joint_count()];
        self.joint_transforms(
            joints,
            identity,
            &mut joint_positions,
            &mut world_transforms,
        )?;

        let mut corners = Vec::with_capacity(landmarks.len());
        for landmark in landmarks.points() {
            let mut point_corners = [CornerDerivative {
                vertex: 0,
                weight: 0.0,
                rotation: [[0.0; 3]; 3],
            }; 3];
            for (slot, vertex) in landmark.indices.iter().copied().enumerate() {
                let mut rotation = [[0.0f32; 3]; 3];
                for (joint, transform) in world_transforms.iter().enumerate() {
                    let skin_weight =
                        self.skinning_weights.values()[joint * self.vertex_count() + vertex];
                    if skin_weight == 0.0 {
                        continue;
                    }
                    for (rotation_row, world_row) in
                        rotation.iter_mut().zip(transform.rotation.iter())
                    {
                        for (entry, world_entry) in rotation_row.iter_mut().zip(world_row.iter()) {
                            *entry += skin_weight * world_entry;
                        }
                    }
                }
                point_corners[slot] = CornerDerivative {
                    vertex,
                    weight: landmark.weights[slot],
                    rotation,
                };
            }
            corners.push(point_corners);
        }
        Ok(SparseSkinningDerivatives { corners })
    }

    /// Evaluates the pre-skinning stage of [`Self::evaluate_sparse`] for one
    /// selected landmark set.
    ///
    /// Pair this with [`SparsePreparedVertices::skin`] to re-skin an unchanged
    /// expression/identity state under many joint poses without repeating the
    /// identity/expression basis loops. When the model ships pose correctives,
    /// [`SparsePreparedVertices::depends_on_joint_state`] reports that the
    /// prepared values are bound to the supplied joint pose.
    ///
    /// # Errors
    ///
    /// Fails on invalid configuration or state dimension mismatch against the
    /// model, mirroring [`Self::evaluate_sparse`].
    pub fn prepare_sparse_vertices(
        &self,
        identity: &GnmIdentityState,
        expression: &GnmExpressionState,
        joints: &GnmJointState,
        landmarks: &crate::SparseLandmarkSet,
    ) -> Result<SparsePreparedVertices, GnmModelError> {
        validate_state(
            "identity",
            identity.values().len(),
            self.identity_dimension(),
        )?;
        validate_state(
            "expression",
            expression.values().len(),
            self.expression_dimension(),
        )?;
        if joints.rotations.len() != self.joint_count() {
            return Err(GnmModelError::Shape {
                field: "joint_rotations".to_owned(),
                expected: format!("[{}, 3]", self.joint_count()),
                actual: format!("[{}, 3]", joints.rotations.len()),
            });
        }
        if let Some(vertex) = landmarks
            .unique_vertices()
            .iter()
            .find(|vertex| **vertex >= self.vertex_count())
        {
            return Err(GnmModelError::InvalidValue {
                field: "landmark vertex index".to_owned(),
                reason: format!("references vertex {vertex}"),
            });
        }

        let mut values = vec![[0.0f32; 3]; landmarks.unique_vertices().len()];
        let mut pose_coefficients = vec![[0.0f32; 9]; self.joint_count()];
        self.deformed_sparse_vertices(
            identity,
            expression,
            joints,
            landmarks,
            &mut values,
            &mut pose_coefficients,
        );
        Ok(SparsePreparedVertices {
            corrective_dependent: self.pose_correctives_regressor.is_some(),
            values,
        })
    }

    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn deformed_sparse_vertices(
        &self,
        identity: &GnmIdentityState,
        expression: &GnmExpressionState,
        joints: &GnmJointState,
        landmarks: &crate::SparseLandmarkSet,
        vertices: &mut [[f32; 3]],
        pose_coefficients: &mut [[f32; 9]],
    ) {
        if self.pose_correctives_regressor.is_some() {
            for (joint, coefficients) in pose_coefficients.iter_mut().enumerate() {
                let rotation = rotation_from_axis_angle(joints.rotations[joint]);
                for (row, rotation_row) in rotation.iter().enumerate() {
                    for (column, rotation_value) in rotation_row.iter().enumerate() {
                        coefficients[row * 3 + column] =
                            *rotation_value - if row == column { 1.0 } else { 0.0 };
                    }
                }
            }
        }
        for (output_vertex, vertex) in vertices.iter_mut().zip(landmarks.unique_vertices()) {
            let mut value = [0.0; 3];
            for (component, value_component) in value.iter_mut().enumerate() {
                *value_component = self.template_vertices.values()[vertex * 3 + component];
            }
            for (basis, coefficient) in identity.values().iter().enumerate() {
                for (component, value_component) in value.iter_mut().enumerate() {
                    *value_component +=
                        coefficient * self.vertex_identity_basis.at3(basis, *vertex, component);
                }
            }
            for (basis, coefficient) in expression.values().iter().enumerate() {
                for (component, value_component) in value.iter_mut().enumerate() {
                    *value_component +=
                        coefficient * self.expression_basis.at3(basis, *vertex, component);
                }
            }
            if let Some(correctives) = &self.pose_correctives_regressor {
                for (joint, coefficients) in pose_coefficients.iter().enumerate() {
                    for (basis, coefficient) in coefficients.iter().enumerate() {
                        for (component, value_component) in value.iter_mut().enumerate() {
                            *value_component += coefficient
                                * correctives.at3(joint * 9 + basis, *vertex, component);
                        }
                    }
                }
            }
            *output_vertex = value;
        }
    }

    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn joint_transforms(
        &self,
        joints: &GnmJointState,
        identity: &GnmIdentityState,
        joint_positions: &mut [[f32; 3]],
        world_transforms: &mut [JointTransform],
    ) -> Result<(), GnmModelError> {
        for (joint, position) in joint_positions.iter_mut().enumerate() {
            for (component, position_component) in position.iter_mut().enumerate() {
                *position_component = self.template_joints.values()[joint * 3 + component];
                for (basis, coefficient) in identity.values().iter().enumerate() {
                    *position_component +=
                        coefficient * self.joint_identity_basis.at3(basis, joint, component);
                }
            }
        }
        for joint in 0..self.joint_count() {
            let parent = self.joint_parent_indices[joint];
            let local_position = if joint == 0 && (parent == -1 || parent == 0) {
                add(joint_positions[0], joints.translation)
            } else {
                let parent = usize::try_from(parent).map_err(|_| GnmModelError::InvalidValue {
                    field: "joint_parent_indices".to_owned(),
                    reason: "non-root joints must have a non-negative parent".to_owned(),
                })?;
                if parent >= joint {
                    return Err(GnmModelError::InvalidValue {
                        field: "joint_parent_indices".to_owned(),
                        reason: "parents must precede their children".to_owned(),
                    });
                }
                subtract(joint_positions[joint], joint_positions[parent])
            };
            let local = JointTransform {
                rotation: rotation_from_axis_angle(joints.rotations[joint]),
                translation: local_position,
            };
            world_transforms[joint] = if joint == 0 && (parent == -1 || parent == 0) {
                local
            } else {
                let parent = usize::try_from(parent).map_err(|_| GnmModelError::InvalidValue {
                    field: "joint_parent_indices".to_owned(),
                    reason: "non-root joints must have a non-negative parent".to_owned(),
                })?;
                world_transforms[parent].compose(local)
            };
        }
        Ok(())
    }
}

fn validate_state(field: &str, actual: usize, expected: usize) -> Result<(), GnmModelError> {
    if actual != expected {
        return Err(GnmModelError::Shape {
            field: field.to_owned(),
            expected: format!("[{expected}]"),
            actual: format!("[{actual}]"),
        });
    }
    Ok(())
}

fn validate_finite(field: &str, values: &[f32]) -> Result<(), GnmModelError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(GnmModelError::NonFinite {
            field: field.to_owned(),
            index,
        });
    }
    Ok(())
}

fn shape2(field: &str, shape: &[usize]) -> Result<[usize; 2], GnmModelError> {
    shape
        .try_into()
        .map_err(|_| shape_error(field, "rank 2", shape))
}

fn shape3(field: &str, shape: &[usize]) -> Result<[usize; 3], GnmModelError> {
    shape
        .try_into()
        .map_err(|_| shape_error(field, "rank 3", shape))
}

fn shape_error(field: &str, expected: &str, actual: &[usize]) -> GnmModelError {
    GnmModelError::Shape {
        field: field.to_owned(),
        expected: expected.to_owned(),
        actual: format!("{actual:?}"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct JointTransform {
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
}

impl JointTransform {
    const fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        }
    }

    fn apply(self, point: [f32; 3]) -> [f32; 3] {
        add(matrix_vector(self.rotation, point), self.translation)
    }

    fn apply_skin(self, point: [f32; 3], rest_joint: [f32; 3]) -> [f32; 3] {
        self.apply([
            point[0] - rest_joint[0],
            point[1] - rest_joint[1],
            point[2] - rest_joint[2],
        ])
    }

    fn compose(self, child: Self) -> Self {
        Self {
            rotation: matrix_matrix(self.rotation, child.rotation),
            translation: add(
                matrix_vector(self.rotation, child.translation),
                self.translation,
            ),
        }
    }
}

/// Per-point linear maps from template-space vertex offsets to skinned
/// world-space offsets under one skinning pose.
///
/// Produced by [`GnmModel::sparse_skinning_derivatives`]; see there for the
/// contract.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseSkinningDerivatives {
    corners: Vec<[CornerDerivative; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CornerDerivative {
    vertex: usize,
    weight: f32,
    rotation: [[f32; 3]; 3],
}

impl SparseSkinningDerivatives {
    /// Returns the number of mapped points.
    pub fn len(&self) -> usize {
        self.corners.len()
    }

    /// Returns whether this set has no points.
    pub fn is_empty(&self) -> bool {
        self.corners.is_empty()
    }

    /// Returns which expression basis columns move any mapped point.
    ///
    /// Real face bases touch only a subset of vertices, so most columns are
    /// exactly zero and can skip offset computation and projection entirely.
    /// Index `k` of the result corresponds to expression coefficient `e_k`.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn active_expression_columns(&self, model: &GnmModel) -> Vec<bool> {
        let mut active = vec![false; model.expression_dimension()];
        for point_corners in self.corners.iter() {
            for corner in point_corners {
                if corner.weight == 0.0 {
                    continue;
                }
                for (basis_index, is_active) in active.iter_mut().enumerate() {
                    if !*is_active
                        && ((0..3).any(|component| {
                            model
                                .expression_basis
                                .at3(basis_index, corner.vertex, component)
                                != 0.0
                        }))
                    {
                        *is_active = true;
                    }
                }
            }
        }
        active
    }

    /// Writes `∂point / ∂e_k` for every mapped point into `deltas`.
    ///
    /// The offsets are exact analytic derivatives of sparse evaluation under
    /// expression coefficient `basis_index`; no surface re-evaluation occurs.
    ///
    /// # Errors
    ///
    /// Fails when `basis_index` is outside the model expression dimension or
    /// when `deltas` does not hold one slot per mapped point.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn expression_point_offsets(
        &self,
        model: &GnmModel,
        basis_index: usize,
        deltas: &mut [[f32; 3]],
    ) -> Result<(), GnmModelError> {
        if basis_index >= model.expression_dimension() {
            return Err(GnmModelError::InvalidValue {
                field: "expression basis index".to_owned(),
                reason: format!(
                    "index {basis_index} is outside 0..{}",
                    model.expression_dimension()
                ),
            });
        }
        if deltas.len() != self.corners.len() {
            return Err(GnmModelError::Shape {
                field: "skinning derivative deltas".to_owned(),
                expected: format!("[{}, 3]", self.corners.len()),
                actual: format!("[{}, 3]", deltas.len()),
            });
        }
        for (delta, point_corners) in deltas.iter_mut().zip(self.corners.iter()) {
            let mut value = [0.0f32; 3];
            for corner in point_corners {
                if corner.weight == 0.0 {
                    continue;
                }
                let basis = [
                    model.expression_basis.at3(basis_index, corner.vertex, 0),
                    model.expression_basis.at3(basis_index, corner.vertex, 1),
                    model.expression_basis.at3(basis_index, corner.vertex, 2),
                ];
                let rotated = matrix_vector(corner.rotation, basis);
                value[0] += corner.weight * rotated[0];
                value[1] += corner.weight * rotated[1];
                value[2] += corner.weight * rotated[2];
            }
            *delta = value;
        }
        Ok(())
    }
}

/// Pre-skinning stage of [`GnmModel::evaluate_sparse`] for one landmark set.
///
/// Stores the deformed template-space vertex values (template + identity +
/// expression, plus pose correctives when the model ships them). When the model
/// has no pose-corrective regressor these values do **not** depend on the
/// joint state, so [`Self::skin`] can be called repeatedly under perturbed
/// joint poses without redoing the basis loops.
#[derive(Clone, Debug, PartialEq)]
pub struct SparsePreparedVertices {
    values: Vec<[f32; 3]>,
    corrective_dependent: bool,
}

impl SparsePreparedVertices {
    /// Returns the number of prepared points.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether nothing was prepared.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns whether the stored values depend on the joint state (pose
    /// correctives), meaning [`Self::skin`] results are only valid for the
    /// joint state used during preparation.
    pub fn depends_on_joint_state(&self) -> bool {
        self.corrective_dependent
    }

    /// Completes sparse evaluation by skinning the stored values under the
    /// supplied joint pose.
    ///
    /// # Errors
    ///
    /// Fails when the joint count or landmark set does not match the model,
    /// mirroring [`GnmModel::evaluate_sparse`] validation, or when any skinned
    /// point becomes non-finite.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn skin(
        &self,
        model: &GnmModel,
        identity: &GnmIdentityState,
        joints: &GnmJointState,
        landmarks: &crate::SparseLandmarkSet,
    ) -> Result<Vec<[f32; 3]>, GnmModelError> {
        validate_state(
            "identity",
            identity.values().len(),
            model.identity_dimension(),
        )?;
        if joints.rotations.len() != model.joint_count() {
            return Err(GnmModelError::Shape {
                field: "joint_rotations".to_owned(),
                expected: format!("[{}, 3]", model.joint_count()),
                actual: format!("[{}, 3]", joints.rotations.len()),
            });
        }
        if self.values.len() != landmarks.unique_vertices().len() {
            return Err(GnmModelError::Shape {
                field: "prepared_vertex_values".to_owned(),
                expected: format!("[{}, 3]", landmarks.unique_vertices().len()),
                actual: format!("[{}, 3]", self.values.len()),
            });
        }

        let mut joint_positions = vec![[0.0f32; 3]; model.joint_count()];
        let mut world_transforms = vec![JointTransform::identity(); model.joint_count()];
        model.joint_transforms(
            joints,
            identity,
            &mut joint_positions,
            &mut world_transforms,
        )?;

        let mut points = Vec::with_capacity(landmarks.len());
        for (landmark, slots) in landmarks
            .points()
            .iter()
            .zip(landmarks.point_vertex_slots())
        {
            let mut point: [f32; 3] = [0.0; 3];
            for (corner, (slot, weight)) in slots.iter().zip(landmark.weights).enumerate() {
                let mut skinned = [0.0; 3];
                let vertex = landmark.indices[corner];
                for (joint, transform) in world_transforms.iter().enumerate() {
                    let skin_weight =
                        model.skinning_weights.values()[joint * model.vertex_count() + vertex];
                    let transformed =
                        transform.apply_skin(self.values[*slot], joint_positions[joint]);
                    for component in 0..3 {
                        skinned[component] += skin_weight * transformed[component];
                    }
                }
                for component in 0..3 {
                    point[component] += weight * skinned[component];
                }
            }
            if point.iter().any(|value| !value.is_finite()) {
                return Err(GnmModelError::NonFinite {
                    field: "sparse_vertices".to_owned(),
                    index: points.len(),
                });
            }
            points.push(point);
        }
        Ok(points)
    }
}

/// Reusable sparse landmark output buffer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GnmSparseVertices {
    values: Vec<[f32; 3]>,
    vertex_values: Vec<[f32; 3]>,
    pose_coefficients: Vec<[f32; 9]>,
    joint_positions: Vec<[f32; 3]>,
    joint_transforms: Vec<JointTransform>,
}

impl GnmSparseVertices {
    /// Creates an output buffer for a fixed number of points.
    pub fn with_len(length: usize) -> Self {
        Self {
            values: vec![[0.0; 3]; length],
            ..Self::default()
        }
    }

    fn resize_points(&mut self, length: usize) {
        self.values.resize(length, [0.0; 3]);
    }

    fn resize_vertex_scratch(&mut self, length: usize) {
        self.vertex_values.resize(length, [0.0; 3]);
    }

    fn resize_joint_scratch(&mut self, length: usize) {
        self.pose_coefficients.resize(length, [0.0; 9]);
        self.joint_positions.resize(length, [0.0; 3]);
        self.joint_transforms
            .resize(length, JointTransform::identity());
    }

    /// Returns the evaluated points.
    pub fn values(&self) -> &[[f32; 3]] {
        &self.values
    }

    /// Returns the number of unique template vertices retained in the scratch buffer.
    pub fn vertex_scratch_len(&self) -> usize {
        self.vertex_values.len()
    }

    /// Returns the scratch capacity used for unique template vertices.
    pub fn vertex_scratch_capacity(&self) -> usize {
        self.vertex_values.capacity()
    }

    /// Returns the capacities of all reusable evaluation buffers.
    pub fn reusable_capacities(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.values.capacity(),
            self.vertex_values.capacity(),
            self.pose_coefficients.capacity(),
            self.joint_positions.capacity(),
            self.joint_transforms.capacity(),
        )
    }
}

fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn subtract(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn matrix_vector(matrix: [[f32; 3]; 3], vector: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn matrix_matrix(left: [[f32; 3]; 3], right: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            for index in 0..3 {
                result[row][column] += left[row][index] * right[index][column];
            }
        }
    }
    result
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn rotation_from_axis_angle(axis_angle: [f32; 3]) -> [[f32; 3]; 3] {
    let theta_squared = axis_angle[0] * axis_angle[0]
        + axis_angle[1] * axis_angle[1]
        + axis_angle[2] * axis_angle[2];
    let theta = theta_squared.max(1.0e-8).sqrt();
    let axis = [
        axis_angle[0] / theta,
        axis_angle[1] / theta,
        axis_angle[2] / theta,
    ];
    let (sin, cos) = theta.sin_cos();
    let mut result = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let delta = if row == column { 1.0 } else { 0.0 };
            result[row][column] = cos * delta
                + (1.0 - cos) * axis[row] * axis[column]
                + sin * skew(axis)[row][column];
        }
    }
    result
}

fn skew(axis: [f32; 3]) -> [[f32; 3]; 3] {
    [
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_model() -> GnmModel {
        let identity = GNM_HEAD_V3_IDENTITY_DIM;
        let expression = GNM_HEAD_V3_EXPRESSION_DIM;
        GnmModel::from_data(GnmModelData {
            version: GnmVersion { major: 3, minor: 0 },
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new(
                "vertices",
                vec![3, 3],
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            .unwrap(),
            template_joints: DenseArray::new("joints", vec![1, 3], vec![0.0, 0.0, 0.0]).unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity, 3, 3],
                vec![0.0; identity * 3 * 3],
            )
            .unwrap(),
            joint_identity_basis: DenseArray::new(
                "joint_identity",
                vec![identity, 1, 3],
                vec![0.0; identity * 3],
            )
            .unwrap(),
            expression_basis: DenseArray::new(
                "expression",
                vec![expression, 3, 3],
                vec![0.0; expression * 3 * 3],
            )
            .unwrap(),
            joint_parent_indices: vec![-1],
            skinning_weights: DenseArray::new("weights", vec![1, 3], vec![1.0; 3]).unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap()
    }

    #[test]
    fn sparse_evaluation_is_barycentric_and_reusable() {
        let model = synthetic_model();
        let landmarks = crate::SparseLandmarkSet::new(vec![
            crate::SparseLandmark::new([0, 1, 2], [0.25, 0.25, 0.5]).unwrap(),
        ])
        .unwrap();
        let identity = model.neutral_identity();
        let expression = model.neutral_expression();
        let joints = GnmJointState::neutral(1);
        let mut output = GnmSparseVertices::with_len(1);
        let capacity = output.values.capacity();
        model
            .evaluate_sparse(&identity, &expression, &joints, &landmarks, &mut output)
            .unwrap();
        assert_eq!(output.values()[0], [0.25, 0.5, 0.0]);
        model
            .evaluate_sparse(&identity, &expression, &joints, &landmarks, &mut output)
            .unwrap();
        assert_eq!(output.values.capacity(), capacity);
    }

    #[test]
    fn same_input_evaluation_is_deterministic() {
        let model = synthetic_model();
        let landmarks = crate::SparseLandmarkSet::new(vec![
            crate::SparseLandmark::new([0, 1, 2], [0.25, 0.25, 0.5]).unwrap(),
        ])
        .unwrap();
        let identity = model.neutral_identity();
        let expression = model.neutral_expression();
        let joints = GnmJointState::neutral(model.joint_count());
        let mut first = GnmSparseVertices::with_len(landmarks.len());
        let mut second = GnmSparseVertices::with_len(landmarks.len());

        model
            .evaluate_sparse(&identity, &expression, &joints, &landmarks, &mut first)
            .unwrap();
        model
            .evaluate_sparse(&identity, &expression, &joints, &landmarks, &mut second)
            .unwrap();

        assert_eq!(first.values(), second.values());
    }

    #[test]
    fn state_non_finite_coefficients_are_typed_errors() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                GnmIdentityState::new(vec![value], 1),
                Err(GnmModelError::NonFinite { field, .. }) if field == "identity"
            ));
            assert!(matches!(
                GnmExpressionState::new(vec![value], 1),
                Err(GnmModelError::NonFinite { field, .. }) if field == "expression"
            ));
            assert!(matches!(
                GnmJointState::new(vec![[value, 0.0, 0.0]], [0.0; 3], 1),
                Err(GnmModelError::NonFinite { field, .. }) if field == "joint_state"
            ));
            assert!(matches!(
                GnmJointState::new(vec![[0.0; 3]], [value, 0.0, 0.0], 1),
                Err(GnmModelError::NonFinite { field, .. }) if field == "joint_state"
            ));
        }
    }

    #[test]
    fn exact_v3_0_is_supported_and_unknown_versions_are_rejected() {
        let model = synthetic_model();
        assert_eq!(model.version(), GNM_HEAD_V3_VERSION);

        for version in [
            GnmVersion { major: 3, minor: 1 },
            GnmVersion { major: 4, minor: 0 },
        ] {
            let data = GnmModelData {
                version,
                variant: model.variant,
                template_vertices: model.template_vertices.clone(),
                template_joints: model.template_joints.clone(),
                vertex_identity_basis: model.vertex_identity_basis.clone(),
                joint_identity_basis: model.joint_identity_basis.clone(),
                expression_basis: model.expression_basis.clone(),
                joint_parent_indices: model.joint_parent_indices.clone(),
                skinning_weights: model.skinning_weights.clone(),
                pose_correctives_regressor: model.pose_correctives_regressor.clone(),
            };
            assert!(matches!(
                GnmModel::from_data(data),
                Err(GnmModelError::UnsupportedVersion(version_text))
                    if version_text == format!("{}.{}", version.major, version.minor)
            ));
        }
    }

    #[test]
    fn identity_applied_joint_positions_drive_fk_and_skinning_offsets() {
        let identity = GNM_HEAD_V3_IDENTITY_DIM;
        let expression = GNM_HEAD_V3_EXPRESSION_DIM;
        let mut joint_identity = vec![0.0; identity * 2 * 3];
        joint_identity[0] = 0.5;
        joint_identity[3] = 0.5;
        let model = GnmModel::from_data(GnmModelData {
            version: GnmVersion { major: 3, minor: 0 },
            variant: GnmVariant::Head,
            template_vertices: DenseArray::new("vertices", vec![1, 3], vec![2.0, 0.0, 0.0])
                .unwrap(),
            template_joints: DenseArray::new(
                "joints",
                vec![2, 3],
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            .unwrap(),
            vertex_identity_basis: DenseArray::new(
                "identity",
                vec![identity, 1, 3],
                vec![0.0; identity * 3],
            )
            .unwrap(),
            joint_identity_basis: DenseArray::new(
                "joint_identity",
                vec![identity, 2, 3],
                joint_identity,
            )
            .unwrap(),
            expression_basis: DenseArray::new(
                "expression",
                vec![expression, 1, 3],
                vec![0.0; expression * 3],
            )
            .unwrap(),
            joint_parent_indices: vec![-1, 0],
            skinning_weights: DenseArray::new("weights", vec![2, 1], vec![0.0, 1.0]).unwrap(),
            pose_correctives_regressor: None,
        })
        .unwrap();
        let landmarks = crate::SparseLandmarkSet::new(vec![
            crate::SparseLandmark::new([0, 0, 0], [1.0, 0.0, 0.0]).unwrap(),
        ])
        .unwrap();
        let mut identity_values = vec![0.0; identity];
        identity_values[0] = 1.0;
        let identity = GnmIdentityState::new(identity_values, identity).unwrap();
        let expression = model.neutral_expression();
        let joints = GnmJointState::new(
            vec![[0.0; 3], [0.0, 0.0, std::f32::consts::FRAC_PI_2]],
            [0.0; 3],
            2,
        )
        .unwrap();
        let mut output = GnmSparseVertices::with_len(1);
        model
            .evaluate_sparse(&identity, &expression, &joints, &landmarks, &mut output)
            .unwrap();
        assert!((output.values()[0][0] - 1.5).abs() < 1e-5);
        assert!((output.values()[0][1] - 0.5).abs() < 1e-5);
        assert!(output.values()[0][2].abs() < 1e-5);
    }

    #[test]
    fn wrong_head_dimensions_are_rejected() {
        let model = synthetic_model();
        let data = GnmModelData {
            version: model.version,
            variant: model.variant,
            template_vertices: model.template_vertices,
            template_joints: model.template_joints,
            vertex_identity_basis: DenseArray::new("identity", vec![1, 3, 3], vec![0.0; 9])
                .unwrap(),
            joint_identity_basis: model.joint_identity_basis,
            expression_basis: model.expression_basis,
            joint_parent_indices: model.joint_parent_indices,
            skinning_weights: model.skinning_weights,
            pose_correctives_regressor: model.pose_correctives_regressor,
        };
        assert!(
            matches!(GnmModel::from_data(data), Err(GnmModelError::Shape { field, .. }) if field == "joint_identity_basis")
        );
    }
}
