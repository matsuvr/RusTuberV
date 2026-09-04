//! Observable non-tongue GNM expression basis artifacts (Issue #15).

use nalgebra::{DMatrix, linalg::SymmetricEigen};
use serde::{Deserialize, Serialize};
use vtuber_gnm::{GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM, GnmNonTongueExpression};

/// Schema version of [`ObservableGnmBasisArtifact`].
pub const OBSERVABLE_GNM_BASIS_SCHEMA_VERSION: u32 = 1;

/// Inputs that bind an observable basis to its model, mapping, and takes.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservableBasisProvenance {
    /// SHA-256 of the pinned GNM model bytes.
    pub model_sha256: String,
    /// Dense mapping schema revision used to build every Jacobian.
    pub mapping_schema_revision: u32,
    /// Explicit, ordered training take ids.
    pub training_takes: Vec<String>,
}

/// Versioned observable non-tongue expression basis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservableGnmBasisArtifact {
    /// Artifact schema version.
    pub schema_version: u32,
    /// SHA-256 of the pinned GNM model bytes.
    pub model_sha256: String,
    /// Dense mapping schema revision.
    pub mapping_schema_revision: u32,
    /// Number of retained observable directions.
    pub rank: usize,
    /// Fixed source expression dimension (351).
    pub source_dimension: usize,
    /// Ordered take ids included in the Gram matrix.
    pub training_takes: Vec<String>,
    /// Retained eigenvalues in descending order.
    pub eigenvalues_descending: Vec<f64>,
    /// Fraction of total Gram energy retained by the selected rank.
    pub retained_energy_fraction: f64,
    /// Row-major `[351, rank]` orthonormal basis.
    pub basis_row_major: Vec<f32>,
    /// Deterministic content hash over every preceding field.
    pub content_hash: u64,
}

/// Typed observable-basis failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ObservableBasisError {
    /// Input shape does not match the fixed Head-v3 contract.
    #[error("invalid observable basis shape: {0}")]
    InvalidShape(&'static str),
    /// Rank is outside `1..=351`.
    #[error("observable basis rank {0} is outside 1..=351")]
    InvalidRank(usize),
    /// Training provenance is incomplete.
    #[error("observable basis provenance is incomplete")]
    InvalidProvenance,
    /// A numeric input or result was non-finite or had no energy.
    #[error("invalid observable basis numeric value: {0}")]
    InvalidNumeric(&'static str),
    /// Artifact hash does not match its contents.
    #[error("observable basis content hash mismatch: expected {expected}, computed {actual}")]
    HashMismatch {
        /// Hash stored in the artifact.
        expected: u64,
        /// Hash recomputed from decoded fields.
        actual: u64,
    },
    /// Compact expression validation failed.
    #[error(transparent)]
    Expression(#[from] vtuber_gnm::GnmModelError),
}

/// Fits the top observable eigenspace from a packed lower-triangle Gram matrix.
///
/// # Errors
///
/// Rejects invalid dimensions, rank, provenance, non-finite values, or a Gram
/// matrix with no positive total energy.
#[allow(clippy::indexing_slicing)]
pub fn fit_observable_gnm_basis(
    gram_lower_triangle: &[f64],
    training_frame_count: usize,
    rank: usize,
    provenance: ObservableBasisProvenance,
) -> Result<ObservableGnmBasisArtifact, ObservableBasisError> {
    let dimension = GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM;
    if gram_lower_triangle.len() != dimension * (dimension + 1) / 2 || training_frame_count == 0 {
        return Err(ObservableBasisError::InvalidShape(
            "Gram triangle or training frame count",
        ));
    }
    if !(1..=dimension).contains(&rank) {
        return Err(ObservableBasisError::InvalidRank(rank));
    }
    if provenance.model_sha256.is_empty() || provenance.training_takes.is_empty() {
        return Err(ObservableBasisError::InvalidProvenance);
    }
    let mut gram = DMatrix::<f64>::zeros(dimension, dimension);
    for row in 0..dimension {
        let start = row * (row + 1) / 2;
        for column in 0..=row {
            let value = gram_lower_triangle[start + column];
            if !value.is_finite() {
                return Err(ObservableBasisError::InvalidNumeric("Gram entry"));
            }
            gram[(row, column)] = value;
            gram[(column, row)] = value;
        }
    }
    let decomposition = SymmetricEigen::new(gram);
    if decomposition
        .eigenvalues
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(ObservableBasisError::InvalidNumeric("eigenvalue"));
    }
    let mut order: Vec<usize> = (0..dimension).collect();
    order.sort_by(|left, right| {
        decomposition.eigenvalues[*right].total_cmp(&decomposition.eigenvalues[*left])
    });
    let total_energy: f64 = decomposition.eigenvalues.iter().sum();
    if !total_energy.is_finite() || total_energy <= 0.0 {
        return Err(ObservableBasisError::InvalidNumeric("Gram energy"));
    }
    let eigenvalues_descending: Vec<f64> = order
        .iter()
        .take(rank)
        .map(|index| decomposition.eigenvalues[*index])
        .collect();
    let retained_energy_fraction = eigenvalues_descending.iter().sum::<f64>() / total_energy;
    if !retained_energy_fraction.is_finite() {
        return Err(ObservableBasisError::InvalidNumeric("retained energy"));
    }
    let mut basis_row_major = vec![0.0_f32; dimension * rank];
    for (basis_column, eigen_column) in order.iter().take(rank).enumerate() {
        let mut sign = 1.0_f64;
        let mut largest = 0.0_f64;
        for row in 0..dimension {
            let value = decomposition.eigenvectors[(row, *eigen_column)];
            if value.abs() > largest {
                largest = value.abs();
                sign = if value < 0.0 { -1.0 } else { 1.0 };
            }
        }
        for row in 0..dimension {
            basis_row_major[row * rank + basis_column] =
                (sign * decomposition.eigenvectors[(row, *eigen_column)]) as f32;
        }
    }
    let mut artifact = ObservableGnmBasisArtifact {
        schema_version: OBSERVABLE_GNM_BASIS_SCHEMA_VERSION,
        model_sha256: provenance.model_sha256,
        mapping_schema_revision: provenance.mapping_schema_revision,
        rank,
        source_dimension: dimension,
        training_takes: provenance.training_takes,
        eigenvalues_descending,
        retained_energy_fraction,
        basis_row_major,
        content_hash: 0,
    };
    artifact.content_hash = observable_basis_hash(&artifact);
    Ok(artifact)
}

/// Projects `q = O^T φ`.
///
/// # Errors
///
/// Rejects an invalid artifact shape/hash or non-finite output.
#[allow(clippy::indexing_slicing)]
pub fn project_non_tongue_expression(
    expression: &GnmNonTongueExpression,
    basis: &ObservableGnmBasisArtifact,
) -> Result<Vec<f32>, ObservableBasisError> {
    validate_artifact(basis)?;
    let mut reduced = vec![0.0_f32; basis.rank];
    for (row, expression_value) in expression.values().iter().enumerate() {
        for (column, reduced_value) in reduced.iter_mut().enumerate() {
            *reduced_value += basis.basis_row_major[row * basis.rank + column] * expression_value;
        }
    }
    if reduced.iter().any(|value| !value.is_finite()) {
        return Err(ObservableBasisError::InvalidNumeric("projected expression"));
    }
    Ok(reduced)
}

/// Reconstructs `φ_hat = O q`.
///
/// # Errors
///
/// Rejects an invalid artifact, reduced dimension, or non-finite result.
#[allow(clippy::indexing_slicing)]
pub fn reconstruct_non_tongue_expression(
    reduced: &[f32],
    basis: &ObservableGnmBasisArtifact,
) -> Result<GnmNonTongueExpression, ObservableBasisError> {
    validate_artifact(basis)?;
    if reduced.len() != basis.rank {
        return Err(ObservableBasisError::InvalidShape("reduced expression"));
    }
    let mut expression = vec![0.0_f32; basis.source_dimension];
    for (row, expression_value) in expression.iter_mut().enumerate() {
        for (column, reduced_value) in reduced.iter().enumerate() {
            *expression_value += basis.basis_row_major[row * basis.rank + column] * reduced_value;
        }
    }
    Ok(GnmNonTongueExpression::try_from_values(expression)?)
}

fn validate_artifact(basis: &ObservableGnmBasisArtifact) -> Result<(), ObservableBasisError> {
    if basis.schema_version != OBSERVABLE_GNM_BASIS_SCHEMA_VERSION
        || basis.source_dimension != GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM
        || !(1..=basis.source_dimension).contains(&basis.rank)
        || basis.eigenvalues_descending.len() != basis.rank
        || basis.basis_row_major.len() != basis.source_dimension * basis.rank
    {
        return Err(ObservableBasisError::InvalidShape("artifact"));
    }
    let actual = observable_basis_hash(basis);
    if actual != basis.content_hash {
        return Err(ObservableBasisError::HashMismatch {
            expected: basis.content_hash,
            actual,
        });
    }
    Ok(())
}

fn observable_basis_hash(artifact: &ObservableGnmBasisArtifact) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut update = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    update(&artifact.schema_version.to_le_bytes());
    update(artifact.model_sha256.as_bytes());
    update(&(artifact.mapping_schema_revision).to_le_bytes());
    update(&(artifact.rank as u64).to_le_bytes());
    update(&(artifact.source_dimension as u64).to_le_bytes());
    for take in &artifact.training_takes {
        update(take.as_bytes());
        update(&[0xff]);
    }
    for value in &artifact.eigenvalues_descending {
        update(&(*value as f32).to_bits().to_le_bytes());
    }
    update(
        &(artifact.retained_energy_fraction as f32)
            .to_bits()
            .to_le_bytes(),
    );
    for value in &artifact.basis_row_major {
        let canonical = if *value == 0.0 { 0.0 } else { *value };
        update(&canonical.to_bits().to_le_bytes());
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal_gram(values: &[f64]) -> Vec<f64> {
        let dimension = GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM;
        let mut gram = vec![0.0; dimension * (dimension + 1) / 2];
        for (index, value) in values.iter().enumerate() {
            gram[index * (index + 1) / 2 + index] = *value;
        }
        gram
    }

    fn provenance() -> ObservableBasisProvenance {
        ObservableBasisProvenance {
            model_sha256: "ABC".to_owned(),
            mapping_schema_revision: 1,
            training_takes: vec!["take-a".to_owned()],
        }
    }

    #[test]
    fn synthetic_gram_recovers_ordered_signed_orthonormal_subspace() {
        let artifact =
            fit_observable_gnm_basis(&diagonal_gram(&[9.0, 4.0, 1.0]), 3, 2, provenance()).unwrap();
        assert_eq!(artifact.eigenvalues_descending, vec![9.0, 4.0]);
        assert!((artifact.retained_energy_fraction - 13.0 / 14.0).abs() < 1.0e-12);
        assert_eq!(artifact.basis_row_major[0], 1.0);
        assert_eq!(artifact.basis_row_major[artifact.rank + 1], 1.0);
        for left in 0..artifact.rank {
            for right in 0..artifact.rank {
                let dot: f32 = (0..artifact.source_dimension)
                    .map(|row| {
                        artifact.basis_row_major[row * artifact.rank + left]
                            * artifact.basis_row_major[row * artifact.rank + right]
                    })
                    .sum();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1.0e-6);
            }
        }

        let repeated =
            fit_observable_gnm_basis(&diagonal_gram(&[9.0, 4.0, 1.0]), 3, 2, provenance()).unwrap();
        assert_eq!(artifact, repeated);
        assert_eq!(
            serde_json::to_vec(&artifact).unwrap(),
            serde_json::to_vec(&repeated).unwrap()
        );
    }

    #[test]
    fn project_and_reconstruct_use_only_the_selected_basis() {
        let artifact =
            fit_observable_gnm_basis(&diagonal_gram(&[9.0, 4.0, 1.0]), 3, 2, provenance()).unwrap();
        let mut values = vec![0.0; GNM_HEAD_V3_NON_TONGUE_EXPRESSION_DIM];
        values[0] = 2.0;
        values[1] = -3.0;
        values[2] = 5.0;
        let expression = GnmNonTongueExpression::try_from_values(values).unwrap();
        let reduced = project_non_tongue_expression(&expression, &artifact).unwrap();
        assert_eq!(reduced, vec![2.0, -3.0]);
        let reconstructed = reconstruct_non_tongue_expression(&reduced, &artifact).unwrap();
        assert_eq!(reconstructed.values()[0], 2.0);
        assert_eq!(reconstructed.values()[1], -3.0);
        assert_eq!(reconstructed.values()[2], 0.0);
    }

    #[test]
    fn json_round_trip_preserves_content_hash_when_basis_contains_negative_zero() {
        let mut artifact =
            fit_observable_gnm_basis(&diagonal_gram(&[9.0, 4.0]), 2, 2, provenance()).unwrap();
        artifact.basis_row_major[2] = -0.0;
        artifact.eigenvalues_descending[0] = 1.562_487_917_078_665_9;
        artifact.retained_energy_fraction = 0.983_849_456_123_456_7;
        artifact.content_hash = observable_basis_hash(&artifact);
        let json = serde_json::to_vec(&artifact).unwrap();
        let loaded: ObservableGnmBasisArtifact = serde_json::from_slice(&json).unwrap();
        validate_artifact(&loaded).unwrap();
    }
}
