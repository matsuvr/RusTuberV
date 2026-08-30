//! Official sparse landmark definitions used by the GNM head model.

use std::sync::OnceLock;

use crate::GnmModelError;

/// One barycentric landmark over three template vertices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseLandmark {
    /// Three vertex indices in the official template mesh.
    pub indices: [usize; 3],
    /// Barycentric weights corresponding to [`Self::indices`].
    pub weights: [f32; 3],
}

impl SparseLandmark {
    /// Creates a finite, non-negative, normalized barycentric landmark.
    pub fn new(indices: [usize; 3], weights: [f32; 3]) -> Result<Self, GnmModelError> {
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(GnmModelError::InvalidValue {
                field: "sparse landmark weights".to_owned(),
                reason: "weights must be finite and non-negative".to_owned(),
            });
        }
        let sum = weights.iter().sum::<f32>();
        if (sum - 1.0).abs() > 0.002 {
            return Err(GnmModelError::InvalidValue {
                field: "sparse landmark weights".to_owned(),
                reason: format!("weights must sum to one, got {sum}"),
            });
        }
        Ok(Self { indices, weights })
    }
}

/// Ordered sparse landmark set.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseLandmarkSet {
    points: Vec<SparseLandmark>,
    unique_vertices: Vec<usize>,
    point_vertex_slots: Vec<[usize; 3]>,
}

impl SparseLandmarkSet {
    /// Creates a non-empty landmark set.
    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    pub fn new(points: Vec<SparseLandmark>) -> Result<Self, GnmModelError> {
        if points.is_empty() {
            return Err(GnmModelError::InvalidValue {
                field: "sparse landmarks".to_owned(),
                reason: "at least one point is required".to_owned(),
            });
        }
        let mut unique_vertices = Vec::new();
        let mut point_vertex_slots = Vec::with_capacity(points.len());
        for point in &points {
            let mut slots = [0; 3];
            for (slot, vertex) in point.indices.iter().copied().enumerate() {
                let unique_slot = match unique_vertices
                    .iter()
                    .position(|candidate| *candidate == vertex)
                {
                    Some(slot) => slot,
                    None => {
                        unique_vertices.push(vertex);
                        unique_vertices.len() - 1
                    }
                };
                slots[slot] = unique_slot;
            }
            point_vertex_slots.push(slots);
        }
        Ok(Self {
            points,
            unique_vertices,
            point_vertex_slots,
        })
    }

    /// Returns the number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns whether the set contains no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns points in their stable source order.
    pub fn points(&self) -> &[SparseLandmark] {
        &self.points
    }

    /// Returns the number of unique template vertices referenced by this set.
    pub fn unique_vertex_count(&self) -> usize {
        self.unique_vertices.len()
    }

    pub(crate) fn unique_vertices(&self) -> &[usize] {
        &self.unique_vertices
    }

    pub(crate) fn point_vertex_slots(&self) -> &[[usize; 3]] {
        &self.point_vertex_slots
    }

    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn from_text(text: &str) -> Result<Self, GnmModelError> {
        let mut points = Vec::new();
        for (line_index, line) in text.lines().enumerate() {
            let values: Vec<&str> = line.split_whitespace().collect();
            if values.len() != 6 {
                return Err(GnmModelError::InvalidValue {
                    field: "head_sparse_68".to_owned(),
                    reason: format!("line {} must contain six values", line_index + 1),
                });
            }
            let mut indices = [0; 3];
            let mut weights = [0.0; 3];
            for index in 0..3 {
                indices[index] =
                    values[index * 2]
                        .parse()
                        .map_err(|_| GnmModelError::InvalidValue {
                            field: "head_sparse_68".to_owned(),
                            reason: format!("invalid vertex index on line {}", line_index + 1),
                        })?;
                weights[index] =
                    values[index * 2 + 1]
                        .parse()
                        .map_err(|_| GnmModelError::InvalidValue {
                            field: "head_sparse_68".to_owned(),
                            reason: format!("invalid weight on line {}", line_index + 1),
                        })?;
            }
            points.push(Self::landmark_from_source(indices, weights)?);
        }
        if points.len() != 68 {
            return Err(GnmModelError::InvalidValue {
                field: "head_sparse_68".to_owned(),
                reason: format!("expected 68 points, got {}", points.len()),
            });
        }
        Self::new(points)
    }

    fn landmark_from_source(
        indices: [usize; 3],
        weights: [f32; 3],
    ) -> Result<SparseLandmark, GnmModelError> {
        SparseLandmark::new(indices, weights)
    }
}

/// The official 68-point GNM Head sparse landmark contract.
pub fn head_sparse_68() -> &'static SparseLandmarkSet {
    static SET: OnceLock<SparseLandmarkSet> = OnceLock::new();
    // Invariant: the committed asset is validated by repository tests.
    #[allow(clippy::expect_used)]
    SET.get_or_init(|| {
        SparseLandmarkSet::from_text(include_str!("../assets/head_sparse_68.txt"))
            .expect("repository-owned head_sparse_68.txt must satisfy its 68-point contract")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_landmark_contract_has_68_points() {
        let set = head_sparse_68();
        assert_eq!(set.len(), 68);
        assert_eq!(set.points()[0].indices, [8777, 8841, 11165]);
        assert!((set.points()[0].weights.iter().sum::<f32>() - 1.0).abs() < 0.002);
    }

    #[test]
    fn official_landmark_contract_order_and_values_have_stable_hash() {
        let set = head_sparse_68();
        let mut hash = 0xcbf29ce484222325u64;
        for point in set.points() {
            for index in point.indices {
                for byte in (index as u64).to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
            for weight in point.weights {
                for byte in weight.to_bits().to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
        }
        assert_eq!(set.len(), 68);
        assert_eq!(hash, 18_219_087_812_053_459_804);
    }

    #[test]
    fn invalid_weights_are_rejected() {
        assert!(SparseLandmark::new([0, 1, 2], [0.5, -0.1, 0.6]).is_err());
        assert!(SparseLandmark::new([0, 1, 2], [0.1, 0.1, 0.1]).is_err());
    }
}
