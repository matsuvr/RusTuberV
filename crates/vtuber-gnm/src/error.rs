//! Errors returned by the GNM model boundary.

use std::fmt::{Display, Formatter};

/// Failure while loading, validating, or evaluating GNM data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GnmModelError {
    /// A required array was not present in the model archive.
    MissingArray(String),
    /// An array had a different shape from the schema contract.
    Shape {
        /// Array name.
        field: String,
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        actual: String,
    },
    /// A numeric value was not finite.
    NonFinite {
        /// Array or state name.
        field: String,
        /// Flattened array index.
        index: usize,
    },
    /// A value violated a semantic constraint.
    InvalidValue {
        /// Array or state name.
        field: String,
        /// Human-readable reason.
        reason: String,
    },
    /// The archive is not a supported GNM version.
    UnsupportedVersion(String),
    /// The archive is not a supported GNM variant.
    UnsupportedVariant(String),
    /// ZIP archive failure.
    Archive(String),
    /// NPY payload failure.
    Npy(String),
    /// File-system failure.
    Io(String),
}

impl Display for GnmModelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingArray(field) => write!(formatter, "missing GNM array `{field}`"),
            Self::Shape {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "invalid shape for `{field}`: expected {expected}, got {actual}"
            ),
            Self::NonFinite { field, index } => {
                write!(formatter, "non-finite value in `{field}` at index {index}")
            }
            Self::InvalidValue { field, reason } => {
                write!(formatter, "invalid value in `{field}`: {reason}")
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported GNM version `{version}`")
            }
            Self::UnsupportedVariant(variant) => {
                write!(formatter, "unsupported GNM variant `{variant}`")
            }
            Self::Archive(message) => write!(formatter, "GNM archive error: {message}"),
            Self::Npy(message) => write!(formatter, "GNM NPY error: {message}"),
            Self::Io(message) => write!(formatter, "GNM I/O error: {message}"),
        }
    }
}

impl std::error::Error for GnmModelError {}
