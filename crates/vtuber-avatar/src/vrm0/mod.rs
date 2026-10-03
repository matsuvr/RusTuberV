//! Import-time VRM 0.x normalization.
//!
//! Converts VRM 0.x sources into VRM 1.0-shaped managed copies so the
//! unmodified upstream runtime loads them without any vendored loader or ECS
//! patch. The modules below are ported from the removed vendored patch and
//! retargeted at file conversion (see each module's docs); [`convert`] ties
//! them together over the GLB JSON document.

pub(crate) mod convert;
pub(crate) mod descriptor;
pub(crate) mod expression_id;
pub(crate) mod materials;
pub(crate) mod normalize;

pub use descriptor::{VrmCompatibilityWarning, VrmCompatibilityWarningCode};
