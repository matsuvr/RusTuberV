//! Import-time VRM 0.x normalization.
//!
//! Converts VRM 0.x sources into VRM 1.0-shaped managed copies so the
//! unmodified upstream runtime loads them without any vendored loader or ECS
//! patch. The modules below are ported from the removed vendored patch and
//! retargeted at file conversion (see each module's docs); [`convert`] ties
//! them together over raw GLB bytes.

pub mod convert;
pub mod descriptor;
pub mod materials;
pub(crate) mod normalize;

pub use convert::{Vrm0ConvertError, convert_vrm0_to_vrm1, prepare_managed_vrm_bytes};
pub use descriptor::{
    LegacyShaderKind, VrmCompatibilityWarning, VrmCompatibilityWarningCode, classify_legacy_shader,
    collect_legacy_compatibility_warnings,
};
