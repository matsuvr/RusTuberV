//! Landmark schema identification for the Peppa_Pig_Face_Landmark student
//! 256x256 model.

use vtuber_core::types::LandmarkSchemaId;

/// Schema ID for the Peppa_Pig_Face_Landmark student 256x256 model.
///
/// This identifies the 98-point landmark set. Expression coefficients are a
/// separate concern: the decoder in [`crate::decode::expressions`] needs named
/// blendshape output, which this model does not provide, so no index set is
/// declared for it here.
pub const SCHEMA_PEPPAPIG_98: LandmarkSchemaId = LandmarkSchemaId("peppapig-98");
