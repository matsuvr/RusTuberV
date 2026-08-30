//! Minimal, deterministic reader for the official GNM `.npz` schema.

use std::{collections::HashMap, fs::File, io::Read, path::Path};

use zip::ZipArchive;

use crate::{DenseArray, GnmModel, GnmModelData, GnmModelError, GnmVariant, GnmVersion};

/// Required keys in the official `gnm_data_schema.py` archive.
pub const GNM_DATA_SCHEMA_KEYS: &[&str] = &[
    "version",
    "variant",
    "template_vertex_positions",
    "template_joint_positions",
    "vertex_identity_basis",
    "joint_identity_basis",
    "expression_basis",
    "identity_names",
    "joint_names",
    "expression_names",
    "joint_parent_indices",
    "skinning_weights",
    "quads",
    "triangles",
    "quad_uvs",
    "triangle_uvs",
    "mesh_component_names",
    "mirror_indices",
    "joint_regressor",
    "pose_correctives_regressor",
    "bone_aligned_template_joint_orientations",
    "vertex_groups",
    "vertex_group_names",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NpyDType {
    I8,
    I16,
    F32,
    F64,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bytes(usize),
    Unicode(usize),
}

#[derive(Clone, Debug)]
struct NpyArray {
    dtype: NpyDType,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Loads and validates an official GNM Head v3 `.npz` archive.
// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
pub fn load_gnm_head_v3(path: impl AsRef<Path>) -> Result<GnmModel, GnmModelError> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|error| GnmModelError::Io(error.to_string()))?;
    let mut archive =
        ZipArchive::new(file).map_err(|error| GnmModelError::Archive(error.to_string()))?;
    let mut arrays = HashMap::new();
    for key in GNM_DATA_SCHEMA_KEYS {
        let name = format!("{key}.npy");
        let mut entry = archive
            .by_name(&name)
            .map_err(|_| GnmModelError::MissingArray((*key).to_owned()))?;
        let mut data = Vec::new();
        entry
            .read_to_end(&mut data)
            .map_err(|error| GnmModelError::Archive(error.to_string()))?;
        arrays.insert((*key).to_owned(), parse_npy(&data)?);
    }

    let version_text = take(&mut arrays, "version")?.single_string("version")?;
    let version = parse_version(&version_text)?;
    let variant_text = take(&mut arrays, "variant")?.single_string("variant")?;
    let variant = match variant_text.as_str() {
        "head" => GnmVariant::Head,
        other => return Err(GnmModelError::UnsupportedVariant(other.to_owned())),
    };
    let template_vertices = dense(
        &take(&mut arrays, "template_vertex_positions")?,
        "template_vertex_positions",
    )?;
    let template_joints = dense(
        &take(&mut arrays, "template_joint_positions")?,
        "template_joint_positions",
    )?;
    let vertex_identity_basis = dense(
        &take(&mut arrays, "vertex_identity_basis")?,
        "vertex_identity_basis",
    )?;
    let joint_identity_basis = dense(
        &take(&mut arrays, "joint_identity_basis")?,
        "joint_identity_basis",
    )?;
    let expression_basis = dense(&take(&mut arrays, "expression_basis")?, "expression_basis")?;

    let vertex_count = exact_shape("template_vertex_positions", template_vertices.shape(), 2)?[0];
    let joint_count = exact_shape("template_joint_positions", template_joints.shape(), 2)?[0];
    if exact_shape("template_vertex_positions", template_vertices.shape(), 2)?[1] != 3
        || exact_shape("template_joint_positions", template_joints.shape(), 2)?[1] != 3
    {
        return Err(GnmModelError::Shape {
            field: "template positions".to_owned(),
            expected: "[count, 3]".to_owned(),
            actual: format!(
                "vertices={:?}, joints={:?}",
                template_vertices.shape(),
                template_joints.shape()
            ),
        });
    }
    let identity_count = exact_shape("vertex_identity_basis", vertex_identity_basis.shape(), 3)?[0];
    let expression_count = exact_shape("expression_basis", expression_basis.shape(), 3)?[0];
    let identity_names = take(&mut arrays, "identity_names")?.strings("identity_names")?;
    let joint_names = take(&mut arrays, "joint_names")?.strings("joint_names")?;
    let expression_names = take(&mut arrays, "expression_names")?.strings("expression_names")?;
    if identity_names.len() != identity_count
        || expression_names.len() != expression_count
        || joint_names.len() != joint_count
    {
        return Err(GnmModelError::InvalidValue {
            field: "name arrays".to_owned(),
            reason: format!(
                "name counts must match identity={identity_count}, joint={joint_count}, expression={expression_count}"
            ),
        });
    }
    let joint_parent_indices = take(&mut arrays, "joint_parent_indices")?
        .i32_values("joint_parent_indices", &[joint_count])?;
    let skinning_weights = dense(&take(&mut arrays, "skinning_weights")?, "skinning_weights")?;
    exact_shape("skinning_weights", skinning_weights.shape(), 2)?;
    let skinning_shape = skinning_weights.shape();
    if skinning_shape != [joint_count, vertex_count] {
        return Err(GnmModelError::Shape {
            field: "skinning_weights".to_owned(),
            expected: format!("[{joint_count}, {vertex_count}]"),
            actual: format!("{skinning_shape:?}"),
        });
    }

    shape_only(&take(&mut arrays, "quads")?, "quads", 2)?;
    shape_only(&take(&mut arrays, "triangles")?, "triangles", 2)?;
    shape_only(&take(&mut arrays, "quad_uvs")?, "quad_uvs", 3)?;
    shape_only(&take(&mut arrays, "triangle_uvs")?, "triangle_uvs", 3)?;
    strings_only(
        &take(&mut arrays, "mesh_component_names")?,
        "mesh_component_names",
    )?;
    let mirror_indices =
        take(&mut arrays, "mirror_indices")?.i32_values("mirror_indices", &[vertex_count])?;
    if mirror_indices
        .iter()
        .any(|index| *index < 0 || *index >= vertex_count as i32)
    {
        return Err(GnmModelError::InvalidValue {
            field: "mirror_indices".to_owned(),
            reason: "every mirror index must address a template vertex".to_owned(),
        });
    }
    let joint_regressor = dense(&take(&mut arrays, "joint_regressor")?, "joint_regressor")?;
    if joint_regressor.shape() != [joint_count, vertex_count] {
        return Err(GnmModelError::Shape {
            field: "joint_regressor".to_owned(),
            expected: format!("[{joint_count}, {vertex_count}]"),
            actual: format!("{:?}", joint_regressor.shape()),
        });
    }
    let pose_correctives = dense(
        &take(&mut arrays, "pose_correctives_regressor")?,
        "pose_correctives_regressor",
    )?;
    let pose_correctives = normalize_pose_correctives(pose_correctives, joint_count, vertex_count)?;
    let orientations = dense(
        &take(&mut arrays, "bone_aligned_template_joint_orientations")?,
        "bone_aligned_template_joint_orientations",
    )?;
    if orientations.shape() != [joint_count, 3, 3] {
        return Err(GnmModelError::Shape {
            field: "bone_aligned_template_joint_orientations".to_owned(),
            expected: format!("[{joint_count}, 3, 3]"),
            actual: format!("{:?}", orientations.shape()),
        });
    }
    shape_only(&take(&mut arrays, "vertex_groups")?, "vertex_groups", 2)?;
    strings_only(
        &take(&mut arrays, "vertex_group_names")?,
        "vertex_group_names",
    )?;
    if identity_count != crate::GNM_HEAD_V3_IDENTITY_DIM
        || expression_count != crate::GNM_HEAD_V3_EXPRESSION_DIM
    {
        return Err(GnmModelError::InvalidValue {
            field: "model dimensions".to_owned(),
            reason: format!(
                "Head v3 requires identity={} and expression={}, got identity={identity_count}, expression={expression_count}",
                crate::GNM_HEAD_V3_IDENTITY_DIM,
                crate::GNM_HEAD_V3_EXPRESSION_DIM
            ),
        });
    }
    GnmModel::from_data(GnmModelData {
        version,
        variant,
        template_vertices,
        template_joints,
        vertex_identity_basis,
        joint_identity_basis,
        expression_basis,
        joint_parent_indices,
        skinning_weights,
        pose_correctives_regressor: Some(pose_correctives),
    })
}

fn take(arrays: &mut HashMap<String, NpyArray>, key: &str) -> Result<NpyArray, GnmModelError> {
    arrays
        .remove(key)
        .ok_or_else(|| GnmModelError::MissingArray(key.to_owned()))
}

fn dense(array: &NpyArray, field: &str) -> Result<DenseArray, GnmModelError> {
    DenseArray::new(field, array.shape.clone(), array.f32_values(field)?)
}

fn exact_shape(field: &str, shape: &[usize], rank: usize) -> Result<Vec<usize>, GnmModelError> {
    if shape.len() != rank {
        return Err(GnmModelError::Shape {
            field: field.to_owned(),
            expected: format!("rank {rank}"),
            actual: format!("{shape:?}"),
        });
    }
    Ok(shape.to_vec())
}

fn shape_only(array: &NpyArray, field: &str, rank: usize) -> Result<(), GnmModelError> {
    exact_shape(field, &array.shape, rank)?;
    array.f32_values(field).map(|_| ())
}

fn strings_only(array: &NpyArray, field: &str) -> Result<(), GnmModelError> {
    array.strings(field).map(|_| ())
}

fn parse_version(version: &str) -> Result<GnmVersion, GnmModelError> {
    let mut parts = version.split('.');
    let major = parts.next().and_then(|part| part.parse().ok());
    let minor = parts.next().and_then(|part| part.parse().ok());
    match (major, minor, parts.next()) {
        (Some(major), Some(minor), None) => Ok(GnmVersion { major, minor }),
        _ => Err(GnmModelError::UnsupportedVersion(version.to_owned())),
    }
}

fn normalize_pose_correctives(
    array: DenseArray,
    joint_count: usize,
    vertex_count: usize,
) -> Result<DenseArray, GnmModelError> {
    let expected_rows = joint_count * 9;
    match array.shape() {
        [rows, columns] if *rows == expected_rows && *columns == vertex_count * 3 => {
            DenseArray::new(
                "pose_correctives_regressor",
                vec![expected_rows, vertex_count, 3],
                array.values().to_vec(),
            )
        }
        [rows, vertices, three]
            if *rows == expected_rows && *vertices == vertex_count && *three == 3 =>
        {
            Ok(array)
        }
        _ => Err(GnmModelError::Shape {
            field: "pose_correctives_regressor".to_owned(),
            expected: format!(
                "[{expected_rows}, {vertex_count}, 3] or [{expected_rows}, {}]",
                vertex_count * 3
            ),
            actual: format!("{:?}", array.shape()),
        }),
    }
}

// Bounds are guaranteed by construction in this numeric kernel
// (loop ranges bounded by buffer lengths / fixed-size dimensions);
// see the AGENTS.md production panic policy.
#[allow(clippy::indexing_slicing)]
fn parse_npy(bytes: &[u8]) -> Result<NpyArray, GnmModelError> {
    if bytes.len() < 8 || bytes.get(0..6) != Some(b"\x93NUMPY") {
        return Err(GnmModelError::Npy("missing NPY magic".to_owned()));
    }
    let major = bytes[6];
    let header_length_size = match major {
        1 => 2,
        2 | 3 => 4,
        other => {
            return Err(GnmModelError::Npy(format!(
                "unsupported NPY major version {other}"
            )));
        }
    };
    let header_start = 8usize;
    let header_length_end = header_start
        .checked_add(header_length_size)
        .ok_or_else(|| GnmModelError::Npy("NPY header length offset overflow".to_owned()))?;
    if bytes.len() < header_length_end {
        return Err(GnmModelError::Npy("truncated NPY header length".to_owned()));
    }
    let header_length = match header_length_size {
        2 => u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        _ => u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
    };
    let data_start = header_length_end
        .checked_add(header_length)
        .ok_or_else(|| GnmModelError::Npy("NPY data offset overflow".to_owned()))?;
    if data_start > bytes.len() {
        return Err(GnmModelError::Npy("truncated NPY header".to_owned()));
    }
    let header = std::str::from_utf8(&bytes[header_length_end..data_start])
        .map_err(|error| GnmModelError::Npy(error.to_string()))?;
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return Err(GnmModelError::Npy(
            "Fortran-order arrays are not supported".to_owned(),
        ));
    }
    let descr = header_value(header, "descr")?;
    let dtype = parse_dtype(descr)?;
    let shape = parse_shape(header)?;
    Ok(NpyArray {
        dtype,
        shape,
        data: bytes[data_start..].to_vec(),
    })
}

fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str, GnmModelError> {
    let single = format!("'{key}': '");
    let double = format!("\"{key}\": \"");
    if let Some(start) = header.find(&single) {
        let start = start + single.len();
        let end = header[start..]
            .find('\'')
            .map(|offset| start + offset)
            .ok_or_else(|| GnmModelError::Npy(format!("unterminated {key}")))?;
        return Ok(&header[start..end]);
    }
    if let Some(start) = header.find(&double) {
        let start = start + double.len();
        let end = header[start..]
            .find('"')
            .map(|offset| start + offset)
            .ok_or_else(|| GnmModelError::Npy(format!("unterminated {key}")))?;
        return Ok(&header[start..end]);
    }
    Err(GnmModelError::Npy(format!("missing {key}")))
}

fn parse_shape(header: &str) -> Result<Vec<usize>, GnmModelError> {
    let marker = if let Some(index) = header.find("'shape':") {
        index + "'shape':".len()
    } else if let Some(index) = header.find("\"shape\":") {
        index + "\"shape\":".len()
    } else {
        return Err(GnmModelError::Npy("missing shape".to_owned()));
    };
    let open = header[marker..]
        .find('(')
        .map(|offset| marker + offset)
        .ok_or_else(|| GnmModelError::Npy("missing shape tuple".to_owned()))?;
    let close = header[open..]
        .find(')')
        .map(|offset| open + offset)
        .ok_or_else(|| GnmModelError::Npy("unterminated shape tuple".to_owned()))?;
    let values = header[open + 1..close]
        .split(',')
        .filter_map(|part| {
            let part = part.trim();
            if part.is_empty() {
                None
            } else {
                Some(
                    part.parse::<usize>()
                        .map_err(|_| GnmModelError::Npy("invalid shape value".to_owned())),
                )
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(values)
}

fn parse_dtype(descr: &str) -> Result<NpyDType, GnmModelError> {
    let mut characters = descr.chars();
    let endian = characters.next();
    let kind = characters.next();
    let width = characters
        .as_str()
        .parse::<usize>()
        .map_err(|_| GnmModelError::Npy(format!("invalid dtype `{descr}`")))?;
    if !matches!(endian, Some('<') | Some('|')) {
        return Err(GnmModelError::Npy(format!(
            "only little-endian NPY dtypes are supported: `{descr}`"
        )));
    }
    match (kind, width) {
        (Some('i'), 1) => Ok(NpyDType::I8),
        (Some('i'), 2) => Ok(NpyDType::I16),
        (Some('f'), 4) => Ok(NpyDType::F32),
        (Some('f'), 8) => Ok(NpyDType::F64),
        (Some('i'), 4) => Ok(NpyDType::I32),
        (Some('i'), 8) => Ok(NpyDType::I64),
        (Some('u'), 1) => Ok(NpyDType::U8),
        (Some('u'), 2) => Ok(NpyDType::U16),
        (Some('u'), 4) => Ok(NpyDType::U32),
        (Some('u'), 8) => Ok(NpyDType::U64),
        (Some('S'), width) => Ok(NpyDType::Bytes(width)),
        (Some('U'), width) => Ok(NpyDType::Unicode(width)),
        _ => Err(GnmModelError::Npy(format!("unsupported dtype `{descr}`"))),
    }
}

impl NpyArray {
    fn item_count(&self, field: &str) -> Result<usize, GnmModelError> {
        self.shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| GnmModelError::Shape {
                    field: field.to_owned(),
                    expected: "finite item count".to_owned(),
                    actual: format!("{:?}", self.shape),
                })
        })
    }

    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn f32_values(&self, field: &str) -> Result<Vec<f32>, GnmModelError> {
        let count = self.item_count(field)?;
        let width = match self.dtype {
            NpyDType::I8 | NpyDType::U8 => 1,
            NpyDType::I16 | NpyDType::U16 => 2,
            NpyDType::F32 => 4,
            NpyDType::F64 => 8,
            NpyDType::I32 | NpyDType::U32 => 4,
            NpyDType::I64 | NpyDType::U64 => 8,
            _ => return Err(GnmModelError::Npy(format!("`{field}` is not numeric"))),
        };
        let expected_bytes = count
            .checked_mul(width)
            .ok_or_else(|| GnmModelError::Npy(format!("`{field}` payload size overflow")))?;
        if self.data.len() != expected_bytes {
            return Err(GnmModelError::Npy(format!(
                "`{field}` payload length mismatch"
            )));
        }
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            let offset = index
                .checked_mul(width)
                .ok_or_else(|| GnmModelError::Npy(format!("`{field}` payload offset overflow")))?;
            let value = match self.dtype {
                NpyDType::F32 => f32::from_le_bytes(read_chunk(&self.data, offset, field)?),
                NpyDType::F64 => f64::from_le_bytes(read_chunk(&self.data, offset, field)?) as f32,
                NpyDType::I8 => i8::from_le_bytes(read_chunk(&self.data, offset, field)?) as f32,
                NpyDType::I16 => i16::from_le_bytes(read_chunk(&self.data, offset, field)?) as f32,
                NpyDType::I32 => i32::from_le_bytes(read_chunk(&self.data, offset, field)?) as f32,
                NpyDType::U8 => self.data[offset] as f32,
                NpyDType::U16 => u16::from_le_bytes(read_chunk(&self.data, offset, field)?) as f32,
                NpyDType::U32 => u32::from_le_bytes(read_chunk(&self.data, offset, field)?) as f32,
                NpyDType::I64 => i64::from_le_bytes(read_chunk(&self.data, offset, field)?) as f32,
                NpyDType::U64 => u64::from_le_bytes(read_chunk(&self.data, offset, field)?) as f32,
                _ => unreachable!("numeric dtype checked above"),
            };
            if !value.is_finite() {
                return Err(GnmModelError::NonFinite {
                    field: field.to_owned(),
                    index,
                });
            }
            values.push(value);
        }
        Ok(values)
    }

    // Bounds are guaranteed by construction in this numeric kernel
    // (loop ranges bounded by buffer lengths / fixed-size dimensions);
    // see the AGENTS.md production panic policy.
    #[allow(clippy::indexing_slicing)]
    fn i32_values(&self, field: &str, expected_shape: &[usize]) -> Result<Vec<i32>, GnmModelError> {
        if self.shape != expected_shape {
            return Err(GnmModelError::Shape {
                field: field.to_owned(),
                expected: format!("{expected_shape:?}"),
                actual: format!("{:?}", self.shape),
            });
        }
        let count = self.item_count(field)?;
        let width = match self.dtype {
            NpyDType::I8 | NpyDType::U8 => 1,
            NpyDType::I16 | NpyDType::U16 => 2,
            NpyDType::I32 | NpyDType::U32 => 4,
            NpyDType::I64 | NpyDType::U64 => 8,
            _ => {
                return Err(GnmModelError::Npy(format!(
                    "`{field}` is not an integer array"
                )));
            }
        };
        let expected_bytes = count
            .checked_mul(width)
            .ok_or_else(|| GnmModelError::Npy(format!("`{field}` payload size overflow")))?;
        if self.data.len() != expected_bytes {
            return Err(GnmModelError::Npy(format!(
                "`{field}` payload length mismatch"
            )));
        }
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            let offset = index
                .checked_mul(width)
                .ok_or_else(|| GnmModelError::Npy(format!("`{field}` payload offset overflow")))?;
            let value = match self.dtype {
                NpyDType::I8 => i8::from_le_bytes(read_chunk(&self.data, offset, field)?) as i32,
                NpyDType::I16 => i16::from_le_bytes(read_chunk(&self.data, offset, field)?) as i32,
                NpyDType::I32 => i32::from_le_bytes(read_chunk(&self.data, offset, field)?),
                NpyDType::U8 => self.data[offset] as i32,
                NpyDType::U16 => u16::from_le_bytes(read_chunk(&self.data, offset, field)?) as i32,
                NpyDType::U32 => {
                    i32::try_from(u32::from_le_bytes(read_chunk(&self.data, offset, field)?))
                        .map_err(|_| GnmModelError::InvalidValue {
                            field: field.to_owned(),
                            reason: "value exceeds i32".to_owned(),
                        })?
                }
                NpyDType::I64 => {
                    i32::try_from(i64::from_le_bytes(read_chunk(&self.data, offset, field)?))
                        .map_err(|_| GnmModelError::InvalidValue {
                            field: field.to_owned(),
                            reason: "value exceeds i32".to_owned(),
                        })?
                }
                NpyDType::U64 => {
                    i32::try_from(u64::from_le_bytes(read_chunk(&self.data, offset, field)?))
                        .map_err(|_| GnmModelError::InvalidValue {
                            field: field.to_owned(),
                            reason: "value exceeds i32".to_owned(),
                        })?
                }
                _ => unreachable!("integer dtype checked above"),
            };
            values.push(value);
        }
        Ok(values)
    }

    fn single_string(&self, field: &str) -> Result<String, GnmModelError> {
        let strings = self.strings(field)?;
        if strings.len() != 1 {
            return Err(GnmModelError::Shape {
                field: field.to_owned(),
                expected: "one string".to_owned(),
                actual: format!("{} strings", strings.len()),
            });
        }
        Ok(strings.into_iter().next().unwrap_or_default())
    }

    fn strings(&self, field: &str) -> Result<Vec<String>, GnmModelError> {
        let count = self.item_count(field)?;
        let width = match self.dtype {
            NpyDType::Bytes(width) => width,
            NpyDType::Unicode(width) => width
                .checked_mul(4)
                .ok_or_else(|| GnmModelError::Npy(format!("`{field}` string width overflow")))?,
            _ => {
                return Err(GnmModelError::Npy(format!(
                    "`{field}` is not a string array"
                )));
            }
        };
        let expected_bytes = count
            .checked_mul(width)
            .ok_or_else(|| GnmModelError::Npy(format!("`{field}` payload size overflow")))?;
        if self.data.len() != expected_bytes {
            return Err(GnmModelError::Npy(format!(
                "`{field}` payload length mismatch"
            )));
        }
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            let start = index
                .checked_mul(width)
                .ok_or_else(|| GnmModelError::Npy(format!("`{field}` payload offset overflow")))?;
            let end = start
                .checked_add(width)
                .ok_or_else(|| GnmModelError::Npy(format!("`{field}` payload range overflow")))?;
            let bytes = self.data.get(start..end).ok_or_else(|| {
                GnmModelError::Npy(format!("`{field}` payload ended unexpectedly"))
            })?;
            let value = match self.dtype {
                NpyDType::Bytes(_) => String::from_utf8(
                    bytes
                        .iter()
                        .copied()
                        .take_while(|byte| *byte != 0)
                        .collect(),
                )
                .map_err(|error| GnmModelError::Npy(error.to_string()))?,
                NpyDType::Unicode(_) => {
                    let mut value = String::new();
                    for code in bytes.chunks_exact(4) {
                        let code = u32::from_le_bytes(code.try_into().map_err(|_| {
                            GnmModelError::Npy(format!("invalid Unicode payload in `{field}`"))
                        })?);
                        if code == 0 {
                            break;
                        }
                        value.push(char::from_u32(code).ok_or_else(|| {
                            GnmModelError::Npy(format!("invalid Unicode in `{field}`"))
                        })?);
                    }
                    value
                }
                _ => unreachable!("string dtype checked above"),
            };
            values.push(value);
        }
        Ok(values)
    }
}

fn read_chunk<const N: usize>(
    data: &[u8],
    offset: usize,
    field: &str,
) -> Result<[u8; N], GnmModelError> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| GnmModelError::Npy(format!("`{field}` payload range overflow")))?;
    data.get(offset..end)
        .and_then(|chunk| chunk.try_into().ok())
        .ok_or_else(|| GnmModelError::Npy(format!("`{field}` payload ended unexpectedly")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(descr: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}")
                .into_bytes();
        let preamble = 10usize;
        let padding = (16 - (preamble + header.len() + 1) % 16) % 16;
        header.extend(std::iter::repeat_n(b' ', padding));
        header.push(b'\n');
        let mut result = b"\x93NUMPY\x01\x00".to_vec();
        result.extend((header.len() as u16).to_le_bytes());
        result.extend(header);
        result.extend(payload);
        result
    }

    #[test]
    fn parses_little_endian_f32_and_rejects_nonfinite_values_at_validation() {
        let mut payload = Vec::new();
        payload.extend(1.5f32.to_le_bytes());
        payload.extend(2.0f32.to_le_bytes());
        let array = parse_npy(&npy("<f4", "(2,)", &payload)).unwrap();
        assert_eq!(array.shape, [2]);
        assert_eq!(array.f32_values("test").unwrap(), [1.5, 2.0]);

        let nonfinite = npy("<f4", "(1,)", &f32::NAN.to_le_bytes());
        let array = parse_npy(&nonfinite).unwrap();
        assert!(
            matches!(dense(&array, "test"), Err(GnmModelError::NonFinite { field, .. }) if field == "test")
        );
    }

    #[test]
    fn malformed_npy_headers_dtypes_and_payloads_are_typed_errors() {
        for major in [2u8, 3u8] {
            let bytes = [b'\x93', b'N', b'U', b'M', b'P', b'Y', major, 0];
            assert!(matches!(
                parse_npy(&bytes),
                Err(GnmModelError::Npy(message)) if message.contains("header length")
            ));
        }

        let wrong_dtype = npy(">f4", "(1,)", &0.0f32.to_le_bytes());
        assert!(matches!(
            parse_npy(&wrong_dtype),
            Err(GnmModelError::Npy(message)) if message.contains("little-endian")
        ));

        let truncated_payload = npy("<f4", "(2,)", &0.0f32.to_le_bytes());
        let array = parse_npy(&truncated_payload).unwrap();
        assert!(matches!(
            array.f32_values("truncated"),
            Err(GnmModelError::Npy(message)) if message.contains("payload length")
        ));
    }

    #[test]
    fn empty_archive_reports_the_first_missing_schema_array() {
        let path =
            std::env::temp_dir().join(format!("vtuber-gnm-empty-{}.npz", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        let writer = zip::ZipWriter::new(file);
        writer.finish().unwrap();
        let error = load_gnm_head_v3(&path).unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert_eq!(error, GnmModelError::MissingArray("version".to_owned()));
    }
}
