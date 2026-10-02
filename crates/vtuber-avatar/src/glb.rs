//! GLB 2 container handling shared by import and VRM JSON conversion.

use serde_json::Value;

const JSON: u32 = 0x4e4f_534a;
const BIN: u32 = 0x004e_4942;

/// Container or JSON failure, including the GLB uint32 size limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GlbError {
    /// Invalid container layout.
    Container(&'static str),
    /// Invalid JSON content.
    Json(String),
    /// The encoded container exceeds the GLB size limit.
    TooLarge,
}

impl std::fmt::Display for GlbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Container(reason) => write!(f, "invalid GLB: {reason}"),
            Self::Json(reason) => write!(f, "invalid GLB JSON: {reason}"),
            Self::TooLarge => write!(f, "GLB exceeds the uint32 size limit"),
        }
    }
}

impl std::error::Error for GlbError {}

/// A mutable JSON document with borrowed binary and extension chunks.
pub struct Glb<'a> {
    /// JSON content, independently edited by the VRM conversion caller.
    pub document: Value,
    /// The optional second, binary chunk, including its authored padding.
    pub bin: Option<&'a [u8]>,
    extra_chunks: Vec<(u32, &'a [u8])>,
}

impl<'a> Glb<'a> {
    /// Creates a container from a JSON document and an optional binary payload.
    pub fn new(document: Value, bin: Option<&'a [u8]>) -> Self {
        Self {
            document,
            bin,
            extra_chunks: Vec::new(),
        }
    }

    /// Parses the GLB 2 header and ordered, aligned chunks.
    /// Unknown trailing chunks are retained without interpreting their payload.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, GlbError> {
        let mut rest = bytes;
        if word(&mut rest)? != 0x4654_6c67 {
            return Err(GlbError::Container("missing glTF magic"));
        }
        if word(&mut rest)? != 2 {
            return Err(GlbError::Container("unsupported version"));
        }
        if usize::try_from(word(&mut rest)?).ok() != Some(bytes.len()) {
            return Err(GlbError::Container("header length does not match input"));
        }
        let (kind, json) = chunk(&mut rest)?;
        if kind != JSON {
            return Err(GlbError::Container("first chunk must be JSON"));
        }
        let document =
            serde_json::from_slice(json).map_err(|error| GlbError::Json(error.to_string()))?;
        let mut result = Self::new(document, None);
        let mut index = 1;
        while !rest.is_empty() {
            let (kind, payload) = chunk(&mut rest)?;
            match kind {
                JSON => return Err(GlbError::Container("duplicate JSON chunk")),
                BIN if index == 1 => result.bin = Some(payload),
                BIN => return Err(GlbError::Container("BIN must be the second chunk")),
                _ => result.extra_chunks.push((kind, payload)),
            }
            index += 1;
        }
        Ok(result)
    }

    /// Encodes the edited JSON and preserves binary and unknown chunk data.
    pub fn to_vec(&self) -> Result<Vec<u8>, GlbError> {
        let json = serde_json::to_vec(&self.document)
            .map_err(|error| GlbError::Json(error.to_string()))?;
        let mut output = Vec::new();
        output.extend_from_slice(b"glTF");
        output.extend_from_slice(&2_u32.to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        append_chunk(&mut output, JSON, &json, b' ')?;
        if let Some(bin) = self.bin {
            append_chunk(&mut output, BIN, bin, 0)?;
        }
        for &(kind, payload) in &self.extra_chunks {
            append_chunk(&mut output, kind, payload, 0)?;
        }
        let length = u32::try_from(output.len()).map_err(|_| GlbError::TooLarge)?;
        output
            .get_mut(8..12)
            .ok_or(GlbError::Container("missing output header"))?
            .copy_from_slice(&length.to_le_bytes());
        Ok(output)
    }
}

fn word(rest: &mut &[u8]) -> Result<u32, GlbError> {
    let (bytes, tail) = rest
        .split_first_chunk::<4>()
        .ok_or(GlbError::Container("truncated header"))?;
    *rest = tail;
    Ok(u32::from_le_bytes(*bytes))
}

fn chunk<'a>(rest: &mut &'a [u8]) -> Result<(u32, &'a [u8]), GlbError> {
    let length = usize::try_from(word(rest)?).map_err(|_| GlbError::TooLarge)?;
    let kind = word(rest)?;
    if length % 4 != 0 {
        return Err(GlbError::Container("unaligned chunk"));
    }
    let (payload, tail) = rest
        .split_at_checked(length)
        .ok_or(GlbError::Container("truncated chunk"))?;
    *rest = tail;
    Ok((kind, payload))
}

fn append_chunk(output: &mut Vec<u8>, kind: u32, payload: &[u8], pad: u8) -> Result<(), GlbError> {
    let padding = (4 - payload.len() % 4) % 4;
    let length = payload
        .len()
        .checked_add(padding)
        .and_then(|length| u32::try_from(length).ok())
        .ok_or(GlbError::TooLarge)?;
    output
        .len()
        .checked_add(8)
        .and_then(|total| total.checked_add(length as usize))
        .and_then(|total| u32::try_from(total).ok())
        .ok_or(GlbError::TooLarge)?;
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(&kind.to_le_bytes());
    output.extend_from_slice(payload);
    output.extend(std::iter::repeat_n(pad, padding));
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn padding_and_unknown_chunks_survive_json_edits() {
        let mut glb = Glb::new(serde_json::json!({"a": 1}), Some(&[1, 2, 3]));
        glb.extra_chunks.push((0x1234, &[4, 5, 6, 7]));
        let bytes = glb.to_vec().unwrap();
        let mut parsed = Glb::parse(&bytes).unwrap();
        assert_eq!(parsed.bin, Some([1, 2, 3, 0].as_slice()));
        parsed.document["a"] = Value::from(2);
        let edited = parsed.to_vec().unwrap();
        let parsed = Glb::parse(&edited).unwrap();
        assert_eq!(parsed.document["a"], 2);
        assert_eq!(parsed.extra_chunks, vec![(0x1234, [4, 5, 6, 7].as_slice())]);
        assert_eq!(parsed.bin, Some([1, 2, 3, 0].as_slice()));
    }

    #[test]
    fn rejects_invalid_headers_lengths_alignment_and_chunk_order() {
        let bytes = Glb::new(serde_json::json!({}), Some(&[1, 2, 3, 4]))
            .to_vec()
            .unwrap();
        for length in 0..bytes.len() {
            assert!(Glb::parse(&bytes[..length]).is_err());
        }
        for (offset, value) in [(4, 1_u32), (8, 0), (12, 3), (16, BIN)] {
            let mut invalid = bytes.clone();
            invalid[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(Glb::parse(&invalid).is_err());
        }
        let mut invalid = Glb::new(serde_json::json!({}), None);
        invalid.extra_chunks.push((0x1234, &[0; 4]));
        invalid.extra_chunks.push((BIN, &[1; 4]));
        assert!(Glb::parse(&invalid.to_vec().unwrap()).is_err());
    }
}
