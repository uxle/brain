//! # Low-Level Binary & Wire Utilities
//!
//! Little-endian numeric decoding, Varint/LEB128 parsing, CRC32 checksums, and binary stream helpers.
#![allow(missing_docs)]

use super::core::{OnnxError, OnnxResult};

/// Decodes unsigned variable-length integer (Varint / LEB128).
pub fn decode_varint(bytes: &[u8], mut offset: usize) -> OnnxResult<(u64, usize)> {
    let mut result = 0u64;
    let mut shift = 0;

    while offset < bytes.len() {
        let byte = bytes[offset];
        offset += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            return Ok((result, offset));
        }
        shift += 7;
        if shift >= 64 {
            return Err(OnnxError::ProtobufDecodeError("Varint overflow".into()));
        }
    }

    Err(OnnxError::ProtobufDecodeError("Unexpected EOF while decoding varint".into()))
}

/// Encodes a u64 into unsigned LEB128 / Varint bytes.
pub fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    while value >= 0x80 {
        buf.push(((value & 0x7F) as u8) | 0x80);
        value >>= 7;
    }
    buf.push((value & 0x7F) as u8);
    buf
}

/// Reads little-endian f32 from slice.
pub fn read_f32_le(slice: &[u8]) -> f32 {
    f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]])
}

/// Reads little-endian f64 from slice.
pub fn read_f64_le(slice: &[u8]) -> f64 {
    f64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3],
        slice[4], slice[5], slice[6], slice[7],
    ])
}

/// Computes CRC32 checksum over arbitrary byte slices.
pub fn compute_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
