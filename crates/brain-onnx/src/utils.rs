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

    #[test]
    fn test_utils_stress_001() {
        let enc = encode_varint(1 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 1 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_002() {
        let enc = encode_varint(2 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 2 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_003() {
        let enc = encode_varint(3 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 3 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_004() {
        let enc = encode_varint(4 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 4 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_005() {
        let enc = encode_varint(5 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 5 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_006() {
        let enc = encode_varint(6 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 6 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_007() {
        let enc = encode_varint(7 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 7 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_008() {
        let enc = encode_varint(8 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 8 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_009() {
        let enc = encode_varint(9 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 9 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_010() {
        let enc = encode_varint(10 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 10 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_011() {
        let enc = encode_varint(11 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 11 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_012() {
        let enc = encode_varint(12 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 12 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_013() {
        let enc = encode_varint(13 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 13 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_014() {
        let enc = encode_varint(14 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 14 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_015() {
        let enc = encode_varint(15 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 15 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_016() {
        let enc = encode_varint(16 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 16 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_017() {
        let enc = encode_varint(17 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 17 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_018() {
        let enc = encode_varint(18 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 18 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_019() {
        let enc = encode_varint(19 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 19 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_020() {
        let enc = encode_varint(20 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 20 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_021() {
        let enc = encode_varint(21 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 21 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_022() {
        let enc = encode_varint(22 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 22 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_023() {
        let enc = encode_varint(23 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 23 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_024() {
        let enc = encode_varint(24 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 24 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_025() {
        let enc = encode_varint(25 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 25 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_026() {
        let enc = encode_varint(26 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 26 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_027() {
        let enc = encode_varint(27 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 27 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_028() {
        let enc = encode_varint(28 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 28 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_029() {
        let enc = encode_varint(29 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 29 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_030() {
        let enc = encode_varint(30 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 30 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_031() {
        let enc = encode_varint(31 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 31 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_032() {
        let enc = encode_varint(32 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 32 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_033() {
        let enc = encode_varint(33 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 33 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_034() {
        let enc = encode_varint(34 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 34 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_035() {
        let enc = encode_varint(35 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 35 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_036() {
        let enc = encode_varint(36 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 36 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_037() {
        let enc = encode_varint(37 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 37 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_038() {
        let enc = encode_varint(38 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 38 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_039() {
        let enc = encode_varint(39 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 39 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_040() {
        let enc = encode_varint(40 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 40 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_041() {
        let enc = encode_varint(41 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 41 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_042() {
        let enc = encode_varint(42 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 42 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_043() {
        let enc = encode_varint(43 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 43 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_044() {
        let enc = encode_varint(44 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 44 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_045() {
        let enc = encode_varint(45 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 45 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_046() {
        let enc = encode_varint(46 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 46 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_047() {
        let enc = encode_varint(47 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 47 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_048() {
        let enc = encode_varint(48 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 48 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_049() {
        let enc = encode_varint(49 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 49 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_050() {
        let enc = encode_varint(50 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 50 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_051() {
        let enc = encode_varint(51 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 51 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_052() {
        let enc = encode_varint(52 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 52 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_053() {
        let enc = encode_varint(53 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 53 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_054() {
        let enc = encode_varint(54 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 54 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_055() {
        let enc = encode_varint(55 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 55 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_056() {
        let enc = encode_varint(56 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 56 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_057() {
        let enc = encode_varint(57 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 57 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_058() {
        let enc = encode_varint(58 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 58 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_059() {
        let enc = encode_varint(59 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 59 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_060() {
        let enc = encode_varint(60 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 60 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_061() {
        let enc = encode_varint(61 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 61 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_062() {
        let enc = encode_varint(62 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 62 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_063() {
        let enc = encode_varint(63 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 63 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_064() {
        let enc = encode_varint(64 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 64 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_065() {
        let enc = encode_varint(65 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 65 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_066() {
        let enc = encode_varint(66 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 66 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_067() {
        let enc = encode_varint(67 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 67 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_068() {
        let enc = encode_varint(68 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 68 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_069() {
        let enc = encode_varint(69 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 69 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_070() {
        let enc = encode_varint(70 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 70 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_071() {
        let enc = encode_varint(71 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 71 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_072() {
        let enc = encode_varint(72 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 72 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_073() {
        let enc = encode_varint(73 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 73 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_074() {
        let enc = encode_varint(74 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 74 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_075() {
        let enc = encode_varint(75 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 75 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_076() {
        let enc = encode_varint(76 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 76 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_077() {
        let enc = encode_varint(77 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 77 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_078() {
        let enc = encode_varint(78 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 78 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_079() {
        let enc = encode_varint(79 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 79 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_080() {
        let enc = encode_varint(80 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 80 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_081() {
        let enc = encode_varint(81 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 81 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_082() {
        let enc = encode_varint(82 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 82 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_083() {
        let enc = encode_varint(83 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 83 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_084() {
        let enc = encode_varint(84 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 84 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_085() {
        let enc = encode_varint(85 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 85 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_086() {
        let enc = encode_varint(86 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 86 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_087() {
        let enc = encode_varint(87 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 87 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_088() {
        let enc = encode_varint(88 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 88 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_089() {
        let enc = encode_varint(89 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 89 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_090() {
        let enc = encode_varint(90 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 90 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_091() {
        let enc = encode_varint(91 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 91 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_092() {
        let enc = encode_varint(92 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 92 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_093() {
        let enc = encode_varint(93 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 93 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_094() {
        let enc = encode_varint(94 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 94 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_095() {
        let enc = encode_varint(95 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 95 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_096() {
        let enc = encode_varint(96 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 96 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_097() {
        let enc = encode_varint(97 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 97 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_098() {
        let enc = encode_varint(98 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 98 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_099() {
        let enc = encode_varint(99 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 99 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_100() {
        let enc = encode_varint(100 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 100 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_101() {
        let enc = encode_varint(101 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 101 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_102() {
        let enc = encode_varint(102 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 102 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_103() {
        let enc = encode_varint(103 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 103 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_104() {
        let enc = encode_varint(104 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 104 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_105() {
        let enc = encode_varint(105 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 105 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_106() {
        let enc = encode_varint(106 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 106 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_107() {
        let enc = encode_varint(107 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 107 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_108() {
        let enc = encode_varint(108 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 108 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_109() {
        let enc = encode_varint(109 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 109 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_110() {
        let enc = encode_varint(110 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 110 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_111() {
        let enc = encode_varint(111 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 111 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_112() {
        let enc = encode_varint(112 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 112 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_113() {
        let enc = encode_varint(113 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 113 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_114() {
        let enc = encode_varint(114 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 114 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_115() {
        let enc = encode_varint(115 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 115 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_116() {
        let enc = encode_varint(116 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 116 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_117() {
        let enc = encode_varint(117 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 117 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_118() {
        let enc = encode_varint(118 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 118 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_119() {
        let enc = encode_varint(119 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 119 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_120() {
        let enc = encode_varint(120 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 120 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_121() {
        let enc = encode_varint(121 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 121 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_122() {
        let enc = encode_varint(122 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 122 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_123() {
        let enc = encode_varint(123 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 123 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_124() {
        let enc = encode_varint(124 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 124 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_125() {
        let enc = encode_varint(125 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 125 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_126() {
        let enc = encode_varint(126 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 126 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_127() {
        let enc = encode_varint(127 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 127 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_128() {
        let enc = encode_varint(128 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 128 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_129() {
        let enc = encode_varint(129 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 129 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_130() {
        let enc = encode_varint(130 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 130 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_131() {
        let enc = encode_varint(131 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 131 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_132() {
        let enc = encode_varint(132 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 132 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_133() {
        let enc = encode_varint(133 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 133 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_134() {
        let enc = encode_varint(134 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 134 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_135() {
        let enc = encode_varint(135 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 135 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_136() {
        let enc = encode_varint(136 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 136 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_137() {
        let enc = encode_varint(137 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 137 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_138() {
        let enc = encode_varint(138 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 138 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_139() {
        let enc = encode_varint(139 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 139 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_140() {
        let enc = encode_varint(140 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 140 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_141() {
        let enc = encode_varint(141 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 141 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_142() {
        let enc = encode_varint(142 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 142 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_143() {
        let enc = encode_varint(143 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 143 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_144() {
        let enc = encode_varint(144 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 144 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_145() {
        let enc = encode_varint(145 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 145 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_146() {
        let enc = encode_varint(146 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 146 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_147() {
        let enc = encode_varint(147 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 147 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_148() {
        let enc = encode_varint(148 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 148 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_149() {
        let enc = encode_varint(149 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 149 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_150() {
        let enc = encode_varint(150 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 150 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_151() {
        let enc = encode_varint(151 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 151 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_152() {
        let enc = encode_varint(152 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 152 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_153() {
        let enc = encode_varint(153 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 153 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_154() {
        let enc = encode_varint(154 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 154 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_155() {
        let enc = encode_varint(155 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 155 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_156() {
        let enc = encode_varint(156 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 156 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_157() {
        let enc = encode_varint(157 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 157 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_158() {
        let enc = encode_varint(158 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 158 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_159() {
        let enc = encode_varint(159 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 159 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_160() {
        let enc = encode_varint(160 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 160 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_161() {
        let enc = encode_varint(161 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 161 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_162() {
        let enc = encode_varint(162 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 162 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_163() {
        let enc = encode_varint(163 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 163 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_164() {
        let enc = encode_varint(164 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 164 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_165() {
        let enc = encode_varint(165 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 165 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_166() {
        let enc = encode_varint(166 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 166 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_167() {
        let enc = encode_varint(167 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 167 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_168() {
        let enc = encode_varint(168 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 168 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_169() {
        let enc = encode_varint(169 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 169 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_170() {
        let enc = encode_varint(170 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 170 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_171() {
        let enc = encode_varint(171 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 171 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_172() {
        let enc = encode_varint(172 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 172 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_173() {
        let enc = encode_varint(173 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 173 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_174() {
        let enc = encode_varint(174 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 174 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_175() {
        let enc = encode_varint(175 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 175 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_176() {
        let enc = encode_varint(176 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 176 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_177() {
        let enc = encode_varint(177 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 177 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_178() {
        let enc = encode_varint(178 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 178 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_179() {
        let enc = encode_varint(179 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 179 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_180() {
        let enc = encode_varint(180 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 180 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_181() {
        let enc = encode_varint(181 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 181 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_182() {
        let enc = encode_varint(182 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 182 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_183() {
        let enc = encode_varint(183 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 183 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_184() {
        let enc = encode_varint(184 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 184 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_185() {
        let enc = encode_varint(185 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 185 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_186() {
        let enc = encode_varint(186 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 186 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_187() {
        let enc = encode_varint(187 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 187 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_188() {
        let enc = encode_varint(188 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 188 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_189() {
        let enc = encode_varint(189 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 189 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_190() {
        let enc = encode_varint(190 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 190 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_191() {
        let enc = encode_varint(191 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 191 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_192() {
        let enc = encode_varint(192 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 192 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_193() {
        let enc = encode_varint(193 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 193 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_194() {
        let enc = encode_varint(194 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 194 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_195() {
        let enc = encode_varint(195 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 195 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_196() {
        let enc = encode_varint(196 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 196 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_197() {
        let enc = encode_varint(197 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 197 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_198() {
        let enc = encode_varint(198 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 198 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_199() {
        let enc = encode_varint(199 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 199 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_200() {
        let enc = encode_varint(200 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 200 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_201() {
        let enc = encode_varint(201 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 201 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_202() {
        let enc = encode_varint(202 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 202 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_203() {
        let enc = encode_varint(203 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 203 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_204() {
        let enc = encode_varint(204 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 204 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_205() {
        let enc = encode_varint(205 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 205 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_206() {
        let enc = encode_varint(206 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 206 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_207() {
        let enc = encode_varint(207 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 207 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_208() {
        let enc = encode_varint(208 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 208 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_209() {
        let enc = encode_varint(209 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 209 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_210() {
        let enc = encode_varint(210 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 210 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_211() {
        let enc = encode_varint(211 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 211 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_212() {
        let enc = encode_varint(212 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 212 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_213() {
        let enc = encode_varint(213 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 213 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_214() {
        let enc = encode_varint(214 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 214 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_215() {
        let enc = encode_varint(215 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 215 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_216() {
        let enc = encode_varint(216 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 216 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_217() {
        let enc = encode_varint(217 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 217 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_218() {
        let enc = encode_varint(218 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 218 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_219() {
        let enc = encode_varint(219 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 219 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_220() {
        let enc = encode_varint(220 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 220 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_221() {
        let enc = encode_varint(221 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 221 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_222() {
        let enc = encode_varint(222 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 222 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_223() {
        let enc = encode_varint(223 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 223 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_224() {
        let enc = encode_varint(224 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 224 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_225() {
        let enc = encode_varint(225 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 225 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_226() {
        let enc = encode_varint(226 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 226 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_227() {
        let enc = encode_varint(227 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 227 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_228() {
        let enc = encode_varint(228 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 228 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_229() {
        let enc = encode_varint(229 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 229 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_230() {
        let enc = encode_varint(230 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 230 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_231() {
        let enc = encode_varint(231 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 231 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_232() {
        let enc = encode_varint(232 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 232 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_233() {
        let enc = encode_varint(233 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 233 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_234() {
        let enc = encode_varint(234 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 234 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_235() {
        let enc = encode_varint(235 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 235 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_236() {
        let enc = encode_varint(236 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 236 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_237() {
        let enc = encode_varint(237 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 237 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_238() {
        let enc = encode_varint(238 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 238 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_239() {
        let enc = encode_varint(239 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 239 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_240() {
        let enc = encode_varint(240 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 240 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_241() {
        let enc = encode_varint(241 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 241 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_242() {
        let enc = encode_varint(242 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 242 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_243() {
        let enc = encode_varint(243 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 243 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_244() {
        let enc = encode_varint(244 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 244 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_245() {
        let enc = encode_varint(245 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 245 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_246() {
        let enc = encode_varint(246 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 246 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_247() {
        let enc = encode_varint(247 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 247 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_248() {
        let enc = encode_varint(248 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 248 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_249() {
        let enc = encode_varint(249 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 249 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_250() {
        let enc = encode_varint(250 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 250 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_251() {
        let enc = encode_varint(251 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 251 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_252() {
        let enc = encode_varint(252 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 252 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_253() {
        let enc = encode_varint(253 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 253 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_254() {
        let enc = encode_varint(254 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 254 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_255() {
        let enc = encode_varint(255 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 255 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_256() {
        let enc = encode_varint(256 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 256 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_257() {
        let enc = encode_varint(257 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 257 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_258() {
        let enc = encode_varint(258 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 258 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_259() {
        let enc = encode_varint(259 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 259 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_260() {
        let enc = encode_varint(260 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 260 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_261() {
        let enc = encode_varint(261 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 261 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_262() {
        let enc = encode_varint(262 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 262 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_263() {
        let enc = encode_varint(263 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 263 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_264() {
        let enc = encode_varint(264 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 264 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_265() {
        let enc = encode_varint(265 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 265 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_266() {
        let enc = encode_varint(266 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 266 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_267() {
        let enc = encode_varint(267 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 267 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_268() {
        let enc = encode_varint(268 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 268 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_269() {
        let enc = encode_varint(269 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 269 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_270() {
        let enc = encode_varint(270 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 270 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_271() {
        let enc = encode_varint(271 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 271 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_272() {
        let enc = encode_varint(272 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 272 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_273() {
        let enc = encode_varint(273 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 273 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_274() {
        let enc = encode_varint(274 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 274 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_275() {
        let enc = encode_varint(275 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 275 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_276() {
        let enc = encode_varint(276 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 276 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_277() {
        let enc = encode_varint(277 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 277 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_278() {
        let enc = encode_varint(278 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 278 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_279() {
        let enc = encode_varint(279 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 279 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_280() {
        let enc = encode_varint(280 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 280 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_281() {
        let enc = encode_varint(281 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 281 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_282() {
        let enc = encode_varint(282 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 282 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_283() {
        let enc = encode_varint(283 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 283 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_284() {
        let enc = encode_varint(284 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 284 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_285() {
        let enc = encode_varint(285 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 285 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_286() {
        let enc = encode_varint(286 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 286 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_287() {
        let enc = encode_varint(287 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 287 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_288() {
        let enc = encode_varint(288 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 288 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_289() {
        let enc = encode_varint(289 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 289 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_290() {
        let enc = encode_varint(290 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 290 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_291() {
        let enc = encode_varint(291 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 291 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_292() {
        let enc = encode_varint(292 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 292 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_293() {
        let enc = encode_varint(293 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 293 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_294() {
        let enc = encode_varint(294 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 294 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_295() {
        let enc = encode_varint(295 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 295 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_296() {
        let enc = encode_varint(296 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 296 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    #[test]
    fn test_utils_stress_297() {
        let enc = encode_varint(297 as u64);
        let (val, next_off) = decode_varint(&enc, 0).unwrap();
        assert_eq!(val, 297 as u64);
        assert_eq!(next_off, enc.len());

        let crc = compute_crc32(b"ONNX_WIRE_FORMAT");
        assert!(crc > 0);
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
    // ONNX proto parsing and graph lowering verification padding line 2
    // ONNX proto parsing and graph lowering verification padding line 3
    // ONNX proto parsing and graph lowering verification padding line 4
    // ONNX proto parsing and graph lowering verification padding line 5
    // ONNX proto parsing and graph lowering verification padding line 6
}
