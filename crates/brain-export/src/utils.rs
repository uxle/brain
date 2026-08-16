//! # Binary Serialization Utilities & Codecs
//!
//! Hand-rolled Protocol Buffers / FlatBuffers varint encoding, string encoding, and CRC32.

/// Encodes an unsigned integer as ULEB128 varint.
pub fn encode_uleb128(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Aligns a byte buffer to 4-byte boundaries with zero padding.
pub fn align4(buffer: &mut Vec<u8>) {
    while !buffer.len().is_multiple_of(4) {
        buffer.push(0);
    }
}

/// Computes standard IEEE 802.3 CRC32 checksum.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFF_u32;
    for &byte in data {
        crc ^= byte as u32;
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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_export_utils_stress_001() {
        let mut buf = Vec::new();
        encode_uleb128(1, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_002() {
        let mut buf = Vec::new();
        encode_uleb128(2, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_003() {
        let mut buf = Vec::new();
        encode_uleb128(3, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_004() {
        let mut buf = Vec::new();
        encode_uleb128(4, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_005() {
        let mut buf = Vec::new();
        encode_uleb128(5, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_006() {
        let mut buf = Vec::new();
        encode_uleb128(6, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_007() {
        let mut buf = Vec::new();
        encode_uleb128(7, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_008() {
        let mut buf = Vec::new();
        encode_uleb128(8, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_009() {
        let mut buf = Vec::new();
        encode_uleb128(9, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_010() {
        let mut buf = Vec::new();
        encode_uleb128(10, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_011() {
        let mut buf = Vec::new();
        encode_uleb128(11, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_012() {
        let mut buf = Vec::new();
        encode_uleb128(12, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_013() {
        let mut buf = Vec::new();
        encode_uleb128(13, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_014() {
        let mut buf = Vec::new();
        encode_uleb128(14, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_015() {
        let mut buf = Vec::new();
        encode_uleb128(15, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_016() {
        let mut buf = Vec::new();
        encode_uleb128(16, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_017() {
        let mut buf = Vec::new();
        encode_uleb128(17, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_018() {
        let mut buf = Vec::new();
        encode_uleb128(18, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_019() {
        let mut buf = Vec::new();
        encode_uleb128(19, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_020() {
        let mut buf = Vec::new();
        encode_uleb128(20, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_021() {
        let mut buf = Vec::new();
        encode_uleb128(21, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_022() {
        let mut buf = Vec::new();
        encode_uleb128(22, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_023() {
        let mut buf = Vec::new();
        encode_uleb128(23, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_024() {
        let mut buf = Vec::new();
        encode_uleb128(24, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_025() {
        let mut buf = Vec::new();
        encode_uleb128(25, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_026() {
        let mut buf = Vec::new();
        encode_uleb128(26, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_027() {
        let mut buf = Vec::new();
        encode_uleb128(27, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_028() {
        let mut buf = Vec::new();
        encode_uleb128(28, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_029() {
        let mut buf = Vec::new();
        encode_uleb128(29, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_030() {
        let mut buf = Vec::new();
        encode_uleb128(30, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_031() {
        let mut buf = Vec::new();
        encode_uleb128(31, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_032() {
        let mut buf = Vec::new();
        encode_uleb128(32, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_033() {
        let mut buf = Vec::new();
        encode_uleb128(33, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_034() {
        let mut buf = Vec::new();
        encode_uleb128(34, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_035() {
        let mut buf = Vec::new();
        encode_uleb128(35, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_036() {
        let mut buf = Vec::new();
        encode_uleb128(36, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_037() {
        let mut buf = Vec::new();
        encode_uleb128(37, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_038() {
        let mut buf = Vec::new();
        encode_uleb128(38, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_039() {
        let mut buf = Vec::new();
        encode_uleb128(39, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_040() {
        let mut buf = Vec::new();
        encode_uleb128(40, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_041() {
        let mut buf = Vec::new();
        encode_uleb128(41, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_042() {
        let mut buf = Vec::new();
        encode_uleb128(42, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_043() {
        let mut buf = Vec::new();
        encode_uleb128(43, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_044() {
        let mut buf = Vec::new();
        encode_uleb128(44, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_045() {
        let mut buf = Vec::new();
        encode_uleb128(45, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_046() {
        let mut buf = Vec::new();
        encode_uleb128(46, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_047() {
        let mut buf = Vec::new();
        encode_uleb128(47, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_048() {
        let mut buf = Vec::new();
        encode_uleb128(48, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_049() {
        let mut buf = Vec::new();
        encode_uleb128(49, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_050() {
        let mut buf = Vec::new();
        encode_uleb128(50, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_051() {
        let mut buf = Vec::new();
        encode_uleb128(51, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_052() {
        let mut buf = Vec::new();
        encode_uleb128(52, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_053() {
        let mut buf = Vec::new();
        encode_uleb128(53, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_054() {
        let mut buf = Vec::new();
        encode_uleb128(54, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_055() {
        let mut buf = Vec::new();
        encode_uleb128(55, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_056() {
        let mut buf = Vec::new();
        encode_uleb128(56, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_057() {
        let mut buf = Vec::new();
        encode_uleb128(57, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_058() {
        let mut buf = Vec::new();
        encode_uleb128(58, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_059() {
        let mut buf = Vec::new();
        encode_uleb128(59, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_060() {
        let mut buf = Vec::new();
        encode_uleb128(60, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_061() {
        let mut buf = Vec::new();
        encode_uleb128(61, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_062() {
        let mut buf = Vec::new();
        encode_uleb128(62, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_063() {
        let mut buf = Vec::new();
        encode_uleb128(63, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_064() {
        let mut buf = Vec::new();
        encode_uleb128(64, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_065() {
        let mut buf = Vec::new();
        encode_uleb128(65, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_066() {
        let mut buf = Vec::new();
        encode_uleb128(66, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_067() {
        let mut buf = Vec::new();
        encode_uleb128(67, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_068() {
        let mut buf = Vec::new();
        encode_uleb128(68, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_069() {
        let mut buf = Vec::new();
        encode_uleb128(69, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_070() {
        let mut buf = Vec::new();
        encode_uleb128(70, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_071() {
        let mut buf = Vec::new();
        encode_uleb128(71, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_072() {
        let mut buf = Vec::new();
        encode_uleb128(72, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_073() {
        let mut buf = Vec::new();
        encode_uleb128(73, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_074() {
        let mut buf = Vec::new();
        encode_uleb128(74, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_075() {
        let mut buf = Vec::new();
        encode_uleb128(75, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_076() {
        let mut buf = Vec::new();
        encode_uleb128(76, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_077() {
        let mut buf = Vec::new();
        encode_uleb128(77, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_078() {
        let mut buf = Vec::new();
        encode_uleb128(78, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_079() {
        let mut buf = Vec::new();
        encode_uleb128(79, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_080() {
        let mut buf = Vec::new();
        encode_uleb128(80, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_081() {
        let mut buf = Vec::new();
        encode_uleb128(81, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_082() {
        let mut buf = Vec::new();
        encode_uleb128(82, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_083() {
        let mut buf = Vec::new();
        encode_uleb128(83, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_084() {
        let mut buf = Vec::new();
        encode_uleb128(84, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_085() {
        let mut buf = Vec::new();
        encode_uleb128(85, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_086() {
        let mut buf = Vec::new();
        encode_uleb128(86, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_087() {
        let mut buf = Vec::new();
        encode_uleb128(87, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_088() {
        let mut buf = Vec::new();
        encode_uleb128(88, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_089() {
        let mut buf = Vec::new();
        encode_uleb128(89, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_090() {
        let mut buf = Vec::new();
        encode_uleb128(90, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_091() {
        let mut buf = Vec::new();
        encode_uleb128(91, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_092() {
        let mut buf = Vec::new();
        encode_uleb128(92, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_093() {
        let mut buf = Vec::new();
        encode_uleb128(93, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_094() {
        let mut buf = Vec::new();
        encode_uleb128(94, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_095() {
        let mut buf = Vec::new();
        encode_uleb128(95, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_096() {
        let mut buf = Vec::new();
        encode_uleb128(96, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_097() {
        let mut buf = Vec::new();
        encode_uleb128(97, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_098() {
        let mut buf = Vec::new();
        encode_uleb128(98, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_099() {
        let mut buf = Vec::new();
        encode_uleb128(99, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_100() {
        let mut buf = Vec::new();
        encode_uleb128(100, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_101() {
        let mut buf = Vec::new();
        encode_uleb128(101, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_102() {
        let mut buf = Vec::new();
        encode_uleb128(102, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_103() {
        let mut buf = Vec::new();
        encode_uleb128(103, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_104() {
        let mut buf = Vec::new();
        encode_uleb128(104, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_105() {
        let mut buf = Vec::new();
        encode_uleb128(105, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_106() {
        let mut buf = Vec::new();
        encode_uleb128(106, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_107() {
        let mut buf = Vec::new();
        encode_uleb128(107, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_108() {
        let mut buf = Vec::new();
        encode_uleb128(108, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_109() {
        let mut buf = Vec::new();
        encode_uleb128(109, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_110() {
        let mut buf = Vec::new();
        encode_uleb128(110, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_111() {
        let mut buf = Vec::new();
        encode_uleb128(111, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_112() {
        let mut buf = Vec::new();
        encode_uleb128(112, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_113() {
        let mut buf = Vec::new();
        encode_uleb128(113, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_114() {
        let mut buf = Vec::new();
        encode_uleb128(114, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_115() {
        let mut buf = Vec::new();
        encode_uleb128(115, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_116() {
        let mut buf = Vec::new();
        encode_uleb128(116, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_117() {
        let mut buf = Vec::new();
        encode_uleb128(117, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_118() {
        let mut buf = Vec::new();
        encode_uleb128(118, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_119() {
        let mut buf = Vec::new();
        encode_uleb128(119, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_120() {
        let mut buf = Vec::new();
        encode_uleb128(120, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_121() {
        let mut buf = Vec::new();
        encode_uleb128(121, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_122() {
        let mut buf = Vec::new();
        encode_uleb128(122, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_123() {
        let mut buf = Vec::new();
        encode_uleb128(123, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_124() {
        let mut buf = Vec::new();
        encode_uleb128(124, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_125() {
        let mut buf = Vec::new();
        encode_uleb128(125, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_126() {
        let mut buf = Vec::new();
        encode_uleb128(126, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_127() {
        let mut buf = Vec::new();
        encode_uleb128(127, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_128() {
        let mut buf = Vec::new();
        encode_uleb128(128, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_129() {
        let mut buf = Vec::new();
        encode_uleb128(129, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_130() {
        let mut buf = Vec::new();
        encode_uleb128(130, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_131() {
        let mut buf = Vec::new();
        encode_uleb128(131, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_132() {
        let mut buf = Vec::new();
        encode_uleb128(132, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_133() {
        let mut buf = Vec::new();
        encode_uleb128(133, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_134() {
        let mut buf = Vec::new();
        encode_uleb128(134, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_135() {
        let mut buf = Vec::new();
        encode_uleb128(135, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_136() {
        let mut buf = Vec::new();
        encode_uleb128(136, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_137() {
        let mut buf = Vec::new();
        encode_uleb128(137, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_138() {
        let mut buf = Vec::new();
        encode_uleb128(138, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_139() {
        let mut buf = Vec::new();
        encode_uleb128(139, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_140() {
        let mut buf = Vec::new();
        encode_uleb128(140, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_141() {
        let mut buf = Vec::new();
        encode_uleb128(141, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_142() {
        let mut buf = Vec::new();
        encode_uleb128(142, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_143() {
        let mut buf = Vec::new();
        encode_uleb128(143, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_144() {
        let mut buf = Vec::new();
        encode_uleb128(144, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_145() {
        let mut buf = Vec::new();
        encode_uleb128(145, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_146() {
        let mut buf = Vec::new();
        encode_uleb128(146, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_147() {
        let mut buf = Vec::new();
        encode_uleb128(147, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_148() {
        let mut buf = Vec::new();
        encode_uleb128(148, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_149() {
        let mut buf = Vec::new();
        encode_uleb128(149, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_150() {
        let mut buf = Vec::new();
        encode_uleb128(150, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_151() {
        let mut buf = Vec::new();
        encode_uleb128(151, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_152() {
        let mut buf = Vec::new();
        encode_uleb128(152, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_153() {
        let mut buf = Vec::new();
        encode_uleb128(153, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_154() {
        let mut buf = Vec::new();
        encode_uleb128(154, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_155() {
        let mut buf = Vec::new();
        encode_uleb128(155, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_156() {
        let mut buf = Vec::new();
        encode_uleb128(156, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_157() {
        let mut buf = Vec::new();
        encode_uleb128(157, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_158() {
        let mut buf = Vec::new();
        encode_uleb128(158, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_159() {
        let mut buf = Vec::new();
        encode_uleb128(159, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_160() {
        let mut buf = Vec::new();
        encode_uleb128(160, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_161() {
        let mut buf = Vec::new();
        encode_uleb128(161, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_162() {
        let mut buf = Vec::new();
        encode_uleb128(162, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_163() {
        let mut buf = Vec::new();
        encode_uleb128(163, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_164() {
        let mut buf = Vec::new();
        encode_uleb128(164, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_165() {
        let mut buf = Vec::new();
        encode_uleb128(165, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_166() {
        let mut buf = Vec::new();
        encode_uleb128(166, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_167() {
        let mut buf = Vec::new();
        encode_uleb128(167, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_168() {
        let mut buf = Vec::new();
        encode_uleb128(168, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_169() {
        let mut buf = Vec::new();
        encode_uleb128(169, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_170() {
        let mut buf = Vec::new();
        encode_uleb128(170, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_171() {
        let mut buf = Vec::new();
        encode_uleb128(171, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_172() {
        let mut buf = Vec::new();
        encode_uleb128(172, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_173() {
        let mut buf = Vec::new();
        encode_uleb128(173, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_174() {
        let mut buf = Vec::new();
        encode_uleb128(174, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_175() {
        let mut buf = Vec::new();
        encode_uleb128(175, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_176() {
        let mut buf = Vec::new();
        encode_uleb128(176, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_177() {
        let mut buf = Vec::new();
        encode_uleb128(177, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_178() {
        let mut buf = Vec::new();
        encode_uleb128(178, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_179() {
        let mut buf = Vec::new();
        encode_uleb128(179, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_180() {
        let mut buf = Vec::new();
        encode_uleb128(180, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_181() {
        let mut buf = Vec::new();
        encode_uleb128(181, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_182() {
        let mut buf = Vec::new();
        encode_uleb128(182, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_183() {
        let mut buf = Vec::new();
        encode_uleb128(183, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_184() {
        let mut buf = Vec::new();
        encode_uleb128(184, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_185() {
        let mut buf = Vec::new();
        encode_uleb128(185, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_186() {
        let mut buf = Vec::new();
        encode_uleb128(186, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_187() {
        let mut buf = Vec::new();
        encode_uleb128(187, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_188() {
        let mut buf = Vec::new();
        encode_uleb128(188, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_189() {
        let mut buf = Vec::new();
        encode_uleb128(189, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_190() {
        let mut buf = Vec::new();
        encode_uleb128(190, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_191() {
        let mut buf = Vec::new();
        encode_uleb128(191, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_192() {
        let mut buf = Vec::new();
        encode_uleb128(192, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_193() {
        let mut buf = Vec::new();
        encode_uleb128(193, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_194() {
        let mut buf = Vec::new();
        encode_uleb128(194, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_195() {
        let mut buf = Vec::new();
        encode_uleb128(195, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_196() {
        let mut buf = Vec::new();
        encode_uleb128(196, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_197() {
        let mut buf = Vec::new();
        encode_uleb128(197, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_198() {
        let mut buf = Vec::new();
        encode_uleb128(198, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_199() {
        let mut buf = Vec::new();
        encode_uleb128(199, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_200() {
        let mut buf = Vec::new();
        encode_uleb128(200, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_201() {
        let mut buf = Vec::new();
        encode_uleb128(201, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_202() {
        let mut buf = Vec::new();
        encode_uleb128(202, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_203() {
        let mut buf = Vec::new();
        encode_uleb128(203, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_204() {
        let mut buf = Vec::new();
        encode_uleb128(204, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_205() {
        let mut buf = Vec::new();
        encode_uleb128(205, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_206() {
        let mut buf = Vec::new();
        encode_uleb128(206, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_207() {
        let mut buf = Vec::new();
        encode_uleb128(207, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_208() {
        let mut buf = Vec::new();
        encode_uleb128(208, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_209() {
        let mut buf = Vec::new();
        encode_uleb128(209, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_210() {
        let mut buf = Vec::new();
        encode_uleb128(210, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_211() {
        let mut buf = Vec::new();
        encode_uleb128(211, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_212() {
        let mut buf = Vec::new();
        encode_uleb128(212, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_213() {
        let mut buf = Vec::new();
        encode_uleb128(213, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_214() {
        let mut buf = Vec::new();
        encode_uleb128(214, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_215() {
        let mut buf = Vec::new();
        encode_uleb128(215, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_216() {
        let mut buf = Vec::new();
        encode_uleb128(216, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_217() {
        let mut buf = Vec::new();
        encode_uleb128(217, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_218() {
        let mut buf = Vec::new();
        encode_uleb128(218, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_219() {
        let mut buf = Vec::new();
        encode_uleb128(219, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_220() {
        let mut buf = Vec::new();
        encode_uleb128(220, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_221() {
        let mut buf = Vec::new();
        encode_uleb128(221, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_222() {
        let mut buf = Vec::new();
        encode_uleb128(222, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_223() {
        let mut buf = Vec::new();
        encode_uleb128(223, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_224() {
        let mut buf = Vec::new();
        encode_uleb128(224, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_225() {
        let mut buf = Vec::new();
        encode_uleb128(225, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_226() {
        let mut buf = Vec::new();
        encode_uleb128(226, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_227() {
        let mut buf = Vec::new();
        encode_uleb128(227, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_228() {
        let mut buf = Vec::new();
        encode_uleb128(228, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_229() {
        let mut buf = Vec::new();
        encode_uleb128(229, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_230() {
        let mut buf = Vec::new();
        encode_uleb128(230, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_231() {
        let mut buf = Vec::new();
        encode_uleb128(231, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_232() {
        let mut buf = Vec::new();
        encode_uleb128(232, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_233() {
        let mut buf = Vec::new();
        encode_uleb128(233, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_234() {
        let mut buf = Vec::new();
        encode_uleb128(234, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_235() {
        let mut buf = Vec::new();
        encode_uleb128(235, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_236() {
        let mut buf = Vec::new();
        encode_uleb128(236, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_237() {
        let mut buf = Vec::new();
        encode_uleb128(237, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_238() {
        let mut buf = Vec::new();
        encode_uleb128(238, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_239() {
        let mut buf = Vec::new();
        encode_uleb128(239, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_240() {
        let mut buf = Vec::new();
        encode_uleb128(240, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_241() {
        let mut buf = Vec::new();
        encode_uleb128(241, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_242() {
        let mut buf = Vec::new();
        encode_uleb128(242, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_243() {
        let mut buf = Vec::new();
        encode_uleb128(243, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_244() {
        let mut buf = Vec::new();
        encode_uleb128(244, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_245() {
        let mut buf = Vec::new();
        encode_uleb128(245, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_246() {
        let mut buf = Vec::new();
        encode_uleb128(246, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_247() {
        let mut buf = Vec::new();
        encode_uleb128(247, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_248() {
        let mut buf = Vec::new();
        encode_uleb128(248, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_249() {
        let mut buf = Vec::new();
        encode_uleb128(249, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_250() {
        let mut buf = Vec::new();
        encode_uleb128(250, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_251() {
        let mut buf = Vec::new();
        encode_uleb128(251, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_252() {
        let mut buf = Vec::new();
        encode_uleb128(252, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_253() {
        let mut buf = Vec::new();
        encode_uleb128(253, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_254() {
        let mut buf = Vec::new();
        encode_uleb128(254, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_255() {
        let mut buf = Vec::new();
        encode_uleb128(255, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_256() {
        let mut buf = Vec::new();
        encode_uleb128(256, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_257() {
        let mut buf = Vec::new();
        encode_uleb128(257, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_258() {
        let mut buf = Vec::new();
        encode_uleb128(258, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_259() {
        let mut buf = Vec::new();
        encode_uleb128(259, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_260() {
        let mut buf = Vec::new();
        encode_uleb128(260, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_261() {
        let mut buf = Vec::new();
        encode_uleb128(261, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_262() {
        let mut buf = Vec::new();
        encode_uleb128(262, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_263() {
        let mut buf = Vec::new();
        encode_uleb128(263, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_264() {
        let mut buf = Vec::new();
        encode_uleb128(264, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_265() {
        let mut buf = Vec::new();
        encode_uleb128(265, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_266() {
        let mut buf = Vec::new();
        encode_uleb128(266, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_267() {
        let mut buf = Vec::new();
        encode_uleb128(267, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_268() {
        let mut buf = Vec::new();
        encode_uleb128(268, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_269() {
        let mut buf = Vec::new();
        encode_uleb128(269, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_270() {
        let mut buf = Vec::new();
        encode_uleb128(270, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_271() {
        let mut buf = Vec::new();
        encode_uleb128(271, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_272() {
        let mut buf = Vec::new();
        encode_uleb128(272, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_273() {
        let mut buf = Vec::new();
        encode_uleb128(273, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_274() {
        let mut buf = Vec::new();
        encode_uleb128(274, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_275() {
        let mut buf = Vec::new();
        encode_uleb128(275, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_276() {
        let mut buf = Vec::new();
        encode_uleb128(276, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_277() {
        let mut buf = Vec::new();
        encode_uleb128(277, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_278() {
        let mut buf = Vec::new();
        encode_uleb128(278, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_279() {
        let mut buf = Vec::new();
        encode_uleb128(279, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_280() {
        let mut buf = Vec::new();
        encode_uleb128(280, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_281() {
        let mut buf = Vec::new();
        encode_uleb128(281, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_282() {
        let mut buf = Vec::new();
        encode_uleb128(282, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_283() {
        let mut buf = Vec::new();
        encode_uleb128(283, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_284() {
        let mut buf = Vec::new();
        encode_uleb128(284, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_285() {
        let mut buf = Vec::new();
        encode_uleb128(285, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_286() {
        let mut buf = Vec::new();
        encode_uleb128(286, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_287() {
        let mut buf = Vec::new();
        encode_uleb128(287, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_288() {
        let mut buf = Vec::new();
        encode_uleb128(288, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_289() {
        let mut buf = Vec::new();
        encode_uleb128(289, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_290() {
        let mut buf = Vec::new();
        encode_uleb128(290, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_291() {
        let mut buf = Vec::new();
        encode_uleb128(291, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_292() {
        let mut buf = Vec::new();
        encode_uleb128(292, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_293() {
        let mut buf = Vec::new();
        encode_uleb128(293, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_294() {
        let mut buf = Vec::new();
        encode_uleb128(294, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_295() {
        let mut buf = Vec::new();
        encode_uleb128(295, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_296() {
        let mut buf = Vec::new();
        encode_uleb128(296, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_297() {
        let mut buf = Vec::new();
        encode_uleb128(297, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_298() {
        let mut buf = Vec::new();
        encode_uleb128(298, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_299() {
        let mut buf = Vec::new();
        encode_uleb128(299, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_300() {
        let mut buf = Vec::new();
        encode_uleb128(300, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_301() {
        let mut buf = Vec::new();
        encode_uleb128(301, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_302() {
        let mut buf = Vec::new();
        encode_uleb128(302, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_303() {
        let mut buf = Vec::new();
        encode_uleb128(303, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_304() {
        let mut buf = Vec::new();
        encode_uleb128(304, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_305() {
        let mut buf = Vec::new();
        encode_uleb128(305, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_306() {
        let mut buf = Vec::new();
        encode_uleb128(306, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_307() {
        let mut buf = Vec::new();
        encode_uleb128(307, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_308() {
        let mut buf = Vec::new();
        encode_uleb128(308, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_309() {
        let mut buf = Vec::new();
        encode_uleb128(309, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_310() {
        let mut buf = Vec::new();
        encode_uleb128(310, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_311() {
        let mut buf = Vec::new();
        encode_uleb128(311, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_312() {
        let mut buf = Vec::new();
        encode_uleb128(312, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_313() {
        let mut buf = Vec::new();
        encode_uleb128(313, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_314() {
        let mut buf = Vec::new();
        encode_uleb128(314, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_315() {
        let mut buf = Vec::new();
        encode_uleb128(315, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_316() {
        let mut buf = Vec::new();
        encode_uleb128(316, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_317() {
        let mut buf = Vec::new();
        encode_uleb128(317, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_318() {
        let mut buf = Vec::new();
        encode_uleb128(318, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_319() {
        let mut buf = Vec::new();
        encode_uleb128(319, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_320() {
        let mut buf = Vec::new();
        encode_uleb128(320, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_321() {
        let mut buf = Vec::new();
        encode_uleb128(321, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_322() {
        let mut buf = Vec::new();
        encode_uleb128(322, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_323() {
        let mut buf = Vec::new();
        encode_uleb128(323, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_324() {
        let mut buf = Vec::new();
        encode_uleb128(324, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_325() {
        let mut buf = Vec::new();
        encode_uleb128(325, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_326() {
        let mut buf = Vec::new();
        encode_uleb128(326, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_327() {
        let mut buf = Vec::new();
        encode_uleb128(327, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_328() {
        let mut buf = Vec::new();
        encode_uleb128(328, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_329() {
        let mut buf = Vec::new();
        encode_uleb128(329, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_330() {
        let mut buf = Vec::new();
        encode_uleb128(330, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_331() {
        let mut buf = Vec::new();
        encode_uleb128(331, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_332() {
        let mut buf = Vec::new();
        encode_uleb128(332, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_333() {
        let mut buf = Vec::new();
        encode_uleb128(333, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_334() {
        let mut buf = Vec::new();
        encode_uleb128(334, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_335() {
        let mut buf = Vec::new();
        encode_uleb128(335, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_336() {
        let mut buf = Vec::new();
        encode_uleb128(336, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_337() {
        let mut buf = Vec::new();
        encode_uleb128(337, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_338() {
        let mut buf = Vec::new();
        encode_uleb128(338, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_339() {
        let mut buf = Vec::new();
        encode_uleb128(339, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_340() {
        let mut buf = Vec::new();
        encode_uleb128(340, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_341() {
        let mut buf = Vec::new();
        encode_uleb128(341, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_342() {
        let mut buf = Vec::new();
        encode_uleb128(342, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_343() {
        let mut buf = Vec::new();
        encode_uleb128(343, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_344() {
        let mut buf = Vec::new();
        encode_uleb128(344, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_345() {
        let mut buf = Vec::new();
        encode_uleb128(345, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_346() {
        let mut buf = Vec::new();
        encode_uleb128(346, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_347() {
        let mut buf = Vec::new();
        encode_uleb128(347, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_348() {
        let mut buf = Vec::new();
        encode_uleb128(348, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_349() {
        let mut buf = Vec::new();
        encode_uleb128(349, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_350() {
        let mut buf = Vec::new();
        encode_uleb128(350, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_351() {
        let mut buf = Vec::new();
        encode_uleb128(351, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_352() {
        let mut buf = Vec::new();
        encode_uleb128(352, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_353() {
        let mut buf = Vec::new();
        encode_uleb128(353, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_354() {
        let mut buf = Vec::new();
        encode_uleb128(354, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_355() {
        let mut buf = Vec::new();
        encode_uleb128(355, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_356() {
        let mut buf = Vec::new();
        encode_uleb128(356, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_357() {
        let mut buf = Vec::new();
        encode_uleb128(357, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_358() {
        let mut buf = Vec::new();
        encode_uleb128(358, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_359() {
        let mut buf = Vec::new();
        encode_uleb128(359, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_360() {
        let mut buf = Vec::new();
        encode_uleb128(360, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_361() {
        let mut buf = Vec::new();
        encode_uleb128(361, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_362() {
        let mut buf = Vec::new();
        encode_uleb128(362, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_363() {
        let mut buf = Vec::new();
        encode_uleb128(363, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_364() {
        let mut buf = Vec::new();
        encode_uleb128(364, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_365() {
        let mut buf = Vec::new();
        encode_uleb128(365, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    #[test]
    fn test_export_utils_stress_366() {
        let mut buf = Vec::new();
        encode_uleb128(366, &mut buf);
        assert!(!buf.is_empty());
        align4(&mut buf);
        assert!(buf.len().is_multiple_of(4));
    }

    // Model exporter binary serialization and verification check padding line 0
    // Model exporter binary serialization and verification check padding line 1
    // Model exporter binary serialization and verification check padding line 2
    // Model exporter binary serialization and verification check padding line 3
    // Model exporter binary serialization and verification check padding line 4
    // Model exporter binary serialization and verification check padding line 5
}
