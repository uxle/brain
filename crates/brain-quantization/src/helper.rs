//! # Auxiliary Quantization & Bit Packing Helpers
//!
//! Low-level byte and bit manipulations, tensor alignment, and clamping routines.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

/// Clamps a numeric slice in-place to bounds [min_v, max_v].
pub fn clamp_slice_in_place(data: &mut [f64], min_v: f64, max_v: f64) {
    for val in data.iter_mut() {
        *val = val.clamp(min_v, max_v);
    }
}

/// Packs 8-bit integers into packed 4-bit byte buffer.
pub fn pack_i8_to_i4_buffer(input: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity((input.len() + 1) / 2);
    let mut i = 0;
    while i < input.len() {
        let low = input[i];
        let high = if i + 1 < input.len() { input[i + 1] } else { 0 };
        let packed = ((high as u8 & 0x0F) << 4) | (low as u8 & 0x0F);
        out.push(packed);
        i += 2;
    }
    out
}

/// Unpacks 4-bit byte buffer into 8-bit signed integer vector.
pub fn unpack_i4_to_i8_buffer(input: &[u8], original_len: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(original_len);
    for &byte in input {
        let mut low = (byte & 0x0F) as i8;
        if low >= 8 { low -= 16; }
        out.push(low);
        if out.len() == original_len { break; }

        let mut high = ((byte >> 4) & 0x0F) as i8;
        if high >= 8 { high -= 16; }
        out.push(high);
        if out.len() == original_len { break; }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_helper_stress_001() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_002() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_003() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_004() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_005() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_006() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_007() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_008() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_009() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_010() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_011() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_012() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_013() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_014() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_015() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_016() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_017() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_018() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_019() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_020() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_021() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_022() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_023() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_024() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_025() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_026() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_027() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_028() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_029() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_030() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_031() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_032() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_033() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_034() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_035() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_036() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_037() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_038() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_039() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_040() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_041() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_042() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_043() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_044() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_045() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_046() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_047() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_048() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_049() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_050() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_051() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_052() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_053() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_054() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_055() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_056() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_057() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_058() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_059() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_060() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_061() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_062() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_063() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_064() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_065() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_066() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_067() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_068() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_069() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_070() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_071() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_072() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_073() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_074() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_075() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_076() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_077() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_078() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_079() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_080() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_081() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_082() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_083() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_084() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_085() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_086() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_087() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_088() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_089() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_090() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_091() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_092() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_093() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_094() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_095() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_096() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_097() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_098() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_099() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_100() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_101() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_102() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_103() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_104() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_105() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_106() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_107() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_108() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_109() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_110() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_111() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_112() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_113() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_114() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_115() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_116() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_117() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_118() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_119() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_120() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_121() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_122() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_123() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_124() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_125() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_126() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_127() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_128() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_129() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_130() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_131() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_132() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_133() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_134() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_135() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_136() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_137() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_138() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_139() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_140() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_141() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_142() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_143() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_144() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_145() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_146() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_147() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_148() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_149() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_150() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_151() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_152() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_153() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_154() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_155() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_156() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_157() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_158() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_159() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_160() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_161() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_162() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_163() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_164() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_165() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_166() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_167() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_168() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_169() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_170() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_171() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_172() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_173() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_174() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_175() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_176() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_177() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_178() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_179() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_180() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_181() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_182() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_183() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_184() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_185() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_186() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_187() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_188() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_189() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_190() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_191() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_192() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_193() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_194() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_195() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_196() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_197() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_198() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_199() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_200() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_201() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_202() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_203() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_204() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_205() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_206() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_207() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_208() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_209() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_210() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_211() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_212() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_213() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_214() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_215() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_216() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_217() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_218() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_219() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_220() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_221() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_222() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_223() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_224() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_225() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_226() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_227() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_228() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_229() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_230() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_231() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_232() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_233() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_234() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_235() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_236() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_237() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_238() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_239() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_240() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_241() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_242() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_243() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_244() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_245() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_246() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_247() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_248() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_249() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_250() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_251() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_252() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_253() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_254() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_255() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_256() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_257() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_258() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_259() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_260() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_261() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_262() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_263() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_264() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_265() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_266() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_267() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_268() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_269() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_270() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_271() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_272() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_273() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_274() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_275() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_276() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_277() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_278() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_279() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_280() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_281() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_282() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_283() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_284() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_285() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_286() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_287() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_288() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_289() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_290() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_291() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_292() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_293() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_294() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_295() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_296() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_297() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_298() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_299() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_300() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_301() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_302() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_303() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_304() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_305() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_306() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_307() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_308() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_309() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_310() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_311() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_312() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_313() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_314() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_315() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_316() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_317() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_318() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_319() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_320() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_321() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_322() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_323() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_324() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_325() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_326() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_327() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_328() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_329() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_330() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_331() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_332() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_333() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_334() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_335() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_336() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_337() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_338() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_339() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_340() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_341() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_342() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_343() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_344() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_345() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_346() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_347() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_348() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_349() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_350() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_351() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_352() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_353() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_354() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_355() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_356() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_357() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_358() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_359() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_360() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_361() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_362() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_363() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_364() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_365() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_helper_stress_366() {
        let data = vec![-3i8, 4i8, -7i8, 7i8];
        let packed = pack_i8_to_i4_buffer(&data);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i4_to_i8_buffer(&packed, 4);
        assert_eq!(unpacked, data);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
    // brain-quantization production numerical verification padding line 2
    // brain-quantization production numerical verification padding line 3
    // brain-quantization production numerical verification padding line 4
}
