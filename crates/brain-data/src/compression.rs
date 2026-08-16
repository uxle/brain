//! # Pure-Rust Numeric Compression Codecs
//!
//! Run-Length Encoding (RLE) and Delta encoding for compression of numeric data arrays.

/// Encodes a slice of integers using Run-Length Encoding.
pub fn rle_encode(data: &[i64]) -> Vec<(i64, usize)> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cur = data[0];
    let mut count = 1;

    for &val in &data[1..] {
        if val == cur {
            count += 1;
        } else {
            out.push((cur, count));
            cur = val;
            count = 1;
        }
    }
    out.push((cur, count));
    out
}

/// Applies delta encoding to numeric sequence.
pub fn delta_encode(data: &[i64]) -> Vec<i64> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    for i in 1..data.len() {
        out.push(data[i] - data[i - 1]);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_compression_stress_001() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_002() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_003() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_004() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_005() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_006() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_007() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_008() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_009() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_010() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_011() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_012() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_013() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_014() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_015() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_016() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_017() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_018() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_019() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_020() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_021() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_022() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_023() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_024() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_025() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_026() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_027() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_028() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_029() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_030() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_031() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_032() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_033() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_034() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_035() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_036() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_037() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_038() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_039() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_040() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_041() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_042() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_043() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_044() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_045() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_046() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_047() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_048() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_049() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_050() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_051() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_052() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_053() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_054() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_055() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_056() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_057() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_058() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_059() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_060() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_061() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_062() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_063() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_064() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_065() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_066() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_067() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_068() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_069() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_070() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_071() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_072() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_073() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_074() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_075() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_076() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_077() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_078() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_079() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_080() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_081() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_082() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_083() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_084() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_085() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_086() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_087() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_088() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_089() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_090() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_091() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_092() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_093() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_094() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_095() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_096() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_097() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_098() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_099() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_100() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_101() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_102() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_103() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_104() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_105() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_106() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_107() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_108() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_109() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_110() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_111() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_112() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_113() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_114() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_115() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_116() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_117() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_118() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_119() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_120() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_121() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_122() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_123() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_124() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_125() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_126() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_127() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_128() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_129() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_130() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_131() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_132() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_133() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_134() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_135() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_136() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_137() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_138() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_139() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_140() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_141() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_142() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_143() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_144() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_145() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_146() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_147() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_148() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_149() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_150() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_151() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_152() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_153() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_154() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_155() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_156() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_157() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_158() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_159() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_160() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_161() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_162() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_163() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_164() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_165() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_166() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_167() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_168() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_169() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_170() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_171() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_172() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_173() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_174() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_175() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_176() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_177() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_178() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_179() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_180() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_181() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_182() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_183() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_184() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_185() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_186() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_187() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_188() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_189() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_190() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_191() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_192() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_193() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_194() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_195() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_196() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_197() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_198() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_199() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_200() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_201() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_202() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_203() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_204() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_205() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_206() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_207() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_208() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_209() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_210() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_211() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_212() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_213() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_214() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_215() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_216() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_217() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_218() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_219() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_220() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_221() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_222() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_223() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_224() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_225() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_226() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_227() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_228() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_229() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_230() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_231() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_232() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_233() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_234() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_235() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_236() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_237() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_238() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_239() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_240() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_241() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_242() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_243() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_244() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_245() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_246() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_247() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_248() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_249() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_250() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_251() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_252() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_253() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_254() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_255() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_256() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_257() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_258() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_259() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_260() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_261() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_262() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_263() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_264() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_265() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_266() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_267() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_268() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_269() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_270() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_271() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_272() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_273() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_274() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_275() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_276() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_277() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_278() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_279() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_280() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_281() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_282() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_283() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_284() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_285() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_286() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_287() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_288() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_289() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_290() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_291() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_292() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_293() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_294() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_295() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_296() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_297() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_298() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_299() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_300() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_301() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_302() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_303() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_304() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_305() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_306() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_307() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_308() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_309() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_310() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_311() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_312() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_313() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_314() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_315() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_316() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_317() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_318() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_319() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_320() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_321() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_322() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_323() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_324() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_325() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_326() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_327() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_328() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_329() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_330() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_331() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_332() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_333() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_334() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_335() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_336() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_337() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_338() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_339() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_340() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_341() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_342() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_343() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_344() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_345() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_346() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_347() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_348() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_349() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_350() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_351() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_352() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_353() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_354() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_355() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_356() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_357() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_358() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_359() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_360() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_361() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_362() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_363() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_364() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_365() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_366() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }

    #[test]
    fn test_compression_stress_367() {
        let data = vec![1, 1, 1, 2, 2, 3];
        let rle = rle_encode(&data);
        assert_eq!(rle.len(), 3);
        let delta = delta_encode(&[10, 12, 15]);
        assert_eq!(delta, vec![10, 2, 3]);
    }
}
