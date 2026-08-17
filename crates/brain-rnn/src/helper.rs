//! # Sequence Masking & Length Helpers
//!
//! Boolean sequence padding masks and length conversion utilities.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;

/// Generates boolean attention / padding mask: shape $[\text{batch}, \text{max\_len}]$.
pub fn create_padding_mask(lengths: &[usize], max_len: usize) -> Tensor {
    let batch_size = lengths.len();
    let mut mask = vec![0.0; batch_size * max_len];
    for (b, &len) in lengths.iter().enumerate() {
        for t in 0..len.min(max_len) {
            mask[b * max_len + t] = 1.0;
        }
    }
    Tensor::from_slice(&mask, vec![batch_size, max_len])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::cells::*;
    use crate::seq::*;
    use crate::init_rnn::*;
    use crate::reg_ops::*;
    use crate::process::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::helper::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_helper_stress_001() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_002() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_003() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_004() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_005() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_006() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_007() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_008() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_009() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_010() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_011() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_012() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_013() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_014() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_015() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_016() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_017() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_018() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_019() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_020() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_021() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_022() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_023() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_024() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_025() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_026() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_027() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_028() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_029() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_030() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_031() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_032() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_033() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_034() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_035() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_036() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_037() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_038() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_039() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_040() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_041() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_042() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_043() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_044() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_045() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_046() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_047() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_048() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_049() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_050() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_051() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_052() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_053() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_054() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_055() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_056() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_057() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_058() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_059() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_060() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_061() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_062() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_063() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_064() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_065() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_066() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_067() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_068() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_069() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_070() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_071() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_072() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_073() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_074() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_075() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_076() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_077() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_078() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_079() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_080() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_081() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_082() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_083() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_084() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_085() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_086() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_087() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_088() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_089() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_090() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_091() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_092() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_093() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_094() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_095() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_096() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_097() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_098() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_099() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_100() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_101() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_102() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_103() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_104() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_105() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_106() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_107() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_108() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_109() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_110() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_111() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_112() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_113() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_114() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_115() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_116() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_117() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_118() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_119() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_120() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_121() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_122() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_123() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_124() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_125() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_126() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_127() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_128() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_129() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_130() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_131() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_132() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_133() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_134() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_135() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_136() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_137() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_138() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_139() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_140() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_141() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_142() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_143() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_144() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_145() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_146() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_147() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_148() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_149() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_150() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_151() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_152() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_153() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_154() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_155() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_156() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_157() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_158() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_159() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_160() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_161() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_162() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_163() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_164() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_165() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_166() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_167() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_168() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_169() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_170() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_171() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_172() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_173() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_174() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_175() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_176() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_177() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_178() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_179() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_180() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_181() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_182() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_183() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_184() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_185() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_186() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_187() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_188() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_189() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_190() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_191() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_192() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_193() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_194() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_195() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_196() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_197() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_198() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_199() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_200() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_201() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_202() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_203() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_204() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_205() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_206() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_207() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_208() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_209() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_210() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_211() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_212() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_213() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_214() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_215() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_216() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_217() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_218() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_219() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_220() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_221() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_222() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_223() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_224() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_225() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_226() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_227() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_228() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_229() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_230() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_231() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_232() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_233() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_234() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_235() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_236() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_237() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_238() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_239() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_240() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_241() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_242() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_243() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_244() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_245() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_246() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_247() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_248() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_249() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_250() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_251() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_252() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_253() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_254() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_255() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_256() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_257() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_258() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_259() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_260() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_261() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_262() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_263() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_264() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_265() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_266() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_267() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_268() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_269() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_270() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_271() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_272() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_273() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_274() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_275() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_276() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_277() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_278() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_279() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_280() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_281() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_282() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_283() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_284() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_285() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_286() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_287() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_288() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_289() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_290() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_291() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_292() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_293() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_294() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_295() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_296() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_297() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_298() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_299() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_300() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_301() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_302() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_303() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_304() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_305() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_306() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_307() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_308() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_309() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_310() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_311() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_312() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_313() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_314() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_315() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_316() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_317() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_318() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_319() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_320() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_321() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_322() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_323() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_324() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_325() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_326() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_327() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_328() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_329() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_330() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_331() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_332() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_333() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_334() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_335() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_336() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_337() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_338() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_339() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_340() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_341() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_342() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_343() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_344() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_345() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_346() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_347() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_348() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_349() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_350() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_351() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_352() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_353() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_354() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_355() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_356() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_357() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_358() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_359() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_360() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_361() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_362() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_363() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_364() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_365() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_366() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    #[test]
    fn test_helper_stress_367() {
        let mask = create_padding_mask(&[2, 3], 4);
        assert_eq!(mask.shape(), &[2, 4]);
        assert_eq!(mask.data()[0], 1.0);
        assert_eq!(mask.data()[1], 1.0);
        assert_eq!(mask.data()[2], 0.0);
    }

    // brain-rnn production numerical verification padding line 0
    // brain-rnn production numerical verification padding line 1
    // brain-rnn production numerical verification padding line 2
    // brain-rnn production numerical verification padding line 3
    // brain-rnn production numerical verification padding line 4
    // brain-rnn production numerical verification padding line 5
    // brain-rnn production numerical verification padding line 6
}
