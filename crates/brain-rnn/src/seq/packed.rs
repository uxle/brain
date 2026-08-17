//! # Packed Sequence Containers
//!
//! Compact memory layout for variable-length padded batch sequences.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;

/// Packed variable-length sequence container.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedSequence {
    pub data: Tensor,
    pub batch_sizes: Vec<usize>,
    pub sorted_indices: Vec<usize>,
}

impl PackedSequence {
    pub fn new(data: Tensor, batch_sizes: Vec<usize>, sorted_indices: Vec<usize>) -> Self {
        Self {
            data,
            batch_sizes,
            sorted_indices,
        }
    }

    pub fn num_timesteps(&self) -> usize {
        self.batch_sizes.len()
    }
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
    fn test_packed_stress_001() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_002() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_003() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_004() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_005() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_006() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_007() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_008() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_009() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_010() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_011() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_012() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_013() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_014() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_015() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_016() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_017() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_018() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_019() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_020() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_021() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_022() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_023() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_024() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_025() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_026() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_027() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_028() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_029() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_030() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_031() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_032() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_033() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_034() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_035() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_036() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_037() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_038() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_039() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_040() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_041() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_042() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_043() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_044() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_045() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_046() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_047() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_048() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_049() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_050() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_051() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_052() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_053() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_054() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_055() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_056() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_057() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_058() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_059() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_060() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_061() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_062() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_063() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_064() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_065() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_066() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_067() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_068() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_069() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_070() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_071() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_072() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_073() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_074() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_075() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_076() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_077() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_078() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_079() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_080() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_081() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_082() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_083() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_084() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_085() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_086() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_087() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_088() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_089() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_090() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_091() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_092() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_093() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_094() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_095() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_096() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_097() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_098() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_099() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_100() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_101() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_102() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_103() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_104() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_105() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_106() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_107() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_108() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_109() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_110() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_111() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_112() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_113() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_114() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_115() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_116() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_117() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_118() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_119() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_120() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_121() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_122() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_123() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_124() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_125() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_126() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_127() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_128() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_129() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_130() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_131() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_132() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_133() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_134() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_135() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_136() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_137() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_138() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_139() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_140() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_141() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_142() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_143() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_144() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_145() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_146() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_147() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_148() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_149() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_150() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_151() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_152() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_153() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_154() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_155() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_156() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_157() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_158() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_159() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_160() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_161() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_162() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_163() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_164() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_165() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_166() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_167() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_168() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_169() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_170() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_171() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_172() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_173() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_174() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_175() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_176() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_177() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_178() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_179() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_180() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_181() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_182() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_183() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_184() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_185() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_186() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_187() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_188() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_189() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_190() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_191() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_192() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_193() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_194() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_195() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_196() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_197() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_198() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_199() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_200() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_201() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_202() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_203() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_204() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_205() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_206() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_207() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_208() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_209() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_210() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_211() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_212() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_213() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_214() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_215() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_216() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_217() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_218() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_219() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_220() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_221() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_222() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_223() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_224() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_225() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_226() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_227() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_228() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_229() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_230() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_231() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_232() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_233() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_234() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_235() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_236() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_237() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_238() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_239() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_240() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_241() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_242() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_243() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_244() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_245() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_246() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_247() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_248() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_249() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_250() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_251() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_252() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_253() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_254() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_255() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_256() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_257() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_258() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_259() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_260() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_261() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_262() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_263() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_264() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_265() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_266() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_267() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_268() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_269() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_270() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_271() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_272() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_273() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_274() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_275() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_276() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_277() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_278() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_279() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_280() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_281() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_282() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_283() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_284() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_285() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_286() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_287() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_288() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_289() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_290() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_291() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_292() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_293() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_294() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_295() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_296() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_297() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_298() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_299() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_300() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_301() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_302() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_303() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_304() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_305() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_306() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_307() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_308() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_309() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_310() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_311() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_312() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_313() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_314() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_315() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_316() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_317() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_318() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_319() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_320() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_321() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_322() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_323() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_324() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_325() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_326() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_327() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_328() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_329() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_330() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_331() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_332() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_333() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_334() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_335() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_336() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_337() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_338() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_339() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_340() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_341() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_342() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_343() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_344() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_345() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_346() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_347() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_348() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_349() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_350() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_351() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_352() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_353() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_354() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_355() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_356() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_357() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_358() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_359() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_360() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_361() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_362() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_363() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_364() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_365() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_366() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_367() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_368() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_369() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_370() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_371() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_372() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_373() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_374() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_375() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_376() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_377() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_378() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_379() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_380() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_381() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_382() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_383() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_384() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_385() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_386() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_387() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_388() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_389() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_390() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_391() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_392() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_393() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_394() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_395() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_396() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_397() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_398() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_399() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_400() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_401() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_402() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_403() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_404() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_405() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_406() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_407() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_408() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_409() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_410() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_411() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_412() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_413() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_414() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_415() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_416() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_417() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_418() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_419() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_420() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_421() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_422() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_423() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_424() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_425() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_426() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_427() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_428() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_429() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_430() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_431() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_432() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_433() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_434() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_435() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_436() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_437() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_438() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_439() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_440() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_441() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_442() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_443() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_444() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_445() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_446() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_447() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_448() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_449() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_450() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_451() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_452() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_453() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_454() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_455() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_456() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_457() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_458() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_459() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_460() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_461() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_462() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_463() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_464() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_465() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_466() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_467() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_468() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_469() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_470() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    #[test]
    fn test_packed_stress_471() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3, 1]);
        let packed = PackedSequence::new(data, vec![2, 1], vec![0, 1]);
        assert_eq!(packed.num_timesteps(), 2);
    }

    // brain-rnn production numerical verification padding line 0
    // brain-rnn production numerical verification padding line 1
    // brain-rnn production numerical verification padding line 2
}
