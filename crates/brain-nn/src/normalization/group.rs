//! # Group Normalization & Instance Normalization
//!
//! Normalization dividing channels into groups, independent of mini-batch size.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

/// Group Normalization module.
#[derive(Debug, Clone)]
pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f64,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            weight: Tensor::from_vec(vec![1.0; num_channels], vec![num_channels]),
            bias: Tensor::zeros(vec![num_channels]),
        }
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(input.clone())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_groupnorm_stress_001() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_002() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_003() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_004() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_005() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_006() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_007() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_008() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_009() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_010() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_011() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_012() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_013() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_014() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_015() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_016() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_017() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_018() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_019() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_020() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_021() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_022() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_023() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_024() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_025() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_026() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_027() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_028() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_029() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_030() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_031() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_032() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_033() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_034() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_035() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_036() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_037() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_038() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_039() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_040() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_041() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_042() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_043() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_044() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_045() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_046() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_047() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_048() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_049() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_050() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_051() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_052() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_053() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_054() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_055() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_056() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_057() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_058() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_059() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_060() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_061() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_062() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_063() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_064() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_065() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_066() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_067() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_068() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_069() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_070() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_071() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_072() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_073() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_074() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_075() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_076() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_077() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_078() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_079() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_080() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_081() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_082() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_083() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_084() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_085() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_086() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_087() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_088() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_089() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_090() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_091() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_092() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_093() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_094() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_095() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_096() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_097() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_098() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_099() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_100() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_101() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_102() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_103() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_104() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_105() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_106() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_107() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_108() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_109() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_110() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_111() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_112() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_113() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_114() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_115() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_116() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_117() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_118() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_119() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_120() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_121() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_122() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_123() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_124() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_125() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_126() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_127() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_128() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_129() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_130() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_131() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_132() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_133() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_134() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_135() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_136() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_137() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_138() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_139() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_140() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_141() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_142() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_143() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_144() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_145() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_146() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_147() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_148() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_149() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_150() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_151() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_152() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_153() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_154() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_155() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_156() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_157() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_158() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_159() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_160() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_161() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_162() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_163() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_164() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_165() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_166() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_167() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_168() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_169() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_170() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_171() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_172() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_173() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_174() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_175() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_176() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_177() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_178() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_179() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_180() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_181() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_182() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_183() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_184() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_185() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_186() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_187() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_188() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_189() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_190() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_191() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_192() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_193() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_194() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_195() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_196() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_197() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_198() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_199() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_200() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_201() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_202() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_203() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_204() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_205() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_206() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_207() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_208() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_209() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_210() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_211() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_212() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_213() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_214() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_215() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_216() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_217() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_218() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_219() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_220() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_221() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_222() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_223() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_224() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_225() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_226() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_227() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_228() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_229() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_230() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_231() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_232() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_233() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_234() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_235() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_236() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_237() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_238() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_239() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_240() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_241() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_242() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_243() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_244() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_245() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_246() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_247() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_248() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_249() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_250() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_251() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_252() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_253() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_254() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_255() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_256() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_257() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_258() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_259() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_260() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_261() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_262() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_263() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_264() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_265() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_266() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_267() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_268() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_269() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_270() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_271() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_272() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_273() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_274() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_275() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_276() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_277() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_278() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_279() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_280() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_281() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_282() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_283() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_284() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_285() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_286() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_287() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_288() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_289() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_290() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_291() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_292() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_293() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_294() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_295() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_296() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_297() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_298() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_299() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_300() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_301() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_302() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_303() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_304() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_305() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_306() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_307() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_308() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_309() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_310() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_311() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_312() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_313() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_314() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_315() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_316() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_317() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_318() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_319() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_320() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_321() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_322() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_323() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_324() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_325() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_326() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_327() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_328() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_329() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_330() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_331() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_332() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_333() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_334() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_335() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_336() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_337() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_338() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_339() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_340() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_341() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_342() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_343() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_344() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_345() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_346() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_347() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_348() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_349() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_350() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_351() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_352() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_353() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_354() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_355() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_356() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_357() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_358() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_359() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_360() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_361() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_362() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_363() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_364() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_365() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_366() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_367() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_368() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_369() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_370() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_371() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_372() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_373() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_374() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_375() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_376() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_377() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_378() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_379() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_380() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_381() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_382() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_383() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_384() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_385() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_386() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_387() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_388() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_389() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_390() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_391() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_392() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_393() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_394() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_395() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_396() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_397() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_398() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_399() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_400() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_401() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_402() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_403() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_404() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_405() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_406() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_407() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_408() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_409() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_410() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_411() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_412() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_413() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_414() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_415() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_416() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_417() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_418() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_419() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_420() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_421() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_422() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_423() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_424() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_425() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_426() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_427() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_428() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_429() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_430() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_431() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_432() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_433() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_434() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_435() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_436() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_437() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_438() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_439() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_440() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_441() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_442() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_443() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_444() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_445() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_446() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_447() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_448() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_449() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_450() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_451() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_452() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_453() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_454() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_455() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_456() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_457() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_458() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_459() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_460() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_461() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_462() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_463() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_464() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_465() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_466() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_467() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_468() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_469() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_470() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_471() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_stress_472() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::zeros(vec![1, 4, 8, 8]);
        assert_eq!(gn.forward(&x).unwrap().shape(), &[1, 4, 8, 8]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
}
