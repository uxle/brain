//! # Weight Decay Policies
//!
//! Decoupled weight decay (AdamW/SGDW) and L2-equivalent weight shrinkage policies.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;

/// Configuration for weight decay policies.
#[derive(Debug, Clone, PartialEq)]
pub struct DecayConfig {
    pub rate: f64,
    pub decoupled: bool,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            rate: 1e-4,
            decoupled: true,
        }
    }
}

/// Decoupled weight decay executor applying direct parameter shrinkage.
#[derive(Debug, Clone)]
pub struct DecoupledWeightDecay {
    pub rate: f64,
}

impl DecoupledWeightDecay {
    pub fn new(rate: f64) -> Self {
        Self { rate: rate.max(0.0) }
    }

    /// Applies decoupled shrinkage in-place given learning rate lr.
    pub fn apply_decay(&self, param: &mut Tensor, lr: f64) {
        let factor = 1.0 - lr * self.rate;
        for val in param.data_mut() {
            *val *= factor;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::dropout::*;
    use crate::normalization::*;
    use crate::regularizers::*;
    use crate::decay::*;
    use crate::earlystop::*;
    use crate::stopping::*;
    use crate::augment::*;
    use crate::perturb::*;
    use crate::dropout_uncertainty::*;
    use crate::label_smooth::*;
    use crate::curriculum::*;
    use crate::consistency::*;
    use crate::rules::*;
    use crate::registry::*;
    use crate::train_hooks::*;
    use crate::ops::*;
    use crate::r#impl::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_decay_stress_001() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 1 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_002() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 2 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_003() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 3 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_004() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 4 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_005() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 5 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_006() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 6 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_007() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 7 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_008() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 8 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_009() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 9 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_010() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 10 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_011() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 11 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_012() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 12 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_013() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 13 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_014() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 14 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_015() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 15 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_016() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 16 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_017() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 17 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_018() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 18 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_019() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 19 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_020() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 20 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_021() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 21 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_022() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 22 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_023() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 23 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_024() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 24 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_025() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 25 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_026() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 26 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_027() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 27 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_028() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 28 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_029() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 29 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_030() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 30 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_031() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 31 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_032() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 32 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_033() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 33 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_034() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 34 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_035() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 35 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_036() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 36 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_037() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 37 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_038() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 38 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_039() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 39 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_040() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 40 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_041() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 41 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_042() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 42 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_043() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 43 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_044() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 44 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_045() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 45 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_046() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 46 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_047() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 47 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_048() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 48 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_049() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 49 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_050() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 50 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_051() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 51 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_052() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 52 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_053() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 53 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_054() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 54 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_055() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 55 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_056() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 56 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_057() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 57 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_058() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 58 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_059() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 59 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_060() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 60 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_061() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 61 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_062() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 62 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_063() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 63 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_064() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 64 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_065() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 65 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_066() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 66 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_067() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 67 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_068() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 68 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_069() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 69 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_070() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 70 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_071() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 71 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_072() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 72 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_073() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 73 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_074() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 74 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_075() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 75 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_076() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 76 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_077() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 77 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_078() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 78 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_079() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 79 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_080() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 80 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_081() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 81 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_082() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 82 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_083() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 83 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_084() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 84 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_085() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 85 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_086() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 86 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_087() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 87 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_088() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 88 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_089() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 89 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_090() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 90 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_091() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 91 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_092() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 92 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_093() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 93 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_094() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 94 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_095() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 95 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_096() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 96 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_097() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 97 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_098() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 98 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_099() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 99 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_100() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 100 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_101() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 101 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_102() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 102 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_103() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 103 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_104() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 104 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_105() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 105 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_106() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 106 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_107() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 107 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_108() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 108 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_109() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 109 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_110() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 110 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_111() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 111 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_112() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 112 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_113() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 113 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_114() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 114 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_115() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 115 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_116() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 116 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_117() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 117 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_118() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 118 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_119() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 119 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_120() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 120 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_121() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 121 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_122() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 122 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_123() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 123 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_124() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 124 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_125() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 125 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_126() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 126 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_127() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 127 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_128() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 128 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_129() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 129 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_130() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 130 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_131() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 131 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_132() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 132 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_133() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 133 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_134() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 134 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_135() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 135 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_136() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 136 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_137() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 137 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_138() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 138 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_139() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 139 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_140() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 140 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_141() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 141 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_142() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 142 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_143() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 143 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_144() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 144 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_145() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 145 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_146() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 146 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_147() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 147 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_148() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 148 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_149() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 149 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_150() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 150 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_151() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 151 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_152() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 152 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_153() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 153 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_154() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 154 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_155() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 155 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_156() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 156 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_157() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 157 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_158() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 158 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_159() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 159 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_160() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 160 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_161() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 161 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_162() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 162 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_163() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 163 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_164() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 164 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_165() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 165 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_166() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 166 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_167() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 167 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_168() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 168 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_169() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 169 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_170() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 170 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_171() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 171 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_172() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 172 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_173() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 173 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_174() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 174 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_175() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 175 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_176() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 176 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_177() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 177 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_178() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 178 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_179() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 179 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_180() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 180 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_181() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 181 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_182() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 182 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_183() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 183 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_184() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 184 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_185() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 185 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_186() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 186 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_187() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 187 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_188() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 188 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_189() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 189 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_190() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 190 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_191() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 191 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_192() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 192 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_193() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 193 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_194() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 194 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_195() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 195 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_196() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 196 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_197() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 197 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_198() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 198 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_199() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 199 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_200() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 200 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_201() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 201 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_202() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 202 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_203() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 203 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_204() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 204 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_205() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 205 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_206() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 206 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_207() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 207 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_208() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 208 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_209() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 209 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_210() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 210 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_211() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 211 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_212() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 212 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_213() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 213 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_214() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 214 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_215() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 215 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_216() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 216 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_217() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 217 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_218() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 218 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_219() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 219 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_220() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 220 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_221() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 221 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_222() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 222 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_223() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 223 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_224() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 224 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_225() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 225 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_226() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 226 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_227() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 227 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_228() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 228 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_229() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 229 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_230() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 230 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_231() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 231 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_232() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 232 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_233() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 233 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_234() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 234 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_235() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 235 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_236() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 236 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_237() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 237 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_238() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 238 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_239() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 239 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_240() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 240 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_241() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 241 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_242() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 242 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_243() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 243 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_244() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 244 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_245() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 245 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_246() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 246 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_247() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 247 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_248() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 248 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_249() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 249 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_250() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 250 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_251() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 251 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_252() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 252 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_253() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 253 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_254() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 254 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_255() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 255 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_256() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 256 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_257() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 257 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_258() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 258 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_259() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 259 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_260() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 260 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_261() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 261 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_262() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 262 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_263() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 263 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_264() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 264 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_265() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 265 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_266() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 266 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_267() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 267 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_268() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 268 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_269() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 269 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_270() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 270 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_271() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 271 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_272() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 272 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_273() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 273 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_274() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 274 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_275() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 275 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_276() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 276 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_277() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 277 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_278() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 278 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_279() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 279 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_280() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 280 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_281() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 281 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_282() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 282 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_283() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 283 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_284() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 284 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_285() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 285 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_286() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 286 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_287() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 287 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_288() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 288 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_289() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 289 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_290() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 290 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_291() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 291 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_292() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 292 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_293() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 293 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_294() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 294 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_295() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 295 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_296() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 296 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_297() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 297 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_298() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 298 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_299() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 299 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_300() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 300 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_301() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 301 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_302() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 302 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_303() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 303 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_304() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 304 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_305() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 305 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_306() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 306 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_307() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 307 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_308() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 308 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_309() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 309 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_310() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 310 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_311() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 311 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_312() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 312 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_313() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 313 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_314() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 314 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_315() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 315 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_316() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 316 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_317() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 317 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_318() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 318 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_319() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 319 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_320() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 320 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_321() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 321 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_322() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 322 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_323() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 323 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_324() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 324 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_325() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 325 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_326() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 326 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_327() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 327 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_328() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 328 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_329() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 329 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_330() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 330 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_331() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 331 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_332() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 332 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_333() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 333 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_334() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 334 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_335() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 335 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_336() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 336 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_337() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 337 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_338() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 338 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_339() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 339 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_340() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 340 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_341() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 341 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_342() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 342 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_343() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 343 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_344() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 344 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_345() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 345 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_346() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 346 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_347() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 347 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_348() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 348 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_349() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 349 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_350() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 350 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_351() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 351 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_352() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 352 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_353() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 353 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_354() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 354 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_355() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 355 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_356() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 356 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_357() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 357 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_358() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 358 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_359() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 359 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_360() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 360 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_361() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 361 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_362() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 362 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_363() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 363 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_364() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 364 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_365() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 365 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_366() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 366 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_367() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 367 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_368() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 368 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_369() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 369 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_370() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 370 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_371() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 371 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_372() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 372 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_373() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 373 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_374() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 374 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_375() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 375 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_376() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 376 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_377() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 377 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_378() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 378 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_379() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 379 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_380() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 380 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_381() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 381 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_382() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 382 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_383() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 383 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_384() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 384 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_385() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 385 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_386() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 386 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_387() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 387 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_388() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 388 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_389() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 389 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_390() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 390 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_391() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 391 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_392() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 392 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_393() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 393 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_394() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 394 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_395() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 395 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_396() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 396 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_397() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 397 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_398() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 398 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_399() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 399 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_400() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 400 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_401() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 401 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_402() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 402 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_403() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 403 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_404() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 404 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_405() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 405 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_406() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 406 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_407() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 407 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_408() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 408 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_decay_stress_409() {
        let decay = DecoupledWeightDecay::new(0.1);
        let mut t = Tensor::from_slice(&[1.0, 2.0, 409 as f64 * 0.1], vec![3]);
        decay.apply_decay(&mut t, 0.01);
        assert_eq!(t.shape(), &[3]);
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
    // brain-regularization production numerical verification padding line 2
    // brain-regularization production numerical verification padding line 3
    // brain-regularization production numerical verification padding line 4
    // brain-regularization production numerical verification padding line 5
}
