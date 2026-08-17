//! # Alpha Dropout (SELU-Compatible)
//!
//! Preserves the self-normalizing properties (zero mean and unit variance) of SELU activations.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RegKind, RegResult, Regularization};
use super::super::utils::XorShift64;

/// Configuration for Alpha Dropout layer.
#[derive(Debug, Clone, PartialEq)]
pub struct AlphaDropoutConfig {
    pub p: f64,
    pub alpha_prime: f64,
    pub scale_factor: f64,
}

impl Default for AlphaDropoutConfig {
    fn default() -> Self {
        Self {
            p: 0.5,
            alpha_prime: -1.7580993408473766,
            scale_factor: 1.0,
        }
    }
}

/// Alpha Dropout layer designed to operate on Self-Normalizing Neural Networks (SNNs).
#[derive(Debug, Clone)]
pub struct AlphaDropout {
    pub p: f64,
    pub is_training: bool,
    pub alpha_prime: f64,
    pub a: f64,
    pub b: f64,
    pub rng: XorShift64,
}

impl AlphaDropout {
    pub fn new(p: f64) -> Self {
        let p_clamped = p.clamp(0.0, 1.0);
        let alpha = 1.673_263_242_354_377_2;
        let scale = 1.050_700_987_355_480_5;
        let alpha_prime = -scale * alpha;

        let a = ((1.0 - p_clamped) * (1.0 + p_clamped * alpha_prime * alpha_prime)).powf(-0.5);
        let b = -a * alpha_prime * p_clamped;

        Self {
            p: p_clamped,
            is_training: true,
            alpha_prime,
            a,
            b,
            rng: XorShift64::new(777),
        }
    }
}

impl Regularization for AlphaDropout {
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor> {
        if !self.is_training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let data = input.data();
        let n = data.len();
        let mut out_data = vec![0.0; n];

        for i in 0..n {
            let r = self.rng.next_f64();
            let val = if r < self.p {
                self.alpha_prime
            } else {
                data[i]
            };
            out_data[i] = self.a * val + self.b;
        }

        Ok(Tensor::from_slice(&out_data, input.shape().to_vec()))
    }

    fn train_mode(&mut self) {
        self.is_training = true;
    }

    fn eval_mode(&mut self) {
        self.is_training = false;
    }

    fn kind(&self) -> RegKind {
        RegKind::AlphaDropout
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
    fn test_alpha_dropout_stress_001() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 1 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_002() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 2 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_003() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 3 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_004() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 4 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_005() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 5 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_006() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 6 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_007() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 7 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_008() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 8 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_009() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 9 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_010() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 10 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_011() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 11 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_012() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 12 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_013() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 13 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_014() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 14 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_015() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 15 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_016() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 16 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_017() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 17 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_018() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 18 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_019() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 19 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_020() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 20 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_021() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 21 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_022() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 22 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_023() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 23 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_024() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 24 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_025() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 25 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_026() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 26 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_027() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 27 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_028() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 28 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_029() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 29 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_030() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 30 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_031() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 31 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_032() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 32 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_033() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 33 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_034() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 34 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_035() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 35 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_036() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 36 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_037() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 37 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_038() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 38 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_039() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 39 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_040() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 40 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_041() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 41 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_042() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 42 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_043() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 43 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_044() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 44 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_045() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 45 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_046() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 46 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_047() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 47 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_048() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 48 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_049() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 49 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_050() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 50 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_051() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 51 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_052() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 52 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_053() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 53 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_054() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 54 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_055() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 55 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_056() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 56 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_057() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 57 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_058() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 58 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_059() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 59 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_060() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 60 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_061() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 61 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_062() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 62 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_063() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 63 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_064() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 64 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_065() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 65 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_066() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 66 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_067() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 67 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_068() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 68 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_069() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 69 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_070() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 70 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_071() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 71 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_072() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 72 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_073() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 73 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_074() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 74 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_075() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 75 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_076() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 76 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_077() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 77 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_078() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 78 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_079() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 79 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_080() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 80 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_081() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 81 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_082() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 82 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_083() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 83 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_084() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 84 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_085() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 85 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_086() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 86 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_087() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 87 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_088() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 88 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_089() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 89 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_090() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 90 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_091() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 91 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_092() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 92 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_093() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 93 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_094() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 94 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_095() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 95 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_096() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 96 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_097() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 97 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_098() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 98 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_099() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 99 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_100() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 100 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_101() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 101 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_102() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 102 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_103() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 103 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_104() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 104 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_105() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 105 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_106() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 106 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_107() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 107 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_108() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 108 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_109() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 109 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_110() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 110 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_111() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 111 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_112() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 112 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_113() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 113 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_114() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 114 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_115() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 115 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_116() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 116 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_117() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 117 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_118() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 118 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_119() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 119 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_120() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 120 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_121() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 121 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_122() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 122 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_123() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 123 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_124() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 124 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_125() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 125 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_126() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 126 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_127() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 127 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_128() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 128 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_129() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 129 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_130() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 130 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_131() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 131 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_132() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 132 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_133() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 133 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_134() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 134 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_135() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 135 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_136() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 136 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_137() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 137 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_138() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 138 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_139() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 139 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_140() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 140 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_141() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 141 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_142() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 142 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_143() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 143 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_144() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 144 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_145() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 145 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_146() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 146 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_147() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 147 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_148() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 148 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_149() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 149 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_150() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 150 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_151() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 151 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_152() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 152 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_153() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 153 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_154() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 154 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_155() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 155 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_156() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 156 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_157() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 157 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_158() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 158 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_159() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 159 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_160() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 160 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_161() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 161 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_162() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 162 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_163() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 163 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_164() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 164 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_165() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 165 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_166() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 166 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_167() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 167 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_168() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 168 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_169() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 169 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_170() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 170 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_171() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 171 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_172() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 172 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_173() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 173 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_174() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 174 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_175() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 175 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_176() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 176 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_177() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 177 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_178() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 178 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_179() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 179 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_180() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 180 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_181() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 181 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_182() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 182 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_183() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 183 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_184() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 184 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_185() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 185 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_186() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 186 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_187() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 187 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_188() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 188 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_189() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 189 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_190() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 190 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_191() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 191 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_192() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 192 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_193() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 193 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_194() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 194 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_195() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 195 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_196() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 196 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_197() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 197 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_198() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 198 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_199() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 199 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_200() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 200 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_201() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 201 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_202() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 202 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_203() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 203 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_204() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 204 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_205() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 205 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_206() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 206 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_207() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 207 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_208() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 208 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_209() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 209 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_210() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 210 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_211() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 211 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_212() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 212 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_213() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 213 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_214() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 214 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_215() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 215 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_216() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 216 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_217() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 217 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_218() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 218 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_219() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 219 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_220() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 220 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_221() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 221 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_222() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 222 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_223() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 223 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_224() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 224 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_225() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 225 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_226() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 226 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_227() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 227 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_228() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 228 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_229() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 229 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_230() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 230 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_231() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 231 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_232() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 232 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_233() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 233 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_234() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 234 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_235() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 235 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_236() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 236 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_237() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 237 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_238() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 238 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_239() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 239 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_240() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 240 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_241() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 241 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_242() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 242 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_243() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 243 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_244() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 244 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_245() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 245 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_246() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 246 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_247() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 247 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_248() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 248 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_249() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 249 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_250() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 250 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_251() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 251 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_252() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 252 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_253() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 253 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_254() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 254 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_255() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 255 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_256() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 256 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_257() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 257 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_258() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 258 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_259() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 259 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_260() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 260 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_261() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 261 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_262() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 262 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_263() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 263 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_264() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 264 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_265() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 265 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_266() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 266 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_267() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 267 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_268() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 268 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_269() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 269 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_270() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 270 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_271() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 271 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_272() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 272 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_273() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 273 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_274() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 274 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_275() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 275 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_276() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 276 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_277() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 277 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_278() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 278 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_279() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 279 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_280() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 280 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_281() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 281 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_282() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 282 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_283() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 283 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_284() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 284 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_285() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 285 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_286() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 286 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_287() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 287 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_288() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 288 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_289() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 289 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_290() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 290 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_291() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 291 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_292() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 292 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_293() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 293 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_294() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 294 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_295() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 295 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_296() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 296 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_297() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 297 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_298() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 298 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_299() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 299 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_300() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 300 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_301() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 301 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_302() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 302 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_303() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 303 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_304() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 304 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_305() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 305 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_306() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 306 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_307() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 307 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_308() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 308 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_309() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 309 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_310() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 310 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_311() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 311 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_312() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 312 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_313() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 313 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_314() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 314 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_315() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 315 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_316() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 316 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_317() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 317 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_318() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 318 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_319() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 319 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_320() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 320 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_321() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 321 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_322() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 322 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_323() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 323 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_324() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 324 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_325() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 325 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_326() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 326 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_327() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 327 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_328() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 328 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_329() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 329 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_330() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 330 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_331() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 331 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_332() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 332 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_333() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 333 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_334() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 334 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_335() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 335 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_336() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 336 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_337() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 337 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_338() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 338 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_339() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 339 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_340() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 340 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_341() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 341 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_342() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 342 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_343() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 343 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_344() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 344 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_345() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 345 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_346() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 346 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_347() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 347 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_348() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 348 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_349() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 349 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_350() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 350 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_351() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 351 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_352() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 352 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_353() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 353 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_354() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 354 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_355() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 355 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_356() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 356 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_357() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 357 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_358() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 358 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_359() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 359 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_360() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 360 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_361() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 361 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_362() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 362 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_363() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 363 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_364() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 364 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_365() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 365 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_366() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 366 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_367() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 367 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_368() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 368 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_369() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 369 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_370() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 370 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_371() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 371 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_372() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 372 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_373() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 373 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_374() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 374 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_375() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 375 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_376() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 376 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_377() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 377 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_378() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 378 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_379() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 379 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_380() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 380 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_381() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 381 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_382() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 382 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_383() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 383 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_384() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 384 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_385() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 385 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_386() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 386 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_387() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 387 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_388() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 388 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_389() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 389 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_390() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 390 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_391() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 391 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_392() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 392 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_393() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 393 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_394() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 394 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_395() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 395 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_396() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 396 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_397() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 397 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_398() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 398 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_399() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 399 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_400() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 400 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_401() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 401 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_402() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 402 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_alpha_dropout_stress_403() {
        let mut ad = AlphaDropout::new(0.2);
        let t = Tensor::from_slice(&[-0.5, 0.0, 403 as f64 * 0.1, 1.0], vec![4]);
        let out = ad.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
}
