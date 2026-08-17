//! # Dropout Family Modules
//!
//! Standard inverted dropout, 2D spatial feature map dropout, 3D volumetric dropout, and fused operations.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod alpha;
pub mod adaptive;

use brain_core::Tensor;
use super::core::{RegError, RegKind, RegResult, Regularization};
use super::utils::XorShift64;

/// Inverted Dropout layer: zeroing elements with probability p and scaling remaining by 1 / (1 - p).
#[derive(Debug, Clone)]
pub struct Dropout {
    pub p: f64,
    pub is_training: bool,
    pub rng: XorShift64,
    pub last_mask: Option<Tensor>,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            is_training: true,
            rng: XorShift64::new(1337),
            last_mask: None,
        }
    }

    pub fn with_seed(p: f64, seed: u64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            is_training: true,
            rng: XorShift64::new(seed),
            last_mask: None,
        }
    }

    /// Computes fused dropout and residual addition.
    pub fn forward_add(&mut self, input: &Tensor, residual: &Tensor) -> RegResult<Tensor> {
        if input.shape() != residual.shape() {
            return Err(RegError::ShapeMismatch {
                expected: input.shape().to_vec(),
                found: residual.shape().to_vec(),
            });
        }
        let dropped = self.apply(input)?;
        let mut out = dropped.clone();
        let out_data = out.data_mut();
        let res_data = residual.data();
        for i in 0..out_data.len() {
            out_data[i] += res_data[i];
        }
        Ok(out)
    }
}

impl Regularization for Dropout {
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor> {
        if !self.is_training || self.p == 0.0 {
            return Ok(input.clone());
        }
        if self.p == 1.0 {
            return Ok(Tensor::zeros(input.shape().to_vec()));
        }

        let scale = 1.0 / (1.0 - self.p);
        let data = input.data();
        let n = data.len();
        let mut out_data = vec![0.0; n];
        let mut mask_data = vec![0.0; n];

        for i in 0..n {
            let r = self.rng.next_f64();
            if r >= self.p {
                mask_data[i] = 1.0;
                out_data[i] = data[i] * scale;
            }
        }

        self.last_mask = Some(Tensor::from_slice(&mask_data, input.shape().to_vec()));
        Ok(Tensor::from_slice(&out_data, input.shape().to_vec()))
    }

    fn train_mode(&mut self) {
        self.is_training = true;
    }

    fn eval_mode(&mut self) {
        self.is_training = false;
    }

    fn kind(&self) -> RegKind {
        RegKind::Dropout
    }
}

/// 2D Spatial Feature Map Dropout (zeroing entire channels independently).
#[derive(Debug, Clone)]
pub struct Dropout2d {
    pub p: f64,
    pub is_training: bool,
    pub rng: XorShift64,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            is_training: true,
            rng: XorShift64::new(42),
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> RegResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(RegError::ShapeMismatch {
                expected: vec![1, 1, 1, 1],
                found: shape.to_vec(),
            });
        }
        if !self.is_training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let batch_size = shape[0];
        let num_channels = shape[1];
        let spatial_size = shape[2] * shape[3];
        let scale = 1.0 / (1.0 - self.p);

        let data = input.data();
        let mut out_data = vec![0.0; data.len()];

        for b in 0..batch_size {
            for c in 0..num_channels {
                let keep = self.rng.next_f64() >= self.p;
                if keep {
                    let start = (b * num_channels + c) * spatial_size;
                    let end = start + spatial_size;
                    for i in start..end {
                        out_data[i] = data[i] * scale;
                    }
                }
            }
        }

        Ok(Tensor::from_slice(&out_data, shape.to_vec()))
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
    fn test_dropout_stress_001() {
        let mut drop = Dropout::with_seed(0.5, 1 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_002() {
        let mut drop = Dropout::with_seed(0.5, 2 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_003() {
        let mut drop = Dropout::with_seed(0.5, 3 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_004() {
        let mut drop = Dropout::with_seed(0.5, 4 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_005() {
        let mut drop = Dropout::with_seed(0.5, 5 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_006() {
        let mut drop = Dropout::with_seed(0.5, 6 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_007() {
        let mut drop = Dropout::with_seed(0.5, 7 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_008() {
        let mut drop = Dropout::with_seed(0.5, 8 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_009() {
        let mut drop = Dropout::with_seed(0.5, 9 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_010() {
        let mut drop = Dropout::with_seed(0.5, 10 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_011() {
        let mut drop = Dropout::with_seed(0.5, 11 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_012() {
        let mut drop = Dropout::with_seed(0.5, 12 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_013() {
        let mut drop = Dropout::with_seed(0.5, 13 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_014() {
        let mut drop = Dropout::with_seed(0.5, 14 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_015() {
        let mut drop = Dropout::with_seed(0.5, 15 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_016() {
        let mut drop = Dropout::with_seed(0.5, 16 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_017() {
        let mut drop = Dropout::with_seed(0.5, 17 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_018() {
        let mut drop = Dropout::with_seed(0.5, 18 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_019() {
        let mut drop = Dropout::with_seed(0.5, 19 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_020() {
        let mut drop = Dropout::with_seed(0.5, 20 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_021() {
        let mut drop = Dropout::with_seed(0.5, 21 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_022() {
        let mut drop = Dropout::with_seed(0.5, 22 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_023() {
        let mut drop = Dropout::with_seed(0.5, 23 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_024() {
        let mut drop = Dropout::with_seed(0.5, 24 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_025() {
        let mut drop = Dropout::with_seed(0.5, 25 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_026() {
        let mut drop = Dropout::with_seed(0.5, 26 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_027() {
        let mut drop = Dropout::with_seed(0.5, 27 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_028() {
        let mut drop = Dropout::with_seed(0.5, 28 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_029() {
        let mut drop = Dropout::with_seed(0.5, 29 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_030() {
        let mut drop = Dropout::with_seed(0.5, 30 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_031() {
        let mut drop = Dropout::with_seed(0.5, 31 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_032() {
        let mut drop = Dropout::with_seed(0.5, 32 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_033() {
        let mut drop = Dropout::with_seed(0.5, 33 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_034() {
        let mut drop = Dropout::with_seed(0.5, 34 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_035() {
        let mut drop = Dropout::with_seed(0.5, 35 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_036() {
        let mut drop = Dropout::with_seed(0.5, 36 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_037() {
        let mut drop = Dropout::with_seed(0.5, 37 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_038() {
        let mut drop = Dropout::with_seed(0.5, 38 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_039() {
        let mut drop = Dropout::with_seed(0.5, 39 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_040() {
        let mut drop = Dropout::with_seed(0.5, 40 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_041() {
        let mut drop = Dropout::with_seed(0.5, 41 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_042() {
        let mut drop = Dropout::with_seed(0.5, 42 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_043() {
        let mut drop = Dropout::with_seed(0.5, 43 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_044() {
        let mut drop = Dropout::with_seed(0.5, 44 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_045() {
        let mut drop = Dropout::with_seed(0.5, 45 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_046() {
        let mut drop = Dropout::with_seed(0.5, 46 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_047() {
        let mut drop = Dropout::with_seed(0.5, 47 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_048() {
        let mut drop = Dropout::with_seed(0.5, 48 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_049() {
        let mut drop = Dropout::with_seed(0.5, 49 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_050() {
        let mut drop = Dropout::with_seed(0.5, 50 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_051() {
        let mut drop = Dropout::with_seed(0.5, 51 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_052() {
        let mut drop = Dropout::with_seed(0.5, 52 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_053() {
        let mut drop = Dropout::with_seed(0.5, 53 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_054() {
        let mut drop = Dropout::with_seed(0.5, 54 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_055() {
        let mut drop = Dropout::with_seed(0.5, 55 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_056() {
        let mut drop = Dropout::with_seed(0.5, 56 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_057() {
        let mut drop = Dropout::with_seed(0.5, 57 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_058() {
        let mut drop = Dropout::with_seed(0.5, 58 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_059() {
        let mut drop = Dropout::with_seed(0.5, 59 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_060() {
        let mut drop = Dropout::with_seed(0.5, 60 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_061() {
        let mut drop = Dropout::with_seed(0.5, 61 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_062() {
        let mut drop = Dropout::with_seed(0.5, 62 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_063() {
        let mut drop = Dropout::with_seed(0.5, 63 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_064() {
        let mut drop = Dropout::with_seed(0.5, 64 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_065() {
        let mut drop = Dropout::with_seed(0.5, 65 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_066() {
        let mut drop = Dropout::with_seed(0.5, 66 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_067() {
        let mut drop = Dropout::with_seed(0.5, 67 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_068() {
        let mut drop = Dropout::with_seed(0.5, 68 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_069() {
        let mut drop = Dropout::with_seed(0.5, 69 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_070() {
        let mut drop = Dropout::with_seed(0.5, 70 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_071() {
        let mut drop = Dropout::with_seed(0.5, 71 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_072() {
        let mut drop = Dropout::with_seed(0.5, 72 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_073() {
        let mut drop = Dropout::with_seed(0.5, 73 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_074() {
        let mut drop = Dropout::with_seed(0.5, 74 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_075() {
        let mut drop = Dropout::with_seed(0.5, 75 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_076() {
        let mut drop = Dropout::with_seed(0.5, 76 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_077() {
        let mut drop = Dropout::with_seed(0.5, 77 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_078() {
        let mut drop = Dropout::with_seed(0.5, 78 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_079() {
        let mut drop = Dropout::with_seed(0.5, 79 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_080() {
        let mut drop = Dropout::with_seed(0.5, 80 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_081() {
        let mut drop = Dropout::with_seed(0.5, 81 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_082() {
        let mut drop = Dropout::with_seed(0.5, 82 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_083() {
        let mut drop = Dropout::with_seed(0.5, 83 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_084() {
        let mut drop = Dropout::with_seed(0.5, 84 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_085() {
        let mut drop = Dropout::with_seed(0.5, 85 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_086() {
        let mut drop = Dropout::with_seed(0.5, 86 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_087() {
        let mut drop = Dropout::with_seed(0.5, 87 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_088() {
        let mut drop = Dropout::with_seed(0.5, 88 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_089() {
        let mut drop = Dropout::with_seed(0.5, 89 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_090() {
        let mut drop = Dropout::with_seed(0.5, 90 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_091() {
        let mut drop = Dropout::with_seed(0.5, 91 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_092() {
        let mut drop = Dropout::with_seed(0.5, 92 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_093() {
        let mut drop = Dropout::with_seed(0.5, 93 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_094() {
        let mut drop = Dropout::with_seed(0.5, 94 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_095() {
        let mut drop = Dropout::with_seed(0.5, 95 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_096() {
        let mut drop = Dropout::with_seed(0.5, 96 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_097() {
        let mut drop = Dropout::with_seed(0.5, 97 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_098() {
        let mut drop = Dropout::with_seed(0.5, 98 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_099() {
        let mut drop = Dropout::with_seed(0.5, 99 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_100() {
        let mut drop = Dropout::with_seed(0.5, 100 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_101() {
        let mut drop = Dropout::with_seed(0.5, 101 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_102() {
        let mut drop = Dropout::with_seed(0.5, 102 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_103() {
        let mut drop = Dropout::with_seed(0.5, 103 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_104() {
        let mut drop = Dropout::with_seed(0.5, 104 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_105() {
        let mut drop = Dropout::with_seed(0.5, 105 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_106() {
        let mut drop = Dropout::with_seed(0.5, 106 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_107() {
        let mut drop = Dropout::with_seed(0.5, 107 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_108() {
        let mut drop = Dropout::with_seed(0.5, 108 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_109() {
        let mut drop = Dropout::with_seed(0.5, 109 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_110() {
        let mut drop = Dropout::with_seed(0.5, 110 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_111() {
        let mut drop = Dropout::with_seed(0.5, 111 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_112() {
        let mut drop = Dropout::with_seed(0.5, 112 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_113() {
        let mut drop = Dropout::with_seed(0.5, 113 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_114() {
        let mut drop = Dropout::with_seed(0.5, 114 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_115() {
        let mut drop = Dropout::with_seed(0.5, 115 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_116() {
        let mut drop = Dropout::with_seed(0.5, 116 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_117() {
        let mut drop = Dropout::with_seed(0.5, 117 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_118() {
        let mut drop = Dropout::with_seed(0.5, 118 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_119() {
        let mut drop = Dropout::with_seed(0.5, 119 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_120() {
        let mut drop = Dropout::with_seed(0.5, 120 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_121() {
        let mut drop = Dropout::with_seed(0.5, 121 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_122() {
        let mut drop = Dropout::with_seed(0.5, 122 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_123() {
        let mut drop = Dropout::with_seed(0.5, 123 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_124() {
        let mut drop = Dropout::with_seed(0.5, 124 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_125() {
        let mut drop = Dropout::with_seed(0.5, 125 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_126() {
        let mut drop = Dropout::with_seed(0.5, 126 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_127() {
        let mut drop = Dropout::with_seed(0.5, 127 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_128() {
        let mut drop = Dropout::with_seed(0.5, 128 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_129() {
        let mut drop = Dropout::with_seed(0.5, 129 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_130() {
        let mut drop = Dropout::with_seed(0.5, 130 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_131() {
        let mut drop = Dropout::with_seed(0.5, 131 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_132() {
        let mut drop = Dropout::with_seed(0.5, 132 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_133() {
        let mut drop = Dropout::with_seed(0.5, 133 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_134() {
        let mut drop = Dropout::with_seed(0.5, 134 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_135() {
        let mut drop = Dropout::with_seed(0.5, 135 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_136() {
        let mut drop = Dropout::with_seed(0.5, 136 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_137() {
        let mut drop = Dropout::with_seed(0.5, 137 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_138() {
        let mut drop = Dropout::with_seed(0.5, 138 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_139() {
        let mut drop = Dropout::with_seed(0.5, 139 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_140() {
        let mut drop = Dropout::with_seed(0.5, 140 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_141() {
        let mut drop = Dropout::with_seed(0.5, 141 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_142() {
        let mut drop = Dropout::with_seed(0.5, 142 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_143() {
        let mut drop = Dropout::with_seed(0.5, 143 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_144() {
        let mut drop = Dropout::with_seed(0.5, 144 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_145() {
        let mut drop = Dropout::with_seed(0.5, 145 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_146() {
        let mut drop = Dropout::with_seed(0.5, 146 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_147() {
        let mut drop = Dropout::with_seed(0.5, 147 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_148() {
        let mut drop = Dropout::with_seed(0.5, 148 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_149() {
        let mut drop = Dropout::with_seed(0.5, 149 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_150() {
        let mut drop = Dropout::with_seed(0.5, 150 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_151() {
        let mut drop = Dropout::with_seed(0.5, 151 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_152() {
        let mut drop = Dropout::with_seed(0.5, 152 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_153() {
        let mut drop = Dropout::with_seed(0.5, 153 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_154() {
        let mut drop = Dropout::with_seed(0.5, 154 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_155() {
        let mut drop = Dropout::with_seed(0.5, 155 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_156() {
        let mut drop = Dropout::with_seed(0.5, 156 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_157() {
        let mut drop = Dropout::with_seed(0.5, 157 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_158() {
        let mut drop = Dropout::with_seed(0.5, 158 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_159() {
        let mut drop = Dropout::with_seed(0.5, 159 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_160() {
        let mut drop = Dropout::with_seed(0.5, 160 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_161() {
        let mut drop = Dropout::with_seed(0.5, 161 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_162() {
        let mut drop = Dropout::with_seed(0.5, 162 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_163() {
        let mut drop = Dropout::with_seed(0.5, 163 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_164() {
        let mut drop = Dropout::with_seed(0.5, 164 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_165() {
        let mut drop = Dropout::with_seed(0.5, 165 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_166() {
        let mut drop = Dropout::with_seed(0.5, 166 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_167() {
        let mut drop = Dropout::with_seed(0.5, 167 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_168() {
        let mut drop = Dropout::with_seed(0.5, 168 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_169() {
        let mut drop = Dropout::with_seed(0.5, 169 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_170() {
        let mut drop = Dropout::with_seed(0.5, 170 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_171() {
        let mut drop = Dropout::with_seed(0.5, 171 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_172() {
        let mut drop = Dropout::with_seed(0.5, 172 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_173() {
        let mut drop = Dropout::with_seed(0.5, 173 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_174() {
        let mut drop = Dropout::with_seed(0.5, 174 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_175() {
        let mut drop = Dropout::with_seed(0.5, 175 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_176() {
        let mut drop = Dropout::with_seed(0.5, 176 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_177() {
        let mut drop = Dropout::with_seed(0.5, 177 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_178() {
        let mut drop = Dropout::with_seed(0.5, 178 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_179() {
        let mut drop = Dropout::with_seed(0.5, 179 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_180() {
        let mut drop = Dropout::with_seed(0.5, 180 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_181() {
        let mut drop = Dropout::with_seed(0.5, 181 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_182() {
        let mut drop = Dropout::with_seed(0.5, 182 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_183() {
        let mut drop = Dropout::with_seed(0.5, 183 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_184() {
        let mut drop = Dropout::with_seed(0.5, 184 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_185() {
        let mut drop = Dropout::with_seed(0.5, 185 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_186() {
        let mut drop = Dropout::with_seed(0.5, 186 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_187() {
        let mut drop = Dropout::with_seed(0.5, 187 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_188() {
        let mut drop = Dropout::with_seed(0.5, 188 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_189() {
        let mut drop = Dropout::with_seed(0.5, 189 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_190() {
        let mut drop = Dropout::with_seed(0.5, 190 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_191() {
        let mut drop = Dropout::with_seed(0.5, 191 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_192() {
        let mut drop = Dropout::with_seed(0.5, 192 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_193() {
        let mut drop = Dropout::with_seed(0.5, 193 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_194() {
        let mut drop = Dropout::with_seed(0.5, 194 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_195() {
        let mut drop = Dropout::with_seed(0.5, 195 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_196() {
        let mut drop = Dropout::with_seed(0.5, 196 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_197() {
        let mut drop = Dropout::with_seed(0.5, 197 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_198() {
        let mut drop = Dropout::with_seed(0.5, 198 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_199() {
        let mut drop = Dropout::with_seed(0.5, 199 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_200() {
        let mut drop = Dropout::with_seed(0.5, 200 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_201() {
        let mut drop = Dropout::with_seed(0.5, 201 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_202() {
        let mut drop = Dropout::with_seed(0.5, 202 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_203() {
        let mut drop = Dropout::with_seed(0.5, 203 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_204() {
        let mut drop = Dropout::with_seed(0.5, 204 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_205() {
        let mut drop = Dropout::with_seed(0.5, 205 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_206() {
        let mut drop = Dropout::with_seed(0.5, 206 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_207() {
        let mut drop = Dropout::with_seed(0.5, 207 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_208() {
        let mut drop = Dropout::with_seed(0.5, 208 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_209() {
        let mut drop = Dropout::with_seed(0.5, 209 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_210() {
        let mut drop = Dropout::with_seed(0.5, 210 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_211() {
        let mut drop = Dropout::with_seed(0.5, 211 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_212() {
        let mut drop = Dropout::with_seed(0.5, 212 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_213() {
        let mut drop = Dropout::with_seed(0.5, 213 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_214() {
        let mut drop = Dropout::with_seed(0.5, 214 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_215() {
        let mut drop = Dropout::with_seed(0.5, 215 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_216() {
        let mut drop = Dropout::with_seed(0.5, 216 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_217() {
        let mut drop = Dropout::with_seed(0.5, 217 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_218() {
        let mut drop = Dropout::with_seed(0.5, 218 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_219() {
        let mut drop = Dropout::with_seed(0.5, 219 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_220() {
        let mut drop = Dropout::with_seed(0.5, 220 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_221() {
        let mut drop = Dropout::with_seed(0.5, 221 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_222() {
        let mut drop = Dropout::with_seed(0.5, 222 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_223() {
        let mut drop = Dropout::with_seed(0.5, 223 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_224() {
        let mut drop = Dropout::with_seed(0.5, 224 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_225() {
        let mut drop = Dropout::with_seed(0.5, 225 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_226() {
        let mut drop = Dropout::with_seed(0.5, 226 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_227() {
        let mut drop = Dropout::with_seed(0.5, 227 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_228() {
        let mut drop = Dropout::with_seed(0.5, 228 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_229() {
        let mut drop = Dropout::with_seed(0.5, 229 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_230() {
        let mut drop = Dropout::with_seed(0.5, 230 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_231() {
        let mut drop = Dropout::with_seed(0.5, 231 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_232() {
        let mut drop = Dropout::with_seed(0.5, 232 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_233() {
        let mut drop = Dropout::with_seed(0.5, 233 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_234() {
        let mut drop = Dropout::with_seed(0.5, 234 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_235() {
        let mut drop = Dropout::with_seed(0.5, 235 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_236() {
        let mut drop = Dropout::with_seed(0.5, 236 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_237() {
        let mut drop = Dropout::with_seed(0.5, 237 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_238() {
        let mut drop = Dropout::with_seed(0.5, 238 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_239() {
        let mut drop = Dropout::with_seed(0.5, 239 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_240() {
        let mut drop = Dropout::with_seed(0.5, 240 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_241() {
        let mut drop = Dropout::with_seed(0.5, 241 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_242() {
        let mut drop = Dropout::with_seed(0.5, 242 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_243() {
        let mut drop = Dropout::with_seed(0.5, 243 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_244() {
        let mut drop = Dropout::with_seed(0.5, 244 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_245() {
        let mut drop = Dropout::with_seed(0.5, 245 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_246() {
        let mut drop = Dropout::with_seed(0.5, 246 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_247() {
        let mut drop = Dropout::with_seed(0.5, 247 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_248() {
        let mut drop = Dropout::with_seed(0.5, 248 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_249() {
        let mut drop = Dropout::with_seed(0.5, 249 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_250() {
        let mut drop = Dropout::with_seed(0.5, 250 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_251() {
        let mut drop = Dropout::with_seed(0.5, 251 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_252() {
        let mut drop = Dropout::with_seed(0.5, 252 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_253() {
        let mut drop = Dropout::with_seed(0.5, 253 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_254() {
        let mut drop = Dropout::with_seed(0.5, 254 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_255() {
        let mut drop = Dropout::with_seed(0.5, 255 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_256() {
        let mut drop = Dropout::with_seed(0.5, 256 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_257() {
        let mut drop = Dropout::with_seed(0.5, 257 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_258() {
        let mut drop = Dropout::with_seed(0.5, 258 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_259() {
        let mut drop = Dropout::with_seed(0.5, 259 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_260() {
        let mut drop = Dropout::with_seed(0.5, 260 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_261() {
        let mut drop = Dropout::with_seed(0.5, 261 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_262() {
        let mut drop = Dropout::with_seed(0.5, 262 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_263() {
        let mut drop = Dropout::with_seed(0.5, 263 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }

    #[test]
    fn test_dropout_stress_264() {
        let mut drop = Dropout::with_seed(0.5, 264 as u64 + 1);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = drop.apply(&t).unwrap();
        assert_eq!(out.shape(), &[4]);

        drop.eval_mode();
        let eval_out = drop.apply(&t).unwrap();
        assert_eq!(eval_out.data(), t.data());
    }
}
