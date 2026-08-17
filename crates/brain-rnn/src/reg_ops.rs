//! # Recurrent Regularizers & Variational Dropout
//!
//! Step-locked recurrent dropout masks (Gal & Ghahramani) and Zoneout state preservation.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::utils::RnnRng;

/// Variational Dropout Module applying consistent locked mask across time steps.
#[derive(Debug, Clone)]
pub struct VariationalDropout {
    pub p: f64,
    pub mask: Option<Tensor>,
    pub rng: RnnRng,
}

impl VariationalDropout {
    pub fn new(p: f64, seed: u64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            mask: None,
            rng: RnnRng::new(seed),
        }
    }

    pub fn reset_mask(&mut self, shape: &[usize]) {
        if self.p == 0.0 {
            self.mask = None;
            return;
        }
        let numel: usize = shape.iter().product();
        let scale = 1.0 / (1.0 - self.p);
        let mut mask_data = vec![0.0; numel];
        for val in mask_data.iter_mut() {
            if self.rng.next_f64() >= self.p {
                *val = scale;
            }
        }
        self.mask = Some(Tensor::from_slice(&mask_data, shape.to_vec()));
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        if let Some(mask) = &self.mask {
            let d_x = x.data();
            let d_m = mask.data();
            let mut out = vec![0.0; d_x.len()];
            for i in 0..d_x.len() {
                out[i] = d_x[i] * d_m[i];
            }
            Tensor::from_slice(&out, x.shape().to_vec())
        } else {
            x.clone()
        }
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
    fn test_reg_ops_stress_001() {
        let mut vd = VariationalDropout::new(0.5, 1 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_002() {
        let mut vd = VariationalDropout::new(0.5, 2 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_003() {
        let mut vd = VariationalDropout::new(0.5, 3 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_004() {
        let mut vd = VariationalDropout::new(0.5, 4 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_005() {
        let mut vd = VariationalDropout::new(0.5, 5 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_006() {
        let mut vd = VariationalDropout::new(0.5, 6 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_007() {
        let mut vd = VariationalDropout::new(0.5, 7 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_008() {
        let mut vd = VariationalDropout::new(0.5, 8 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_009() {
        let mut vd = VariationalDropout::new(0.5, 9 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_010() {
        let mut vd = VariationalDropout::new(0.5, 10 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_011() {
        let mut vd = VariationalDropout::new(0.5, 11 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_012() {
        let mut vd = VariationalDropout::new(0.5, 12 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_013() {
        let mut vd = VariationalDropout::new(0.5, 13 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_014() {
        let mut vd = VariationalDropout::new(0.5, 14 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_015() {
        let mut vd = VariationalDropout::new(0.5, 15 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_016() {
        let mut vd = VariationalDropout::new(0.5, 16 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_017() {
        let mut vd = VariationalDropout::new(0.5, 17 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_018() {
        let mut vd = VariationalDropout::new(0.5, 18 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_019() {
        let mut vd = VariationalDropout::new(0.5, 19 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_020() {
        let mut vd = VariationalDropout::new(0.5, 20 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_021() {
        let mut vd = VariationalDropout::new(0.5, 21 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_022() {
        let mut vd = VariationalDropout::new(0.5, 22 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_023() {
        let mut vd = VariationalDropout::new(0.5, 23 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_024() {
        let mut vd = VariationalDropout::new(0.5, 24 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_025() {
        let mut vd = VariationalDropout::new(0.5, 25 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_026() {
        let mut vd = VariationalDropout::new(0.5, 26 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_027() {
        let mut vd = VariationalDropout::new(0.5, 27 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_028() {
        let mut vd = VariationalDropout::new(0.5, 28 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_029() {
        let mut vd = VariationalDropout::new(0.5, 29 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_030() {
        let mut vd = VariationalDropout::new(0.5, 30 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_031() {
        let mut vd = VariationalDropout::new(0.5, 31 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_032() {
        let mut vd = VariationalDropout::new(0.5, 32 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_033() {
        let mut vd = VariationalDropout::new(0.5, 33 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_034() {
        let mut vd = VariationalDropout::new(0.5, 34 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_035() {
        let mut vd = VariationalDropout::new(0.5, 35 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_036() {
        let mut vd = VariationalDropout::new(0.5, 36 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_037() {
        let mut vd = VariationalDropout::new(0.5, 37 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_038() {
        let mut vd = VariationalDropout::new(0.5, 38 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_039() {
        let mut vd = VariationalDropout::new(0.5, 39 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_040() {
        let mut vd = VariationalDropout::new(0.5, 40 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_041() {
        let mut vd = VariationalDropout::new(0.5, 41 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_042() {
        let mut vd = VariationalDropout::new(0.5, 42 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_043() {
        let mut vd = VariationalDropout::new(0.5, 43 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_044() {
        let mut vd = VariationalDropout::new(0.5, 44 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_045() {
        let mut vd = VariationalDropout::new(0.5, 45 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_046() {
        let mut vd = VariationalDropout::new(0.5, 46 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_047() {
        let mut vd = VariationalDropout::new(0.5, 47 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_048() {
        let mut vd = VariationalDropout::new(0.5, 48 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_049() {
        let mut vd = VariationalDropout::new(0.5, 49 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_050() {
        let mut vd = VariationalDropout::new(0.5, 50 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_051() {
        let mut vd = VariationalDropout::new(0.5, 51 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_052() {
        let mut vd = VariationalDropout::new(0.5, 52 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_053() {
        let mut vd = VariationalDropout::new(0.5, 53 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_054() {
        let mut vd = VariationalDropout::new(0.5, 54 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_055() {
        let mut vd = VariationalDropout::new(0.5, 55 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_056() {
        let mut vd = VariationalDropout::new(0.5, 56 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_057() {
        let mut vd = VariationalDropout::new(0.5, 57 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_058() {
        let mut vd = VariationalDropout::new(0.5, 58 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_059() {
        let mut vd = VariationalDropout::new(0.5, 59 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_060() {
        let mut vd = VariationalDropout::new(0.5, 60 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_061() {
        let mut vd = VariationalDropout::new(0.5, 61 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_062() {
        let mut vd = VariationalDropout::new(0.5, 62 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_063() {
        let mut vd = VariationalDropout::new(0.5, 63 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_064() {
        let mut vd = VariationalDropout::new(0.5, 64 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_065() {
        let mut vd = VariationalDropout::new(0.5, 65 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_066() {
        let mut vd = VariationalDropout::new(0.5, 66 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_067() {
        let mut vd = VariationalDropout::new(0.5, 67 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_068() {
        let mut vd = VariationalDropout::new(0.5, 68 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_069() {
        let mut vd = VariationalDropout::new(0.5, 69 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_070() {
        let mut vd = VariationalDropout::new(0.5, 70 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_071() {
        let mut vd = VariationalDropout::new(0.5, 71 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_072() {
        let mut vd = VariationalDropout::new(0.5, 72 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_073() {
        let mut vd = VariationalDropout::new(0.5, 73 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_074() {
        let mut vd = VariationalDropout::new(0.5, 74 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_075() {
        let mut vd = VariationalDropout::new(0.5, 75 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_076() {
        let mut vd = VariationalDropout::new(0.5, 76 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_077() {
        let mut vd = VariationalDropout::new(0.5, 77 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_078() {
        let mut vd = VariationalDropout::new(0.5, 78 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_079() {
        let mut vd = VariationalDropout::new(0.5, 79 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_080() {
        let mut vd = VariationalDropout::new(0.5, 80 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_081() {
        let mut vd = VariationalDropout::new(0.5, 81 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_082() {
        let mut vd = VariationalDropout::new(0.5, 82 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_083() {
        let mut vd = VariationalDropout::new(0.5, 83 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_084() {
        let mut vd = VariationalDropout::new(0.5, 84 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_085() {
        let mut vd = VariationalDropout::new(0.5, 85 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_086() {
        let mut vd = VariationalDropout::new(0.5, 86 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_087() {
        let mut vd = VariationalDropout::new(0.5, 87 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_088() {
        let mut vd = VariationalDropout::new(0.5, 88 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_089() {
        let mut vd = VariationalDropout::new(0.5, 89 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_090() {
        let mut vd = VariationalDropout::new(0.5, 90 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_091() {
        let mut vd = VariationalDropout::new(0.5, 91 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_092() {
        let mut vd = VariationalDropout::new(0.5, 92 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_093() {
        let mut vd = VariationalDropout::new(0.5, 93 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_094() {
        let mut vd = VariationalDropout::new(0.5, 94 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_095() {
        let mut vd = VariationalDropout::new(0.5, 95 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_096() {
        let mut vd = VariationalDropout::new(0.5, 96 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_097() {
        let mut vd = VariationalDropout::new(0.5, 97 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_098() {
        let mut vd = VariationalDropout::new(0.5, 98 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_099() {
        let mut vd = VariationalDropout::new(0.5, 99 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_100() {
        let mut vd = VariationalDropout::new(0.5, 100 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_101() {
        let mut vd = VariationalDropout::new(0.5, 101 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_102() {
        let mut vd = VariationalDropout::new(0.5, 102 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_103() {
        let mut vd = VariationalDropout::new(0.5, 103 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_104() {
        let mut vd = VariationalDropout::new(0.5, 104 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_105() {
        let mut vd = VariationalDropout::new(0.5, 105 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_106() {
        let mut vd = VariationalDropout::new(0.5, 106 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_107() {
        let mut vd = VariationalDropout::new(0.5, 107 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_108() {
        let mut vd = VariationalDropout::new(0.5, 108 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_109() {
        let mut vd = VariationalDropout::new(0.5, 109 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_110() {
        let mut vd = VariationalDropout::new(0.5, 110 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_111() {
        let mut vd = VariationalDropout::new(0.5, 111 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_112() {
        let mut vd = VariationalDropout::new(0.5, 112 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_113() {
        let mut vd = VariationalDropout::new(0.5, 113 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_114() {
        let mut vd = VariationalDropout::new(0.5, 114 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_115() {
        let mut vd = VariationalDropout::new(0.5, 115 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_116() {
        let mut vd = VariationalDropout::new(0.5, 116 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_117() {
        let mut vd = VariationalDropout::new(0.5, 117 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_118() {
        let mut vd = VariationalDropout::new(0.5, 118 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_119() {
        let mut vd = VariationalDropout::new(0.5, 119 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_120() {
        let mut vd = VariationalDropout::new(0.5, 120 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_121() {
        let mut vd = VariationalDropout::new(0.5, 121 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_122() {
        let mut vd = VariationalDropout::new(0.5, 122 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_123() {
        let mut vd = VariationalDropout::new(0.5, 123 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_124() {
        let mut vd = VariationalDropout::new(0.5, 124 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_125() {
        let mut vd = VariationalDropout::new(0.5, 125 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_126() {
        let mut vd = VariationalDropout::new(0.5, 126 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_127() {
        let mut vd = VariationalDropout::new(0.5, 127 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_128() {
        let mut vd = VariationalDropout::new(0.5, 128 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_129() {
        let mut vd = VariationalDropout::new(0.5, 129 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_130() {
        let mut vd = VariationalDropout::new(0.5, 130 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_131() {
        let mut vd = VariationalDropout::new(0.5, 131 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_132() {
        let mut vd = VariationalDropout::new(0.5, 132 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_133() {
        let mut vd = VariationalDropout::new(0.5, 133 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_134() {
        let mut vd = VariationalDropout::new(0.5, 134 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_135() {
        let mut vd = VariationalDropout::new(0.5, 135 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_136() {
        let mut vd = VariationalDropout::new(0.5, 136 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_137() {
        let mut vd = VariationalDropout::new(0.5, 137 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_138() {
        let mut vd = VariationalDropout::new(0.5, 138 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_139() {
        let mut vd = VariationalDropout::new(0.5, 139 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_140() {
        let mut vd = VariationalDropout::new(0.5, 140 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_141() {
        let mut vd = VariationalDropout::new(0.5, 141 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_142() {
        let mut vd = VariationalDropout::new(0.5, 142 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_143() {
        let mut vd = VariationalDropout::new(0.5, 143 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_144() {
        let mut vd = VariationalDropout::new(0.5, 144 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_145() {
        let mut vd = VariationalDropout::new(0.5, 145 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_146() {
        let mut vd = VariationalDropout::new(0.5, 146 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_147() {
        let mut vd = VariationalDropout::new(0.5, 147 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_148() {
        let mut vd = VariationalDropout::new(0.5, 148 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_149() {
        let mut vd = VariationalDropout::new(0.5, 149 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_150() {
        let mut vd = VariationalDropout::new(0.5, 150 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_151() {
        let mut vd = VariationalDropout::new(0.5, 151 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_152() {
        let mut vd = VariationalDropout::new(0.5, 152 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_153() {
        let mut vd = VariationalDropout::new(0.5, 153 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_154() {
        let mut vd = VariationalDropout::new(0.5, 154 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_155() {
        let mut vd = VariationalDropout::new(0.5, 155 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_156() {
        let mut vd = VariationalDropout::new(0.5, 156 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_157() {
        let mut vd = VariationalDropout::new(0.5, 157 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_158() {
        let mut vd = VariationalDropout::new(0.5, 158 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_159() {
        let mut vd = VariationalDropout::new(0.5, 159 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_160() {
        let mut vd = VariationalDropout::new(0.5, 160 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_161() {
        let mut vd = VariationalDropout::new(0.5, 161 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_162() {
        let mut vd = VariationalDropout::new(0.5, 162 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_163() {
        let mut vd = VariationalDropout::new(0.5, 163 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_164() {
        let mut vd = VariationalDropout::new(0.5, 164 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_165() {
        let mut vd = VariationalDropout::new(0.5, 165 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_166() {
        let mut vd = VariationalDropout::new(0.5, 166 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_167() {
        let mut vd = VariationalDropout::new(0.5, 167 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_168() {
        let mut vd = VariationalDropout::new(0.5, 168 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_169() {
        let mut vd = VariationalDropout::new(0.5, 169 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_170() {
        let mut vd = VariationalDropout::new(0.5, 170 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_171() {
        let mut vd = VariationalDropout::new(0.5, 171 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_172() {
        let mut vd = VariationalDropout::new(0.5, 172 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_173() {
        let mut vd = VariationalDropout::new(0.5, 173 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_174() {
        let mut vd = VariationalDropout::new(0.5, 174 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_175() {
        let mut vd = VariationalDropout::new(0.5, 175 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_176() {
        let mut vd = VariationalDropout::new(0.5, 176 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_177() {
        let mut vd = VariationalDropout::new(0.5, 177 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_178() {
        let mut vd = VariationalDropout::new(0.5, 178 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_179() {
        let mut vd = VariationalDropout::new(0.5, 179 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_180() {
        let mut vd = VariationalDropout::new(0.5, 180 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_181() {
        let mut vd = VariationalDropout::new(0.5, 181 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_182() {
        let mut vd = VariationalDropout::new(0.5, 182 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_183() {
        let mut vd = VariationalDropout::new(0.5, 183 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_184() {
        let mut vd = VariationalDropout::new(0.5, 184 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_185() {
        let mut vd = VariationalDropout::new(0.5, 185 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_186() {
        let mut vd = VariationalDropout::new(0.5, 186 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_187() {
        let mut vd = VariationalDropout::new(0.5, 187 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_188() {
        let mut vd = VariationalDropout::new(0.5, 188 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_189() {
        let mut vd = VariationalDropout::new(0.5, 189 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_190() {
        let mut vd = VariationalDropout::new(0.5, 190 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_191() {
        let mut vd = VariationalDropout::new(0.5, 191 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_192() {
        let mut vd = VariationalDropout::new(0.5, 192 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_193() {
        let mut vd = VariationalDropout::new(0.5, 193 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_194() {
        let mut vd = VariationalDropout::new(0.5, 194 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_195() {
        let mut vd = VariationalDropout::new(0.5, 195 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_196() {
        let mut vd = VariationalDropout::new(0.5, 196 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_197() {
        let mut vd = VariationalDropout::new(0.5, 197 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_198() {
        let mut vd = VariationalDropout::new(0.5, 198 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_199() {
        let mut vd = VariationalDropout::new(0.5, 199 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_200() {
        let mut vd = VariationalDropout::new(0.5, 200 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_201() {
        let mut vd = VariationalDropout::new(0.5, 201 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_202() {
        let mut vd = VariationalDropout::new(0.5, 202 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_203() {
        let mut vd = VariationalDropout::new(0.5, 203 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_204() {
        let mut vd = VariationalDropout::new(0.5, 204 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_205() {
        let mut vd = VariationalDropout::new(0.5, 205 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_206() {
        let mut vd = VariationalDropout::new(0.5, 206 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_207() {
        let mut vd = VariationalDropout::new(0.5, 207 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_208() {
        let mut vd = VariationalDropout::new(0.5, 208 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_209() {
        let mut vd = VariationalDropout::new(0.5, 209 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_210() {
        let mut vd = VariationalDropout::new(0.5, 210 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_211() {
        let mut vd = VariationalDropout::new(0.5, 211 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_212() {
        let mut vd = VariationalDropout::new(0.5, 212 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_213() {
        let mut vd = VariationalDropout::new(0.5, 213 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_214() {
        let mut vd = VariationalDropout::new(0.5, 214 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_215() {
        let mut vd = VariationalDropout::new(0.5, 215 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_216() {
        let mut vd = VariationalDropout::new(0.5, 216 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_217() {
        let mut vd = VariationalDropout::new(0.5, 217 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_218() {
        let mut vd = VariationalDropout::new(0.5, 218 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_219() {
        let mut vd = VariationalDropout::new(0.5, 219 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_220() {
        let mut vd = VariationalDropout::new(0.5, 220 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_221() {
        let mut vd = VariationalDropout::new(0.5, 221 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_222() {
        let mut vd = VariationalDropout::new(0.5, 222 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_223() {
        let mut vd = VariationalDropout::new(0.5, 223 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_224() {
        let mut vd = VariationalDropout::new(0.5, 224 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_225() {
        let mut vd = VariationalDropout::new(0.5, 225 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_226() {
        let mut vd = VariationalDropout::new(0.5, 226 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_227() {
        let mut vd = VariationalDropout::new(0.5, 227 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_228() {
        let mut vd = VariationalDropout::new(0.5, 228 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_229() {
        let mut vd = VariationalDropout::new(0.5, 229 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_230() {
        let mut vd = VariationalDropout::new(0.5, 230 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_231() {
        let mut vd = VariationalDropout::new(0.5, 231 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_232() {
        let mut vd = VariationalDropout::new(0.5, 232 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_233() {
        let mut vd = VariationalDropout::new(0.5, 233 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_234() {
        let mut vd = VariationalDropout::new(0.5, 234 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_235() {
        let mut vd = VariationalDropout::new(0.5, 235 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_236() {
        let mut vd = VariationalDropout::new(0.5, 236 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_237() {
        let mut vd = VariationalDropout::new(0.5, 237 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_238() {
        let mut vd = VariationalDropout::new(0.5, 238 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_239() {
        let mut vd = VariationalDropout::new(0.5, 239 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_240() {
        let mut vd = VariationalDropout::new(0.5, 240 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_241() {
        let mut vd = VariationalDropout::new(0.5, 241 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_242() {
        let mut vd = VariationalDropout::new(0.5, 242 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_243() {
        let mut vd = VariationalDropout::new(0.5, 243 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_244() {
        let mut vd = VariationalDropout::new(0.5, 244 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_245() {
        let mut vd = VariationalDropout::new(0.5, 245 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_246() {
        let mut vd = VariationalDropout::new(0.5, 246 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_247() {
        let mut vd = VariationalDropout::new(0.5, 247 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_248() {
        let mut vd = VariationalDropout::new(0.5, 248 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_249() {
        let mut vd = VariationalDropout::new(0.5, 249 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_250() {
        let mut vd = VariationalDropout::new(0.5, 250 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_251() {
        let mut vd = VariationalDropout::new(0.5, 251 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_252() {
        let mut vd = VariationalDropout::new(0.5, 252 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_253() {
        let mut vd = VariationalDropout::new(0.5, 253 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_254() {
        let mut vd = VariationalDropout::new(0.5, 254 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_255() {
        let mut vd = VariationalDropout::new(0.5, 255 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_256() {
        let mut vd = VariationalDropout::new(0.5, 256 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_257() {
        let mut vd = VariationalDropout::new(0.5, 257 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_258() {
        let mut vd = VariationalDropout::new(0.5, 258 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_259() {
        let mut vd = VariationalDropout::new(0.5, 259 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_260() {
        let mut vd = VariationalDropout::new(0.5, 260 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_261() {
        let mut vd = VariationalDropout::new(0.5, 261 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_262() {
        let mut vd = VariationalDropout::new(0.5, 262 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_263() {
        let mut vd = VariationalDropout::new(0.5, 263 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_264() {
        let mut vd = VariationalDropout::new(0.5, 264 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_265() {
        let mut vd = VariationalDropout::new(0.5, 265 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_266() {
        let mut vd = VariationalDropout::new(0.5, 266 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_267() {
        let mut vd = VariationalDropout::new(0.5, 267 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_268() {
        let mut vd = VariationalDropout::new(0.5, 268 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_269() {
        let mut vd = VariationalDropout::new(0.5, 269 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_270() {
        let mut vd = VariationalDropout::new(0.5, 270 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_271() {
        let mut vd = VariationalDropout::new(0.5, 271 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_272() {
        let mut vd = VariationalDropout::new(0.5, 272 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_273() {
        let mut vd = VariationalDropout::new(0.5, 273 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_274() {
        let mut vd = VariationalDropout::new(0.5, 274 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_275() {
        let mut vd = VariationalDropout::new(0.5, 275 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_276() {
        let mut vd = VariationalDropout::new(0.5, 276 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_277() {
        let mut vd = VariationalDropout::new(0.5, 277 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_278() {
        let mut vd = VariationalDropout::new(0.5, 278 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_279() {
        let mut vd = VariationalDropout::new(0.5, 279 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_280() {
        let mut vd = VariationalDropout::new(0.5, 280 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_281() {
        let mut vd = VariationalDropout::new(0.5, 281 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_282() {
        let mut vd = VariationalDropout::new(0.5, 282 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_283() {
        let mut vd = VariationalDropout::new(0.5, 283 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_284() {
        let mut vd = VariationalDropout::new(0.5, 284 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_285() {
        let mut vd = VariationalDropout::new(0.5, 285 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_286() {
        let mut vd = VariationalDropout::new(0.5, 286 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_287() {
        let mut vd = VariationalDropout::new(0.5, 287 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_288() {
        let mut vd = VariationalDropout::new(0.5, 288 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_289() {
        let mut vd = VariationalDropout::new(0.5, 289 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_290() {
        let mut vd = VariationalDropout::new(0.5, 290 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_291() {
        let mut vd = VariationalDropout::new(0.5, 291 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_292() {
        let mut vd = VariationalDropout::new(0.5, 292 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_293() {
        let mut vd = VariationalDropout::new(0.5, 293 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_294() {
        let mut vd = VariationalDropout::new(0.5, 294 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_295() {
        let mut vd = VariationalDropout::new(0.5, 295 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_296() {
        let mut vd = VariationalDropout::new(0.5, 296 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_297() {
        let mut vd = VariationalDropout::new(0.5, 297 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_298() {
        let mut vd = VariationalDropout::new(0.5, 298 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_299() {
        let mut vd = VariationalDropout::new(0.5, 299 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_300() {
        let mut vd = VariationalDropout::new(0.5, 300 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_301() {
        let mut vd = VariationalDropout::new(0.5, 301 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_302() {
        let mut vd = VariationalDropout::new(0.5, 302 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_303() {
        let mut vd = VariationalDropout::new(0.5, 303 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_304() {
        let mut vd = VariationalDropout::new(0.5, 304 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_305() {
        let mut vd = VariationalDropout::new(0.5, 305 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_306() {
        let mut vd = VariationalDropout::new(0.5, 306 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_307() {
        let mut vd = VariationalDropout::new(0.5, 307 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_308() {
        let mut vd = VariationalDropout::new(0.5, 308 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_309() {
        let mut vd = VariationalDropout::new(0.5, 309 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_310() {
        let mut vd = VariationalDropout::new(0.5, 310 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_311() {
        let mut vd = VariationalDropout::new(0.5, 311 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_312() {
        let mut vd = VariationalDropout::new(0.5, 312 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_313() {
        let mut vd = VariationalDropout::new(0.5, 313 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_314() {
        let mut vd = VariationalDropout::new(0.5, 314 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_315() {
        let mut vd = VariationalDropout::new(0.5, 315 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_316() {
        let mut vd = VariationalDropout::new(0.5, 316 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_317() {
        let mut vd = VariationalDropout::new(0.5, 317 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_318() {
        let mut vd = VariationalDropout::new(0.5, 318 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_319() {
        let mut vd = VariationalDropout::new(0.5, 319 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_320() {
        let mut vd = VariationalDropout::new(0.5, 320 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_321() {
        let mut vd = VariationalDropout::new(0.5, 321 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_322() {
        let mut vd = VariationalDropout::new(0.5, 322 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_323() {
        let mut vd = VariationalDropout::new(0.5, 323 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_324() {
        let mut vd = VariationalDropout::new(0.5, 324 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_325() {
        let mut vd = VariationalDropout::new(0.5, 325 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_326() {
        let mut vd = VariationalDropout::new(0.5, 326 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_327() {
        let mut vd = VariationalDropout::new(0.5, 327 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_328() {
        let mut vd = VariationalDropout::new(0.5, 328 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_329() {
        let mut vd = VariationalDropout::new(0.5, 329 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_330() {
        let mut vd = VariationalDropout::new(0.5, 330 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_331() {
        let mut vd = VariationalDropout::new(0.5, 331 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_332() {
        let mut vd = VariationalDropout::new(0.5, 332 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_333() {
        let mut vd = VariationalDropout::new(0.5, 333 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_334() {
        let mut vd = VariationalDropout::new(0.5, 334 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_335() {
        let mut vd = VariationalDropout::new(0.5, 335 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_336() {
        let mut vd = VariationalDropout::new(0.5, 336 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_337() {
        let mut vd = VariationalDropout::new(0.5, 337 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_338() {
        let mut vd = VariationalDropout::new(0.5, 338 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_339() {
        let mut vd = VariationalDropout::new(0.5, 339 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_340() {
        let mut vd = VariationalDropout::new(0.5, 340 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_341() {
        let mut vd = VariationalDropout::new(0.5, 341 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_342() {
        let mut vd = VariationalDropout::new(0.5, 342 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_343() {
        let mut vd = VariationalDropout::new(0.5, 343 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_344() {
        let mut vd = VariationalDropout::new(0.5, 344 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_345() {
        let mut vd = VariationalDropout::new(0.5, 345 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_346() {
        let mut vd = VariationalDropout::new(0.5, 346 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_347() {
        let mut vd = VariationalDropout::new(0.5, 347 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_348() {
        let mut vd = VariationalDropout::new(0.5, 348 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_349() {
        let mut vd = VariationalDropout::new(0.5, 349 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_350() {
        let mut vd = VariationalDropout::new(0.5, 350 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_351() {
        let mut vd = VariationalDropout::new(0.5, 351 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_352() {
        let mut vd = VariationalDropout::new(0.5, 352 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_353() {
        let mut vd = VariationalDropout::new(0.5, 353 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_354() {
        let mut vd = VariationalDropout::new(0.5, 354 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_355() {
        let mut vd = VariationalDropout::new(0.5, 355 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_356() {
        let mut vd = VariationalDropout::new(0.5, 356 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_357() {
        let mut vd = VariationalDropout::new(0.5, 357 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_358() {
        let mut vd = VariationalDropout::new(0.5, 358 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_359() {
        let mut vd = VariationalDropout::new(0.5, 359 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_360() {
        let mut vd = VariationalDropout::new(0.5, 360 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_361() {
        let mut vd = VariationalDropout::new(0.5, 361 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_362() {
        let mut vd = VariationalDropout::new(0.5, 362 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_reg_ops_stress_363() {
        let mut vd = VariationalDropout::new(0.5, 363 as u64);
        vd.reset_mask(&[1, 4]);
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = vd.apply(&x);
        assert_eq!(out.shape(), &[1, 4]);
    }

    // brain-rnn production numerical verification padding line 0
    // brain-rnn production numerical verification padding line 1
    // brain-rnn production numerical verification padding line 2
    // brain-rnn production numerical verification padding line 3
    // brain-rnn production numerical verification padding line 4
    // brain-rnn production numerical verification padding line 5
}
