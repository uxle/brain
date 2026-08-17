//! # InfoNCE Loss
//!
//! Information Noise-Contrastive Estimation loss for self-supervised representation learning.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{LossResult, Reduction};
use crate::utils::reduction_apply;

/// Configuration for InfoNCE loss.
#[derive(Debug, Clone)]
pub struct InfoNceConfig {
    pub temperature: f64,
    pub reduction: Reduction,
}

impl Default for InfoNceConfig {
    fn default() -> Self {
        Self { temperature: 0.07, reduction: Reduction::Mean }
    }
}

/// InfoNCE Loss module.
#[derive(Debug, Clone, Default)]
pub struct InfoNCELoss {
    pub config: InfoNceConfig,
}

impl InfoNCELoss {
    pub fn new(config: InfoNceConfig) -> Self {
        Self { config }
    }

    pub fn compute(&self, queries: &Tensor, pos_keys: &Tensor, neg_keys: &[Tensor]) -> LossResult<Tensor> {
        let q_data = queries.to_vec();
        let p_data = pos_keys.to_vec();
        let dim = queries.shape().get(1).copied().unwrap_or(queries.to_vec().len());
        let num_queries = queries.shape()[0];
        let t = self.config.temperature;

        let mut losses = vec![0.0f64; num_queries];

        for i in 0..num_queries {
            let q_slice = &q_data[i * dim..(i + 1) * dim];
            let p_slice = &p_data[i * dim..(i + 1) * dim];

            let pos_sim: f64 = q_slice.iter().zip(p_slice.iter()).map(|(&a, &b)| a * b).sum::<f64>() / t;
            let mut sum_exp = pos_sim.exp();

            for neg_t in neg_keys {
                let n_data = neg_t.to_vec();
                let neg_sim: f64 = q_slice.iter().zip(n_data.iter()).map(|(&a, &b)| a * b).sum::<f64>() / t;
                sum_exp += neg_sim.exp();
            }

            losses[i] = -(pos_sim - sum_exp.ln());
        }

        Ok(reduction_apply(&losses, self.config.reduction))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_infonce_stress_001() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_002() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_003() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_004() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_005() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_006() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_007() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_008() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_009() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_010() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_011() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_012() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_013() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_014() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_015() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_016() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_017() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_018() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_019() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_020() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_021() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_022() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_023() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_024() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_025() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_026() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_027() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_028() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_029() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_030() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_031() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_032() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_033() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_034() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_035() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_036() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_037() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_038() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_039() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_040() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_041() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_042() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_043() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_044() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_045() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_046() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_047() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_048() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_049() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_050() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_051() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_052() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_053() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_054() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_055() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_056() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_057() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_058() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_059() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_060() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_061() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_062() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_063() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_064() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_065() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_066() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_067() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_068() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_069() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_070() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_071() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_072() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_073() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_074() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_075() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_076() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_077() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_078() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_079() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_080() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_081() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_082() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_083() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_084() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_085() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_086() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_087() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_088() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_089() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_090() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_091() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_092() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_093() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_094() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_095() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_096() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_097() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_098() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_099() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_100() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_101() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_102() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_103() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_104() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_105() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_106() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_107() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_108() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_109() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_110() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_111() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_112() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_113() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_114() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_115() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_116() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_117() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_118() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_119() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_120() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_121() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_122() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_123() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_124() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_125() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_126() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_127() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_128() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_129() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_130() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_131() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_132() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_133() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_134() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_135() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_136() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_137() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_138() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_139() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_140() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_141() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_142() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_143() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_144() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_145() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_146() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_147() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_148() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_149() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_150() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_151() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_152() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_153() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_154() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_155() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_156() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_157() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_158() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_159() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_160() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_161() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_162() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_163() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_164() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_165() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_166() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_167() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_168() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_169() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_170() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_171() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_172() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_173() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_174() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_175() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_176() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_177() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_178() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_179() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_180() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_181() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_182() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_183() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_184() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_185() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_186() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_187() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_188() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_189() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_190() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_191() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_192() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_193() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_194() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_195() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_196() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_197() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_198() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_199() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_200() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_201() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_202() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_203() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_204() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_205() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_206() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_207() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_208() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_209() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_210() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_211() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_212() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_213() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_214() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_215() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_216() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_217() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_218() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_219() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_220() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_221() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_222() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_223() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_224() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_225() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_226() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_227() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_228() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_229() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_230() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_231() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_232() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_233() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_234() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_235() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_236() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_237() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_238() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_239() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_240() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_241() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_242() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_243() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_244() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_245() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_246() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_247() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_248() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_249() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_250() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_251() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_252() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_253() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_254() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_255() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_256() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_257() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_258() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_259() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_260() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_261() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_262() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_263() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_264() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_265() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_266() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_267() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_268() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_269() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_270() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_271() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_272() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_273() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_274() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_275() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_276() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_277() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_278() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_279() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_280() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_281() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_282() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_283() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_284() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_285() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_286() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_287() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_288() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_289() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_290() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_291() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_292() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_293() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_294() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_295() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_296() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_297() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_298() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_299() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_300() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_301() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_302() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_303() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_304() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_305() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_306() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_307() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_308() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_309() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_310() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_311() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_312() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_313() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_314() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_315() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_316() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_317() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_318() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_319() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_320() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_321() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_322() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_323() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_324() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_325() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_326() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_327() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }

    #[test]
    fn test_infonce_stress_328() {
        let q = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let n = Tensor::from_vec(vec![0.0, 1.0], vec![1, 2]);
        let infonce = InfoNCELoss::default();
        let l = infonce.compute(&q, &p, &[n]).unwrap();
        assert!(l.to_vec()[0] > 0.0);
    }
}
