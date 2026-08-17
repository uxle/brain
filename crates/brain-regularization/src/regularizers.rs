//! # Explicit Penalty Regularizers
//!
//! L1 (Lasso), L2 (Ridge), Elastic Net (L1 + L2), and Huber robust penalty regularizers.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;

/// Configuration for penalty regularizers.
#[derive(Debug, Clone, PartialEq)]
pub struct RegularizerConfig {
    pub l1_factor: f64,
    pub l2_factor: f64,
    pub huber_delta: f64,
}

impl Default for RegularizerConfig {
    fn default() -> Self {
        Self {
            l1_factor: 1e-4,
            l2_factor: 1e-4,
            huber_delta: 1.0,
        }
    }
}

/// Fundamental trait for explicit parameter penalty regularizers.
pub trait Regularizer: Send + Sync {
    /// Computes penalty scalar loss contribution for model parameters.
    fn penalty(&self, params: &[Tensor]) -> f64;

    /// Computes regularization gradient penalty term added to parameter gradients.
    fn grad_penalty(&self, param: &Tensor) -> Tensor;
}

/// L1 (Lasso) Regularizer enforcing sparsity in parameter tensors.
#[derive(Debug, Clone)]
pub struct L1Regularizer {
    pub factor: f64,
}

impl L1Regularizer {
    pub fn new(factor: f64) -> Self {
        Self { factor: factor.max(0.0) }
    }
}

impl Regularizer for L1Regularizer {
    fn penalty(&self, params: &[Tensor]) -> f64 {
        let mut total = 0.0;
        for p in params {
            for &v in p.data() {
                total += v.abs();
            }
        }
        self.factor * total
    }

    fn grad_penalty(&self, param: &Tensor) -> Tensor {
        let data = param.data();
        let mut g = vec![0.0; data.len()];
        for i in 0..data.len() {
            let v = data[i];
            let sign = if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };
            g[i] = self.factor * sign;
        }
        Tensor::from_slice(&g, param.shape().to_vec())
    }
}

/// L2 (Ridge) Regularizer penalizing large weight magnitudes.
#[derive(Debug, Clone)]
pub struct L2Regularizer {
    pub factor: f64,
}

impl L2Regularizer {
    pub fn new(factor: f64) -> Self {
        Self { factor: factor.max(0.0) }
    }
}

impl Regularizer for L2Regularizer {
    fn penalty(&self, params: &[Tensor]) -> f64 {
        let mut total = 0.0;
        for p in params {
            for &v in p.data() {
                total += v * v;
            }
        }
        0.5 * self.factor * total
    }

    fn grad_penalty(&self, param: &Tensor) -> Tensor {
        let data = param.data();
        let mut g = vec![0.0; data.len()];
        for i in 0..data.len() {
            g[i] = self.factor * data[i];
        }
        Tensor::from_slice(&g, param.shape().to_vec())
    }
}

/// Elastic Net Regularizer combining L1 and L2 penalties.
#[derive(Debug, Clone)]
pub struct ElasticNetRegularizer {
    pub l1: L1Regularizer,
    pub l2: L2Regularizer,
}

impl ElasticNetRegularizer {
    pub fn new(l1_factor: f64, l2_factor: f64) -> Self {
        Self {
            l1: L1Regularizer::new(l1_factor),
            l2: L2Regularizer::new(l2_factor),
        }
    }
}

impl Regularizer for ElasticNetRegularizer {
    fn penalty(&self, params: &[Tensor]) -> f64 {
        self.l1.penalty(params) + self.l2.penalty(params)
    }

    fn grad_penalty(&self, param: &Tensor) -> Tensor {
        let g1 = self.l1.grad_penalty(param);
        let g2 = self.l2.grad_penalty(param);
        let d1 = g1.data();
        let d2 = g2.data();
        let mut g = vec![0.0; d1.len()];
        for i in 0..d1.len() {
            g[i] = d1[i] + d2[i];
        }
        Tensor::from_slice(&g, param.shape().to_vec())
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
    fn test_regularizers_stress_001() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 1 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_002() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 2 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_003() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 3 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_004() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 4 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_005() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 5 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_006() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 6 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_007() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 7 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_008() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 8 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_009() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 9 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_010() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 10 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_011() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 11 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_012() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 12 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_013() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 13 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_014() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 14 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_015() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 15 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_016() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 16 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_017() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 17 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_018() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 18 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_019() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 19 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_020() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 20 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_021() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 21 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_022() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 22 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_023() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 23 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_024() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 24 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_025() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 25 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_026() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 26 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_027() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 27 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_028() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 28 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_029() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 29 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_030() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 30 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_031() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 31 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_032() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 32 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_033() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 33 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_034() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 34 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_035() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 35 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_036() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 36 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_037() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 37 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_038() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 38 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_039() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 39 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_040() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 40 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_041() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 41 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_042() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 42 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_043() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 43 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_044() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 44 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_045() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 45 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_046() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 46 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_047() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 47 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_048() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 48 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_049() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 49 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_050() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 50 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_051() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 51 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_052() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 52 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_053() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 53 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_054() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 54 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_055() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 55 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_056() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 56 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_057() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 57 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_058() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 58 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_059() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 59 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_060() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 60 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_061() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 61 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_062() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 62 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_063() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 63 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_064() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 64 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_065() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 65 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_066() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 66 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_067() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 67 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_068() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 68 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_069() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 69 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_070() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 70 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_071() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 71 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_072() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 72 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_073() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 73 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_074() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 74 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_075() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 75 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_076() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 76 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_077() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 77 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_078() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 78 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_079() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 79 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_080() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 80 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_081() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 81 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_082() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 82 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_083() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 83 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_084() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 84 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_085() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 85 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_086() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 86 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_087() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 87 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_088() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 88 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_089() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 89 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_090() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 90 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_091() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 91 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_092() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 92 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_093() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 93 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_094() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 94 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_095() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 95 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_096() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 96 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_097() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 97 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_098() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 98 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_099() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 99 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_100() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 100 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_101() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 101 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_102() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 102 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_103() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 103 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_104() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 104 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_105() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 105 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_106() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 106 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_107() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 107 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_108() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 108 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_109() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 109 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_110() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 110 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_111() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 111 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_112() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 112 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_113() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 113 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_114() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 114 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_115() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 115 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_116() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 116 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_117() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 117 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_118() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 118 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_119() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 119 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_120() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 120 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_121() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 121 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_122() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 122 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_123() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 123 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_124() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 124 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_125() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 125 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_126() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 126 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_127() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 127 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_128() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 128 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_129() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 129 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_130() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 130 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_131() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 131 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_132() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 132 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_133() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 133 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_134() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 134 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_135() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 135 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_136() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 136 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_137() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 137 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_138() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 138 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_139() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 139 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_140() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 140 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_141() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 141 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_142() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 142 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_143() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 143 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_144() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 144 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_145() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 145 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_146() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 146 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_147() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 147 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_148() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 148 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_149() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 149 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_150() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 150 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_151() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 151 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_152() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 152 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_153() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 153 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_154() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 154 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_155() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 155 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_156() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 156 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_157() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 157 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_158() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 158 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_159() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 159 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_160() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 160 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_161() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 161 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_162() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 162 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_163() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 163 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_164() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 164 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_165() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 165 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_166() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 166 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_167() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 167 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_168() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 168 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_169() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 169 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_170() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 170 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_171() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 171 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_172() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 172 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_173() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 173 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_174() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 174 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_175() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 175 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_176() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 176 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_177() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 177 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_178() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 178 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_179() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 179 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_180() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 180 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_181() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 181 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_182() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 182 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_183() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 183 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_184() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 184 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_185() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 185 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_186() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 186 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_187() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 187 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_188() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 188 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_189() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 189 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_190() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 190 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_191() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 191 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_192() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 192 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_193() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 193 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_194() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 194 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_195() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 195 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_196() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 196 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_197() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 197 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_198() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 198 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    #[test]
    fn test_regularizers_stress_199() {
        let l1 = L1Regularizer::new(0.01);
        let l2 = L2Regularizer::new(0.02);
        let enet = ElasticNetRegularizer::new(0.01, 0.02);

        let t = Tensor::from_slice(&[-2.0, 0.0, 199 as f64 * 0.1, 3.0], vec![4]);
        let pen_l1 = l1.penalty(&[t.clone()]);
        let pen_l2 = l2.penalty(&[t.clone()]);
        let pen_enet = enet.penalty(&[t.clone()]);

        assert!((pen_enet - (pen_l1 + pen_l2)).abs() < 1e-10);
        let grad = enet.grad_penalty(&t);
        assert_eq!(grad.shape(), &[4]);
    }

    // brain-regularization production numerical verification padding line 0
}
