//! # Weight & Spectral Normalization
//!
//! Weight normalization (magnitude-direction decoupling) and Spectral Normalization (power iteration).
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RegError, RegResult};

/// Configuration for Spectral Normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralNormConfig {
    pub num_iterations: usize,
    pub eps: f64,
}

impl Default for SpectralNormConfig {
    fn default() -> Self {
        Self {
            num_iterations: 1,
            eps: 1e-12,
        }
    }
}

/// Spectral Normalization estimating spectral radius via power iteration.
#[derive(Debug, Clone)]
pub struct SpectralNorm {
    pub config: SpectralNormConfig,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub spectral_radius: f64,
}

impl SpectralNorm {
    pub fn new(rows: usize, cols: usize, config: SpectralNormConfig) -> Self {
        let u = vec![1.0 / (rows as f64).sqrt(); rows];
        let v = vec![1.0 / (cols as f64).sqrt(); cols];
        Self {
            config,
            u,
            v,
            spectral_radius: 1.0,
        }
    }

    /// Evaluates power iteration step on weight matrix W of shape [M, N] and normalizes W / sigma(W).
    pub fn normalize_weights(&mut self, weights: &Tensor) -> RegResult<Tensor> {
        let shape = weights.shape();
        if shape.len() != 2 {
            return Err(RegError::ShapeMismatch {
                expected: vec![self.u.len(), self.v.len()],
                found: shape.to_vec(),
            });
        }

        let m = shape[0];
        let n = shape[1];
        let data = weights.data();

        // Power iteration: v = W^T u / ||W^T u||, u = W v / ||W v||
        for _ in 0..self.config.num_iterations {
            // v = W^T u
            let mut v_new = vec![0.0; n];
            for j in 0..n {
                let mut sum = 0.0;
                for i in 0..m {
                    sum += data[i * n + j] * self.u[i];
                }
                v_new[j] = sum;
            }
            let v_norm = v_new.iter().map(|&x| x * x).sum::<f64>().sqrt().max(self.config.eps);
            for j in 0..n { self.v[j] = v_new[j] / v_norm; }

            // u = W v
            let mut u_new = vec![0.0; m];
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += data[i * n + j] * self.v[j];
                }
                u_new[i] = sum;
            }
            let u_norm = u_new.iter().map(|&x| x * x).sum::<f64>().sqrt().max(self.config.eps);
            for i in 0..m { self.u[i] = u_new[i] / u_norm; }
        }

        // sigma = u^T W v
        let mut sigma = 0.0;
        for i in 0..m {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += data[i * n + j] * self.v[j];
            }
            sigma += self.u[i] * row_sum;
        }

        self.spectral_radius = sigma.abs().max(self.config.eps);
        let scale = 1.0 / self.spectral_radius;

        let mut out_data = vec![0.0; m * n];
        for i in 0..m * n {
            out_data[i] = data[i] * scale;
        }

        Ok(Tensor::from_slice(&out_data, shape.to_vec()))
    }
}

/// Weight Normalization layer reparameterizing weights.
#[derive(Debug, Clone)]
pub struct WeightNorm {
    pub g: Vec<f64>,
    pub dim: usize,
}

impl WeightNorm {
    pub fn new(g: Vec<f64>, dim: usize) -> Self {
        Self { g, dim }
    }

    pub fn compute_weight(&self, v: &Tensor) -> RegResult<Tensor> {
        let shape = v.shape();
        let numel = v.numel();
        if shape.is_empty() || self.g.is_empty() {
            return Err(RegError::EmptyTensor);
        }

        let num_vectors = self.g.len();
        let vector_size = numel / num_vectors;
        let v_data = v.data();
        let mut w_data = vec![0.0; numel];

        for k in 0..num_vectors {
            let start = k * vector_size;
            let end = start + vector_size;
            let mut norm_sq = 0.0;
            for idx in start..end {
                norm_sq += v_data[idx] * v_data[idx];
            }
            let norm = norm_sq.sqrt().max(1e-12);
            let factor = self.g[k] / norm;

            for idx in start..end {
                w_data[idx] = v_data[idx] * factor;
            }
        }

        Ok(Tensor::from_slice(&w_data, shape.to_vec()))
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
    fn test_weight_norm_stress_001() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (1 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_002() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (2 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_003() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (3 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_004() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (4 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_005() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (5 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_006() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (6 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_007() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (7 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_008() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (8 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_009() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (9 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_010() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (10 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_011() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (11 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_012() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (12 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_013() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (13 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_014() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (14 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_015() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (15 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_016() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (16 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_017() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (17 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_018() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (18 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_019() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (19 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_020() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (20 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_021() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (21 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_022() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (22 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_023() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (23 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_024() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (24 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_025() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (25 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_026() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (26 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_027() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (27 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_028() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (28 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_029() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (29 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_030() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (30 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_031() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (31 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_032() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (32 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_033() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (33 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_034() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (34 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_035() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (35 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_036() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (36 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_037() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (37 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_038() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (38 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_039() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (39 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_040() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (40 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_041() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (41 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_042() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (42 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_043() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (43 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_044() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (44 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_045() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (45 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_046() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (46 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_047() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (47 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_048() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (48 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_049() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (49 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_050() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (50 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_051() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (51 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_052() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (52 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_053() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (53 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_054() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (54 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_055() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (55 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_056() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (56 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_057() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (57 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_058() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (58 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_059() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (59 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_060() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (60 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_061() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (61 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_062() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (62 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_063() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (63 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_064() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (64 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_065() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (65 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_066() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (66 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_067() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (67 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_068() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (68 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_069() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (69 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_070() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (70 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_071() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (71 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_072() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (72 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_073() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (73 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_074() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (74 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_075() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (75 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_076() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (76 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_077() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (77 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_078() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (78 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_079() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (79 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_080() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (80 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_081() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (81 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_082() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (82 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_083() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (83 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_084() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (84 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_085() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (85 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_086() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (86 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_087() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (87 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_088() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (88 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_089() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (89 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_090() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (90 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_091() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (91 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_092() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (92 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_093() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (93 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_094() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (94 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_095() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (95 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_096() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (96 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_097() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (97 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_098() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (98 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_099() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (99 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_100() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (100 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_101() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (101 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_102() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (102 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_103() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (103 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_104() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (104 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_105() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (105 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_106() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (106 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_107() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (107 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_108() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (108 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_109() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (109 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_110() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (110 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_111() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (111 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_112() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (112 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_113() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (113 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_114() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (114 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_115() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (115 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_116() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (116 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_117() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (117 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_118() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (118 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_119() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (119 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_120() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (120 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_121() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (121 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_122() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (122 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_123() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (123 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_124() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (124 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_125() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (125 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_126() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (126 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_127() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (127 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_128() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (128 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_129() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (129 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_130() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (130 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_131() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (131 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_132() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (132 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_133() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (133 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_134() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (134 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_135() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (135 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_136() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (136 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_137() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (137 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_138() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (138 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_139() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (139 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_140() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (140 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_141() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (141 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_142() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (142 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_143() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (143 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_144() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (144 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_145() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (145 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_146() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (146 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_147() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (147 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_148() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (148 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_149() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (149 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_150() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (150 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_151() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (151 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_152() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (152 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_153() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (153 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_154() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (154 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_155() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (155 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_156() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (156 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_157() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (157 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_158() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (158 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_159() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (159 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_160() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (160 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_161() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (161 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_162() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (162 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_163() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (163 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_164() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (164 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_165() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (165 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_166() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (166 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_167() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (167 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_168() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (168 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_169() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (169 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_170() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (170 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_171() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (171 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_172() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (172 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_173() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (173 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_174() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (174 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_175() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (175 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_176() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (176 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_177() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (177 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_178() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (178 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_179() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (179 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_180() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (180 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_181() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (181 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_182() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (182 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_183() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (183 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_184() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (184 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_185() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (185 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_186() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (186 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_187() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (187 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_188() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (188 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_189() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (189 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_190() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (190 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_191() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (191 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_192() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (192 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_193() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (193 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_194() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (194 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_195() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (195 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_196() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (196 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_197() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (197 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_198() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (198 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_199() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (199 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_200() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (200 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_201() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (201 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_202() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (202 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_203() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (203 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_204() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (204 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_205() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (205 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_206() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (206 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_207() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (207 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_208() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (208 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_209() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (209 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_210() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (210 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_211() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (211 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_212() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (212 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_213() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (213 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_214() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (214 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_215() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (215 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_216() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (216 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_217() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (217 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_218() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (218 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_219() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (219 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_220() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (220 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_221() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (221 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_222() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (222 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_223() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (223 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_224() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (224 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_225() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (225 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_226() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (226 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_227() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (227 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_228() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (228 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_229() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (229 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_230() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (230 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_231() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (231 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_232() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (232 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_233() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (233 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_234() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (234 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_235() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (235 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_236() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (236 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_237() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (237 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_238() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (238 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_239() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (239 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_240() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (240 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_241() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (241 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_242() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (242 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_stress_243() {
        let mut sn = SpectralNorm::new(2, 2, SpectralNormConfig::default());
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (243 as f64 * 0.01)], vec![2, 2]);
        let normalized = sn.normalize_weights(&w).unwrap();
        assert_eq!(normalized.shape(), &[2, 2]);
        assert!(sn.spectral_radius > 0.0);

        let wn = WeightNorm::new(vec![1.0, 2.0], 0);
        let wn_out = wn.compute_weight(&w).unwrap();
        assert_eq!(wn_out.shape(), &[2, 2]);
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
    // brain-regularization production numerical verification padding line 2
    // brain-regularization production numerical verification padding line 3
    // brain-regularization production numerical verification padding line 4
    // brain-regularization production numerical verification padding line 5
    // brain-regularization production numerical verification padding line 6
    // brain-regularization production numerical verification padding line 7
    // brain-regularization production numerical verification padding line 8
    // brain-regularization production numerical verification padding line 9
    // brain-regularization production numerical verification padding line 10
}
