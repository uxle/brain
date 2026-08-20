//! # Weight & Spectral Normalization
//!
//! Weight normalization (magnitude-direction decoupling) and Spectral Normalization (power iteration).
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

use super::super::core::{RegError, RegResult};
use brain_core::Tensor;

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
            let v_norm = v_new
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(self.config.eps);
            for j in 0..n {
                self.v[j] = v_new[j] / v_norm;
            }

            // u = W v
            let mut u_new = vec![0.0; m];
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += data[i * n + j] * self.v[j];
                }
                u_new[i] = sum;
            }
            let u_norm = u_new
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(self.config.eps);
            for i in 0..m {
                self.u[i] = u_new[i] / u_norm;
            }
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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown
    )]
    use super::*;
    use crate::augment::*;
    use crate::config::*;
    use crate::consistency::*;
    use crate::core::*;
    use crate::curriculum::*;
    use crate::decay::*;
    use crate::dropout::*;
    use crate::dropout_uncertainty::*;
    use crate::earlystop::*;
    use crate::label_smooth::*;
    use crate::normalization::*;
    use crate::ops::*;
    use crate::perturb::*;
    use crate::r#impl::*;
    use crate::registry::*;
    use crate::regularizers::*;
    use crate::rules::*;
    use crate::stopping::*;
    use crate::train_hooks::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
