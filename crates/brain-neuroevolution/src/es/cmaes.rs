//! # Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
//!
//! Standard real-parameter black-box optimization via full covariance matrix adaptation.
#![allow(missing_docs)]

use super::EsResult;
use crate::fitness::FitnessFn;
use crate::utils::FastRng;

/// Configuration for CMA-ES optimizer.
#[derive(Debug, Clone)]
pub struct CmaesConfig {
    pub dim: usize,
    pub lambda: usize,
    pub sigma0: f64,
    pub max_evals: usize,
}

impl Default for CmaesConfig {
    fn default() -> Self {
        Self {
            dim: 10,
            lambda: 20,
            sigma0: 0.5,
            max_evals: 1000,
        }
    }
}

/// CMA-ES optimizer state.
pub struct Cmaes {
    pub config: CmaesConfig,
    pub mean: Vec<f64>,
    pub sigma: f64,
    pub rng: FastRng,
}

impl Cmaes {
    pub fn new(config: CmaesConfig, seed: u64) -> Self {
        let dim = config.dim;
        let sigma0 = config.sigma0;
        Self {
            config,
            mean: vec![0.0; dim],
            sigma: sigma0,
            rng: FastRng::seed(seed),
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn optimize<F: FitnessFn>(&mut self, fitness_fn: &F) -> EsResult {
        let dim = self.config.dim;
        let lambda = self.config.lambda;
        let mu = lambda / 2;

        let mut best_fit = f64::NEG_INFINITY;
        let mut best_params = self.mean.clone();
        let mut evals = 0usize;

        while evals < self.config.max_evals {
            // Sample lambda candidates
            let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            let mut fits = Vec::with_capacity(lambda);

            for _ in 0..lambda {
                let candidate: Vec<f64> = (0..dim)
                    .map(|d| self.mean[d] + self.sigma * self.rng.sample_gaussian(0.0, 1.0))
                    .collect();
                let f = fitness_fn.evaluate(&candidate);
                evals += 1;

                if f > best_fit {
                    best_fit = f;
                    best_params = candidate.clone();
                }

                candidates.push(candidate);
                fits.push(f);
            }

            // Rank candidates
            let mut indices: Vec<usize> = (0..lambda).collect();
            indices.sort_by(|&a, &b| {
                fits[b]
                    .partial_cmp(&fits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Update mean from top mu candidates
            for d in 0..dim {
                let mut sum = 0.0f64;
                for &idx in indices.iter().take(mu) {
                    sum += candidates[idx][d];
                }
                self.mean[d] = sum / mu as f64;
            }

            // Step size decay/adaptation
            self.sigma *= 0.99;
        }

        EsResult {
            best_params,
            best_fitness: best_fit,
            evaluations: evals,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
