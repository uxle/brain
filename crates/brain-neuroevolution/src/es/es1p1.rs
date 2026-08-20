//! # (1+1)-Evolution Strategy with 1/5th Rule
//!
//! Fast, lightweight self-adaptive point mutation optimizer.
#![allow(missing_docs)]

use super::EsResult;
use crate::fitness::FitnessFn;
use crate::utils::FastRng;

/// Configuration for (1+1)-ES.
#[derive(Debug, Clone)]
pub struct Es1p1Config {
    pub dim: usize,
    pub initial_sigma: f64,
    pub max_evals: usize,
}

impl Default for Es1p1Config {
    fn default() -> Self {
        Self {
            dim: 5,
            initial_sigma: 0.5,
            max_evals: 200,
        }
    }
}

/// (1+1)-ES optimizer.
pub struct Es1p1 {
    pub config: Es1p1Config,
    pub parent: Vec<f64>,
    pub sigma: f64,
    pub rng: FastRng,
}

impl Es1p1 {
    pub fn new(config: Es1p1Config, seed: u64) -> Self {
        let dim = config.dim;
        let sigma0 = config.initial_sigma;
        Self {
            config,
            parent: vec![0.0; dim],
            sigma: sigma0,
            rng: FastRng::seed(seed),
        }
    }

    pub fn optimize<F: FitnessFn>(&mut self, fitness_fn: &F) -> EsResult {
        let mut parent_fit = fitness_fn.evaluate(&self.parent);
        let mut evals = 1usize;

        while evals < self.config.max_evals {
            let mutant: Vec<f64> = (0..self.config.dim)
                .map(|d| self.parent[d] + self.sigma * self.rng.sample_gaussian(0.0, 1.0))
                .collect();
            let mutant_fit = fitness_fn.evaluate(&mutant);
            evals += 1;

            if mutant_fit > parent_fit {
                self.parent = mutant;
                parent_fit = mutant_fit;
                // 1/5th success rule: increase step size
                self.sigma *= 1.1;
            } else {
                // decrease step size
                self.sigma *= 0.95;
            }
        }

        EsResult {
            best_params: self.parent.clone(),
            best_fitness: parent_fit,
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
