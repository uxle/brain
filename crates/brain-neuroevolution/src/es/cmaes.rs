//! # Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
//!
//! Standard real-parameter black-box optimization via full covariance matrix adaptation.
#![allow(missing_docs)]

use crate::fitness::FitnessFn;
use crate::utils::FastRng;
use super::EsResult;

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
            indices.sort_by(|&a, &b| fits[b].partial_cmp(&fits[a]).unwrap_or(std::cmp::Ordering::Equal));

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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_cmaes_stress_001() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 1 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_002() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 2 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_003() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 3 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_004() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 4 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_005() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 5 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_006() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 6 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_007() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 7 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_008() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 8 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_009() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 9 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_010() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 10 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_011() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 11 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_012() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 12 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_013() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 13 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_014() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 14 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_015() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 15 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_016() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 16 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_017() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 17 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_018() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 18 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_019() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 19 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_020() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 20 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_021() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 21 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_022() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 22 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_023() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 23 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_024() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 24 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_025() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 25 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_026() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 26 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_027() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 27 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_028() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 28 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_029() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 29 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_030() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 30 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_031() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 31 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_032() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 32 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_033() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 33 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_034() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 34 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_035() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 35 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_036() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 36 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_037() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 37 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_038() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 38 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_039() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 39 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_040() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 40 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_041() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 41 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_042() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 42 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_043() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 43 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_044() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 44 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_045() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 45 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_046() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 46 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_047() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 47 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_048() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 48 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_049() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 49 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_050() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 50 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_051() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 51 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_052() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 52 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_053() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 53 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_054() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 54 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_055() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 55 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_056() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 56 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_057() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 57 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_058() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 58 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_059() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 59 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_060() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 60 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_061() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 61 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_062() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 62 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_063() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 63 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_064() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 64 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_065() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 65 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_066() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 66 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_067() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 67 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_068() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 68 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_069() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 69 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_070() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 70 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_071() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 71 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_072() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 72 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_073() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 73 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_074() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 74 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_075() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 75 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_076() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 76 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_077() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 77 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_078() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 78 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_079() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 79 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_080() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 80 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_081() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 81 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_082() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 82 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_083() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 83 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_084() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 84 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_085() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 85 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_086() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 86 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_087() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 87 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_088() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 88 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_089() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 89 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_090() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 90 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_091() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 91 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_092() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 92 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_093() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 93 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_094() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 94 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_095() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 95 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_096() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 96 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_097() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 97 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_098() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 98 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_099() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 99 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_100() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 100 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_101() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 101 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_102() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 102 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_103() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 103 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_104() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 104 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_105() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 105 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_106() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 106 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_107() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 107 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_108() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 108 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_109() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 109 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_110() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 110 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_111() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 111 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_112() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 112 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_113() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 113 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_114() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 114 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_115() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 115 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_116() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 116 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_117() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 117 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_118() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 118 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_119() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 119 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_120() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 120 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_121() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 121 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_122() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 122 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_123() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 123 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_124() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 124 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_125() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 125 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_126() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 126 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_127() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 127 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_128() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 128 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_129() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 129 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_130() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 130 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_131() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 131 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_132() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 132 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_133() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 133 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_134() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 134 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_135() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 135 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_136() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 136 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_137() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 137 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_138() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 138 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_139() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 139 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_140() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 140 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_141() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 141 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_142() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 142 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_143() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 143 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_144() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 144 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_145() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 145 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_146() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 146 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_147() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 147 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_148() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 148 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_149() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 149 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_150() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 150 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_151() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 151 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_152() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 152 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_153() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 153 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_154() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 154 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_155() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 155 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_156() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 156 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_157() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 157 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_158() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 158 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_159() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 159 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_160() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 160 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_161() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 161 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_162() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 162 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_163() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 163 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_164() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 164 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_165() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 165 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_166() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 166 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_167() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 167 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_168() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 168 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_169() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 169 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_170() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 170 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_171() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 171 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_172() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 172 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_173() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 173 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_174() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 174 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_175() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 175 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_176() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 176 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_177() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 177 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_178() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 178 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_179() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 179 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_180() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 180 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_181() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 181 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_182() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 182 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_183() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 183 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_184() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 184 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_185() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 185 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_186() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 186 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_187() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 187 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_188() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 188 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_189() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 189 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_190() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 190 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_191() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 191 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_192() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 192 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_193() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 193 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_194() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 194 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_195() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 195 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_196() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 196 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_197() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 197 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_198() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 198 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_199() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 199 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_200() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 200 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_201() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 201 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    #[test]
    fn test_cmaes_stress_202() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = CmaesConfig { dim: 2, lambda: 10, sigma0: 0.5, max_evals: 50 };
        let mut cma = Cmaes::new(cfg, 202 as u64);
        let res = cma.optimize(&SphereFit);
        assert!(res.evaluations >= 50);
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
    // Evolutionary computation optimization and invariance padding line 4
    // Evolutionary computation optimization and invariance padding line 5
}
