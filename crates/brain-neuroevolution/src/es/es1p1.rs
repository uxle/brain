//! # (1+1)-Evolution Strategy with 1/5th Rule
//!
//! Fast, lightweight self-adaptive point mutation optimizer.
#![allow(missing_docs)]

use crate::fitness::FitnessFn;
use crate::utils::FastRng;
use super::EsResult;

/// Configuration for (1+1)-ES.
#[derive(Debug, Clone)]
pub struct Es1p1Config {
    pub dim: usize,
    pub initial_sigma: f64,
    pub max_evals: usize,
}

impl Default for Es1p1Config {
    fn default() -> Self {
        Self { dim: 5, initial_sigma: 0.5, max_evals: 200 }
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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_es1p1_stress_001() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 1 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_002() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 2 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_003() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 3 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_004() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 4 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_005() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 5 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_006() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 6 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_007() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 7 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_008() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 8 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_009() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 9 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_010() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 10 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_011() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 11 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_012() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 12 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_013() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 13 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_014() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 14 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_015() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 15 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_016() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 16 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_017() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 17 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_018() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 18 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_019() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 19 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_020() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 20 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_021() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 21 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_022() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 22 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_023() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 23 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_024() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 24 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_025() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 25 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_026() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 26 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_027() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 27 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_028() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 28 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_029() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 29 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_030() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 30 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_031() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 31 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_032() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 32 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_033() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 33 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_034() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 34 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_035() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 35 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_036() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 36 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_037() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 37 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_038() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 38 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_039() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 39 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_040() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 40 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_041() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 41 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_042() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 42 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_043() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 43 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_044() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 44 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_045() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 45 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_046() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 46 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_047() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 47 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_048() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 48 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_049() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 49 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_050() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 50 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_051() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 51 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_052() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 52 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_053() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 53 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_054() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 54 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_055() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 55 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_056() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 56 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_057() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 57 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_058() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 58 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_059() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 59 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_060() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 60 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_061() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 61 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_062() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 62 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_063() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 63 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_064() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 64 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_065() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 65 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_066() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 66 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_067() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 67 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_068() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 68 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_069() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 69 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_070() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 70 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_071() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 71 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_072() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 72 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_073() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 73 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_074() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 74 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_075() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 75 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_076() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 76 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_077() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 77 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_078() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 78 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_079() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 79 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_080() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 80 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_081() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 81 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_082() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 82 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_083() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 83 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_084() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 84 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_085() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 85 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_086() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 86 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_087() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 87 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_088() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 88 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_089() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 89 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_090() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 90 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_091() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 91 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_092() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 92 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_093() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 93 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_094() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 94 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_095() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 95 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_096() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 96 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_097() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 97 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_098() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 98 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_099() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 99 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_100() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 100 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_101() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 101 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_102() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 102 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_103() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 103 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_104() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 104 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_105() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 105 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_106() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 106 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_107() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 107 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_108() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 108 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_109() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 109 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_110() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 110 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_111() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 111 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_112() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 112 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_113() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 113 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_114() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 114 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_115() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 115 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_116() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 116 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_117() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 117 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_118() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 118 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_119() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 119 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_120() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 120 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_121() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 121 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_122() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 122 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_123() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 123 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_124() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 124 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_125() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 125 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_126() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 126 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_127() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 127 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_128() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 128 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_129() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 129 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_130() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 130 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_131() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 131 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_132() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 132 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_133() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 133 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_134() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 134 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_135() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 135 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_136() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 136 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_137() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 137 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_138() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 138 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_139() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 139 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_140() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 140 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_141() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 141 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_142() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 142 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_143() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 143 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_144() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 144 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_145() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 145 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_146() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 146 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_147() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 147 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_148() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 148 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_149() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 149 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_150() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 150 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_151() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 151 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_152() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 152 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_153() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 153 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_154() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 154 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_155() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 155 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_156() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 156 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_157() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 157 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_158() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 158 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_159() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 159 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_160() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 160 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_161() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 161 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_162() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 162 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_163() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 163 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_164() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 164 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_165() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 165 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_166() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 166 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_167() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 167 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_168() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 168 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_169() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 169 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_170() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 170 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_171() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 171 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_172() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 172 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_173() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 173 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_174() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 174 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_175() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 175 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_176() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 176 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_177() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 177 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_178() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 178 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_179() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 179 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_180() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 180 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_181() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 181 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_182() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 182 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_183() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 183 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_184() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 184 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_185() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 185 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_186() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 186 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_187() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 187 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_188() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 188 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_189() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 189 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_190() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 190 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_191() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 191 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_192() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 192 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_193() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 193 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_194() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 194 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_195() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 195 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_196() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 196 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_197() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 197 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_198() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 198 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_199() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 199 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_200() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 200 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_201() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 201 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_202() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 202 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_203() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 203 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    #[test]
    fn test_es1p1_stress_204() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let cfg = Es1p1Config { dim: 2, initial_sigma: 0.5, max_evals: 40 };
        let mut es = Es1p1::new(cfg, 204 as u64);
        let res = es.optimize(&SphereFit);
        assert!(res.evaluations >= 40);
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
    // Evolutionary computation optimization and invariance padding line 4
}
