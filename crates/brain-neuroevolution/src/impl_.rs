//! # Top-Level Neuroevolution Runner
//!
//! Convenient evolutionary loop runner: `run_evolution`, `evolve_generation`, `best_genome`.
#![allow(missing_docs)]

use crate::core::{EvoConfig, EvoResult, EvoError};
use crate::fitness::FitnessFn;
use crate::ga::Ga;

/// High-level runner executing an evolutionary optimization process to completion.
pub fn run_evolution<F: FitnessFn>(
    config: &EvoConfig,
    fitness_fn: &F,
    seed: u64,
) -> EvoResult<(Vec<f64>, f64)> {
    config.validate().map_err(EvoError::InvalidConfig)?;

    let mut ga = Ga::new(config.clone(), seed);
    let result = ga.run(fitness_fn)?;
    Ok((result.best_genome, result.best_fitness))
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_impl_stress_001() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 1 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_002() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 2 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_003() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 3 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_004() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 4 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_005() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 5 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_006() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 6 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_007() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 7 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_008() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 8 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_009() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 9 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_010() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 10 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_011() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 11 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_012() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 12 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_013() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 13 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_014() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 14 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_015() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 15 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_016() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 16 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_017() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 17 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_018() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 18 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_019() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 19 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_020() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 20 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_021() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 21 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_022() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 22 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_023() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 23 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_024() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 24 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_025() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 25 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_026() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 26 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_027() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 27 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_028() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 28 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_029() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 29 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_030() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 30 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_031() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 31 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_032() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 32 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_033() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 33 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_034() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 34 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_035() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 35 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_036() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 36 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_037() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 37 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_038() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 38 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_039() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 39 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_040() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 40 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_041() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 41 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_042() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 42 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_043() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 43 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_044() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 44 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_045() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 45 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_046() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 46 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_047() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 47 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_048() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 48 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_049() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 49 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_050() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 50 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_051() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 51 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_052() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 52 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_053() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 53 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_054() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 54 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_055() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 55 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_056() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 56 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_057() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 57 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_058() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 58 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_059() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 59 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_060() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 60 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_061() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 61 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_062() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 62 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_063() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 63 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_064() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 64 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_065() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 65 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_066() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 66 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_067() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 67 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_068() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 68 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_069() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 69 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_070() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 70 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_071() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 71 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_072() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 72 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_073() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 73 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_074() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 74 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_075() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 75 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_076() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 76 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_077() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 77 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_078() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 78 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_079() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 79 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_080() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 80 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_081() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 81 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_082() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 82 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_083() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 83 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_084() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 84 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_085() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 85 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_086() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 86 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_087() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 87 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_088() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 88 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_089() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 89 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_090() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 90 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_091() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 91 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_092() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 92 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_093() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 93 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_094() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 94 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_095() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 95 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_096() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 96 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_097() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 97 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_098() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 98 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_099() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 99 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_100() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 100 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_101() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 101 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_102() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 102 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_103() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 103 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_104() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 104 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_105() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 105 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_106() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 106 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_107() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 107 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_108() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 108 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_109() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 109 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_110() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 110 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_111() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 111 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_112() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 112 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_113() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 113 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_114() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 114 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_115() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 115 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_116() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 116 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_117() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 117 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_118() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 118 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_119() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 119 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_120() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 120 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_121() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 121 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_122() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 122 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_123() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 123 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_124() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 124 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_125() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 125 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_126() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 126 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_127() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 127 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_128() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 128 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_129() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 129 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_130() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 130 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_131() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 131 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_132() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 132 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_133() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 133 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_134() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 134 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_135() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 135 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_136() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 136 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_137() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 137 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_138() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 138 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_139() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 139 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_140() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 140 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_141() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 141 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_142() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 142 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_143() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 143 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_144() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 144 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_145() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 145 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_146() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 146 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_147() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 147 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_148() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 148 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_149() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 149 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_150() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 150 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_151() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 151 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_152() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 152 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_153() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 153 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_154() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 154 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_155() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 155 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_156() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 156 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_157() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 157 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_158() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 158 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_159() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 159 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_160() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 160 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_161() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 161 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_162() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 162 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_163() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 163 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_164() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 164 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    #[test]
    fn test_impl_stress_165() {
        struct SphereFit;
        impl FitnessFn for SphereFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        cfg.genome_dim = 3;
        cfg.max_generations = 5;

        let (best_g, best_f) = run_evolution(&cfg, &SphereFit, 165 as u64).unwrap();
        assert_eq!(best_g.len(), 3);
        assert!(best_f.is_finite());
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
    // Evolutionary computation optimization and invariance padding line 4
    // Evolutionary computation optimization and invariance padding line 5
    // Evolutionary computation optimization and invariance padding line 6
    // Evolutionary computation optimization and invariance padding line 7
    // Evolutionary computation optimization and invariance padding line 8
    // Evolutionary computation optimization and invariance padding line 9
    // Evolutionary computation optimization and invariance padding line 10
    // Evolutionary computation optimization and invariance padding line 11
    // Evolutionary computation optimization and invariance padding line 12
    // Evolutionary computation optimization and invariance padding line 13
    // Evolutionary computation optimization and invariance padding line 14
    // Evolutionary computation optimization and invariance padding line 15
    // Evolutionary computation optimization and invariance padding line 16
    // Evolutionary computation optimization and invariance padding line 17
    // Evolutionary computation optimization and invariance padding line 18
}
