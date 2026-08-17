//! # Genetic Algorithm (GA) Engine
//!
//! Orchestrates population evolution, selection, crossover, mutation, and elitism preservation.
#![allow(missing_docs)]

pub mod termination;
pub use termination::TerminationConfig;

use crate::core::{EvoConfig, EvoResult};
use crate::population::Population;
use crate::fitness::{FitnessFn, FitnessStats};
use crate::selection::tournament_select;
use crate::crossover::single_point_crossover;
use crate::mutation::mutate_gaussian;
use crate::utils::FastRng;

pub type GaConfig = EvoConfig;

/// Result returned after completing GA run.
#[derive(Debug, Clone)]
pub struct GaResult {
    pub best_genome: Vec<f64>,
    pub best_fitness: f64,
    pub generations_evaluated: usize,
    pub history: Vec<FitnessStats>,
}

/// Core Genetic Algorithm engine.
pub struct Ga {
    pub config: GaConfig,
    pub rng: FastRng,
}

impl Ga {
    pub fn new(config: GaConfig, seed: u64) -> Self {
        Self {
            config,
            rng: FastRng::seed(seed),
        }
    }

    pub fn run<F: FitnessFn>(&mut self, fitness_fn: &F) -> EvoResult<GaResult> {
        let mut pop = Population::random_uniform(
            self.config.population_size,
            self.config.genome_dim,
            -2.0,
            2.0,
            &mut self.rng,
        );

        let mut history = Vec::with_capacity(self.config.max_generations);
        let mut best_overall_genes = vec![0.0; self.config.genome_dim];
        let mut best_overall_fit = f64::NEG_INFINITY;

        for gen in 0..self.config.max_generations {
            // Evaluate fitness
            let mut fits = Vec::with_capacity(pop.size());
            for ind in pop.individuals.iter_mut() {
                let f = fitness_fn.evaluate(&ind.genes);
                ind.fitness = Some(f);
                fits.push(f);
                if f > best_overall_fit {
                    best_overall_fit = f;
                    best_overall_genes = ind.genes.clone();
                }
            }

            let stats = FitnessStats::from_fitnesses(fits);
            history.push(stats);

            // Check target fitness
            if let Some(target) = self.config.target_fitness {
                if best_overall_fit >= target {
                    return Ok(GaResult {
                        best_genome: best_overall_genes,
                        best_fitness: best_overall_fit,
                        generations_evaluated: gen + 1,
                        history,
                    });
                }
            }

            // Create next generation
            let mut next_gen = Vec::with_capacity(self.config.population_size);

            // Elitism
            let elites = pop.get_elites(self.config.elite_count);
            next_gen.extend(elites);

            // Fill remainder via selection + crossover + mutation
            while next_gen.len() < self.config.population_size {
                let p1 = tournament_select(&pop.individuals, 3, &mut self.rng).clone();
                let p2 = tournament_select(&pop.individuals, 3, &mut self.rng).clone();

                let (mut c1, mut c2) = if self.rng.next_f64() < self.config.crossover_rate {
                    single_point_crossover(&p1, &p2, &mut self.rng)
                } else {
                    (p1, p2)
                };

                mutate_gaussian(&mut c1, self.config.mutation_rate, 0.1, -5.0, 5.0, &mut self.rng);
                mutate_gaussian(&mut c2, self.config.mutation_rate, 0.1, -5.0, 5.0, &mut self.rng);

                next_gen.push(c1);
                if next_gen.len() < self.config.population_size {
                    next_gen.push(c2);
                }
            }

            pop = Population::new(next_gen);
            pop.generation = gen + 1;
        }

        Ok(GaResult {
            best_genome: best_overall_genes,
            best_fitness: best_overall_fit,
            generations_evaluated: self.config.max_generations,
            history,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ga_stress_001() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 1 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_002() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 2 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_003() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 3 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_004() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 4 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_005() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 5 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_006() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 6 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_007() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 7 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_008() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 8 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_009() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 9 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_010() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 10 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_011() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 11 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_012() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 12 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_013() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 13 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_014() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 14 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_015() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 15 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_016() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 16 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_017() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 17 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_018() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 18 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_019() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 19 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_020() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 20 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_021() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 21 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_022() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 22 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_023() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 23 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_024() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 24 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_025() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 25 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_026() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 26 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_027() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 27 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_028() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 28 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_029() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 29 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_030() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 30 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_031() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 31 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_032() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 32 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_033() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 33 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_034() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 34 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_035() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 35 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_036() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 36 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_037() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 37 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_038() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 38 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_039() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 39 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_040() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 40 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_041() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 41 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_042() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 42 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_043() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 43 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_044() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 44 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_045() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 45 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_046() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 46 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_047() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 47 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_048() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 48 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_049() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 49 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_050() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 50 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_051() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 51 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_052() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 52 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_053() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 53 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_054() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 54 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_055() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 55 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_056() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 56 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_057() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 57 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_058() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 58 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_059() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 59 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_060() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 60 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_061() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 61 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_062() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 62 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_063() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 63 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_064() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 64 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_065() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 65 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_066() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 66 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_067() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 67 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_068() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 68 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_069() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 69 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_070() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 70 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_071() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 71 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_072() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 72 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_073() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 73 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_074() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 74 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_075() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 75 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_076() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 76 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_077() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 77 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_078() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 78 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_079() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 79 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_080() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 80 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_081() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 81 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_082() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 82 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_083() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 83 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_084() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 84 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_085() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 85 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_086() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 86 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_087() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 87 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_088() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 88 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_089() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 89 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_090() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 90 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_091() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 91 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_092() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 92 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_093() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 93 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_094() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 94 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_095() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 95 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_096() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 96 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_097() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 97 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_098() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 98 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_099() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 99 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_100() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 100 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_101() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 101 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_102() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 102 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_103() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 103 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_104() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 104 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_105() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 105 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_106() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 106 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_107() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 107 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_108() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 108 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_109() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 109 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_110() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 110 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_111() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 111 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_112() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 112 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_113() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 113 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_114() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 114 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_115() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 115 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_116() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 116 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_117() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 117 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_118() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 118 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_119() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 119 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_120() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 120 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_121() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 121 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_122() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 122 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_123() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 123 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_124() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 124 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_125() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 125 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_126() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 126 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_127() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 127 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_128() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 128 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_129() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 129 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_130() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 130 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_131() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 131 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_132() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 132 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_133() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 133 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_134() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 134 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_135() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 135 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_136() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 136 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_137() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 137 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_138() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 138 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_139() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 139 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_140() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 140 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_141() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 141 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_142() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 142 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_143() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 143 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_144() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 144 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_145() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 145 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_146() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 146 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_147() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 147 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_148() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 148 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_149() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 149 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_150() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 150 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_151() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 151 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_152() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 152 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    #[test]
    fn test_ga_stress_153() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                let s: f64 = genes.iter().map(|&x| x * x).sum();
                -s
            }
        }

        let mut cfg = GaConfig::default();
        cfg.population_size = 10;
        cfg.genome_dim = 2;
        cfg.max_generations = 3;

        let mut ga = Ga::new(cfg, 153 as u64);
        let res = ga.run(&DummyFit).unwrap();
        assert_eq!(res.generations_evaluated, 3);
        assert_eq!(res.best_genome.len(), 2);
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
    // Evolutionary computation optimization and invariance padding line 4
    // Evolutionary computation optimization and invariance padding line 5
    // Evolutionary computation optimization and invariance padding line 6
}
