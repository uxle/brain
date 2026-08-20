//! # Genetic Algorithm (GA) Engine
//!
//! Orchestrates population evolution, selection, crossover, mutation, and elitism preservation.
#![allow(missing_docs)]

pub mod termination;
pub use termination::TerminationConfig;

use crate::core::{EvoConfig, EvoResult};
use crate::crossover::single_point_crossover;
use crate::fitness::{FitnessFn, FitnessStats};
use crate::mutation::mutate_gaussian;
use crate::population::Population;
use crate::selection::tournament_select;
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

                mutate_gaussian(
                    &mut c1,
                    self.config.mutation_rate,
                    0.1,
                    -5.0,
                    5.0,
                    &mut self.rng,
                );
                mutate_gaussian(
                    &mut c2,
                    self.config.mutation_rate,
                    0.1,
                    -5.0,
                    5.0,
                    &mut self.rng,
                );

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
