//! # Population Management & Bookkeeping
//!
//! Generational population pool, elitism selection, best individual tracking, and diversity stats.
#![allow(missing_docs)]

use crate::genome::Genome;
use crate::utils::{rank_fitness, FastRng};

/// Collection of evolutionary individuals.
#[derive(Debug, Clone)]
pub struct Population {
    pub individuals: Vec<Genome>,
    pub generation: usize,
}

impl Population {
    pub fn new(individuals: Vec<Genome>) -> Self {
        Self {
            individuals,
            generation: 0,
        }
    }

    pub fn random_uniform(
        size: usize,
        dim: usize,
        min_val: f64,
        max_val: f64,
        rng: &mut FastRng,
    ) -> Self {
        let inds: Vec<Genome> = (0..size)
            .map(|_| Genome::random_uniform(dim, min_val, max_val, rng))
            .collect();
        Self::new(inds)
    }

    pub fn size(&self) -> usize {
        self.individuals.len()
    }

    pub fn best_individual(&self) -> Option<&Genome> {
        self.individuals
            .iter()
            .filter(|ind| ind.fitness.is_some())
            .max_by(|a, b| {
                a.fitness
                    .unwrap()
                    .partial_cmp(&b.fitness.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn get_elites(&self, count: usize) -> Vec<Genome> {
        let fitnesses: Vec<f64> = self
            .individuals
            .iter()
            .map(|ind| ind.fitness.unwrap_or(f64::NEG_INFINITY))
            .collect();
        let ranked = rank_fitness(&fitnesses);
        ranked
            .into_iter()
            .take(count)
            .map(|i| self.individuals[i].clone())
            .collect()
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
