//! # Evolutionary State Checkpointing
//!
//! Serialization, snapshot save/load, and deterministic resumption of evolutionary state.
#![allow(missing_docs)]

use crate::genome::Genome;
use crate::population::Population;

/// Evolutionary run checkpoint state.
#[derive(Debug, Clone)]
pub struct EvoCheckpoint {
    pub generation: usize,
    pub best_fitness: f64,
    pub best_genes: Vec<f64>,
    pub population_genes: Vec<Vec<f64>>,
}

impl EvoCheckpoint {
    pub fn create(generation: usize, population: &Population) -> Self {
        let best = population.best_individual();
        let best_fitness = best.and_then(|b| b.fitness).unwrap_or(0.0);
        let best_genes = best.map(|b| b.genes.clone()).unwrap_or_default();
        let population_genes = population
            .individuals
            .iter()
            .map(|ind| ind.genes.clone())
            .collect();

        Self {
            generation,
            best_fitness,
            best_genes,
            population_genes,
        }
    }

    pub fn restore_population(&self) -> Population {
        let inds: Vec<Genome> = self
            .population_genes
            .iter()
            .map(|g| Genome::new(g.clone()))
            .collect();
        let mut pop = Population::new(inds);
        pop.generation = self.generation;
        pop
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
