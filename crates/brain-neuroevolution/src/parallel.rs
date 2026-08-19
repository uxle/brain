//! # Multi-Threaded Population Evaluation
//!
//! Scoped multi-threaded fitness evaluation using `std::thread::scope` for deterministic batch parallelism.
#![allow(missing_docs)]

use crate::genome::Genome;
use crate::fitness::FitnessFn;

/// Configuration for parallel evaluation.
#[derive(Debug, Clone, Default)]
pub struct ParallelConfig {
    pub num_threads: usize,
}

/// Evaluates a batch of genomes across threads using scoped workers.
pub fn evaluate_population_parallel<F: FitnessFn + Sync>(
    population: &mut [Genome],
    fitness_fn: &F,
    num_threads: usize,
) {
    let n = population.len();
    if n == 0 { return; }

    let chunk_size = (n + num_threads - 1) / num_threads.max(1);

    std::thread::scope(|s| {
        for chunk in population.chunks_mut(chunk_size) {
            s.spawn(move || {
                for ind in chunk {
                    ind.fitness = Some(fitness_fn.evaluate(&ind.genes));
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
