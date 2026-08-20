//! # Parent Selection Operators
//!
//! Tournament selection, Roulette-wheel (fitness proportional), and Rank-based selection.
#![allow(missing_docs)]

use crate::genome::Genome;
use crate::utils::FastRng;

/// Selection strategy enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionKind {
    #[default]
    Tournament,
    Roulette,
    Rank,
}

/// Executes tournament selection to pick one individual.
pub fn tournament_select<'a>(
    population: &'a [Genome],
    tournament_size: usize,
    rng: &mut FastRng,
) -> &'a Genome {
    let n = population.len();
    assert!(n > 0, "Population cannot be empty for selection");

    let mut best_idx = (rng.next_u64() as usize) % n;
    let mut best_fit = population[best_idx].fitness.unwrap_or(f64::NEG_INFINITY);

    for _ in 1..tournament_size {
        let candidate_idx = (rng.next_u64() as usize) % n;
        let candidate_fit = population[candidate_idx]
            .fitness
            .unwrap_or(f64::NEG_INFINITY);
        if candidate_fit > best_fit {
            best_fit = candidate_fit;
            best_idx = candidate_idx;
        }
    }

    &population[best_idx]
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
