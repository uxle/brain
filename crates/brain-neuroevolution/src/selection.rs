//! # Parent Selection Operators
//!
//! Tournament selection, Roulette-wheel (fitness proportional), and Linear Rank-based selection.
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

/// Executes Roulette-wheel (fitness-proportionate) selection.
pub fn roulette_wheel_select<'a>(population: &'a [Genome], rng: &mut FastRng) -> &'a Genome {
    let n = population.len();
    assert!(n > 0, "Population cannot be empty for selection");

    let min_fit = population
        .iter()
        .map(|g| g.fitness.unwrap_or(0.0))
        .fold(f64::INFINITY, f64::min);
    let offset = if min_fit < 0.0 { -min_fit + 1e-4 } else { 1e-4 };

    let fitnesses: Vec<f64> = population
        .iter()
        .map(|g| g.fitness.unwrap_or(0.0) + offset)
        .collect();
    let total_fitness: f64 = fitnesses.iter().sum();

    let pick = rng.next_f64() * total_fitness;
    let mut accum = 0.0;

    for (idx, &fit) in fitnesses.iter().enumerate() {
        accum += fit;
        if accum >= pick {
            return &population[idx];
        }
    }

    &population[n - 1]
}

/// Executes linear rank-based selection.
pub fn rank_based_select<'a>(population: &'a [Genome], rng: &mut FastRng) -> &'a Genome {
    let n = population.len();
    assert!(n > 0, "Population cannot be empty for selection");

    let mut indexed_pop: Vec<(usize, f64)> = population
        .iter()
        .enumerate()
        .map(|(i, g)| (i, g.fitness.unwrap_or(f64::NEG_INFINITY)))
        .collect();

    indexed_pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Linear ranks from 1 to N
    let total_rank_sum = (n * (n + 1)) / 2;
    let pick = (rng.next_u64() as usize % total_rank_sum) + 1;

    let mut accum = 0;
    for (rank, &(orig_idx, _)) in indexed_pop.iter().enumerate() {
        accum += rank + 1;
        if accum >= pick {
            return &population[orig_idx];
        }
    }

    &population[indexed_pop.last().unwrap().0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_methods() {
        let mut rng = FastRng::seed(42);
        let mut p1 = Genome::new(vec![1.0]);
        p1.fitness = Some(10.0);
        let mut p2 = Genome::new(vec![2.0]);
        p2.fitness = Some(100.0);

        let pop = vec![p1, p2];

        let tour = tournament_select(&pop, 2, &mut rng);
        assert_eq!(tour.fitness, Some(100.0));

        let roul = roulette_wheel_select(&pop, &mut rng);
        assert!(roul.fitness.is_some());

        let rank = rank_based_select(&pop, &mut rng);
        assert!(rank.fitness.is_some());
    }
}
