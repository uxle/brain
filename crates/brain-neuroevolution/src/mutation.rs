//! # Genetic Mutation Operators
//!
//! Gaussian perturbation (adaptive sigma), polynomial mutation, uniform reset, and gene swapping.
#![allow(missing_docs)]

use crate::genome::Genome;
use crate::utils::FastRng;

/// Mutates a genome in-place using Gaussian perturbations with clamping.
pub fn mutate_gaussian(
    genome: &mut Genome,
    mutation_rate: f64,
    sigma: f64,
    min_val: f64,
    max_val: f64,
    rng: &mut FastRng,
) {
    for gene in genome.genes.iter_mut() {
        if rng.next_f64() < mutation_rate {
            let delta = rng.sample_gaussian(0.0, sigma);
            *gene = (*gene + delta).clamp(min_val, max_val);
        }
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
