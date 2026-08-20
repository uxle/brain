//! # Genetic Crossover Operators
//!
//! Single-point, Two-point, Uniform, and Arithmetic (BLX-alpha / Simulated Binary) crossover.
#![allow(missing_docs)]

use crate::genome::Genome;
use crate::utils::FastRng;

/// Performs single-point crossover between two parent genomes.
pub fn single_point_crossover(
    parent_a: &Genome,
    parent_b: &Genome,
    rng: &mut FastRng,
) -> (Genome, Genome) {
    let dim = parent_a.len().min(parent_b.len());
    if dim <= 1 {
        return (parent_a.clone(), parent_b.clone());
    }

    let point = (rng.next_u64() as usize % (dim - 1)) + 1;

    let mut child1_genes = Vec::with_capacity(dim);
    let mut child2_genes = Vec::with_capacity(dim);

    child1_genes.extend_from_slice(&parent_a.genes[..point]);
    child1_genes.extend_from_slice(&parent_b.genes[point..dim]);

    child2_genes.extend_from_slice(&parent_b.genes[..point]);
    child2_genes.extend_from_slice(&parent_a.genes[point..dim]);

    (Genome::new(child1_genes), Genome::new(child2_genes))
}

/// Performs uniform crossover where each gene is independently inherited from either parent.
pub fn uniform_crossover(
    parent_a: &Genome,
    parent_b: &Genome,
    rng: &mut FastRng,
) -> (Genome, Genome) {
    let dim = parent_a.len().min(parent_b.len());
    let mut child1_genes = Vec::with_capacity(dim);
    let mut child2_genes = Vec::with_capacity(dim);

    for i in 0..dim {
        if rng.next_f64() < 0.5 {
            child1_genes.push(parent_a.genes[i]);
            child2_genes.push(parent_b.genes[i]);
        } else {
            child1_genes.push(parent_b.genes[i]);
            child2_genes.push(parent_a.genes[i]);
        }
    }

    (Genome::new(child1_genes), Genome::new(child2_genes))
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
