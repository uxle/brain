//! # Genetic Crossover Operators
//!
//! Single-point, Two-point, Uniform, BLX-alpha (Blend Crossover), and Simulated Binary (SBX) Crossover.
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

/// Performs Blend Crossover (BLX-alpha) exploring continuous intervals around parent genes.
pub fn blx_alpha_crossover(
    parent_a: &Genome,
    parent_b: &Genome,
    alpha: f64,
    rng: &mut FastRng,
) -> (Genome, Genome) {
    let dim = parent_a.len().min(parent_b.len());
    let mut child1_genes = Vec::with_capacity(dim);
    let mut child2_genes = Vec::with_capacity(dim);

    for i in 0..dim {
        let p1 = parent_a.genes[i];
        let p2 = parent_b.genes[i];
        let min_val = p1.min(p2);
        let max_val = p1.max(p2);
        let range = max_val - min_val;

        let lower = min_val - alpha * range;
        let upper = max_val + alpha * range;

        child1_genes.push(rng.sample_range(lower, upper));
        child2_genes.push(rng.sample_range(lower, upper));
    }

    (Genome::new(child1_genes), Genome::new(child2_genes))
}

/// Performs Simulated Binary Crossover (SBX) with distribution index eta_c.
pub fn sbx_crossover(
    parent_a: &Genome,
    parent_b: &Genome,
    eta_c: f64,
    rng: &mut FastRng,
) -> (Genome, Genome) {
    let dim = parent_a.len().min(parent_b.len());
    let mut child1_genes = Vec::with_capacity(dim);
    let mut child2_genes = Vec::with_capacity(dim);

    for i in 0..dim {
        let p1 = parent_a.genes[i];
        let p2 = parent_b.genes[i];

        let u = rng.next_f64();
        let beta = if u <= 0.5 {
            (2.0 * u).powf(1.0 / (eta_c + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta_c + 1.0))
        };

        let c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2);
        let c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2);

        child1_genes.push(c1);
        child2_genes.push(c2);
    }

    (Genome::new(child1_genes), Genome::new(child2_genes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossover_operators() {
        let mut rng = FastRng::seed(42);
        let p1 = Genome::new(vec![1.0, 2.0, 3.0, 4.0]);
        let p2 = Genome::new(vec![10.0, 20.0, 30.0, 40.0]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 4);
        assert_eq!(c2.len(), 4);

        let (b1, b2) = blx_alpha_crossover(&p1, &p2, 0.5, &mut rng);
        assert_eq!(b1.len(), 4);
        assert_eq!(b2.len(), 4);

        let (s1, s2) = sbx_crossover(&p1, &p2, 2.0, &mut rng);
        assert_eq!(s1.len(), 4);
        assert_eq!(s2.len(), 4);
    }
}
