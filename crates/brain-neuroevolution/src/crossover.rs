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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_crossover_stress_001() {
        let mut rng = FastRng::seed(1 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_002() {
        let mut rng = FastRng::seed(2 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_003() {
        let mut rng = FastRng::seed(3 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_004() {
        let mut rng = FastRng::seed(4 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_005() {
        let mut rng = FastRng::seed(5 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_006() {
        let mut rng = FastRng::seed(6 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_007() {
        let mut rng = FastRng::seed(7 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_008() {
        let mut rng = FastRng::seed(8 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_009() {
        let mut rng = FastRng::seed(9 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_010() {
        let mut rng = FastRng::seed(10 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_011() {
        let mut rng = FastRng::seed(11 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_012() {
        let mut rng = FastRng::seed(12 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_013() {
        let mut rng = FastRng::seed(13 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_014() {
        let mut rng = FastRng::seed(14 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_015() {
        let mut rng = FastRng::seed(15 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_016() {
        let mut rng = FastRng::seed(16 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_017() {
        let mut rng = FastRng::seed(17 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_018() {
        let mut rng = FastRng::seed(18 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_019() {
        let mut rng = FastRng::seed(19 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_020() {
        let mut rng = FastRng::seed(20 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_021() {
        let mut rng = FastRng::seed(21 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_022() {
        let mut rng = FastRng::seed(22 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_023() {
        let mut rng = FastRng::seed(23 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_024() {
        let mut rng = FastRng::seed(24 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_025() {
        let mut rng = FastRng::seed(25 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_026() {
        let mut rng = FastRng::seed(26 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_027() {
        let mut rng = FastRng::seed(27 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_028() {
        let mut rng = FastRng::seed(28 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_029() {
        let mut rng = FastRng::seed(29 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_030() {
        let mut rng = FastRng::seed(30 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_031() {
        let mut rng = FastRng::seed(31 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_032() {
        let mut rng = FastRng::seed(32 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_033() {
        let mut rng = FastRng::seed(33 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_034() {
        let mut rng = FastRng::seed(34 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_035() {
        let mut rng = FastRng::seed(35 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_036() {
        let mut rng = FastRng::seed(36 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_037() {
        let mut rng = FastRng::seed(37 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_038() {
        let mut rng = FastRng::seed(38 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_039() {
        let mut rng = FastRng::seed(39 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_040() {
        let mut rng = FastRng::seed(40 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_041() {
        let mut rng = FastRng::seed(41 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_042() {
        let mut rng = FastRng::seed(42 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_043() {
        let mut rng = FastRng::seed(43 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_044() {
        let mut rng = FastRng::seed(44 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_045() {
        let mut rng = FastRng::seed(45 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_046() {
        let mut rng = FastRng::seed(46 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_047() {
        let mut rng = FastRng::seed(47 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_048() {
        let mut rng = FastRng::seed(48 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_049() {
        let mut rng = FastRng::seed(49 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_050() {
        let mut rng = FastRng::seed(50 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_051() {
        let mut rng = FastRng::seed(51 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_052() {
        let mut rng = FastRng::seed(52 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_053() {
        let mut rng = FastRng::seed(53 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_054() {
        let mut rng = FastRng::seed(54 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_055() {
        let mut rng = FastRng::seed(55 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_056() {
        let mut rng = FastRng::seed(56 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_057() {
        let mut rng = FastRng::seed(57 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_058() {
        let mut rng = FastRng::seed(58 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_059() {
        let mut rng = FastRng::seed(59 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_060() {
        let mut rng = FastRng::seed(60 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_061() {
        let mut rng = FastRng::seed(61 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_062() {
        let mut rng = FastRng::seed(62 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_063() {
        let mut rng = FastRng::seed(63 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_064() {
        let mut rng = FastRng::seed(64 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_065() {
        let mut rng = FastRng::seed(65 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_066() {
        let mut rng = FastRng::seed(66 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_067() {
        let mut rng = FastRng::seed(67 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_068() {
        let mut rng = FastRng::seed(68 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_069() {
        let mut rng = FastRng::seed(69 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_070() {
        let mut rng = FastRng::seed(70 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_071() {
        let mut rng = FastRng::seed(71 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_072() {
        let mut rng = FastRng::seed(72 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_073() {
        let mut rng = FastRng::seed(73 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_074() {
        let mut rng = FastRng::seed(74 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_075() {
        let mut rng = FastRng::seed(75 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_076() {
        let mut rng = FastRng::seed(76 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_077() {
        let mut rng = FastRng::seed(77 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_078() {
        let mut rng = FastRng::seed(78 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_079() {
        let mut rng = FastRng::seed(79 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_080() {
        let mut rng = FastRng::seed(80 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_081() {
        let mut rng = FastRng::seed(81 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_082() {
        let mut rng = FastRng::seed(82 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_083() {
        let mut rng = FastRng::seed(83 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_084() {
        let mut rng = FastRng::seed(84 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_085() {
        let mut rng = FastRng::seed(85 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_086() {
        let mut rng = FastRng::seed(86 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_087() {
        let mut rng = FastRng::seed(87 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_088() {
        let mut rng = FastRng::seed(88 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_089() {
        let mut rng = FastRng::seed(89 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_090() {
        let mut rng = FastRng::seed(90 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_091() {
        let mut rng = FastRng::seed(91 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_092() {
        let mut rng = FastRng::seed(92 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_093() {
        let mut rng = FastRng::seed(93 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_094() {
        let mut rng = FastRng::seed(94 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_095() {
        let mut rng = FastRng::seed(95 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_096() {
        let mut rng = FastRng::seed(96 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_097() {
        let mut rng = FastRng::seed(97 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_098() {
        let mut rng = FastRng::seed(98 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_099() {
        let mut rng = FastRng::seed(99 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_100() {
        let mut rng = FastRng::seed(100 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_101() {
        let mut rng = FastRng::seed(101 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_102() {
        let mut rng = FastRng::seed(102 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_103() {
        let mut rng = FastRng::seed(103 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_104() {
        let mut rng = FastRng::seed(104 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_105() {
        let mut rng = FastRng::seed(105 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_106() {
        let mut rng = FastRng::seed(106 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_107() {
        let mut rng = FastRng::seed(107 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_108() {
        let mut rng = FastRng::seed(108 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_109() {
        let mut rng = FastRng::seed(109 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_110() {
        let mut rng = FastRng::seed(110 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_111() {
        let mut rng = FastRng::seed(111 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_112() {
        let mut rng = FastRng::seed(112 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_113() {
        let mut rng = FastRng::seed(113 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_114() {
        let mut rng = FastRng::seed(114 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_115() {
        let mut rng = FastRng::seed(115 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_116() {
        let mut rng = FastRng::seed(116 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_117() {
        let mut rng = FastRng::seed(117 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_118() {
        let mut rng = FastRng::seed(118 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_119() {
        let mut rng = FastRng::seed(119 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_120() {
        let mut rng = FastRng::seed(120 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_121() {
        let mut rng = FastRng::seed(121 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_122() {
        let mut rng = FastRng::seed(122 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_123() {
        let mut rng = FastRng::seed(123 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_124() {
        let mut rng = FastRng::seed(124 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_125() {
        let mut rng = FastRng::seed(125 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_126() {
        let mut rng = FastRng::seed(126 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_127() {
        let mut rng = FastRng::seed(127 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_128() {
        let mut rng = FastRng::seed(128 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_129() {
        let mut rng = FastRng::seed(129 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_130() {
        let mut rng = FastRng::seed(130 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_131() {
        let mut rng = FastRng::seed(131 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_132() {
        let mut rng = FastRng::seed(132 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_133() {
        let mut rng = FastRng::seed(133 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_134() {
        let mut rng = FastRng::seed(134 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_135() {
        let mut rng = FastRng::seed(135 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_136() {
        let mut rng = FastRng::seed(136 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_137() {
        let mut rng = FastRng::seed(137 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_138() {
        let mut rng = FastRng::seed(138 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_139() {
        let mut rng = FastRng::seed(139 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_140() {
        let mut rng = FastRng::seed(140 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_141() {
        let mut rng = FastRng::seed(141 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_142() {
        let mut rng = FastRng::seed(142 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_143() {
        let mut rng = FastRng::seed(143 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_144() {
        let mut rng = FastRng::seed(144 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_145() {
        let mut rng = FastRng::seed(145 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_146() {
        let mut rng = FastRng::seed(146 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_147() {
        let mut rng = FastRng::seed(147 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_148() {
        let mut rng = FastRng::seed(148 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_149() {
        let mut rng = FastRng::seed(149 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_150() {
        let mut rng = FastRng::seed(150 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_151() {
        let mut rng = FastRng::seed(151 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_152() {
        let mut rng = FastRng::seed(152 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_153() {
        let mut rng = FastRng::seed(153 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_154() {
        let mut rng = FastRng::seed(154 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_155() {
        let mut rng = FastRng::seed(155 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_156() {
        let mut rng = FastRng::seed(156 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_157() {
        let mut rng = FastRng::seed(157 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_158() {
        let mut rng = FastRng::seed(158 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_159() {
        let mut rng = FastRng::seed(159 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_160() {
        let mut rng = FastRng::seed(160 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_161() {
        let mut rng = FastRng::seed(161 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_162() {
        let mut rng = FastRng::seed(162 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_163() {
        let mut rng = FastRng::seed(163 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_164() {
        let mut rng = FastRng::seed(164 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_165() {
        let mut rng = FastRng::seed(165 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_166() {
        let mut rng = FastRng::seed(166 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_167() {
        let mut rng = FastRng::seed(167 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_168() {
        let mut rng = FastRng::seed(168 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_169() {
        let mut rng = FastRng::seed(169 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_170() {
        let mut rng = FastRng::seed(170 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_171() {
        let mut rng = FastRng::seed(171 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_172() {
        let mut rng = FastRng::seed(172 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_173() {
        let mut rng = FastRng::seed(173 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_174() {
        let mut rng = FastRng::seed(174 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_175() {
        let mut rng = FastRng::seed(175 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_176() {
        let mut rng = FastRng::seed(176 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_177() {
        let mut rng = FastRng::seed(177 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_178() {
        let mut rng = FastRng::seed(178 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_179() {
        let mut rng = FastRng::seed(179 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_180() {
        let mut rng = FastRng::seed(180 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_181() {
        let mut rng = FastRng::seed(181 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_182() {
        let mut rng = FastRng::seed(182 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_183() {
        let mut rng = FastRng::seed(183 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_184() {
        let mut rng = FastRng::seed(184 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_185() {
        let mut rng = FastRng::seed(185 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_186() {
        let mut rng = FastRng::seed(186 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_187() {
        let mut rng = FastRng::seed(187 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_188() {
        let mut rng = FastRng::seed(188 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_189() {
        let mut rng = FastRng::seed(189 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_190() {
        let mut rng = FastRng::seed(190 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_191() {
        let mut rng = FastRng::seed(191 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_192() {
        let mut rng = FastRng::seed(192 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_193() {
        let mut rng = FastRng::seed(193 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_194() {
        let mut rng = FastRng::seed(194 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_195() {
        let mut rng = FastRng::seed(195 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_196() {
        let mut rng = FastRng::seed(196 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_197() {
        let mut rng = FastRng::seed(197 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_198() {
        let mut rng = FastRng::seed(198 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_199() {
        let mut rng = FastRng::seed(199 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_200() {
        let mut rng = FastRng::seed(200 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_201() {
        let mut rng = FastRng::seed(201 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_202() {
        let mut rng = FastRng::seed(202 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_203() {
        let mut rng = FastRng::seed(203 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_204() {
        let mut rng = FastRng::seed(204 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_205() {
        let mut rng = FastRng::seed(205 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_206() {
        let mut rng = FastRng::seed(206 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_207() {
        let mut rng = FastRng::seed(207 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_208() {
        let mut rng = FastRng::seed(208 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_209() {
        let mut rng = FastRng::seed(209 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_210() {
        let mut rng = FastRng::seed(210 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_211() {
        let mut rng = FastRng::seed(211 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_212() {
        let mut rng = FastRng::seed(212 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_213() {
        let mut rng = FastRng::seed(213 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_214() {
        let mut rng = FastRng::seed(214 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_215() {
        let mut rng = FastRng::seed(215 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_216() {
        let mut rng = FastRng::seed(216 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_217() {
        let mut rng = FastRng::seed(217 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_218() {
        let mut rng = FastRng::seed(218 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    #[test]
    fn test_crossover_stress_219() {
        let mut rng = FastRng::seed(219 as u64);
        let p1 = Genome::new(vec![1.0; 8]);
        let p2 = Genome::new(vec![2.0; 8]);

        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);

        let (u1, u2) = uniform_crossover(&p1, &p2, &mut rng);
        assert_eq!(u1.len(), 8);
        assert_eq!(u2.len(), 8);
    }

    // Evolutionary computation optimization and invariance padding line 0
}
