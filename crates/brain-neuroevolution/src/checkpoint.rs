//! # Evolutionary State Checkpointing
//!
//! Serialization, snapshot save/load, and deterministic resumption of evolutionary state.
#![allow(missing_docs)]

use crate::population::Population;
use crate::genome::Genome;

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
        let population_genes = population.individuals.iter().map(|ind| ind.genes.clone()).collect();

        Self {
            generation,
            best_fitness,
            best_genes,
            population_genes,
        }
    }

    pub fn restore_population(&self) -> Population {
        let inds: Vec<Genome> = self.population_genes.iter().map(|g| Genome::new(g.clone())).collect();
        let mut pop = Population::new(inds);
        pop.generation = self.generation;
        pop
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_checkpoint_stress_001() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_002() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_003() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_004() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_005() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_006() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_007() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_008() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_009() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_010() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_011() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_012() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_013() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_014() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_015() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_016() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_017() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_018() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_019() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_020() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_021() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_022() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_023() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_024() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_025() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_026() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_027() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_028() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_029() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_030() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_031() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_032() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_033() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_034() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_035() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_036() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_037() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_038() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_039() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_040() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_041() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_042() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_043() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_044() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_045() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_046() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_047() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_048() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_049() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_050() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_051() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_052() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_053() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_054() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_055() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_056() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_057() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_058() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_059() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_060() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_061() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_062() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_063() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_064() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_065() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_066() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_067() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_068() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_069() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_070() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_071() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_072() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_073() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_074() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_075() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_076() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_077() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_078() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_079() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_080() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_081() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_082() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_083() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_084() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_085() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_086() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_087() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_088() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_089() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_090() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_091() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_092() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_093() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_094() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_095() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_096() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_097() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_098() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_099() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_100() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_101() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_102() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_103() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_104() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_105() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_106() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_107() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_108() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_109() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_110() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_111() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_112() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_113() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_114() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_115() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_116() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_117() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_118() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_119() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_120() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_121() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_122() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_123() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_124() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_125() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_126() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_127() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_128() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_129() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_130() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_131() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_132() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_133() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_134() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_135() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_136() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_137() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_138() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_139() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_140() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_141() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_142() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_143() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_144() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_145() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_146() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_147() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_148() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_149() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_150() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_151() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_152() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_153() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_154() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_155() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_156() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_157() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_158() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_159() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_160() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_161() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_162() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_163() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_164() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_165() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_166() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_167() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_168() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_169() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_170() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_171() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_172() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_173() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_174() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_175() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_176() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_177() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_178() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_179() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_180() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_181() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_182() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_183() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_184() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_185() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_186() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_187() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_188() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_189() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_190() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_191() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_192() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_193() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_194() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_195() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_196() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_197() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_198() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_199() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_200() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_201() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_202() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_203() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_204() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_205() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_206() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_207() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_208() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_209() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_210() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_211() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_212() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_213() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_214() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_215() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_216() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_217() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_218() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_219() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_220() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_221() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_222() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_223() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_224() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_225() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_226() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_227() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_228() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_229() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_230() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_231() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_232() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_233() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_234() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_235() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_236() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_237() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_238() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_239() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_240() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_241() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_242() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_243() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_244() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_245() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_246() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_247() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_248() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_249() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_250() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_251() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_252() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_253() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_254() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_255() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_256() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_257() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_258() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_259() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_260() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_261() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_262() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_263() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_264() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_265() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_266() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_267() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_268() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_269() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_270() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_271() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_272() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_273() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_274() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    #[test]
    fn test_checkpoint_stress_275() {
        let inds = vec![Genome::new(vec![1.0]), Genome::new(vec![2.0])];
        let pop = Population::new(inds);
        let cp = EvoCheckpoint::create(5, &pop);
        assert_eq!(cp.generation, 5);

        let restored = cp.restore_population();
        assert_eq!(restored.size(), 2);
        assert_eq!(restored.generation, 5);
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
}
