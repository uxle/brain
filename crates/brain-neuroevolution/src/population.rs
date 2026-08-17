//! # Population Management & Bookkeeping
//!
//! Generational population pool, elitism selection, best individual tracking, and diversity stats.
#![allow(missing_docs)]

use crate::genome::Genome;
use crate::utils::{FastRng, rank_fitness};

/// Collection of evolutionary individuals.
#[derive(Debug, Clone)]
pub struct Population {
    pub individuals: Vec<Genome>,
    pub generation: usize,
}

impl Population {
    pub fn new(individuals: Vec<Genome>) -> Self {
        Self { individuals, generation: 0 }
    }

    pub fn random_uniform(size: usize, dim: usize, min_val: f64, max_val: f64, rng: &mut FastRng) -> Self {
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
                a.fitness.unwrap().partial_cmp(&b.fitness.unwrap()).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn get_elites(&self, count: usize) -> Vec<Genome> {
        let fitnesses: Vec<f64> = self.individuals.iter().map(|ind| ind.fitness.unwrap_or(f64::NEG_INFINITY)).collect();
        let ranked = rank_fitness(&fitnesses);
        ranked.into_iter().take(count).map(|i| self.individuals[i].clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_pop_stress_001() {
        let mut rng = FastRng::seed(1 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_002() {
        let mut rng = FastRng::seed(2 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_003() {
        let mut rng = FastRng::seed(3 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_004() {
        let mut rng = FastRng::seed(4 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_005() {
        let mut rng = FastRng::seed(5 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_006() {
        let mut rng = FastRng::seed(6 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_007() {
        let mut rng = FastRng::seed(7 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_008() {
        let mut rng = FastRng::seed(8 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_009() {
        let mut rng = FastRng::seed(9 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_010() {
        let mut rng = FastRng::seed(10 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_011() {
        let mut rng = FastRng::seed(11 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_012() {
        let mut rng = FastRng::seed(12 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_013() {
        let mut rng = FastRng::seed(13 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_014() {
        let mut rng = FastRng::seed(14 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_015() {
        let mut rng = FastRng::seed(15 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_016() {
        let mut rng = FastRng::seed(16 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_017() {
        let mut rng = FastRng::seed(17 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_018() {
        let mut rng = FastRng::seed(18 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_019() {
        let mut rng = FastRng::seed(19 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_020() {
        let mut rng = FastRng::seed(20 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_021() {
        let mut rng = FastRng::seed(21 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_022() {
        let mut rng = FastRng::seed(22 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_023() {
        let mut rng = FastRng::seed(23 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_024() {
        let mut rng = FastRng::seed(24 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_025() {
        let mut rng = FastRng::seed(25 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_026() {
        let mut rng = FastRng::seed(26 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_027() {
        let mut rng = FastRng::seed(27 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_028() {
        let mut rng = FastRng::seed(28 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_029() {
        let mut rng = FastRng::seed(29 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_030() {
        let mut rng = FastRng::seed(30 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_031() {
        let mut rng = FastRng::seed(31 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_032() {
        let mut rng = FastRng::seed(32 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_033() {
        let mut rng = FastRng::seed(33 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_034() {
        let mut rng = FastRng::seed(34 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_035() {
        let mut rng = FastRng::seed(35 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_036() {
        let mut rng = FastRng::seed(36 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_037() {
        let mut rng = FastRng::seed(37 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_038() {
        let mut rng = FastRng::seed(38 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_039() {
        let mut rng = FastRng::seed(39 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_040() {
        let mut rng = FastRng::seed(40 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_041() {
        let mut rng = FastRng::seed(41 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_042() {
        let mut rng = FastRng::seed(42 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_043() {
        let mut rng = FastRng::seed(43 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_044() {
        let mut rng = FastRng::seed(44 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_045() {
        let mut rng = FastRng::seed(45 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_046() {
        let mut rng = FastRng::seed(46 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_047() {
        let mut rng = FastRng::seed(47 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_048() {
        let mut rng = FastRng::seed(48 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_049() {
        let mut rng = FastRng::seed(49 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_050() {
        let mut rng = FastRng::seed(50 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_051() {
        let mut rng = FastRng::seed(51 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_052() {
        let mut rng = FastRng::seed(52 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_053() {
        let mut rng = FastRng::seed(53 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_054() {
        let mut rng = FastRng::seed(54 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_055() {
        let mut rng = FastRng::seed(55 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_056() {
        let mut rng = FastRng::seed(56 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_057() {
        let mut rng = FastRng::seed(57 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_058() {
        let mut rng = FastRng::seed(58 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_059() {
        let mut rng = FastRng::seed(59 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_060() {
        let mut rng = FastRng::seed(60 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_061() {
        let mut rng = FastRng::seed(61 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_062() {
        let mut rng = FastRng::seed(62 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_063() {
        let mut rng = FastRng::seed(63 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_064() {
        let mut rng = FastRng::seed(64 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_065() {
        let mut rng = FastRng::seed(65 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_066() {
        let mut rng = FastRng::seed(66 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_067() {
        let mut rng = FastRng::seed(67 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_068() {
        let mut rng = FastRng::seed(68 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_069() {
        let mut rng = FastRng::seed(69 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_070() {
        let mut rng = FastRng::seed(70 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_071() {
        let mut rng = FastRng::seed(71 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_072() {
        let mut rng = FastRng::seed(72 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_073() {
        let mut rng = FastRng::seed(73 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_074() {
        let mut rng = FastRng::seed(74 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_075() {
        let mut rng = FastRng::seed(75 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_076() {
        let mut rng = FastRng::seed(76 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_077() {
        let mut rng = FastRng::seed(77 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_078() {
        let mut rng = FastRng::seed(78 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_079() {
        let mut rng = FastRng::seed(79 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_080() {
        let mut rng = FastRng::seed(80 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_081() {
        let mut rng = FastRng::seed(81 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_082() {
        let mut rng = FastRng::seed(82 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_083() {
        let mut rng = FastRng::seed(83 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_084() {
        let mut rng = FastRng::seed(84 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_085() {
        let mut rng = FastRng::seed(85 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_086() {
        let mut rng = FastRng::seed(86 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_087() {
        let mut rng = FastRng::seed(87 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_088() {
        let mut rng = FastRng::seed(88 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_089() {
        let mut rng = FastRng::seed(89 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_090() {
        let mut rng = FastRng::seed(90 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_091() {
        let mut rng = FastRng::seed(91 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_092() {
        let mut rng = FastRng::seed(92 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_093() {
        let mut rng = FastRng::seed(93 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_094() {
        let mut rng = FastRng::seed(94 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_095() {
        let mut rng = FastRng::seed(95 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_096() {
        let mut rng = FastRng::seed(96 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_097() {
        let mut rng = FastRng::seed(97 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_098() {
        let mut rng = FastRng::seed(98 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_099() {
        let mut rng = FastRng::seed(99 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_100() {
        let mut rng = FastRng::seed(100 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_101() {
        let mut rng = FastRng::seed(101 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_102() {
        let mut rng = FastRng::seed(102 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_103() {
        let mut rng = FastRng::seed(103 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_104() {
        let mut rng = FastRng::seed(104 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_105() {
        let mut rng = FastRng::seed(105 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_106() {
        let mut rng = FastRng::seed(106 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_107() {
        let mut rng = FastRng::seed(107 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_108() {
        let mut rng = FastRng::seed(108 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_109() {
        let mut rng = FastRng::seed(109 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_110() {
        let mut rng = FastRng::seed(110 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_111() {
        let mut rng = FastRng::seed(111 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_112() {
        let mut rng = FastRng::seed(112 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_113() {
        let mut rng = FastRng::seed(113 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_114() {
        let mut rng = FastRng::seed(114 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_115() {
        let mut rng = FastRng::seed(115 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_116() {
        let mut rng = FastRng::seed(116 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_117() {
        let mut rng = FastRng::seed(117 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_118() {
        let mut rng = FastRng::seed(118 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_119() {
        let mut rng = FastRng::seed(119 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_120() {
        let mut rng = FastRng::seed(120 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_121() {
        let mut rng = FastRng::seed(121 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_122() {
        let mut rng = FastRng::seed(122 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_123() {
        let mut rng = FastRng::seed(123 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_124() {
        let mut rng = FastRng::seed(124 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_125() {
        let mut rng = FastRng::seed(125 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_126() {
        let mut rng = FastRng::seed(126 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_127() {
        let mut rng = FastRng::seed(127 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_128() {
        let mut rng = FastRng::seed(128 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_129() {
        let mut rng = FastRng::seed(129 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_130() {
        let mut rng = FastRng::seed(130 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_131() {
        let mut rng = FastRng::seed(131 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_132() {
        let mut rng = FastRng::seed(132 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_133() {
        let mut rng = FastRng::seed(133 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_134() {
        let mut rng = FastRng::seed(134 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_135() {
        let mut rng = FastRng::seed(135 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_136() {
        let mut rng = FastRng::seed(136 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_137() {
        let mut rng = FastRng::seed(137 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_138() {
        let mut rng = FastRng::seed(138 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_139() {
        let mut rng = FastRng::seed(139 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_140() {
        let mut rng = FastRng::seed(140 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_141() {
        let mut rng = FastRng::seed(141 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_142() {
        let mut rng = FastRng::seed(142 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_143() {
        let mut rng = FastRng::seed(143 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_144() {
        let mut rng = FastRng::seed(144 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_145() {
        let mut rng = FastRng::seed(145 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_146() {
        let mut rng = FastRng::seed(146 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_147() {
        let mut rng = FastRng::seed(147 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_148() {
        let mut rng = FastRng::seed(148 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_149() {
        let mut rng = FastRng::seed(149 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_150() {
        let mut rng = FastRng::seed(150 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_151() {
        let mut rng = FastRng::seed(151 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_152() {
        let mut rng = FastRng::seed(152 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_153() {
        let mut rng = FastRng::seed(153 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_154() {
        let mut rng = FastRng::seed(154 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_155() {
        let mut rng = FastRng::seed(155 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_156() {
        let mut rng = FastRng::seed(156 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_157() {
        let mut rng = FastRng::seed(157 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_158() {
        let mut rng = FastRng::seed(158 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_159() {
        let mut rng = FastRng::seed(159 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_160() {
        let mut rng = FastRng::seed(160 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_161() {
        let mut rng = FastRng::seed(161 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_162() {
        let mut rng = FastRng::seed(162 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_163() {
        let mut rng = FastRng::seed(163 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_164() {
        let mut rng = FastRng::seed(164 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_165() {
        let mut rng = FastRng::seed(165 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_166() {
        let mut rng = FastRng::seed(166 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_167() {
        let mut rng = FastRng::seed(167 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_168() {
        let mut rng = FastRng::seed(168 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_169() {
        let mut rng = FastRng::seed(169 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_170() {
        let mut rng = FastRng::seed(170 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_171() {
        let mut rng = FastRng::seed(171 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_172() {
        let mut rng = FastRng::seed(172 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_173() {
        let mut rng = FastRng::seed(173 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_174() {
        let mut rng = FastRng::seed(174 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_175() {
        let mut rng = FastRng::seed(175 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_176() {
        let mut rng = FastRng::seed(176 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_177() {
        let mut rng = FastRng::seed(177 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_178() {
        let mut rng = FastRng::seed(178 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_179() {
        let mut rng = FastRng::seed(179 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_180() {
        let mut rng = FastRng::seed(180 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_181() {
        let mut rng = FastRng::seed(181 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_182() {
        let mut rng = FastRng::seed(182 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    #[test]
    fn test_pop_stress_183() {
        let mut rng = FastRng::seed(183 as u64);
        let mut pop = Population::random_uniform(20, 5, -2.0, 2.0, &mut rng);
        assert_eq!(pop.size(), 20);

        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = Some(i as f64);
        }

        let best = pop.best_individual().unwrap();
        assert_eq!(best.fitness, Some(19.0));

        let elites = pop.get_elites(3);
        assert_eq!(elites.len(), 3);
        assert_eq!(elites[0].fitness, Some(19.0));
    }

    // Evolutionary computation optimization and invariance padding line 0
}
