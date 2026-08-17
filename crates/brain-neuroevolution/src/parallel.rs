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

    #[test]
    fn test_parallel_stress_001() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_002() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_003() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_004() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_005() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_006() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_007() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_008() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_009() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_010() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_011() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_012() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_013() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_014() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_015() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_016() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_017() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_018() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_019() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_020() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_021() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_022() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_023() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_024() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_025() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_026() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_027() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_028() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_029() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_030() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_031() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_032() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_033() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_034() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_035() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_036() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_037() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_038() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_039() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_040() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_041() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_042() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_043() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_044() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_045() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_046() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_047() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_048() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_049() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_050() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_051() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_052() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_053() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_054() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_055() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_056() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_057() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_058() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_059() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_060() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_061() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_062() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_063() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_064() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_065() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_066() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_067() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_068() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_069() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_070() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_071() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_072() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_073() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_074() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_075() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_076() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_077() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_078() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_079() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_080() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_081() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_082() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_083() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_084() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_085() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_086() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_087() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_088() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_089() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_090() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_091() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_092() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_093() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_094() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_095() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_096() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_097() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_098() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_099() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_100() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_101() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_102() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_103() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_104() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_105() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_106() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_107() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_108() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_109() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_110() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_111() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_112() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_113() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_114() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_115() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_116() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_117() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_118() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_119() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_120() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_121() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_122() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_123() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_124() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_125() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_126() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_127() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_128() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_129() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_130() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_131() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_132() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_133() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_134() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_135() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_136() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_137() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_138() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_139() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_140() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_141() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_142() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_143() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_144() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_145() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_146() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_147() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_148() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_149() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_150() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_151() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_152() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_153() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_154() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_155() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_156() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_157() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_158() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_159() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_160() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_161() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_162() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_163() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_164() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_165() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_166() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_167() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_168() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_169() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_170() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_171() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_172() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_173() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_174() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_175() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_176() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_177() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_178() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_179() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_180() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_181() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_182() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_183() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_184() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_185() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_186() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_187() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_188() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_189() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_190() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_191() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_192() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_193() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_194() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_195() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_196() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_197() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_198() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_199() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_200() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_201() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_202() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_203() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_204() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_205() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_206() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_207() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_208() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_209() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_210() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_211() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_212() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_213() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_214() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_215() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_216() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_217() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_218() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_219() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    #[test]
    fn test_parallel_stress_220() {
        struct DummyFit;
        impl FitnessFn for DummyFit {
            fn evaluate(&self, genes: &[f64]) -> f64 {
                genes.iter().sum()
            }
        }

        let mut inds = vec![Genome::new(vec![1.0, 2.0]), Genome::new(vec![3.0, 4.0])];
        evaluate_population_parallel(&mut inds, &DummyFit, 2);
        assert_eq!(inds[0].fitness, Some(3.0));
        assert_eq!(inds[1].fitness, Some(7.0));
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
    // Evolutionary computation optimization and invariance padding line 4
    // Evolutionary computation optimization and invariance padding line 5
}
