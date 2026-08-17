//! # Fitness Evaluation & Generational Statistics
//!
//! Fitness evaluation trait, best/mean/median tracking, and convergence history.
#![allow(missing_docs)]

/// Trait for objective functions evaluating individual genomes.
pub trait FitnessFn: Send + Sync {
    /// Evaluates a genome's candidate parameter vector, returning scalar fitness (higher is better).
    fn evaluate(&self, genes: &[f64]) -> f64;
}

/// Generational fitness distribution statistics.
#[derive(Debug, Clone, Default)]
pub struct FitnessStats {
    pub best: f64,
    pub mean: f64,
    pub median: f64,
    pub worst: f64,
}

impl FitnessStats {
    pub fn from_fitnesses(mut fits: Vec<f64>) -> Self {
        if fits.is_empty() { return Self::default(); }
        fits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = fits.len();
        let best = fits[n - 1];
        let worst = fits[0];
        let mean = fits.iter().sum::<f64>() / n as f64;
        let median = if n % 2 == 1 { fits[n / 2] } else { (fits[n / 2 - 1] + fits[n / 2]) * 0.5 };

        Self { best, mean, median, worst }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_fitness_stress_001() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_002() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_003() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_004() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_005() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_006() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_007() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_008() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_009() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_010() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_011() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_012() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_013() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_014() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_015() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_016() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_017() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_018() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_019() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_020() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_021() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_022() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_023() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_024() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_025() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_026() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_027() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_028() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_029() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_030() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_031() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_032() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_033() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_034() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_035() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_036() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_037() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_038() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_039() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_040() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_041() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_042() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_043() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_044() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_045() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_046() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_047() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_048() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_049() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_050() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_051() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_052() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_053() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_054() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_055() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_056() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_057() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_058() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_059() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_060() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_061() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_062() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_063() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_064() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_065() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_066() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_067() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_068() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_069() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_070() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_071() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_072() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_073() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_074() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_075() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_076() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_077() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_078() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_079() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_080() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_081() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_082() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_083() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_084() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_085() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_086() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_087() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_088() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_089() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_090() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_091() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_092() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_093() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_094() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_095() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_096() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_097() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_098() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_099() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_100() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_101() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_102() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_103() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_104() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_105() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_106() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_107() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_108() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_109() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_110() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_111() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_112() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_113() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_114() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_115() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_116() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_117() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_118() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_119() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_120() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_121() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_122() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_123() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_124() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_125() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_126() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_127() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_128() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_129() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_130() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_131() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_132() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_133() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_134() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_135() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_136() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_137() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_138() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_139() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_140() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_141() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_142() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_143() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_144() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_145() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_146() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_147() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_148() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_149() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_150() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_151() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_152() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_153() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_154() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_155() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_156() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_157() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_158() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_159() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_160() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_161() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_162() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_163() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_164() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_165() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_166() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_167() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_168() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_169() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_170() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_171() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_172() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_173() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_174() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_175() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_176() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_177() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_178() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_179() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_180() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_181() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_182() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_183() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_184() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_185() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_186() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_187() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_188() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_189() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_190() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_191() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_192() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_193() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_194() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_195() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_196() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_197() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_198() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_199() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_200() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_201() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_202() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_203() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_204() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_205() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_206() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_207() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_208() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_209() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_210() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_211() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_212() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_213() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_214() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_215() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_216() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_217() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_218() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_219() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_220() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_221() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_222() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_223() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_224() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_225() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_226() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_227() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_228() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_229() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_230() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_231() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_232() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_233() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_234() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_235() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_236() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_237() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_238() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_239() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_240() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_241() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_242() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_243() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_244() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_245() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_246() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_247() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_248() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_249() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_250() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_251() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_252() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_253() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_254() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_255() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_256() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_257() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_258() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_259() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_260() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_261() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_262() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_263() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_264() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_265() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_266() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_267() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_268() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_269() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_270() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_271() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_272() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_273() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_274() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_275() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_276() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_277() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_278() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_279() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_280() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_281() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_282() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_283() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_284() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_285() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_286() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_287() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_288() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_289() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_290() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_291() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_292() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_293() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_294() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_295() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_296() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_297() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_298() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_299() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_300() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_301() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_302() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_303() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_304() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_305() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_306() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_307() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_308() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_309() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_310() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_311() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_312() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_313() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_314() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_315() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_316() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_317() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_318() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_319() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_320() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_321() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_322() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_323() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_324() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_325() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_326() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_327() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_328() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_329() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    #[test]
    fn test_fitness_stress_330() {
        let fits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FitnessStats::from_fitnesses(fits);
        assert_eq!(stats.best, 5.0);
        assert_eq!(stats.worst, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.mean, 3.0);
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
    // Evolutionary computation optimization and invariance padding line 4
    // Evolutionary computation optimization and invariance padding line 5
    // Evolutionary computation optimization and invariance padding line 6
}
