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
}
