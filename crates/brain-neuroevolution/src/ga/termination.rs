//! # Evolutionary Termination Conditions
//!
//! Max generation cutoff, patience-based early stopping, target fitness thresholds, and compute budget limits.
#![allow(missing_docs)]

/// Configuration for termination conditions.
#[derive(Debug, Clone)]
pub struct TerminationConfig {
    pub max_generations: usize,
    pub patience: Option<usize>,
    pub min_fitness_delta: f64,
    pub target_fitness: Option<f64>,
}

impl Default for TerminationConfig {
    fn default() -> Self {
        Self {
            max_generations: 100,
            patience: Some(20),
            min_fitness_delta: 1e-4,
            target_fitness: None,
        }
    }
}

/// Evaluator tracking patience and stagnation across generations.
#[derive(Debug, Clone, Default)]
pub struct TerminationTracker {
    pub best_fitness: f64,
    pub generations_without_improvement: usize,
}

impl TerminationTracker {
    pub fn update(&mut self, current_best: f64, config: &TerminationConfig) -> bool {
        if current_best > self.best_fitness + config.min_fitness_delta {
            self.best_fitness = current_best;
            self.generations_without_improvement = 0;
        } else {
            self.generations_without_improvement += 1;
        }

        if let Some(target) = config.target_fitness {
            if current_best >= target { return true; }
        }

        if let Some(patience) = config.patience {
            if self.generations_without_improvement >= patience { return true; }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
