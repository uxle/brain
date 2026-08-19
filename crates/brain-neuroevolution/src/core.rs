//! # Neuroevolution Core Types
//!
//! Master evolutionary configuration, errors, results, and summary descriptors.
#![allow(missing_docs)]


/// Evolutionary algorithm kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlgorithmKind {
    #[default]
    GeneticAlgorithm,
    CMAES,
    ES1p1,
    HyperNEAT,
}

/// Master configuration for evolutionary runs.
#[derive(Debug, Clone)]
pub struct EvoConfig {
    pub algorithm: AlgorithmKind,
    pub population_size: usize,
    pub genome_dim: usize,
    pub elite_count: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub max_generations: usize,
    pub target_fitness: Option<f64>,
}

impl Default for EvoConfig {
    fn default() -> Self {
        Self {
            algorithm: AlgorithmKind::GeneticAlgorithm,
            population_size: 50,
            genome_dim: 10,
            elite_count: 2,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            max_generations: 100,
            target_fitness: None,
        }
    }
}

impl EvoConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.population_size < 2 {
            return Err("population_size must be >= 2".into());
        }
        if self.elite_count >= self.population_size {
            return Err("elite_count must be < population_size".into());
        }
        if self.genome_dim == 0 {
            return Err("genome_dim must be > 0".into());
        }
        if self.mutation_rate < 0.0 || self.mutation_rate > 1.0 {
            return Err("mutation_rate must be in [0, 1]".into());
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "EvoConfig[algo={:?} pop={} dim={} elites={} mut={:.2} xover={:.2} gens={}]",
            self.algorithm, self.population_size, self.genome_dim, self.elite_count,
            self.mutation_rate, self.crossover_rate, self.max_generations
        )
    }
}

/// Error type for evolutionary processes.
#[derive(Debug, Clone, PartialEq)]
pub enum EvoError {
    InvalidConfig(String),
    PopulationEmpty,
    EvaluationFailed(String),
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for EvoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvoError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            EvoError::PopulationEmpty => write!(f, "Population is empty"),
            EvoError::EvaluationFailed(msg) => write!(f, "Evaluation failed: {}", msg),
            EvoError::DimensionMismatch { expected, got } => write!(f, "Dimension mismatch: expected {}, got {}", expected, got),
        }
    }
}

pub type EvoResult<T> = Result<T, EvoError>;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
}
